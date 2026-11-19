"""3D volumetric metrics, patient-balanced/episode estimands, and paired bootstrap contrasts."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from .selectors import ARMS

NEW_ARMS = (
    "resp_magnitude_k2",
    "resp_signed_global_k2",
    "resp_signed_localring_k2",
    "resp_signed_adaptive",
    "local_contrast_near2",
    "pba_fixed_k2",
)


def surface_6conn(mask: np.ndarray) -> np.ndarray:
    """Extract 6-connected binary surface voxels matching analysis/gate_a/measurements.py."""
    m = np.asarray(mask, dtype=bool)
    return m & ~ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(m.ndim, 1), border_value=0)


def independent_hd95_mm(pred: np.ndarray, gt: np.ndarray, spacing_zyx: tuple[float, ...]) -> tuple[float, float, float]:
    """Pooled symmetric 95th-percentile physical surface distance in mm."""
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool)
    if not p.any() and not g.any():
        return 0.0, 0.0, 0.0
    if not p.any() or not g.any():
        return float("inf"), float("inf"), float("inf")
    ps, gs = surface_6conn(p), surface_6conn(g)
    sp = np.asarray(spacing_zyx, dtype=np.float64)
    pc = np.argwhere(ps) * sp
    gc = np.argwhere(gs) * sp
    p_to_g = cKDTree(gc).query(pc, workers=1)[0]
    g_to_p = cKDTree(pc).query(gc, workers=1)[0]
    return (
        float(np.percentile(np.concatenate([p_to_g, g_to_p]), 95)),
        float(np.percentile(p_to_g, 95)),
        float(np.percentile(g_to_p, 95)),
    )


def disjoint_fp_3d(pred: np.ndarray, gt: np.ndarray, other: np.ndarray) -> dict[str, int]:
    """Partition 3D false positives into detached, other annotated organs, and residual."""
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool)
    o = np.asarray(other, dtype=bool)
    labels, _ = ndimage.label(p, structure=ndimage.generate_binary_structure(3, 1))
    hits = np.bincount(labels[g].ravel(), minlength=int(labels.max()) + 1)
    no_overlap = hits == 0
    no_overlap[0] = False
    detached = p & no_overlap[labels]
    fp = p & ~g
    other_fp = fp & ~detached & o
    residual = fp & ~detached & ~o
    assert np.array_equal(detached | other_fp | residual, fp)
    return {
        "detached_fp": int(detached.sum()),
        "other_annotated_organ_fp": int(other_fp.sum()),
        "residual_fp": int(residual.sum()),
    }


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.bool_).tobytes()).hexdigest()


def measure_episode(
    episode_id: int,
    fold: int,
    patient_id: str,
    organ: str,
    spacing_xyz: tuple[float, float, float] | list[float],
    gt_3d: np.ndarray,
    other_3d: np.ndarray,
    raw_by_arm: dict[str, np.ndarray],
    cc_by_arm: dict[str, np.ndarray],
) -> list[dict]:
    """Compute all Raw and CC 3D episode metrics across all 9 Option A arms."""
    gt = np.asarray(gt_3d, dtype=bool)
    other = np.asarray(other_3d, dtype=bool)
    sp_xyz = tuple(float(x) for x in spacing_xyz)
    sp_zyx = (sp_xyz[2], sp_xyz[1], sp_xyz[0])
    voxel_vol = float(np.prod(sp_xyz))
    gt_vox = int(gt.sum())

    rows: list[dict] = []
    for arm in ARMS:
        raw = np.asarray(raw_by_arm[arm], dtype=bool)
        cc = np.asarray(cc_by_arm[arm], dtype=bool)
        assert raw.shape == cc.shape == gt.shape
        assert not (cc & ~raw).any()
        removed_tp = int((raw & ~cc & gt).sum())
        removed_fp = int((raw & ~cc & ~gt).sum())

        for post, pred in (("raw", raw), ("cc", cc)):
            tp = int((pred & gt).sum())
            fp = int((pred & ~gt).sum())
            fn = int((~pred & gt).sum())
            denom = 2 * tp + fp + fn
            dice = (2.0 * tp / denom) if denom > 0 else 1.0
            recall = (tp / gt_vox) if gt_vox > 0 else 1.0
            hd95, dir_p, dir_g = independent_hd95_mm(pred, gt, sp_zyx)
            part = disjoint_fp_3d(pred, gt, other)
            rows.append(
                {
                    "episode_id": int(episode_id),
                    "fold": int(fold),
                    "patient_id": str(patient_id),
                    "organ": str(organ),
                    "arm": arm,
                    "post": post,
                    "removed_tp": removed_tp,
                    "removed_fp": removed_fp,
                    "dice": float(dice),
                    "hd95": float(hd95),
                    "recall": float(recall),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "detached_fp": part["detached_fp"],
                    "other_annotated_organ_fp": part["other_annotated_organ_fp"],
                    "residual_fp": part["residual_fp"],
                    "directed_pred_to_gt95": float(dir_p),
                    "directed_gt_to_pred95": float(dir_g),
                    "fp_mm3": float(fp * voxel_vol),
                    "fn_mm3": float(fn * voxel_vol),
                    "empty_prediction": bool(not pred.any()),
                    "gt_voxels": gt_vox,
                    "metric_status": "finite" if np.isfinite(hd95) else "undefined_one_empty",
                    "mask_sha256": mask_sha256(pred),
                }
            )
    return rows


def _patient_level_table(df: pd.DataFrame, post: str, metric: str) -> pd.DataFrame:
    d = df[df["post"] == post]
    rep = d.groupby(["patient_id", "organ", "arm"], as_index=False)[metric].mean()
    pat = rep.groupby(["patient_id", "arm"])[metric].mean().unstack("arm")
    patients = sorted(d["patient_id"].unique())
    arms = [a for a in ARMS if a in pat.columns]
    return pat.reindex(index=patients, columns=arms)


def compute_estimands(df: pd.DataFrame) -> pd.DataFrame:
    """Compute both patient-balanced and episode-weighted summaries with SD and tail quantiles."""
    rows: list[dict] = []
    for post in ("raw", "cc"):
        d_post = df[df["post"] == post]
        for arm in ARMS:
            g = d_post[d_post["arm"] == arm]
            if g.empty:
                continue
            rep = g.groupby(["patient_id", "organ"])[["dice", "hd95", "recall", "detached_fp", "fp", "fn", "removed_tp", "removed_fp"]].mean()
            pat = rep.groupby("patient_id").mean()
            hd_vals = g["hd95"].to_numpy(dtype=float)
            pat_hd = pat["hd95"].to_numpy(dtype=float)
            rows.append(
                {
                    "arm": arm,
                    "post": post,
                    "episodes": int(len(g)),
                    "patients": int(g["patient_id"].nunique()),
                    "patient_dice_pct": float(pat["dice"].mean() * 100.0),
                    "patient_dice_sd_pct": float(pat["dice"].std(ddof=1) * 100.0),
                    "episode_dice_pct": float(g["dice"].mean() * 100.0),
                    "episode_dice_sd_pct": float(g["dice"].std(ddof=1) * 100.0),
                    "patient_hd95_mm": float(np.mean(pat_hd)) if np.isfinite(pat_hd).all() else float("inf"),
                    "episode_hd95_mm": float(np.mean(hd_vals)) if np.isfinite(hd_vals).all() else float("inf"),
                    "episode_hd95_median_mm": float(np.nanmedian(np.where(np.isfinite(hd_vals), hd_vals, np.nan))),
                    "episode_hd95_p90_mm": float(np.nanquantile(np.where(np.isfinite(hd_vals), hd_vals, np.nan), 0.90)),
                    "patient_recall_pct": float(pat["recall"].mean() * 100.0),
                    "episode_recall_pct": float(g["recall"].mean() * 100.0),
                    "patient_detached_fp": float(pat["detached_fp"].mean()),
                    "patient_removed_tp": float(pat["removed_tp"].mean()),
                    "patient_removed_fp": float(pat["removed_fp"].mean()),
                    "empty_predictions": int(g["empty_prediction"].sum()),
                    "nonfinite_hd95": int((~np.isfinite(hd_vals)).sum()),
                }
            )
    return pd.DataFrame(rows)


def json_safe(obj):
    """Recursively replace non-finite floats with None for strict JSON output."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def paired_bootstrap_contrasts(
    df: pd.DataFrame,
    baselines: tuple[str, ...] = ("full10", "positive0", "boundary_near2"),
    replicates: int = 5000,
    seed: int = 42,
) -> dict:
    """Compute paired 5,000-rep patient-cluster bootstrap intervals matching controls_v1."""
    patients = sorted(df["patient_id"].unique())
    n_pat = len(patients)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, n_pat, size=(replicates, n_pat))

    tables = {
        ("cc", "dice"): _patient_level_table(df, "cc", "dice").to_numpy() * 100.0,
        ("raw", "dice"): _patient_level_table(df, "raw", "dice").to_numpy() * 100.0,
        ("cc", "hd95"): _patient_level_table(df, "cc", "hd95").to_numpy(),
        ("cc", "recall"): _patient_level_table(df, "cc", "recall").to_numpy() * 100.0,
    }
    cols = list(_patient_level_table(df, "cc", "dice").columns)

    contrasts: list[dict] = []
    for base in baselines:
        if base not in cols:
            continue
        bi = cols.index(base)
        for arm in cols:
            if arm == base:
                continue
            ai = cols.index(arm)
            entry: dict = {"arm": arm, "baseline": base}
            for (post, metric), mat in tables.items():
                key = f"{post}_{metric}"
                diff = mat[:, ai] - mat[:, bi]
                if not np.isfinite(diff).all():
                    entry[key] = {"point": None, "ci95": None, "nonfinite": int((~np.isfinite(diff)).sum())}
                    continue
                point = float(diff.mean())
                boot = diff[sample].mean(axis=1)
                ci = [float(x) for x in np.quantile(boot, [0.025, 0.975])]
                entry[key] = {"point": point, "ci95": ci, "nonfinite": 0}
            contrasts.append(entry)

    # Per-organ patient-balanced breakdown on CC Dice, HD95, and Recall.
    d_cc = df[df["post"] == "cc"]
    per_organ: list[dict] = []
    for organ in sorted(d_cc["organ"].unique()):
        sub = d_cc[d_cc["organ"] == organ]
        pat_org = sub.groupby(["patient_id", "arm"])[["dice", "hd95", "recall"]].mean().reset_index()
        for arm in cols:
            g = pat_org[pat_org["arm"] == arm]
            hd = g["hd95"].to_numpy(dtype=float)
            per_organ.append(
                {
                    "organ": str(organ),
                    "arm": arm,
                    "patients": int(g["patient_id"].nunique()),
                    "patient_dice_pct": float(g["dice"].mean() * 100.0),
                    "patient_hd95_mm": float(np.mean(hd)) if np.isfinite(hd).all() else float("inf"),
                    "patient_recall_pct": float(g["recall"].mean() * 100.0),
                }
            )

    return json_safe({
        "replicates": int(replicates),
        "seed": int(seed),
        "patients": int(n_pat),
        "contrasts": contrasts,
        "per_organ_cc": per_organ,
    })


def evaluate_preregistered_decision(estimands_df: pd.DataFrame, contrasts: dict) -> dict:
    """Apply pre-registered Setting I decision criteria against boundary_near2 and full10."""
    by_pair = {(c["arm"], c["baseline"]): c for c in contrasts["contrasts"]}
    summary_rows: list[dict] = []
    passing_arms: list[str] = []

    cc_est = estimands_df[estimands_df["post"] == "cc"].set_index("arm")
    for arm in NEW_ARMS:
        if arm not in cc_est.index:
            continue
        c_near = by_pair.get((arm, "boundary_near2"), {})
        c_full = by_pair.get((arm, "full10"), {})
        d_near = c_near.get("cc_dice", {})
        r_full = c_full.get("cc_recall", {})
        h_near = c_near.get("cc_hd95", {})

        delta_near = d_near.get("point")
        ci_near = d_near.get("ci95")
        delta_rec_full = r_full.get("point")
        ci_rec_full = r_full.get("ci95")

        passed = bool(
            delta_near is not None
            and ci_near is not None
            and delta_rec_full is not None
            and ci_rec_full is not None
            and delta_near >= 0.50
            and ci_near[0] > 0.0
            and delta_rec_full >= -0.15
            and ci_rec_full[0] >= -0.50
        )
        if passed:
            passing_arms.append(arm)

        summary_rows.append(
            {
                "arm": arm,
                "patient_cc_dice_pct": float(cc_est.loc[arm, "patient_dice_pct"]),
                "episode_cc_dice_pct": float(cc_est.loc[arm, "episode_dice_pct"]),
                "patient_cc_hd95_mm": float(cc_est.loc[arm, "patient_hd95_mm"]),
                "patient_cc_recall_pct": float(cc_est.loc[arm, "patient_recall_pct"]),
                "delta_vs_boundary_near2_pp": delta_near,
                "ci95_vs_boundary_near2_pp": ci_near,
                "delta_hd95_vs_boundary_near2_mm": h_near.get("point"),
                "ci95_hd95_vs_boundary_near2_mm": h_near.get("ci95"),
                "delta_vs_full10_pp": c_full.get("cc_dice", {}).get("point"),
                "ci95_vs_full10_pp": c_full.get("cc_dice", {}).get("ci95"),
                "recall_vs_full10_pp": delta_rec_full,
                "recall_ci95_vs_full10_pp": ci_rec_full,
                "passes_preregistered_bar": passed,
            }
        )

    decision = "CANDIDATE_PASS" if passing_arms else "STOP_BOUNDARY_NEAR2_SUFFICIENT"
    return json_safe({
        "decision": decision,
        "passing_arms": passing_arms,
        "criteria": {
            "delta_vs_boundary_near2_pp_min": 0.50,
            "ci95_low_vs_boundary_near2_gt": 0.0,
            "recall_vs_full10_pp_min": -0.15,
            "recall_ci95_low_vs_full10_min": -0.50,
        },
        "arms": summary_rows,
    })
