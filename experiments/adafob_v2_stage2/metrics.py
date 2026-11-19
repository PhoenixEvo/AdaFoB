"""3D volumetric metrics, patient-balanced estimands, and Stage-2 decision evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.option_a_reassessment_v1.metrics import (
    disjoint_fp_3d,
    independent_hd95_mm,
    json_safe,
    mask_sha256,
)

from .selectors import ARMS

NEW_ARMS = ARMS[4:]


def measure_episode_stage2(
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
    """Compute Raw and CC 3D episode metrics across all 11 Stage-2 arms."""
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


def compute_estimands_stage2(df: pd.DataFrame) -> pd.DataFrame:
    """Compute patient-balanced and episode-weighted summaries across Stage-2 arms."""
    rows: list[dict] = []
    for post in ("raw", "cc"):
        d_post = df[df["post"] == post]
        for arm in ARMS:
            g = d_post[d_post["arm"] == arm]
            if g.empty:
                continue
            rep = g.groupby(["patient_id", "organ"])[
                ["dice", "hd95", "recall", "detached_fp", "fp", "fn", "removed_tp", "removed_fp"]
            ].mean()
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


def paired_bootstrap_contrasts_stage2(
    df: pd.DataFrame,
    baselines: tuple[str, ...] = ("full10", "positive0", "boundary_near2", "pba_f10_int_k2"),
    replicates: int = 5000,
    seed: int = 42,
) -> dict:
    """Compute 5,000-rep patient-cluster bootstrap CIs and per-organ Dice/HD95/Recall."""
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

    return json_safe(
        {
            "replicates": int(replicates),
            "seed": int(seed),
            "patients": int(n_pat),
            "contrasts": contrasts,
            "per_organ_cc": per_organ,
        }
    )


def evaluate_stage2_decision(estimands_df: pd.DataFrame, contrasts: dict) -> dict:
    """Evaluate all 7 Stage-2 arms against locked Stage-1 pass criteria plus Pareto diagnostic."""
    by_pair = {(c["arm"], c["baseline"]): c for c in contrasts["contrasts"]}
    per_org = {(r["organ"], r["arm"]): r for r in contrasts["per_organ_cc"]}
    cc_est = estimands_df[estimands_df["post"] == "cc"].set_index("arm")

    summary_rows: list[dict] = []
    passing_arms: list[str] = []
    pareto_arms: list[str] = []

    for arm in NEW_ARMS:
        if arm not in cc_est.index:
            continue
        c_near = by_pair.get((arm, "boundary_near2"), {})
        c_full = by_pair.get((arm, "full10"), {})
        c_pba = by_pair.get((arm, "pba_f10_int_k2"), {})

        d_near = c_near.get("cc_dice", {})
        h_near = c_near.get("cc_hd95", {})
        r_full = c_full.get("cc_recall", {})

        delta_near = d_near.get("point")
        ci_near = d_near.get("ci95")
        delta_hd_near = h_near.get("point")
        delta_rec_full = r_full.get("point")
        ci_rec_full = r_full.get("ci95")
        liver_rec = per_org.get(("LIVER", arm), {}).get("patient_recall_pct")
        liver_dice = per_org.get(("LIVER", arm), {}).get("patient_dice_pct")
        spleen_dice = per_org.get(("SPLEEN", arm), {}).get("patient_dice_pct")

        passed = bool(
            delta_near is not None
            and ci_near is not None
            and delta_hd_near is not None
            and delta_rec_full is not None
            and ci_rec_full is not None
            and liver_rec is not None
            and delta_near >= 1.00
            and ci_near[0] > 0.0
            and delta_rec_full >= -0.15
            and ci_rec_full[0] >= -0.50
            and liver_rec >= 91.0
            and delta_hd_near < 0.0
        )
        pareto_strong = bool(
            delta_near is not None
            and ci_near is not None
            and delta_hd_near is not None
            and liver_rec is not None
            and delta_near >= 1.00
            and ci_near[0] > 0.0
            and liver_rec >= 90.0
            and delta_hd_near <= -2.0
        )
        if passed:
            passing_arms.append(arm)
        if pareto_strong:
            pareto_arms.append(arm)

        summary_rows.append(
            {
                "arm": arm,
                "patient_cc_dice_pct": float(cc_est.loc[arm, "patient_dice_pct"]),
                "episode_cc_dice_pct": float(cc_est.loc[arm, "episode_dice_pct"]),
                "patient_cc_hd95_mm": float(cc_est.loc[arm, "patient_hd95_mm"]),
                "patient_cc_recall_pct": float(cc_est.loc[arm, "patient_recall_pct"]),
                "liver_recall_pct": liver_rec,
                "liver_dice_pct": liver_dice,
                "spleen_dice_pct": spleen_dice,
                "delta_vs_boundary_near2_pp": delta_near,
                "ci95_vs_boundary_near2_pp": ci_near,
                "delta_hd95_vs_boundary_near2_mm": delta_hd_near,
                "ci95_hd95_vs_boundary_near2_mm": h_near.get("ci95"),
                "delta_vs_pba_f10_int_k2_pp": c_pba.get("cc_dice", {}).get("point"),
                "ci95_vs_pba_f10_int_k2_pp": c_pba.get("cc_dice", {}).get("ci95"),
                "recall_vs_pba_f10_int_k2_pp": c_pba.get("cc_recall", {}).get("point"),
                "delta_vs_full10_pp": c_full.get("cc_dice", {}).get("point"),
                "ci95_vs_full10_pp": c_full.get("cc_dice", {}).get("ci95"),
                "recall_vs_full10_pp": delta_rec_full,
                "recall_ci95_vs_full10_pp": ci_rec_full,
                "passes_stage2_bar": passed,
                "pareto_strong": pareto_strong,
            }
        )

    if passing_arms:
        decision = "STAGE2_PASS_PROCEED_MULTIDATASET"
    elif pareto_arms:
        decision = "STAGE2_PARETO_STRONG_RECALL_REVIEW"
    else:
        decision = "STAGE2_ITERATE_OR_STOP"

    return json_safe(
        {
            "decision": decision,
            "passing_arms": passing_arms,
            "pareto_strong_arms": pareto_arms,
            "criteria": {
                "delta_vs_boundary_near2_pp_min": 1.00,
                "ci95_low_vs_boundary_near2_gt": 0.0,
                "recall_vs_full10_pp_min": -0.15,
                "recall_ci95_low_vs_full10_min": -0.50,
                "liver_recall_pct_min": 91.0,
                "delta_hd95_vs_boundary_near2_mm_lt": 0.0,
            },
            "secondary_pareto_criteria": {
                "delta_vs_boundary_near2_pp_min": 1.00,
                "ci95_low_vs_boundary_near2_gt": 0.0,
                "liver_recall_pct_min": 90.0,
                "delta_hd95_vs_boundary_near2_mm_max": -2.0,
            },
            "arms": summary_rows,
        }
    )
