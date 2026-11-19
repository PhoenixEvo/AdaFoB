"""3D volumetric metrics, 5-seed random aggregation, and Stage-3 cross-modality decision checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.option_a_reassessment_v1.metrics import (
    disjoint_fp_3d,
    independent_hd95_mm,
    json_safe,
    mask_sha256,
)

from .selectors import ALL_REPORTED_ARMS, METHOD_ARMS, RANDOM_SEEDS, RAW_ARMS, SUMMARY_ARMS


def measure_episode_stage3(
    backbone: str,
    orientation: str,
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
    """Compute Raw and CC 3D episode metrics across all 12 RAW_ARMS + random2_mean5 aggregate."""
    gt = np.asarray(gt_3d, dtype=bool)
    other = np.asarray(other_3d, dtype=bool)
    sp_xyz = tuple(float(x) for x in spacing_xyz)
    sp_zyx = (sp_xyz[2], sp_xyz[1], sp_xyz[0])
    voxel_vol = float(np.prod(sp_xyz))
    gt_vox = int(gt.sum())

    rows: list[dict] = []
    for arm in RAW_ARMS:
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
                    "backbone": str(backbone),
                    "orientation": str(orientation),
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

    mean_fields = (
        "removed_tp",
        "removed_fp",
        "dice",
        "hd95",
        "recall",
        "tp",
        "fp",
        "fn",
        "detached_fp",
        "other_annotated_organ_fp",
        "residual_fp",
        "directed_pred_to_gt95",
        "directed_gt_to_pred95",
        "fp_mm3",
        "fn_mm3",
    )
    seed_arms = {f"random2_s{s}" for s in RANDOM_SEEDS}
    for post in ("raw", "cc"):
        sub = [r for r in rows if r["post"] == post and r["arm"] in seed_arms]
        assert len(sub) == len(RANDOM_SEEDS)
        agg = {
            "backbone": str(backbone),
            "orientation": str(orientation),
            "episode_id": int(episode_id),
            "fold": int(fold),
            "patient_id": str(patient_id),
            "organ": str(organ),
            "arm": "random2_mean5",
            "post": post,
            "empty_prediction": any(bool(r["empty_prediction"]) for r in sub),
            "gt_voxels": gt_vox,
            "metric_status": "finite" if all(np.isfinite(r["hd95"]) for r in sub) else "undefined_one_empty",
            "mask_sha256": "mean5_aggregate",
        }
        for field in mean_fields:
            vals = [float(r[field]) for r in sub]
            agg[field] = float(np.mean(vals)) if all(np.isfinite(v) for v in vals) else float("inf")
        rows.append(agg)

    return rows


def _patient_level_table(df_bb: pd.DataFrame, post: str, metric: str) -> pd.DataFrame:
    d = df_bb[df_bb["post"] == post]
    rep = d.groupby(["patient_id", "organ", "arm"], as_index=False)[metric].mean()
    pat = rep.groupby(["patient_id", "arm"])[metric].mean().unstack("arm")
    patients = sorted(d["patient_id"].unique())
    arms = [a for a in ALL_REPORTED_ARMS if a in pat.columns]
    return pat.reindex(index=patients, columns=arms)


def compute_estimands_stage3(df: pd.DataFrame) -> pd.DataFrame:
    """Compute patient-balanced and episode-weighted summaries per backbone and arm."""
    rows: list[dict] = []
    for backbone in sorted(df["backbone"].unique()):
        d_bb = df[df["backbone"] == backbone]
        for post in ("raw", "cc"):
            d_post = d_bb[d_bb["post"] == post]
            for arm in ALL_REPORTED_ARMS:
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
                        "backbone": str(backbone),
                        "arm": arm,
                        "post": post,
                        "is_summary_arm": bool(arm in SUMMARY_ARMS),
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


def paired_bootstrap_contrasts_stage3(
    df: pd.DataFrame,
    baselines: tuple[str, ...] = ("full10", "positive0", "random2_mean5", "boundary_near2", "pba_f10_int_k2"),
    replicates: int = 5000,
    seed: int = 42,
) -> dict:
    """Compute 5,000-rep patient-cluster bootstrap CIs per backbone and vit_h-vs-vit_b deltas."""
    contrasts: list[dict] = []
    per_organ: list[dict] = []
    backbones = sorted(df["backbone"].unique())
    pat_tables_by_bb: dict[tuple[str, str, str], pd.DataFrame] = {}

    rng_seed = int(seed)
    for backbone in backbones:
        d_bb = df[df["backbone"] == backbone]
        patients = sorted(d_bb["patient_id"].unique())
        n_pat = len(patients)
        rng = np.random.default_rng(rng_seed)
        sample = rng.integers(0, n_pat, size=(replicates, n_pat))

        tbl_cc_dice = _patient_level_table(d_bb, "cc", "dice") * 100.0
        tbl_raw_dice = _patient_level_table(d_bb, "raw", "dice") * 100.0
        tbl_cc_hd = _patient_level_table(d_bb, "cc", "hd95")
        tbl_cc_rec = _patient_level_table(d_bb, "cc", "recall") * 100.0
        pat_tables_by_bb[(backbone, "cc", "dice")] = tbl_cc_dice
        pat_tables_by_bb[(backbone, "raw", "dice")] = tbl_raw_dice
        pat_tables_by_bb[(backbone, "cc", "hd95")] = tbl_cc_hd
        pat_tables_by_bb[(backbone, "cc", "recall")] = tbl_cc_rec

        tables = {
            ("cc", "dice"): tbl_cc_dice.to_numpy(),
            ("raw", "dice"): tbl_raw_dice.to_numpy(),
            ("cc", "hd95"): tbl_cc_hd.to_numpy(),
            ("cc", "recall"): tbl_cc_rec.to_numpy(),
        }
        cols = list(tbl_cc_dice.columns)

        for base in baselines:
            if base not in cols:
                continue
            bi = cols.index(base)
            for arm in cols:
                if arm == base or arm.startswith("random2_s"):
                    continue
                ai = cols.index(arm)
                entry: dict = {"backbone": str(backbone), "arm": arm, "baseline": base}
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

        d_cc = d_bb[d_bb["post"] == "cc"]
        for organ in sorted(d_cc["organ"].unique()):
            sub = d_cc[d_cc["organ"] == organ]
            pat_org = sub.groupby(["patient_id", "arm"])[["dice", "hd95", "recall"]].mean().reset_index()
            for arm in cols:
                g = pat_org[pat_org["arm"] == arm]
                hd = g["hd95"].to_numpy(dtype=float)
                per_organ.append(
                    {
                        "backbone": str(backbone),
                        "organ": str(organ),
                        "arm": arm,
                        "patients": int(g["patient_id"].nunique()),
                        "patient_dice_pct": float(g["dice"].mean() * 100.0),
                        "patient_hd95_mm": float(np.mean(hd)) if np.isfinite(hd).all() else float("inf"),
                        "patient_recall_pct": float(g["recall"].mean() * 100.0),
                    }
                )

    backbone_scaling: list[dict] = []
    if "vit_b" in backbones and "vit_h" in backbones:
        b_cc = pat_tables_by_bb[("vit_b", "cc", "dice")]
        h_cc = pat_tables_by_bb[("vit_h", "cc", "dice")]
        common_pats = sorted(set(b_cc.index) & set(h_cc.index))
        common_arms = [a for a in SUMMARY_ARMS if a in b_cc.columns and a in h_cc.columns]
        rng = np.random.default_rng(rng_seed)
        sample_bb = rng.integers(0, len(common_pats), size=(replicates, len(common_pats)))
        for arm in common_arms:
            entry: dict = {"arm": arm, "comparison": "vit_h_minus_vit_b"}
            for (post, metric) in (("cc", "dice"), ("raw", "dice"), ("cc", "hd95"), ("cc", "recall")):
                tb = pat_tables_by_bb[("vit_b", post, metric)].reindex(index=common_pats)
                th = pat_tables_by_bb[("vit_h", post, metric)].reindex(index=common_pats)
                diff = th[arm].to_numpy(dtype=float) - tb[arm].to_numpy(dtype=float)
                key = f"{post}_{metric}"
                if not np.isfinite(diff).all():
                    entry[key] = {"point": None, "ci95": None, "nonfinite": int((~np.isfinite(diff)).sum())}
                else:
                    boot = diff[sample_bb].mean(axis=1)
                    entry[key] = {
                        "point": float(diff.mean()),
                        "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
                        "nonfinite": 0,
                    }
            backbone_scaling.append(entry)

    return json_safe(
        {
            "replicates": int(replicates),
            "seed": int(seed),
            "patients": int(df["patient_id"].nunique()),
            "backbones": backbones,
            "contrasts": contrasts,
            "per_organ_cc": per_organ,
            "backbone_scaling": backbone_scaling,
        }
    )


def evaluate_stage3_decision(estimands_df: pd.DataFrame, contrasts: dict) -> dict:
    """Evaluate Stage-3 OOD CHAOST2 generalization and recall protection per backbone."""
    by_pair = {(c["backbone"], c["arm"], c["baseline"]): c for c in contrasts["contrasts"]}
    per_org = {(r["backbone"], r["organ"], r["arm"]): r for r in contrasts["per_organ_cc"]}
    cc_est = estimands_df[estimands_df["post"] == "cc"].set_index(["backbone", "arm"])

    rows: list[dict] = []
    ood_passing: list[dict] = []
    recall_protected_passing: list[dict] = []

    for backbone in sorted(estimands_df["backbone"].unique()):
        for arm in METHOD_ARMS:
            if (backbone, arm) not in cc_est.index:
                continue
            c_near = by_pair.get((backbone, arm, "boundary_near2"), {})
            c_rand = by_pair.get((backbone, arm, "random2_mean5"), {})
            c_full = by_pair.get((backbone, arm, "full10"), {})
            c_pba = by_pair.get((backbone, arm, "pba_f10_int_k2"), {})

            d_near = c_near.get("cc_dice", {})
            d_rand = c_rand.get("cc_dice", {})
            d_full = c_full.get("cc_dice", {})
            h_near = c_near.get("cc_hd95", {})
            r_full = c_full.get("cc_recall", {})
            r_pba = c_pba.get("cc_recall", {})

            delta_near = d_near.get("point")
            ci_near = d_near.get("ci95")
            delta_rand = d_rand.get("point")
            delta_full = d_full.get("point")
            delta_hd_near = h_near.get("point")
            rec_vs_full = r_full.get("point")
            rec_vs_pba = r_pba.get("point")
            ci_rec_pba = r_pba.get("ci95")

            ood_pass = bool(
                delta_near is not None
                and ci_near is not None
                and delta_rand is not None
                and delta_full is not None
                and delta_hd_near is not None
                and delta_near > 0.0
                and ci_near[0] > 0.0
                and delta_rand > 0.0
                and delta_full > 0.0
                and delta_hd_near < 0.0
            )
            recall_ok = bool(
                (rec_vs_full is not None and rec_vs_full >= -0.50)
                or (rec_vs_pba is not None and ci_rec_pba is not None and rec_vs_pba >= 0.25 and ci_rec_pba[0] > 0.0)
            )
            recall_prot_pass = bool(ood_pass and recall_ok)
            if ood_pass:
                ood_passing.append({"backbone": backbone, "arm": arm})
            if recall_prot_pass:
                recall_protected_passing.append({"backbone": backbone, "arm": arm})

            rows.append(
                {
                    "backbone": backbone,
                    "arm": arm,
                    "patient_cc_dice_pct": float(cc_est.loc[(backbone, arm), "patient_dice_pct"]),
                    "episode_cc_dice_pct": float(cc_est.loc[(backbone, arm), "episode_dice_pct"]),
                    "patient_cc_hd95_mm": float(cc_est.loc[(backbone, arm), "patient_hd95_mm"]),
                    "patient_cc_recall_pct": float(cc_est.loc[(backbone, arm), "patient_recall_pct"]),
                    "liver_dice_pct": per_org.get((backbone, "LIVER", arm), {}).get("patient_dice_pct"),
                    "liver_recall_pct": per_org.get((backbone, "LIVER", arm), {}).get("patient_recall_pct"),
                    "spleen_dice_pct": per_org.get((backbone, "SPLEEN", arm), {}).get("patient_dice_pct"),
                    "delta_vs_boundary_near2_pp": delta_near,
                    "ci95_vs_boundary_near2_pp": ci_near,
                    "delta_vs_random2_mean5_pp": delta_rand,
                    "ci95_vs_random2_mean5_pp": d_rand.get("ci95"),
                    "delta_vs_full10_pp": delta_full,
                    "ci95_vs_full10_pp": d_full.get("ci95"),
                    "delta_hd95_vs_boundary_near2_mm": delta_hd_near,
                    "recall_vs_full10_pp": rec_vs_full,
                    "recall_vs_pba_f10_int_k2_pp": rec_vs_pba,
                    "recall_ci95_vs_pba_f10_int_k2_pp": ci_rec_pba,
                    "passes_ood_bar": ood_pass,
                    "passes_recall_protected_bar": recall_prot_pass,
                }
            )

    if recall_protected_passing:
        decision = "STAGE3_CHAOST2_RECALL_PROTECTED_GENERALIZATION"
    elif ood_passing:
        decision = "STAGE3_CHAOST2_OOD_GAIN_WITH_RECALL_TRADEOFF"
    else:
        decision = "STAGE3_CHAOST2_INCONCLUSIVE_OR_NEGATIVE"

    return json_safe(
        {
            "decision": decision,
            "ood_passing": ood_passing,
            "recall_protected_passing": recall_protected_passing,
            "arms": rows,
        }
    )
