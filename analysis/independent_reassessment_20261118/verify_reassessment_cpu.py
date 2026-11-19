"""Independent read-only CPU verification script for AdaFoB reassessment.

Reads authorized CSVs, JSON receipts, manifests, and saved prediction NPZs
(without reading raw query-label NIfTI files) and exports a machine-verifiable
audit summary to cpu_verification_summary.json.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(args: list[str]) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout.strip()


def verify_environment_and_provenance() -> dict:
    import SimpleITK as sitk
    import medpy
    import scipy
    import torch

    head = run_git(["rev-parse", "HEAD"])
    tracked_dirty = run_git(["diff", "--name-only"]).splitlines()

    missing_or_present = {
        "sam_vit_b_local_found": len(list(REPO_ROOT.glob("**/sam_vit_b*.pth"))),
        "sam_vit_h_local_found": len(list(REPO_ROOT.glob("**/sam_vit_h*.pth"))),
        "step5_subset_masks_present": False,
        "brats_raw_data_present": False,
        "local_sabs_ct_normalized_files": len(
            list((REPO_ROOT / "data" / "sabs_CT_normalized").glob("image_*.nii*"))
        )
        if (REPO_ROOT / "data" / "sabs_CT_normalized").exists()
        else 0,
    }
    step5_zip = REPO_ROOT / "results" / "step5_diagnostics" / "step5_diagnostics_artifacts.zip"
    if step5_zip.is_file():
        import zipfile

        with zipfile.ZipFile(step5_zip, "r") as zf:
            npz_members = [n for n in zf.namelist() if n.endswith(".npz")]
            missing_or_present["step5_zip_npz_count"] = len(npz_members)
            missing_or_present["step5_subset_masks_present"] = len(npz_members) > 0

    key_files = [
        "protocol/paired_episode_manifest.csv",
        "protocol/pilot_manifest_v1.csv",
        "results/experiment_manifest.csv",
        "results/r3_global_params.json",
        "results/clean_rerun_rank4_seed42_combined.csv",
        "results/clean_rerun_rank16_unseeded_combined.csv",
        "results/lora_eval_rank2_combined.csv",
        "results/lora_eval_rank4_combined.csv",
        "results/lora_eval_rank8_combined.csv",
        "results/lora_eval_rank16_combined.csv",
        "results/step9_controls_20261024T034851Z/cpu_analysis_v3_20261025T024935616Z_1e6a9939/measurements.csv",
        "results/step9_controls_20261024T034851Z/cpu_analysis_v3_20261025T024935616Z_1e6a9939/both_estimands.csv",
        "results/step9_controls_20261024T034851Z/cpu_analysis_v3_20261025T024935616Z_1e6a9939/slice_records.csv",
        "third_party/FoB_SAM/models/FoB.py",
        "third_party/FoB_SAM/models/allocator.py",
        "third_party/FoB_SAM_upstream/models/FoB.py",
        "models/postprocessing.py",
        "experiments/eval_lora.py",
        "experiments/gate_d_adafob_eval.py",
        "experiments/train_lora.py",
        "experiments/utility_feasibility_v1/scoring_v1.py",
    ]
    hashes = {}
    for rel in key_files:
        p = REPO_ROOT / rel
        if p.is_file():
            hashes[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}

    ckpt_hashes = {}
    for fold in range(5):
        p = (
            REPO_ROOT
            / "checkpoints"
            / "fob_checkpoints"
            / "exps_train_on_SABS_FSMIS_FoB"
            / f"FSMIS_train_SABS_cv{fold}"
            / "ckpt.pth"
        )
        if p.is_file():
            ckpt_hashes[f"fob_cv{fold}"] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    for rel in [
        "outputs/rank4_fold0_1_results/lora_fold0_best.pth",
        "outputs/rank4_fold0_1_results/lora_fold1_best.pth",
        "outputs/rank4_fold2_3_results/lora_fold2_best.pth",
        "outputs/rank4_fold2_3_results/lora_fold3_best.pth",
        "outputs/rank4_fold4_results/lora_fold4_best.pth",
    ]:
        p = REPO_ROOT / rel
        if p.is_file():
            ckpt_hashes[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "SimpleITK": sitk.__version__,
            "medpy": getattr(medpy, "__version__", "unknown"),
            "torch": torch.__version__,
        },
        "git_head": head,
        "tracked_dirty_files": tracked_dirty,
        "local_artifact_availability": missing_or_present,
        "file_hashes": hashes,
        "checkpoint_hashes": ckpt_hashes,
    }


def verify_patient_roles() -> dict:
    m_df = pd.read_csv(REPO_ROOT / "protocol" / "paired_episode_manifest.csv")
    m_df["patient_id"] = (
        m_df["vol_id"].str.replace("\\", "/", regex=False).str.split("/").str[-1].str.split(".nii").str[0]
    )

    folds = {
        0: set(range(0, 7)),
        1: set(range(6, 13)),
        2: set(range(12, 19)),
        3: set(range(18, 25)),
        4: set(range(24, 30)) | {0},
    }
    fold_summary = {}
    all_query_occurrences = 0
    for f, pset in folds.items():
        sorted_pids = sorted(pset)
        supp_idx = sorted_pids[2]
        query_pids = [p for i, p in enumerate(sorted_pids) if i != 2]
        train_pids = sorted(set(range(30)) - pset)
        manifest_q = sorted(
            m_df[m_df["fold"] == f]["patient_id"]
            .apply(lambda x: int(x.split("_")[1]))
            .unique()
            .tolist()
        )
        assert manifest_q == query_pids, f"Fold {f} query mismatch: {manifest_q} vs {query_pids}"
        all_query_occurrences += len(manifest_q)
        fold_summary[str(f)] = {
            "fold_patients": sorted_pids,
            "support_patient": f"image_{supp_idx}",
            "query_patients": [f"image_{p}" for p in query_pids],
            "train_patients_count": len(train_pids),
            "episodes_in_fold": int((m_df["fold"] == f).sum()),
        }

    pat_folds = m_df.groupby("patient_id")["fold"].unique().apply(lambda x: sorted(x.tolist())).to_dict()
    repeated_patients = {k: v for k, v in pat_folds.items() if len(v) > 1}
    support_patients = {fold_summary[str(f)]["support_patient"] for f in range(5)}
    overlap_support_query = sorted(support_patients & set(pat_folds.keys()))

    rb_manifest = pd.read_csv(REPO_ROOT / "protocol" / "pilot_manifest_v1.csv")
    rb_roles = rb_manifest[["patient", "role", "organ", "class_id", "slice_index", "support_patient", "support_slice_index"]].to_dict(orient="records")

    return {
        "total_episodes": int(len(m_df)),
        "total_patient_fold_occurrences": int(all_query_occurrences),
        "unique_query_patients": int(m_df["patient_id"].nunique()),
        "repeated_query_patients_across_folds": repeated_patients,
        "support_patients_5fold": sorted(support_patients),
        "support_in_120_query_episodes": overlap_support_query,
        "fold_summary": fold_summary,
        "route_b_manifest_cases": rb_roles,
        "route_b_patients_in_fob_cv0_supervoxel_train": {
            "image_8": 8 not in folds[0],
            "image_14": 14 not in folds[0],
            "image_20": 20 not in folds[0],
        },
    }


def verify_step9_controls() -> dict:
    base = REPO_ROOT / "results" / "step9_controls_20261024T034851Z" / "cpu_analysis_v3_20261025T024935616Z_1e6a9939"
    df = pd.read_csv(base / "measurements.csv")
    be = pd.read_csv(base / "both_estimands.csv")
    sr = pd.read_csv(base / "slice_records.csv")
    ea = pd.read_csv(base / "episode_audit.csv")
    gc = pd.read_csv(base / "greedy_counterexamples.csv")
    pc_ref = json.loads((base / "paired_contrasts.json").read_text(encoding="utf-8"))
    gap_ref = json.loads((base / "gap_closure.json").read_text(encoding="utf-8"))
    dec_ref = json.loads((base / "decision.json").read_text(encoding="utf-8"))

    assert df.shape == (7440, 27)
    assert df["episode_id"].nunique() == 120
    assert df["patient_id"].nunique() == 25
    assert df["arm"].nunique() == 31
    assert int(df["empty_prediction"].sum()) == 0
    assert float(df["reference_dice_difference"].abs().max()) == 0.0
    assert float(df["hd95_difference"].abs().max()) == 0.0

    # Add 5-seed random averages matching controls_v1.statistics.add_seed_means
    rows = []
    keys = ["episode_id", "fold", "patient_id", "organ", "post"]
    num_cols = [c for c in df.columns if c not in keys + ["arm"] and pd.api.types.is_numeric_dtype(df[c])]
    for k in (2, 4, 6):
        g = df[df["arm"].isin([f"random{k}_s{s}" for s in range(42, 47)])]
        m = g.groupby(keys)[num_cols].mean().reset_index()
        m["arm"] = f"random{k}_mean5"
        rows.append(m)
    df_ext = pd.concat([df, *rows], ignore_index=True)

    # Verify all estimands against both_estimands.csv
    max_be_diff = 0.0
    arms_summary = {}
    key_arms = [
        "full10",
        "positive0",
        "prefix2",
        "prefix4",
        "prefix6",
        "random2_mean5",
        "random4_mean5",
        "random6_mean5",
        "fps2",
        "fps4",
        "fps6",
        "boundary_near2",
        "boundary_near4",
        "boundary_near6",
        "boundary_far2",
        "boundary_far4",
        "boundary_far6",
        "GT_prefix_reference",
        "GT_greedy_reference",
    ]

    for post in ("raw", "cc"):
        arms_summary[post] = {}
        sub = df_ext[df_ext["post"] == post]
        for arm in sorted(df_ext["arm"].unique()):
            adf = sub[sub["arm"] == arm]
            po = adf.groupby(["patient_id", "organ"])[["dice", "hd95", "recall", "fp", "fn", "fp_mm3", "fn_mm3", "detached_fp", "other_annotated_organ_fp", "residual_fp", "removed_tp", "removed_fp"]].mean()
            pat_df = po.groupby("patient_id").mean()
            ep_means = adf[["dice", "hd95", "recall", "fp", "fn", "fp_mm3", "fn_mm3"]].mean()
            ep_stds = adf[["dice", "hd95", "recall"]].std(ddof=1)
            pat_means = pat_df.mean()
            pat_stds = pat_df[["dice", "hd95", "recall"]].std(ddof=1)

            for metric in ("dice", "hd95", "recall", "fp", "fn", "fp_mm3", "fn_mm3", "detached_fp", "other_annotated_organ_fp", "residual_fp", "removed_tp", "removed_fp"):
                ref_row = be[(be["organ"] == "ALL") & (be["arm"] == arm) & (be["post"] == post) & (be["metric"] == metric)]
                assert len(ref_row) == 1
                ep_val = float(ep_means[metric]) if metric in ep_means else float(adf[metric].mean())
                diff_ep = abs(ep_val - float(ref_row["episode_weighted"].iloc[0]))
                diff_pat = abs(float(pat_means[metric]) - float(ref_row["patient_balanced"].iloc[0]))
                max_be_diff = max(max_be_diff, diff_ep, diff_pat)

            if arm in key_arms:
                arms_summary[post][arm] = {
                    "episode_weighted": {
                        "dice_pct": float(ep_means["dice"] * 100.0),
                        "dice_sd_pct": float(ep_stds["dice"] * 100.0),
                        "hd95_mm": float(ep_means["hd95"]),
                        "hd95_sd_mm": float(ep_stds["hd95"]),
                        "hd95_median_mm": float(adf["hd95"].median()),
                        "hd95_p90_mm": float(np.quantile(adf["hd95"], 0.90)),
                        "recall_pct": float(ep_means["recall"] * 100.0),
                        "fp_voxels": float(ep_means["fp"]),
                        "fn_voxels": float(ep_means["fn"]),
                    },
                    "patient_balanced": {
                        "dice_pct": float(pat_means["dice"] * 100.0),
                        "dice_sd_pct": float(pat_stds["dice"] * 100.0),
                        "hd95_mm": float(pat_means["hd95"]),
                        "hd95_sd_mm": float(pat_stds["hd95"]),
                        "recall_pct": float(pat_means["recall"] * 100.0),
                        "fp_voxels": float(pat_means["fp"]),
                        "fn_voxels": float(pat_means["fn"]),
                        "detached_fp_voxels": float(pat_means["detached_fp"]),
                        "other_organ_fp_voxels": float(pat_means["other_annotated_organ_fp"]),
                        "residual_fp_voxels": float(pat_means["residual_fp"]),
                        "removed_tp_voxels": float(pat_means["removed_tp"]),
                        "removed_fp_voxels": float(pat_means["removed_fp"]),
                    },
                }

    assert max_be_diff < 1e-10, f"both_estimands discrepancy too large: {max_be_diff}"

    random_seed_spread = {}
    for k in (2, 4, 6):
        seed_vals = {}
        for s in range(42, 47):
            s_arm = f"random{k}_s{s}"
            s_df = df_ext[(df_ext["post"] == "cc") & (df_ext["arm"] == s_arm)]
            pat_val = s_df.groupby(["patient_id", "organ"])["dice"].mean().groupby("patient_id").mean().mean() * 100.0
            ep_val = s_df["dice"].mean() * 100.0
            seed_vals[s_arm] = {"patient_dice_pct": float(pat_val), "episode_dice_pct": float(ep_val)}
        random_seed_spread[f"random{k}"] = seed_vals

    # Independent 5,000-replicate paired patient-cluster bootstrap matching gap_closure.json
    cc = df_ext[df_ext["post"] == "cc"]
    eligible = sorted([
        "full10", "positive0",
        "prefix2", "prefix4", "prefix6",
        "fps2", "fps4", "fps6",
        "boundary_near2", "boundary_near4", "boundary_near6",
        "boundary_far2", "boundary_far4", "boundary_far6",
        "random2_mean5", "random4_mean5", "random6_mean5",
    ])
    arms_boot = eligible + ["GT_greedy_reference"]
    pm = cc.groupby(["patient_id", "organ", "arm"])[["dice", "hd95", "recall"]].mean().groupby(["patient_id", "arm"]).mean()
    patients = sorted(cc["patient_id"].unique())
    dice_mat = pm["dice"].unstack("arm").reindex(index=patients, columns=arms_boot).to_numpy()
    hd95_mat = pm["hd95"].unstack("arm").reindex(index=patients, columns=arms_boot).to_numpy()
    rec_mat = pm["recall"].unstack("arm").reindex(index=patients, columns=arms_boot).to_numpy()

    rng = np.random.default_rng(42)
    sample = rng.integers(0, len(patients), size=(5000, len(patients)))
    boot_dice = dice_mat[sample].mean(axis=1)
    boot_hd95 = hd95_mat[sample].mean(axis=1)
    boot_rec = rec_mat[sample].mean(axis=1)

    idx_full = arms_boot.index("full10")
    idx_near2 = arms_boot.index("boundary_near2")
    idx_greedy = arms_boot.index("GT_greedy_reference")
    idx_pos0 = arms_boot.index("positive0")
    idx_rand2 = arms_boot.index("random2_mean5")

    point_dice = dice_mat.mean(axis=0)
    gap_pp = float((point_dice[idx_greedy] - point_dice[idx_full]) * 100.0)
    closure_near2 = float((point_dice[idx_near2] - point_dice[idx_full]) / (point_dice[idx_greedy] - point_dice[idx_full]))
    boot_den = boot_dice[:, idx_greedy] - boot_dice[:, idx_full]
    winner_idx = np.argmax(boot_dice[:, :len(eligible)], axis=1)
    reselected_closure_ci = np.quantile((boot_dice[np.arange(5000), winner_idx] - boot_dice[:, idx_full]) / boot_den, [0.025, 0.975]).tolist()

    assert abs(gap_pp - gap_ref["corrected_gap_pp"]) < 1e-12
    assert abs(closure_near2 - gap_ref["best_gap_closure"]) < 1e-12
    assert np.allclose(reselected_closure_ci, gap_ref["best_reselected_ci95"], atol=1e-12)

    contrasts = {
        "boundary_near2_minus_full10_dice_pp": {
            "point": float((point_dice[idx_near2] - point_dice[idx_full]) * 100.0),
            "ci95": np.quantile((boot_dice[:, idx_near2] - boot_dice[:, idx_full]) * 100.0, [0.025, 0.975]).tolist(),
        },
        "boundary_near2_minus_full10_hd95_mm": {
            "point": float(hd95_mat.mean(0)[idx_near2] - hd95_mat.mean(0)[idx_full]),
            "ci95": np.quantile(boot_hd95[:, idx_near2] - boot_hd95[:, idx_full], [0.025, 0.975]).tolist(),
        },
        "boundary_near2_minus_full10_recall_pp": {
            "point": float((rec_mat.mean(0)[idx_near2] - rec_mat.mean(0)[idx_full]) * 100.0),
            "ci95": np.quantile((boot_rec[:, idx_near2] - boot_rec[:, idx_full]) * 100.0, [0.025, 0.975]).tolist(),
        },
        "boundary_near2_minus_random2_mean5_dice_pp": {
            "point": float((point_dice[idx_near2] - point_dice[idx_rand2]) * 100.0),
            "ci95": np.quantile((boot_dice[:, idx_near2] - boot_dice[:, idx_rand2]) * 100.0, [0.025, 0.975]).tolist(),
        },
        "boundary_near2_minus_positive0_dice_pp": {
            "point": float((point_dice[idx_near2] - point_dice[idx_pos0]) * 100.0),
            "ci95": np.quantile((boot_dice[:, idx_near2] - boot_dice[:, idx_pos0]) * 100.0, [0.025, 0.975]).tolist(),
        },
        "boundary_near2_minus_positive0_recall_pp": {
            "point": float((rec_mat.mean(0)[idx_near2] - rec_mat.mean(0)[idx_pos0]) * 100.0),
            "ci95": np.quantile((boot_rec[:, idx_near2] - boot_rec[:, idx_pos0]) * 100.0, [0.025, 0.975]).tolist(),
        },
        "greedy_minus_boundary_near2_dice_pp": {
            "point": float((point_dice[idx_greedy] - point_dice[idx_near2]) * 100.0),
            "ci95_reselected": np.quantile((boot_dice[:, idx_greedy] - boot_dice[np.arange(5000), winner_idx]) * 100.0, [0.025, 0.975]).tolist(),
        },
    }

    # Check per-organ breakdown for key arms in CC
    per_organ_cc = {}
    for org in sorted(cc["organ"].unique()):
        per_organ_cc[org] = {}
        for arm in ("full10", "positive0", "random2_mean5", "boundary_near2", "GT_greedy_reference"):
            odf = cc[(cc["organ"] == org) & (cc["arm"] == arm)]
            pat_org = odf.groupby("patient_id")[["dice", "hd95", "recall"]].mean().mean()
            per_organ_cc[org][arm] = {
                "patient_dice_pct": float(pat_org["dice"] * 100.0),
                "episode_dice_pct": float(odf["dice"].mean() * 100.0),
                "patient_hd95_mm": float(pat_org["hd95"]),
                "patient_recall_pct": float(pat_org["recall"] * 100.0),
            }

    # Slice records accounting
    sr_full = sr[sr["arm"] == "full10"]
    sr_near2 = sr[sr["arm"] == "boundary_near2"]
    sr_greedy = sr[sr["arm"] == "GT_greedy_reference"]
    worst_cc = {}
    for arm in ("full10", "positive0", "boundary_near2"):
        w = cc[cc["arm"] == arm].nsmallest(5, "dice")[["episode_id", "fold", "patient_id", "organ", "dice", "hd95", "recall"]]
        worst_cc[arm] = w.to_dict(orient="records")

    return {
        "max_both_estimands_discrepancy": float(max_be_diff),
        "decision_json": dec_ref,
        "gap_closure_summary": {
            "corrected_gap_pp": gap_pp,
            "best_control": gap_ref["best_simple_control"],
            "best_gap_closure_pct": closure_near2 * 100.0,
            "reselected_closure_ci95_pct": [x * 100.0 for x in reselected_closure_ci],
            "undefined_replicates": int((~(boot_den > 0)).sum()),
        },
        "recomputed_bootstrap_contrasts": contrasts,
        "random_seed_spread_cc": random_seed_spread,
        "arms": arms_summary,
        "per_organ_cc": per_organ_cc,
        "worst_5_episodes_cc": worst_cc,
        "slice_accounting": {
            "total_slices": int(sr_full.shape[0]),
            "total_encoder_calls": int(ea["encoder_calls"].sum()),
            "total_decoder_calls": int(ea["decoder_calls"].sum()),
            "total_greedy_logical_trials": int(ea["greedy_logical_trials"].sum()),
            "mean_positive_prompts": float(sr_full["positive_count"].mean()),
            "full10_gt_contaminated_negatives": int(sr_full["gt_contaminated_negatives"].sum()),
            "full10_oob_rounded_negatives": int(sr_full["negative_oob_rounded"].sum()),
            "near2_gt_contaminated_negatives": int(sr_near2["gt_contaminated_negatives"].sum()),
            "greedy_zero_negative_slices": int((sr_greedy["negative_count"] == 0).sum()),
            "greedy_mean_negative_count": float(sr_greedy["negative_count"].mean()),
            "greedy_counterexamples_rows": int(len(gc)),
            "greedy_worse_than_control_cc_episodes": int(gc[gc["post"] == "cc"]["episode_id"].nunique()),
        },
    }


def verify_historical_lora_and_gate_d() -> dict:
    files = {
        "clean_rerun_rank4_seed42": REPO_ROOT / "results/clean_rerun_rank4_seed42_combined.csv",
        "clean_rerun_rank16_unseeded": REPO_ROOT / "results/clean_rerun_rank16_unseeded_combined.csv",
        "legacy_rank2": REPO_ROOT / "results/lora_eval_rank2_combined.csv",
        "legacy_rank4": REPO_ROOT / "results/lora_eval_rank4_combined.csv",
        "legacy_rank8": REPO_ROOT / "results/lora_eval_rank8_combined.csv",
        "legacy_rank16": REPO_ROOT / "results/lora_eval_rank16_combined.csv",
        "legacy_final": REPO_ROOT / "results/lora_eval_final_combined.csv",
    }
    summaries = {}
    dfs = {}
    for name, path in files.items():
        df = pd.read_csv(path).sort_values(["fold", "organ", "vol_id"]).reset_index(drop=True)
        df["patient_id"] = df["vol_id"].apply(lambda x: str(x).replace("\\", "/").split("/")[-1].split(".nii")[0])
        dfs[name] = df
        arm_stats = {}
        for dcol in ("dice_baseline", "dice_adafob_2d", "dice_lora_fob", "dice_lora_ada"):
            hcol = dcol.replace("dice_", "hd95_")
            po_d = df.groupby(["patient_id", "organ"])[dcol].mean().groupby("patient_id").mean()
            po_h = df.groupby(["patient_id", "organ"])[hcol].mean().groupby("patient_id").mean()
            arm_stats[dcol] = {
                "episode_dice_pct": float(df[dcol].mean() * 100.0),
                "episode_dice_sd_pct": float(df[dcol].std(ddof=1) * 100.0),
                "episode_hd95_mm": float(df[hcol].mean()),
                "episode_hd95_sd_mm": float(df[hcol].std(ddof=1)),
                "episode_hd95_median_mm": float(df[hcol].median()),
                "patient_dice_pct": float(po_d.mean() * 100.0),
                "patient_hd95_mm": float(po_h.mean()),
                "hd95_gt_300mm_count": int((df[hcol] > 300.0).sum()),
                "hd95_max_mm": float(df[hcol].max()),
            }
        summaries[name] = {"rows": int(len(df)), "unique_patients": int(df["patient_id"].nunique()), "arms": arm_stats}

    # Prove clean_rerun_rank16_unseeded == 108 rows of legacy_rank2 + 12 rows of legacy_rank4 (Fold 0 LK, LIVER)
    c16 = dfs["clean_rerun_rank16_unseeded"]
    r2 = dfs["legacy_rank2"]
    r4 = dfs["legacy_rank4"]
    num_cols = ["dice_baseline", "hd95_baseline", "dice_adafob_2d", "hd95_adafob_2d", "dice_lora_fob", "hd95_lora_fob", "dice_lora_ada", "hd95_lora_ada"]
    mask_f0_gpu1 = (c16["fold"] == 0) & (c16["organ"].isin(["LK", "LIVER"]))
    spliced = r2.copy()
    spliced.loc[mask_f0_gpu1, num_cols] = r4.loc[mask_f0_gpu1, num_cols]
    splice_max_diff = float((c16[num_cols] - spliced[num_cols]).abs().max().max())

    # Check manifest commit hashes against git
    manifest_df = pd.read_csv(REPO_ROOT / "results/experiment_manifest.csv")
    manifest_git_checks = []
    for _, row in manifest_df.iterrows():
        recorded = str(row["git_commit_hash"])
        cat = subprocess.run(["git", "cat-file", "-t", recorded], cwd=str(REPO_ROOT), capture_output=True, text=True)
        short_resolve = run_git(["rev-parse", recorded[:7]])
        manifest_git_checks.append({
            "result_file": row["result_file"],
            "recorded_40char_hash": recorded,
            "exact_hash_exists_in_git": cat.returncode == 0,
            "short_7char_resolves_to": short_resolve,
        })

    # Gate D verification
    gd = pd.concat([pd.read_csv(REPO_ROOT / "results/gate_d_results_gpu0.csv"), pd.read_csv(REPO_ROOT / "results/gate_d_results_gpu1.csv")], ignore_index=True)
    ada_plateau = {f"ada_dice_{k}": float(gd[f"ada_dice_{k}"].mean() * 100.0) for k in (0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24)}
    uni_plateau = {f"uni_dice_{k}": float(gd[f"uni_dice_{k}"].mean() * 100.0) for k in (0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24)}
    phase4 = pd.read_csv(REPO_ROOT / "results/phase4_validation.csv")
    e3 = pd.read_csv(REPO_ROOT / "results/e3_evaluation.csv")
    r1 = pd.read_csv(REPO_ROOT / "results/r1_budget_sweep.csv")
    r2_mat = pd.read_csv(REPO_ROOT / "results/r2_raw_metrics.csv")

    return {
        "lora_csv_summaries": summaries,
        "clean_rank16_splice_proof": {
            "max_abs_diff_vs_108_rank2_plus_12_rank4": splice_max_diff,
            "fold0_lk_liver_rows_from_rank4": int(mask_f0_gpu1.sum()),
            "remaining_rows_from_rank2": int((~mask_f0_gpu1).sum()),
        },
        "experiment_manifest_git_checks": manifest_git_checks,
        "gate_d_truncation_evidence": {
            "mean_adaptive_budget": float(gd["mean_adaptive_budget"].mean()),
            "mean_adaptive_budget_by_organ": gd.groupby("organ")["mean_adaptive_budget"].mean().to_dict(),
            "dice_fob_base_pct": float(gd["dice_fob_base"].mean() * 100.0),
            "dice_adabudget_uni_pct": float(gd["dice_adabudget_uni"].mean() * 100.0),
            "dice_fixed_10_ada_pct": float(gd["dice_fixed_10_ada"].mean() * 100.0),
            "dice_adafob_pct": float(gd["dice_adafob"].mean() * 100.0),
            "ada_sweep_pct": ada_plateau,
            "uni_sweep_pct": uni_plateau,
        },
        "phase4_zero_budget_collapse": {
            "rows": int(len(phase4)),
            "ada_n_neg_counts": {int(k): int(v) for k, v in phase4["ada_n_neg"].value_counts().to_dict().items()},
            "ada_mean_dice_pct": float(phase4["ada_dice"].mean() * 100.0),
            "base_mean_dice_pct": float(phase4["base_dice"].mean() * 100.0),
        },
        "e3_evaluation_summary": e3.groupby("Organ")[["Baseline_Dice", "AdaFoB_Dice", "Gain"]].mean().to_dict(orient="index"),
        "r1_budget_sweep_summary": {
            "rows": int(len(r1)),
            "dice_0_pct": float(r1["dice_0"].mean() * 100.0),
            "dice_10_pct": float(r1["dice_10"].mean() * 100.0),
            "oracle_dice_pct": float(r1["oracle_dice"].mean() * 100.0),
        },
        "r2_raw_metrics_summary": {
            "rows": int(len(r2_mat)),
            "uni_dice_0_pct": float(r2_mat["uni_dice_0"].mean() * 100.0),
            "uni_dice_10_pct": float(r2_mat["uni_dice_10"].mean() * 100.0),
            "ada_dice_10_pct": float(r2_mat["ada_dice_10"].mean() * 100.0),
        },
    }


def verify_route_b_and_mechanisms() -> dict:
    locked_json = json.loads((REPO_ROOT / "audit/route_b_locked_reading_20261028T092629817154Z/analysis.json").read_text(encoding="utf-8-sig"))
    rb_dir = REPO_ROOT / "audit/route_b_lean_20261028T031431779702Z/owner_terminal/gpu_outputs_v2/route_b_gpu_20261028-031826-80d323"
    case_results = json.loads((rb_dir / "case_results.json").read_text(encoding="utf-8"))
    v3_results = json.loads((REPO_ROOT / "analysis/route_b_saved_bank_diagnostic_20261028_v3/results.json").read_text(encoding="utf-8"))

    # Recompute check/construction arm means directly from case_results.json
    arms = ["full10", "boundary_near2", "random_mean", "random_s42", "random_s43", "random_s44", "random_s45", "magnitude", "coordinate", "signed", "protected"]
    role_means = {}
    for role in ("construction", "check"):
        role_cases = [c for c in case_results if c["role"] == role]
        role_means[role] = {}
        for arm in arms:
            if arm == "random_mean":
                d_vals = [np.mean([c["record"]["evaluation"][f"random_s{s}_cc"]["dice"] for s in range(42, 46)]) for c in role_cases]
                r_vals = [np.mean([c["record"]["evaluation"][f"random_s{s}_cc"]["recall"] for s in range(42, 46)]) for c in role_cases]
            else:
                d_vals = [c["record"]["evaluation"][f"{arm}_cc"]["dice"] for c in role_cases]
                r_vals = [c["record"]["evaluation"][f"{arm}_cc"]["recall"] for c in role_cases]
            role_means[role][arm] = {"mean_cc_dice": float(np.mean(d_vals)), "mean_cc_recall": float(np.mean(r_vals))}

    # Independent verification of Route B prediction/evidence arrays without reading query NIfTI labels
    sys.path.insert(0, str(REPO_ROOT / "experiments/utility_feasibility_v1"))
    from scoring_v1 import score_subsets

    case_mechanisms = []
    for idx, c in enumerate(case_results):
        cid = c["case"]
        npz = np.load(rb_dir / "predictions" / f"{cid}.npz", allow_pickle=False)
        F = npz["full10_raw"]
        D = npz["single_deletion_raw"]
        e = npz["evidence"]
        xy = npz["negative_restored_xy"]
        N = F.size

        scored = score_subsets(F, D, xy, e)
        for k_arm in ("magnitude", "coordinate", "signed", "protected"):
            assert list(scored[k_arm]) == c["record"]["selected_ids"][k_arm], f"Mismatch in {cid} {k_arm}"

        added = D & ~F
        lost = F & ~D
        mag = (added | lost).mean(axis=(1, 2))
        add_area = added.mean(axis=(1, 2))
        rho_mag_signed = float(spearmanr(mag, scored["signed_scores"]).statistic)
        rho_add_signed = float(spearmanr(add_area, scored["signed_scores"]).statistic)

        case_mechanisms.append({
            "case": cid,
            "role": c["role"],
            "patient": c["patient"],
            "organ": c["organ"],
            "actual_positive_count": int(c["record"]["actual_positive_count"]),
            "full10_raw_pixels": int(F.sum()),
            "pct_evidence_positive": float((e > 0).mean() * 100.0),
            "mean_evidence": float(e.mean()),
            "eligible_count": int(len(scored["eligible"])),
            "selected_ids": c["record"]["selected_ids"],
            "signed_equals_magnitude": c["record"]["selected_ids"]["signed"] == c["record"]["selected_ids"]["magnitude"],
            "spearman_mag_vs_signed": rho_mag_signed,
            "spearman_added_area_vs_signed": rho_add_signed,
            "full10_cc_dice": float(c["record"]["evaluation"]["full10_cc"]["dice"]),
            "near2_cc_dice": float(c["record"]["evaluation"]["boundary_near2_cc"]["dice"]),
            "signed_cc_dice": float(c["record"]["evaluation"]["signed_cc"]["dice"]),
            "protected_cc_dice": float(c["record"]["evaluation"]["protected_cc"]["dice"]),
        })

    # Detailed case7 (image_20 / pancreas) veto & Q1 decomposition
    npz7 = np.load(rb_dir / "predictions/case7.npz", allow_pickle=False)
    F7, D7, e7 = npz7["full10_raw"], npz7["single_deletion_raw"], npz7["evidence"]
    added7, lost7 = D7 & ~F7, F7 & ~D7
    N7 = F7.size
    case7_candidates = []
    for i in range(10):
        a_i = float((added7[i] * e7).sum() / N7)
        l_i = float((lost7[i] * e7).sum() / N7)
        af_i = float((added7[i] * np.maximum(e7, 0)).sum() / N7)
        lf_i = float((lost7[i] * np.maximum(e7, 0)).sum() / N7)
        case7_candidates.append({
            "id": i,
            "added_px": int(added7[i].sum()),
            "lost_px": int(lost7[i].sum()),
            "a_i": a_i,
            "l_i": l_i,
            "signed_score": l_i - a_i,
            "af_i": af_i,
            "lf_i": lf_i,
            "eligible": bool(af_i <= lf_i),
        })

    m34 = npz7["signed_raw"]
    m67 = npz7["coordinate_raw"]
    both = m34 & m67
    extra_67 = m67 & ~m34
    extra_34 = m34 & ~m67
    case7_q1_vs_signed = {
        "shared_pixels": int(both.sum()),
        "shared_mean_evidence": float(e7[both].mean()),
        "extra_67_only_pixels": int(extra_67.sum()),
        "extra_67_only_mean_evidence": float(e7[extra_67].mean()),
        "extra_34_only_pixels": int(extra_34.sum()),
        "extra_34_only_mean_evidence": float(e7[extra_34].mean()),
    }

    return {
        "locked_pilot_role_means": role_means,
        "locked_decision": locked_json["decision"],
        "v3_saved_bank_means": v3_results["means"],
        "case_mechanisms": case_mechanisms,
        "case7_pancreas_veto_breakdown": case7_candidates,
        "case7_pancreas_q1_extra_fp_breakdown": case7_q1_vs_signed,
    }


def verify_pba_code_bugs_synthetically() -> dict:
    sys.path.insert(0, str(REPO_ROOT / "third_party/FoB_SAM"))
    from models.allocator import PromptBudgetAllocator

    alloc = PromptBudgetAllocator(max_points=24, device="cpu")
    # 1. Test Bug A3: z-score scale vs sigma=20.0 in leak risk l(s)
    # Create a binary circle mask in 256x256 and a z-score query image with step edge 2.0 sigma
    yy, xx = np.ogrid[:256, :256]
    m_tilde = (((yy - 128) ** 2 + (xx - 128) ** 2) <= 30**2).astype(np.uint8)
    contours, _ = ndimage.label(m_tilde), None
    import cv2

    cv_contours, _ = cv2.findContours(m_tilde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    z_img = np.where(m_tilde == 1, 1.0, -1.0).astype(np.float32)  # huge 2.0-std contrast!
    pts24 = alloc.sample_placement(z_img, m_tilde, cv_contours, 24, r=10)
    # Angle coverage of first 2 points out of 24 (Bug A1 capturing_allocate slice)
    angles_all = np.degrees(np.arctan2(pts24[:, 1] - 128, pts24[:, 0] - 128))
    angles_first2 = angles_all[:2]
    max_ls_deviation_from_1 = 1.0 - float(np.exp(-(2.0**2) / (2.0 * 20.0**2)))

    return {
        "bug_a1_pts24_count": int(len(pts24)),
        "bug_a1_first2_points_xy": pts24[:2].tolist(),
        "bug_a1_first2_angles_deg": angles_first2.tolist(),
        "bug_a1_first2_angular_span_deg": float(np.max(angles_first2) - np.min(angles_first2)),
        "bug_a3_zscore_2std_ls_drop_below_1": max_ls_deviation_from_1,
    }


def main() -> None:
    summary = {
        "environment_and_provenance": verify_environment_and_provenance(),
        "patient_roles_and_label_use": verify_patient_roles(),
        "step9_controls_verification": verify_step9_controls(),
        "historical_lora_and_gate_d": verify_historical_lora_and_gate_d(),
        "route_b_verification": verify_route_b_and_mechanisms(),
        "pba_synthetic_bug_checks": verify_pba_code_bugs_synthetically(),
    }
    out_path = OUT_DIR / "cpu_verification_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"VERIFICATION_COMPLETE: {out_path}")
    print(f"Step9 max_be_diff={summary['step9_controls_verification']['max_both_estimands_discrepancy']}")
    print(f"Step9 best_closure_pct={summary['step9_controls_verification']['gap_closure_summary']['best_gap_closure_pct']:.6f}%")
    print(f"Rank16 splice max_diff={summary['historical_lora_and_gate_d']['clean_rank16_splice_proof']['max_abs_diff_vs_108_rank2_plus_12_rank4']}")
    print(f"Route B check signed={summary['route_b_verification']['locked_pilot_role_means']['check']['signed']['mean_cc_dice']:.6f} protected={summary['route_b_verification']['locked_pilot_role_means']['check']['protected']['mean_cc_dice']:.6f}")
    print(f"PBA Bug A1 first2 span deg={summary['pba_synthetic_bug_checks']['bug_a1_first2_angular_span_deg']:.2f}, Bug A3 ls drop={summary['pba_synthetic_bug_checks']['bug_a3_zscore_2std_ls_drop_below_1']:.6f}")


if __name__ == "__main__":
    main()
