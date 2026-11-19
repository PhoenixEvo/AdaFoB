"""Independent local verification and English report generation after Kaggle download."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (
    NEW_ARMS,
    compute_estimands,
    evaluate_preregistered_decision,
    json_safe,
    paired_bootstrap_contrasts,
)
from .selectors import ARMS

ROOT = Path(__file__).resolve().parents[2]
STEP9_MEASUREMENTS = (
    ROOT / "results/step9_controls_20261024T034851Z/cpu_analysis_v3_20261025T024935616Z_1e6a9939/measurements.csv"
)
STEP9_ENV = ROOT / "results/step9_controls_20261024T034851Z/manual_verified_20261024T133636Z/environment_kaggle.json"
DEFAULT_REPORT = ROOT / "reports/option_a_reassessment_results_20261119.md"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _load_kaggle_artifacts(
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, dict, dict, dict | None, dict | None]:
    zip_path = out_dir / "option_a_outputs.zip"
    if zip_path.is_file() and not (out_dir / "measurements.csv").is_file():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir / "unpacked")
        src = out_dir / "unpacked"
    else:
        src = out_dir

    df_meas = pd.read_csv(src / "measurements.csv")
    df_est_k = pd.read_csv(src / "both_estimands.csv")
    contrasts_k = json.loads((src / "paired_contrasts.json").read_text(encoding="utf-8"))
    df_audit = pd.read_csv(src / "prompt_audit.csv")
    worst_k = json.loads((src / "worst_cases.json").read_text(encoding="utf-8"))
    status_k = json.loads((src / "remote_status.json").read_text(encoding="utf-8"))
    pilot_p = src / "pilot_summary.json"
    if not pilot_p.is_file():
        pilot_p = src / "pilot/pilot_summary.json"
    pilot_k = json.loads(pilot_p.read_text(encoding="utf-8")) if pilot_p.is_file() else None
    env_p = src / "environment_kaggle.json"
    env_k = json.loads(env_p.read_text(encoding="utf-8")) if env_p.is_file() else None
    return df_meas, df_est_k, contrasts_k, df_audit, worst_k, status_k, pilot_k, env_k


def _audit_prompts(df_audit: pd.DataFrame) -> dict:
    total = int(len(df_audit))
    k_counts = df_audit["resp_signed_adaptive_k"].value_counts().sort_index().to_dict()
    same_local_mag = int(
        (df_audit["resp_signed_localring_k2_ids"].astype(str) == df_audit["resp_magnitude_k2_ids"].astype(str)).sum()
    )
    same_local_near = int(
        (df_audit["resp_signed_localring_k2_ids"].astype(str) == df_audit["boundary_near2_ids"].astype(str)).sum()
    )
    same_global_mag = int(
        (df_audit["resp_signed_global_k2_ids"].astype(str) == df_audit["resp_magnitude_k2_ids"].astype(str)).sum()
    )
    same_lc_near = int(
        (df_audit["local_contrast_near2_ids"].astype(str) == df_audit["boundary_near2_ids"].astype(str)).sum()
    )
    return {
        "slices": total,
        "adaptive_k_histogram": {str(int(k)): int(v) for k, v in k_counts.items()},
        "adaptive_mean_k": float(df_audit["resp_signed_adaptive_k"].mean()),
        "resp_local_equals_magnitude_slices": same_local_mag,
        "resp_global_equals_magnitude_slices": same_global_mag,
        "resp_local_equals_near2_slices": same_local_near,
        "local_contrast_equals_near2_slices": same_lc_near,
        "pba_fixed_fallbacks": int(df_audit["pba_fixed_k2_fallback"].notna().sum()),
        "local_contrast_fallbacks": int(df_audit["local_contrast_fallback"].notna().sum()),
        "mean_decoder_calls_per_slice": float(df_audit["decoder_calls"].mean()),
    }


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return "inf"
    return f"{v:.{digits}f}"


def _build_report(
    run_dir: Path,
    receipt: dict,
    estimands: pd.DataFrame,
    contrasts: dict,
    decision: dict,
    prompt_stats: dict,
    status_k: dict,
    pilot_k: dict | None,
) -> str:
    cc_est = estimands[estimands["post"] == "cc"].set_index("arm")
    raw_est = estimands[estimands["post"] == "raw"].set_index("arm")
    env_cmp = receipt.get("runtime_comparison", {})

    lines = [
        "# Option A Bounded Setting I Reassessment Results (Frozen SAM ViT-B)",
        "",
        f"- **Run Directory:** `{run_dir}`",
        f"- **Kaggle Kernel Status:** `{status_k.get('status')}` (`episodes={status_k.get('episodes')}`, `patients={status_k.get('patients')}`, `slices={status_k.get('slices')}`)",
        f"- **Runtime Drift Note:** Step 9 ran on `torch {env_cmp.get('step9_torch', 'unknown')}` (`CUDA {env_cmp.get('step9_cuda', 'unknown')}`); Option A ran on `torch {env_cmp.get('option_a_torch', 'unknown')}` (`CUDA {env_cmp.get('option_a_cuda', 'unknown')}`). All 9 Option A arms ran inside the same matched session.",
        f"- **Pre-Registered Decision:** **`{decision['decision']}`** (Passing arms: `{decision['passing_arms']}`)",
        f"- **Cross-CUDA Anchor Agreement vs Step 9 (720 rows):** `exact_mask_sha_matches={receipt['step9_exact_mask_sha_matches']}/720`, `max_patient_cc_dice_diff_pp={receipt['step9_max_patient_dice_diff_pp']:.4f}`, `max_patient_cc_hd95_diff_mm={receipt['step9_max_patient_hd95_diff_mm']:.4f}`, `max_episode_dice_diff={receipt['step9_max_episode_dice_diff']:.4e}`",
        "",
        "## 1. Pre-Registered Criteria Evaluation",
        "",
        "Criteria require beating matched `boundary_near2` by `>= +0.50 pp` Patient-balanced CC Dice with `95% CI lower bound > 0`, while keeping Recall delta vs `full10` `>= -0.15 pp` (`95% CI lower bound >= -0.50 pp`).",
        "",
        "| New Arm | Pat CC Dice (%) | Delta vs `boundary_near2` (pp) [95% CI] | Delta vs `full10` (pp) [95% CI] | Pat HD95 (mm) | Recall vs `full10` (pp) [95% CI] | Passes Bar |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for r in decision["arms"]:
        c_n = r["ci95_vs_boundary_near2_pp"]
        c_f = r["ci95_vs_full10_pp"]
        c_r = r["recall_ci95_vs_full10_pp"]
        s_dn = f"{_fmt(r['delta_vs_boundary_near2_pp'])} [{_fmt(c_n[0])}, {_fmt(c_n[1])}]" if c_n else "N/A"
        s_df = f"{_fmt(r['delta_vs_full10_pp'])} [{_fmt(c_f[0])}, {_fmt(c_f[1])}]" if c_f else "N/A"
        s_rf = f"{_fmt(r['recall_vs_full10_pp'])} [{_fmt(c_r[0])}, {_fmt(c_r[1])}]" if c_r else "N/A"
        lines.append(
            f"| `{r['arm']}` | {_fmt(r['patient_cc_dice_pct'])} | {s_dn} | {s_df} | {_fmt(r['patient_cc_hd95_mm'])} | {s_rf} | `{r['passes_preregistered_bar']}` |"
        )

    lines.extend(
        [
            "",
            "## 2. Full 9-Arm Comparison Table (120 Episodes / 25 Patients / 4,009 Slices)",
            "",
            "| Arm | Pat Raw Dice (%) | Pat CC Dice (%) [SD] | Ep CC Dice (%) [SD] | Pat CC HD95 (mm) | Ep Med/p90 HD95 (mm) | Pat CC Recall (%) | Empty Preds |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ARMS:
        if arm not in cc_est.index:
            continue
        r_cc = cc_est.loc[arm]
        r_rw = raw_est.loc[arm]
        lines.append(
            f"| `{arm}` | {_fmt(r_rw['patient_dice_pct'])} | {_fmt(r_cc['patient_dice_pct'])} [{_fmt(r_cc['patient_dice_sd_pct'])}] | "
            f"{_fmt(r_cc['episode_dice_pct'])} [{_fmt(r_cc['episode_dice_sd_pct'])}] | {_fmt(r_cc['patient_hd95_mm'])} | "
            f"{_fmt(r_cc['episode_hd95_median_mm'])} / {_fmt(r_cc['episode_hd95_p90_mm'])} | {_fmt(r_cc['patient_recall_pct'])} | "
            f"{int(r_cc['empty_predictions'])} |"
        )

    lines.extend(
        [
            "",
            "## 3. Per-Organ Patient-Balanced CC Dice (%)",
            "",
            "| Arm | LIVER (%) | LK (%) | RK (%) | SPLEEN (%) |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    po = pd.DataFrame(contrasts["per_organ_cc"]).pivot(index="arm", columns="organ", values="patient_dice_pct")
    for arm in ARMS:
        if arm in po.index:
            lines.append(
                f"| `{arm}` | {_fmt(po.loc[arm, 'LIVER'])} | {_fmt(po.loc[arm, 'LK'])} | {_fmt(po.loc[arm, 'RK'])} | {_fmt(po.loc[arm, 'SPLEEN'])} |"
            )

    lines.extend(
        [
            "",
            "## 4. Prompt Behavior and Parity Summary",
            "",
            f"- Total audited slices: `{prompt_stats['slices']}`",
            f"- Mean SAM decoder calls per slice: `{prompt_stats['mean_decoder_calls_per_slice']:.2f}`",
            f"- `resp_signed_localring_k2 == resp_magnitude_k2` slices: `{prompt_stats['resp_local_equals_magnitude_slices']}/{prompt_stats['slices']}` (vs `resp_signed_global_k2 == resp_magnitude_k2`: `{prompt_stats['resp_global_equals_magnitude_slices']}/{prompt_stats['slices']}`)",
            f"- `local_contrast_near2 == boundary_near2` slices: `{prompt_stats['local_contrast_equals_near2_slices']}/{prompt_stats['slices']}`",
            f"- `resp_signed_adaptive` mean negative points `k`: `{prompt_stats['adaptive_mean_k']:.3f}` (histogram: `{prompt_stats['adaptive_k_histogram']}`)",
            f"- `pba_fixed_k2` fallbacks: `{prompt_stats['pba_fixed_fallbacks']}/{prompt_stats['slices']}`",
            f"- Remote aggregate T4 GPU-hours: `{status_k.get('aggregate_gpu_hours', 0.0):.4f}` (wall `{status_k.get('wall_seconds', 0.0):.1f}s`)",
        ]
    )
    if pilot_k:
        lines.append(
            f"- Stage-0 Pilot intra-run exact determinism: `{pilot_k.get('intra_run_exact_all_slices')}`, "
            f"Step 9 CUDA tolerance passed: `{pilot_k.get('step9_cuda_tolerance_passed')}` "
            f"(`max_neg_coord_diff_px={pilot_k.get('max_neg_coord_diff_px', 0.0):.5f}`, "
            f"`min_full10_cc_dice_vs_step9={pilot_k.get('min_full10_cc_dice_vs_step9', 0.0):.5f}`, "
            f"`projected_total_gpu_hours={pilot_k.get('projected_total_gpu_hours', 0.0):.4f}`)"
        )
    lines.append("")
    return "\n".join(lines)


def verify_run(run_dir: Path, report_path: Path) -> dict:
    out_dir = run_dir / "kaggle_outputs"
    df_meas, df_est_k, contrasts_k, df_audit, worst_k, status_k, pilot_k, env_k = _load_kaggle_artifacts(out_dir)

    assert len(df_meas) == 120 * len(ARMS) * 2, f"Expected 2160 rows, got {len(df_meas)}"
    assert len(df_audit) == 4009, f"Expected 4009 slice audit rows, got {len(df_audit)}"

    step9 = pd.read_csv(STEP9_MEASUREMENTS)
    anchors = ("full10", "positive0", "boundary_near2")
    s9_sub = step9[step9["arm"].isin(anchors)].sort_values(["episode_id", "arm", "post"]).reset_index(drop=True)
    oa_sub = df_meas[df_meas["arm"].isin(anchors)].sort_values(["episode_id", "arm", "post"]).reset_index(drop=True)
    assert len(s9_sub) == len(oa_sub) == 720
    assert (s9_sub["episode_id"] == oa_sub["episode_id"]).all()
    assert (s9_sub["arm"] == oa_sub["arm"]).all()
    assert (s9_sub["post"] == oa_sub["post"]).all()

    exact_mask_sha_matches = int((s9_sub["mask_sha256"] == oa_sub["mask_sha256"]).sum())
    max_ep_dice_diff = float((s9_sub["dice"] - oa_sub["dice"]).abs().max())
    max_ep_hd95_diff = float((s9_sub["hd95"] - oa_sub["hd95"]).abs().max())

    local_est = compute_estimands(df_meas)
    num_cols = [c for c in local_est.columns if c not in ("arm", "post") and np.isfinite(local_est[c]).all()]
    max_est_diff = float((local_est[num_cols].to_numpy() - df_est_k[num_cols].to_numpy()).__abs__().max())
    assert max_est_diff <= 1e-10, f"Local estimand mismatch: {max_est_diff}"

    s9_est = compute_estimands(s9_sub)
    s9_cc = s9_est[s9_est["post"] == "cc"].set_index("arm")
    oa_cc = local_est[local_est["post"] == "cc"].set_index("arm")
    anchor_diffs = {}
    for arm in anchors:
        d_diff = float(abs(oa_cc.loc[arm, "patient_dice_pct"] - s9_cc.loc[arm, "patient_dice_pct"]))
        h_diff = float(abs(oa_cc.loc[arm, "patient_hd95_mm"] - s9_cc.loc[arm, "patient_hd95_mm"]))
        anchor_diffs[arm] = {"patient_cc_dice_diff_pp": d_diff, "patient_cc_hd95_diff_mm": h_diff}
    max_pat_dice_diff_pp = max(v["patient_cc_dice_diff_pp"] for v in anchor_diffs.values())
    max_pat_hd95_diff_mm = max(v["patient_cc_hd95_diff_mm"] for v in anchor_diffs.values())
    assert max_pat_dice_diff_pp <= 0.10, f"Step 9 aggregate patient CC Dice drift too large: {max_pat_dice_diff_pp:.4f} pp"
    assert max_pat_hd95_diff_mm <= 0.50, f"Step 9 aggregate patient CC HD95 drift too large: {max_pat_hd95_diff_mm:.4f} mm"

    local_contrasts = paired_bootstrap_contrasts(df_meas, baselines=("full10", "positive0", "boundary_near2"), replicates=5000, seed=42)
    decision = evaluate_preregistered_decision(local_est, local_contrasts)
    prompt_stats = _audit_prompts(df_audit)

    s9_env = json.loads(STEP9_ENV.read_text(encoding="utf-8")) if STEP9_ENV.is_file() else {}
    runtime_cmp = {
        "step9_torch": s9_env.get("packages", {}).get("torch"),
        "step9_cuda": s9_env.get("torch_cuda"),
        "option_a_torch": (env_k or {}).get("packages", {}).get("torch"),
        "option_a_cuda": (env_k or {}).get("torch_cuda"),
    }

    receipt = {
        "run_dir": str(run_dir),
        "measurements_rows": int(len(df_meas)),
        "audit_slices": int(len(df_audit)),
        "step9_anchor_rows": 720,
        "step9_exact_mask_sha_matches": exact_mask_sha_matches,
        "step9_max_episode_dice_diff": max_ep_dice_diff,
        "step9_max_episode_hd95_diff_mm": max_ep_hd95_diff,
        "step9_max_patient_dice_diff_pp": max_pat_dice_diff_pp,
        "step9_max_patient_hd95_diff_mm": max_pat_hd95_diff_mm,
        "step9_anchor_diffs": anchor_diffs,
        "runtime_comparison": runtime_cmp,
        "local_estimand_max_diff": max_est_diff,
        "prompt_stats": prompt_stats,
        "decision": decision,
        "remote_status": status_k,
        "worst_cases": worst_k,
    }
    (run_dir / "local_verification.json").write_text(json.dumps(json_safe(receipt), indent=2), encoding="utf-8")

    report_md = _build_report(run_dir, receipt, local_est, local_contrasts, decision, prompt_stats, status_k, pilot_k)
    report_path.write_text(report_md, encoding="utf-8")
    (run_dir / "option_a_reassessment_results_20261119.md").write_text(report_md, encoding="utf-8")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    a = p.parse_args()
    res = verify_run(a.run_dir, a.report_path)
    print(
        json.dumps(
            {
                "decision": res["decision"]["decision"],
                "passing_arms": res["decision"]["passing_arms"],
                "step9_exact_mask_sha_matches": res["step9_exact_mask_sha_matches"],
                "step9_max_patient_dice_diff_pp": res["step9_max_patient_dice_diff_pp"],
                "step9_max_patient_hd95_diff_mm": res["step9_max_patient_hd95_diff_mm"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
