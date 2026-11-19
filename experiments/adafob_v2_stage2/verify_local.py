"""Independent local verification and MICCAI Stage-2 results report generator."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (
    compute_estimands_stage2,
    evaluate_stage2_decision,
    json_safe,
    paired_bootstrap_contrasts_stage2,
)
from .runner import CBV_ARMS
from .selectors import ARMS

ROOT = Path(__file__).resolve().parents[2]
STEP9_MEASUREMENTS = (
    ROOT / "results/step9_controls_20261024T034851Z/cpu_analysis_v3_20261025T024935616Z_1e6a9939/measurements.csv"
)
STAGE1_MEASUREMENTS = ROOT / "experiments/runs/adafob_v2_stage1_20261119/kaggle_outputs/measurements.csv"
DEFAULT_REPORT = ROOT / "reports/adafob_v2_stage2_results_20261119.md"


def _load_kaggle_artifacts(
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, dict, dict, dict | None, dict | None]:
    zip_path = out_dir / "adafob_v2_stage2_outputs.zip"
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


def _audit_prompts_stage2(df_audit: pd.DataFrame) -> dict:
    cbv_stats: dict[str, dict] = {}
    for arm in CBV_ARMS:
        ch_col = f"{arm}__choice"
        k_col = f"{arm}__k"
        fb_col = f"{arm}__fallback"
        vc = df_audit[ch_col].value_counts().to_dict()
        cbv_stats[arm] = {
            "mean_k": float(df_audit[k_col].mean()),
            "choice_counts": {str(k): int(v) for k, v in sorted(vc.items())},
            "fallbacks": int(df_audit[fb_col].notna().sum()),
        }
    organ_scale = (
        df_audit.groupby("organ")[["r_eq_px", "r_offset_px", "tau_cal"]]
        .median()
        .reset_index()
        .to_dict("records")
    )
    return {
        "slices": int(len(df_audit)),
        "mean_decoder_calls_per_slice": float(df_audit["decoder_calls"].mean()),
        "pba_f10_fallbacks": int(df_audit["pba_f10_int_k2_fallback"].notna().sum()),
        "pba_scale_fallbacks": int(df_audit["pba_f10_scale_k2_fallback"].notna().sum()),
        "mean_tau_cal": float(df_audit["tau_cal"].mean()),
        "no_sep_slices": int((df_audit["cal_note"] == "no_sep").sum()),
        "organ_median_scale": organ_scale,
        "per_cbv_arm": cbv_stats,
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
    per_org_dice = pd.DataFrame(contrasts["per_organ_cc"]).pivot(index="arm", columns="organ", values="patient_dice_pct")
    per_org_rec = pd.DataFrame(contrasts["per_organ_cc"]).pivot(index="arm", columns="organ", values="patient_recall_pct")
    per_org_hd = pd.DataFrame(contrasts["per_organ_cc"]).pivot(index="arm", columns="organ", values="patient_hd95_mm")

    lines = [
        "# AdaFoB v2 Stage 2 Results: Scale-Adaptive Counterfactual Boundary Verification (Frozen SAM ViT-B, Setting I)",
        "",
        f"- **Run Directory:** `{run_dir}`",
        f"- **Kaggle Kernel Status:** `{status_k.get('status')}` (`episodes={status_k.get('episodes')}`, `patients={status_k.get('patients')}`, `slices={status_k.get('slices')}`)",
        f"- **Pre-Registered Stage-2 Decision:** **`{decision['decision']}`** (Primary pass: `{decision['passing_arms']}`, Secondary strong Pareto: `{decision['pareto_strong_arms']}`)",
        f"- **Anchor Agreement vs Stage 1 (`cu130`):** `max_patient_cc_dice_diff_pp={receipt['stage1_max_patient_dice_diff_pp']:.4f}` across `full10`, `positive0`, `boundary_near2`, `pba_f10_int_k2`",
        f"- **Cross-CUDA Agreement vs Step 9 (`cu128 -> cu130`):** `max_patient_cc_dice_diff_pp={receipt['step9_max_patient_dice_diff_pp']:.4f}`, `max_patient_cc_hd95_diff_mm={receipt['step9_max_patient_hd95_diff_mm']:.4f}`",
        "",
        "## 1. Stage-2 Criteria Evaluation (`NEW_ARMS` vs `boundary_near2` & `full10`)",
        "",
        "Primary criteria: `delta_vs_boundary_near2 >= +1.00 pp` (`95% CI low > 0`), `recall_vs_full10 >= -0.15 pp` (`95% CI low >= -0.50 pp`), `LIVER recall >= 91.0%`, `delta_hd95_vs_boundary_near2 < 0`. Secondary Pareto diagnostic: `delta >= +1.00 pp`, `LIVER recall >= 90.0%`, `delta_hd95 <= -2.0 mm`.",
        "",
        "| Arm | Pat CC Dice (%) | Delta vs `boundary_near2` (pp) [95% CI] | Delta vs `pba_f10` (pp) | Pat HD95 (mm) | Recall vs `full10` (pp) [95% CI] | LIVER Rec (%) | Primary / Pareto |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for r in decision["arms"]:
        c_n = r["ci95_vs_boundary_near2_pp"]
        c_r = r["recall_ci95_vs_full10_pp"]
        s_dn = f"{_fmt(r['delta_vs_boundary_near2_pp'])} [{_fmt(c_n[0])}, {_fmt(c_n[1])}]" if c_n else "N/A"
        s_rf = f"{_fmt(r['recall_vs_full10_pp'])} [{_fmt(c_r[0])}, {_fmt(c_r[1])}]" if c_r else "N/A"
        lines.append(
            f"| `{r['arm']}` | {_fmt(r['patient_cc_dice_pct'])} | {s_dn} | {_fmt(r['delta_vs_pba_f10_int_k2_pp'])} | "
            f"{_fmt(r['patient_cc_hd95_mm'])} | {s_rf} | {_fmt(r['liver_recall_pct'])} | `{r['passes_stage2_bar']}` / `{r['pareto_strong']}` |"
        )

    lines.extend(
        [
            "",
            "## 2. Full 11-Arm Comparison Table (120 Episodes / 25 Patients / 4,009 Slices)",
            "",
            "| Arm | Pat Raw Dice (%) | Pat CC Dice (%) [SD] | Ep CC Dice (%) | Pat CC HD95 (mm) | Ep Med/p90 HD95 (mm) | Pat CC Recall (%) | Empty Preds |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ARMS:
        r_cc = cc_est.loc[arm]
        r_rw = raw_est.loc[arm]
        lines.append(
            f"| `{arm}` | {_fmt(r_rw['patient_dice_pct'])} | {_fmt(r_cc['patient_dice_pct'])} [{_fmt(r_cc['patient_dice_sd_pct'])}] | "
            f"{_fmt(r_cc['episode_dice_pct'])} | {_fmt(r_cc['patient_hd95_mm'])} | "
            f"{_fmt(r_cc['episode_hd95_median_mm'])} / {_fmt(r_cc['episode_hd95_p90_mm'])} | {_fmt(r_cc['patient_recall_pct'])} | "
            f"{int(r_cc['empty_predictions'])} |"
        )

    lines.extend(
        [
            "",
            "## 3. Per-Organ Patient-Balanced Breakdown (`CC Dice % / CC Recall % / CC HD95 mm`)",
            "",
            "| Arm | LIVER | LK | RK | SPLEEN |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for arm in ARMS:
        cells = []
        for org in ("LIVER", "LK", "RK", "SPLEEN"):
            cells.append(f"{_fmt(per_org_dice.loc[arm, org], 2)} / {_fmt(per_org_rec.loc[arm, org], 2)} / {_fmt(per_org_hd.loc[arm, org], 1)}")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 4. CBV Subset Selection, Calibration & Compute Accounting",
            "",
            f"- Audited slices: `{prompt_stats['slices']}`, Mean SAM decoder calls per slice: `{prompt_stats['mean_decoder_calls_per_slice']:.2f}`",
            f"- Mean `tau_cal`: `{prompt_stats['mean_tau_cal']:.4f}`, `no_sep` slices: `{prompt_stats['no_sep_slices']}`",
            f"- Remote aggregate T4 GPU-hours: `{status_k.get('aggregate_gpu_hours', 0.0):.4f}` (wall `{status_k.get('wall_seconds', 0.0):.1f}s`)",
            "",
            "| CBV Arm | Mean `k` | Choice Distribution | Fallbacks |",
            "| :--- | ---: | :--- | ---: |",
        ]
    )
    for arm, st in prompt_stats["per_cbv_arm"].items():
        lines.append(f"| `{arm}` | {st['mean_k']:.3f} | `{st['choice_counts']}` | {st['fallbacks']} |")
    if pilot_k:
        lines.append("")
        lines.append(
            f"- Stage-0 Pilot intra-run exact determinism: `{pilot_k.get('intra_run_exact_all_slices')}`, "
            f"Step 9 tolerance passed: `{pilot_k.get('step9_cuda_tolerance_passed')}` "
            f"(`max_neg_coord_diff_px={pilot_k.get('max_neg_coord_diff_px', 0.0):.5f}`)"
        )
    lines.append("")
    return "\n".join(lines)


def verify_run(run_dir: Path, report_path: Path) -> dict:
    out_dir = run_dir / "kaggle_outputs"
    df_meas, df_est_k, contrasts_k, df_audit, worst_k, status_k, pilot_k, env_k = _load_kaggle_artifacts(out_dir)

    assert len(df_meas) == 120 * len(ARMS) * 2, f"Expected 2640 rows, got {len(df_meas)}"
    assert len(df_audit) == 4009, f"Expected 4009 audit rows, got {len(df_audit)}"

    local_est = compute_estimands_stage2(df_meas)
    num_cols = [c for c in local_est.columns if c not in ("arm", "post") and np.isfinite(local_est[c]).all()]
    max_est_diff = float((local_est[num_cols].to_numpy() - df_est_k[num_cols].to_numpy()).__abs__().max())
    assert max_est_diff <= 1e-10, f"Local vs Kaggle estimand mismatch: {max_est_diff}"

    s2_cc = local_est[local_est["post"] == "cc"].set_index("arm")

    stage1_df = pd.read_csv(STAGE1_MEASUREMENTS)
    s1_anchors = ("full10", "positive0", "boundary_near2", "pba_f10_int_k2")
    s1_est = compute_estimands_stage2(stage1_df[stage1_df["arm"].isin(s1_anchors)])
    s1_cc = s1_est[s1_est["post"] == "cc"].set_index("arm")
    stage1_diffs = {
        arm: float(abs(s2_cc.loc[arm, "patient_dice_pct"] - s1_cc.loc[arm, "patient_dice_pct"]))
        for arm in s1_anchors
    }
    max_stage1_diff = max(stage1_diffs.values())
    assert max_stage1_diff <= 0.05, f"Stage 1 anchor drift exceeded 0.05 pp: {stage1_diffs}"

    step9 = pd.read_csv(STEP9_MEASUREMENTS)
    s9_anchors = ("full10", "positive0", "boundary_near2")
    s9_est = compute_estimands_stage2(step9[step9["arm"].isin(s9_anchors)])
    s9_cc = s9_est[s9_est["post"] == "cc"].set_index("arm")
    s9_dice_diff = max(float(abs(s2_cc.loc[a, "patient_dice_pct"] - s9_cc.loc[a, "patient_dice_pct"])) for a in s9_anchors)
    s9_hd_diff = max(float(abs(s2_cc.loc[a, "patient_hd95_mm"] - s9_cc.loc[a, "patient_hd95_mm"])) for a in s9_anchors)
    assert s9_dice_diff <= 0.10, f"Step 9 patient CC Dice drift exceeded 0.10 pp: {s9_dice_diff}"
    assert s9_hd_diff <= 0.50, f"Step 9 patient CC HD95 drift exceeded 0.50 mm: {s9_hd_diff}"

    local_contrasts = paired_bootstrap_contrasts_stage2(
        df_meas,
        baselines=("full10", "positive0", "boundary_near2", "pba_f10_int_k2"),
        replicates=5000,
        seed=42,
    )
    decision = evaluate_stage2_decision(local_est, local_contrasts)
    prompt_stats = _audit_prompts_stage2(df_audit)

    receipt = {
        "run_dir": str(run_dir),
        "measurements_rows": int(len(df_meas)),
        "audit_slices": int(len(df_audit)),
        "stage1_anchor_diffs_pp": stage1_diffs,
        "stage1_max_patient_dice_diff_pp": max_stage1_diff,
        "step9_max_patient_dice_diff_pp": s9_dice_diff,
        "step9_max_patient_hd95_diff_mm": s9_hd_diff,
        "local_estimand_max_diff": max_est_diff,
        "prompt_stats": prompt_stats,
        "decision": decision,
        "remote_status": status_k,
        "worst_cases": worst_k,
    }
    (run_dir / "local_verification.json").write_text(json.dumps(json_safe(receipt), indent=2), encoding="utf-8")

    report_md = _build_report(run_dir, receipt, local_est, local_contrasts, decision, prompt_stats, status_k, pilot_k)
    report_path.write_text(report_md, encoding="utf-8")
    (run_dir / "adafob_v2_stage2_results_20261119.md").write_text(report_md, encoding="utf-8")
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
                "pareto_strong_arms": res["decision"]["pareto_strong_arms"],
                "stage1_max_patient_dice_diff_pp": res["stage1_max_patient_dice_diff_pp"],
                "step9_max_patient_dice_diff_pp": res["step9_max_patient_dice_diff_pp"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
