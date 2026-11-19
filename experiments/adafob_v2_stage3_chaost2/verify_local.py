"""Independent local verification and Stage-3 CHAOST2 cross-modality report generator."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (
    compute_estimands_stage3,
    evaluate_stage3_decision,
    json_safe,
    paired_bootstrap_contrasts_stage3,
)
from .selectors import ALL_REPORTED_ARMS, SUMMARY_ARMS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports/adafob_v2_stage3_chaost2_results_20261119.md"


def _load_kaggle_artifacts(out_dir: Path):
    zip_path = out_dir / "adafob_v2_stage3_outputs.zip"
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
    gate_p = src / "orientation_gate.json"
    if not gate_p.is_file():
        gate_p = src / "pilot/orientation_gate.json"
    gate_k = json.loads(gate_p.read_text(encoding="utf-8")) if gate_p.is_file() else {}
    return df_meas, df_est_k, contrasts_k, df_audit, worst_k, status_k, gate_k


def _audit_prompts_stage3(df_audit: pd.DataFrame) -> dict:
    out: dict = {"total_rows": int(len(df_audit)), "by_backbone": {}}
    for bb in sorted(df_audit["backbone"].unique()):
        g = df_audit[df_audit["backbone"] == bb]
        out["by_backbone"][str(bb)] = {
            "slices": int(len(g)),
            "mean_decoder_calls_per_slice": float(g["decoder_calls"].mean()),
            "pba_fallbacks": int(g["pba_f10_int_k2_fallback"].notna().sum()),
            "cbv_fixed_mean_k": float(g["cbv_f10_fixed_supp__k"].mean()),
            "cbv_fixed_choices": {str(k): int(v) for k, v in sorted(g["cbv_f10_fixed_supp__choice"].value_counts().items())},
            "cbv_no0_mean_k": float(g["cbv_f10_no0_supp__k"].mean()),
            "cbv_no0_choices": {str(k): int(v) for k, v in sorted(g["cbv_f10_no0_supp__choice"].value_counts().items())},
        }
    return out


def _fmt(v: float | None, digits: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "inf"
    return f"{v:.{digits}f}"


def _build_report(
    run_dir: Path,
    estimands: pd.DataFrame,
    contrasts: dict,
    decision: dict,
    prompt_stats: dict,
    status_k: dict,
    gate_k: dict,
) -> str:
    lines = [
        "# AdaFoB v2 Stage 3 Results: Cross-Modality CHAOST2 (Abd-MRI, Frozen SAM ViT-B & ViT-H, Setting I)",
        "",
        f"- **Run Directory:** `{run_dir}`",
        f"- **Kaggle Kernel Status:** `{status_k.get('status')}` (`episodes_per_backbone=80`, `patients=15`, `slices=1354`, `aggregate_gpu_hours={status_k.get('aggregate_gpu_hours', 0.0):.4f}`, `wall={status_k.get('wall_seconds', 0.0):.1f}s`)",
        f"- **Stage-0 Locked Orientation:** `{status_k.get('orientation_locked')}` (`best_fold0_r1_full10_pat_cc_dice_pct={gate_k.get('best_fold0_r1_full10_pat_cc_dice_pct', 0.0):.2f}%`, `intra_run_exact_5folds={gate_k.get('intra_run_exact_5folds')}`)",
        f"- **Pre-Registered Stage-3 Decision:** **`{decision['decision']}`** (OOD pass: `{decision['ood_passing']}`, Recall-protected pass: `{decision['recall_protected_passing']}`)",
        "",
        "## 1. Stage-0 Orientation & Multimask Parity Check (Fold 0, SAM ViT-B)",
        "",
        "| Orientation | Multimask Output | `full10` Raw Dice (%) | `full10` CC Dice (%) | `positive0` CC Dice (%) |",
        "| :--- | :---: | ---: | ---: | ---: |",
    ]
    for r in gate_k.get("orientation_comparisons", []):
        lines.append(
            f"| `{r['orientation']}` | `{r['multimask_output']}` | {_fmt(r['full10_pat_raw_dice_pct'])} | "
            f"{_fmt(r['full10_pat_cc_dice_pct'])} | {_fmt(r['positive0_pat_cc_dice_pct'])} |"
        )

    for bb in ("vit_b", "vit_h"):
        lines.extend(
            [
                "",
                f"## 2. Patient-Balanced 3D Metrics on CHAOST2 (`{bb}`, 80 Episodes / 15 Patients / 1,354 Slices)",
                "",
                "| Arm | Pat Raw Dice (%) | Pat CC Dice (%) [SD] | Ep CC Dice (%) | Pat HD95 (mm) | Pat Recall (%) | Detached FP / pat |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        cc_sub = estimands[(estimands["backbone"] == bb) & (estimands["post"] == "cc")].set_index("arm")
        rw_sub = estimands[(estimands["backbone"] == bb) & (estimands["post"] == "raw")].set_index("arm")
        for arm in SUMMARY_ARMS:
            if arm not in cc_sub.index:
                continue
            rc, rr = cc_sub.loc[arm], rw_sub.loc[arm]
            lines.append(
                f"| `{arm}` | {_fmt(rr['patient_dice_pct'])} | {_fmt(rc['patient_dice_pct'])} [{_fmt(rc['patient_dice_sd_pct'])}] | "
                f"{_fmt(rc['episode_dice_pct'])} | {_fmt(rc['patient_hd95_mm'])} | {_fmt(rc['patient_recall_pct'])} | "
                f"{_fmt(rr['patient_detached_fp'], 1)} |"
            )

        lines.extend(
            [
                "",
                f"### Per-Organ Patient-Balanced Breakdown (`{bb}`, `CC Dice % / CC Recall % / CC HD95 mm`)",
                "",
                "| Arm | LIVER | RK | LK | SPLEEN |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]
        )
        po = pd.DataFrame([r for r in contrasts["per_organ_cc"] if r["backbone"] == bb])
        pod = po.pivot(index="arm", columns="organ", values="patient_dice_pct")
        por = po.pivot(index="arm", columns="organ", values="patient_recall_pct")
        poh = po.pivot(index="arm", columns="organ", values="patient_hd95_mm")
        for arm in SUMMARY_ARMS:
            if arm not in pod.index:
                continue
            cells = [f"{_fmt(pod.loc[arm, o])} / {_fmt(por.loc[arm, o])} / {_fmt(poh.loc[arm, o], 1)}" for o in ("LIVER", "RK", "LK", "SPLEEN")]
            lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 3. Paired Patient-Cluster Bootstrap Contrasts & Backbone Scaling (`vit_h - vit_b`)",
            "",
            "| Backbone | Method Arm | Delta vs `boundary_near2` (pp) [95% CI] | Delta vs `random2_mean5` (pp) | Delta vs `full10` (pp) | Delta HD95 vs `near2` (mm) | Rec vs `full10` (pp) | OOD / Rec-Prot |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for r in decision["arms"]:
        ci = r["ci95_vs_boundary_near2_pp"]
        s_ci = f"{_fmt(r['delta_vs_boundary_near2_pp'])} [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "N/A"
        lines.append(
            f"| `{r['backbone']}` | `{r['arm']}` | {s_ci} | {_fmt(r['delta_vs_random2_mean5_pp'])} | "
            f"{_fmt(r['delta_vs_full10_pp'])} | {_fmt(r['delta_hd95_vs_boundary_near2_mm'])} | "
            f"{_fmt(r['recall_vs_full10_pp'])} | `{r['passes_ood_bar']}` / `{r['passes_recall_protected_bar']}` |"
        )

    lines.extend(
        [
            "",
            "### Backbone Scaling (`SAM ViT-H` minus `SAM ViT-B` on Same 15 Patients)",
            "",
            "| Arm | Delta Pat CC Dice (pp) [95% CI] | Delta Pat HD95 (mm) [95% CI] | Delta Pat Recall (pp) |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for b in contrasts.get("backbone_scaling", []):
        cd = b["cc_dice"]
        ch = b["cc_hd95"]
        cr = b["cc_recall"]
        sd = f"{_fmt(cd['point'])} [{_fmt(cd['ci95'][0])}, {_fmt(cd['ci95'][1])}]" if cd.get("ci95") else "N/A"
        sh = f"{_fmt(ch['point'])} [{_fmt(ch['ci95'][0])}, {_fmt(ch['ci95'][1])}]" if ch.get("ci95") else "N/A"
        lines.append(f"| `{b['arm']}` | {sd} | {sh} | {_fmt(cr.get('point'))} |")

    lines.extend(["", "## 4. CBV Choice Audit", ""])
    for bb, st in prompt_stats["by_backbone"].items():
        lines.append(
            f"- `{bb}`: mean decoder calls/slice `{st['mean_decoder_calls_per_slice']:.2f}`, "
            f"`pba` fallbacks `{st['pba_fallbacks']}`, "
            f"`cbv_f10_fixed_supp` mean `k={st['cbv_fixed_mean_k']:.3f}` choices `{st['cbv_fixed_choices']}`, "
            f"`cbv_f10_no0_supp` mean `k={st['cbv_no0_mean_k']:.3f}` choices `{st['cbv_no0_choices']}`."
        )
    lines.append("")
    return "\n".join(lines)


def verify_run(run_dir: Path, report_path: Path) -> dict:
    out_dir = run_dir / "kaggle_outputs"
    df_meas, df_est_k, contrasts_k, df_audit, worst_k, status_k, gate_k = _load_kaggle_artifacts(out_dir)

    expected_rows = 2 * 80 * len(ALL_REPORTED_ARMS) * 2
    assert len(df_meas) == expected_rows, f"Expected {expected_rows} measurement rows, got {len(df_meas)}"
    assert len(df_audit) == 2 * 1354, f"Expected 2708 audit rows, got {len(df_audit)}"

    local_est = compute_estimands_stage3(df_meas)
    num_cols = [c for c in local_est.columns if c not in ("backbone", "arm", "post", "is_summary_arm") and np.isfinite(local_est[c]).all()]
    max_est_diff = float((local_est[num_cols].to_numpy(dtype=float) - df_est_k[num_cols].to_numpy(dtype=float)).__abs__().max())
    assert max_est_diff <= 1e-10, f"Local vs Kaggle estimand mismatch: {max_est_diff}"

    local_contrasts = paired_bootstrap_contrasts_stage3(
        df_meas,
        baselines=("full10", "positive0", "random2_mean5", "boundary_near2", "pba_f10_int_k2"),
        replicates=5000,
        seed=42,
    )
    decision = evaluate_stage3_decision(local_est, local_contrasts)
    prompt_stats = _audit_prompts_stage3(df_audit)

    receipt = {
        "run_dir": str(run_dir),
        "measurements_rows": int(len(df_meas)),
        "audit_rows": int(len(df_audit)),
        "local_estimand_max_diff": max_est_diff,
        "orientation_gate": gate_k,
        "prompt_stats": prompt_stats,
        "decision": decision,
        "remote_status": status_k,
        "worst_cases": worst_k,
    }
    (run_dir / "local_verification.json").write_text(json.dumps(json_safe(receipt), indent=2), encoding="utf-8")

    report_md = _build_report(run_dir, local_est, local_contrasts, decision, prompt_stats, status_k, gate_k)
    report_path.write_text(report_md, encoding="utf-8")
    (run_dir / "adafob_v2_stage3_chaost2_results_20261119.md").write_text(report_md, encoding="utf-8")
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
                "ood_passing": res["decision"]["ood_passing"],
                "recall_protected_passing": res["decision"]["recall_protected_passing"],
                "local_estimand_max_diff": res["local_estimand_max_diff"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
