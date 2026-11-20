"""Independent local verification, anchor parity check, and Stage-4 report generator."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (
    compute_estimands_stage4,
    evaluate_stage4_decision,
    json_safe,
    paired_bootstrap_contrasts_stage4,
)
from .selectors import ALL_REPORTED_ARMS, CBV_ARMS, SUMMARY_ARMS

ROOT = Path(__file__).resolve().parents[2]
STAGE3_ESTIMANDS = ROOT / "experiments/runs/adafob_v2_stage3_chaost2_20261119/kaggle_outputs/both_estimands.csv"
STAGE2_ESTIMANDS = ROOT / "experiments/runs/adafob_v2_stage2_20261119/kaggle_outputs/both_estimands.csv"


def _load_kaggle_artifacts(out_dir: Path):
    zip_path = out_dir / "adafob_v2_stage4_outputs.zip"
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
    entry_k = json.loads((src / "adafob_v2_stage4_entry_status.json").read_text(encoding="utf-8"))
    return df_meas, df_est_k, contrasts_k, df_audit, worst_k, status_k, entry_k


def _audit_prompts_stage4(df_audit: pd.DataFrame) -> dict:
    out: dict = {"total_rows": int(len(df_audit)), "by_backbone": {}}
    for bb in sorted(df_audit["backbone"].unique()):
        g = df_audit[df_audit["backbone"] == bb]
        cbv_summary: dict = {}
        for arm in CBV_ARMS:
            cbv_summary[arm] = {
                "mean_k": float(g[f"{arm}__k"].mean()),
                "choices": {str(k): int(v) for k, v in sorted(g[f"{arm}__choice"].value_counts().items())},
                "fallbacks": int(g[f"{arm}__fallback"].notna().sum()),
            }
        out["by_backbone"][str(bb)] = {
            "slices": int(len(g)),
            "mean_decoder_calls_per_slice": float(g["decoder_calls"].mean()),
            "pba_fallbacks": int(g["pba_f10_int_k2_fallback"].notna().sum()),
            "fob_prior_mean": float(g["fob_prior_mean"].mean()),
            "cbv_arms": cbv_summary,
        }
    return out


def _check_anchor_parity(dataset: str, local_est: pd.DataFrame) -> dict:
    if dataset == "chaost2":
        ref_path = STAGE3_ESTIMANDS
        anchor_map = {
            "full10": "full10",
            "positive0": "positive0",
            "random2_mean5": "random2_mean5",
            "boundary_near2": "boundary_near2",
            "pba_f10_int_k2": "pba_f10_int_k2",
            "cbv_sam_localring_all": "cbv_f10_fixed_supp",
            "cbv_sam_localring_no0": "cbv_f10_no0_supp",
        }
        ref = pd.read_csv(ref_path)
    elif dataset == "sabs":
        ref_path = STAGE2_ESTIMANDS
        anchor_map = {
            "full10": "full10",
            "positive0": "positive0",
            "boundary_near2": "boundary_near2",
            "pba_f10_int_k2": "pba_f10_int_k2",
            "cbv_sam_localring_all": "cbv_f10_fixed_supp",
        }
        ref = pd.read_csv(ref_path)
        ref["backbone"] = "vit_b"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    diffs: list[dict] = []
    max_diff = 0.0
    for _, row in local_est.iterrows():
        bb = str(row["backbone"])
        arm = str(row["arm"])
        post = str(row["post"])
        if arm not in anchor_map:
            continue
        ref_arm = anchor_map[arm]
        m = ref[(ref["backbone"] == bb) & (ref["arm"] == ref_arm) & (ref["post"] == post)]
        if m.empty:
            continue
        ref_val = float(m.iloc[0]["patient_dice_pct"])
        cur_val = float(row["patient_dice_pct"])
        d = abs(cur_val - ref_val)
        max_diff = max(max_diff, d)
        diffs.append(
            {
                "backbone": bb,
                "arm": arm,
                "ref_arm": ref_arm,
                "post": post,
                "stage4_dice_pct": cur_val,
                "prior_dice_pct": ref_val,
                "abs_diff_pp": d,
            }
        )
    assert max_diff <= 0.005, f"Anchor parity drift exceeded 0.005 pp: {max_diff} ({diffs})"
    return {"reference_file": str(ref_path), "checked_rows": len(diffs), "max_abs_diff_pp": float(max_diff), "details": diffs}


def _fmt(v: float | None, digits: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "inf"
    return f"{v:.{digits}f}"


def _build_report(
    dataset: str,
    run_dir: Path,
    estimands: pd.DataFrame,
    contrasts: dict,
    decision: dict,
    prompt_stats: dict,
    anchor_check: dict,
    status_k: dict,
) -> str:
    title = "CHAOST2 (Abd-MRI)" if dataset == "chaost2" else "SABS-CT (Abd-CT)"
    ep_n = status_k.get("episodes_per_backbone")
    pat_n = status_k.get("patients_per_backbone")
    sl_n = status_k.get("slices_per_backbone")
    lines = [
        f"# AdaFoB v2 Stage 4 Results: {title} Multi-Backbone CBV Verification (Setting I)",
        "",
        f"- **Run Directory:** `{run_dir}`",
        f"- **Kaggle Kernel Status:** `{status_k.get('status')}` (`episodes_per_backbone={ep_n}`, `patients={pat_n}`, `slices={sl_n}`, `aggregate_gpu_hours={status_k.get('aggregate_gpu_hours', 0.0):.4f}`, `wall={status_k.get('wall_seconds', 0.0):.1f}s`)",
        f"- **Anchor Parity vs Prior Locked Stage:** `max_abs_diff_pp={anchor_check['max_abs_diff_pp']:.4f}` (`checked_rows={anchor_check['checked_rows']}`)",
        f"- **Pre-Registered Stage-4 Decision:** **`{decision['decision']}`** (`passing_vit_b={decision['passing_vit_b']}`, `passing_vit_h={decision['passing_vit_h']}`, `backbone_invariant_passing={decision['backbone_invariant_passing']}`)",
        "",
    ]

    for bb in sorted(estimands["backbone"].unique()):
        lines.extend(
            [
                f"## Patient-Balanced 3D Metrics (`{bb}`, {ep_n} Episodes / {pat_n} Patients / {sl_n} Slices)",
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
        lines.append("")

    lines.extend(
        [
            "## Paired Patient-Cluster Bootstrap Contrasts (5,000 Replicates, `seed=42`)",
            "",
            "| Backbone | Arm | vs `near2` (pp) [95% CI] | vs `full10` (pp) [95% CI] | vs `pba_f10` (pp) [95% CI] | vs `legacy_cbv_no0` (pp) [95% CI] | Liver Rec vs `pba` (pp) | Passes BB |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for r in decision["arms"]:
        def _sci(delta, ci):
            if delta is None or ci is None:
                return "N/A"
            return f"{_fmt(delta)} [{_fmt(ci[0])}, {_fmt(ci[1])}]"
        lines.append(
            f"| `{r['backbone']}` | `{r['arm']}` | "
            f"{_sci(r['delta_vs_boundary_near2_pp'], r['ci95_vs_boundary_near2_pp'])} | "
            f"{_sci(r['delta_vs_full10_pp'], r['ci95_vs_full10_pp'])} | "
            f"{_sci(r['delta_vs_pba_pp'], r['ci95_vs_pba_pp'])} | "
            f"{_sci(r['delta_vs_legacy_cbv_pp'], r['ci95_vs_legacy_cbv_pp'])} | "
            f"{_fmt(r['liver_recall_vs_pba_pp'])} | `{r['passes_backbone_bar']}` |"
        )

    lines.extend(["", "## CBV Candidate Selection Audit", ""])
    for bb, st in prompt_stats["by_backbone"].items():
        lines.append(f"- `{bb}`: mean decoder calls/slice `{st['mean_decoder_calls_per_slice']:.2f}`, `pba` fallbacks `{st['pba_fallbacks']}`, mean FoB prior `{st['fob_prior_mean']:.4f}`")
        for arm, info in st["cbv_arms"].items():
            lines.append(f"  - `{arm}`: mean `k={info['mean_k']:.3f}`, choices `{info['choices']}`")
    lines.append("")
    return "\n".join(lines)


def verify_run(dataset: str, run_dir: Path, report_path: Path | None = None) -> dict:
    out_dir = run_dir / "kaggle_outputs"
    df_meas, df_est_k, contrasts_k, df_audit, worst_k, status_k, entry_k = _load_kaggle_artifacts(out_dir)

    pkg_receipt = json.loads((run_dir / "package_receipt.json").read_text(encoding="utf-8"))
    bundle_match = pkg_receipt["source_bundle_sha256"] == entry_k["source_bundle_sha256"]
    assert bundle_match, "Source bundle SHA256 mismatch between package receipt and Kaggle run"

    ep_n = 80 if dataset == "chaost2" else 120
    sl_n = 1354 if dataset == "chaost2" else 4009
    n_bb = int(df_meas["backbone"].nunique())
    expected_rows = n_bb * ep_n * len(ALL_REPORTED_ARMS) * 2
    assert len(df_meas) == expected_rows, f"Expected {expected_rows} rows, got {len(df_meas)}"
    assert len(df_audit) == n_bb * sl_n, f"Expected {n_bb * sl_n} audit rows, got {len(df_audit)}"

    local_est = compute_estimands_stage4(df_meas)
    num_cols = [c for c in local_est.columns if c not in ("backbone", "arm", "post", "is_summary_arm") and np.isfinite(local_est[c]).all()]
    max_est_diff = float((local_est[num_cols].to_numpy(dtype=float) - df_est_k[num_cols].to_numpy(dtype=float)).__abs__().max())
    assert max_est_diff <= 1e-10, f"Local vs Kaggle estimand mismatch: {max_est_diff}"

    anchor_check = _check_anchor_parity(dataset, local_est)
    local_contrasts = paired_bootstrap_contrasts_stage4(df_meas, replicates=5000, seed=42)
    decision = evaluate_stage4_decision(local_est, local_contrasts)
    prompt_stats = _audit_prompts_stage4(df_audit)

    receipt = {
        "dataset": dataset,
        "run_dir": str(run_dir),
        "bundle_sha256_match": bundle_match,
        "measurements_rows": int(len(df_meas)),
        "audit_rows": int(len(df_audit)),
        "local_estimand_max_diff": max_est_diff,
        "anchor_parity": anchor_check,
        "prompt_stats": prompt_stats,
        "decision": decision,
        "remote_status": status_k,
        "worst_cases": worst_k,
    }
    (run_dir / "local_verification.json").write_text(json.dumps(json_safe(receipt), indent=2), encoding="utf-8")

    if report_path is None:
        report_path = ROOT / f"reports/adafob_v2_stage4_{dataset}_results_20261119.md"
    report_md = _build_report(dataset, run_dir, local_est, local_contrasts, decision, prompt_stats, anchor_check, status_k)
    report_path.write_text(report_md, encoding="utf-8")
    (run_dir / f"adafob_v2_stage4_{dataset}_results_20261119.md").write_text(report_md, encoding="utf-8")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["chaost2", "sabs"], required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--report-path", type=Path, default=None)
    a = p.parse_args()
    res = verify_run(a.dataset, a.run_dir, a.report_path)
    print(
        json.dumps(
            {
                "dataset": res["dataset"],
                "bundle_sha256_match": res["bundle_sha256_match"],
                "local_estimand_max_diff": res["local_estimand_max_diff"],
                "anchor_max_abs_diff_pp": res["anchor_parity"]["max_abs_diff_pp"],
                "decision": res["decision"]["decision"],
                "passing_vit_b": res["decision"]["passing_vit_b"],
                "passing_vit_h": res["decision"]["passing_vit_h"],
                "backbone_invariant_passing": res["decision"]["backbone_invariant_passing"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
