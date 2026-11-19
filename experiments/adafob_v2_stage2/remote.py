"""Kaggle T4/T4x2 orchestrator for AdaFoB v2 Stage 2."""
from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from analysis.rebaseline_v1.common import read_config, sha256, write_json
from experiments.adafob_v2_stage1.remote import _worst_and_delta_cases_v2
from experiments.option_a_reassessment_v1.remote import monitor, spawn_child, verify_data

from .metrics import (
    compute_estimands_stage2,
    evaluate_stage2_decision,
    paired_bootstrap_contrasts_stage2,
)
from .selectors import ARMS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lock-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    start = float(os.environ.get("ADAF_OB_ALLOC_START", time.time()))
    c = read_config(a.lock_dir)
    import torch

    n_gpus = int(torch.cuda.device_count())
    names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    assert n_gpus in (1, 2) and all("T4" in n for n in names), f"Unexpected hardware: {names}"

    budget_gpu_hours = float(c.get("stage2_budget_gpu_hours", 2.5))
    wall_cap = min((budget_gpu_hours * 3600.0) / float(n_gpus), 5150.0)
    deadline = start + wall_cap
    hw_info = {
        "allocation_start_epoch": start,
        "deadline_epoch": deadline,
        "wall_cap_seconds": wall_cap,
        "n_gpus": n_gpus,
        "device_names": names,
        "budget_gpu_hours": budget_gpu_hours,
    }
    write_json(a.output / "hardware_budget.json", hw_info)
    print(json.dumps(hw_info), flush=True)

    if importlib.util.find_spec("SimpleITK") is None:
        print("Installing SimpleITK==2.4.1 (--no-deps) for NIfTI reading", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "SimpleITK==2.4.1"], check=True, timeout=120)

    inventory = {d.metadata["Name"]: d.version for d in md.distributions() if d.metadata["Name"]}
    write_json(
        a.output / "environment_kaggle.json",
        {
            "python": sys.version,
            "packages": inventory,
            "gpu_names": names,
            "torch_cuda": torch.version.cuda,
            "precision": "float32 TF32 off; no autocast",
        },
    )

    roots = list(Path("/kaggle/input").glob("**/sabs_CT_normalized"))
    data_folder = next(p for p in roots if (p / "image_0.nii").exists() or (p / "image_0.nii.gz").exists())
    verify_data(data_folder, c, a.output)

    cv0 = list(Path("/kaggle/input").glob("**/FSMIS_train_SABS_cv0/ckpt.pth"))
    assert cv0, "FoB checkpoints unavailable under /kaggle/input"
    ckpt_root = cv0[0].parents[2]
    for fold in range(5):
        assert sha256(ckpt_root / c["fob_relative_paths"][str(fold)]) == c["fob_sha256"][str(fold)]

    sam_candidates = list(Path("/kaggle/input").glob("**/sam_vit_b*.pth"))
    sam = next((p for p in sam_candidates if sha256(p) == c["sam_b_sha256"]), None)
    if sam is None:
        sam = Path("/tmp/adafob_v2_stage2_sam_b.pth")
        assert not sam.exists()
        print("Fetching official SAM ViT-B release and verifying SHA256", flush=True)
        with urllib.request.urlopen(c["sam_b_official_url"], timeout=60) as resp:
            if urllib.parse.urlparse(resp.geturl()).hostname != "dl.fbaipublicfiles.com":
                raise RuntimeError("Unexpected redirect host for SAM-B download")
            with sam.open("xb") as f:
                total = 0
                while True:
                    block = resp.read(8 * 1024 * 1024)
                    if not block:
                        break
                    f.write(block)
                    total += len(block)
                    if total % (64 * 1024 * 1024) == 0:
                        print(f"SAM-B bytes {total}", flush=True)
    assert sha256(sam) == c["sam_b_sha256"], "SAM-B SHA256 mismatch"

    base_args = [
        "--lock-dir",
        str(a.lock_dir),
        "--data-dir",
        str(data_folder),
        "--checkpoint-root",
        str(ckpt_root),
        "--sam-checkpoint",
        str(sam),
    ]

    try:
        pilot_dir = a.output / "pilot"
        pilot_cmd = [
            sys.executable,
            "-u",
            "-m",
            "experiments.adafob_v2_stage2.remote",
            "--mode",
            "pilot",
            "--n-gpus",
            str(n_gpus),
            "--output",
            str(pilot_dir),
            *base_args,
        ]
        # Invoke runner directly for pilot and worker subprocesses
        pilot_cmd[3] = "experiments.adafob_v2_stage2.runner"
        pilot_proc = spawn_child(pilot_cmd, a.output / "pilot.log", 0)
        assert monitor([pilot_proc], deadline - 60) == [0], "Stage-0 pilot failed; see pilot.log"
        pilot_res = json.loads((pilot_dir / "pilot_summary.json").read_text(encoding="utf-8"))
        proj_hours = float(pilot_res["projected_total_gpu_hours"]) + float(n_gpus) * (time.time() - start) / 3600.0
        pilot_ok = bool(pilot_res["intra_run_exact_all_slices"] and pilot_res["step9_cuda_tolerance_passed"])
        proj = {
            "intra_run_exact_all_slices": bool(pilot_res["intra_run_exact_all_slices"]),
            "step9_cuda_tolerance_passed": bool(pilot_res["step9_cuda_tolerance_passed"]),
            "max_neg_coord_diff_px": float(pilot_res["max_neg_coord_diff_px"]),
            "min_full10_cc_dice_vs_step9": float(pilot_res["min_full10_cc_dice_vs_step9"]),
            "elapsed_gpu_hours_before_full": float(n_gpus) * (time.time() - start) / 3600.0,
            "projected_full_run_gpu_hours": float(pilot_res["projected_total_gpu_hours"]),
            "projected_total_gpu_hours": float(proj_hours),
            "budget_gpu_hours": budget_gpu_hours,
            "within_budget": bool(proj_hours <= budget_gpu_hours),
        }
        write_json(a.output / "projection_before_full_run.json", proj)
        print(json.dumps(proj), flush=True)
        if not pilot_ok or not proj["within_budget"]:
            write_json(
                a.output / "remote_status.json",
                {"status": "BLOCKED", "reason": "Stage-0 parity or projected budget check failed", "projection": proj},
            )
            return

        children = []
        for worker_id in range(n_gpus):
            w_cmd = [
                sys.executable,
                "-u",
                "-m",
                "experiments.adafob_v2_stage2.runner",
                "--mode",
                "worker",
                "--worker",
                str(worker_id),
                "--num-workers",
                str(n_gpus),
                "--deadline-epoch",
                str(deadline - 60),
                "--output",
                str(a.output / "inference" / f"worker{worker_id}"),
                *base_args,
            ]
            children.append(spawn_child(w_cmd, a.output / f"worker{worker_id}.log", worker_id))
        states = monitor(children, deadline - 30)
        assert all(s == 0 for s in states), f"Worker failed with exit codes {states}"

        metric_files = sorted((a.output / "inference").glob("worker*/episode_*/episode_*_metrics.csv"))
        audit_files = sorted((a.output / "inference").glob("worker*/episode_*/episode_*_audit.json"))
        assert len(metric_files) == 120 and len(audit_files) == 120

        df = pd.concat([pd.read_csv(f) for f in metric_files], ignore_index=True)
        df = df.sort_values(["episode_id", "arm", "post"]).reset_index(drop=True)
        assert len(df) == 120 * len(ARMS) * 2
        df.to_csv(a.output / "measurements.csv", index=False)

        audits = []
        for f in audit_files:
            audits.extend(json.loads(f.read_text(encoding="utf-8")))
        df_audit = pd.DataFrame(audits).sort_values(["episode_id", "compact_z"]).reset_index(drop=True)
        assert len(df_audit) == int(c["total_slices"])
        df_audit.to_csv(a.output / "prompt_audit.csv", index=False)

        estimands = compute_estimands_stage2(df)
        estimands.to_csv(a.output / "both_estimands.csv", index=False)

        contrasts = paired_bootstrap_contrasts_stage2(
            df,
            baselines=("full10", "positive0", "boundary_near2", "pba_f10_int_k2"),
            replicates=int(c.get("bootstrap_replicates", 5000)),
            seed=int(c.get("seed", 42)),
        )
        write_json(a.output / "paired_contrasts.json", contrasts)

        decision = evaluate_stage2_decision(estimands, contrasts)
        write_json(a.output / "decision_summary.json", decision)
        write_json(a.output / "worst_cases.json", _worst_and_delta_cases_v2(df))

        write_json(
            a.output / "remote_status.json",
            {
                "status": "COMPLETED",
                "episodes": 120,
                "patients": int(df["patient_id"].nunique()),
                "slices": int(len(df_audit)),
                "measurement_rows": int(len(df)),
                "decision": decision["decision"],
                "passing_arms": decision["passing_arms"],
                "pareto_strong_arms": decision["pareto_strong_arms"],
                "wall_seconds": float(time.time() - start),
                "aggregate_gpu_hours": float(n_gpus) * (time.time() - start) / 3600.0,
            },
        )
        print(json.dumps({"remote_status": "COMPLETED", "decision": decision["decision"]}), flush=True)
    except Exception as exc:
        write_json(a.output / "remote_failure.json", {"status": "FAIL", "type": type(exc).__name__, "reason": str(exc)})
        raise


if __name__ == "__main__":
    main()
