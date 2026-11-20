"""Kaggle T4/T4x2 orchestrator for AdaFoB v2 Stage 4 (CHAOST2 & SABS-CT, ViT-B & ViT-H)."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata as md
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from analysis.gate_a.runtime import sha256
from analysis.rebaseline_v1.common import read_config, write_json
from experiments.adafob_v2_stage3_chaost2.dataset import verify_chaost2_inputs
from experiments.adafob_v2_stage3_chaost2.remote import SAM_H_OFFICIAL_URL, _fetch_official_checkpoint
from experiments.option_a_reassessment_v1.metrics import json_safe
from experiments.option_a_reassessment_v1.remote import monitor, spawn_child, verify_data

from .metrics import (
    compute_estimands_stage4,
    evaluate_stage4_decision,
    paired_bootstrap_contrasts_stage4,
)
from .runner import load_tracked_sam
from .selectors import ALL_REPORTED_ARMS, METHOD_ARMS


def _worst_and_delta_cases_stage4(df: pd.DataFrame) -> dict:
    cc = df[df["post"] == "cc"]
    out: dict = {}
    for backbone in sorted(cc["backbone"].unique()):
        dbb = cc[cc["backbone"] == backbone]
        base_near = dbb[dbb["arm"] == "boundary_near2"].set_index("episode_id")
        base_full = dbb[dbb["arm"] == "full10"].set_index("episode_id")
        base_pba = dbb[dbb["arm"] == "pba_f10_int_k2"].set_index("episode_id")
        bb_out: dict = {}
        for arm in sorted(dbb["arm"].unique()):
            sub = dbb[dbb["arm"] == arm].sort_values("dice", ascending=True)
            worst = sub.head(10)[["episode_id", "fold", "patient_id", "organ", "dice", "hd95", "recall"]].to_dict("records")
            entry: dict = {"worst_10_cc": worst}
            if arm in METHOD_ARMS:
                s_idx = sub.set_index("episode_id")
                merged = sub[["episode_id", "fold", "patient_id", "organ", "dice", "hd95", "recall"]].copy()
                merged["delta_vs_near2_pp"] = merged["episode_id"].map(((s_idx["dice"] - base_near["dice"]) * 100.0).to_dict())
                merged["delta_vs_full10_pp"] = merged["episode_id"].map(((s_idx["dice"] - base_full["dice"]) * 100.0).to_dict())
                merged["delta_vs_pba_pp"] = merged["episode_id"].map(((s_idx["dice"] - base_pba["dice"]) * 100.0).to_dict())
                merged["recall_vs_full10_pp"] = merged["episode_id"].map(((s_idx["recall"] - base_full["recall"]) * 100.0).to_dict())
                entry["top_10_losses_vs_pba"] = merged.sort_values("delta_vs_pba_pp", ascending=True).head(10).to_dict("records")
                entry["top_10_gains_vs_pba"] = merged.sort_values("delta_vs_pba_pp", ascending=False).head(10).to_dict("records")
            bb_out[arm] = entry
        out[backbone] = bb_out
    return json_safe(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["chaost2", "sabs"], required=True)
    p.add_argument("--lock-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backbones", default="vit_b,vit_h")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    start = float(os.environ.get("ADAF_OB_ALLOC_START", time.time()))
    c = read_config(a.lock_dir)
    import torch

    n_gpus = int(torch.cuda.device_count())
    names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    assert n_gpus in (1, 2) and all("T4" in n for n in names), f"Unexpected hardware: {names}"

    backbones = tuple(x.strip() for x in a.backbones.split(",") if x.strip())
    assert set(backbones) <= {"vit_b", "vit_h"} and len(backbones) >= 1

    budget_gpu_hours = float(c.get("stage4_budget_gpu_hours", 2.5 if a.dataset == "chaost2" else 9.0))
    wall_cap = min((budget_gpu_hours * 3600.0) / float(n_gpus), 18000.0)
    deadline = start + wall_cap
    hw_info = {
        "dataset": a.dataset,
        "backbones": list(backbones),
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
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "SimpleITK==2.4.1"], check=True, timeout=180)

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

    if a.dataset == "chaost2":
        orientation = "flip_y"
        prep_manifest = json.loads((a.lock_dir / "chaost2_prep_manifest.json").read_text(encoding="utf-8"))
        roots = list(Path("/kaggle/input").glob("**/chaos_MR_T2_normalized"))
        data_folder = next(p for p in roots if (p / "image_1.nii.gz").is_file() or (p / "image_1.nii").is_file())
        cv0 = list(Path("/kaggle/input").glob("**/FSMIS_train_CHAOST2_cv0/ckpt.pth"))
        assert cv0, "CHAOST2 FoB checkpoints unavailable under /kaggle/input"
        ckpt_root = cv0[0].parents[2]
        input_check = verify_chaost2_inputs(data_folder, ckpt_root, prep_manifest)
        write_json(a.output / "remote_data_verification.json", input_check)
        ckpt_map = {
            fold: str(ckpt_root / input_check["checkpoint_rel_paths"][fold])
            for fold in input_check["checkpoint_rel_paths"]
        }
        expected_episodes = 80
        expected_slices = 1354
    else:
        orientation = "identity"
        roots = list(Path("/kaggle/input").glob("**/sabs_CT_normalized"))
        data_folder = next(p for p in roots if (p / "image_0.nii").exists() or (p / "image_0.nii.gz").exists())
        verify_data(data_folder, c, a.output)
        cv0 = list(Path("/kaggle/input").glob("**/FSMIS_train_SABS_cv0/ckpt.pth"))
        assert cv0, "SABS FoB checkpoints unavailable under /kaggle/input"
        ckpt_root = cv0[0].parents[2]
        ckpt_map: dict[str, str] = {}
        for fold in range(5):
            rel = c["fob_relative_paths"][str(fold)]
            p_ckpt = ckpt_root / rel
            assert sha256(p_ckpt) == c["fob_sha256"][str(fold)]
            ckpt_map[str(fold)] = str(p_ckpt)
        expected_episodes = 120
        expected_slices = int(c["total_slices"])

    ckpt_map_path = a.output / "ckpt_map.json"
    write_json(ckpt_map_path, ckpt_map)

    sam_b_candidates = list(Path("/kaggle/input").glob("**/sam_vit_b*.pth"))
    sam_b = next((p for p in sam_b_candidates if sha256(p) == c["sam_b_sha256"]), None)
    if sam_b is None:
        sam_b = _fetch_official_checkpoint(c["sam_b_official_url"], Path("/tmp/adafob_stage4_sam_b.pth"))
    assert sha256(sam_b) == c["sam_b_sha256"], "SAM-B SHA256 mismatch"

    sam_ckpts: dict[str, Path] = {"vit_b": sam_b}
    if "vit_h" in backbones:
        sam_h_candidates = sorted(Path("/kaggle/input").glob("**/sam_vit_h*.pth"))
        sam_h = sam_h_candidates[0] if sam_h_candidates else None
        if sam_h is None:
            sam_h = _fetch_official_checkpoint(SAM_H_OFFICIAL_URL, Path("/tmp/adafob_stage4_sam_h.pth"))
        pred_h, sam_h_meta = load_tracked_sam("vit_h", sam_h)
        del pred_h
        gc.collect()
        torch.cuda.empty_cache()
        write_json(a.output / "sam_h_verification.json", sam_h_meta)
        sam_ckpts["vit_h"] = sam_h

    try:
        pilot_dir = a.output / "pilot"
        pilot_cmd = [
            sys.executable,
            "-u",
            "-m",
            "experiments.adafob_v2_stage4.runner",
            "--dataset",
            a.dataset,
            "--mode",
            "pilot",
            "--orientation",
            orientation,
            "--lock-dir",
            str(a.lock_dir),
            "--data-dir",
            str(data_folder),
            "--checkpoint-map",
            str(ckpt_map_path),
            "--sam-checkpoint",
            str(sam_b),
            "--n-gpus",
            str(n_gpus),
            "--output",
            str(pilot_dir),
        ]
        pilot_proc = spawn_child(pilot_cmd, a.output / "pilot.log", 0)
        assert monitor([pilot_proc], deadline - 60) == [0], "Stage-0 determinism pilot failed; see pilot.log"
        pilot_res = json.loads((pilot_dir / "pilot_summary.json").read_text(encoding="utf-8"))
        proj_hours = float(pilot_res["projected_total_gpu_hours_both_backbones"]) + (
            float(n_gpus) * (time.time() - start) / 3600.0
        )
        proj = {
            "dataset": a.dataset,
            "orientation_locked": orientation,
            "intra_run_exact_5folds": bool(pilot_res["intra_run_exact_5folds"]),
            "vit_b_p90_slice_seconds": float(pilot_res["vit_b_p90_slice_seconds"]),
            "mean_decoder_calls": float(pilot_res["mean_decoder_calls"]),
            "projected_total_gpu_hours_both_backbones": proj_hours,
            "budget_gpu_hours": budget_gpu_hours,
            "within_budget": bool(proj_hours <= budget_gpu_hours),
        }
        write_json(a.output / "projection_before_full_run.json", proj)
        print(json.dumps(proj), flush=True)
        if not pilot_res["intra_run_exact_5folds"] or not proj["within_budget"]:
            write_json(
                a.output / "remote_status.json",
                {"status": "BLOCKED", "reason": "Stage-0 pilot or projected budget check failed", "projection": proj},
            )
            return

        for backbone in backbones:
            children = []
            for worker_id in range(n_gpus):
                w_cmd = [
                    sys.executable,
                    "-u",
                    "-m",
                    "experiments.adafob_v2_stage4.runner",
                    "--dataset",
                    a.dataset,
                    "--mode",
                    "worker",
                    "--backbone",
                    backbone,
                    "--orientation",
                    orientation,
                    "--lock-dir",
                    str(a.lock_dir),
                    "--data-dir",
                    str(data_folder),
                    "--checkpoint-map",
                    str(ckpt_map_path),
                    "--sam-checkpoint",
                    str(sam_ckpts[backbone]),
                    "--worker",
                    str(worker_id),
                    "--num-workers",
                    str(n_gpus),
                    "--deadline-epoch",
                    str(deadline - 60),
                    "--output",
                    str(a.output / "inference" / backbone / f"worker{worker_id}"),
                ]
                children.append(spawn_child(w_cmd, a.output / f"worker{worker_id}_{backbone}.log", worker_id))
            states = monitor(children, deadline - 30)
            assert all(s == 0 for s in states), f"Workers failed for {backbone}: {states}"

        metric_files = sorted((a.output / "inference").glob("*/worker*/episode_*/episode_*_metrics.csv"))
        audit_files = sorted((a.output / "inference").glob("*/worker*/episode_*/episode_*_audit.json"))
        n_bb = len(backbones)
        assert len(metric_files) == n_bb * expected_episodes and len(audit_files) == n_bb * expected_episodes, (
            f"Expected {n_bb * expected_episodes} episode files, got {len(metric_files)} / {len(audit_files)}"
        )

        df = pd.concat([pd.read_csv(f) for f in metric_files], ignore_index=True)
        df = df.sort_values(["backbone", "episode_id", "arm", "post"]).reset_index(drop=True)
        expected_rows = n_bb * expected_episodes * len(ALL_REPORTED_ARMS) * 2
        assert len(df) == expected_rows, f"Expected {expected_rows} measurement rows, got {len(df)}"
        df.to_csv(a.output / "measurements.csv", index=False)

        audits = []
        for f in audit_files:
            audits.extend(json.loads(f.read_text(encoding="utf-8")))
        df_audit = pd.DataFrame(audits).sort_values(["backbone", "episode_id", "compact_z"]).reset_index(drop=True)
        assert len(df_audit) == n_bb * expected_slices, (
            f"Expected {n_bb * expected_slices} audit rows, got {len(df_audit)}"
        )
        df_audit.to_csv(a.output / "prompt_audit.csv", index=False)

        estimands = compute_estimands_stage4(df)
        estimands.to_csv(a.output / "both_estimands.csv", index=False)

        contrasts = paired_bootstrap_contrasts_stage4(
            df,
            replicates=int(c.get("bootstrap_replicates", 5000)),
            seed=int(c.get("seed", 42)),
        )
        write_json(a.output / "paired_contrasts.json", contrasts)

        decision = evaluate_stage4_decision(estimands, contrasts)
        write_json(a.output / "decision_summary.json", decision)
        write_json(a.output / "worst_cases.json", _worst_and_delta_cases_stage4(df))

        write_json(
            a.output / "remote_status.json",
            {
                "status": "COMPLETED",
                "dataset": a.dataset,
                "backbones": list(backbones),
                "orientation_locked": orientation,
                "episodes_per_backbone": expected_episodes,
                "patients_per_backbone": int(df["patient_id"].nunique()),
                "slices_per_backbone": expected_slices,
                "measurement_rows": int(len(df)),
                "decision": decision["decision"],
                "passing_vit_b": decision["passing_vit_b"],
                "passing_vit_h": decision["passing_vit_h"],
                "backbone_invariant_passing": decision["backbone_invariant_passing"],
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
