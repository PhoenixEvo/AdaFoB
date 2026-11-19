"""Kaggle T4/T4x2 orchestrator for AdaFoB v2 Stage 3 (CHAOST2 MRI, SAM ViT-B & ViT-H)."""
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

from analysis.gate_a.runtime import sha256
from analysis.rebaseline_v1.common import read_config, write_json
from experiments.option_a_reassessment_v1.metrics import json_safe
from experiments.option_a_reassessment_v1.remote import monitor, spawn_child

from .dataset import verify_chaost2_inputs
from .metrics import (
    compute_estimands_stage3,
    evaluate_stage3_decision,
    paired_bootstrap_contrasts_stage3,
)
from .runner import load_tracked_sam
from .selectors import ALL_REPORTED_ARMS, METHOD_ARMS

SAM_H_OFFICIAL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


def _worst_and_delta_cases_stage3(df: pd.DataFrame) -> dict:
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
                entry["top_10_losses_vs_near2"] = merged.sort_values("delta_vs_near2_pp", ascending=True).head(10).to_dict("records")
                entry["top_10_gains_vs_near2"] = merged.sort_values("delta_vs_near2_pp", ascending=False).head(10).to_dict("records")
            bb_out[arm] = entry
        out[backbone] = bb_out
    return json_safe(out)


def _fetch_official_checkpoint(url: str, dst: Path) -> Path:
    assert not dst.exists()
    print(f"Downloading SAM weights from {url} -> {dst}", flush=True)
    with urllib.request.urlopen(url, timeout=180) as resp:
        if urllib.parse.urlparse(resp.geturl()).hostname != "dl.fbaipublicfiles.com":
            raise RuntimeError(f"Unexpected redirect host for {url}: {resp.geturl()}")
        with dst.open("xb") as f:
            total = 0
            while True:
                block = resp.read(16 * 1024 * 1024)
                if not block:
                    break
                f.write(block)
                total += len(block)
                if total % (128 * 1024 * 1024) == 0:
                    print(f"Downloaded bytes: {total}", flush=True)
    return dst


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lock-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    start = float(os.environ.get("ADAF_OB_ALLOC_START", time.time()))
    c = read_config(a.lock_dir)
    prep_manifest = json.loads((a.lock_dir / "chaost2_prep_manifest.json").read_text(encoding="utf-8"))
    import torch

    n_gpus = int(torch.cuda.device_count())
    names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    assert n_gpus in (1, 2) and all("T4" in n for n in names), f"Unexpected hardware: {names}"

    budget_gpu_hours = float(c.get("stage3_budget_gpu_hours", 2.5))
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
    ckpt_map_path = a.output / "chaost2_ckpt_map.json"
    write_json(ckpt_map_path, ckpt_map)

    sam_b_candidates = list(Path("/kaggle/input").glob("**/sam_vit_b*.pth"))
    sam_b = next((p for p in sam_b_candidates if sha256(p) == c["sam_b_sha256"]), None)
    if sam_b is None:
        sam_b = _fetch_official_checkpoint(c["sam_b_official_url"], Path("/tmp/adafob_stage3_sam_b.pth"))
    assert sha256(sam_b) == c["sam_b_sha256"], "SAM-B SHA256 mismatch"

    sam_h_candidates = sorted(Path("/kaggle/input").glob("**/sam_vit_h*.pth"))
    sam_h = sam_h_candidates[0] if sam_h_candidates else None
    if sam_h is None:
        sam_h = _fetch_official_checkpoint(SAM_H_OFFICIAL_URL, Path("/tmp/adafob_stage3_sam_h.pth"))
    # Pre-verify strict SAM ViT-H loading on CPU/GPU cleanly before worker launch.
    pred_h, sam_h_meta = load_tracked_sam("vit_h", sam_h)
    del pred_h
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    write_json(a.output / "sam_h_verification.json", sam_h_meta)

    try:
        pilot_dir = a.output / "pilot"
        pilot_cmd = [
            sys.executable,
            "-u",
            "-m",
            "experiments.adafob_v2_stage3_chaost2.runner",
            "--mode",
            "pilot",
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
        assert monitor([pilot_proc], deadline - 60) == [0], "Stage-0 orientation/determinism pilot failed; see pilot.log"
        gate = json.loads((pilot_dir / "orientation_gate.json").read_text(encoding="utf-8"))
        shutil_orientation = gate["best_orientation"]
        proj_hours = float(gate["projected_total_gpu_hours_both_backbones"])
        proj = {
            "best_orientation": shutil_orientation,
            "best_fold0_r1_full10_pat_cc_dice_pct": float(gate["best_fold0_r1_full10_pat_cc_dice_pct"]),
            "intra_run_exact_5folds": bool(gate["intra_run_exact_5folds"]),
            "projected_total_gpu_hours_both_backbones": proj_hours,
            "budget_gpu_hours": budget_gpu_hours,
            "within_budget": bool(proj_hours <= budget_gpu_hours),
        }
        write_json(a.output / "projection_before_full_run.json", proj)
        print(json.dumps(proj), flush=True)
        if not gate["intra_run_exact_5folds"] or not proj["within_budget"]:
            write_json(
                a.output / "remote_status.json",
                {"status": "BLOCKED", "reason": "Stage-0 gate or projected budget check failed", "projection": proj},
            )
            return

        for backbone, sam_ckpt in (("vit_b", sam_b), ("vit_h", sam_h)):
            children = []
            for worker_id in range(n_gpus):
                w_cmd = [
                    sys.executable,
                    "-u",
                    "-m",
                    "experiments.adafob_v2_stage3_chaost2.runner",
                    "--mode",
                    "worker",
                    "--backbone",
                    backbone,
                    "--orientation",
                    shutil_orientation,
                    "--data-dir",
                    str(data_folder),
                    "--checkpoint-map",
                    str(ckpt_map_path),
                    "--sam-checkpoint",
                    str(sam_ckpt),
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
        assert len(metric_files) == 160 and len(audit_files) == 160, (
            f"Expected 160 episode files (80 vit_b + 80 vit_h), got {len(metric_files)} / {len(audit_files)}"
        )

        df = pd.concat([pd.read_csv(f) for f in metric_files], ignore_index=True)
        df = df.sort_values(["backbone", "episode_id", "arm", "post"]).reset_index(drop=True)
        expected_rows = 2 * 80 * len(ALL_REPORTED_ARMS) * 2
        assert len(df) == expected_rows, f"Expected {expected_rows} measurement rows, got {len(df)}"
        df.to_csv(a.output / "measurements.csv", index=False)

        audits = []
        for f in audit_files:
            audits.extend(json.loads(f.read_text(encoding="utf-8")))
        df_audit = pd.DataFrame(audits).sort_values(["backbone", "episode_id", "compact_z"]).reset_index(drop=True)
        assert len(df_audit) == 2 * 1354, f"Expected 2708 audit rows (2 backbones * 1354 slices), got {len(df_audit)}"
        df_audit.to_csv(a.output / "prompt_audit.csv", index=False)

        estimands = compute_estimands_stage3(df)
        estimands.to_csv(a.output / "both_estimands.csv", index=False)

        contrasts = paired_bootstrap_contrasts_stage3(
            df,
            baselines=("full10", "positive0", "random2_mean5", "boundary_near2", "pba_f10_int_k2"),
            replicates=int(c.get("bootstrap_replicates", 5000)),
            seed=int(c.get("seed", 42)),
        )
        write_json(a.output / "paired_contrasts.json", contrasts)

        decision = evaluate_stage3_decision(estimands, contrasts)
        write_json(a.output / "decision_summary.json", decision)
        write_json(a.output / "worst_cases.json", _worst_and_delta_cases_stage3(df))

        write_json(
            a.output / "remote_status.json",
            {
                "status": "COMPLETED",
                "orientation_locked": shutil_orientation,
                "episodes_per_backbone": 80,
                "patients_per_backbone": int(df["patient_id"].nunique()),
                "slices_per_backbone": 1354,
                "measurement_rows": int(len(df)),
                "decision": decision["decision"],
                "ood_passing": decision["ood_passing"],
                "recall_protected_passing": decision["recall_protected_passing"],
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
