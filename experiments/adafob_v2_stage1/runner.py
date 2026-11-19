"""Slice execution, Stage-0 parity pilot, and resumable workers for AdaFoB v2 Stage 1."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.controls_v1.core import baseline_digest, coordinate_digest, proximity
from analysis.gate_a.runtime import seed_all
from analysis.rebaseline_v1.common import (
    cc_mask,
    episode_inputs,
    generate,
    load_models,
    predict,
    read_config,
    read_manifest,
    restore_xy,
    rgb_image,
    tracked_sam,
    write_json,
)
from experiments.option_a_reassessment_v1.runner import SupportFeatureCache, _dice2d

from .metrics import measure_episode_v2
from .selectors import (
    ARMS,
    compute_support_evidence,
    mask_on_feature_grid,
    place_pba_fixed_k2,
    place_semgate_pba,
)


def run_stage1_slice(
    model,
    predictor,
    support_cache: SupportFeatureCache,
    case: dict,
    expected_fp: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict, dict[str, np.ndarray]]:
    """Execute all 11 Stage-1 arms on one slice (only <= 11 SAM decoder passes)."""
    t0 = time.perf_counter()
    neg, pos = generate(model, case)
    assert neg.shape == (10, 2), f"Unexpected neg shape {neg.shape}"
    nr = restore_xy(neg, (256, 256), (257, 257))
    pr = restore_xy(pos, (256, 256), (257, 257))
    step8_coord_sha_exact = coordinate_digest(neg, pos, nr, pr) == expected_fp["coordinates"]

    supp_feat, supp_miss = support_cache.get(predictor, case)
    supp_mask_fg = mask_on_feature_grid(case["support_mask"], supp_feat.shape[1:])
    supp_img_hw = case["supp"][0, 0]
    qry_img_hw = case["query"][0, 0]

    enc_before = predictor.audit_encoder_calls
    predictor.set_image(rgb_image(case))
    query_encoder_calls = predictor.audit_encoder_calls - enc_before
    assert query_encoder_calls == 1
    qry_feat = predictor.features[0].detach().cpu().numpy().copy()

    dec_before = predictor.audit_decoder_calls
    pool_cache: dict[tuple[int, ...], np.ndarray] = {}
    custom_cache: dict[bytes, np.ndarray] = {}

    def decode_pool(ids: list[int] | tuple[int, ...]) -> np.ndarray:
        key = tuple(int(i) for i in ids)
        if key not in pool_cache:
            mask, _, _, _ = predict(predictor, nr[list(key)], pr)
            pool_cache[key] = np.asarray(mask, dtype=bool)
        return pool_cache[key]

    full_raw = decode_pool(list(range(10)))
    full_cc = cc_mask(full_raw, pr)

    pos0_raw = decode_pool([])
    pos0_cc = cc_mask(pos0_raw, pr)

    dig = baseline_digest(
        {
            "full10_raw": full_raw,
            "full10_cc": full_cc,
            "positive0_raw": pos0_raw,
            "positive0_cc": pos0_cc,
        }
    )
    step8_mask_sha_exact = dig == expected_fp["masks"]

    near2_ids = [int(i) for i in proximity(nr, np.arange(10), 2, full_raw, far=False)]
    near2_raw = decode_pool(near2_ids)
    near2_cc = cc_mask(near2_raw, pr)

    e_local, l_note = compute_support_evidence(
        supp_feat,
        supp_mask_fg,
        qry_feat,
        full_raw.shape,
        mode="localring",
        support_image_hw=supp_img_hw,
    )

    def decode_placed(pts: np.ndarray | None) -> np.ndarray:
        if pts is None:
            return near2_raw
        arr = np.asarray(pts, dtype=np.float64)
        if len(arr) == 0:
            return pos0_raw
        key = arr.tobytes()
        if key not in custom_cache:
            m_raw, _, _, _ = predict(predictor, arr, pr)
            custom_cache[key] = np.asarray(m_raw, dtype=bool)
        return custom_cache[key]

    # Exact parity anchor matching Option A v2 pba_fixed_k2 (contour=full_raw, offset=5.0).
    pba_f10_pts, pba_f10_meta = place_pba_fixed_k2(full_raw, qry_img_hw, n_samples=2, offset_px=5.0)
    # Contour ablation matching pba_fixed_k2 except using connected positive0_cc contour.
    pba_pos0_pts, pba_pos0_meta = place_pba_fixed_k2(pos0_cc, qry_img_hw, n_samples=2, offset_px=5.0)

    sg_specs = {
        "semgate_f10_k2_r5": (full_cc, (5.0,), True),
        "semgate_pos0_k2_r5": (pos0_cc, (5.0,), True),
        "semgate_pos0_only_k2_r5": (pos0_cc, (5.0,), False),
        "semgate_pos0_raymarch_k2": (pos0_cc, (5.0, 8.0, 11.0, 14.0), True),
        "semgate_pos0_safe_k2": (pos0_cc, (8.0, 11.0, 14.0), True),
        "semgate_f10_safe_k2": (full_cc, (8.0, 11.0, 14.0), True),
    }

    raw_masks: dict[str, np.ndarray] = {
        "full10": full_raw,
        "positive0": pos0_raw,
        "boundary_near2": near2_raw,
        "pba_f10_int_k2": decode_placed(pba_f10_pts),
        "pba_pos0_int_k2": decode_placed(pba_pos0_pts),
    }
    sg_audits: dict[str, dict] = {}
    for arm, (cmask, offs, use_int) in sg_specs.items():
        pts, meta = place_semgate_pba(
            cmask,
            qry_img_hw,
            e_local,
            n_samples=2,
            offsets=offs,
            use_intensity=use_int,
        )
        raw_masks[arm] = decode_placed(pts)
        sg_audits[arm] = meta

    cc_masks: dict[str, np.ndarray] = {
        "full10": full_cc,
        "positive0": pos0_cc,
        "boundary_near2": near2_cc,
    }
    for arm in ARMS:
        if arm not in cc_masks:
            cc_masks[arm] = cc_mask(raw_masks[arm], pr)

    decoder_calls = predictor.audit_decoder_calls - dec_before
    audit_row = {
        "episode_id": int(case["episode_id"]),
        "fold": int(case["fold"]),
        "patient_id": str(case["patient_id"]),
        "organ": str(case["organ"]),
        "native_z": int(case["native_z"]),
        "compact_z": int(case["compact_z"]),
        "support_miss": bool(supp_miss),
        "query_encoder_calls": int(query_encoder_calls),
        "decoder_calls": int(decoder_calls),
        "local_evidence_note": l_note,
        "pba_f10_int_k2_fallback": pba_f10_meta.get("fallback"),
        "pba_pos0_int_k2_fallback": pba_pos0_meta.get("fallback"),
        "step8_coord_sha_exact": bool(step8_coord_sha_exact),
        "step8_mask_sha_exact": bool(step8_mask_sha_exact),
        "slice_seconds": float(time.perf_counter() - t0),
    }
    for arm, meta in sg_audits.items():
        audit_row[f"{arm}__k"] = int(meta.get("k_placed", 0))
        audit_row[f"{arm}__fallback"] = meta.get("fallback")
        audit_row[f"{arm}__valid_arc_frac"] = float(meta.get("valid_arc_frac", 0.0))
        audit_row[f"{arm}__mean_offset_px"] = float(meta.get("mean_offset_px", 0.0))

    return raw_masks, cc_masks, audit_row, {"neg": neg, "pos": pos}


def run_pilot(
    lock_dir: Path,
    data_dir: str,
    checkpoint_root: str,
    sam_checkpoint: str,
    output_dir: Path,
    n_gpus: int,
) -> dict:
    """Run Stage-0 intra-run determinism, Step 9 parity tolerance, and timing projection."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    c = read_config(lock_dir)
    seed_all(c["seed"])
    expected = json.loads((lock_dir / "step8_fingerprints.json").read_text(encoding="utf-8"))
    diag = pd.read_csv(lock_dir / "diagnostic_manifest.csv")
    manifest = read_manifest(lock_dir / "paired_episode_manifest.csv")

    predictor, sam_meta = tracked_sam(sam_checkpoint)
    support_cache = SupportFeatureCache()
    loads = []
    timings = []
    drift_records = []
    started = time.perf_counter()
    repeats = max(1, int(c.get("stage1_pilot_repeats", 1)))

    for fold in sorted(diag["fold"].unique()):
        model, meta = load_models(c, checkpoint_root, sam_checkpoint, int(fold))
        loads.append(meta)
        for row in diag[diag["fold"] == fold].to_dict("records"):
            ep = manifest[manifest["episode_id"] == row["episode_id"]].iloc[0].to_dict()
            case = next(v for v in episode_inputs(ep, data_dir, c) if v["native_z"] == row["native_z"])
            ref_path = lock_dir / "step9_pilot_ref" / f"ep{int(case['episode_id']):03d}_z{int(case['native_z'])}.npz"
            with np.load(ref_path) as rnpz:
                ref = {k: rnpz[k].copy() for k in rnpz.files}

            rep0 = None
            for rep in range(repeats + 1):
                raw_masks, cc_masks, audit, coords = run_stage1_slice(
                    model, predictor, support_cache, case, expected[f"{case['episode_id']}:{case['native_z']}"]
                )
                if rep == 0:
                    rep0 = {"coords": coords, "raw": raw_masks, "cc": cc_masks}
                else:
                    np.testing.assert_array_equal(coords["neg"], rep0["coords"]["neg"])
                    np.testing.assert_array_equal(coords["pos"], rep0["coords"]["pos"])
                    for arm in ARMS:
                        np.testing.assert_array_equal(raw_masks[arm], rep0["raw"][arm])
                        np.testing.assert_array_equal(cc_masks[arm], rep0["cc"][arm])

                if rep == 1:
                    max_neg_diff = float(
                        np.abs(coords["neg"].astype(np.float64) - ref["negative_source_xy"].astype(np.float64)).max()
                    )
                    max_pos_diff = float(
                        np.abs(coords["pos"].astype(np.float64) - ref["positive_source_xy"].astype(np.float64)).max()
                    )
                    d_full = _dice2d(cc_masks["full10"], ref["full10_cc"])
                    d_pos0 = _dice2d(cc_masks["positive0"], ref["positive0_cc"])
                    d_near2 = _dice2d(cc_masks["boundary_near2"], ref["boundary_near2_cc"])
                    assert max_neg_diff <= 0.25, (
                        f"Excessive neg coord drift ep={case['episode_id']} z={case['native_z']}: {max_neg_diff:.4f}"
                    )
                    assert min(d_full, d_pos0, d_near2) >= 0.985, (
                        f"Excessive mask drift vs Step 9 ep={case['episode_id']} z={case['native_z']}"
                    )
                    drift_records.append(
                        {
                            "episode_id": int(case["episode_id"]),
                            "native_z": int(case["native_z"]),
                            "max_neg_coord_diff_px": max_neg_diff,
                            "max_pos_coord_diff_px": max_pos_diff,
                            "full10_cc_dice_vs_step9": d_full,
                            "positive0_cc_dice_vs_step9": d_pos0,
                            "near2_cc_dice_vs_step9": d_near2,
                        }
                    )
                if rep > 0:
                    timings.append(
                        {
                            "episode_id": int(case["episode_id"]),
                            "native_z": int(case["native_z"]),
                            "repeat": int(rep),
                            "slice_seconds": float(audit["slice_seconds"]),
                            "decoder_calls": int(audit["decoder_calls"]),
                        }
                    )
                print(
                    f"pilot ep={case['episode_id']} z={case['native_z']} rep={rep} "
                    f"sec={audit['slice_seconds']:.4f} calls={audit['decoder_calls']} intra_exact={rep > 0}",
                    flush=True,
                )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    df_t = pd.DataFrame(timings)
    df_t.to_csv(output_dir / "timing_repeats.csv", index=False)
    df_d = pd.DataFrame(drift_records)
    df_d.to_csv(output_dir / "step9_drift_check.csv", index=False)

    p90_sec = float(df_t["slice_seconds"].quantile(0.90))
    overhead = float(c.get("projection_overhead_factor", 1.25))
    startup = float(c.get("projection_startup_seconds", 180.0))
    wall_proj = p90_sec * float(c["total_slices"]) / float(max(1, n_gpus)) * overhead + startup
    gpu_hours_proj = float(n_gpus) * wall_proj / 3600.0

    summary = {
        "hardware": torch.cuda.get_device_name(0),
        "n_gpus": int(n_gpus),
        "unique_slices": int(len(diag)),
        "repeats": int(repeats),
        "intra_run_exact_all_slices": True,
        "step9_cuda_tolerance_passed": True,
        "max_neg_coord_diff_px": float(df_d["max_neg_coord_diff_px"].max()),
        "max_pos_coord_diff_px": float(df_d["max_pos_coord_diff_px"].max()),
        "min_full10_cc_dice_vs_step9": float(df_d["full10_cc_dice_vs_step9"].min()),
        "min_positive0_cc_dice_vs_step9": float(df_d["positive0_cc_dice_vs_step9"].min()),
        "min_near2_cc_dice_vs_step9": float(df_d["near2_cc_dice_vs_step9"].min()),
        "p90_slice_seconds": p90_sec,
        "mean_decoder_calls": float(df_t["decoder_calls"].mean()),
        "projection_wall_seconds": float(wall_proj),
        "projected_total_gpu_hours": float(gpu_hours_proj),
        "pilot_stage_wall_seconds": float(time.perf_counter() - started),
        "loads": loads,
        "sam_load": sam_meta,
    }
    write_json(output_dir / "pilot_summary.json", summary)
    return summary


def _pack_episode_masks(raw_slices: dict[str, list[np.ndarray]], cc_slices: dict[str, list[np.ndarray]], path: Path) -> None:
    payload: dict[str, np.ndarray] = {}
    for arm in ARMS:
        r3 = np.stack(raw_slices[arm], axis=0).astype(bool)
        c3 = np.stack(cc_slices[arm], axis=0).astype(bool)
        payload[f"{arm}__raw_bits"] = np.packbits(r3.ravel())
        payload[f"{arm}__cc_bits"] = np.packbits(c3.ravel())
    z, h, w = np.stack(raw_slices[ARMS[0]], axis=0).shape
    payload["shape_zhw"] = np.asarray([z, h, w], dtype=np.int32)
    with path.open("wb") as f:
        np.savez_compressed(f, **payload)


def run_worker(
    lock_dir: Path,
    data_dir: str,
    checkpoint_root: str,
    sam_checkpoint: str,
    output_dir: Path,
    worker_id: int,
    num_workers: int,
    deadline_epoch: float,
) -> dict:
    """Run assigned episodes with exact 3D metrics and compact packed binary masks."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    c = read_config(lock_dir)
    seed_all(c["seed"])
    expected = json.loads((lock_dir / "step8_fingerprints.json").read_text(encoding="utf-8"))
    manifest = read_manifest(lock_dir / "paired_episode_manifest.csv")
    if num_workers == 2:
        assign = pd.read_csv(lock_dir / "worker_assignment.csv")
        ids = assign[assign["worker"] == worker_id]["episode_id"].tolist()
        manifest = manifest[manifest["episode_id"].isin(ids)]
    elif num_workers != 1:
        raise ValueError(f"Expected 1 or 2 workers, got {num_workers}")

    predictor, _ = tracked_sam(sam_checkpoint)
    support_cache = SupportFeatureCache()
    t_start = time.perf_counter()
    completed_eps = 0
    slice_count = 0

    for fold in sorted(manifest["fold"].unique()):
        model, _ = load_models(c, checkpoint_root, sam_checkpoint, int(fold))
        for row in manifest[manifest["fold"] == fold].to_dict("records"):
            eid = int(row["episode_id"])
            ep_dir = output_dir / f"episode_{eid:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = ep_dir / f"episode_{eid:03d}_metrics.csv"
            audit_path = ep_dir / f"episode_{eid:03d}_audit.json"
            packed_path = ep_dir / f"episode_{eid:03d}_packed.npz"
            if metrics_path.is_file() and audit_path.is_file() and packed_path.is_file():
                completed_eps += 1
                continue

            if time.time() > deadline_epoch - 60:
                write_json(output_dir / f"budget_stop_{time.time_ns()}.json", {"worker": worker_id, "episode_id": eid, "slices": slice_count})
                raise SystemExit(75)

            raw_slices: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
            cc_slices: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
            gt_slices: list[np.ndarray] = []
            other_slices: list[np.ndarray] = []
            slice_audits: list[dict] = []
            spacing_xyz = None

            for case in episode_inputs(row, data_dir, c):
                spacing_xyz = case["spacing_xyz"]
                fp = expected[f"{case['episode_id']}:{case['native_z']}"]
                r_m, c_m, s_aud, _ = run_stage1_slice(model, predictor, support_cache, case, fp)
                for arm in ARMS:
                    raw_slices[arm].append(r_m[arm])
                    cc_slices[arm].append(c_m[arm])
                gt_slices.append(np.asarray(case["gt"], dtype=bool))
                other_slices.append(np.asarray(case["other"], dtype=bool))
                slice_audits.append(s_aud)
                slice_count += 1

            gt_3d = np.stack(gt_slices, axis=0)
            other_3d = np.stack(other_slices, axis=0)
            raw_3d = {arm: np.stack(raw_slices[arm], axis=0) for arm in ARMS}
            cc_3d = {arm: np.stack(cc_slices[arm], axis=0) for arm in ARMS}

            ep_metrics = measure_episode_v2(
                episode_id=eid,
                fold=int(fold),
                patient_id=str(row["patient_id"]),
                organ=str(row["organ"]),
                spacing_xyz=spacing_xyz.tolist(),
                gt_3d=gt_3d,
                other_3d=other_3d,
                raw_by_arm=raw_3d,
                cc_by_arm=cc_3d,
            )
            _pack_episode_masks(raw_slices, cc_slices, packed_path)
            pd.DataFrame(ep_metrics).to_csv(metrics_path, index=False)
            audit_path.write_text(json.dumps(slice_audits), encoding="utf-8")
            completed_eps += 1

            if completed_eps % 5 == 0 or completed_eps == len(manifest):
                print(
                    f"worker={worker_id} fold={fold} ep={eid} completed={completed_eps}/{len(manifest)} "
                    f"slices={slice_count} elapsed={time.perf_counter() - t_start:.1f}s",
                    flush=True,
                )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "worker": int(worker_id),
        "episodes_completed": int(completed_eps),
        "slices_processed": int(slice_count),
        "support_encoder_calls": int(support_cache.encoder_calls),
        "wall_seconds": float(time.perf_counter() - t_start),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(output_dir / "worker_complete.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pilot", "worker"], required=True)
    p.add_argument("--lock-dir", type=Path, required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--sam-checkpoint", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-gpus", type=int, default=2)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--deadline-epoch", type=float, default=0.0)
    a = p.parse_args()

    if a.mode == "pilot":
        run_pilot(a.lock_dir, a.data_dir, a.checkpoint_root, a.sam_checkpoint, a.output, a.n_gpus)
    else:
        run_worker(
            a.lock_dir,
            a.data_dir,
            a.checkpoint_root,
            a.sam_checkpoint,
            a.output,
            a.worker,
            a.num_workers,
            a.deadline_epoch,
        )


if __name__ == "__main__":
    main()
