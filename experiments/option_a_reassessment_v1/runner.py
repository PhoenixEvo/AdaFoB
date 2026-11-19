"""Slice execution, Stage-0 parity pilot, and resumable per-episode GPU workers."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.controls_v1.core import baseline_digest, coordinate_digest, proximity
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
from analysis.gate_a.runtime import seed_all

from .metrics import measure_episode
from .selectors import (
    ARMS,
    compute_support_evidence,
    mask_on_feature_grid,
    place_pba_fixed_k2,
    select_local_contrast_near2,
    select_response_arms,
)


class SupportFeatureCache:
    """Cache 64x64 frozen SAM ViT-B support features keyed by (patient, native_z)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self.encoder_calls: int = 0

    def get(self, predictor, case: dict) -> tuple[np.ndarray, bool]:
        key = (str(case["support_patient"]), int(case["support_native_z"]))
        if key in self._cache:
            return self._cache[key], False
        predictor.set_image(rgb_image({"query": case["supp"]}))
        feat = predictor.features[0].detach().cpu().numpy().copy()
        self._cache[key] = feat
        self.encoder_calls += 1
        return feat, True


def _dice2d(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    s = int(aa.sum()) + int(bb.sum())
    if s == 0:
        return 1.0
    return float(2.0 * np.logical_and(aa, bb).sum() / s)


def run_option_a_slice(
    model,
    predictor,
    support_cache: SupportFeatureCache,
    case: dict,
    expected_fp: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict, dict[str, np.ndarray]]:
    """Execute all 9 Option A arms on one slice and record Step 8 byte-parity flags."""
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
    cache: dict[tuple[int, ...], np.ndarray] = {}

    def decode(ids: list[int] | tuple[int, ...]) -> np.ndarray:
        key = tuple(int(i) for i in ids)
        if key not in cache:
            mask, _, _, _ = predict(predictor, nr[list(key)], pr)
            cache[key] = np.asarray(mask, dtype=bool)
        return cache[key]

    full_ids = list(range(10))
    full_raw = decode(full_ids)
    full_cc = cc_mask(full_raw, pr)

    pos0_ids: list[int] = []
    pos0_raw = decode(pos0_ids)
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

    # Exact Step 9 distance-ascending order for boundary_near2.
    near2_ids = [int(i) for i in proximity(nr, np.arange(10), 2, full_raw, far=False)]
    near2_raw = decode(near2_ids)
    near2_cc = cc_mask(near2_raw, pr)

    deleted_masks = np.stack([decode([j for j in range(10) if j != i]) for i in range(10)], axis=0)

    e_global, g_note = compute_support_evidence(
        supp_feat, supp_mask_fg, qry_feat, full_raw.shape, mode="global"
    )
    e_local, l_note = compute_support_evidence(
        supp_feat,
        supp_mask_fg,
        qry_feat,
        full_raw.shape,
        mode="localring",
        support_image_hw=supp_img_hw,
    )

    resp = select_response_arms(
        full_raw=full_raw,
        deleted_masks=deleted_masks,
        e_global=e_global,
        e_local=e_local,
        fallback_ids=near2_ids,
        k_max_adaptive=6,
    )
    lc_ids, lc_meta = select_local_contrast_near2(nr, full_raw, qry_img_hw)
    pba_pts, pba_meta = place_pba_fixed_k2(full_raw, qry_img_hw, n_samples=2, offset_px=5.0)

    raw_masks: dict[str, np.ndarray] = {
        "full10": full_raw,
        "positive0": pos0_raw,
        "boundary_near2": near2_raw,
        "resp_magnitude_k2": decode(resp["resp_magnitude_k2"]),
        "resp_signed_global_k2": decode(resp["resp_signed_global_k2"]),
        "resp_signed_localring_k2": decode(resp["resp_signed_localring_k2"]),
        "resp_signed_adaptive": decode(resp["resp_signed_adaptive"]),
        "local_contrast_near2": decode(lc_ids),
    }
    if pba_pts is not None:
        pba_raw, _, _, _ = predict(predictor, pba_pts, pr)
        raw_masks["pba_fixed_k2"] = np.asarray(pba_raw, dtype=bool)
    else:
        raw_masks["pba_fixed_k2"] = near2_raw

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
        "global_evidence_note": g_note,
        "local_evidence_note": l_note,
        "boundary_near2_ids": near2_ids,
        "resp_magnitude_k2_ids": resp["resp_magnitude_k2"],
        "resp_signed_global_k2_ids": resp["resp_signed_global_k2"],
        "resp_signed_localring_k2_ids": resp["resp_signed_localring_k2"],
        "resp_signed_adaptive_ids": resp["resp_signed_adaptive"],
        "resp_signed_adaptive_k": int(len(resp["resp_signed_adaptive"])),
        "resp_signed_adaptive_note": resp["notes"]["resp_signed_adaptive"],
        "local_contrast_near2_ids": lc_ids,
        "local_contrast_fallback": lc_meta.get("fallback"),
        "pba_fixed_k2_fallback": pba_meta.get("fallback"),
        "pba_fixed_k2_points": pba_pts.tolist() if pba_pts is not None else None,
        "step8_coord_sha_exact": bool(step8_coord_sha_exact),
        "step8_mask_sha_exact": bool(step8_mask_sha_exact),
        "slice_seconds": float(time.perf_counter() - t0),
    }
    return raw_masks, cc_masks, audit_row, {"neg": neg, "pos": pos}


def run_pilot(
    lock_dir: Path,
    data_dir: str,
    checkpoint_root: str,
    sam_checkpoint: str,
    output_dir: Path,
    n_gpus: int,
) -> dict:
    """Run Stage-0 intra-run determinism, cross-CUDA tolerance vs Step 9, and timing projection."""
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
    repeats = max(1, int(c.get("option_a_pilot_repeats", 1)))

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
                raw_masks, cc_masks, audit, coords = run_option_a_slice(
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
                        f"Excessive neg coord drift ep={case['episode_id']} z={case['native_z']}: {max_neg_diff:.4f} px"
                    )
                    assert min(d_full, d_pos0, d_near2) >= 0.985, (
                        f"Excessive mask drift vs Step 9 ep={case['episode_id']} z={case['native_z']}: "
                        f"full10={d_full:.4f} pos0={d_pos0:.4f} near2={d_near2:.4f}"
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
                            "step8_coord_sha_exact": audit["step8_coord_sha_exact"],
                            "step8_mask_sha_exact": audit["step8_mask_sha_exact"],
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
    """Run assigned episodes, computing exact 3D metrics and compact packed masks."""
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
                r_m, c_m, s_aud, _ = run_option_a_slice(model, predictor, support_cache, case, fp)
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

            ep_metrics = measure_episode(
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
