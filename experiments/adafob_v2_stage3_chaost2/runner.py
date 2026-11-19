"""Slice execution, Stage-0 orientation/determinism pilot, and dual-backbone CHAOST2 workers."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.controls_v1.core import proximity, random_ids
from analysis.gate_a.runtime import ROOT, seed_all, sha256
from analysis.rebaseline_v1.common import cc_mask, generate, load_models, restore_xy, rgb_image, write_json
from experiments.adafob_v2_stage2.runner import _decode_placed_candidates
from experiments.option_a_reassessment_v1.metrics import json_safe
from experiments.option_a_reassessment_v1.runner import SupportFeatureCache, _dice2d

from .dataset import (
    ORGANS_CHAOST2,
    build_chaost2_manifest,
    episode_inputs_chaost2,
)
from .metrics import measure_episode_stage3
from .selectors import (
    RANDOM_SEEDS,
    RAW_ARMS,
    compute_support_evidence,
    mask_on_feature_grid,
    place_pba_fixed_k2,
    select_best_candidate,
    select_response_arms,
)


def load_tracked_sam(model_type: str, checkpoint: Path):
    """Strictly load frozen SAM ViT-B or ViT-H with call-counting forward pre-hooks."""
    import torch

    sa_root = str(ROOT / "third_party/segment-anything")
    if sa_root not in sys.path:
        sys.path.insert(0, sa_root)
    from segment_anything import SamPredictor, sam_model_registry

    if model_type not in ("vit_b", "vit_h"):
        raise ValueError(f"Unsupported SAM model_type: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    res = sam.load_state_dict(state, strict=True)
    predictor = SamPredictor(sam.cuda().eval())
    predictor.audit_decoder_calls = 0
    predictor.audit_encoder_calls = 0

    def _dec_hook(module, inputs):
        predictor.audit_decoder_calls += 1

    def _enc_hook(module, inputs):
        predictor.audit_encoder_calls += 1

    predictor.model.mask_decoder.register_forward_pre_hook(_dec_hook)
    predictor.model.image_encoder.register_forward_pre_hook(_enc_hook)
    meta = {
        "model_type": model_type,
        "checkpoint": str(checkpoint),
        "bytes": int(Path(checkpoint).stat().st_size),
        "sha256": sha256(checkpoint),
        "missing_keys": list(res.missing_keys),
        "unexpected_keys": list(res.unexpected_keys),
    }
    print(json.dumps({"sam_loaded": meta}), flush=True)
    return predictor, meta


def predict_sam(
    predictor,
    neg: np.ndarray,
    pos: np.ndarray,
    hw: tuple[int, int] = (256, 256),
    multimask_output: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Run SAM mask decoder on native (H, W) grid without hardcoded 257 shape."""
    import torch

    pts = np.concatenate([pos, neg])
    lbls = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    if not len(pts):
        return (
            np.zeros(hw, dtype=bool),
            np.zeros((1, *hw), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            0,
        )
    with torch.inference_mode():
        logits, quality, _ = predictor.predict(
            point_coords=pts,
            point_labels=lbls,
            multimask_output=multimask_output,
            return_logits=True,
        )
    if multimask_output:
        logits = logits[:1]
        quality = quality[:1]
    return logits[0] > predictor.model.mask_threshold, logits, quality, 1


def _load_fob_chaost2(ckpt_path: Path):
    return load_models({"fob_relative_paths": {"0": ckpt_path.name}, "fob_sha256": {"0": sha256(ckpt_path)}}, ckpt_path.parent, "", 0)


def _dice3d(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool)
    denom = 2 * int((p & g).sum()) + int((p & ~g).sum()) + int((~p & g).sum())
    return float((2.0 * int((p & g).sum()) / denom) if denom > 0 else 1.0)


def run_stage3_slice(
    model,
    predictor,
    support_cache: SupportFeatureCache,
    case: dict,
    expected_hw: tuple[int, int] = (256, 256),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict, dict[str, np.ndarray]]:
    """Execute all 12 RAW_ARMS on one CHAOST2 slice under matched Setting I protocol."""
    t0 = time.perf_counter()
    neg, pos = generate(model, case)
    assert neg.shape == (10, 2), f"Unexpected FoB neg shape {neg.shape}"
    nr = restore_xy(neg, (256, 256), expected_hw)
    pr = restore_xy(pos, (256, 256), expected_hw)

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
            mask, _, _, _ = predict_sam(predictor, nr[list(key)], pr, hw=expected_hw, multimask_output=False)
            pool_cache[key] = np.asarray(mask, dtype=bool)
        return pool_cache[key]

    def decode_placed(pts: np.ndarray) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float64)
        if len(arr) == 0:
            return pos0_raw
        key = arr.tobytes()
        if key not in custom_cache:
            m_raw, _, _, _ = predict_sam(predictor, arr, pr, hw=expected_hw, multimask_output=False)
            custom_cache[key] = np.asarray(m_raw, dtype=bool)
        return custom_cache[key]

    full_raw = decode_pool(list(range(10)))
    full_cc = cc_mask(full_raw, pr)

    pos0_raw = decode_pool([])
    pos0_cc = cc_mask(pos0_raw, pr)

    near2_ids = [int(i) for i in proximity(nr, np.arange(10), 2, full_raw, far=False)]
    near2_raw = decode_pool(near2_ids)
    near2_cc = cc_mask(near2_raw, pr)

    rand_raw: dict[str, np.ndarray] = {}
    for seed in RANDOM_SEEDS:
        rids = random_ids(range(10), 2, seed, int(case["episode_id"]), int(case["native_z"]))
        rand_raw[f"random2_s{seed}"] = decode_pool(rids)

    deleted_masks = np.stack([decode_pool([j for j in range(10) if j != i]) for i in range(10)], axis=0)
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
        e_global=None,
        e_local=e_local,
        fallback_ids=near2_ids,
        k_max_adaptive=6,
    )
    localring_ids = resp["resp_signed_localring_k2"]
    localring_raw = decode_pool(localring_ids)

    fixed_pts, fixed_meta = place_pba_fixed_k2(full_raw, qry_img_hw, n_samples=2, offset_px=5.0)
    fixed_raw, fixed_cc, fixed_k_map = _decode_placed_candidates(
        fixed_pts, decode_placed, pos0_raw, pos0_cc, near2_raw, near2_cc, pr
    )
    fixed_default_key = "c12" if "c12" in fixed_raw else ("c1" if "c1" in fixed_raw else "pos0")

    cbv_key, cbv_meta = select_best_candidate(fixed_cc, e_local, fallback_key=fixed_default_key)
    no0_pool = {k: v for k, v in fixed_cc.items() if k != "pos0"}
    if not no0_pool:
        no0_pool = {"pos0": pos0_cc}
    no0_key, no0_meta = select_best_candidate(no0_pool, e_local, fallback_key=fixed_default_key)

    raw_masks: dict[str, np.ndarray] = {
        "full10": full_raw,
        "positive0": pos0_raw,
        **rand_raw,
        "boundary_near2": near2_raw,
        "resp_signed_localring_k2": localring_raw,
        "pba_f10_int_k2": fixed_raw[fixed_default_key],
        "cbv_f10_fixed_supp": fixed_raw[cbv_key],
        "cbv_f10_no0_supp": fixed_raw[no0_key],
    }
    cc_masks: dict[str, np.ndarray] = {
        "full10": full_cc,
        "positive0": pos0_cc,
        "boundary_near2": near2_cc,
        "pba_f10_int_k2": fixed_cc[fixed_default_key],
        "cbv_f10_fixed_supp": fixed_cc[cbv_key],
        "cbv_f10_no0_supp": fixed_cc[no0_key],
    }
    for arm in RAW_ARMS:
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
        "pba_f10_int_k2_fallback": fixed_meta.get("fallback"),
        "cbv_f10_fixed_supp__choice": str(cbv_key),
        "cbv_f10_fixed_supp__k": int(fixed_k_map[cbv_key]),
        "cbv_f10_fixed_supp__fallback": cbv_meta.get("fallback"),
        "cbv_f10_no0_supp__choice": str(no0_key),
        "cbv_f10_no0_supp__k": int(fixed_k_map[no0_key]),
        "cbv_f10_no0_supp__fallback": no0_meta.get("fallback"),
        "slice_seconds": float(time.perf_counter() - t0),
    }
    return raw_masks, cc_masks, audit_row, {"neg": neg, "pos": pos}


def run_orientation_and_determinism_pilot(
    data_dir: Path,
    ckpt_paths: dict[int, Path],
    sam_b_ckpt: Path,
    output_dir: Path,
    n_gpus: int,
    seed: int = 42,
) -> dict:
    """Test identity vs flip_y on Fold 0 (16 eps) under R1/multimask, then check 5-fold determinism."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    manifest = build_chaost2_manifest()
    fold0 = manifest[manifest["fold"] == 0].to_dict("records")

    predictor, sam_b_meta = load_tracked_sam("vit_b", sam_b_ckpt)
    model_f0, f0_meta = _load_fob_chaost2(ckpt_paths[0])

    ori_rows: list[dict] = []
    started = time.perf_counter()
    for orientation in ("identity", "flip_y"):
        for mm in (False, True):
            ep_records = []
            for row in fold0:
                preds_full_cc = []
                preds_pos0_cc = []
                preds_full_raw = []
                gts = []
                for case in episode_inputs_chaost2(row, data_dir, orientation=orientation):
                    neg, pos = generate(model_f0, case)
                    nr = restore_xy(neg, (256, 256), (256, 256))
                    pr = restore_xy(pos, (256, 256), (256, 256))
                    predictor.set_image(rgb_image(case))
                    r_f10, _, _, _ = predict_sam(predictor, nr, pr, hw=(256, 256), multimask_output=mm)
                    r_p0, _, _, _ = predict_sam(predictor, nr[:0], pr, hw=(256, 256), multimask_output=mm)
                    preds_full_raw.append(np.asarray(r_f10, dtype=bool))
                    preds_full_cc.append(cc_mask(np.asarray(r_f10, dtype=bool), pr))
                    preds_pos0_cc.append(cc_mask(np.asarray(r_p0, dtype=bool), pr))
                    gts.append(np.asarray(case["gt"], dtype=bool))
                gt_3d = np.stack(gts, axis=0)
                ep_records.append(
                    {
                        "patient_id": row["patient_id"],
                        "organ": row["organ"],
                        "full10_raw_dice": _dice3d(np.stack(preds_full_raw, axis=0), gt_3d),
                        "full10_cc_dice": _dice3d(np.stack(preds_full_cc, axis=0), gt_3d),
                        "pos0_cc_dice": _dice3d(np.stack(preds_pos0_cc, axis=0), gt_3d),
                    }
                )
            df_o = pd.DataFrame(ep_records)
            pat_f10 = float(df_o.groupby(["patient_id", "organ"])["full10_cc_dice"].mean().groupby("patient_id").mean().mean() * 100.0)
            pat_f10_raw = float(df_o.groupby(["patient_id", "organ"])["full10_raw_dice"].mean().groupby("patient_id").mean().mean() * 100.0)
            pat_pos0 = float(df_o.groupby(["patient_id", "organ"])["pos0_cc_dice"].mean().groupby("patient_id").mean().mean() * 100.0)
            rec = {
                "orientation": orientation,
                "multimask_output": bool(mm),
                "fold0_episodes": len(df_o),
                "full10_pat_raw_dice_pct": pat_f10_raw,
                "full10_pat_cc_dice_pct": pat_f10,
                "positive0_pat_cc_dice_pct": pat_pos0,
                "by_organ_full10_cc_pct": {
                    k: float(v * 100.0) for k, v in df_o.groupby("organ")["full10_cc_dice"].mean().to_dict().items()
                },
            }
            ori_rows.append(rec)
            print(json.dumps({"orientation_check": rec}), flush=True)

    del model_f0
    gc.collect()
    torch.cuda.empty_cache()

    r1_id = next(r for r in ori_rows if r["orientation"] == "identity" and not r["multimask_output"])
    r1_fy = next(r for r in ori_rows if r["orientation"] == "flip_y" and not r["multimask_output"])
    best_orientation = "identity" if r1_id["full10_pat_cc_dice_pct"] >= r1_fy["full10_pat_cc_dice_pct"] else "flip_y"
    best_dice = max(r1_id["full10_pat_cc_dice_pct"], r1_fy["full10_pat_cc_dice_pct"])
    assert best_dice >= 50.0, f"Orientation gate failed: both orientations below 50% ({ori_rows})"

    # Run 2 exact determinism passes on 5 representative slices (1 per fold) using best_orientation.
    support_cache = SupportFeatureCache()
    timings: list[dict] = []
    for fold in range(5):
        model, _ = _load_fob_chaost2(ckpt_paths[fold])
        ep_row = manifest[manifest["fold"] == fold].iloc[0].to_dict()
        slices = list(episode_inputs_chaost2(ep_row, data_dir, orientation=best_orientation))
        case = slices[len(slices) // 2]

        rep0 = None
        for rep in range(2):
            raw_m, cc_m, aud, coords = run_stage3_slice(model, predictor, support_cache, case, expected_hw=(256, 256))
            if rep == 0:
                rep0 = {"coords": coords, "raw": raw_m, "cc": cc_m}
            else:
                np.testing.assert_array_equal(coords["neg"], rep0["coords"]["neg"])
                np.testing.assert_array_equal(coords["pos"], rep0["coords"]["pos"])
                for arm in RAW_ARMS:
                    np.testing.assert_array_equal(raw_m[arm], rep0["raw"][arm])
                    np.testing.assert_array_equal(cc_m[arm], rep0["cc"][arm])
                timings.append(
                    {
                        "fold": fold,
                        "episode_id": int(case["episode_id"]),
                        "slice_seconds_vit_b": float(aud["slice_seconds"]),
                        "decoder_calls": int(aud["decoder_calls"]),
                    }
                )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    del predictor
    gc.collect()
    torch.cuda.empty_cache()

    p90_b = float(pd.DataFrame(timings)["slice_seconds_vit_b"].quantile(0.90))
    total_slices = 1354
    wall_b = p90_b * total_slices / float(max(1, n_gpus)) * 1.25 + 60.0
    # Conservatively project ViT-H at 2.5x ViT-B slice wall time (heavier image encoder).
    wall_h = (p90_b * 2.5) * total_slices / float(max(1, n_gpus)) * 1.25 + 60.0
    gpu_hours_proj = float(n_gpus) * (wall_b + wall_h + (time.perf_counter() - started)) / 3600.0

    gate = {
        "best_orientation": best_orientation,
        "best_fold0_r1_full10_pat_cc_dice_pct": float(best_dice),
        "orientation_comparisons": ori_rows,
        "intra_run_exact_5folds": True,
        "vit_b_p90_slice_seconds": p90_b,
        "projected_total_gpu_hours_both_backbones": float(gpu_hours_proj),
        "pilot_wall_seconds": float(time.perf_counter() - started),
        "sam_b_load": sam_b_meta,
        "fold0_fob_load": f0_meta,
    }
    write_json(output_dir / "orientation_gate.json", gate)
    return gate


def run_worker(
    backbone: str,
    orientation: str,
    data_dir: Path,
    ckpt_paths: dict[int, Path],
    sam_checkpoint: Path,
    output_dir: Path,
    worker_id: int,
    num_workers: int,
    deadline_epoch: float,
    seed: int = 42,
) -> dict:
    """Run sharded CHAOST2 episodes for a given frozen SAM backbone (vit_b or vit_h)."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    manifest = build_chaost2_manifest()
    if num_workers == 2:
        manifest = manifest[(manifest["episode_id"] % 2) == worker_id].copy()
    elif num_workers != 1:
        raise ValueError(f"Expected 1 or 2 workers, got {num_workers}")

    predictor, sam_meta = load_tracked_sam(backbone, sam_checkpoint)
    support_cache = SupportFeatureCache()
    t_start = time.perf_counter()
    completed_eps = 0
    slice_count = 0

    for fold in sorted(manifest["fold"].unique()):
        model, _ = _load_fob_chaost2(ckpt_paths[int(fold)])
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

            if deadline_epoch > 0 and time.time() > deadline_epoch - 60:
                write_json(
                    output_dir / f"budget_stop_{time.time_ns()}.json",
                    {"backbone": backbone, "worker": worker_id, "episode_id": eid, "slices": slice_count},
                )
                raise SystemExit(75)

            raw_slices: dict[str, list[np.ndarray]] = {arm: [] for arm in RAW_ARMS}
            cc_slices: dict[str, list[np.ndarray]] = {arm: [] for arm in RAW_ARMS}
            gt_slices: list[np.ndarray] = []
            other_slices: list[np.ndarray] = []
            slice_audits: list[dict] = []
            spacing_xyz = None

            for case in episode_inputs_chaost2(row, data_dir, orientation=orientation):
                spacing_xyz = case["spacing_xyz"]
                r_m, c_m, s_aud, _ = run_stage3_slice(model, predictor, support_cache, case, expected_hw=(256, 256))
                s_aud["backbone"] = backbone
                s_aud["orientation"] = orientation
                for arm in RAW_ARMS:
                    raw_slices[arm].append(r_m[arm])
                    cc_slices[arm].append(c_m[arm])
                gt_slices.append(np.asarray(case["gt"], dtype=bool))
                other_slices.append(np.asarray(case["other"], dtype=bool))
                slice_audits.append(s_aud)
                slice_count += 1

            gt_3d = np.stack(gt_slices, axis=0)
            other_3d = np.stack(other_slices, axis=0)
            raw_3d = {arm: np.stack(raw_slices[arm], axis=0) for arm in RAW_ARMS}
            cc_3d = {arm: np.stack(cc_slices[arm], axis=0) for arm in RAW_ARMS}

            ep_metrics = measure_episode_stage3(
                backbone=backbone,
                orientation=orientation,
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

            payload: dict[str, np.ndarray] = {}
            for arm in RAW_ARMS:
                r3 = np.stack(raw_slices[arm], axis=0).astype(bool)
                c3 = np.stack(cc_slices[arm], axis=0).astype(bool)
                payload[f"{arm}__raw_bits"] = np.packbits(r3.ravel())
                payload[f"{arm}__cc_bits"] = np.packbits(c3.ravel())
            z, h, w = np.stack(raw_slices[RAW_ARMS[0]], axis=0).shape
            payload["shape_zhw"] = np.asarray([z, h, w], dtype=np.int32)
            with packed_path.open("wb") as f:
                np.savez_compressed(f, **payload)

            pd.DataFrame(ep_metrics).to_csv(metrics_path, index=False)
            audit_path.write_text(json.dumps(slice_audits), encoding="utf-8")
            completed_eps += 1

            if completed_eps % 5 == 0 or completed_eps == len(manifest):
                print(
                    f"backbone={backbone} worker={worker_id} fold={fold} ep={eid} "
                    f"completed={completed_eps}/{len(manifest)} slices={slice_count} elapsed={time.perf_counter() - t_start:.1f}s",
                    flush=True,
                )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "backbone": backbone,
        "orientation": orientation,
        "worker": int(worker_id),
        "episodes_completed": int(completed_eps),
        "slices_processed": int(slice_count),
        "support_encoder_calls": int(support_cache.encoder_calls),
        "wall_seconds": float(time.perf_counter() - t_start),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "sam_meta": sam_meta,
    }
    write_json(output_dir / "worker_complete.json", summary)
    return json_safe(summary)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pilot", "worker"], required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--checkpoint-map", type=Path, required=True)
    p.add_argument("--sam-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backbone", choices=["vit_b", "vit_h"], default="vit_b")
    p.add_argument("--orientation", choices=["identity", "flip_y"], default="identity")
    p.add_argument("--n-gpus", type=int, default=2)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--deadline-epoch", type=float, default=0.0)
    a = p.parse_args()

    ckpt_map_raw = json.loads(a.checkpoint_map.read_text(encoding="utf-8"))
    ckpt_paths = {int(k): Path(v) for k, v in ckpt_map_raw.items()}
    if a.mode == "pilot":
        run_orientation_and_determinism_pilot(a.data_dir, ckpt_paths, a.sam_checkpoint, a.output, a.n_gpus)
    else:
        run_worker(
            backbone=a.backbone,
            orientation=a.orientation,
            data_dir=a.data_dir,
            ckpt_paths=ckpt_paths,
            sam_checkpoint=a.sam_checkpoint,
            output_dir=a.output,
            worker_id=a.worker,
            num_workers=a.num_workers,
            deadline_epoch=a.deadline_epoch,
        )


if __name__ == "__main__":
    main()
