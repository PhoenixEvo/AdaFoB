"""Unified slice execution, LPT workload sharding, determinism pilot, and dual-backbone workers."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.controls_v1.core import baseline_digest, coordinate_digest, proximity, random_ids
from analysis.gate_a.runtime import seed_all, sha256
from analysis.rebaseline_v1.common import (
    ORGANS as ORGANS_SABS,
    cc_mask,
    episode_inputs as episode_inputs_sabs,
    label_volume,
    load_models,
    nifti_path,
    read_config,
    read_manifest as read_sabs_manifest,
    restore_xy,
    rgb_image,
    write_json,
)
from experiments.adafob_v2_stage3_chaost2.dataset import (
    ORGANS_CHAOST2,
    build_chaost2_manifest,
    episode_inputs_chaost2,
    load_chaost2_volume,
)
from experiments.adafob_v2_stage3_chaost2.runner import load_tracked_sam, predict_sam
from experiments.option_a_reassessment_v1.metrics import json_safe
from experiments.option_a_reassessment_v1.runner import SupportFeatureCache, _dice2d
from experiments.option_a_reassessment_v1.selectors import compute_support_evidence

from .fob_hook import generate_with_fob_prior
from .metrics import measure_episode_stage4
from .selectors import (
    CBV_EVIDENCE_FAMILIES,
    RANDOM_SEEDS,
    RAW_ARMS,
    compute_dualbg_evidence,
    compute_fob_prob_evidence,
    compute_joint_evidence,
    mask_on_feature_grid,
    place_pba_fixed_k2,
    select_best_candidate_logit_weighted,
)
from experiments.adafob_v2_stage2.selectors import select_best_candidate


def count_episode_slices(
    manifest: pd.DataFrame,
    dataset: str,
    data_dir: Path,
    orientation: str = "identity",
) -> dict[int, int]:
    """Count target-positive slices per episode_id for balanced multi-GPU sharding."""
    counts: dict[int, int] = {}
    for row in manifest.to_dict("records"):
        eid = int(row["episode_id"])
        pid = str(row["patient_id"])
        organ = str(row["organ"])
        lstem = pid.replace("image_", "label_")
        if dataset == "chaost2":
            lbl, _ = load_chaost2_volume(data_dir, lstem, is_label=True, orientation=orientation)
            lid = int(ORGANS_CHAOST2[organ])
        elif dataset == "sabs":
            lbl, _ = label_volume(str(nifti_path(data_dir, lstem)))
            lid = int(ORGANS_SABS[organ])
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        counts[eid] = int((lbl == lid).any(axis=(1, 2)).sum())
    return counts


def shard_manifest_lpt(
    manifest: pd.DataFrame,
    slice_counts_by_ep: dict[int, int],
    worker_id: int,
    num_workers: int,
) -> pd.DataFrame:
    """Assign episodes to workers using deterministic Longest Processing Time (LPT) scheduling."""
    if num_workers < 1:
        raise ValueError(f"Expected num_workers >= 1, got {num_workers}")
    if not (0 <= worker_id < num_workers):
        raise ValueError(f"worker_id={worker_id} out of range for num_workers={num_workers}")
    if num_workers == 1:
        return manifest.sort_values(["fold", "episode_id"]).reset_index(drop=True)

    ordered_eids = sorted(
        (int(eid) for eid in manifest["episode_id"].tolist()),
        key=lambda eid: (-int(slice_counts_by_ep[eid]), int(eid)),
    )
    loads = [0] * num_workers
    assigned: set[int] = set()
    for eid in ordered_eids:
        target_w = min(range(num_workers), key=lambda w: (loads[w], w))
        loads[target_w] += int(slice_counts_by_ep[eid])
        if target_w == worker_id:
            assigned.add(eid)
    return (
        manifest[manifest["episode_id"].isin(assigned)]
        .sort_values(["fold", "episode_id"])
        .reset_index(drop=True)
    )


def _decode_placed_candidates_with_logits(
    pts: np.ndarray | None,
    decode_placed,
    pos0_raw: np.ndarray,
    pos0_cc: np.ndarray,
    pos0_logits: np.ndarray,
    near2_raw: np.ndarray,
    near2_cc: np.ndarray,
    near2_logits: np.ndarray,
    pr: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    """Decode {pos0, c1, c2, c12} raw, CC, and SAM logit maps with exact fallback."""
    raws: dict[str, np.ndarray] = {"pos0": pos0_raw}
    ccs: dict[str, np.ndarray] = {"pos0": pos0_cc}
    logits_map: dict[str, np.ndarray] = {"pos0": pos0_logits}
    k_map: dict[str, int] = {"pos0": 0}

    if pts is None:
        raws["c12"] = near2_raw
        ccs["c12"] = near2_cc
        logits_map["c12"] = near2_logits
        k_map["c12"] = 2
        return raws, ccs, logits_map, k_map

    arr = np.asarray(pts, dtype=np.float64)
    if len(arr) == 0:
        return raws, ccs, logits_map, k_map

    if len(arr) == 1:
        r1, l1 = decode_placed(arr[:1])
        raws["c1"] = r1
        ccs["c1"] = cc_mask(r1, pr)
        logits_map["c1"] = l1
        k_map["c1"] = 1
        return raws, ccs, logits_map, k_map

    r1, l1 = decode_placed(arr[:1])
    r2, l2 = decode_placed(arr[1:2])
    r12, l12 = decode_placed(arr[:2])
    raws.update({"c1": r1, "c2": r2, "c12": r12})
    ccs.update({"c1": cc_mask(r1, pr), "c2": cc_mask(r2, pr), "c12": cc_mask(r12, pr)})
    logits_map.update({"c1": l1, "c2": l2, "c12": l12})
    k_map.update({"c1": 1, "c2": 1, "c12": 2})
    return raws, ccs, logits_map, k_map


def _load_fob_fold_ckpt(ckpt_path: Path):
    return load_models({"fob_relative_paths": {"0": ckpt_path.name}, "fob_sha256": {"0": sha256(ckpt_path)}}, ckpt_path.parent, "", 0)


def _get_dataset_context(
    dataset: str,
    data_dir: Path,
    orientation: str,
    lock_dir: Path | None,
):
    if dataset == "chaost2":
        manifest = build_chaost2_manifest()
        native_hw = (256, 256)

        def iter_cases(row: dict):
            return episode_inputs_chaost2(row, data_dir, orientation=orientation)

        return manifest, native_hw, iter_cases
    if dataset == "sabs":
        assert lock_dir is not None, "--lock-dir is required for dataset=sabs"
        cfg = read_config(lock_dir)
        manifest = read_sabs_manifest(lock_dir / "paired_episode_manifest.csv")
        native_hw = (257, 257)

        def iter_cases(row: dict):
            return episode_inputs_sabs(row, str(data_dir), cfg)

        return manifest, native_hw, iter_cases
    raise ValueError(f"Unsupported dataset: {dataset}")


def run_stage4_slice(
    model,
    predictor,
    support_cache: SupportFeatureCache,
    case: dict,
    expected_hw: tuple[int, int],
    expected_fp: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict, dict[str, np.ndarray]]:
    """Execute all 22 RAW_ARMS cleanly on one slice while caching embeddings & logits."""
    t0 = time.perf_counter()
    neg, pos, p_fob_hw, fob_supp_feat, fob_qry_feat = generate_with_fob_prior(model, case, expected_hw)
    assert neg.shape == (10, 2), f"Unexpected FoB neg shape {neg.shape}"
    nr = restore_xy(neg, (256, 256), expected_hw)
    pr = restore_xy(pos, (256, 256), expected_hw)
    coord_sha_exact = (
        bool(coordinate_digest(neg, pos, nr, pr) == expected_fp["coordinates"])
        if expected_fp is not None
        else None
    )

    fob_prior_raw = np.asarray(p_fob_hw >= 0.5, dtype=bool)
    fob_prior_cc = cc_mask(fob_prior_raw, pr)

    supp_feat, supp_miss = support_cache.get(predictor, case)
    sam_supp_fg = mask_on_feature_grid(case["support_mask"], supp_feat.shape[1:])
    fob_supp_fg = mask_on_feature_grid(case["support_mask"], fob_supp_feat.shape[1:])
    supp_img_hw = case["supp"][0, 0]
    qry_img_hw = case["query"][0, 0]

    enc_before = predictor.audit_encoder_calls
    predictor.set_image(rgb_image(case))
    query_encoder_calls = predictor.audit_encoder_calls - enc_before
    assert query_encoder_calls == 1
    qry_feat = predictor.features[0].detach().cpu().numpy().copy()

    dec_before = predictor.audit_decoder_calls
    pool_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    custom_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def decode_pool(ids: list[int] | tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(int(i) for i in ids)
        if key not in pool_cache:
            m_raw, logits, _, _ = predict_sam(predictor, nr[list(key)], pr, hw=expected_hw, multimask_output=False)
            pool_cache[key] = (np.asarray(m_raw, dtype=bool), np.asarray(logits[0], dtype=np.float32))
        return pool_cache[key]

    def decode_placed(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(pts, dtype=np.float64)
        if len(arr) == 0:
            return pos0_raw, pos0_logits
        key = arr.tobytes()
        if key not in custom_cache:
            m_raw, logits, _, _ = predict_sam(predictor, arr, pr, hw=expected_hw, multimask_output=False)
            custom_cache[key] = (np.asarray(m_raw, dtype=bool), np.asarray(logits[0], dtype=np.float32))
        return custom_cache[key]

    full_raw, _ = decode_pool(list(range(10)))
    full_cc = cc_mask(full_raw, pr)

    pos0_raw, pos0_logits = decode_pool([])
    pos0_cc = cc_mask(pos0_raw, pr)

    mask_sha_exact = None
    if expected_fp is not None:
        dig = baseline_digest(
            {
                "full10_raw": full_raw,
                "full10_cc": full_cc,
                "positive0_raw": pos0_raw,
                "positive0_cc": pos0_cc,
            }
        )
        mask_sha_exact = bool(dig == expected_fp["masks"])

    near2_ids = [int(i) for i in proximity(nr, np.arange(10), 2, full_raw, far=False)]
    near2_raw, near2_logits = decode_pool(near2_ids)
    near2_cc = cc_mask(near2_raw, pr)

    rand_raw: dict[str, np.ndarray] = {}
    for seed in RANDOM_SEEDS:
        rids = random_ids(range(10), 2, seed, int(case["episode_id"]), int(case["native_z"]))
        rmask, _ = decode_pool(rids)
        rand_raw[f"random2_s{seed}"] = rmask

    fixed_pts, fixed_meta = place_pba_fixed_k2(full_raw, qry_img_hw, n_samples=2, offset_px=5.0)
    fixed_raw, fixed_cc, fixed_logits, fixed_k_map = _decode_placed_candidates_with_logits(
        fixed_pts, decode_placed, pos0_raw, pos0_cc, pos0_logits, near2_raw, near2_cc, near2_logits, pr
    )
    fixed_default_key = "c12" if "c12" in fixed_raw else ("c1" if "c1" in fixed_raw else "pos0")
    no0_cc = {k: v for k, v in fixed_cc.items() if k != "pos0"}
    no0_logits = {k: v for k, v in fixed_logits.items() if k != "pos0"}
    if not no0_cc:
        no0_cc = {"pos0": pos0_cc}
        no0_logits = {"pos0": pos0_logits}

    e_sam_localring, l_note = compute_support_evidence(
        supp_feat, sam_supp_fg, qry_feat, full_raw.shape, mode="localring", support_image_hw=supp_img_hw
    )
    e_sam_dualbg, s_dual_note = compute_dualbg_evidence(
        supp_feat, sam_supp_fg, qry_feat, full_raw.shape, support_image_hw=supp_img_hw
    )
    if float(case["supp"].max()) == 0.0 or float(case["support_mask"].max()) == 0.0:
        e_fob_prob = None
        e_fob_dualbg, f_dual_note = None, "empty_support"
    else:
        e_fob_prob = compute_fob_prob_evidence(p_fob_hw)
        e_fob_dualbg, f_dual_note = compute_dualbg_evidence(
            fob_supp_feat, fob_supp_fg, fob_qry_feat, full_raw.shape, support_image_hw=supp_img_hw
        )
    e_joint_prob_sam = compute_joint_evidence(e_fob_prob, e_sam_dualbg)

    evidence_map = {
        "sam_localring": e_sam_localring,
        "sam_dualbg": e_sam_dualbg,
        "fob_prob": e_fob_prob,
        "fob_dualbg": e_fob_dualbg,
        "joint_prob_sam": e_joint_prob_sam,
    }

    raw_masks: dict[str, np.ndarray] = {
        "fob_coarse_prior": fob_prior_raw,
        "full10": full_raw,
        "positive0": pos0_raw,
        **rand_raw,
        "boundary_near2": near2_raw,
        "pba_f10_int_k2": fixed_raw[fixed_default_key],
    }
    cc_masks: dict[str, np.ndarray] = {
        "fob_coarse_prior": fob_prior_cc,
        "full10": full_cc,
        "positive0": pos0_cc,
        "boundary_near2": near2_cc,
        "pba_f10_int_k2": fixed_cc[fixed_default_key],
    }

    cbv_audit: dict[str, dict] = {}
    for fam in CBV_EVIDENCE_FAMILIES:
        for subset_mode, pool_cc, pool_logits in (("all", fixed_cc, fixed_logits), ("no0", no0_cc, no0_logits)):
            arm = f"cbv_{fam}_{subset_mode}"
            if fam == "logit_fob_prob":
                chosen, meta = select_best_candidate_logit_weighted(
                    pool_cc, pool_logits, e_fob_prob, fallback_key=fixed_default_key
                )
            else:
                chosen, meta = select_best_candidate(
                    pool_cc, evidence_map[fam], fallback_key=fixed_default_key
                )
            raw_masks[arm] = fixed_raw[chosen]
            cc_masks[arm] = fixed_cc[chosen]
            cbv_audit[arm] = {
                "choice": str(chosen),
                "k": int(fixed_k_map[chosen]),
                "fallback": meta.get("fallback"),
            }

    for arm in RAW_ARMS:
        if arm not in cc_masks:
            cc_masks[arm] = cc_mask(raw_masks[arm], pr)

    decoder_calls = predictor.audit_decoder_calls - dec_before
    audit_row: dict = {
        "episode_id": int(case["episode_id"]),
        "fold": int(case["fold"]),
        "patient_id": str(case["patient_id"]),
        "organ": str(case["organ"]),
        "native_z": int(case["native_z"]),
        "compact_z": int(case["compact_z"]),
        "support_miss": bool(supp_miss),
        "query_encoder_calls": int(query_encoder_calls),
        "decoder_calls": int(decoder_calls),
        "sam_localring_note": l_note,
        "sam_dualbg_note": s_dual_note,
        "fob_dualbg_note": f_dual_note,
        "fob_prior_mean": float(np.mean(p_fob_hw)),
        "fob_prior_fg_frac": float(np.mean(fob_prior_cc)),
        "pba_f10_int_k2_fallback": fixed_meta.get("fallback"),
        "step8_coord_sha_exact": coord_sha_exact,
        "step8_mask_sha_exact": mask_sha_exact,
        "pos0_area": int(pos0_cc.sum()),
        "c12_area": int(fixed_cc[fixed_default_key].sum()),
        "slice_seconds": float(time.perf_counter() - t0),
    }
    for arm, info in cbv_audit.items():
        audit_row[f"{arm}__choice"] = info["choice"]
        audit_row[f"{arm}__k"] = info["k"]
        audit_row[f"{arm}__fallback"] = info["fallback"]

    return raw_masks, cc_masks, audit_row, {"neg": neg, "pos": pos, "p_fob_hw": p_fob_hw}


def run_pilot(
    dataset: str,
    orientation: str,
    lock_dir: Path | None,
    data_dir: Path,
    ckpt_paths: dict[int, Path],
    sam_b_ckpt: Path,
    output_dir: Path,
    n_gpus: int,
    seed: int = 42,
) -> dict:
    """Verify intra-run exact determinism, FoB prior sanity, and wall-time projection."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    manifest, expected_hw, iter_cases = _get_dataset_context(dataset, data_dir, orientation, lock_dir)

    expected_fp_map = None
    step9_ref_map: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    if dataset == "sabs":
        assert lock_dir is not None
        expected_fp_map = json.loads((lock_dir / "step8_fingerprints.json").read_text(encoding="utf-8"))
        diag = pd.read_csv(lock_dir / "diagnostic_manifest.csv")
        pilot_rows = []
        for fold in sorted(diag["fold"].unique()):
            row_d = diag[diag["fold"] == fold].iloc[0].to_dict()
            ep_row = manifest[manifest["episode_id"] == int(row_d["episode_id"])].iloc[0].to_dict()
            ep_row["_target_z"] = int(row_d["native_z"])
            pilot_rows.append(ep_row)
            ref_p = lock_dir / "step9_pilot_ref" / f"ep{int(row_d['episode_id']):03d}_z{int(row_d['native_z'])}.npz"
            with np.load(ref_p) as npz:
                step9_ref_map[(int(row_d["episode_id"]), int(row_d["native_z"]))] = {k: npz[k].copy() for k in npz.files}
    else:
        pilot_rows = [manifest[manifest["fold"] == fold].iloc[0].to_dict() for fold in range(5)]

    predictor, sam_b_meta = load_tracked_sam("vit_b", sam_b_ckpt)
    support_cache = SupportFeatureCache()
    timings: list[dict] = []
    drift_records: list[dict] = []
    started = time.perf_counter()

    for ep_row in pilot_rows:
        fold = int(ep_row["fold"])
        model, _ = _load_fob_fold_ckpt(ckpt_paths[fold])
        slices = list(iter_cases(ep_row))
        if "_target_z" in ep_row:
            case = next(c for c in slices if int(c["native_z"]) == int(ep_row["_target_z"]))
        else:
            case = slices[len(slices) // 2]

        fp_key = f"{int(case['episode_id'])}:{int(case['native_z'])}"
        expected_fp = expected_fp_map[fp_key] if expected_fp_map is not None else None

        rep0 = None
        for rep in range(2):
            raw_m, cc_m, aud, coords = run_stage4_slice(
                model, predictor, support_cache, case, expected_hw=expected_hw, expected_fp=expected_fp
            )
            p_fob = coords["p_fob_hw"]
            assert np.isfinite(p_fob).all() and float(p_fob.min()) >= -1e-6 and float(p_fob.max()) <= 1.0 + 1e-6
            if rep == 0:
                rep0 = {"coords": coords, "raw": raw_m, "cc": cc_m}
            else:
                np.testing.assert_array_equal(coords["neg"], rep0["coords"]["neg"])
                np.testing.assert_array_equal(coords["pos"], rep0["coords"]["pos"])
                np.testing.assert_array_equal(coords["p_fob_hw"], rep0["coords"]["p_fob_hw"])
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
                ref = step9_ref_map.get((int(case["episode_id"]), int(case["native_z"])))
                if ref is not None:
                    max_neg_diff = float(
                        np.abs(coords["neg"].astype(np.float64) - ref["negative_source_xy"].astype(np.float64)).max()
                    )
                    d_full = _dice2d(cc_m["full10"], ref["full10_cc"])
                    d_pos0 = _dice2d(cc_m["positive0"], ref["positive0_cc"])
                    d_near2 = _dice2d(cc_m["boundary_near2"], ref["boundary_near2_cc"])
                    assert max_neg_diff <= 0.25 and min(d_full, d_pos0, d_near2) >= 0.985
                    drift_records.append(
                        {
                            "episode_id": int(case["episode_id"]),
                            "native_z": int(case["native_z"]),
                            "max_neg_coord_diff_px": max_neg_diff,
                            "full10_cc_dice_vs_step9": d_full,
                        }
                    )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    del predictor
    gc.collect()
    torch.cuda.empty_cache()

    slice_counts = count_episode_slices(manifest, dataset, data_dir, orientation=orientation)
    total_slices = int(sum(slice_counts.values()))
    p90_b = float(pd.DataFrame(timings)["slice_seconds_vit_b"].quantile(0.90))
    wall_b = p90_b * total_slices / float(max(1, n_gpus)) * 1.15 + 60.0
    wall_h = (p90_b * 2.5) * total_slices / float(max(1, n_gpus)) * 1.15 + 60.0
    proj_hours = float(n_gpus) * (wall_b + wall_h + (time.perf_counter() - started)) / 3600.0

    summary = {
        "dataset": dataset,
        "orientation": orientation,
        "total_slices": total_slices,
        "intra_run_exact_5folds": True,
        "vit_b_p90_slice_seconds": p90_b,
        "mean_decoder_calls": float(pd.DataFrame(timings)["decoder_calls"].mean()),
        "projected_total_gpu_hours_both_backbones": float(proj_hours),
        "pilot_wall_seconds": float(time.perf_counter() - started),
        "drift_records": drift_records,
        "sam_b_load": sam_b_meta,
    }
    write_json(output_dir / "pilot_summary.json", summary)
    return summary


def run_worker(
    dataset: str,
    backbone: str,
    orientation: str,
    lock_dir: Path | None,
    data_dir: Path,
    ckpt_paths: dict[int, Path],
    sam_checkpoint: Path,
    output_dir: Path,
    worker_id: int,
    num_workers: int,
    deadline_epoch: float,
    seed: int = 42,
) -> dict:
    """Execute LPT-sharded episodes for one dataset and frozen SAM backbone."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    manifest, expected_hw, iter_cases = _get_dataset_context(dataset, data_dir, orientation, lock_dir)
    slice_counts = count_episode_slices(manifest, dataset, data_dir, orientation=orientation)
    manifest = shard_manifest_lpt(manifest, slice_counts, worker_id=worker_id, num_workers=num_workers)

    expected_fp_map = (
        json.loads((lock_dir / "step8_fingerprints.json").read_text(encoding="utf-8"))
        if (dataset == "sabs" and lock_dir is not None)
        else None
    )

    predictor, sam_meta = load_tracked_sam(backbone, sam_checkpoint)
    support_cache = SupportFeatureCache()
    t_start = time.perf_counter()
    completed_eps = 0
    slice_count = 0

    for fold in sorted(manifest["fold"].unique()):
        model, _ = _load_fob_fold_ckpt(ckpt_paths[int(fold)])
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
                    {"dataset": dataset, "backbone": backbone, "worker": worker_id, "episode_id": eid, "slices": slice_count},
                )
                raise SystemExit(75)

            raw_slices: dict[str, list[np.ndarray]] = {arm: [] for arm in RAW_ARMS}
            cc_slices: dict[str, list[np.ndarray]] = {arm: [] for arm in RAW_ARMS}
            gt_slices: list[np.ndarray] = []
            other_slices: list[np.ndarray] = []
            slice_audits: list[dict] = []
            spacing_xyz = None

            for case in iter_cases(row):
                spacing_xyz = case["spacing_xyz"]
                fp_key = f"{int(case['episode_id'])}:{int(case['native_z'])}"
                exp_fp = expected_fp_map[fp_key] if expected_fp_map is not None else None
                r_m, c_m, s_aud, _ = run_stage4_slice(
                    model, predictor, support_cache, case, expected_hw=expected_hw, expected_fp=exp_fp
                )
                s_aud["dataset"] = dataset
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

            ep_metrics = measure_episode_stage4(
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
                    f"dataset={dataset} backbone={backbone} worker={worker_id} fold={fold} ep={eid} "
                    f"completed={completed_eps}/{len(manifest)} slices={slice_count} elapsed={time.perf_counter() - t_start:.1f}s",
                    flush=True,
                )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "dataset": dataset,
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
    p.add_argument("--dataset", choices=["chaost2", "sabs"], required=True)
    p.add_argument("--mode", choices=["pilot", "worker"], required=True)
    p.add_argument("--lock-dir", type=Path, default=None)
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
        run_pilot(
            dataset=a.dataset,
            orientation=a.orientation,
            lock_dir=a.lock_dir,
            data_dir=a.data_dir,
            ckpt_paths=ckpt_paths,
            sam_b_ckpt=a.sam_checkpoint,
            output_dir=a.output,
            n_gpus=a.n_gpus,
        )
    else:
        run_worker(
            dataset=a.dataset,
            backbone=a.backbone,
            orientation=a.orientation,
            lock_dir=a.lock_dir,
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
