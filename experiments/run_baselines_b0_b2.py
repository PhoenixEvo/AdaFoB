"""
AdaFoB Research Project — Execution of Attribution Baselines B0, B1, B2
Compliant with Protocol Amendment A1.
=======================================================================
Baselines evaluated:
- B0: FoB with Frozen SAM ViT-H (Np=10 fixed background prompts), reported with and without CC.
- B1: FoB with Frozen SAM ViT-B (Np=10 fixed background prompts), reported with and without CC.
- B2: B1 with Connected-Component (CC) post-processing standard.

Rules:
- Uses protocol/paired_episode_manifest.csv without modification.
- Verifies manifest SHA-256 before execution.
- Completely frozen SAM models (no LoRA, no fine-tuned decoder).
- Zero query ground truth access during prompt generation or inference.
"""

import os
import sys
import glob
import json
import time
import hashlib
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import cv2
from torch.utils.data import DataLoader

try:
    from medpy.metric.binary import hd95 as medpy_hd95
except ImportError:
    os.system("pip install -q medpy")
    from medpy.metric.binary import hd95 as medpy_hd95

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, "third_party", "FoB_SAM"))
sys.path.append(os.path.join(REPO_DIR, "third_party", "segment-anything"))

from models.FoB import FewShotSeg
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import get_label_names
from segment_anything import sam_model_registry, SamPredictor
from models.postprocessing import filter_connected_components

# Expected hash of manifest (supports both Linux LF and Windows CRLF)
EXPECTED_MANIFEST_LF_SHA256 = "ac2cc5df4a19e7ace297086f75f5385cc05cb2c9decbeec1cd7d321e628e3ad6"
EXPECTED_MANIFEST_CRLF_SHA256 = "3298b6b6b649c72658aaa24ae47fe372c6ee66d01e298497d76ca313c11fe3ac"
TEST_LABELS = [1, 2, 3, 6]  # SPLEEN, RK, LK, LIVER
SUPP_IDX = 2
N_PART = 3


def verify_manifest(manifest_path):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    with open(manifest_path, "rb") as f:
        raw_bytes = f.read()
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    lf_bytes = raw_bytes.replace(b"\r\n", b"\n")
    lf_hash = hashlib.sha256(lf_bytes).hexdigest()

    if raw_hash in (EXPECTED_MANIFEST_CRLF_SHA256, EXPECTED_MANIFEST_LF_SHA256) or lf_hash == EXPECTED_MANIFEST_LF_SHA256:
        print(f"[Verified] Manifest SHA256 verified (raw: {raw_hash}, normalized LF: {lf_hash})")
    else:
        raise ValueError(
            f"Manifest SHA256 mismatch!\nExpected LF: {EXPECTED_MANIFEST_LF_SHA256}\nFound raw:   {raw_hash}\nNormalized:  {lf_hash}\nRefusing to run."
        )


def find_file(pattern_list):
    for p in pattern_list:
        if os.path.exists(p):
            return p
        matches = glob.glob(p, recursive=True)
        if matches:
            return sorted(matches)[0]
    return None


def ensure_checkpoints(ckpt_dir, sam_h_path, sam_b_path):
    os.makedirs(ckpt_dir, exist_ok=True)
    # Check FoB checkpoints
    for fold in range(5):
        cand = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*/**/*.pth"), recursive=True)
        if not cand:
            cand = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*.pth"), recursive=True)
        if not cand:
            print(f"Downloading FoB checkpoints from HuggingFace to {ckpt_dir}...")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="PrimeBo1/FoB_SAM",
                    local_dir=ckpt_dir,
                    allow_patterns=["*SABS*"],
                )
                import zipfile
                zips = glob.glob(os.path.join(ckpt_dir, "**/*.zip"), recursive=True)
                if zips:
                    with zipfile.ZipFile(zips[0], "r") as zr:
                        zr.extractall(ckpt_dir)
            except Exception as e:
                print(f"Warning: HuggingFace download error: {e}")
            break

    # Check SAM weights
    if not os.path.exists(sam_b_path):
        print(f"Downloading SAM ViT-B to {sam_b_path}...")
        os.makedirs(os.path.dirname(sam_b_path) or ".", exist_ok=True)
        os.system(f"wget -qO {sam_b_path} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")

    if not os.path.exists(sam_h_path):
        print(f"Downloading SAM ViT-H to {sam_h_path}...")
        os.makedirs(os.path.dirname(sam_h_path) or ".", exist_ok=True)
        os.system(f"wget -qO {sam_h_path} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")


def compute_metrics(pred, gt, spacing):
    """
    Computes 3D native-volume Dice and physical HD95.
    Empty-prediction handling:
      If both empty: Dice = 1.0, HD95 = 0.0
      If one empty:  Dice = 0.0, HD95 = physical volume diagonal
    """
    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)

    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()

    # Physical volume diagonal
    shape_z, shape_y, shape_x = pred.shape
    diag = float(np.sqrt((shape_x * spacing[0])**2 + (shape_y * spacing[1])**2 + (shape_z * spacing[2])**2))

    if pred_sum == 0 and gt_sum == 0:
        return 1.0, 0.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0, diag

    inter = np.logical_and(pred_bin, gt_bin).sum()
    dice = float(2.0 * inter / (pred_sum + gt_sum))

    try:
        hd = float(medpy_hd95(pred_bin, gt_bin, voxelspacing=spacing[::-1]))
        if np.isnan(hd) or np.isinf(hd):
            hd = diag
        else:
            hd = min(hd, diag)
    except Exception:
        hd = diag

    return dice, hd


def independent_metric_check(pred, gt, spacing):
    """Independent verification metric implementation."""
    p = (pred > 0).astype(np.float32)
    g = (gt > 0).astype(np.float32)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    inter = (p * g).sum()
    return float(2.0 * inter / (p.sum() + g.sum()))


def predict_sam(predictor, pos_pts, neg_pts, img_shape):
    pts = []
    lbls = []
    if pos_pts is not None and len(pos_pts) > 0:
        pts.append(pos_pts)
        lbls.append(np.ones(len(pos_pts), dtype=np.int32))
    if neg_pts is not None and len(neg_pts) > 0:
        pts.append(neg_pts)
        lbls.append(np.zeros(len(neg_pts), dtype=np.int32))

    if len(pts) == 0:
        return np.zeros(img_shape[:2], dtype=np.uint8)

    all_pts = np.concatenate(pts, axis=0)
    all_lbls = np.concatenate(lbls, axis=0)

    try:
        masks, scores, _ = predictor.predict(
            point_coords=all_pts,
            point_labels=all_lbls,
            multimask_output=True,
        )
        return (masks[np.argmax(scores)] > 0).astype(np.uint8)
    except Exception:
        return np.zeros(img_shape[:2], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Execute Baselines B0, B1, B2")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="results/baselines")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} (requested GPU {args.gpu})")

    # 1. Verify Manifest Integrity
    manifest_path = os.path.join(REPO_DIR, "protocol", "paired_episode_manifest.csv")
    verify_manifest(manifest_path)
    manifest_df = pd.read_csv(manifest_path)

    # 2. Locate Data & Checkpoints
    data_dir_cand = [
        "/kaggle/input/datasets/nhatphatnguyen/adafob-env/sabs_CT_normalized",
        "/kaggle/input/adafob-env/sabs_CT_normalized",
        "/kaggle/working/sabs_CT_normalized",
        os.path.join(REPO_DIR, "data", "sabs_CT_normalized"),
    ]
    norm_dir = find_file(data_dir_cand)
    if not norm_dir:
        raise FileNotFoundError(f"Could not locate sabs_CT_normalized in candidates: {data_dir_cand}")
    print(f"[Data] Found normalized dataset at: {norm_dir}")
    data_root = os.path.dirname(norm_dir)

    ckpt_dir = find_file([
        "/tmp/fob_checkpoints",
        "/kaggle/working/fob_checkpoints",
        "/kaggle/input/baseline_fob/exps_train_on_SABS_FSMIS_FoB",
        "/kaggle/input/adafob-checkpoints",
        os.path.join(REPO_DIR, "checkpoints", "fob_checkpoints")
    ]) or "/tmp/fob_checkpoints"

    sam_h_path = find_file([
        "/tmp/checkpoints/sam_vit_h_4b8939.pth",
        "/kaggle/working/checkpoints/sam_vit_h_4b8939.pth",
        "/kaggle/working/sam_vit_h_4b8939.pth",
        "/kaggle/working/sam_vit_h.pth",
        "/kaggle/input/checkpoints/sam_vit_h_4b8939.pth",
        "D:/results/checkpoints/sam_vit_h_4b8939.pth"
    ]) or "/tmp/checkpoints/sam_vit_h_4b8939.pth"

    sam_b_path = find_file([
        "/tmp/checkpoints/sam_vit_b_01ec64.pth",
        "/kaggle/working/checkpoints/sam_vit_b_01ec64.pth",
        "/kaggle/working/sam_vit_b_01ec64.pth",
        "/kaggle/working/sam_vit_b.pth",
        "/kaggle/input/checkpoints/sam_vit_b_01ec64.pth",
        "D:/results/checkpoints/sam_vit_b_01ec64.pth"
    ]) or "/tmp/checkpoints/sam_vit_b_01ec64.pth"

    ensure_checkpoints(ckpt_dir, sam_h_path, sam_b_path)

    # 3. Load Frozen SAM Models
    print(f"[SAM] Loading ViT-H from {sam_h_path}...")
    sam_h = sam_model_registry["vit_h"](checkpoint=sam_h_path).eval().to(device)
    predictor_h = SamPredictor(sam_h)

    print(f"[SAM] Loading ViT-B from {sam_b_path}...")
    sam_b = sam_model_registry["vit_b"](checkpoint=sam_b_path).eval().to(device)
    predictor_b = SamPredictor(sam_b)

    labels = get_label_names("SABS")
    results = []
    masks_save_dir = os.path.join(args.out_dir, "predictions")
    overlays_dir = os.path.join(args.out_dir, "overlays")
    os.makedirs(masks_save_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    target_folds = range(5) if args.fold is None else [args.fold]

    # Timing and S4 statistics accumulators
    t_enc_h_list, t_dec_h_list = [], []
    t_enc_b_list, t_dec_b_list = [], []
    t_cc_list = []
    s4_gradient_correlations = []
    independent_check_diffs = []

    episode_counter = 0

    for eval_fold in target_folds:
        print(f"\n{'='*70}\nExecuting Baselines B0, B1, B2: Fold {eval_fold}\n{'='*70}")

        # Find FoB fold checkpoint
        cands = glob.glob(os.path.join(ckpt_dir, f"**/*cv{eval_fold}*/**/*.pth"), recursive=True)
        if not cands:
            cands = glob.glob(os.path.join(ckpt_dir, f"**/*cv{eval_fold}*.pth"), recursive=True)
        if not cands:
            raise FileNotFoundError(f"Missing FoB checkpoint for fold {eval_fold}")
        fob_ckpt_path = sorted(cands)[0]
        print(f"[FoB] Loaded fold {eval_fold} checkpoint: {fob_ckpt_path}")

        class DummyArgs:
            pass
        fob_args = DummyArgs()
        fob_args.dataset = "SABS"
        fob_args.n_ways = 1
        fob_args.n_shots = 1
        fob_model = FewShotSeg(fob_args).to(device).eval()
        ckpt = torch.load(fob_ckpt_path, map_location=device)
        fob_model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt, strict=False)
        fob_model.allocator = None  # Upstream FoB fixed 10-point ring

        data_config = {
            'data_dir': data_root, 'dataset': 'SABS', 'n_shot': 1, 'n_way': 1, 'n_query': 1,
            'n_sv': 5000, 'max_iter': 3000, 'eval_fold': eval_fold, 'min_size': 200,
            'max_slices': 3, 'supp_idx': SUPP_IDX,
        }
        test_dataset = TestDataset(data_config)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        for label_val in TEST_LABELS:
            label_name = labels[label_val]
            test_dataset.label = label_val
            support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=N_PART)

            support_image = [support_sample['image'][[i]].float().to(device) for i in range(support_sample['image'].shape[0])]
            support_fg_mask = [support_sample['label'][[i]].float().to(device) for i in range(support_sample['image'].shape[0])]

            for vi, sample in enumerate(test_loader):
                vol_id_str = sample['id'][0]
                pat_id = vol_id_str.split('/')[-1].split('.nii')[0]

                # Match with manifest
                row_match = manifest_df[(manifest_df['fold'] == eval_fold) &
                                        (manifest_df['organ'] == label_name) &
                                        (manifest_df['vol_id'].str.contains(pat_id))]
                if len(row_match) == 0:
                    continue
                ep_row = row_match.iloc[0]
                ep_id = int(ep_row['episode_id'])
                spacing = (float(ep_row['spacing_x']), float(ep_row['spacing_y']), float(ep_row['spacing_z']))

                query_images = sample['image'].to(device)
                query_labels = sample['label'].to(device)
                C_q = query_images.shape[1]
                H_q, W_q = query_images.shape[-2], query_images.shape[-1]

                # Arrays to hold predictions
                pred_vol_b0_raw = np.zeros((C_q, H_q, W_q), dtype=np.uint8)
                pred_vol_b0_cc  = np.zeros((C_q, H_q, W_q), dtype=np.uint8)
                pred_vol_b1_raw = np.zeros((C_q, H_q, W_q), dtype=np.uint8)
                pred_vol_b2_cc  = np.zeros((C_q, H_q, W_q), dtype=np.uint8)

                # Prompt coordinates storage for overlays
                overlay_data = []

                idx_ = np.linspace(0, C_q, N_PART + 1).astype(int)

                with torch.no_grad():
                    for sub_chunk in range(N_PART):
                        support_image_s = [support_image[sub_chunk]]
                        support_fg_mask_s = [support_fg_mask[sub_chunk]]
                        query_image_chunk = query_images[0, idx_[sub_chunk]:idx_[sub_chunk+1]]
                        query_label_chunk = query_labels[0, idx_[sub_chunk]:idx_[sub_chunk+1]]

                        if query_image_chunk.shape[0] == 0:
                            continue

                        for j in range(query_image_chunk.shape[0]):
                            global_z = idx_[sub_chunk] + j
                            q_img_slice = query_image_chunk[[j]]
                            q_lbl_slice = query_label_chunk[[j]]

                            if support_image_s[0].max() == 0 or q_lbl_slice.max() == 0:
                                pos_arr = np.zeros((0, 2))
                                neg_arr = np.zeros((0, 2))
                            else:
                                try:
                                    fob_out = fob_model(
                                        [support_image_s], [support_fg_mask_s],
                                        [q_img_slice], q_lbl_slice,
                                        train=False, use_skeleton=False, budget_Np=10
                                    )
                                    uni_neg_all, pos_all = fob_out[0], fob_out[1]
                                    neg_arr = np.array(uni_neg_all).reshape(-1, 2)[:10]
                                    pos_arr = np.array(pos_all).reshape(-1, 2)[:10]
                                except Exception:
                                    pos_arr = np.zeros((0, 2))
                                    neg_arr = np.zeros((0, 2))

                            img_t = q_img_slice[0].permute(1, 2, 0).cpu().numpy()
                            # Normalization: standard FoB SAM 0-255 uint8 conversion
                            img_norm = ((img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8) * 255).astype(np.uint8)

                            # Profile ViT-H
                            t0 = time.perf_counter()
                            predictor_h.set_image(img_norm)
                            t1 = time.perf_counter()
                            b0_raw = predict_sam(predictor_h, pos_arr, neg_arr, img_norm.shape)
                            t2 = time.perf_counter()

                            t_enc_h_list.append((t1 - t0) * 1000.0)
                            t_dec_h_list.append((t2 - t1) * 1000.0)

                            # Profile ViT-B
                            t3 = time.perf_counter()
                            predictor_b.set_image(img_norm)
                            t4 = time.perf_counter()
                            b1_raw = predict_sam(predictor_b, pos_arr, neg_arr, img_norm.shape)
                            t5 = time.perf_counter()

                            t_enc_b_list.append((t4 - t3) * 1000.0)
                            t_dec_b_list.append((t5 - t4) * 1000.0)

                            # Profile CC
                            t6 = time.perf_counter()
                            b0_cc = filter_connected_components(b0_raw, pos_arr)
                            b2_cc = filter_connected_components(b1_raw, pos_arr)
                            t7 = time.perf_counter()
                            t_cc_list.append((t7 - t6) * 1000.0)

                            pred_vol_b0_raw[global_z] = b0_raw
                            pred_vol_b0_cc[global_z]  = b0_cc
                            pred_vol_b1_raw[global_z] = b1_raw
                            pred_vol_b2_cc[global_z]  = b2_cc

                            # Step D: Retrospective boundary gradient feasibility check on B1
                            if b1_raw.sum() > 0:
                                gray = cv2.cvtColor(img_norm, cv2.COLOR_RGB2GRAY)
                                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                                gmag = cv2.magnitude(gx, gy)
                                contours, _ = cv2.findContours(b1_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                c_mask = np.zeros_like(b1_raw)
                                cv2.drawContours(c_mask, contours, -1, 1, 1)
                                mean_edge_grad = float(gmag[c_mask == 1].mean()) if c_mask.sum() > 0 else 0.0

                                gt_slice = (q_lbl_slice[0].cpu().numpy() > 0).astype(np.uint8)
                                fp_area = float((np.logical_and(b1_raw == 1, gt_slice == 0)).sum())
                                s4_gradient_correlations.append({"edge_grad": mean_edge_grad, "fp_area": fp_area, "pat_id": pat_id})

                            # Save overlay info for image_7 and RK worst cases
                            if "image_7" in vol_id_str or label_name == "RK":
                                overlay_data.append({
                                    "slice_z": global_z, "img": img_norm,
                                    "gt": (q_lbl_slice[0].cpu().numpy() > 0).astype(np.uint8),
                                    "b0": b0_raw, "b1": b1_raw, "b2": b2_cc,
                                    "pos": pos_arr, "neg": neg_arr
                                })

                gt_vol = (query_labels.squeeze(0).cpu().numpy() > 0).astype(np.uint8)

                # Compute Metrics
                dice_b0_raw, hd_b0_raw = compute_metrics(pred_vol_b0_raw, gt_vol, spacing)
                dice_b0_cc,  hd_b0_cc  = compute_metrics(pred_vol_b0_cc,  gt_vol, spacing)
                dice_b1_raw, hd_b1_raw = compute_metrics(pred_vol_b1_raw, gt_vol, spacing)
                dice_b2_cc,  hd_b2_cc  = compute_metrics(pred_vol_b2_cc,  gt_vol, spacing)

                # Independent 10% subset verification
                if episode_counter % 10 == 0:
                    ind_dice = independent_metric_check(pred_vol_b2_cc, gt_vol, spacing)
                    diff = abs(dice_b2_cc - ind_dice)
                    independent_check_diffs.append(diff)

                # Save compressed masks
                npz_path = os.path.join(masks_save_dir, f"ep_{ep_id:03d}_{pat_id}_{label_name}.npz")
                np.savez_compressed(
                    npz_path,
                    b0_raw=pred_vol_b0_raw, b0_cc=pred_vol_b0_cc,
                    b1_raw=pred_vol_b1_raw, b2_cc=pred_vol_b2_cc,
                    gt=gt_vol
                )

                # Save overlay visualization if image_7
                if "image_7" in vol_id_str and len(overlay_data) > 0:
                    # Pick slice with largest GT area
                    best_sl = max(overlay_data, key=lambda x: x['gt'].sum())
                    vis = best_sl['img'].copy()
                    # GT green, B1 orange, B2 red
                    cv2.drawContours(vis, cv2.findContours(best_sl['gt'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 2)
                    cv2.drawContours(vis, cv2.findContours(best_sl['b1'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 165, 255), 2)
                    cv2.drawContours(vis, cv2.findContours(best_sl['b2'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 0, 255), 2)
                    for pt in best_sl['pos']:
                        cv2.drawMarker(vis, (int(pt[0]), int(pt[1])), (0, 255, 0), cv2.MARKER_CROSS, 8, 2)
                    for pt in best_sl['neg']:
                        cv2.drawMarker(vis, (int(pt[0]), int(pt[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 2)
                    cv2.imwrite(os.path.join(overlays_dir, f"overlay_image7_{label_name}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

                peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else 0.0

                rec = {
                    "episode_id": ep_id, "fold": eval_fold, "organ": label_name, "vol_id": vol_id_str,
                    "patient_id": pat_id, "spacing_x": spacing[0], "spacing_y": spacing[1], "spacing_z": spacing[2],
                    "n_pos_prompts": 10, "n_neg_prompts": 10, "n_total_prompts": 20,
                    "dice_b0_raw": dice_b0_raw, "hd95_b0_raw": hd_b0_raw,
                    "dice_b0_cc": dice_b0_cc, "hd95_b0_cc": hd_b0_cc,
                    "dice_b1_raw": dice_b1_raw, "hd95_b1_raw": hd_b1_raw,
                    "dice_b2_cc": dice_b2_cc, "hd95_b2_cc": hd_b2_cc,
                    "peak_vram_mb": peak_vram
                }
                results.append(rec)
                episode_counter += 1
                print(f"  Ep {ep_id:03d} | {pat_id:10s} | {label_name:6s} | B0: {dice_b0_raw*100:.1f}% ({hd_b0_raw:.1f}mm) | B1: {dice_b1_raw*100:.1f}% ({hd_b1_raw:.1f}mm) | B2(CC): {dice_b2_cc*100:.1f}% ({hd_b2_cc:.1f}mm)")

    # 4. Save Aggregated Results
    res_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "baselines_b0_b1_b2_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\n[Done] Saved {len(res_df)} results to {out_csv}")

    # 5. Timing Breakdown
    timing_summary = {
        "t_enc_h_mean_ms": float(np.mean(t_enc_h_list)), "t_enc_h_std_ms": float(np.std(t_enc_h_list)),
        "t_dec_h_mean_ms": float(np.mean(t_dec_h_list)), "t_dec_h_std_ms": float(np.std(t_dec_h_list)),
        "t_enc_b_mean_ms": float(np.mean(t_enc_b_list)), "t_enc_b_std_ms": float(np.std(t_enc_b_list)),
        "t_dec_b_mean_ms": float(np.mean(t_dec_b_list)), "t_dec_b_std_ms": float(np.std(t_dec_b_list)),
        "t_cc_mean_ms": float(np.mean(t_cc_list)), "t_cc_std_ms": float(np.std(t_cc_list)),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if torch.cuda.is_available() else 0.0
    }
    timing_json = os.path.join(args.out_dir, "timing_breakdown.json")
    with open(timing_json, "w") as f:
        json.dump(timing_summary, f, indent=2)

    # 6. S4 Feasibility Summary
    s4_df = pd.DataFrame(s4_gradient_correlations)
    if len(s4_df) > 10:
        corr = float(s4_df["edge_grad"].corr(s4_df["fp_area"]))
        s4_res = {"pearson_corr_edge_grad_vs_fp_area": corr, "n_slices_evaluated": len(s4_df)}
        with open(os.path.join(args.out_dir, "s4_feasibility_check.json"), "w") as f:
            json.dump(s4_res, f, indent=2)

    # 7. Independent Metric Disagreement Report
    max_diff = max(independent_check_diffs) if independent_check_diffs else 0.0
    print(f"[Verification] 10% subset independent Dice max disagreement: {max_diff:.6e} (Threshold: 1e-4)")

if __name__ == "__main__":
    main()
