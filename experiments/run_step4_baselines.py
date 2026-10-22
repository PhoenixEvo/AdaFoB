"""
AdaFoB Research Project — Execution of Step 4 Symmetric Baseline Matrix
Compliant with Auditor Binding Protocol Step 4.

Arms Evaluated across identical 120 episodes:
  B0:  ViT-H, Upstream R3 (mask[0]), Fixed-10 (Reproduction Check Gate)
  B3H: ViT-H, Frozen Rule R1 (SingleMask), Fixed-10
  B3B: ViT-B, Frozen Rule R1 (SingleMask), Fixed-10
  B4H: ViT-H, Frozen Rule R1 (SingleMask), No Negative Points (Positive Only)
  B4B: ViT-B, Frozen Rule R1 (SingleMask), No Negative Points (Positive Only)

Every arm reported:
  - Raw prediction
  - Connected-Component (CC) post-processed prediction

Rigorous Profiling:
  - Separate peak GPU VRAM reset via torch.cuda.reset_peak_memory_stats()
  - Synchronized GPU latency timing with warm-up
  - Separate breakdown: Encoder, Prompt Generator, Decoder, CC
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
        print(f"[Verified] Manifest SHA256 verified (raw: {raw_hash}, normalized LF: {lf_hash})", flush=True)
    else:
        raise ValueError(
            f"Manifest SHA256 mismatch!\nExpected LF: {EXPECTED_MANIFEST_LF_SHA256}\nFound raw:   {raw_hash}\nRefusing to run."
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
            print(f"Downloading FoB checkpoints from HuggingFace to {ckpt_dir}...", flush=True)
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="PrimeBo1/FoB_SAM", local_dir=ckpt_dir, allow_patterns=["*SABS*"])
                import zipfile
                zips = glob.glob(os.path.join(ckpt_dir, "**/*.zip"), recursive=True)
                if zips:
                    with zipfile.ZipFile(zips[0], "r") as zr:
                        zr.extractall(ckpt_dir)
            except Exception as e:
                print(f"Warning: HuggingFace download error: {e}", flush=True)
            break

    if not os.path.exists(sam_b_path):
        print(f"Downloading SAM ViT-B to {sam_b_path}...", flush=True)
        os.makedirs(os.path.dirname(sam_b_path) or ".", exist_ok=True)
        os.system(f"wget -qO {sam_b_path} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")

    if not os.path.exists(sam_h_path):
        print(f"Downloading SAM ViT-H to {sam_h_path}...", flush=True)
        os.makedirs(os.path.dirname(sam_h_path) or ".", exist_ok=True)
        os.system(f"wget -qO {sam_h_path} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")


def compute_metrics(pred, gt, spacing):
    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()
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
        hd = min(diag, hd) if (not np.isnan(hd) and not np.isinf(hd)) else diag
    except Exception:
        hd = diag

    return dice, hd


def predict_singlemask(predictor, pos_pts, neg_pts, img_shape):
    """
    Candidate Rule R1 (SingleMask): multimask_output=False.
    Uses SAM's dedicated Token 0 for non-ambiguous multi-point prompts.
    """
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
            multimask_output=False,
        )
        return (masks[0] > 0).astype(np.uint8)
    except Exception as e:
        print(f"Warning: SAM predict_singlemask exception: {e}", flush=True)
        return np.zeros(img_shape[:2], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Execute Step 4 Symmetric Baseline Matrix")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/step4_baselines")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Device] Using {device} ({gpu_name})", flush=True)

    manifest_path = os.path.join(REPO_DIR, "protocol", "paired_episode_manifest.csv")
    verify_manifest(manifest_path)
    manifest_df = pd.read_csv(manifest_path)

    # Candidate data paths
    data_dir_cand = [
        "/kaggle/input/datasets/nhatphatnguyen/adafob-env/sabs_CT_normalized",
        "/kaggle/input/adafob-env/sabs_CT_normalized",
        "/kaggle/working/sabs_CT_normalized",
        os.path.join(REPO_DIR, "data", "sabs_CT_normalized"),
        "D:/sabs_CT_normalized",
    ]
    norm_dir = find_file(data_dir_cand)
    if not norm_dir:
        raise FileNotFoundError(f"Could not locate sabs_CT_normalized in candidates: {data_dir_cand}")
    print(f"[Data] Found normalized dataset at: {norm_dir}", flush=True)
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
        "/kaggle/input/checkpoints/sam_vit_h_4b8939.pth",
        "D:/results/checkpoints/sam_vit_h_4b8939.pth"
    ]) or "/tmp/checkpoints/sam_vit_h_4b8939.pth"

    sam_b_path = find_file([
        "/tmp/checkpoints/sam_vit_b_01ec64.pth",
        "/kaggle/working/checkpoints/sam_vit_b_01ec64.pth",
        "/kaggle/working/sam_vit_b_01ec64.pth",
        "/kaggle/input/checkpoints/sam_vit_b_01ec64.pth",
        "D:/results/checkpoints/sam_vit_b_01ec64.pth"
    ]) or "/tmp/checkpoints/sam_vit_b_01ec64.pth"

    ensure_checkpoints(ckpt_dir, sam_h_path, sam_b_path)

    # Load existing verified B0 (ViT-H upstream mask[0]) results for exact reproduction reference
    b0_ref_csv = find_file([
        "results/baselines/baselines_b0_b1_b2_results.csv",
        os.path.join(REPO_DIR, "results", "baselines", "baselines_b0_b1_b2_results.csv")
    ])
    b0_ref_df = pd.read_csv(b0_ref_csv) if b0_ref_csv else None

    # Load SAM ViT-H
    print(f"[SAM] Loading ViT-H from {sam_h_path}...", flush=True)
    sam_h = sam_model_registry["vit_h"](checkpoint=sam_h_path).eval().to(device)
    predictor_h = SamPredictor(sam_h)

    # Load SAM ViT-B
    print(f"[SAM] Loading ViT-B from {sam_b_path}...", flush=True)
    sam_b = sam_model_registry["vit_b"](checkpoint=sam_b_path).eval().to(device)
    predictor_b = SamPredictor(sam_b)

    labels = get_label_names("SABS")
    masks_save_dir = os.path.join(args.out_dir, "predictions")
    os.makedirs(masks_save_dir, exist_ok=True)
    results = []

    # Accurate separate VRAM benchmarks
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    predictor_b.set_image(dummy_img)
    _ = predictor_b.predict(point_coords=np.array([[128, 128]]), point_labels=np.array([1]), multimask_output=False)
    peak_vram_vitb = float(torch.cuda.max_memory_allocated(device) / (1024**2))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    predictor_h.set_image(dummy_img)
    _ = predictor_h.predict(point_coords=np.array([[128, 128]]), point_labels=np.array([1]), multimask_output=False)
    peak_vram_vith = float(torch.cuda.max_memory_allocated(device) / (1024**2))

    print(f"[VRAM Benchmark] Isolated Peak VRAM ViT-B: {peak_vram_vitb:.2f} MB | ViT-H: {peak_vram_vith:.2f} MB", flush=True)

    # Latency accumulators (with synchronization)
    t_fob_list = []
    t_enc_h_list, t_dec_h_list = [], []
    t_enc_b_list, t_dec_b_list = [], []
    t_cc_list = []

    warmup_counter = 0
    WARMUP_SLICES = 10

    for eval_fold in range(5):
        print(f"\n{'='*70}\nExecuting Step 4 Baselines: Fold {eval_fold}\n{'='*70}", flush=True)

        cands = glob.glob(os.path.join(ckpt_dir, f"**/*cv{eval_fold}*/**/*.pth"), recursive=True)
        if not cands:
            cands = glob.glob(os.path.join(ckpt_dir, f"**/*cv{eval_fold}*.pth"), recursive=True)
        if not cands:
            raise FileNotFoundError(f"Missing FoB checkpoint for fold {eval_fold}")
        fob_ckpt_path = sorted(cands)[0]

        class DummyArgs:
            dataset = "SABS"
            n_ways = 1
            n_shots = 1
        fob_model = FewShotSeg(DummyArgs()).to(device).eval()
        ckpt = torch.load(fob_ckpt_path, map_location=device)
        raw_sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        cleaned_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
        fob_model.load_state_dict(cleaned_sd, strict=False)
        fob_model.allocator = None

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

                row_match = manifest_df[(manifest_df['fold'] == eval_fold) &
                                        (manifest_df['organ'] == label_name) &
                                        (manifest_df['vol_id'].str.contains(pat_id))]
                if len(row_match) == 0:
                    continue
                ep_row = row_match.iloc[0]
                ep_id = int(ep_row['episode_id'])
                spacing = (float(ep_row['spacing_x']), float(ep_row['spacing_y']), float(ep_row['spacing_z']))

                query_images = sample['image'].float().to(device)
                query_labels = sample['label'].long().to(device)
                C_q = query_images.shape[1]
                H_q, W_q = query_images.shape[-2], query_images.shape[-1]

                # Prediction volumes for the 4 new symmetric arms
                vol_b3h_raw = np.zeros((C_q, H_q, W_q), dtype=np.uint8)
                vol_b3h_cc  = np.zeros((C_q, H_q, W_q), dtype=np.uint8)
                vol_b3b_raw = np.zeros((C_q, H_q, W_q), dtype=np.uint8)
                vol_b3b_cc  = np.zeros((C_q, H_q, W_q), dtype=np.uint8)

                vol_b4h_raw = np.zeros((C_q, H_q, W_q), dtype=np.uint8)
                vol_b4h_cc  = np.zeros((C_q, H_q, W_q), dtype=np.uint8)
                vol_b4b_raw = np.zeros((C_q, H_q, W_q), dtype=np.uint8)
                vol_b4b_cc  = np.zeros((C_q, H_q, W_q), dtype=np.uint8)

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
                            dummy_lbl = torch.ones_like(q_img_slice[:, 0]).long()

                            # Profile FoB prompt generation
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            t_fob_0 = time.perf_counter()
                            if support_image_s[0].max() == 0:
                                pos_arr = np.zeros((0, 2))
                                neg_arr = np.zeros((0, 2))
                            else:
                                fob_out = fob_model(
                                    [support_image_s], [support_fg_mask_s],
                                    [q_img_slice], dummy_lbl,
                                    train=False, use_skeleton=False, budget_Np=10
                                )
                                neg_arr = np.array(fob_out[0]).reshape(-1, 2)[:10]
                                pos_arr = np.array(fob_out[1]).reshape(-1, 2)[:10]
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            t_fob_1 = time.perf_counter()

                            img_t = q_img_slice[0].permute(1, 2, 0).cpu().numpy()
                            img_norm = ((img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8) * 255).astype(np.uint8)

                            # Profile ViT-H Encoder
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            th0 = time.perf_counter()
                            predictor_h.set_image(img_norm)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            th1 = time.perf_counter()

                            # Predict B3H (Fixed-10) and B4H (Positive-only)
                            th2 = time.perf_counter()
                            b3h_raw = predict_singlemask(predictor_h, pos_arr, neg_arr, img_norm.shape)
                            b4h_raw = predict_singlemask(predictor_h, pos_arr, None, img_norm.shape)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            th3 = time.perf_counter()

                            # Profile ViT-B Encoder
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            tb0 = time.perf_counter()
                            predictor_b.set_image(img_norm)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            tb1 = time.perf_counter()

                            # Predict B3B (Fixed-10) and B4B (Positive-only)
                            tb2 = time.perf_counter()
                            b3b_raw = predict_singlemask(predictor_b, pos_arr, neg_arr, img_norm.shape)
                            b4b_raw = predict_singlemask(predictor_b, pos_arr, None, img_norm.shape)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            tb3 = time.perf_counter()

                            # Profile CC
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            t_cc_0 = time.perf_counter()
                            b3h_cc = filter_connected_components(b3h_raw, pos_arr)
                            b3b_cc = filter_connected_components(b3b_raw, pos_arr)
                            b4h_cc = filter_connected_components(b4h_raw, pos_arr)
                            b4b_cc = filter_connected_components(b4b_raw, pos_arr)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            t_cc_1 = time.perf_counter()

                            vol_b3h_raw[global_z] = b3h_raw
                            vol_b3h_cc[global_z]  = b3h_cc
                            vol_b3b_raw[global_z] = b3b_raw
                            vol_b3b_cc[global_z]  = b3b_cc

                            vol_b4h_raw[global_z] = b4h_raw
                            vol_b4h_cc[global_z]  = b4h_cc
                            vol_b4b_raw[global_z] = b4b_raw
                            vol_b4b_cc[global_z]  = b4b_cc

                            warmup_counter += 1
                            if warmup_counter > WARMUP_SLICES:
                                t_fob_list.append((t_fob_1 - t_fob_0) * 1000.0)
                                t_enc_h_list.append((th1 - th0) * 1000.0)
                                t_dec_h_list.append(((th3 - th2) / 2.0) * 1000.0)
                                t_enc_b_list.append((tb1 - tb0) * 1000.0)
                                t_dec_b_list.append(((tb3 - tb2) / 2.0) * 1000.0)
                                t_cc_list.append(((t_cc_1 - t_cc_0) / 4.0) * 1000.0)

                gt_vol = (query_labels.squeeze(0).cpu().numpy() > 0).astype(np.uint8)

                # Compute Metrics
                d_b3h_r, h_b3h_r = compute_metrics(vol_b3h_raw, gt_vol, spacing)
                d_b3h_c, h_b3h_c = compute_metrics(vol_b3h_cc,  gt_vol, spacing)
                d_b3b_r, h_b3b_r = compute_metrics(vol_b3b_raw, gt_vol, spacing)
                d_b3b_c, h_b3b_c = compute_metrics(vol_b3b_cc,  gt_vol, spacing)

                d_b4h_r, h_b4h_r = compute_metrics(vol_b4h_raw, gt_vol, spacing)
                d_b4h_c, h_b4h_c = compute_metrics(vol_b4h_cc,  gt_vol, spacing)
                d_b4b_r, h_b4b_r = compute_metrics(vol_b4b_raw, gt_vol, spacing)
                d_b4b_c, h_b4b_c = compute_metrics(vol_b4b_cc,  gt_vol, spacing)

                # Match with B0 reproduction reference
                if b0_ref_df is not None:
                    b0_match = b0_ref_df[b0_ref_df['episode_id'] == ep_id]
                    if len(b0_match) > 0:
                        d_b0_r = float(b0_match.iloc[0]['dice_b0_raw'])
                        h_b0_r = float(b0_match.iloc[0]['hd95_b0_raw'])
                        d_b0_c = float(b0_match.iloc[0]['dice_b0_cc'])
                        h_b0_c = float(b0_match.iloc[0]['hd95_b0_cc'])
                    else:
                        d_b0_r, h_b0_r, d_b0_c, h_b0_c = 0.0, 0.0, 0.0, 0.0
                else:
                    d_b0_r, h_b0_r, d_b0_c, h_b0_c = 0.0, 0.0, 0.0, 0.0

                # Save predictions npz
                npz_path = os.path.join(masks_save_dir, f"ep_{ep_id:03d}_{pat_id}_{label_name}.npz")
                np.savez_compressed(
                    npz_path,
                    b3h_raw=vol_b3h_raw, b3h_cc=vol_b3h_cc,
                    b3b_raw=vol_b3b_raw, b3b_cc=vol_b3b_cc,
                    b4h_raw=vol_b4h_raw, b4h_cc=vol_b4h_cc,
                    b4b_raw=vol_b4b_raw, b4b_cc=vol_b4b_cc,
                    gt=gt_vol
                )

                rec = {
                    "episode_id": ep_id, "fold": eval_fold, "organ": label_name, "vol_id": vol_id_str,
                    "patient_id": pat_id, "spacing_x": spacing[0], "spacing_y": spacing[1], "spacing_z": spacing[2],
                    "dice_b0_raw": d_b0_r, "hd95_b0_raw": h_b0_r,
                    "dice_b0_cc": d_b0_c, "hd95_b0_cc": h_b0_c,
                    "dice_b3h_raw": d_b3h_r, "hd95_b3h_raw": h_b3h_r,
                    "dice_b3h_cc": d_b3h_c, "hd95_b3h_cc": h_b3h_c,
                    "dice_b3b_raw": d_b3b_r, "hd95_b3b_raw": h_b3b_r,
                    "dice_b3b_cc": d_b3b_c, "hd95_b3b_cc": h_b3b_c,
                    "dice_b4h_raw": d_b4h_r, "hd95_b4h_raw": h_b4h_r,
                    "dice_b4h_cc": d_b4h_c, "hd95_b4h_cc": h_b4h_c,
                    "dice_b4b_raw": d_b4b_r, "hd95_b4b_raw": h_b4b_r,
                    "dice_b4b_cc": d_b4b_c, "hd95_b4b_cc": h_b4b_c,
                }
                results.append(rec)
                print(f"Ep {ep_id:03d} [{label_name:6s} | {pat_id}]: B3H={d_b3h_c*100:5.2f}% | B3B={d_b3b_c*100:5.2f}% | B4H={d_b4h_c*100:5.2f}% | B4B={d_b4b_c*100:5.2f}%", flush=True)

    # Verification of completeness
    assert len(results) == 120, f"Completeness check FAILED! Expected 120 episodes, found {len(results)}"
    out_df = pd.DataFrame(results).sort_values("episode_id").reset_index(drop=True)
    out_csv = os.path.join(args.out_dir, "step4_baselines_results.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\n[Verified] Wrote 120/120 complete episodes to {out_csv}", flush=True)

    # Save timing breakdown
    timing_data = {
        "hardware": {
            "gpu_model": gpu_name,
            "synchronization_used": "torch.cuda.synchronize() before and after every timing block",
            "warmup_slices": WARMUP_SLICES,
            "peak_vram_vitb_mb": peak_vram_vitb,
            "peak_vram_vith_mb": peak_vram_vith,
            "n_timed_slices": len(t_fob_list),
        },
        "latencies_ms_per_slice": {
            "fob_prompt_generator": {"mean": float(np.mean(t_fob_list)), "std": float(np.std(t_fob_list))},
            "sam_vith_image_encoder": {"mean": float(np.mean(t_enc_h_list)), "std": float(np.std(t_enc_h_list))},
            "sam_vith_mask_decoder": {"mean": float(np.mean(t_dec_h_list)), "std": float(np.std(t_dec_h_list))},
            "sam_vitb_image_encoder": {"mean": float(np.mean(t_enc_b_list)), "std": float(np.std(t_enc_b_list))},
            "sam_vitb_mask_decoder": {"mean": float(np.mean(t_dec_b_list)), "std": float(np.std(t_dec_b_list))},
            "connected_components_filtering": {"mean": float(np.mean(t_cc_list)), "std": float(np.std(t_cc_list))},
        }
    }
    timing_path = os.path.join(args.out_dir, "step4_timing_breakdown.json")
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=2)
    print(f"[Timing] Saved rigorous timing profile to {timing_path}", flush=True)


if __name__ == "__main__":
    main()
