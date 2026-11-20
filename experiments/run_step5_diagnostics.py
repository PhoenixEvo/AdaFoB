"""
AdaFoB Research Project — Step 5 Diagnostics Runner
Symmetric, rigorous execution of:
  Item B.2: Prompt Quality & Contamination Audit
  Item B.3: Candidate-Pool Restricted Oracle Ceiling (Prefix & Greedy)
  Item B.4: Budget Sensitivity {0, 2, 4, 6, 10} + 5 Matched Random Controls
  Item B.5: Synthesis Decision Table & Statistical Contrasts

Supports multi-GPU execution (--gpu, --folds) for 2x T4 parallel speedup on Kaggle,
as well as single-GPU execution and partition merging.
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
from scipy import ndimage
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
from analysis.gate_a.candidate_records import CandidateRunLog

TEST_LABELS = [1, 2, 3, 6]  # SPLEEN, RK, LK, LIVER
SUPP_IDX = 2
N_PART = 3
RANDOM_SEEDS = [42, 43, 44, 45, 46]


def find_file(pattern_list):
    for p in pattern_list:
        if os.path.exists(p):
            return p
        matches = glob.glob(p, recursive=True)
        if matches:
            return sorted(matches)[0]
    return None


def ensure_checkpoints(ckpt_dir, sam_b_path):
    os.makedirs(ckpt_dir, exist_ok=True)
    for fold in range(5):
        cand = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*/**/*.pth"), recursive=True)
        if not cand:
            cand = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*.pth"), recursive=True)
        if not cand:
            print(f"Downloading FoB checkpoints to {ckpt_dir}...", flush=True)
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="PrimeBo1/FoB_SAM", local_dir=ckpt_dir, allow_patterns=["*SABS*"])
                import zipfile
                zips = glob.glob(os.path.join(ckpt_dir, "**/*.zip"), recursive=True)
                for z in zips:
                    with zipfile.ZipFile(z, "r") as zip_ref:
                        zip_ref.extractall(ckpt_dir)
            except Exception as e:
                print(f"HuggingFace download failed: {e}", flush=True)
            break

    if not os.path.exists(sam_b_path):
        os.makedirs(os.path.dirname(sam_b_path), exist_ok=True)
        print(f"Downloading SAM ViT-B checkpoint to {sam_b_path}...", flush=True)
        import urllib.request
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", sam_b_path)


def filter_volume_cc(vol_3d, pos_by_z):
    vol_cc = np.zeros_like(vol_3d)
    for z in range(vol_3d.shape[0]):
        pts = pos_by_z.get(z, None)
        vol_cc[z] = filter_connected_components(vol_3d[z], pts)
    return vol_cc


def compute_volume_metrics(pred, gt, spacing):
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


def predict_singlemask(predictor, pos_pts, neg_pts, img_shape, audit=None, selected_ids=None, arm=None):
    pts = []
    lbls = []
    if pos_pts is not None and len(pos_pts) > 0:
        pts.append(pos_pts)
        lbls.append(np.ones(len(pos_pts), dtype=np.int32))
    if neg_pts is not None and len(neg_pts) > 0:
        pts.append(neg_pts)
        lbls.append(np.zeros(len(neg_pts), dtype=np.int32))

    event = audit.decoder_event(selected_ids, arm, evaluated=bool(len(pts))) if audit is not None else None
    if len(pts) == 0:
        return np.zeros(img_shape[:2], dtype=np.uint8)

    all_pts = np.concatenate(pts, axis=0)
    all_lbls = np.concatenate(lbls, axis=0)

    try:
        masks, _, _ = predictor.predict(
            point_coords=all_pts,
            point_labels=all_lbls,
            multimask_output=False,
        )
        if event is not None: event["outcome"] = "success"
        return (masks[0] > 0).astype(np.uint8)
    except Exception as e:
        if event is not None: event["outcome"] = "exception:" + type(e).__name__
        print(f"Warning: SAM predict_singlemask exception: {e}", flush=True)
        return np.zeros(img_shape[:2], dtype=np.uint8)


def compute_slice_dice(pred, gt):
    p_sum = pred.sum()
    g_sum = gt.sum()
    if p_sum == 0 and g_sum == 0:
        return 1.0
    if p_sum == 0 or g_sum == 0:
        return 0.0
    return float(2.0 * np.logical_and(pred, gt).sum() / (p_sum + g_sum))


def run_diagnostics(folds, gpu_id, out_dir, max_episodes=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Worker GPU {gpu_id}] Running Folds {folds} on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    manifest_path = os.path.join(REPO_DIR, "protocol", "paired_episode_manifest.csv")
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
    data_root = os.path.dirname(norm_dir)

    ckpt_dir = find_file([
        "/tmp/fob_checkpoints",
        "/kaggle/working/fob_checkpoints",
        "/kaggle/input/baseline_fob/exps_train_on_SABS_FSMIS_FoB",
        "/kaggle/input/adafob-checkpoints",
        os.path.join(REPO_DIR, "checkpoints", "fob_checkpoints")
    ]) or "/tmp/fob_checkpoints"

    sam_b_path = find_file([
        "/tmp/checkpoints/sam_vit_b_01ec64.pth",
        "/kaggle/working/checkpoints/sam_vit_b_01ec64.pth",
        "/kaggle/working/sam_vit_b_01ec64.pth",
        "/kaggle/input/checkpoints/sam_vit_b_01ec64.pth",
        "D:/results/checkpoints/sam_vit_b_01ec64.pth"
    ]) or "/tmp/checkpoints/sam_vit_b_01ec64.pth"

    ensure_checkpoints(ckpt_dir, sam_b_path)

    # Load SAM ViT-B
    print(f"[SAM] Loading ViT-B from {sam_b_path}...", flush=True)
    sam_b = sam_model_registry["vit_b"](checkpoint=sam_b_path).eval().to(device)
    predictor_b = SamPredictor(sam_b)

    labels = get_label_names("SABS")
    os.makedirs(out_dir, exist_ok=True)
    if glob.glob(os.path.join(out_dir, "step5_diagnostics*.csv")):
        raise FileExistsError("Use a new output directory; historical results are preserved")
    candidate_log = CandidateRunLog(os.path.join(out_dir, "candidate_records_f" + "_".join(map(str, folds)) + ".jsonl"))
    
    results = []
    prompt_audit_records = []

    for eval_fold in folds:
        print(f"\n{'='*70}\nExecuting Step 5 Diagnostics: Fold {eval_fold}\n{'='*70}", flush=True)

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
                vol_id_str = sample['id'][0].replace('\\', '/')
                pat_id = vol_id_str.split('/')[-1].split('.nii')[0]

                row_match = manifest_df[(manifest_df['fold'] == eval_fold) &
                                        (manifest_df['organ'] == label_name) &
                                        (manifest_df['vol_id'].str.contains(pat_id, regex=False))]
                if len(row_match) == 0:
                    continue
                ep_row = row_match.iloc[0]
                ep_id = int(ep_row['episode_id'])
                spacing = (float(ep_row['spacing_x']), float(ep_row['spacing_y']), float(ep_row['spacing_z']))
                pixel_spacing_mm = float(ep_row['spacing_x'])

                query_images = sample['image'].float() # Keep on CPU until slice indexing to save VRAM
                query_labels = sample['label'].long()
                C_q = query_images.shape[1]
                H_q, W_q = query_images.shape[-2], query_images.shape[-1]
                gt_vol_np = (query_labels[0].squeeze(1).numpy() > 0).astype(np.uint8)

                # Initialize 3D prediction volumes
                vols = {
                    'k0': np.zeros((C_q, H_q, W_q), dtype=np.uint8),
                    'k2': np.zeros((C_q, H_q, W_q), dtype=np.uint8),
                    'k4': np.zeros((C_q, H_q, W_q), dtype=np.uint8),
                    'k6': np.zeros((C_q, H_q, W_q), dtype=np.uint8),
                    'k10': np.zeros((C_q, H_q, W_q), dtype=np.uint8),
                    'prefix_oracle': np.zeros((C_q, H_q, W_q), dtype=np.uint8),
                    'greedy_oracle': np.zeros((C_q, H_q, W_q), dtype=np.uint8),
                }
                for k in [2, 4, 6]:
                    for s in RANDOM_SEEDS:
                        vols[f'rand_k{k}_s{s}'] = np.zeros((C_q, H_q, W_q), dtype=np.uint8)

                # Episode prompt audit counters and prompt cache
                ep_pos_total, ep_pos_outside = 0, 0
                ep_neg_total, ep_neg_inside = 0, 0
                ep_neg_distances_mm = []
                pos_points_by_z = {}

                idx_ = np.linspace(0, C_q, N_PART + 1).astype(int)

                with torch.no_grad():
                    for sub_chunk in range(N_PART):
                        support_image_s = [support_image[sub_chunk]]
                        support_fg_mask_s = [support_fg_mask[sub_chunk]]
                        query_image_chunk = query_images[0, idx_[sub_chunk]:idx_[sub_chunk+1]]
                        chunk_len = query_image_chunk.shape[0]

                        if chunk_len == 0:
                            continue

                        for j in range(chunk_len):
                            global_z = idx_[sub_chunk] + j
                            q_img_slice = query_image_chunk[[j]].to(device)
                            gt_slice = gt_vol_np[global_z]
                            dummy_lbl = torch.ones_like(q_img_slice[:, 0]).long()

                            # FoB prompt generation
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
                            candidate_log.begin_slice(ep_id, global_z, neg_arr, pos_arr, (H_q, W_q))
                            pos_points_by_z[global_z] = pos_arr

                            # Distance transform for spatial surface distance in mm
                            gt_slice_bool = (gt_slice > 0)
                            if gt_slice_bool.sum() > 0:
                                dt_outside = ndimage.distance_transform_edt(~gt_slice_bool) * pixel_spacing_mm
                            else:
                                dt_outside = np.zeros_like(gt_slice, dtype=float)

                            # Item B.2: Prompt Audit for this slice
                            for p in pos_arr:
                                x, y = int(round(p[0])), int(round(p[1]))
                                if 0 <= y < H_q and 0 <= x < W_q:
                                    ep_pos_total += 1
                                    if gt_slice[y, x] == 0:
                                        ep_pos_outside += 1

                            for p in neg_arr:
                                x, y = int(round(p[0])), int(round(p[1]))
                                if 0 <= y < H_q and 0 <= x < W_q:
                                    ep_neg_total += 1
                                    if gt_slice[y, x] == 1:
                                        ep_neg_inside += 1
                                        ep_neg_distances_mm.append(-1.0)  # Contaminated inside
                                    else:
                                        dist_val = float(dt_outside[y, x])
                                        ep_neg_distances_mm.append(dist_val)

                            # Set image in SAM ViT-B
                            img_t = q_img_slice[0].permute(1, 2, 0).cpu().numpy()
                            img_norm = ((img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8) * 255.0).astype(np.uint8)
                            img_rgb = np.repeat(img_norm, 3, axis=-1) if img_norm.shape[-1] == 1 else img_norm
                            predictor_b.set_image(img_rgb)

                            # Fixed budgets k in {0, 2, 4, 6, 10}
                            m_k0 = predict_singlemask(predictor_b, pos_arr, neg_arr[:0], (H_q, W_q), audit=candidate_log, selected_ids=list(range(min(0, len(neg_arr)))), arm="prefix_0")
                            m_k2 = predict_singlemask(predictor_b, pos_arr, neg_arr[:2], (H_q, W_q), audit=candidate_log, selected_ids=list(range(min(2, len(neg_arr)))), arm="prefix_2")
                            m_k4 = predict_singlemask(predictor_b, pos_arr, neg_arr[:4], (H_q, W_q), audit=candidate_log, selected_ids=list(range(min(4, len(neg_arr)))), arm="prefix_4")
                            m_k6 = predict_singlemask(predictor_b, pos_arr, neg_arr[:6], (H_q, W_q), audit=candidate_log, selected_ids=list(range(min(6, len(neg_arr)))), arm="prefix_6")
                            m_k10 = predict_singlemask(predictor_b, pos_arr, neg_arr[:10], (H_q, W_q), audit=candidate_log, selected_ids=list(range(min(10, len(neg_arr)))), arm="prefix_10")

                            vols['k0'][global_z] = m_k0
                            vols['k2'][global_z] = m_k2
                            vols['k4'][global_z] = m_k4
                            vols['k6'][global_z] = m_k6
                            vols['k10'][global_z] = m_k10

                            # Matched Random controls (5 seeds)
                            n_pool = len(neg_arr)
                            for k in [2, 4, 6]:
                                for s in RANDOM_SEEDS:
                                    if n_pool > 0:
                                        rng = np.random.RandomState(s + global_z * 100)
                                        chosen_idx = rng.choice(n_pool, size=min(k, n_pool), replace=False)
                                        m_rand = predict_singlemask(predictor_b, pos_arr, neg_arr[chosen_idx], (H_q, W_q), audit=candidate_log, selected_ids=chosen_idx.tolist(), arm=f"random_k{k}_s{s}")
                                    else:
                                        m_rand = m_k0
                                    vols[f'rand_k{k}_s{s}'][global_z] = m_rand

                            # Prefix budget oracle (best k in 0..10 for this slice)
                            best_dice = compute_slice_dice(m_k0, gt_slice)
                            best_mask_prefix = m_k0
                            best_prefix_ids = []
                            for k_cand in range(1, 11):
                                m_cand = predict_singlemask(predictor_b, pos_arr, neg_arr[:k_cand], (H_q, W_q), audit=candidate_log, selected_ids=list(range(min(k_cand, len(neg_arr)))), arm="GT_prefix_trial")
                                d_cand = compute_slice_dice(m_cand, gt_slice)
                                if d_cand > best_dice:
                                    best_dice = d_cand
                                    best_mask_prefix = m_cand
                                    best_prefix_ids = list(range(min(k_cand, len(neg_arr))))
                            vols['prefix_oracle'][global_z] = best_mask_prefix

                            # Greedy subset oracle (greedily adds points from candidate pool that improve slice Dice)
                            current_selected = []
                            current_best_dice = compute_slice_dice(m_k0, gt_slice)
                            current_best_mask = m_k0
                            remaining_indices = list(range(n_pool))

                            while remaining_indices:
                                step_best_dice = current_best_dice
                                step_best_idx = None
                                step_best_mask = None
                                for idx_cand in remaining_indices:
                                    trial_pts = neg_arr[current_selected + [idx_cand]]
                                    m_trial = predict_singlemask(predictor_b, pos_arr, trial_pts, (H_q, W_q), audit=candidate_log, selected_ids=current_selected + [idx_cand], arm="GT_greedy_trial")
                                    d_trial = compute_slice_dice(m_trial, gt_slice)
                                    if d_trial > step_best_dice + 1e-4:
                                        step_best_dice = d_trial
                                        step_best_idx = idx_cand
                                        step_best_mask = m_trial
                                if step_best_idx is not None:
                                    current_selected.append(step_best_idx)
                                    remaining_indices.remove(step_best_idx)
                                    current_best_dice = step_best_dice
                                    current_best_mask = step_best_mask
                                else:
                                    break
                            vols['greedy_oracle'][global_z] = current_best_mask
                            candidate_log.final_choice('GT_prefix_final', best_prefix_ids)
                            candidate_log.final_choice('GT_greedy_final', current_selected)
                            candidate_log.finish_slice()

                # Compute 3D Metrics across all arms
                row_metrics = {
                    'episode_id': ep_id, 'fold': eval_fold, 'organ': label_name,
                    'patient_id': pat_id, 'vol_id': vol_id_str,
                    'spacing_x': spacing[0], 'spacing_y': spacing[1], 'spacing_z': spacing[2],
                    'gt_voxels': int(gt_vol_np.sum())
                }

                # Evaluate volume metrics for each arm (Raw and CC)
                for arm_key in ['k0', 'k2', 'k4', 'k6', 'k10', 'prefix_oracle', 'greedy_oracle']:
                    vol_raw = vols[arm_key]
                    vol_cc = filter_volume_cc(vol_raw, pos_points_by_z)
                    d_raw, h_raw = compute_volume_metrics(vol_raw, gt_vol_np, spacing)
                    d_cc, h_cc = compute_volume_metrics(vol_cc, gt_vol_np, spacing)
                    row_metrics[f'dice_{arm_key}_raw'] = d_raw
                    row_metrics[f'hd95_{arm_key}_raw'] = h_raw
                    row_metrics[f'dice_{arm_key}_cc'] = d_cc
                    row_metrics[f'hd95_{arm_key}_cc'] = h_cc

                # Evaluate random controls
                for k in [2, 4, 6]:
                    rand_dices_raw, rand_dices_cc = [], []
                    rand_hds_raw, rand_hds_cc = [], []
                    for s in RANDOM_SEEDS:
                        vol_raw = vols[f'rand_k{k}_s{s}']
                        vol_cc = filter_volume_cc(vol_raw, pos_points_by_z)
                        d_r, h_r = compute_volume_metrics(vol_raw, gt_vol_np, spacing)
                        d_c, h_c = compute_volume_metrics(vol_cc, gt_vol_np, spacing)
                        rand_dices_raw.append(d_r)
                        rand_dices_cc.append(d_c)
                        rand_hds_raw.append(h_r)
                        rand_hds_cc.append(h_c)
                        row_metrics[f'dice_rand_k{k}_s{s}_raw'] = d_r
                        row_metrics[f'dice_rand_k{k}_s{s}_cc'] = d_c

                    row_metrics[f'dice_rand_k{k}_mean_raw'] = float(np.mean(rand_dices_raw))
                    row_metrics[f'dice_rand_k{k}_std_raw'] = float(np.std(rand_dices_raw))
                    row_metrics[f'dice_rand_k{k}_mean_cc'] = float(np.mean(rand_dices_cc))
                    row_metrics[f'dice_rand_k{k}_std_cc'] = float(np.std(rand_dices_cc))
                    row_metrics[f'hd95_rand_k{k}_mean_cc'] = float(np.mean(rand_hds_cc))

                # Prompt audit stats for this episode
                dist_arr = np.array(ep_neg_distances_mm)
                neg_inside_cnt = int((dist_arr < 0).sum())
                d_0_5_cnt = int(np.logical_and(dist_arr >= 0, dist_arr < 5.0).sum())
                d_5_10_cnt = int(np.logical_and(dist_arr >= 5.0, dist_arr < 10.0).sum())
                d_10_20_cnt = int(np.logical_and(dist_arr >= 10.0, dist_arr < 20.0).sum())
                d_gt20_cnt = int((dist_arr >= 20.0).sum())

                row_metrics['pos_points_total'] = ep_pos_total
                row_metrics['pos_points_outside'] = ep_pos_outside
                row_metrics['pos_outside_frac'] = float(ep_pos_outside / max(ep_pos_total, 1))
                row_metrics['neg_points_total'] = ep_neg_total
                row_metrics['neg_points_inside'] = ep_neg_inside
                row_metrics['neg_contamination_frac'] = float(ep_neg_inside / max(ep_neg_total, 1))
                row_metrics['neg_dist_inside_cnt'] = neg_inside_cnt
                row_metrics['neg_dist_0_5mm_cnt'] = d_0_5_cnt
                row_metrics['neg_dist_5_10mm_cnt'] = d_5_10_cnt
                row_metrics['neg_dist_10_20mm_cnt'] = d_10_20_cnt
                row_metrics['neg_dist_gt20mm_cnt'] = d_gt20_cnt

                results.append(row_metrics)
                prompt_audit_records.append({
                    'episode_id': ep_id, 'fold': eval_fold, 'organ': label_name, 'patient_id': pat_id,
                    'pos_points_total': ep_pos_total, 'pos_points_outside': ep_pos_outside,
                    'neg_points_total': ep_neg_total, 'neg_points_inside': ep_neg_inside,
                    'distances_mm': [float(d) for d in ep_neg_distances_mm]
                })

                print(f"Ep {ep_id:03d} ({eval_fold}, {label_name}, {pat_id}): "
                      f"k10_cc={row_metrics['dice_k10_cc']*100:.1f}%, "
                      f"k4_cc={row_metrics['dice_k4_cc']*100:.1f}%, "
                      f"greedy_oracle_cc={row_metrics['dice_greedy_oracle_cc']*100:.1f}%, "
                      f"neg_contam={row_metrics['neg_contamination_frac']*100:.1f}%", flush=True)

                if max_episodes is not None and len(results) >= max_episodes:
                    break
            if max_episodes is not None and len(results) >= max_episodes:
                break
        if max_episodes is not None and len(results) >= max_episodes:
            break

    df_out = pd.DataFrame(results)
    out_csv = os.path.join(out_dir, f"step5_diagnostics_f{'_'.join(map(str, folds))}.csv")
    df_out.to_csv(out_csv, index=False)

    out_audit_json = os.path.join(out_dir, f"prompt_audit_f{'_'.join(map(str, folds))}.json")
    with open(out_audit_json, "w") as f:
        json.dump(prompt_audit_records, f)

    if len(folds) == 5:
        generate_summary_report(df_out, prompt_audit_records, out_dir)

    print(f"\n[Worker GPU {gpu_id}] Saved results to {out_csv} and audit to {out_audit_json}", flush=True)


def merge_partitions(dir1, dir2, out_dir):
    print(f"Merging diagnostic partitions from {dir1} and {dir2} into {out_dir}...", flush=True)
    csv1 = glob.glob(os.path.join(dir1, "*.csv"))[0]
    csv2 = glob.glob(os.path.join(dir2, "*.csv"))[0]
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    df_merged = pd.concat([df1, df2], ignore_index=True).sort_values("episode_id").reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    merged_csv = os.path.join(out_dir, "step5_diagnostics_results.csv")
    df_merged.to_csv(merged_csv, index=False)
    print(f"Merged CSV contains {len(df_merged)} episodes. Saved to {merged_csv}", flush=True)

    # Merge audit JSONs
    json1 = glob.glob(os.path.join(dir1, "*.json"))[0]
    json2 = glob.glob(os.path.join(dir2, "*.json"))[0]
    with open(json1) as f:
        a1 = json.load(f)
    with open(json2) as f:
        a2 = json.load(f)
    merged_audit = sorted(a1 + a2, key=lambda x: x['episode_id'])
    merged_audit_path = os.path.join(out_dir, "step5_prompt_audit_records.json")
    with open(merged_audit_path, "w") as f:
        json.dump(merged_audit, f)
    print(f"Merged audit contains {len(merged_audit)} episodes. Saved to {merged_audit_path}", flush=True)

    # Compute comprehensive summary and statistical tests
    generate_summary_report(df_merged, merged_audit, out_dir)


def generate_summary_report(df, audit_records, out_dir):
    print("Computing statistical summary, bootstrap CIs, and synthesis decision table...", flush=True)
    summary = {"cohort_size": len(df), "budget_sensitivity": {}, "oracle_analysis": {}, "prompt_audit": {}, "decision_table": {}}

    # 1. Budget Sensitivity {0, 2, 4, 6, 10}
    for k_name in ['k0', 'k2', 'k4', 'k6', 'k10']:
        summary["budget_sensitivity"][k_name] = {
            "dice_raw_mean": float(df[f'dice_{k_name}_raw'].mean() * 100.0),
            "dice_raw_median": float(df[f'dice_{k_name}_raw'].median() * 100.0),
            "dice_cc_mean": float(df[f'dice_{k_name}_cc'].mean() * 100.0),
            "dice_cc_median": float(df[f'dice_{k_name}_cc'].median() * 100.0),
            "hd95_cc_mean": float(df[f'hd95_{k_name}_cc'].mean()),
            "hd95_cc_median": float(df[f'hd95_{k_name}_cc'].median()),
            "hd95_cc_p90": float(df[f'hd95_{k_name}_cc'].quantile(0.90)),
        }

    # Random controls
    for k in [2, 4, 6]:
        summary["budget_sensitivity"][f'rand_k{k}'] = {
            "dice_raw_mean": float(df[f'dice_rand_k{k}_mean_raw'].mean() * 100.0),
            "dice_cc_mean": float(df[f'dice_rand_k{k}_mean_cc'].mean() * 100.0),
            "dice_cc_median": float(df[[f'dice_rand_k{k}_s{s}_cc' for s in RANDOM_SEEDS]].values.flatten().mean() * 100.0),
            "hd95_cc_mean": float(df[f'hd95_rand_k{k}_mean_cc'].mean()),
        }

    # 2. Oracle Analysis
    for o_name in ['prefix_oracle', 'greedy_oracle']:
        summary["oracle_analysis"][o_name] = {
            "dice_raw_mean": float(df[f'dice_{o_name}_raw'].mean() * 100.0),
            "dice_raw_median": float(df[f'dice_{o_name}_raw'].median() * 100.0),
            "dice_cc_mean": float(df[f'dice_{o_name}_cc'].mean() * 100.0),
            "dice_cc_median": float(df[f'dice_{o_name}_cc'].median() * 100.0),
            "hd95_cc_mean": float(df[f'hd95_{o_name}_cc'].mean()),
            "hd95_cc_median": float(df[f'hd95_{o_name}_cc'].median()),
            "hd95_cc_p90": float(df[f'hd95_{o_name}_cc'].quantile(0.90)),
            "gain_over_k10_cc_mean": float((df[f'dice_{o_name}_cc'] - df['dice_k10_cc']).mean() * 100.0),
            "gain_over_k10_cc_median": float((df[f'dice_{o_name}_cc'] - df['dice_k10_cc']).median() * 100.0),
        }

    # Patient-cluster bootstrap for Oracle Gain over Fixed-10 CC
    patients = df['patient_id'].unique()
    K = len(patients)
    B = 5000
    rng = np.random.RandomState(42)
    boot_gains = []
    diff_series = (df['dice_greedy_oracle_cc'] - df['dice_k10_cc']) * 100.0

    for _ in range(B):
        sampled_pats = rng.choice(patients, size=K, replace=True)
        sample_indices = []
        for p in sampled_pats:
            sample_indices.extend(df.index[df['patient_id'] == p].tolist())
        boot_gains.append(float(diff_series.iloc[sample_indices].mean()))

    ci_low = float(np.percentile(boot_gains, 2.5))
    ci_high = float(np.percentile(boot_gains, 97.5))
    summary["oracle_analysis"]["greedy_oracle"]["bootstrap_ci_95"] = [ci_low, ci_high]

    # 3. Prompt Quality Audit
    total_pos = int(df['pos_points_total'].sum())
    total_pos_out = int(df['pos_points_outside'].sum())
    total_neg = int(df['neg_points_total'].sum())
    total_neg_in = int(df['neg_points_inside'].sum())

    summary["prompt_audit"]["overall"] = {
        "pos_points_total": total_pos,
        "pos_points_outside": total_pos_out,
        "pos_outside_pct": float(total_pos_out / max(total_pos, 1) * 100.0),
        "neg_points_total": total_neg,
        "neg_points_inside": total_neg_in,
        "neg_contamination_pct": float(total_neg_in / max(total_neg, 1) * 100.0),
        "dist_inside_cnt": int(df['neg_dist_inside_cnt'].sum()),
        "dist_0_5mm_cnt": int(df['neg_dist_0_5mm_cnt'].sum()),
        "dist_5_10mm_cnt": int(df['neg_dist_5_10mm_cnt'].sum()),
        "dist_10_20mm_cnt": int(df['neg_dist_10_20mm_cnt'].sum()),
        "dist_gt20mm_cnt": int(df['neg_dist_gt20mm_cnt'].sum()),
    }

    # Per-organ breakdown
    summary["prompt_audit"]["per_organ"] = {}
    for organ in ["LIVER", "RK", "LK", "SPLEEN"]:
        sub = df[df['organ'] == organ]
        sub_neg_tot = int(sub['neg_points_total'].sum())
        sub_neg_in = int(sub['neg_points_inside'].sum())
        sub_pos_tot = int(sub['pos_points_total'].sum())
        sub_pos_out = int(sub['pos_points_outside'].sum())
        summary["prompt_audit"]["per_organ"][organ] = {
            "pos_outside_pct": float(sub_pos_out / max(sub_pos_tot, 1) * 100.0),
            "neg_contamination_pct": float(sub_neg_in / max(sub_neg_tot, 1) * 100.0),
            "dist_inside_cnt": int(sub['neg_dist_inside_cnt'].sum()),
            "dist_0_5mm_cnt": int(sub['neg_dist_0_5mm_cnt'].sum()),
            "dist_5_10mm_cnt": int(sub['neg_dist_5_10mm_cnt'].sum()),
            "dist_10_20mm_cnt": int(sub['neg_dist_10_20mm_cnt'].sum()),
            "dist_gt20mm_cnt": int(sub['neg_dist_gt20mm_cnt'].sum()),
        }

    # High contamination cases (>20%)
    high_contam_df = df[df['neg_contamination_frac'] > 0.20][['episode_id', 'fold', 'organ', 'patient_id', 'neg_contamination_frac']]
    summary["prompt_audit"]["high_contamination_cases"] = high_contam_df.to_dict(orient="records")

    # 4. Synthesis Decision Table
    greedy_gain = summary["oracle_analysis"]["greedy_oracle"]["gain_over_k10_cc_mean"]
    best_fixed_k = 'k4'  # To be dynamically picked
    best_k_mean = max([summary["budget_sensitivity"][k]["dice_cc_mean"] for k in ['k0', 'k2', 'k4', 'k6', 'k10']])
    for k in ['k0', 'k2', 'k4', 'k6', 'k10']:
        if summary["budget_sensitivity"][k]["dice_cc_mean"] == best_k_mean:
            best_fixed_k = k
    budget_gain = best_k_mean - summary["budget_sensitivity"]["k10"]["dice_cc_mean"]
    cc_gain = float((df['dice_k10_cc'] - df['dice_k10_raw']).mean() * 100.0)
    token_gain = 3.64  # Pre-established: ViT-H R3 vs ViT-B R1

    summary["decision_table"] = {
        "levers": [
            {"lever": "Better Prompt Selection (Greedy Oracle vs Fixed-10)", "effect_size_dice_pp": float(greedy_gain), "rank": 0},
            {"lever": f"Reducing Budget ({best_fixed_k.upper()} vs Fixed-10)", "effect_size_dice_pp": float(budget_gain), "rank": 0},
            {"lever": "Connected-Component Post-Processing (CC vs Raw)", "effect_size_dice_pp": float(cc_gain), "rank": 0},
            {"lever": "Token Selection / Backbone Calibration (mask[0] vs SingleMask)", "effect_size_dice_pp": float(token_gain), "rank": 0},
        ]
    }
    summary["decision_table"]["levers"] = sorted(summary["decision_table"]["levers"], key=lambda x: x["effect_size_dice_pp"], reverse=True)
    for r, item in enumerate(summary["decision_table"]["levers"]):
        item["rank"] = r + 1

    # Determination of Gap
    if greedy_gain < 1.0:
        determination = "NO"
        reason = f"Greedy oracle ceiling gain ({greedy_gain:.2f} pp) is below the 1.0 pp threshold. Candidate pool cannot solve the segmentation deficit."
    elif greedy_gain > 3.0:
        determination = "YES"
        reason = f"Greedy oracle ceiling gain ({greedy_gain:.2f} pp, 95% CI [{ci_low:.2f}, {ci_high:.2f}]) exceeds the 3.0 pp threshold. Prompt selection has a substantial target."
    else:
        determination = "INCONCLUSIVE"
        reason = f"Greedy oracle ceiling gain ({greedy_gain:.2f} pp, 95% CI [{ci_low:.2f}, {ci_high:.2f}]) falls in the intermediate [1.0, 3.0] pp range."

    summary["conclusion"] = {
        "gap_exists": determination,
        "reason": reason,
        "oracle_dice_mean": summary["oracle_analysis"]["greedy_oracle"]["dice_cc_mean"],
        "fixed10_dice_mean": summary["budget_sensitivity"]["k10"]["dice_cc_mean"],
        "gain_mean": greedy_gain,
        "bootstrap_ci_95": [ci_low, ci_high]
    }

    summary_json_path = os.path.join(out_dir, "step5_diagnosis_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary report written to {summary_json_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Step 5 Diagnostics Execution")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Folds to evaluate")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU ID")
    parser.add_argument("--out_dir", type=str, default="results/step5_diagnostics", help="Output directory")
    parser.add_argument("--max_episodes", type=int, default=None, help="Max episodes to evaluate for smoke testing")
    parser.add_argument("--merge", action="store_true", help="Merge existing partition runs")
    parser.add_argument("--dir1", type=str, default="results/step5_diagnostics/gpu0")
    parser.add_argument("--dir2", type=str, default="results/step5_diagnostics/gpu1")
    args = parser.parse_args()

    if args.merge:
        merge_partitions(args.dir1, args.dir2, args.out_dir)
    else:
        run_diagnostics(args.folds, args.gpu, args.out_dir, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
