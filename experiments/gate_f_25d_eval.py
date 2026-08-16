"""
Gate F: AdaFoB-2.5D Evaluation with Mask-Prompt Propagation
=============================================================
This script extends Gate D by propagating SAM's low-res mask logits from
slice z to slice z+1, giving SAM spatial context from adjacent slices.

Key insight: SAM already has a built-in mask_input API that accepts
(1, 256, 256) logits and adds them to image embeddings. We simply
"wire up" the output of one slice to the input of the next.

Methods evaluated:
  1. FoB Baseline (Np=10, no mask propagation)
  2. AdaFoB 2D (adaptive Np, no mask propagation)
  3. FoB + Mask Propagation (Np=10, with mask propagation)
  4. AdaFoB-2.5D (adaptive Np + mask propagation)  ← OUR METHOD

Designed for Kaggle T4x2 (split organs across GPUs).
"""

import os
import sys
import glob
import json
import numpy as np
import torch
import cv2
import argparse

from tqdm import tqdm
import SimpleITK as sitk

try:
    from medpy.metric.binary import hd95
except ImportError:
    os.system("pip install medpy")
    from medpy.metric.binary import hd95

REPO_DIR = "/kaggle/working/AdaFoB"
sys.path.insert(0, REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, "third_party", "FoB_SAM"))

from models.FoB import FewShotSeg
from models.allocator import PromptBudgetAllocator
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import get_label_names, get_folds
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry, SamPredictor

# Reuse eval.py functions
sys.path.insert(0, os.path.join(REPO_DIR, "experiments"))
from eval import predict_sam_from_points, sanitize_prompts


# ── Auto-detect paths ──────────────────────────────────────────────────────
def find_path(target_name, is_file=False):
    working_path = os.path.join("/kaggle/working", target_name)
    if os.path.exists(working_path):
        return working_path
    for pat in [f"/kaggle/input/**/{target_name}"]:
        candidates = glob.glob(pat, recursive=True)
        if candidates:
            return candidates[0]
    return working_path

NORMALIZED_DIR = find_path("sabs_CT_normalized")
CKPT_DIR = find_path("fob_checkpoints")
SAM_CKPT = find_path("sam_vit_h.pth", is_file=True)
GLOBAL_PARAMS_PATH = os.path.join(REPO_DIR, "results", "r3_global_params.json")

TEST_LABELS = [1, 2, 3, 6]  # Spleen, RK, LK, Liver
N_PART = 3
SUPP_IDX = 2
VALID_NPS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]


def load_fob_checkpoint(fold, ckpt_dir):
    candidates = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*/**/*.pth"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*.pth"), recursive=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No checkpoint found for fold {fold} in {ckpt_dir}")


def get_voxel_spacing(volume_idx):
    img_path = os.path.join(NORMALIZED_DIR, f"image_{volume_idx}.nii.gz")
    if not os.path.exists(img_path):
        img_path = os.path.join(NORMALIZED_DIR, f"image_{volume_idx}.nii")
    if os.path.exists(img_path):
        return sitk.ReadImage(img_path).GetSpacing()
    return (1.0, 1.0, 1.0)


def compute_volume_dice(pred, gt):
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = np.sum(pred * gt)
    total = np.sum(pred) + np.sum(gt)
    if total == 0:
        return 1.0
    return 2.0 * inter / total


def compute_volume_hd95(pred, gt, spacing):
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return 100.0
    try:
        return hd95(pred, gt, voxelspacing=spacing)
    except RuntimeError:
        return 100.0


def query_sam_with_propagation(predictor, pts_pos, pts_neg, prev_logits, alpha,
                                img_shape):
    """Call SAM with point prompts + optional mask propagation from previous slice."""
    all_pts = []
    all_lbl = []
    if len(pts_pos) > 0:
        all_pts.extend(pts_pos)
        all_lbl.extend([1] * len(pts_pos))
    if len(pts_neg) > 0:
        all_pts.extend(pts_neg)
        all_lbl.extend([0] * len(pts_neg))

    # Prepare mask_input from previous slice logits
    mask_input = None
    if prev_logits is not None:
        mask_input = prev_logits * alpha  # Decay to prevent error accumulation

    if len(all_pts) > 0:
        masks, scores, low_res_logits = predictor.predict(
            point_coords=np.array(all_pts),
            point_labels=np.array(all_lbl),
            multimask_output=True,
            mask_input=mask_input,
        )
        idx = 0  # FoB parity: always take index 0
        return (masks[idx] > 0.5).astype(np.uint8), low_res_logits[idx:idx+1]
    elif mask_input is not None:
        # No point prompts but we have mask from previous slice — use mask-only
        masks, scores, low_res_logits = predictor.predict(
            point_coords=np.zeros((1, 2)),  # dummy point
            point_labels=np.array([-1]),     # SAM ignores label -1
            multimask_output=True,
            mask_input=mask_input,
        )
        idx = 0
        return (masks[idx] > 0.5).astype(np.uint8), low_res_logits[idx:idx+1]
    else:
        H, W = img_shape[:2]
        return np.zeros((H, W), dtype=np.uint8), None


def evaluate_25d(gpu=0, target_organs=None, alpha=0.5, target_fold=None):
    """Main evaluation loop with 2.5D mask propagation."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if target_organs is None:
        target_organs = TEST_LABELS

    # Load global params
    global_params = {"nu": 0.02, "lam": 0.5, "a0": 0.6, "tau": 0.05}
    if os.path.exists(GLOBAL_PARAMS_PATH):
        with open(GLOBAL_PARAMS_PATH, "r") as f:
            global_params.update(json.load(f))
        print(f"Loaded Global Params: {global_params}")

    labels = get_label_names("SABS")
    data_dir = os.path.dirname(NORMALIZED_DIR)

    # Init SAM
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).eval().cuda()
    predictor = SamPredictor(sam)

    results_data = []

    fold_list = range(5) if target_fold is None else [target_fold]
    for eval_fold in fold_list:
        print(f"\n{'='*70}")
        print(f"GATE F: Fold {eval_fold} (GPU {gpu}, alpha={alpha})")
        print(f"{'='*70}")

        ckpt_path = load_fob_checkpoint(eval_fold, CKPT_DIR)

        class DummyArgs:
            pass
        args = DummyArgs()
        args.dataset = "SABS"
        args.max_points = 24

        model = FewShotSeg(args)
        model.cuda()
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        model.eval()

        data_config = {
            'data_dir': data_dir, 'dataset': 'SABS', 'n_shot': 1, 'n_way': 1, 'n_query': 1,
            'n_sv': 5000, 'max_iter': 3000, 'eval_fold': eval_fold, 'min_size': 200,
            'max_slices': 3, 'supp_idx': SUPP_IDX,
        }
        test_dataset = TestDataset(data_config)
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=True, drop_last=False
        )

        for label_val, label_name in labels.items():
            if label_name == 'BG' or label_val not in target_organs:
                continue

            print(f"\n  Testing: {label_name} (label={label_val})")
            support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=N_PART)
            test_dataset.label = label_val

            support_image = [support_sample['image'][[i]].float().cuda()
                             for i in range(support_sample['image'].shape[0])]
            support_fg_mask = [support_sample['label'][[i]].float().cuda()
                               for i in range(support_sample['image'].shape[0])]

            with torch.no_grad():
                for vi, sample in enumerate(test_loader):
                    vol_id_str = sample.get('id', [f"image_{vi}.nii.gz"])[0]
                    try:
                        vol_idx = int(vol_id_str.split("image_")[-1].split(".nii")[0])
                    except:
                        vol_idx = vi
                    spacing = get_voxel_spacing(vol_idx)

                    query_image = [sample['image'][j].float().cuda()
                                   for j in range(sample['image'].shape[0])]
                    query_label = sample['label'].long()
                    C_q = sample['image'].shape[1]

                    idx_ = np.linspace(0, C_q, N_PART + 1).astype('int')
                    
                    # ── Identify Chunk Boundaries (Q2) ────────────────────
                    is_boundary = np.zeros(C_q, dtype=bool)
                    boundary_indices = idx_[1:-1]
                    for b_idx in boundary_indices:
                        is_boundary[max(0, b_idx - 2) : min(C_q, b_idx + 3)] = True

                    # ── Accumulators ──────────────────────────────────────
                    # Method 1: FoB Baseline (Np=10, no propagation)
                    pred_baseline = np.zeros(query_label.shape[-3:], dtype=np.uint8)
                    # Method 2: AdaFoB 2D (adaptive Np, no propagation)
                    pred_adafob_2d = np.zeros(query_label.shape[-3:], dtype=np.uint8)
                    # Method 3: FoB + Mask Prop (Np=10, with propagation)
                    pred_fob_prop = np.zeros(query_label.shape[-3:], dtype=np.uint8)
                    # Method 4: AdaFoB-2.5D (adaptive Np + propagation)
                    pred_adafob_25d = np.zeros(query_label.shape[-3:], dtype=np.uint8)

                    # ── Forward pass: collect all slices' points first ────
                    slice_data = []  # [(global_z, uni_neg, ada_neg, pos, budget, img_t)]

                    for sub_chunk in range(N_PART):
                        support_image_s = [support_image[sub_chunk]]
                        support_fg_mask_s = [support_fg_mask[sub_chunk]]
                        query_image_s = query_image[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]
                        query_label_s = query_label[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]

                        for j in range(query_image_s.shape[0]):
                            global_z = idx_[sub_chunk] + j

                            # 1. Uniform FoB (no allocator)
                            model.allocator = None
                            try:
                                uni_neg_p, pos_p = model(
                                    [support_image_s], [support_fg_mask_s],
                                    [query_image_s[[j]]], query_label_s[[j]], None
                                )
                                uni_neg = np.array(uni_neg_p).reshape(-1, 2) if uni_neg_p is not None else np.zeros((0, 2))
                                pos_arr = np.array(pos_p).reshape(-1, 2) if pos_p is not None else np.zeros((0, 2))
                            except:
                                uni_neg, pos_arr = np.zeros((0, 2)), np.zeros((0, 2))

                            # 2. Adaptive FoB (with allocator)
                            allocator = PromptBudgetAllocator(max_points=24).cuda()
                            allocator.nu = global_params.get("nu", 0.02)
                            allocator.lam = global_params.get("lam", 0.5)
                            allocator.a0 = global_params.get("a0", 0.6)
                            allocator.tau = global_params.get("tau", 0.05)
                            model.allocator = allocator

                            original_allocate = model.allocator.allocate
                            def capturing_allocate(qry_img, qry_pred_coarse, spt_fg_proto,
                                                   supp_m, mdl, supp_fts):
                                a, contours, M_tilde = mdl.allocator.get_ambiguity_score(
                                    qry_img, qry_pred_coarse, spt_fg_proto, supp_m, mdl, supp_fts)
                                budget = mdl.allocator.compute_budget(a, contours, M_tilde)
                                mdl.allocator.last_budget = budget
                                r = mdl.allocator.get_scale_adaptive_offset(M_tilde)
                                pts = mdl.allocator.sample_placement(
                                    qry_img, M_tilde, contours, 24, r)  # Always 24 for sweep
                                return pts, budget
                            model.allocator.allocate = capturing_allocate

                            try:
                                ada_neg_p, _ = model(
                                    [support_image_s], [support_fg_mask_s],
                                    [query_image_s[[j]]], query_label_s[[j]], None
                                )
                                ada_neg = np.array(ada_neg_p).reshape(-1, 2) if ada_neg_p is not None else np.zeros((0, 2))
                            except:
                                ada_neg = np.zeros((0, 2))

                            slice_budget = getattr(model.allocator, 'last_budget', 10)

                            # Prepare SAM image
                            img_t = query_image_s[[j]][0].permute(1, 2, 0).cpu().numpy()
                            img_t = ((img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8) * 255).astype(np.uint8)

                            slice_data.append({
                                'global_z': global_z,
                                'uni_neg': uni_neg,
                                'ada_neg': ada_neg,
                                'pos': pos_arr,
                                'budget': slice_budget,
                                'img_t': img_t,
                            })

                    # ── SAM predictions ───────────────────────────────────
                    # Method 1 & 2: No propagation (independent slices)
                    for sd in slice_data:
                        predictor.set_image(sd['img_t'])
                        z = sd['global_z']
                        budget_np = min(VALID_NPS, key=lambda x: abs(x - sd['budget']))

                        # Method 1: Baseline (Np=10, uniform)
                        m1, _, _ = predictor.predict(
                            point_coords=np.concatenate([sd['pos'][:10], sd['uni_neg'][:10]], axis=0) if len(sd['pos']) > 0 else sd['uni_neg'][:10],
                            point_labels=np.concatenate([np.ones(min(10, len(sd['pos']))), np.zeros(min(10, len(sd['uni_neg'])))]),
                            multimask_output=True,
                        ) if (len(sd['pos']) + len(sd['uni_neg'])) > 0 else (np.zeros((3, 256, 256)),)
                        pred_baseline[z] = (m1[0] > 0.5).astype(np.uint8) if isinstance(m1, np.ndarray) and m1.ndim == 3 else np.zeros((256, 256), dtype=np.uint8)

                        # Method 2: AdaFoB 2D (adaptive Np, no prop)
                        mask_2d, logits_2d = query_sam_with_propagation(
                            predictor, sd['pos'], sd['ada_neg'][:budget_np],
                            None, alpha, sd['img_t'].shape
                        )
                        pred_adafob_2d[z] = mask_2d

                    # ── Method 3 & 4: Forward propagation ─────────────────
                    # Forward pass (z=0 → z=Z-1)
                    fwd_logits_fob = None
                    fwd_logits_ada = None
                    fwd_masks_fob = {}
                    fwd_masks_ada = {}
                    fwd_logits_fob_store = {}
                    fwd_logits_ada_store = {}
                    
                    fwd_prev_area_fob = 0
                    fwd_prev_area_ada = 0

                    for sd in slice_data:
                        predictor.set_image(sd['img_t'])
                        z = sd['global_z']
                        budget_np = min(VALID_NPS, key=lambda x: abs(x - sd['budget']))
                        
                        # Reset Condition 1: Confidence Drop (No positive points found)
                        if len(sd['pos']) == 0:
                            fwd_logits_fob = None
                            fwd_logits_ada = None

                        # Method 3: FoB + Prop (Np=10)
                        mask_fob, logits_fob = query_sam_with_propagation(
                            predictor, sd['pos'], sd['uni_neg'][:10],
                            fwd_logits_fob, alpha, sd['img_t'].shape
                        )
                        fwd_masks_fob[z] = mask_fob
                        fwd_logits_fob_store[z] = logits_fob
                        
                        # Reset Condition 2: Area Drop (FoB)
                        current_area_fob = np.sum(mask_fob)
                        if fwd_prev_area_fob > 0 and current_area_fob < 0.1 * fwd_prev_area_fob:
                            fwd_logits_fob = None
                        else:
                            fwd_logits_fob = logits_fob
                        fwd_prev_area_fob = current_area_fob

                        # Method 4: AdaFoB-2.5D (adaptive + prop)
                        mask_ada, logits_ada = query_sam_with_propagation(
                            predictor, sd['pos'], sd['ada_neg'][:budget_np],
                            fwd_logits_ada, alpha, sd['img_t'].shape
                        )
                        fwd_masks_ada[z] = mask_ada
                        fwd_logits_ada_store[z] = logits_ada
                        
                        # Reset Condition 2: Area Drop (AdaFoB)
                        current_area_ada = np.sum(mask_ada)
                        if fwd_prev_area_ada > 0 and current_area_ada < 0.1 * fwd_prev_area_ada:
                            fwd_logits_ada = None
                        else:
                            fwd_logits_ada = logits_ada
                        fwd_prev_area_ada = current_area_ada

                    # ── Backward pass (z=Z-1 → z=0) ──────────────────────
                    bwd_logits_fob = None
                    bwd_logits_ada = None
                    bwd_masks_fob = {}
                    bwd_masks_ada = {}
                    bwd_logits_fob_store = {}
                    bwd_logits_ada_store = {}
                    
                    bwd_prev_area_fob = 0
                    bwd_prev_area_ada = 0

                    for sd in reversed(slice_data):
                        predictor.set_image(sd['img_t'])
                        z = sd['global_z']
                        budget_np = min(VALID_NPS, key=lambda x: abs(x - sd['budget']))
                        
                        if len(sd['pos']) == 0:
                            bwd_logits_fob = None
                            bwd_logits_ada = None

                        mask_fob, logits_fob = query_sam_with_propagation(
                            predictor, sd['pos'], sd['uni_neg'][:10],
                            bwd_logits_fob, alpha, sd['img_t'].shape
                        )
                        bwd_masks_fob[z] = mask_fob
                        bwd_logits_fob_store[z] = logits_fob
                        
                        current_area_fob = np.sum(mask_fob)
                        if bwd_prev_area_fob > 0 and current_area_fob < 0.1 * bwd_prev_area_fob:
                            bwd_logits_fob = None
                        else:
                            bwd_logits_fob = logits_fob
                        bwd_prev_area_fob = current_area_fob

                        mask_ada, logits_ada = query_sam_with_propagation(
                            predictor, sd['pos'], sd['ada_neg'][:budget_np],
                            bwd_logits_ada, alpha, sd['img_t'].shape
                        )
                        bwd_masks_ada[z] = mask_ada
                        bwd_logits_ada_store[z] = logits_ada
                        
                        current_area_ada = np.sum(mask_ada)
                        if bwd_prev_area_ada > 0 and current_area_ada < 0.1 * bwd_prev_area_ada:
                            bwd_logits_ada = None
                        else:
                            bwd_logits_ada = logits_ada
                        bwd_prev_area_ada = current_area_ada

                    # ── Merge forward + backward ──────────────────────────
                    for sd in slice_data:
                        z = sd['global_z']
                        H_img, W_img = sd['img_t'].shape[:2]
                        
                        def merge_logits(f_log, b_log):
                            if f_log is not None and b_log is not None:
                                return (f_log + b_log) / 2.0
                            elif f_log is not None:
                                return f_log
                            elif b_log is not None:
                                return b_log
                            return None

                        # Method 3: FoB + Prop Merge
                        m_fob = np.zeros((H_img, W_img), dtype=np.uint8)
                        f_l_fob = fwd_logits_fob_store.get(z)
                        b_l_fob = bwd_logits_fob_store.get(z)
                        m_l_fob = merge_logits(f_l_fob, b_l_fob)
                        if m_l_fob is not None:
                            import cv2
                            m_fob = (cv2.resize(m_l_fob[0].astype(np.float32), (W_img, H_img)) > 0.0).astype(np.uint8)
                        pred_fob_prop[z] = m_fob

                        # Method 4: AdaFoB-2.5D Merge
                        m_ada = np.zeros((H_img, W_img), dtype=np.uint8)
                        f_l_ada = fwd_logits_ada_store.get(z)
                        b_l_ada = bwd_logits_ada_store.get(z)
                        m_l_ada = merge_logits(f_l_ada, b_l_ada)
                        if m_l_ada is not None:
                            import cv2
                            m_ada = (cv2.resize(m_l_ada[0].astype(np.float32), (W_img, H_img)) > 0.0).astype(np.uint8)
                        pred_adafob_25d[z] = m_ada

                    # ── Compute 3D metrics ────────────────────────────────
                    gt = (query_label.squeeze(0).cpu().numpy() > 0).astype(np.uint8)
                    
                    # Full Volume Masks
                    gt_core = gt[~is_boundary]
                    pred_base_core = pred_baseline[~is_boundary]
                    pred_a2d_core = pred_adafob_2d[~is_boundary]
                    pred_fobp_core = pred_fob_prop[~is_boundary]
                    pred_a25d_core = pred_adafob_25d[~is_boundary]

                    row = {
                        "fold": eval_fold,
                        "organ": label_name,
                        "vol_id": vol_id_str,
                        "alpha": alpha,
                        "spacing_x": spacing[0],
                        "spacing_y": spacing[1],
                        "spacing_z": spacing[2],
                        # Method 1: FoB Baseline
                        "dice_baseline": compute_volume_dice(pred_baseline, gt),
                        "hd95_baseline": compute_volume_hd95(pred_baseline, gt, spacing),
                        "dice_base_core": compute_volume_dice(pred_base_core, gt_core),
                        # Method 2: AdaFoB 2D
                        "dice_adafob_2d": compute_volume_dice(pred_adafob_2d, gt),
                        "hd95_adafob_2d": compute_volume_hd95(pred_adafob_2d, gt, spacing),
                        "dice_a2d_core": compute_volume_dice(pred_a2d_core, gt_core),
                        # Method 3: FoB + Mask Prop
                        "dice_fob_prop": compute_volume_dice(pred_fob_prop, gt),
                        "hd95_fob_prop": compute_volume_hd95(pred_fob_prop, gt, spacing),
                        "dice_fobp_core": compute_volume_dice(pred_fobp_core, gt_core),
                        # Method 4: AdaFoB-2.5D
                        "dice_adafob_25d": compute_volume_dice(pred_adafob_25d, gt),
                        "hd95_adafob_25d": compute_volume_hd95(pred_adafob_25d, gt, spacing),
                        "dice_a25d_core": compute_volume_dice(pred_a25d_core, gt_core),
                    }

                    results_data.append(row)
                    print(f"    Vol {vi}: Baseline={row['dice_baseline']*100:.2f}%, "
                          f"AdaFoB2D={row['dice_adafob_2d']*100:.2f}%, "
                          f"FoB+Prop={row['dice_fob_prop']*100:.2f}%, "
                          f"AdaFoB-2.5D={row['dice_adafob_25d']*100:.2f}%")

    # ── Save results ──────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(results_data)
    os.makedirs(os.path.join(REPO_DIR, "results"), exist_ok=True)
    out_csv = os.path.join(REPO_DIR, "results", f"gate_f_results_gpu{gpu}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("GATE F: 2.5D MASK PROPAGATION SUMMARY")
    print("=" * 80)
    for organ in df['organ'].unique():
        odf = df[df['organ'] == organ]
        print(f"\n--- {organ} (n={len(odf)} volumes) ---")
        print(f"  FoB Baseline (Np=10):     Dice={odf['dice_baseline'].mean()*100:.2f}%  "
              f"HD95={odf['hd95_baseline'].median():.1f}mm")
        print(f"  AdaFoB 2D:                Dice={odf['dice_adafob_2d'].mean()*100:.2f}%  "
              f"HD95={odf['hd95_adafob_2d'].median():.1f}mm")
        print(f"  FoB + Mask Prop:          Dice={odf['dice_fob_prop'].mean()*100:.2f}%  "
              f"HD95={odf['hd95_fob_prop'].median():.1f}mm")
        print(f"  AdaFoB-2.5D (Ours):       Dice={odf['dice_adafob_25d'].mean()*100:.2f}%  "
              f"HD95={odf['hd95_adafob_25d'].median():.1f}mm")

    print(f"\n{'='*80}")
    print("OVERALL MEAN")
    print(f"  FoB Baseline:       {df['dice_baseline'].mean()*100:.2f}%")
    print(f"  AdaFoB 2D:          {df['dice_adafob_2d'].mean()*100:.2f}%")
    print(f"  FoB + Mask Prop:    {df['dice_fob_prop'].mean()*100:.2f}%")
    print(f"  AdaFoB-2.5D (Ours): {df['dice_adafob_25d'].mean()*100:.2f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--organs', type=int, nargs='+', default=None,
                        help='Organ IDs (default: split by GPU)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Decay factor for mask propagation logits')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to run (0-4). Runs all if None.')
    args = parser.parse_args()

    gpu_id = int(args.gpu)
    if args.organs is None:
        if gpu_id == 0:
            organs = [1, 2]   # Spleen, RK
        else:
            organs = [3, 6]   # LK, Liver
    else:
        organs = args.organs

    evaluate_25d(gpu=gpu_id, target_organs=organs, alpha=args.alpha, target_fold=args.fold)
