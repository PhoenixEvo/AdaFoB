"""
Gate D Notebook: Full AdaFoB Matrix Evaluation (3D Volume)
============================================================
This script mimics the exact FoB volume-based evaluation protocol (Gate C)
but extracts metrics for multiple Np budgets, and both Uniform and Adaptive
placements, to build the final paper tables.

It calculates:
- FoB Baseline (Np=10)
- Ada. Budget + Uni. Placement
- Fixed Np=10 + Ada. Placement
- AdaFoB (Ours)
- Oracle Np
"""

import os
import sys
import glob
import json
import numpy as np
import torch
import cv2

from tqdm import tqdm
import SimpleITK as sitk

# Missing dependencies
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

# Paths
NORMALIZED_DIR = "/kaggle/working/sabs_CT_normalized"
CKPT_DIR = "/kaggle/working/fob_checkpoints"
SAM_CKPT = "/kaggle/working/sam_vit_h.pth"
GLOBAL_PARAMS_PATH = os.path.join(REPO_DIR, "results", "r3_global_params.json")

TEST_LABELS = [1, 2, 3, 6]  # Spleen, RK, LK, Liver
N_PART = 3   # number of support slices / query chunks
SUPP_IDX = 2  # support volume index within fold
VALID_NPS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]

def load_fob_checkpoint(fold, ckpt_dir):
    candidates = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*/**/*.pth"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*.pth"), recursive=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No checkpoint found for fold {fold} in {ckpt_dir}")

def get_voxel_spacing(volume_idx):
    """Fetch the voxel spacing from the original normalized .nii.gz file to compute true physical HD95."""
    img_path = os.path.join(NORMALIZED_DIR, f"image_{volume_idx}.nii.gz")
    if os.path.exists(img_path):
        img_obj = sitk.ReadImage(img_path)
        return img_obj.GetSpacing()
    return (1.0, 1.0, 1.0)  # Fallback

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
        return 100.0  # Penalty for empty prediction
    try:
        # Pass spacing to compute HD95 in true physical mm
        return hd95(pred, gt, voxelspacing=spacing)
    except RuntimeError:
        return 100.0

def evaluate_adafob_matrix():
    print("\n" + "=" * 70)
    print("GATE D: Full AdaFoB Matrix Evaluation (3D Volume)")
    print("=" * 70)

    # Load global params if available
    global_params = {"nu": 0.02, "lam": 0.5, "a0": 0.6, "tau": 0.05}
    if os.path.exists(GLOBAL_PARAMS_PATH):
        with open(GLOBAL_PARAMS_PATH, "r") as f:
            global_params.update(json.load(f))
        print(f"Loaded Global Params: {global_params}")
    else:
        print(f"Using Default Params: {global_params}")

    labels = get_label_names("SABS")
    data_dir = os.path.dirname(NORMALIZED_DIR)

    # Prepare SAM Predictor once
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).eval().cuda()
    predictor = SamPredictor(sam)

    results_data = []

    for eval_fold in range(5):
        print(f"\n--- Fold {eval_fold} ---")
        ckpt_path = load_fob_checkpoint(eval_fold, CKPT_DIR)

        class DummyArgs: pass
        args = DummyArgs()
        args.dataset = "SABS"
        args.max_points = 24  # We need up to 24 points

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
            test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
        )

        for label_val, label_name in labels.items():
            if label_name == 'BG' or label_val not in TEST_LABELS:
                continue

            print(f"  Testing: {label_name} (label={label_val})")
            support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=N_PART)
            test_dataset.label = label_val

            support_image = [support_sample['image'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]
            support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]

            with torch.no_grad():
                for vi, sample in enumerate(test_loader):
                    # We need the original volume ID to get spacing!
                    # test_loader doesn't give us the ID directly. We can parse it from sample['id'] if available.
                    vol_id_str = sample.get('id', [f"image_{vi}.nii.gz"])[0]
                    try:
                        vol_idx = int(vol_id_str.split("image_")[-1].split(".nii.gz")[0])
                    except:
                        vol_idx = vi
                    
                    spacing = get_voxel_spacing(vol_idx)

                    query_image = [sample['image'][j].float().cuda() for j in range(sample['image'].shape[0])]
                    query_label = sample['label'].long()
                    C_q = sample['image'].shape[1]

                    # 3D Accumulators for ALL combinations: [Placement_Type][Np_Val] -> [D, H, W]
                    preds_uni = {np_val: np.zeros(query_label.shape[-3:], dtype=np.uint8) for np_val in VALID_NPS}
                    preds_ada = {np_val: np.zeros(query_label.shape[-3:], dtype=np.uint8) for np_val in VALID_NPS}
                    
                    # Accumulator for mixed AdaFoB and Fixed placements where Np changes per slice
                    pred_adabudget_uni = np.zeros(query_label.shape[-3:], dtype=np.uint8)
                    pred_adabudget_ada = np.zeros(query_label.shape[-3:], dtype=np.uint8)
                    pred_fob_base = np.zeros(query_label.shape[-3:], dtype=np.uint8)
                    pred_fob_ada  = np.zeros(query_label.shape[-3:], dtype=np.uint8)

                    idx_ = np.linspace(0, C_q, N_PART + 1).astype('int')
                    
                    vol_budgets = []
                    
                    for sub_chunk in range(N_PART):
                        support_image_s = [support_image[sub_chunk]]
                        support_fg_mask_s = [support_fg_mask[sub_chunk]]
                        query_image_s = query_image[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]
                        query_label_s = query_label[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]

                        for j in range(query_image_s.shape[0]):
                            global_z = idx_[sub_chunk] + j
                            
                            # 1. Run Uniform FoB (max_points=24)
                            model.allocator = None
                            try:
                                uni_neg_p, pos_p = model([support_image_s], [support_fg_mask_s], [query_image_s[[j]]], query_label_s[[j]], None)
                                uni_neg_p_all = np.array(uni_neg_p).reshape(-1, 2) if uni_neg_p is not None else np.zeros((0, 2))
                                pos_p_arr = np.array(pos_p).reshape(-1, 2) if pos_p is not None else np.zeros((0, 2))
                            except:
                                uni_neg_p_all, pos_p_arr = np.zeros((0, 2)), np.zeros((0, 2))

                            # 2. Run Adaptive FoB (AdaFoB) (max_points=24)
                            allocator = PromptBudgetAllocator(max_points=24).cuda()
                            allocator.nu = global_params["nu"]
                            allocator.lam = global_params["lam"]
                            allocator.a0 = global_params["a0"]
                            allocator.tau = global_params["tau"]
                            model.allocator = allocator
                            
                            # Monkey-patch allocate to force 24 points while capturing budget
                            original_allocate = model.allocator.allocate
                            def capturing_allocate(qry_img, qry_pred_coarse, spt_fg_proto, supp_m, mdl, supp_fts):
                                a, contours, M_tilde = mdl.allocator.get_ambiguity_score(qry_img, qry_pred_coarse, spt_fg_proto, supp_m, mdl, supp_fts)
                                budget = mdl.allocator.compute_budget(a, contours, M_tilde)
                                mdl.allocator.last_budget = budget
                                r = mdl.allocator.get_scale_adaptive_offset(M_tilde)
                                pts = mdl.allocator.sample_placement(qry_img, M_tilde, contours, 24, r)
                                return pts, budget

                            model.allocator.allocate = capturing_allocate
                            try:
                                ada_neg_p, _ = model([support_image_s], [support_fg_mask_s], [query_image_s[[j]]], query_label_s[[j]], None)
                                ada_neg_p_all = np.array(ada_neg_p).reshape(-1, 2) if ada_neg_p is not None else np.zeros((0, 2))
                            except:
                                ada_neg_p_all = np.zeros((0, 2))
                                
                            slice_budget = getattr(model.allocator, 'last_budget', 0)
                            vol_budgets.append(slice_budget)
                            
                            # Preprocess image and set SAM image ONCE for this slice
                            img_t = query_image_s[[j]][0].permute(1, 2, 0).cpu().numpy()
                            img_t = ((img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8) * 255).astype(np.uint8)
                            predictor.set_image(img_t)
                            
                            def query_sam(pts_pos, pts_neg):
                                all_pts = []
                                all_lbl = []
                                if len(pts_pos) > 0:
                                    all_pts.extend(pts_pos)
                                    all_lbl.extend([1] * len(pts_pos))
                                if len(pts_neg) > 0:
                                    all_pts.extend(pts_neg)
                                    all_lbl.extend([0] * len(pts_neg))
                                
                                if len(all_pts) > 0:
                                    mask, _, _ = predictor.predict(
                                        point_coords=np.array(all_pts),
                                        point_labels=np.array(all_lbl),
                                        multimask_output=True
                                    )
                                    return (mask[0] > 0.5).astype(np.uint8) # best idx = 0
                                else:
                                    return np.zeros((img_t.shape[0], img_t.shape[1]), dtype=np.uint8)

                            # 3. Sweep all Nps and compute the 4 main settings
                            closest_budget_val = min(VALID_NPS, key=lambda x: abs(x - slice_budget))
                            
                            for np_val in VALID_NPS:
                                # Uniform Prediction
                                mask_uni = query_sam(pos_p_arr, uni_neg_p_all[:np_val])
                                preds_uni[np_val][global_z] = mask_uni
                                
                                # Adaptive Prediction
                                mask_ada = query_sam(pos_p_arr, ada_neg_p_all[:np_val])
                                preds_ada[np_val][global_z] = mask_ada
                                
                                # Dynamic assignments for the mixed tables
                                if np_val == closest_budget_val:
                                    pred_adabudget_uni[global_z] = mask_uni
                                    pred_adabudget_ada[global_z] = mask_ada
                                
                                if np_val == 10:
                                    pred_fob_base[global_z] = mask_uni
                                    pred_fob_ada[global_z] = mask_ada

                    # After collecting all slices, compute 3D Volume Metrics
                    gt = (query_label.squeeze(0).cpu().numpy() > 0).astype(np.uint8)
                    mean_ada = np.mean(vol_budgets) if vol_budgets else 0
                    fixed_mean_np = min(VALID_NPS, key=lambda x: abs(x - int(np.round(mean_ada))))
                    
                    row = {
                        "fold": eval_fold,
                        "organ": label_name,
                        "vol_id": vol_id_str,
                        "mean_adaptive_budget": mean_ada,
                        "spacing_x": spacing[0],
                        "spacing_y": spacing[1],
                        "spacing_z": spacing[2],
                    }
                    
                    # Store sweeps
                    for np_val in VALID_NPS:
                        row[f"uni_dice_{np_val}"] = compute_volume_dice(preds_uni[np_val], gt)
                        row[f"uni_hd95_{np_val}"] = compute_volume_hd95(preds_uni[np_val], gt, spacing)
                        row[f"ada_dice_{np_val}"] = compute_volume_dice(preds_ada[np_val], gt)
                        row[f"ada_hd95_{np_val}"] = compute_volume_hd95(preds_ada[np_val], gt, spacing)

                    # Store combinations directly to avoid mismatch later
                    row["dice_fob_base"] = compute_volume_dice(pred_fob_base, gt)
                    row["hd95_fob_base"] = compute_volume_hd95(pred_fob_base, gt, spacing)
                    
                    row["dice_adabudget_uni"] = compute_volume_dice(pred_adabudget_uni, gt)
                    row["hd95_adabudget_uni"] = compute_volume_hd95(pred_adabudget_uni, gt, spacing)
                    
                    row["dice_fixed_10_ada"] = compute_volume_dice(pred_fob_ada, gt)
                    row["hd95_fixed_10_ada"] = compute_volume_hd95(pred_fob_ada, gt, spacing)
                    
                    row["dice_adafob"] = compute_volume_dice(pred_adabudget_ada, gt)
                    row["hd95_adafob"] = compute_volume_hd95(pred_adabudget_ada, gt, spacing)
                    
                    # Compute Oracle
                    best_u_dice = max([row[f"uni_dice_{n}"] for n in VALID_NPS])
                    row["oracle_dice"] = best_u_dice
                    
                    results_data.append(row)
                    
                    print(f"    Vol {vi}: Baseline(Np=10)={row['dice_fob_base']*100:.2f}%, " 
                          f"AdaFoB={row['dice_adafob']*100:.2f}%, "
                          f"MeanBudget={mean_ada:.1f}, Spacing={spacing}")

    import pandas as pd
    df = pd.DataFrame(results_data)
    os.makedirs(os.path.join(REPO_DIR, "results"), exist_ok=True)
    df.to_csv(os.path.join(REPO_DIR, "results", "gate_d_results.csv"), index=False)
    print("\nResults saved to results/gate_d_results.csv")


if __name__ == "__main__":
    # If the user restarted the Kaggle session, the working directory is empty.
    # We can automatically trigger the Gate C setup steps by simply importing it.
    # Gate C's evaluation is wrapped in __main__, so it will safely skip the 30-min evaluation.
    import experiments.gate_c_reproduce_fob
    
    evaluate_adafob_matrix()

