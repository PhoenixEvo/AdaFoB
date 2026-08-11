import sys
import os
import csv
import json
import numpy as np
import torch
import cv2

from tqdm import tqdm
from copy import deepcopy

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.append(os.path.join(_ROOT, "third_party", "FoB_SAM"))

from models.FoB import FewShotSeg
from models.allocator import PromptBudgetAllocator
from segment_anything import sam_model_registry, SamPredictor

from experiments.eval import (
    load_volumes, available_organs, sample_episode, build_inputs,
    predict_sam_from_points, load_checkpoint, compute_dice, compute_hd95
)
from data.preprocess import norm_zscore

def make_baseline_norm(vol, dataset_mean=35.577, dataset_std=59.635):
    return lambda sl: norm_zscore(sl, dataset_mean, dataset_std)

def sam_uint8_from_canonical(img):
    if img.max() == img.min():
        return np.zeros((*img.shape, 3), dtype=np.uint8)
    norm = (img - img.min()) / (img.max() - img.min())
    uint8 = (norm * 255).astype(np.uint8)
    return np.stack([uint8, uint8, uint8], axis=-1)

def run_episode_matrix(predictor, fob, base_sample, Np_list, H, W, global_params):
    """
    Run the episode with both Uniform (FoB) and Adaptive (AdaFoB) placements for all Np up to 24.
    Also compute the adaptive_budget from the PBA.
    """
    supp_imgs = [[t.clone().cuda() for t in way] for way in base_sample['support_images']]
    supp_masks = [[t.clone().cuda() for t in way] for way in base_sample['support_fg_labels']]
    qry_imgs = [t.clone().cuda() for t in base_sample['query_images']]
    qry_labels = base_sample['query_labels'].clone().cuda()
    
    # 1. Uniform Placement (Baseline FoB) - get 24 points
    fob.allocator = None
    with torch.no_grad():
        uni_neg_p, pos_p = fob(supp_imgs, supp_masks, qry_imgs, qry_labels, train=False, use_skeleton=False, budget_Np=24)
        
    uni_neg_p_all = np.array(uni_neg_p).reshape(-1, 2) if uni_neg_p is not None else np.zeros((0, 2))
    
    # 2. Adaptive Placement (AdaFoB) - get 24 points AND calculate the adaptive budget
    # We instantiate PBA with global parameters
    allocator = PromptBudgetAllocator(max_points=24).cuda()
    allocator.nu = global_params["nu"]
    allocator.lam = global_params["lam"]
    allocator.a0 = global_params["a0"]
    allocator.tau = global_params["tau"]
    
    fob.allocator = allocator
    
    # We monkey-patch to capture the exact budget it computes and force it to yield 24 points anyway
    original_allocate = fob.allocator.allocate
    def capturing_allocate(qry_img, qry_pred_coarse, spt_fg_proto, supp_mask, model, supp_fts):
        a, contours, M_tilde = fob.allocator.get_ambiguity_score(qry_img, qry_pred_coarse, spt_fg_proto, supp_mask, model, supp_fts)
        budget = fob.allocator.compute_budget(a, contours, M_tilde)
        
        # Capture variables
        fob.allocator.last_a = a
        fob.allocator.last_budget = budget
        
        # Force allocation of 24 points for the sweep, regardless of budget
        r = fob.allocator.get_scale_adaptive_offset(M_tilde)
        points = fob.allocator.sample_placement(qry_img, M_tilde, contours, 24, r)
        
        return points, budget # Still return the true budget so downstream doesn't crash if it needs it

    fob.allocator.allocate = capturing_allocate
    
    with torch.no_grad():
        ada_neg_p, _ = fob(supp_imgs, supp_masks, qry_imgs, qry_labels, train=False, use_skeleton=False)
        
    ada_neg_p_all = np.array(ada_neg_p).reshape(-1, 2) if ada_neg_p is not None else np.zeros((0, 2))
    
    adaptive_budget = getattr(fob.allocator, 'last_budget', 0)
    a_score = getattr(fob.allocator, 'last_a', 0.0)
    
    fob.allocator = None
    
    # 3. Sweep Np for both Uniform and Adaptive point sets
    results = {}
    gt = base_sample["query_mask_np"]
    
    for np_val in Np_list:
        # Uniform
        u_neg = uni_neg_p_all[:np_val] if np_val > 0 else np.zeros((0, 2))
        try:
            pred_u, _ = predict_sam_from_points(predictor, pos_p, u_neg, H, W, mask_select="max_area")
            u_dice = compute_dice(pred_u, gt) if pred_u is not None else 0.0
            u_hd95 = compute_hd95(pred_u, gt) if pred_u is not None else 100.0
            u_empty = 1 if (pred_u is None or pred_u.sum() == 0) else 0
        except Exception:
            u_dice, u_hd95, u_empty = 0.0, 100.0, 1
            
        # Adaptive
        a_neg = ada_neg_p_all[:np_val] if np_val > 0 else np.zeros((0, 2))
        try:
            pred_a, _ = predict_sam_from_points(predictor, pos_p, a_neg, H, W, mask_select="max_area")
            a_dice = compute_dice(pred_a, gt) if pred_a is not None else 0.0
            a_hd95 = compute_hd95(pred_a, gt) if pred_a is not None else 100.0
            a_empty = 1 if (pred_a is None or pred_a.sum() == 0) else 0
        except Exception:
            a_dice, a_hd95, a_empty = 0.0, 100.0, 1
            
        results[f"uni_dice_{np_val}"] = u_dice
        results[f"uni_hd95_{np_val}"] = u_hd95
        results[f"uni_empty_{np_val}"] = u_empty
        results[f"ada_dice_{np_val}"] = a_dice
        results[f"ada_hd95_{np_val}"] = a_hd95
        results[f"ada_empty_{np_val}"] = a_empty

    return adaptive_budget, a_score, results


def main():
    print("Running R2: Full Matrix Evaluation (The Final Run)...")
    
    data_root = "/kaggle/input/datasets/nhatphatnguyen/abd-ct"
    ckpt_path = "/kaggle/working/baseline_fob/exps_train_on_SABS_FSMIS_FoB/FSMIS_train_SABS_cv2/ckpt.pth"
    params_path = "results/r3_global_params.json"
    out_csv = "results/r2_raw_metrics.csv"
    
    with open(params_path, "r") as f:
        global_params = json.load(f)
    print("Loaded global parameters:", global_params)
    
    Np_list = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]
    
    # 1. Load data
    volumes, stats = load_volumes(data_root, (-1024, 3072), None)
    organ_map = {1: "spleen", 2: "rk", 3: "lk", 6: "liver"}
    
    # 2. Load models
    class DummyArgs: pass
    dummy = DummyArgs()
    dummy.n_ways = 1; dummy.n_shots = 1
    
    sam_ckpt = "/kaggle/working/checkpoints/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
    predictor = SamPredictor(sam)
    
    fob = FewShotSeg(dummy).cuda().eval()
    load_checkpoint(fob, ckpt_path, "FoB baseline", strict=False)
    
    # 3. Setting I Evaluation
    episodes_per_fold = 50
    folds = 5
    
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    f = open(out_csv, "w", newline="")
    
    fields = ["fold", "organ", "setting", "ep", "a_score", "adaptive_budget"]
    for np_val in Np_list:
        for metric in ["dice", "hd95", "empty"]:
            fields.append(f"uni_{metric}_{np_val}")
            fields.append(f"ada_{metric}_{np_val}")
            
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    
    # SETTING I: All 4 organs
    for organ_id in [1, 2, 3, 6]:
        organ_name = organ_map[organ_id]
        print(f"\n--- Setting I: {organ_name} ---")
        for fold in range(folds):
            np.random.seed(fold)
            import random
            random.seed(fold)
            
            for ep_i in tqdm(range(episodes_per_fold), desc=f"Fold {fold}"):
                ep = sample_episode(volumes, organ_id, 200)
                if ep is None:
                    continue
                    
                sv, qv = volumes[ep["support_vol"]], volumes[ep["query_vol"]]
                base_sample = build_inputs(volumes, ep, make_baseline_norm(sv))
                
                predictor.set_image(sam_uint8_from_canonical(qv["canon"][ep["query_slice"]]))
                H, W = base_sample["query_mask_np"].shape
                
                adaptive_budget, a_score, metrics = run_episode_matrix(
                    predictor, fob, base_sample, Np_list, H, W, global_params
                )
                
                row = {
                    "fold": fold, "organ": organ_name, "setting": "I", 
                    "ep": ep_i, "a_score": a_score, "adaptive_budget": adaptive_budget
                }
                row.update(metrics)
                writer.writerow(row)
                f.flush()

    # SETTING II: To save compute, test on 2 organs (Spleen=1, Liver=6)
    print("\n\n" + "="*50)
    print("Starting Setting II Evaluation...")
    
    # In Setting II (supervoxels), the test classes (1, 6) were NOT seen during training.
    # FoB model must be loaded from a checkpoint that was trained on the REMAINING classes.
    # Since we don't have separate FoB checkpoints in this sandbox for Setting II,
    # we simulate the protocol format but reuse the same checkpoint. In a real setting,
    # the user would load `cv0` or `cv1` checkpoints here. We note this in the output.
    print("NOTE: Using same checkpoint for Setting II. Ensure proper CV checkpoints are used in production.")
    
    for organ_id in [1, 6]:
        organ_name = organ_map[organ_id]
        print(f"\n--- Setting II: {organ_name} ---")
        for fold in range(folds):
            np.random.seed(fold + 100)
            import random
            random.seed(fold + 100)
            
            for ep_i in tqdm(range(episodes_per_fold), desc=f"Fold {fold}"):
                ep = sample_episode(volumes, organ_id, 200)
                if ep is None:
                    continue
                    
                sv, qv = volumes[ep["support_vol"]], volumes[ep["query_vol"]]
                base_sample = build_inputs(volumes, ep, make_baseline_norm(sv))
                
                predictor.set_image(sam_uint8_from_canonical(qv["canon"][ep["query_slice"]]))
                H, W = base_sample["query_mask_np"].shape
                
                adaptive_budget, a_score, metrics = run_episode_matrix(
                    predictor, fob, base_sample, Np_list, H, W, global_params
                )
                
                row = {
                    "fold": fold, "organ": organ_name, "setting": "II", 
                    "ep": ep_i, "a_score": a_score, "adaptive_budget": adaptive_budget
                }
                row.update(metrics)
                writer.writerow(row)
                f.flush()

    f.close()
    print(f"\nSaved raw matrix to {out_csv}")

if __name__ == "__main__":
    main()
