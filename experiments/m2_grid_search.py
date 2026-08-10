import sys
import os
import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.append(os.path.join(_ROOT, "third_party", "FoB_SAM"))

from models.FoB import FewShotSeg
from models.allocator import PromptBudgetAllocator
from segment_anything import sam_model_registry, SamPredictor

from experiments.eval import (
    load_volumes, available_organs, 
    sample_episode, build_inputs,
    predict_sam_from_points, load_checkpoint, compute_dice
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

def compute_L_and_kappa(contours):
    L = 0
    total_kappa = 0
    valid_kappa_pts = 0
    
    for contour in contours:
        if len(contour) < 3:
            continue
        L += cv2.arcLength(contour, closed=True)
        
        pts = contour[:, 0, :]
        if len(pts) >= 5:
            dx = np.gradient(pts[:, 0])
            dy = np.gradient(pts[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            
            denom = (dx**2 + dy**2)**1.5
            valid_idx = denom > 1e-5
            if valid_idx.sum() > 0:
                kappa = np.abs(dx[valid_idx]*ddy[valid_idx] - dy[valid_idx]*ddx[valid_idx]) / denom[valid_idx]
                total_kappa += kappa.sum()
                valid_kappa_pts += len(kappa)
    
    mean_kappa = total_kappa / valid_kappa_pts if valid_kappa_pts > 0 else 0
    return L, mean_kappa

def sweep_episode(predictor, fob, base_sample, Np_list, H, W):
    supp_imgs = [[t.clone().cuda() for t in way] for way in base_sample['support_images']]
    supp_masks = [[t.clone().cuda() for t in way] for way in base_sample['support_fg_labels']]
    qry_imgs = [t.clone().cuda() for t in base_sample['query_images']]
    qry_labels = base_sample['query_labels'].clone().cuda()
    
    # 1. Get negative points from baseline
    fob.allocator = None
    with torch.no_grad():
        neg_p, pos_p = fob(supp_imgs, supp_masks, qry_imgs, qry_labels, train=False, use_skeleton=False, budget_Np=24)
        
    # 2. Get ambiguity score, L, mean_kappa
    fob.allocator = PromptBudgetAllocator(max_points=24).cuda()
    
    original_allocate = fob.allocator.allocate
    def capturing_allocate(qry_img, qry_pred_coarse, spt_fg_proto, supp_mask, model, supp_fts):
        a, contours, M_tilde = fob.allocator.get_ambiguity_score(qry_img, qry_pred_coarse, spt_fg_proto, supp_mask, model, supp_fts)
        fob.allocator.last_a = a
        fob.allocator.last_contours = contours
        return original_allocate(qry_img, qry_pred_coarse, spt_fg_proto, supp_mask, model, supp_fts)
        
    fob.allocator.allocate = capturing_allocate
    
    with torch.no_grad():
        _ = fob(supp_imgs, supp_masks, qry_imgs, qry_labels, train=False, use_skeleton=False)
        
    a = getattr(fob.allocator, 'last_a', 0.0)
    contours = getattr(fob.allocator, 'last_contours', [])
    fob.allocator = None
    
    L, mean_kappa = compute_L_and_kappa(contours)
    
    # 3. Sweep Np
    results = {}
    if neg_p is not None:
        neg_p_all = np.array(neg_p).reshape(-1, 2)
    else:
        neg_p_all = np.zeros((0, 2))
        
    for np_val in Np_list:
        neg_p_subset = neg_p_all[:np_val] if np_val > 0 else np.zeros((0, 2))
        gt = base_sample["query_mask_np"]
        
        try:
            pred_mask, _ = predict_sam_from_points(
                predictor, pos_p, neg_p_subset, H, W, mask_select="max_area"
            )
            dice = compute_dice(pred_mask, gt) if pred_mask is not None else 0.0
            results[np_val] = dice
        except Exception:
            results[np_val] = 0.0
            
    # Find optimal Np
    optimal_Np = max(results, key=results.get)
    return L, mean_kappa, a, optimal_Np

def grid_search_budget(dataset):
    print("Grid Searching Parameters...")
    nu_space = np.linspace(0.005, 0.05, 10)
    lam_space = [0.0, 0.2, 0.5, 1.0]
    a0_space = [0.4, 0.5, 0.6, 0.7, 0.8]
    tau_space = [0.05, 0.1, 0.2]
    
    best_params = None
    best_mse = float('inf')
    
    for nu in nu_space:
        for lam in lam_space:
            for a0 in a0_space:
                for tau in tau_space:
                    
                    mse_sum = 0
                    for (L, mean_kappa, a, opt_Np) in dataset:
                        # calculate budget
                        g_a = 1.0 / (1.0 + np.exp(-(a - a0) / tau))
                        budget_float = nu * L * (1 + lam * mean_kappa) * g_a
                        budget = int(np.round(budget_float))
                        budget = np.clip(budget, 0, 24)
                        
                        mse_sum += (budget - opt_Np)**2
                    
                    mse = mse_sum / len(dataset)
                    if mse < best_mse:
                        best_mse = mse
                        best_params = {"nu": nu, "lam": lam, "a0": a0, "tau": tau}
                        
    return best_params, best_mse

def main():
    print("Running m2 grid search...")
    
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    
    data_root = "/kaggle/input/datasets/nhatphatnguyen/abd-ct"
    ckpt_path = "/kaggle/working/baseline_fob/exps_train_on_SABS_FSMIS_FoB/FSMIS_train_SABS_cv2/ckpt.pth"
    sam_ckpt = "/kaggle/working/checkpoints/sam_vit_h_4b8939.pth"
    
    organ_map = {8: "aorta", 4: "gallbladder"}
    
    print("Loading datasets...")
    # Load volumes but limit organs
    volumes, stats = load_volumes(data_root, (-1024, 3072), None)
    
    class DummyArgs: pass
    dummy = DummyArgs()
    dummy.n_ways = 1
    dummy.n_shots = 1
    
    print("Loading models...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
    predictor = SamPredictor(sam)
    
    fob = FewShotSeg(dummy).cuda().eval()
    load_checkpoint(fob, ckpt_path, "FoB baseline", strict=False)
    
    Np_list = [0, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24]
    
    organs = available_organs(volumes, organ_map, 100) # Ensure 100 min pixels
    # filter to only aorta and gallbladder
    organs = [o for o in organs if o['class_id'] in organ_map]
    
    organ_datasets = {8: [], 4: []}
    
    for class_id in organ_map.keys():
        print(f"Sampling episodes for {organ_map[class_id]} (ID {class_id})")
        cls_organs = [o for o in organs if o['class_id'] == class_id]
        
        if not cls_organs:
            print(f"No organs found for {class_id}")
            continue
            
        for _ in tqdm(range(20)):
            organ = random.choice(cls_organs)
            episode = sample_episode(volumes, organ, 100)
            base_sample = build_inputs(episode, predictor, norm_fn=make_baseline_norm, rgb_fn=sam_uint8_from_canonical)
            
            # extract features
            H, W = base_sample["query_mask_np"].shape
            L, mean_kappa, a, opt_Np = sweep_episode(predictor, fob, base_sample, Np_list, H, W)
            organ_datasets[class_id].append((L, mean_kappa, a, opt_Np))
            
        # Run grid search
        print(f"\nRunning grid search for {organ_map[class_id]}...")
        best_params, best_mse = grid_search_budget(organ_datasets[class_id])
        print(f"Best params for {organ_map[class_id]}: {best_params} (MSE={best_mse:.2f})")
        
    print("\nAll done.")

if __name__ == "__main__":
    main()
