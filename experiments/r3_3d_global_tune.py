import os
import sys
import numpy as np
import torch
import json
import argparse
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.append(os.path.join(_ROOT, "third_party", "FoB_SAM"))

from models.FoB import FewShotSeg
from models.allocator import PromptBudgetAllocator
from segment_anything import sam_model_registry, SamPredictor
from experiments.eval import (
    load_volumes, available_organs, sample_episode, build_inputs,
    run_model, predict_sam_from_points, load_checkpoint
)

# 1. We must cache [organ, vol_id, slice_idx, a_score, I(Np), U(Np)] for all valid Nps.
VALID_NPS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]

def evaluate_and_cache(gpu=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = "cuda"
    
    # Init SAM and FoB
    sam = sam_model_registry["vit_b"](checkpoint=os.path.join(_ROOT, "weights", "sam_vit_b_01ec64.pth"))
    sam.to(device)
    predictor = SamPredictor(sam)
    
    fob = FewShotSeg().to(device)
    load_checkpoint(fob, os.path.join(_ROOT, "weights", "FoB_SAM.pth"))
    fob.eval()
    
    organs_to_run = [1, 2, 3, 6]  # Spleen, RK, LK, Liver
    if gpu == 0:
        organs_to_run = [1, 2]
    elif gpu == 1:
        organs_to_run = [3, 6]
    
    cache_data = []
    
    for org_id in organs_to_run:
        label_name = available_organs()[org_id]
        
        for fold in range(5):
            print(f"Caching Fold {fold} - Organ {label_name}")
            
            # Since this is a global cross-val, we process both train and val volumes
            test_vols = load_volumes(fold, "test")
            train_vols = load_volumes(fold, "train")
            all_vols = train_vols + test_vols
            
            for vol_path, label_path in tqdm(all_vols):
                try:
                    sample = sample_episode(vol_path, label_path, org_id)
                except Exception as e:
                    continue
                
                inputs = build_inputs(sample, device)
                if inputs is None: continue
                supp_imgs, supp_masks, qry_imgs, qry_labels = inputs
                
                H, W = sample['query_images'][0].shape[1], sample['query_images'][0].shape[2]
                
                # 1. Run FoB Baseline to get 24 negative points
                fob.allocator = None
                fob.max_points = 24
                
                with torch.no_grad():
                    neg_p, pos_p = fob(supp_imgs, supp_masks, qry_imgs, qry_labels, train=False, use_skeleton=False, budget_Np=24)
                
                neg_p_all = np.array(neg_p).reshape(-1, 2) if neg_p is not None else np.zeros((0, 2))
                pos_p_arr = np.array(pos_p).reshape(-1, 2) if pos_p is not None else np.zeros((0, 2))
                
                # 2. Get ambiguity score 'a' for each slice using our allocator
                alloc_dummy = PromptBudgetAllocator(max_points=24).to(device)
                
                # We need to capture 'a' per slice
                # Since FoB is built to process slices in a batch, let's just use the allocator method directly
                # However, FoB internal variables like qry_pred_coarse are not exposed
                # We can intercept it like we did in e1_budget_sweep
                a_scores = []
                original_allocate = alloc_dummy.allocate
                def intercept_allocate(qry_img, qry_pred_coarse, spt_fg_proto, supp_m, mdl, supp_fts):
                    a, _, _ = alloc_dummy.get_ambiguity_score(qry_img, qry_pred_coarse, spt_fg_proto, supp_m, mdl, supp_fts)
                    a_scores.append(a)
                    return original_allocate(qry_img, qry_pred_coarse, spt_fg_proto, supp_m, mdl, supp_fts)
                
                alloc_dummy.allocate = intercept_allocate
                fob.allocator = alloc_dummy
                with torch.no_grad():
                    _ = fob(supp_imgs, supp_masks, qry_imgs, qry_labels, train=False, use_skeleton=False)
                
                # Restore
                fob.allocator = None
                
                # 3. For each slice, predict SAM for all valid Nps
                n_slices = len(sample['query_images'])
                vol_id_str = os.path.basename(vol_path)
                
                for j in range(n_slices):
                    img_t = sample['query_images'][j].permute(1, 2, 0).numpy()
                    img_t = ((img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8) * 255).astype(np.uint8)
                    predictor.set_image(img_t)
                    
                    gt = (sample['query_labels'][j].numpy() > 0).astype(np.uint8)
                    a_score = a_scores[j] if j < len(a_scores) else 0.0
                    
                    slice_data = {
                        "fold": fold,
                        "organ": label_name,
                        "vol_id": vol_id_str,
                        "slice_idx": j,
                        "a_score": float(a_score),
                        "I": {},
                        "U": {}
                    }
                    
                    for np_val in VALID_NPS:
                        pts = []
                        lbls = []
                        if len(pos_p_arr) > 0:
                            pts.extend(pos_p_arr)
                            lbls.extend([1] * len(pos_p_arr))
                        if np_val > 0 and len(neg_p_all) > 0:
                            n_neg = min(np_val, len(neg_p_all))
                            pts.extend(neg_p_all[:n_neg])
                            lbls.extend([0] * n_neg)
                            
                        if len(pts) > 0:
                            mask, _, _ = predictor.predict(
                                point_coords=np.array(pts),
                                point_labels=np.array(lbls),
                                multimask_output=True
                            )
                            pred = (mask[0] > 0.5).astype(np.uint8)
                        else:
                            pred = np.zeros_like(gt)
                            
                        I = np.sum(pred & gt)
                        U = np.sum(pred) + np.sum(gt)
                        
                        slice_data["I"][np_val] = int(I)
                        slice_data["U"][np_val] = int(U)
                        
                    cache_data.append(slice_data)
                    
    # Save cache
    os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
    out_file = os.path.join(_ROOT, "results", f"3d_tuning_cache_gpu{gpu}.json")
    with open(out_file, "w") as f:
        json.dump(cache_data, f)
    print(f"Cached saved to {out_file}")


def grid_search():
    cache_path_0 = os.path.join(_ROOT, "results", "3d_tuning_cache_gpu0.json")
    cache_path_1 = os.path.join(_ROOT, "results", "3d_tuning_cache_gpu1.json")
    
    cache_data = []
    for p in [cache_path_0, cache_path_1]:
        if os.path.exists(p):
            with open(p, "r") as f:
                cache_data.extend(json.load(f))
        else:
            print(f"Warning: Cache file {p} not found!")
            
    if not cache_data:
        print("No cache data found. Please run evaluate_and_cache first.")
        return
        
    # We want to tune on Fold 0 train set, and apply to all folds?
    # Or tune on Fold 0 training, evaluate on Fold 0 test?
    # Opus said: "single global set, declared in the text... fit within each fold's training volumes, evaluate on that fold's test volumes"
    # Actually, a single global set means we pool ALL training volumes across all 5 folds?
    # Let's do a fast grid search to find ONE global parameter set.
    
    # For now, let's simulate compute_budget
    def compute_budget(a, nu, tau, a0):
        # We cap it at max=24, min=0
        budget = nu * (a ** tau) + a0
        budget = max(0, min(24, int(np.round(budget))))
        return min(VALID_NPS, key=lambda x: abs(x - budget))
        
    # Let's grid search parameters
    grid_nu = [10, 15, 20, 24]
    grid_tau = [0.5, 1.0, 2.0]
    grid_a0 = [-5, 0, 5]
    
    # We only care about overall Dice on the 3D volume level.
    # To do cross-validation, for each fold, we find the best params on the training volumes (other 4 folds).
    # Then we evaluate on the test fold.
    
    folds = [0, 1, 2, 3, 4]
    
    for fold in folds:
        print(f"\n--- Tuning for Fold {fold} ---")
        train_data = [d for d in cache_data if d["fold"] != fold]
        test_data = [d for d in cache_data if d["fold"] == fold]
        
        best_dice = -1
        best_params = None
        
        for nu, tau, a0 in product(grid_nu, grid_tau, grid_a0):
            total_I = 0
            total_U = 0
            
            for slice_data in train_data:
                a = slice_data["a_score"]
                budget = compute_budget(a, nu, tau, a0)
                total_I += slice_data["I"][str(budget)]
                total_U += slice_data["U"][str(budget)]
                
            dice = 2 * total_I / (total_U + 1e-8)
            if dice > best_dice:
                best_dice = dice
                best_params = (nu, tau, a0)
                
        print(f"Best Training Params: nu={best_params[0]}, tau={best_params[1]}, a0={best_params[2]} (Dice: {best_dice*100:.2f}%)")
        
        # Evaluate on Test Fold
        total_I = 0
        total_U = 0
        for slice_data in test_data:
            a = slice_data["a_score"]
            budget = compute_budget(a, *best_params)
            total_I += slice_data["I"][str(budget)]
            total_U += slice_data["U"][str(budget)]
            
        test_dice = 2 * total_I / (total_U + 1e-8)
        print(f"Test Dice: {test_dice*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Run the SAM evaluations and cache I, U per slice")
    parser.add_argument("--search", action="store_true", help="Run the grid search over the cache")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (0 or 1)")
    args = parser.parse_args()
    
    if args.cache:
        evaluate_and_cache(args.gpu)
    if args.search:
        grid_search()
