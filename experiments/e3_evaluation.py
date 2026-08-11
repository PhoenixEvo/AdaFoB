import sys
import os
import random
import numpy as np
import torch
import pandas as pd
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
    run_model, predict_sam_from_points,
    load_checkpoint, compute_dice, compute_hd95
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

def main():
    print("Running E3: Full Evaluation Matrix (Aorta & Gallbladder)")
    
    data_root = "/kaggle/input/datasets/nhatphatnguyen/abd-ct"
    ckpt_path = "/kaggle/working/baseline_fob/exps_train_on_SABS_FSMIS_FoB/FSMIS_train_SABS_cv2/ckpt.pth"
    sam_ckpt = "/kaggle/working/checkpoints/sam_vit_h_4b8939.pth"
    
    # Try dynamic root resolution like eval.py
    for pat in ["/kaggle/input/**/*sabs_CT_normalized*", "/kaggle/input/**/*abd*ct*"]:
        import glob
        for h in glob.glob(pat, recursive=True):
            if os.path.isdir(h):
                data_root = h
                break
                
    print(f"Data root: {data_root}")
    
    volumes, stats = load_volumes(data_root, (-1024, 3072), None)
    organ_map = {8: "aorta", 4: "gallbladder"}
    available_classes = available_organs(volumes, organ_map, 100)
    
    dummy = type("A", (), {})()
    dummy.n_ways = 1
    dummy.n_shots = 1
    
    print("Loading models...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
    predictor = SamPredictor(sam)
    
    baseline_fob = FewShotSeg(dummy).cuda().eval()
    load_checkpoint(baseline_fob, ckpt_path, "FoB baseline", strict=False)
    baseline_fob.allocator = None
    
    adafob = FewShotSeg(dummy).cuda().eval()
    load_checkpoint(adafob, ckpt_path, "AdaFoB", strict=False)
    
    # We will run 5 independent "folds" (by using 5 different seeds) to get robust mean and std
    n_folds = 5
    episodes_per_fold = 50
    
    results = []
    
    for fold in range(n_folds):
        random.seed(fold + 2024)
        np.random.seed(fold + 2024)
        torch.manual_seed(fold + 2024)
        
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        for class_id in [8, 4]:
            if class_id not in available_classes:
                continue
                
            organ_name = organ_map[class_id]
            print(f"Evaluating {organ_name}...")
            
            # Set optimal lam for AdaFoB based on grid search results
            lam_val = 0.0 if class_id == 8 else 1.0
            adafob.allocator = PromptBudgetAllocator(nu=0.015, lam=lam_val, a0=0.4, tau=0.2).cuda()
            
            base_dices = []
            ada_dices = []
            
            for _ in tqdm(range(episodes_per_fold)):
                ep = sample_episode(volumes, class_id, 100)
                sv = volumes[ep["support_vol"]]
                qv = volumes[ep["query_vol"]]
                
                # Same input for both
                sample = build_inputs(volumes, ep, make_baseline_norm(sv))
                
                q = ep["query_slice"]
                qry_img_canonical = qv["canon"][q]
                sam_img = sam_uint8_from_canonical(qry_img_canonical)
                predictor.set_image(sam_img)
                
                H, W = sample["query_mask_np"].shape
                
                # Run Baseline FoB
                try:
                    neg_p_base, pos_p_base = run_model(baseline_fob, sample, train=False, use_skeleton=False, budget_Np=10)
                    mask_base, _ = predict_sam_from_points(predictor, pos_p_base, neg_p_base, H, W, mask_select="max_area")
                    dice_base = compute_dice(mask_base, sample["query_mask_np"]) if mask_base is not None else 0.0
                except Exception:
                    dice_base = 0.0
                    
                # Run AdaFoB
                try:
                    neg_p_ada, pos_p_ada = run_model(adafob, sample, train=False, use_skeleton=False)
                    mask_ada, _ = predict_sam_from_points(predictor, pos_p_ada, neg_p_ada, H, W, mask_select="max_area")
                    dice_ada = compute_dice(mask_ada, sample["query_mask_np"]) if mask_ada is not None else 0.0
                except Exception:
                    dice_ada = 0.0
                    
                base_dices.append(dice_base)
                ada_dices.append(dice_ada)
                
            results.append({
                "Fold": fold,
                "Organ": organ_name,
                "Baseline_Dice": np.mean(base_dices),
                "AdaFoB_Dice": np.mean(ada_dices),
                "Gain": np.mean(ada_dices) - np.mean(base_dices)
            })
            
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/e3_evaluation.csv", index=False)
    print("\n--- Final Results ---")
    print(df.groupby("Organ").mean())

if __name__ == "__main__":
    main()
