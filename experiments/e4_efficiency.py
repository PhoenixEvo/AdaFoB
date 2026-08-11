import sys
import os
import random
import time
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
    load_checkpoint, compute_dice
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
    print("Running E4: Efficiency and Budget Distribution Ablation")
    
    data_root = "/kaggle/input/datasets/nhatphatnguyen/abd-ct"
    ckpt_path = "/kaggle/working/baseline_fob/exps_train_on_SABS_FSMIS_FoB/FSMIS_train_SABS_cv2/ckpt.pth"
    sam_ckpt = "/kaggle/working/checkpoints/sam_vit_h_4b8939.pth"
    
    for pat in ["/kaggle/input/**/*sabs_CT_normalized*", "/kaggle/input/**/*abd*ct*"]:
        import glob
        for h in glob.glob(pat, recursive=True):
            if os.path.isdir(h):
                data_root = h
                break
                
    volumes, stats = load_volumes(data_root, (-1024, 3072), None)
    organ_map = {8: "aorta", 4: "gallbladder"}
    available_classes = available_organs(volumes, organ_map, 100)
    
    dummy = type("A", (), {})()
    dummy.n_ways = 1
    dummy.n_shots = 1
    
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
    predictor = SamPredictor(sam)
    
    baseline_fob = FewShotSeg(dummy).cuda().eval()
    load_checkpoint(baseline_fob, ckpt_path, "FoB baseline", strict=False)
    baseline_fob.allocator = None
    
    adafob = FewShotSeg(dummy).cuda().eval()
    load_checkpoint(adafob, ckpt_path, "AdaFoB", strict=False)
    
    episodes_per_organ = 100
    results = []
    budget_distributions = {8: [], 4: []}
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    for class_id in [8, 4]:
        if class_id not in available_classes:
            continue
            
        organ_name = organ_map[class_id]
        print(f"\nEvaluating Efficiency for {organ_name}...")
        
        # Set optimal lam for AdaFoB
        lam_val = 0.0 if class_id == 8 else 1.0
        adafob.allocator = PromptBudgetAllocator(nu=0.015, lam=lam_val, a0=0.4, tau=0.2).cuda()
        
        # Monkey patch allocator to record the budget chosen
        original_allocate = adafob.allocator.allocate
        def capturing_allocate(qry_img, qry_pred_coarse, spt_fg_proto, supp_mask, model, supp_fts):
            budget = original_allocate(qry_img, qry_pred_coarse, spt_fg_proto, supp_mask, model, supp_fts)
            adafob.allocator.last_budget = budget
            return budget
        adafob.allocator.allocate = capturing_allocate
        
        base_times = []
        ada_times = []
        
        for _ in tqdm(range(episodes_per_organ)):
            ep = sample_episode(volumes, class_id, 100)
            sv = volumes[ep["support_vol"]]
            qv = volumes[ep["query_vol"]]
            sample = build_inputs(volumes, ep, make_baseline_norm(sv))
            
            q = ep["query_slice"]
            qry_img_canonical = qv["canon"][q]
            sam_img = sam_uint8_from_canonical(qry_img_canonical)
            predictor.set_image(sam_img)
            
            H, W = sample["query_mask_np"].shape
            
            # WARMUP (don't measure first few runs to avoid CUDA initialization overhead)
            # but we won't strictly enforce it here for simplicity. We just run directly.
            
            # --- BASELINE TIMING ---
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                neg_p_base, pos_p_base = run_model(baseline_fob, sample, train=False, use_skeleton=False, budget_Np=10)
                _, _ = predict_sam_from_points(predictor, pos_p_base, neg_p_base, H, W, mask_select="max_area")
            except Exception:
                pass
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            base_times.append((t1 - t0) * 1000) # in ms
            
            # --- ADAFOB TIMING ---
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            try:
                adafob.allocator.last_budget = 0
                neg_p_ada, pos_p_ada = run_model(adafob, sample, train=False, use_skeleton=False)
                _, _ = predict_sam_from_points(predictor, pos_p_ada, neg_p_ada, H, W, mask_select="max_area")
                budget_distributions[class_id].append(adafob.allocator.last_budget)
            except Exception:
                pass
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            ada_times.append((t3 - t2) * 1000)
            
        mean_base_ms = np.mean(base_times)
        mean_ada_ms = np.mean(ada_times)
        
        results.append({
            "Organ": organ_name,
            "Baseline_ms": mean_base_ms,
            "AdaFoB_ms": mean_ada_ms,
            "Speedup_Ratio": mean_base_ms / mean_ada_ms if mean_ada_ms > 0 else 1.0,
            "Mean_Allocated_Budget": np.mean(budget_distributions[class_id]) if budget_distributions[class_id] else 0
        })
        
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/e4_efficiency.csv", index=False)
    print("\n--- Efficiency Results ---")
    print(df)
    
    # Save budget distributions
    np.save("results/e4_budget_distributions.npy", budget_distributions)

if __name__ == "__main__":
    main()
