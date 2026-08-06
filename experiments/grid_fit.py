"""Grid search for Prompt Budget Allocator parameters (M2) on Fold 0."""

import os
import sys
import glob
import json
import random
import argparse
import itertools
import numpy as np
from tqdm import tqdm

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..")))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..", "third_party", "FoB_SAM")))

from models.FoB import FewShotSeg
import experiments.eval as EV
from segment_anything import sam_model_registry, SamPredictor
from data.preprocess import sam_uint8_from_canonical, sanitize_prompts

def predict_all(predictor, pos, neg, H=256, W=256):
    pos, _ = sanitize_prompts(pos, H, W, mode="drop")
    neg, _ = sanitize_prompts(neg, H, W, mode="drop")
    if len(pos) == 0 and len(neg) == 0:
        return None, None
    pts = np.concatenate([pos, neg], axis=0)
    lbl = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))], axis=0)
    masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbl, multimask_output=True)
    return masks, scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", type=str, default=None)
    ap.add_argument("--sam_ckpt", type=str, default="/kaggle/working/checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--data_root", type=str, default="/kaggle/input/datasets/nhatphatnguyen/abd-ct/RawData/Training")
    ap.add_argument("--n_cases", type=int, default=50) # Use 50 cases for grid search
    ap.add_argument("--organs", type=str, default="1,6,11")
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--hu_window", type=float, nargs=2, default=[-125.0, 275.0])
    args = ap.parse_args()

    volumes, stats = EV.load_volumes(args.data_root, tuple(args.hu_window))
    organs = [int(x) for x in args.organs.split(",")]
    
    # Load SAM
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).cuda().eval()
    predictor = SamPredictor(sam)

    # Load FoB
    dummy = type("A", (), {})()
    base_ckpt = args.baseline_ckpt
    if not base_ckpt:
        # Search multiple possible locations
        search_paths = [
            "/kaggle/working/baseline_fob/**/*.pth",
            "/kaggle/working/**/*.pth",
            "/kaggle/input/**/*.pth",
            "outputs/checkpoints/**/*.pth"
        ]
        for path in search_paths:
            hits = glob.glob(path, recursive=True)
            hits = [h for h in hits if 'sam_vit' not in h] # filter out SAM
            if hits:
                base_ckpt = hits[0]
                break
                
    if not base_ckpt:
        raise FileNotFoundError("No baseline FoB checkpoint found! Please provide --baseline_ckpt.")
    
    model = FewShotSeg(dummy).cuda().eval()
    if base_ckpt:
        obj = torch.load(base_ckpt, map_location="cpu")
        for key in ("state_dict", "model", "net"):
            if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
        cleaned = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in obj.items()}
        model.load_state_dict(cleaned, strict=False)

    np.random.seed(args.seed)
    random.seed(args.seed)

    b_mu, b_sd = stats["dataset_mean"], stats["dataset_std"]
    norm_factory = lambda vol: lambda sl: EV.norm_fob(sl, b_mu, b_sd)

    # Grid definition
    grid = {
        "nu": [0.02, 0.05, 0.1],
        "lam": [0.5, 1.0, 2.0],
        "gamma": [0.5, 1.0, 2.0],
        "a0": [0.4, 0.5, 0.6],
        "tau": [0.05, 0.1, 0.2],
        "alpha": [0.2, 0.35, 0.5]
    }
    
    keys = list(grid.keys())
    combinations = list(itertools.product(*(grid[k] for k in keys)))
    print(f"Grid size: {len(combinations)} combinations")
    
    # Cache variable for allocator arguments
    alloc_cache = {}
    original_allocate = model.allocator.allocate
    
    def caching_allocate(*args, **kwargs):
        alloc_cache['args'] = args
        alloc_cache['kwargs'] = kwargs
        return original_allocate(*args, **kwargs)
        
    model.allocator.allocate = caching_allocate

    all_episode_dices = {i: [] for i in range(len(combinations))}
    
    for cls in organs:
        print(f"\n--- Organ {cls} ---")
        
        cases = []
        for _ in range(args.n_cases):
            ep = EV.sample_episode(volumes, cls)
            if ep is None: continue
            qv = volumes[ep["query_vol"]]
            gt = (qv["label"][ep["query_slice"]] == cls).astype(np.uint8)
            if gt.sum() < 20: continue
            cases.append((ep, gt))
            
        for ep_idx, (ep, gt) in enumerate(tqdm(cases, desc=f"Grid fitting organ {cls}")):
            sv, qv = volumes[ep["support_vol"]], volumes[ep["query_vol"]]
            sample = EV.build_inputs(volumes, ep, norm_factory(sv))
            
            # 1. Run model ONCE to extract pos_p and cache allocator args
            alloc_cache.clear()
            try:
                # We don't care about the returned neg_p because we will sweep it
                _, pos_p = EV.run_model(model, sample, train=False, use_skeleton=False)
            except Exception as e:
                continue
                
            if 'args' not in alloc_cache:
                continue # allocator wasn't called
                
            pos_pts = EV._as_points(pos_p).copy()
            
            # 2. Cache SAM embedding ONCE
            predictor.set_image(sam_uint8_from_canonical(qv["canon"][ep["query_slice"]]))
            
            # 3. Sweep all combinations
            for comb_idx, vals in enumerate(combinations):
                params = dict(zip(keys, vals))
                
                # Apply params to allocator
                for k, v in params.items():
                    setattr(model.allocator, k, v)
                    
                # Run allocator without re-running encoder
                pred_point_alloc, budget_Np = original_allocate(*alloc_cache['args'], **alloc_cache['kwargs'])
                
                # Format neg_pts
                if budget_Np > 0 and pred_point_alloc is not None and len(pred_point_alloc) > 0:
                    neg_pts = pred_point_alloc[:budget_Np]
                else:
                    neg_pts = np.zeros((0, 2), dtype=np.float32)
                    
                # SAM Decoder
                masks, _ = predict_all(predictor, pos_pts, neg_pts)
                d = EV.compute_dice(masks[0], gt) if masks is not None else 0.0
                
                all_episode_dices[comb_idx].append(d)

    # Analyze results
    mean_dices = {i: np.mean(dices) for i, dices in all_episode_dices.items() if len(dices) > 0}
    if not mean_dices:
        print("No valid episodes found.")
        return
        
    best_idx = max(mean_dices, key=mean_dices.get)
    best_params = dict(zip(keys, combinations[best_idx]))
    best_dice = mean_dices[best_idx]
    
    print(f"\nBest Grid Configuration (Dice: {best_dice:.4f}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
        
    os.makedirs("results", exist_ok=True)
    with open("results/pba_params.json", "w") as f:
        json.dump({
            "best_dice": float(best_dice),
            "best_params": best_params,
            "grid_results": {str(dict(zip(keys, combinations[i]))): float(mean_dices[i]) for i in mean_dices}
        }, f, indent=2)

if __name__ == "__main__":
    main()
