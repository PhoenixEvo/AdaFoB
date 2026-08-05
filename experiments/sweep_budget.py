"""Sweep background prompt budget (E1) on Abd-CT to find optimal N_p per episode."""

import os
import sys
import glob
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..")))

from third_party.FoB_SAM.models.FoB import FewShotSeg
import experiments.eval as EV
from data.preprocess import sam_uint8_from_canonical

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="outputs/checkpoints/adafob_abdct.pth")
    ap.add_argument("--baseline_ckpt", type=str, default=None)
    ap.add_argument("--sam_ckpt", type=str, default="/kaggle/working/checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--n_cases", type=int, default=30)
    ap.add_argument("--organs", type=str, default="1,6,11")
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--hu_window", type=float, nargs=2, default=[-125.0, 275.0])
    args = ap.parse_args()

    data_root = args.data_root or "/kaggle/input/datasets/nhatphatnguyen/abd-ct/RawData/Training"
    volumes, stats = EV.load_volumes(data_root, tuple(args.hu_window))

    organs = [int(x) for x in args.organs.split(",")]
    
    # Load SAM
    from third_party.segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).cuda().eval()
    predictor = SamPredictor(sam)

    # Load Baseline FoB
    dummy = type("A", (), {})()
    base_ckpt = args.baseline_ckpt
    if not base_ckpt:
        hits = glob.glob("/kaggle/working/baseline_fob/**/*.pth", recursive=True)
        base_ckpt = hits[0] if hits else None
    
    model = FewShotSeg(dummy).cuda().eval()
    if base_ckpt:
        obj = torch.load(base_ckpt, map_location="cpu")
        for key in ("state_dict", "model", "net"):
            if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
        cleaned = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in obj.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"FoB baseline: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("!! no baseline checkpoint found")

    np.random.seed(args.seed)
    random.seed(args.seed)

    b_mu, b_sd = stats["dataset_mean"], stats["dataset_std"]
    norm_factory = lambda vol: lambda sl: EV.norm_fob(sl, b_mu, b_sd)

    budgets = [0, 1, 2, 3, 4, 6, 8, 10]
    
    report = {}
    
    for cls in organs:
        print(f"\n{'='*60}\nOrgan {cls}\n{'='*60}")
        
        cases = []
        for _ in range(args.n_cases):
            ep = EV.sample_episode(volumes, cls)
            if ep is None: continue
            qv = volumes[ep["query_vol"]]
            gt = (qv["label"][ep["query_slice"]] == cls).astype(np.uint8)
            if gt.sum() < 20: continue
            cases.append((ep, gt))
            
        print(f"Using {len(cases)} episodes")
        if not cases: continue
        
        results = []
        oracle_dices = []
        best_n_dices = {n: [] for n in budgets}
        n_stars = []
        
        for ep, gt in tqdm(cases, desc=f"Organ {cls}"):
            sv, qv = volumes[ep["support_vol"]], volumes[ep["query_vol"]]
            sample = EV.build_inputs(volumes, ep, norm_factory(sv))
            
            # 1. Get predicted prompts
            try:
                neg_p, pos_p = EV.run_model(model, sample, train=False, use_skeleton=False)
            except Exception as e:
                print(f"Forward failed: {e}")
                continue
                
            pos_pts = EV._as_points(pos_p)[:, ::-1].copy() # [N, 2]
            neg_pts = EV._as_points(neg_p)[:, ::-1].copy() # [10, 2]
            
            # 2. Get oracle prompts (for ceiling)
            o_pos, o_neg = EV.oracle_prompts(gt, n_pos=10, n_neg=10, rng=random.Random(args.seed))
            
            # 3. Cache SAM embedding ONCE
            predictor.set_image(sam_uint8_from_canonical(qv["canon"][ep["query_slice"]]))
            
            # Oracle ceiling
            dice_oracle = EV.dice_of(predictor, o_pos, o_neg, gt)
            oracle_dices.append(dice_oracle)
            
            # Budget sweep
            ep_dices = {}
            for Np in budgets:
                cur_neg = neg_pts[:Np] if Np > 0 else np.zeros((0,2), dtype=np.float32)
                d = EV.dice_of(predictor, pos_pts, cur_neg, gt)
                ep_dices[Np] = d
                best_n_dices[Np].append(d)
                
            # Find N*
            # Ties broken towards smaller budget
            best_n = 0
            best_d = ep_dices[0]
            for Np in budgets:
                if ep_dices[Np] > best_d + 1e-4:
                    best_n = Np
                    best_d = ep_dices[Np]
                    
            n_stars.append(best_n)
            results.append({
                "N_star": best_n,
                "dices": ep_dices,
                "oracle": dice_oracle
            })
            
        if not results: continue
        
        # Analyze
        print(f"\n--- Organ {cls} Analysis ---")
        
        # N* histogram
        hist = {n: n_stars.count(n) for n in budgets}
        print("N* Histogram:")
        for n in budgets:
            print(f"  N* = {n:2d}: {hist[n]} episodes ({hist[n]/len(cases)*100:.1f}%)")
            
        # Oracle vs best-global
        mean_oracle = np.mean(oracle_dices)
        global_means = {n: np.mean(best_n_dices[n]) for n in budgets}
        best_global_n = max(budgets, key=lambda n: global_means[n])
        best_global_dice = global_means[best_global_n]
        
        print(f"\nOracle Dice      : {mean_oracle:.4f}")
        print(f"Best global N_p={best_global_n} : {best_global_dice:.4f}")
        print(f"Headroom         : {mean_oracle - best_global_dice:.4f}  (G1 requires >= 0.015)")
        
        # Cost of forcing 10
        n_zero_cases = [r for r in results if r["N_star"] == 0]
        if n_zero_cases:
            cost = np.mean([r["dices"][0] - r["dices"][10] for r in n_zero_cases])
            frac = len(n_zero_cases) / len(cases)
            print(f"\nFraction where N*=0: {frac*100:.1f}%  (G1 requires >= 20%)")
            print(f"Cost of forcing 10 on this subset: {cost:.4f}  (G1 requires >= 0.03)")
        else:
            print("\nFraction where N*=0: 0.0%")
            cost = 0
            
        report[cls] = {
            "hist": hist,
            "oracle": float(mean_oracle),
            "best_global_dice": float(best_global_dice),
            "zero_frac": float(len(n_zero_cases) / len(cases)),
            "cost_of_10": float(cost)
        }
        
    os.makedirs("results", exist_ok=True)
    with open("results/sweep_budget.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
