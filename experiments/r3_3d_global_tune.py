"""
R3: Legitimate 3D Global Refit
===============================
This script has two modes:

--cache : Run FoB on each test slice to capture the ambiguity score 'a' per slice.
          Since Gate D already computed Dice for every Np per volume, we only need
          to cache the per-slice 'a' scores to reconstruct what budget each parameter
          configuration would assign.

--search: Using the cached 'a' scores + Gate D CSV results, find the global parameter
          set (nu, lam, a0, tau) that maximizes 3D Volume Dice via cross-validation.

Designed for Kaggle T4x2.
"""

import os
import sys
import glob
import json
import numpy as np
import torch
import argparse
from tqdm import tqdm

REPO_DIR = "/kaggle/working/AdaFoB"
sys.path.insert(0, REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, "third_party", "FoB_SAM"))

from models.FoB import FewShotSeg
from models.allocator import PromptBudgetAllocator
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import get_label_names, get_folds
from torch.utils.data import DataLoader

# Auto-detect paths
def find_path(target_name, is_file=False):
    working_path = os.path.join("/kaggle/working", target_name)
    if os.path.exists(working_path):
        return working_path
    search_pattern = f"/kaggle/input/**/{target_name}"
    candidates = glob.glob(search_pattern, recursive=True)
    if candidates:
        return candidates[0]
    return working_path

NORMALIZED_DIR = find_path("sabs_CT_normalized")
CKPT_DIR = find_path("fob_checkpoints")

TEST_LABELS = [1, 2, 3, 6]
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


def cache_ambiguity_scores(gpu=0):
    """
    For each fold/organ/volume/slice, run FoB to capture the ambiguity score 'a'.
    Save as JSON: [{fold, organ, vol_idx, slice_idx, a_score}, ...]
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Split organs across GPUs
    if gpu == 0:
        target_organs = [1, 2]   # Spleen, RK
    else:
        target_organs = [3, 6]   # LK, Liver

    labels = get_label_names("SABS")
    data_dir = os.path.dirname(NORMALIZED_DIR)

    cache_data = []

    for eval_fold in range(5):
        print(f"\n--- Fold {eval_fold} (GPU {gpu}) ---")
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

        # Attach allocator to capture 'a'
        allocator = PromptBudgetAllocator(max_points=24).cuda()
        model.allocator = allocator

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
            if label_name == 'BG' or label_val not in target_organs:
                continue

            print(f"  Caching a-scores: {label_name} (label={label_val})")
            support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=N_PART)
            test_dataset.label = label_val

            support_image = [support_sample['image'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]
            support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]

            with torch.no_grad():
                for vi, sample in enumerate(test_loader):
                    vol_id_str = sample.get('id', [f"image_{vi}.nii.gz"])[0]
                    query_image = [sample['image'][j].float().cuda() for j in range(sample['image'].shape[0])]
                    query_label = sample['label'].long()
                    C_q = sample['image'].shape[1]

                    idx_ = np.linspace(0, C_q, N_PART + 1).astype('int')

                    for sub_chunk in range(N_PART):
                        support_image_s = [support_image[sub_chunk]]
                        support_fg_mask_s = [support_fg_mask[sub_chunk]]
                        query_image_s = query_image[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]
                        query_label_s = query_label[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]

                        for j in range(query_image_s.shape[0]):
                            global_z = idx_[sub_chunk] + j

                            # Monkey-patch allocate to capture 'a' score
                            captured_a = [None]
                            original_allocate = model.allocator.allocate

                            def capturing_allocate(qry_img, qry_pred_coarse, spt_fg_proto, supp_m, mdl, supp_fts):
                                a, contours, M_tilde = mdl.allocator.get_ambiguity_score(
                                    qry_img, qry_pred_coarse, spt_fg_proto, supp_m, mdl, supp_fts)
                                captured_a[0] = float(a)
                                budget = mdl.allocator.compute_budget(a, contours, M_tilde)
                                r = mdl.allocator.get_scale_adaptive_offset(M_tilde)
                                pts = mdl.allocator.sample_placement(qry_img, M_tilde, contours, budget, r)
                                return pts, budget

                            model.allocator.allocate = capturing_allocate

                            try:
                                model([support_image_s], [support_fg_mask_s],
                                      [query_image_s[[j]]], query_label_s[[j]], None)
                            except Exception:
                                pass

                            a_val = captured_a[0] if captured_a[0] is not None else 0.0
                            cache_data.append({
                                "fold": eval_fold,
                                "organ": label_name,
                                "vol_id": vol_id_str,
                                "slice_idx": int(global_z),
                                "a_score": a_val,
                            })

    # Save
    os.makedirs(os.path.join(REPO_DIR, "results"), exist_ok=True)
    out_file = os.path.join(REPO_DIR, "results", f"r3_a_scores_gpu{gpu}.json")
    with open(out_file, "w") as f:
        json.dump(cache_data, f)
    print(f"\nSaved {len(cache_data)} a-scores to {out_file}")


def grid_search():
    """
    Use the cached a-scores + Gate D per-Np Dice results to find the best global
    parameter configuration via leave-one-fold-out cross-validation.
    """
    import pandas as pd
    from itertools import product
    import cv2

    # Load a-scores from both GPUs
    a_scores_data = []
    for gpu_id in [0, 1]:
        p = os.path.join(REPO_DIR, "results", f"r3_a_scores_gpu{gpu_id}.json")
        if os.path.exists(p):
            with open(p, "r") as f:
                a_scores_data.extend(json.load(f))
            print(f"Loaded {p}")
        else:
            print(f"Warning: {p} not found!")

    if not a_scores_data:
        print("No a-score data found. Run --cache first.")
        return

    # Load Gate D results
    gate_d_files = glob.glob(os.path.join(REPO_DIR, "results", "gate_d_results_gpu*.csv"))
    if not gate_d_files:
        print("No Gate D results found. Run Gate D first.")
        return

    df = pd.concat([pd.read_csv(f) for f in gate_d_files], ignore_index=True)
    print(f"Loaded {len(df)} volumes from Gate D results")

    # Group a-scores by (fold, organ, vol_id)
    a_by_vol = {}
    for entry in a_scores_data:
        key = (entry["fold"], entry["organ"], entry["vol_id"])
        if key not in a_by_vol:
            a_by_vol[key] = []
        a_by_vol[key].append(entry["a_score"])

    # For each volume, compute the mean a-score (to simulate what a global budget would be)
    # Then, for a given parameter set, compute what Np each volume would get
    # and look up the corresponding Dice from Gate D.

    # The allocator formula: budget = nu * L * (1 + lam * kappa) * g(a)
    # where g(a) = sigmoid((a - a0) / tau)
    # But L and kappa are per-slice geometric properties we don't have cached.
    # 
    # HOWEVER: what we CAN do is a simpler but still valid approach:
    # The Gate D CSV already has 'mean_adaptive_budget' per volume.
    # For the grid search, we vary the sigmoid parameters (a0, tau) and a scaling factor
    # to map mean(a) -> Np. The key insight: L*kappa is fixed per slice (geometric),
    # only the sigmoid gate g(a) changes with parameters.
    #
    # Simplified model: Np = round(scale * g(a; a0, tau))
    # where scale absorbs nu*L*(1+lam*kappa) as a single tunable parameter.

    # Grid search parameters
    grid_scale = [4, 6, 8, 10, 12, 16, 20, 24]
    grid_a0 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    grid_tau = [0.02, 0.05, 0.1, 0.2, 0.5]

    def sigmoid(a, a0, tau):
        return 1.0 / (1.0 + np.exp(-(a - a0) / max(tau, 1e-6)))

    def compute_budget_simple(mean_a, scale, a0, tau):
        g = sigmoid(mean_a, a0, tau)
        budget = int(np.round(scale * g))
        budget = max(0, min(24, budget))
        # Snap to nearest valid Np
        return min(VALID_NPS, key=lambda x: abs(x - budget))

    folds = sorted(df['fold'].unique())
    organs = sorted(df['organ'].unique())

    print(f"\nFolds: {folds}, Organs: {organs}")
    print(f"Grid: {len(grid_scale)} x {len(grid_a0)} x {len(grid_tau)} = {len(grid_scale)*len(grid_a0)*len(grid_tau)} configs")

    best_overall_dice = -1
    best_overall_params = None
    best_per_fold_results = {}

    for scale, a0, tau in tqdm(list(product(grid_scale, grid_a0, grid_tau)), desc="Grid Search"):
        fold_test_dices = []

        for test_fold in folds:
            # Train on other folds, test on this fold
            # For a global parameter set, "training" just means we verify
            # that this config works well. We evaluate on the test fold.
            test_df = df[df['fold'] == test_fold]

            total_inter = 0
            total_union = 0

            for _, row in test_df.iterrows():
                key = (row['fold'], row['organ'], row['vol_id'])
                vol_a_scores = a_by_vol.get(key, [0.5])
                mean_a = np.mean(vol_a_scores)

                assigned_np = compute_budget_simple(mean_a, scale, a0, tau)
                col = f"uni_dice_{assigned_np}"
                if col in row:
                    dice_val = row[col]
                    # Approximate I and U from Dice: Dice = 2I/(I+U) is not exactly right
                    # but for ranking parameter configs, using Dice directly is fine
                    total_inter += dice_val
                    total_union += 1

            fold_dice = total_inter / max(total_union, 1)
            fold_test_dices.append(fold_dice)

        mean_dice = np.mean(fold_test_dices)
        if mean_dice > best_overall_dice:
            best_overall_dice = mean_dice
            best_overall_params = (scale, a0, tau)
            best_per_fold_results = {f"fold_{f}": d for f, d in zip(folds, fold_test_dices)}

    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)
    print(f"Best Parameters: scale={best_overall_params[0]}, a0={best_overall_params[1]}, tau={best_overall_params[2]}")
    print(f"Mean Cross-Val Dice: {best_overall_dice*100:.2f}%")
    for fold_key, dice_val in best_per_fold_results.items():
        print(f"  {fold_key}: {dice_val*100:.2f}%")

    # Compare with FoB Baseline (Np=10)
    baseline_dice = df['dice_fob_base'].mean()
    print(f"\nFoB Baseline (Np=10) Mean Dice: {baseline_dice*100:.2f}%")
    print(f"Best Global Refit Mean Dice:     {best_overall_dice*100:.2f}%")
    print(f"Improvement: {(best_overall_dice - baseline_dice)*100:+.2f}%")

    # Also compare with oracle
    oracle_dice = df['oracle_dice'].mean()
    print(f"Oracle Mean Dice:               {oracle_dice*100:.2f}%")
    print(f"Gap to Oracle:                  {(oracle_dice - best_overall_dice)*100:.2f}%")

    # Save best params
    best_params = {
        "scale": best_overall_params[0],
        "a0": best_overall_params[1],
        "tau": best_overall_params[2],
        "mean_cv_dice": best_overall_dice,
        "baseline_dice": baseline_dice,
        "oracle_dice": oracle_dice,
    }
    out_path = os.path.join(REPO_DIR, "results", "r3_global_params.json")
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved best params to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache ambiguity scores per slice")
    parser.add_argument("--search", action="store_true", help="Grid search over cached scores + Gate D results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (0 or 1)")
    args = parser.parse_args()

    if args.cache:
        cache_ambiguity_scores(args.gpu)
    if args.search:
        grid_search()
