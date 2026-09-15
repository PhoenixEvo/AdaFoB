import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

def cohen_d(x, y):
    """Cohen's d for two groups."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / dof)
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std

def main():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(repo_dir, 'results')
    
    seeds = [42, 43, 44]
    ranks = [4, 16]
    
    data = {4: {}, 16: {}}
    
    # Check if files exist
    missing = []
    for r in ranks:
        for s in seeds:
            # Try finding files in results/
            p = os.path.join(results_dir, f'lora_eval_fold0_rank{r}_seed{s}_combined.csv')
            if not os.path.exists(p):
                missing.append(f'rank{r}_seed{s}')
            else:
                data[r][s] = pd.read_csv(p)
                
    if missing:
        print("Waiting for CSV files. Missing:", missing)
        print("Please place the downloaded files in results/:")
        for m in missing:
            print(f"  - results/lora_eval_fold0_{m}_combined.csv")
        return
        
    print("========================================================================")
    print("REPRESENTATIVE-FOLD (FOLD 0) MULTI-SEED STATISTICAL ANALYSIS")
    print("========================================================================")
    
    # 1. POOLED COMPARISON (All 24 cases across 3 seeds = 72 observations per rank)
    r4_dice_all = np.concatenate([data[4][s]['dice_lora_ada'].values for s in seeds])
    r16_dice_all = np.concatenate([data[16][s]['dice_lora_ada'].values for s in seeds])
    
    r4_hd95_all = np.concatenate([data[4][s]['hd95_lora_ada'].values for s in seeds])
    r16_hd95_all = np.concatenate([data[16][s]['hd95_lora_ada'].values for s in seeds])
    
    w_dice_stat, w_dice_p = stats.wilcoxon(r16_dice_all, r4_dice_all, alternative='two-sided')
    d_dice = cohen_d(r16_dice_all, r4_dice_all)
    
    w_hd95_stat, w_hd95_p = stats.wilcoxon(r16_hd95_all, r4_hd95_all, alternative='two-sided')
    d_hd95 = cohen_d(r16_hd95_all, r4_hd95_all)
    
    print("\n--- 1. PRIMARY POOLED COMPARISON (n=72 observations: 24 cases x 3 seeds) ---")
    print(f"Dice  | Rank 4: {np.mean(r4_dice_all)*100:.2f}% ± {np.std(r4_dice_all)*100:.2f}% | Rank 16: {np.mean(r16_dice_all)*100:.2f}% ± {np.std(r16_dice_all)*100:.2f}% | p={w_dice_p:.4e} | Cohen's d={d_dice:+.3f}")
    print(f"HD95  | Rank 4: {np.mean(r4_hd95_all):.2f} ± {np.std(r4_hd95_all):.2f} mm | Rank 16: {np.mean(r16_hd95_all):.2f} ± {np.std(r16_hd95_all):.2f} mm | p={w_hd95_p:.4e} | Cohen's d={d_hd95:+.3f}")
    
    # 2. SEED-TO-SEED VARIANCE CHECK
    print("\n--- 2. PER-SEED MACRO MEANS (Fold 0 Macro Average) ---")
    r4_seed_means = [data[4][s]['dice_lora_ada'].mean() * 100 for s in seeds]
    r16_seed_means = [data[16][s]['dice_lora_ada'].mean() * 100 for s in seeds]
    
    print(f"Rank 4  Per-Seed Dice: {', '.join([f's{s}={m:.2f}%' for s, m in zip(seeds, r4_seed_means)])} | Mean: {np.mean(r4_seed_means):.2f}% ± {np.std(r4_seed_means, ddof=1):.2f}%")
    print(f"Rank 16 Per-Seed Dice: {', '.join([f's{s}={m:.2f}%' for s, m in zip(seeds, r16_seed_means)])} | Mean: {np.mean(r16_seed_means):.2f}% ± {np.std(r16_seed_means, ddof=1):.2f}%")
    
    seed_variance_r4 = np.std(r4_seed_means, ddof=1)
    seed_variance_r16 = np.std(r16_seed_means, ddof=1)
    rank_gap = np.mean(r16_seed_means) - np.mean(r4_seed_means)
    
    print(f"\nObserved Rank 16 vs Rank 4 Gap on Fold 0: {rank_gap:+.2f}%")
    print(f"Seed Std (Variance): Rank 4 = {seed_variance_r4:.2f}%, Rank 16 = {seed_variance_r16:.2f}%")
    if max(seed_variance_r4, seed_variance_r16) >= abs(rank_gap):
        print(">> VERDICT: Seed variance is comparable to/larger than the rank gap! Ranking CANNOT be confidently attributed to rank alone.")
    else:
        print(">> VERDICT: Rank 16 gain exceeds seed variance. Ranking is robust across random seeds.")
        
    # 3. DESCRIPTIVE PER-ORGAN BREAKDOWN (Unpowered, descriptive only)
    print("\n--- 3. PER-ORGAN DESCRIPTIVE BREAKDOWN (n=6 per seed, strictly descriptive, NOT powered) ---")
    organs = sorted(data[4][seeds[0]]['organ'].unique())
    for org in organs:
        o_r4_seeds = [data[4][s][data[4][s]['organ'] == org]['dice_lora_ada'].mean() * 100 for s in seeds]
        o_r16_seeds = [data[16][s][data[16][s]['organ'] == org]['dice_lora_ada'].mean() * 100 for s in seeds]
        print(f"  {org:8s} | r=4: {np.mean(o_r4_seeds):.2f}% ± {np.std(o_r4_seeds, ddof=1):.2f}% | r=16: {np.mean(o_r16_seeds):.2f}% ± {np.std(o_r16_seeds, ddof=1):.2f}%")

if __name__ == '__main__':
    main()
