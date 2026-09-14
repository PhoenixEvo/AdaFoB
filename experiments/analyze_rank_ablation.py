import pandas as pd
import numpy as np
from scipy import stats
import os

def cohen_d(x, y):
    """Calculate Cohen's d for two groups."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

def analyze_ranks(results_dir='results'):
    ranks = [2, 4, 8, 16]
    dfs = {}
    
    # Load all CSVs
    for r in ranks:
        csv_path = os.path.join(results_dir, f'lora_eval_rank{r}_combined.csv')
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping.")
            return
        dfs[r] = pd.read_csv(csv_path)
    
    # Merge on vol_id and organ to ensure exact paired alignment
    # Wait, we can just use the first dataframe as a base and join
    merged_df = dfs[ranks[0]][['vol_id', 'organ', 'dice_lora_ada', 'hd95_lora_ada']].rename(
        columns={'dice_lora_ada': f'dice_r{ranks[0]}', 'hd95_lora_ada': f'hd95_r{ranks[0]}'}
    )
    
    for r in ranks[1:]:
        temp_df = dfs[r][['vol_id', 'organ', 'dice_lora_ada', 'hd95_lora_ada']].rename(
            columns={'dice_lora_ada': f'dice_r{r}', 'hd95_lora_ada': f'hd95_r{r}'}
        )
        merged_df = pd.merge(merged_df, temp_df, on=['vol_id', 'organ'])
        
    print(f"Total matched paired samples: {len(merged_df)}")
    
    pairs = [(2, 4), (4, 8), (8, 16), (2, 16)]
    metrics = [('dice', 'Higher is better'), ('hd95', 'Lower is better')]
    organs = merged_df['organ'].unique().tolist() + ['ALL']
    
    results = []
    
    for metric, direction in metrics:
        for org in organs:
            if org == 'ALL':
                sub_df = merged_df
            else:
                sub_df = merged_df[merged_df['organ'] == org]
            
            for r1, r2 in pairs:
                col1 = f'{metric}_r{r1}'
                col2 = f'{metric}_r{r2}'
                
                x1 = sub_df[col1].values
                x2 = sub_df[col2].values
                
                mean1 = np.mean(x1)
                mean2 = np.mean(x2)
                
                # Wilcoxon signed-rank test
                try:
                    stat, p_val = stats.wilcoxon(x1, x2, alternative='two-sided')
                except Exception as e:
                    p_val = np.nan
                    
                d = cohen_d(x2, x1) # We use x2 - x1 so a positive Cohen's d means r2 is larger than r1
                
                results.append({
                    'metric': metric,
                    'organ': org,
                    'pair': f'r{r1}_vs_r{r2}',
                    'mean_r1': mean1,
                    'mean_r2': mean2,
                    'p_value': p_val,
                    'cohens_d': d,
                    'n_samples': len(x1)
                })
                
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(results_dir, 'rank_ablation_stats.csv'), index=False)
    print("Saved stats to results/rank_ablation_stats.csv")
    
    # Print summary for ALL organs
    print("\n--- Summary (ALL ORGANS) ---")
    print(res_df[res_df['organ'] == 'ALL'].to_string(index=False))

if __name__ == '__main__':
    analyze_ranks()
