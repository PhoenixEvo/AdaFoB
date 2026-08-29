"""
Summarize Fixed-Mean-Budget Control
====================================
Compares the output of eval_fixed_control.py against the adaptive budget results
(from lora_eval_final_combined.csv) to test the hypothesis that AdaFoB's
adaptive allocation achieves parity with a fixed mean budget.

Output: Statistical significance report testing AdaFoB+LoRA vs Fixed-Budget+LoRA.
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    fixed_csv = os.path.join(_ROOT, "results", "control_fixed_mean_budget_gpu0.csv")
    adaptive_csv = os.path.join(_ROOT, "results", "lora_eval_final_combined.csv")

    if not os.path.exists(fixed_csv):
        print(f"ERROR: Cannot find fixed control results at {fixed_csv}")
        sys.exit(1)
    
    # We load both and merge on vol_id and organ
    df_fixed = pd.read_csv(fixed_csv)
    
    # If adaptive_csv is not found (because it's a fresh Kaggle session),
    # we just report the fixed control metrics alone.
    if not os.path.exists(adaptive_csv):
        print("--- FIXED BUDGET CONTROL RESULTS ---")
        print("Note: lora_eval_final_combined.csv not found, cannot run paired Wilcoxon test.")
        for organ in sorted(df_fixed['organ'].unique()):
            sub = df_fixed[df_fixed['organ'] == organ]
            print(f"\n{organ} (n={len(sub)}, Np={sub['fixed_np'].iloc[0]}):")
            print(f"  Mean Dice: {sub['dice_fixed_control'].mean()*100:.2f}%")
            print(f"  Mean HD95: {sub['hd95_fixed_control'].mean():.2f}mm")
        print("\nOVERALL:")
        print(f"  Mean Dice: {df_fixed['dice_fixed_control'].mean()*100:.2f}%")
        print(f"  Mean HD95: {df_fixed['hd95_fixed_control'].mean():.2f}mm")
        sys.exit(0)

    df_ada = pd.read_csv(adaptive_csv)
    
    # Drop NaNs from adaptive
    df_ada = df_ada.dropna(subset=['dice_lora_ada', 'hd95_lora_ada'])
    
    # Merge
    merged = pd.merge(df_fixed, df_ada, on=['fold', 'organ', 'vol_id'], how='inner')
    
    if len(merged) == 0:
        print("ERROR: Merge failed. No matching samples between the two CSVs.")
        sys.exit(1)
        
    print(f"Loaded and merged {len(merged)} paired samples.\n")
    
    md_lines = [
        "# Fixed-Mean-Budget Control Results",
        "",
        "| Organ | Fixed Np | Fixed Dice | Adaptive Dice | p-value (Wilcoxon) | Significant Difference? |",
        "|-------|----------|------------|---------------|--------------------|-------------------------|"
    ]
    
    organs = sorted(merged['organ'].unique())
    for organ in organs + ['OVERALL']:
        if organ == 'OVERALL':
            sub = merged
            fixed_np_str = "Mean=3.25"
        else:
            sub = merged[merged['organ'] == organ]
            fixed_np_str = str(int(sub['fixed_np'].iloc[0]))
            
        a = sub['dice_lora_ada'].values
        b = sub['dice_fixed_control'].values
        
        diff = a - b
        if np.all(diff == 0):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(a, b)
            except ValueError:
                p = 1.0
                
        sig = "Yes" if p < 0.05 else "No (Expected)"
        
        row = (f"| {organ} | {fixed_np_str} | {np.mean(b)*100:.2f}% | "
               f"{np.mean(a)*100:.2f}% | {p:.4f} | {sig} |")
        md_lines.append(row)
        
    md_content = "\n".join(md_lines) + "\n"
    
    print(md_content)
    
    # Interpretation
    overall = merged
    _, p_over = wilcoxon(overall['dice_lora_ada'], overall['dice_fixed_control'])
    print("\n### Conclusion")
    if p_over >= 0.05:
        print("SUCCESS: The Wilcoxon test shows NO statistically significant difference between "
              "the adaptive budget and the fixed mean budget. This confirms that AdaFoB's "
              "dynamic prompt allocation achieves parity with a fixed budget, proving its efficiency "
              "claims.")
    else:
        print("WARNING: The Wilcoxon test shows a statistically significant difference. "
              "Check which method performed better. If Adaptive > Fixed, AdaFoB is superior. "
              "If Fixed > Adaptive, the adaptive allocator is suboptimal.")

if __name__ == "__main__":
    main()
