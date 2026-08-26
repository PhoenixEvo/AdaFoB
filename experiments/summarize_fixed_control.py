import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

def cohen_d(x, y):
    diff = x - y
    if np.std(diff) == 0:
        return 0.0
    return np.mean(diff) / np.std(diff)

def main():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(repo_dir, "results")
    
    # Load control CSVs
    csv_files = glob.glob(os.path.join(results_dir, "control_fixed_mean_budget_gpu*.csv"))
    if not csv_files:
        print("No control evaluation CSVs found!")
        return
        
    dfs = [pd.read_csv(f) for f in csv_files]
    df_control = pd.concat(dfs, ignore_index=True)
    
    # Load original combined CSV
    orig_csv = os.path.join(results_dir, "lora_eval_final_combined.csv")
    if not os.path.exists(orig_csv):
        print(f"Cannot find original results at {orig_csv}")
        return
    df_orig = pd.read_csv(orig_csv)
    
    # Merge on fold, organ, vol_id
    df_merged = pd.merge(df_orig, df_control, on=["fold", "organ", "vol_id"], how="inner")
    
    if len(df_merged) == 0:
        print("Merged dataframe is empty. Check if fold/organ/vol_id match.")
        return
        
    print(f"Merged {len(df_merged)} samples successfully.\n")
    
    organs = df_merged['organ'].unique()
    
    print("| Organ | Mean Dice AdaFoB | Mean Dice Fixed-Control | AdaFoB vs Control (p-value) | Cohen's d | FoB(10) vs Control (p-value) |")
    print("|---|---|---|---|---|---|")
    
    def report_row(name, df_sub):
        ada_dice = df_sub['dice_lora_ada']
        fob_dice = df_sub['dice_lora_fob']
        ctrl_dice = df_sub['dice_fixed_control']
        
        mean_ada = ada_dice.mean() * 100
        mean_ctrl = ctrl_dice.mean() * 100
        
        # AdaFoB vs Control
        if np.allclose(ada_dice, ctrl_dice):
            p_ada_ctrl = 1.0
        else:
            _, p_ada_ctrl = wilcoxon(ada_dice, ctrl_dice)
        d_ada_ctrl = cohen_d(ada_dice, ctrl_dice)
        
        # FoB(10) vs Control
        if np.allclose(fob_dice, ctrl_dice):
            p_fob_ctrl = 1.0
        else:
            _, p_fob_ctrl = wilcoxon(fob_dice, ctrl_dice)
            
        print(f"| {name} | {mean_ada:.2f}% | {mean_ctrl:.2f}% | {p_ada_ctrl:.4f} | {d_ada_ctrl:.3f} | {p_fob_ctrl:.4f} |")

    for organ in sorted(organs):
        df_organ = df_merged[df_merged['organ'] == organ]
        report_row(organ, df_organ)
        
    # Overall
    report_row("OVERALL", df_merged)

if __name__ == "__main__":
    main()
