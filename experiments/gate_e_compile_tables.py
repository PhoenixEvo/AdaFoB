import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import os

def pval_to_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

import glob

def compile_table(csv_pattern):
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        print(f"Error: No files matching {csv_pattern} found.")
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    
    print(f"\n{'='*80}")
    print(f"                       GATE E: 3D VOLUME ABLATION TABLE")
    print(f"{'='*80}")
    
    for organ in df["organ"].unique():
        df_org = df[df["organ"] == organ].copy()
        n_eps = len(df_org)
        
        print(f"\n--- {organ.upper()} (n={n_eps} volumes) ---")
        
        # Base FoB Np=10
        base_dice = df_org["dice_fob_base"].values
        
        rows = {
            "FoB Baseline (Np=10)": (df_org["dice_fob_base"].values, df_org["hd95_fob_base"].values),
            "Ada. Budget + Uniform Placement": (df_org["dice_adabudget_uni"].values, df_org["hd95_adabudget_uni"].values),
            "Fixed Np=10 + Ada. Placement": (df_org["dice_fixed_10_ada"].values, df_org["hd95_fixed_10_ada"].values),
            "AdaFoB (Ours)": (df_org["dice_adafob"].values, df_org["hd95_adafob"].values),
            "Per-case Oracle Np": (df_org["oracle_dice"].values, np.zeros_like(base_dice)) # HD95 oracle not saved properly in row, but we don't care much
        }
        
        print(f"{'Method':<40} | {'Dice (%)':<18} | {'HD95 (mm)':<18} | {'P-value (vs Base)'}")
        print("-" * 105)
        
        for name, (d_arr, h_arr) in rows.items():
            mean_d = np.mean(d_arr) * 100
            std_d = np.std(d_arr) * 100
            mean_h = np.mean(h_arr)
            std_h = np.std(h_arr)
            
            if name.startswith("FoB"):
                pval_str = "-"
            else:
                diff = d_arr - base_dice
                if np.all(diff == 0):
                    pval = 1.0
                else:
                    try:
                        _, pval = wilcoxon(d_arr, base_dice)
                    except ValueError:
                        pval = 1.0
                pval_str = f"{pval:.4f} {pval_to_stars(pval)}"
                
            print(f"{name:<40} | {mean_d:>6.2f} ± {std_d:>5.2f} | {mean_h:>6.2f} ± {std_h:>5.2f} | {pval_str}")

if __name__ == "__main__":
    import os
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    compile_table(os.path.join(repo_dir, "results", "gate_d_results*.csv"))
