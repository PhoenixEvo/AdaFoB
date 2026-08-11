import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import os

def pval_to_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

def compile_table(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Identify Fold 0 for best global Np calculation
    df_fold0 = df[df["fold"] == 0]
    best_global_np = 0
    best_global_dice = -1
    for np_val in [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]:
        col = f"uni_dice_{np_val}"
        mean_dice = df_fold0[col].mean()
        if mean_dice > best_global_dice:
            best_global_dice = mean_dice
            best_global_np = np_val
            
    print(f"Best global fixed Np (computed from Fold 0): {best_global_np} (Dice={best_global_dice:.4f})")
    
    # Process each Setting and Organ
    for setting in df["setting"].unique():
        print(f"\n{'='*80}")
        print(f"                       SETTING {setting} ABLATION TABLE")
        print(f"{'='*80}")
        
        df_set = df[df["setting"] == setting]
        
        for organ in df_set["organ"].unique():
            df_org = df_set[df_set["organ"] == organ].copy()
            n_eps = len(df_org)
            
            print(f"\n--- {organ.upper()} (n={n_eps}) ---")
            
            # Row 1: FoB (Np=10)
            base_dice = df_org["uni_dice_10"].values
            base_hd = df_org["uni_hd95_10"].values
            
            # Row 2: Equal-mean budget
            mean_ada = df_org["adaptive_budget"].mean()
            fixed_mean_np = int(np.round(mean_ada))
            
            # Find closest valid Np list value for fixed_mean_np
            valid_nps = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]
            closest_mean_np = min(valid_nps, key=lambda x: abs(x - fixed_mean_np))
            
            # Precompute row arrays
            rows = {
                "FoB Baseline (Np=10)": (base_dice, base_hd),
                f"Fixed Np = {closest_mean_np} (Mean-adaptive)": (df_org[f"uni_dice_{closest_mean_np}"].values, df_org[f"uni_hd95_{closest_mean_np}"].values),
                f"Fixed Np = {best_global_np} (Best global)": (df_org[f"uni_dice_{best_global_np}"].values, df_org[f"uni_hd95_{best_global_np}"].values),
            }
            
            # Compute dynamic arrays
            ada_budget_vals = df_org["adaptive_budget"].values
            
            # Row 4: Adaptive Budget + Uniform Placement
            dice_4, hd_4 = [], []
            for i, row in df_org.iterrows():
                b = int(row["adaptive_budget"])
                cb = min(valid_nps, key=lambda x: abs(x - b))
                dice_4.append(row[f"uni_dice_{cb}"])
                hd_4.append(row[f"uni_hd95_{cb}"])
            rows["Ada. Budget + Uniform Placement"] = (np.array(dice_4), np.array(hd_4))
            
            # Row 5: Fixed Budget (10) + Adaptive Placement
            rows["Fixed Np=10 + Ada. Placement"] = (df_org["ada_dice_10"].values, df_org["ada_hd95_10"].values)
            
            # Row 6: AdaFoB (Ours)
            dice_6, hd_6 = [], []
            for i, row in df_org.iterrows():
                b = int(row["adaptive_budget"])
                cb = min(valid_nps, key=lambda x: abs(x - b))
                dice_6.append(row[f"ada_dice_{cb}"])
                hd_6.append(row[f"ada_hd95_{cb}"])
            rows["AdaFoB (Ours)"] = (np.array(dice_6), np.array(hd_6))
            
            # Row 7: Oracle
            dice_7, hd_7 = [], []
            for i, row in df_org.iterrows():
                best_d = -1; best_h = 100
                for n in valid_nps:
                    if row[f"uni_dice_{n}"] > best_d:
                        best_d = row[f"uni_dice_{n}"]
                        best_h = row[f"uni_hd95_{n}"]
                dice_7.append(best_d)
                hd_7.append(best_h)
            rows["Per-case Oracle Np"] = (np.array(dice_7), np.array(hd_7))
            
            # Print table
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
                    # Wilcoxon signed-rank test
                    diff = d_arr - base_dice
                    # If all differences are 0, wilcoxon raises ValueError
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
    compile_table("results/r2_raw_metrics.csv")
