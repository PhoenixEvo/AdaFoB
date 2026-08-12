import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import os
import glob

def pval_to_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

VALID_NPS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]

def compile_table(csv_pattern):
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        print(f"Error: No files matching {csv_pattern} found.")
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    print(f"\n{'='*110}")
    print(f"                       GATE E: 3D VOLUME ABLATION TABLE (CORRECTED)")
    print(f"{'='*110}")
    
    for organ in df["organ"].unique():
        df_org = df[df["organ"] == organ].copy()
        n_eps = len(df_org)
        
        # 1. Find Best Global Np
        best_global_np = 10
        best_global_mean_dice = -1
        for n in VALID_NPS:
            mean_d = df_org[f"uni_dice_{n}"].mean()
            if mean_d > best_global_mean_dice:
                best_global_mean_dice = mean_d
                best_global_np = n
                
        base_dice = df_org[f"uni_dice_{best_global_np}"].values
        base_hd95 = df_org[f"uni_hd95_{best_global_np}"].values
        
        # 2. Compute Oracle HD95
        oracle_hd95 = []
        for _, row in df_org.iterrows():
            best_n = 10
            best_d = -1
            for n in VALID_NPS:
                if row[f"uni_dice_{n}"] > best_d:
                    best_d = row[f"uni_dice_{n}"]
                    best_n = n
            oracle_hd95.append(row[f"uni_hd95_{best_n}"])
        oracle_hd95 = np.array(oracle_hd95)
        
        print(f"\n--- {organ.upper()} (n={n_eps} volumes) | Best Global Np = {best_global_np} ---")
        
        rows = {
            f"Best Global Baseline (Np={best_global_np})": (base_dice, base_hd95),
            "FoB Baseline (Np=10)": (df_org["dice_fob_base"].values, df_org["hd95_fob_base"].values),
            "Ada. Budget + Uniform Placement": (df_org["dice_adabudget_uni"].values, df_org["hd95_adabudget_uni"].values),
            "Fixed Np=10 + Ada. Placement": (df_org["dice_fixed_10_ada"].values, df_org["hd95_fixed_10_ada"].values),
            "AdaFoB (Ours)": (df_org["dice_adafob"].values, df_org["hd95_adafob"].values),
            "Per-case Oracle Np": (df_org["oracle_dice"].values, oracle_hd95)
        }
        
        print(f"{'Method':<35} | {'Dice (%)':<17} | {'HD95 Median [IQR]':<20} | {'EmptyRate':<10} | {'P-val (vs BestG)'}")
        print("-" * 110)
        
        for name, (d_arr, h_arr) in rows.items():
            mean_d = np.mean(d_arr) * 100
            std_d = np.std(d_arr) * 100
            
            med_h = np.median(h_arr)
            q25_h = np.percentile(h_arr, 25)
            q75_h = np.percentile(h_arr, 75)
            
            # Compute empty rate (Dice < 1e-4 or HD95 > 100 as proxy)
            empty_rate = np.mean((d_arr < 1e-4) | (h_arr > 100)) * 100
            
            if name.startswith("Best Global Baseline"):
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
                
            hd95_str = f"{med_h:>5.1f} [{q25_h:>4.1f}-{q75_h:>4.1f}]"
            dice_str = f"{mean_d:>6.2f} ± {std_d:>5.2f}"
            print(f"{name:<35} | {dice_str:<17} | {hd95_str:<20} | {empty_rate:>5.1f}%    | {pval_str}")

if __name__ == "__main__":
    import os
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 1. Try local results folder first
    local_pattern = os.path.join(repo_dir, "results", "gate_d_results*.csv")
    csv_files = glob.glob(local_pattern)
    
    # 2. Try Kaggle input folders (if user mounted the output as a dataset)
    if not csv_files:
        kaggle_pattern = "/kaggle/input/**/gate_d_results*.csv"
        csv_files = glob.glob(kaggle_pattern, recursive=True)
        
    if csv_files:
        print(f"Found CSV files: {csv_files}")
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        # Create a temporary unified CSV to pass to the original compile_table logic
        tmp_path = "unified_gate_d.csv"
        df.to_csv(tmp_path, index=False)
        compile_table(tmp_path)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    else:
        print("Error: Could not find any gate_d_results*.csv files in /kaggle/working or /kaggle/input")
