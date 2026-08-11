import pandas as pd
import numpy as np
import os

def check_g1_gate(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    print("\n--- Gate G1 Evaluation ---")
    
    # Per-organ breakdown
    for organ in df["organ"].unique():
        sub_df = df[df["organ"] == organ].copy()
        n_episodes = len(sub_df)
        
        # Best global Np
        global_dices = []
        n_cols = [c for c in df.columns if c.startswith("dice_")]
        np_list = [int(c.split("_")[1]) for c in n_cols]
        
        for c in n_cols:
            global_dices.append(sub_df[c].mean())
            
        best_global_idx = np.argmax(global_dices)
        best_global_np = np_list[best_global_idx]
        best_global_dice = global_dices[best_global_idx]
        
        # Per-case oracle
        oracle_mean = sub_df["oracle_dice"].mean()
        
        # Headroom
        headroom = oracle_mean - best_global_dice
        
        # Zeros
        zeros_df = sub_df[sub_df["optimal_Np"] == 0]
        zeros_frac = len(zeros_df) / n_episodes
        
        if len(zeros_df) > 0:
            zeros_oracle_mean = zeros_df["oracle_dice"].mean()
            zeros_fob10_mean = zeros_df["fob_10_dice"].mean()
            cost_of_10 = zeros_oracle_mean - zeros_fob10_mean
        else:
            cost_of_10 = 0.0
            
        print(f"\n{organ.upper()} (n={n_episodes}):")
        print(f"  Best Global Np:   {best_global_np} (Dice: {best_global_dice:.4f})")
        print(f"  Per-case Oracle:  {oracle_mean:.4f}")
        print(f"  -> Headroom:      {headroom*100:.2f} Dice points")
        print(f"  Zeros fraction:   {zeros_frac*100:.1f}% ({len(zeros_df)} eps)")
        print(f"  -> Cost of 10:    {cost_of_10*100:.2f} Dice points")
        
        # Check Gate 1
        pass_cond1 = headroom >= 0.015
        pass_cond2 = (zeros_frac >= 0.20) and (cost_of_10 >= 0.03)
        passed = pass_cond1 or pass_cond2
        print(f"  GATE G1 STATUS:   {'PASS' if passed else 'FAIL'}")

if __name__ == "__main__":
    check_g1_gate("results/r1_budget_sweep.csv")
