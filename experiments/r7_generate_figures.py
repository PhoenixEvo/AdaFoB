import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def plot_r7_figures(csv_path="results/r2_raw_metrics.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    # Only use Setting I for the main figures
    df = df[df["setting"] == "I"].copy()
    
    os.makedirs("results/figures", exist_ok=True)
    
    # Pre-calculate optimal uniform Np (oracle Np) for each episode
    valid_nps = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]
    
    optimal_nps = []
    oracle_dices = []
    
    for _, row in df.iterrows():
        best_d = -1
        best_np = 0
        for n in valid_nps:
            d = row[f"uni_dice_{n}"]
            if d > best_d + 1e-4: # tie break to smaller Np
                best_d = d
                best_np = n
        optimal_nps.append(best_np)
        oracle_dices.append(best_d)
        
    df["optimal_Np"] = optimal_nps
    df["oracle_dice"] = oracle_dices
    df["fob_10_dice"] = df["uni_dice_10"]
    
    # 1. Histogram of N* per organ
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="optimal_Np", hue="organ", multiple="dodge", 
                 bins=np.arange(-0.5, 25.5, 1), palette="Set2")
    plt.title("Fig 1: Distribution of Oracle Budget (N*) by Organ")
    plt.xlabel("Optimal Number of Negative Prompts (N*)")
    plt.ylabel("Count of Episodes")
    plt.xlim(-1, 25)
    plt.tight_layout()
    plt.savefig("results/figures/fig1_optimal_Np_hist.png", dpi=300)
    plt.close()
    
    # 2. Scatter: Negative-Prompt Gain vs Ambiguity Score (a)
    # Gain = Dice(N=10) - Dice(N=0)
    df["gain_10_vs_0"] = df["uni_dice_10"] - df["uni_dice_0"]
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="a_score", y="gain_10_vs_0", hue="organ", 
                    alpha=0.6, palette="Set2", s=50)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.title("Fig 2: Gain from 10 Negatives vs. Ambiguity Score")
    plt.xlabel("Ambiguity Score (a)")
    plt.ylabel("Dice(N=10) - Dice(N=0)")
    plt.tight_layout()
    plt.savefig("results/figures/fig2_gain_vs_a.png", dpi=300)
    plt.close()
    
    # 3. Cost of forcing budget stratified by a
    # We bin 'a' into low (0-0.3), med (0.3-0.6), high (0.6-1.0)
    # Then plot Dice cost = Oracle Dice - FoB(N=10) Dice
    df["cost_of_10"] = df["oracle_dice"] - df["fob_10_dice"]
    df["a_bin"] = pd.cut(df["a_score"], bins=[0, 0.3, 0.6, 1.0], labels=["Low (<0.3)", "Med (0.3-0.6)", "High (>0.6)"])
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="a_bin", y="cost_of_10", palette="pastel")
    plt.title("Fig 3: Cost of Fixed N=10 Stratified by Ambiguity")
    plt.xlabel("Ambiguity Score Bin")
    plt.ylabel("Dice Cost (Oracle N* - N=10)")
    plt.tight_layout()
    plt.savefig("results/figures/fig3_cost_vs_a.png", dpi=300)
    plt.close()
    
    print("Generated Figures 1, 2, 3 in results/figures/")

if __name__ == "__main__":
    plot_r7_figures()
