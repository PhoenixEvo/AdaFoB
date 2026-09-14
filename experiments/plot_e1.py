import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    df = pd.read_csv('results/e1_budget_sweep.csv')
    os.makedirs('results/figures', exist_ok=True)
    
    # 1. Histogram of optimal Np
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='optimal_Np', discrete=True, color='#2c3e50', alpha=0.8)
    plt.title('Distribution of Optimal Negative Prompt Budget ($N^*$)', fontsize=14, pad=15)
    plt.xlabel('Optimal Budget $N^*$', fontsize=12)
    plt.ylabel('Number of Episodes', fontsize=12)
    plt.xticks(np.arange(0, 25, 2))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/figures/fig1_optimal_Np_hist.png', dpi=300)
    plt.close()
    
    # 2. Negative-prompt gain vs ambiguity score a
    # Gain = oracle_dice - dice_0
    df['neg_gain'] = df['oracle_dice'] - df['dice_0']
    
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x='a_score', y='neg_gain', scatter_kws={'alpha':0.5, 'color': '#3498db'}, line_kws={'color': '#e74c3c'})
    plt.title('Negative-Prompt Gain vs. Ambiguity Score $a$', fontsize=14, pad=15)
    plt.xlabel('Ambiguity Score $a$', fontsize=12)
    plt.ylabel('Gain (Oracle Dice - $N_p=0$ Dice)', fontsize=12)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/figures/fig2_gain_vs_a.png', dpi=300)
    plt.close()
    
    # 3. Dice cost of a fixed budget: FoB Np=10 minus per-case oracle
    # Cost = oracle_dice - fob_10_dice
    df['fob_cost'] = df['oracle_dice'] - df['fob_10_dice']
    
    # Stratify by a
    df['a_bin'] = pd.qcut(df['a_score'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='a_bin', y='fob_cost', palette='Set3')
    sns.stripplot(data=df, x='a_bin', y='fob_cost', color='black', alpha=0.3, jitter=True)
    plt.title('Dice Cost of Fixed Budget ($N_p=10$) Stratified by Ambiguity', fontsize=14, pad=15)
    plt.xlabel('Ambiguity Score Quintile', fontsize=12)
    plt.ylabel('Dice Cost (Oracle - Fixed FoB)', fontsize=12)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/figures/fig3_cost_vs_a.png', dpi=300)
    plt.close()
    
    print("Figures saved to results/figures/")

if __name__ == '__main__':
    main()
