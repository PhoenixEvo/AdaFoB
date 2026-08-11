import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Plotting E4 Budget Distribution...")
    
    npy_path = "results/e4_budget_distributions.npy"
    if not os.path.exists(npy_path):
        print(f"Error: {npy_path} not found.")
        return
        
    distributions = np.load(npy_path, allow_pickle=True).item()
    
    # distributions[8] is Aorta, distributions[4] is Gallbladder
    aorta_budgets = distributions.get(8, [])
    gallbladder_budgets = distributions.get(4, [])
    
    if not aorta_budgets or not gallbladder_budgets:
        print("Error: Missing data in npy file.")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Histogram for Aorta
    plt.hist(aorta_budgets, bins=np.arange(-0.5, 25.5, 1), alpha=0.6, label='Aorta (Regime I)', color='red', edgecolor='black')
    
    # Histogram for Gallbladder
    plt.hist(gallbladder_budgets, bins=np.arange(-0.5, 25.5, 1), alpha=0.6, label='Gallbladder (Regime II)', color='blue', edgecolor='black')
    
    plt.axvline(x=24, color='black', linestyle='--', label='Baseline FoB Fixed Budget (Np=24)')
    plt.axvline(x=10, color='gray', linestyle='--', label='Baseline FoB Fallback Budget (Np=10)')
    
    plt.title('AdaFoB Dynamic Budget Allocation Distribution', fontsize=14)
    plt.xlabel('Allocated Budget (Np)', fontsize=12)
    plt.ylabel('Frequency (Episodes)', fontsize=12)
    plt.xticks(np.arange(0, 25, 2))
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    out_path = "results/fig4_budget_distribution.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
