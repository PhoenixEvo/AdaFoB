import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

def find_outliers(rank4_csv, rank16_csv, organ='SPLEEN', threshold=100.0):
    """Find cases where HD95 is greater than threshold."""
    df4 = pd.read_csv(rank4_csv)
    df16 = pd.read_csv(rank16_csv)
    
    # Filter by organ
    df4 = df4[df4['organ'] == organ]
    df16 = df16[df16['organ'] == organ]
    
    outliers4 = df4[df4['hd95_lora_ada'] > threshold]
    outliers16 = df16[df16['hd95_lora_ada'] > threshold]
    
    print(f"Rank 4 {organ} outliers (HD95 > {threshold}mm):")
    print(outliers4[['vol_id', 'hd95_lora_ada']])
    
    print(f"\nRank 16 {organ} outliers (HD95 > {threshold}mm):")
    print(outliers16[['vol_id', 'hd95_lora_ada']])
    
    # Return worst case from rank 4
    if len(outliers4) > 0:
        worst_case = outliers4.sort_values(by='hd95_lora_ada', ascending=False).iloc[0]
        return worst_case['vol_id']
    return None

def visualize_case(vol_id):
    # This is a stub for the visualization part since we don't have predictions saved locally.
    # We will need the user to either run this on Kaggle or download the predictions.
    print(f"\nTo visualize {vol_id}:")
    print("We need the NIfTI prediction files for Rank 4 and Rank 16.")
    print("Since Kaggle evaluation only computed metrics and didn't save the full 3D prediction arrays, we'll need to re-run inference for this specific volume to generate the plots.")
    
if __name__ == '__main__':
    r4_csv = 'results/lora_eval_rank4_combined.csv'
    r16_csv = 'results/lora_eval_rank16_combined.csv'
    
    if os.path.exists(r4_csv) and os.path.exists(r16_csv):
        worst_vol = find_outliers(r4_csv, r16_csv)
        if worst_vol:
            visualize_case(worst_vol)
    else:
        print(f"Missing CSV files. Ensure {r4_csv} and {r16_csv} exist.")
