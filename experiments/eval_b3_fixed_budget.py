"""
Baseline B3: Fixed Budget Grid Selection (PREPARATION ONLY — DO NOT RUN CONFIRMATORY)
Complies with Amendment A1.1 (Patient-Isolated Tuning).
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

MANIFEST_PATH = os.path.join(REPO_DIR, "protocol", "paired_episode_manifest.csv")
BUDGET_GRID = [2, 3, 4, 5, 6, 10]

def select_best_fixed_budget_for_fold(test_fold: int, validation_results_df: pd.DataFrame) -> int:
    """
    Selects k* in {2, 3, 4, 5, 6, 10} using ONLY records from patients that do NOT appear in test fold.
    """
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest['patient_id'] = manifest['vol_id'].apply(lambda x: int(x.split('/')[-1].split('.nii')[0].split('_')[-1]))
    test_patients = set(manifest[manifest['fold'] == test_fold]['patient_id'].unique())

    # Filter validation results to strictly exclude test patients
    valid_records = validation_results_df[~validation_results_df['patient_id'].isin(test_patients)]
    assert len(valid_records['patient_id'].unique()) <= 19

    # Find budget that maximizes mean Dice
    best_budget = None
    best_dice = -1.0
    for b in BUDGET_GRID:
        col = f"dice_fixed_{b}"
        if col in valid_records.columns:
            m_dice = valid_records[col].mean()
            if m_dice > best_dice:
                best_dice = m_dice
                best_budget = b

    return best_budget or 4

def smoke_test_dry_run():
    print("[SMOKE TEST] Dry-run on 3 episodes for B3 Fixed Budget...")
    manifest = pd.read_csv(MANIFEST_PATH).head(3)
    print(f"Loaded {len(manifest)} mock episodes.")
    mock_data = []
    for idx, row in manifest.iterrows():
        pat_id = int(row['vol_id'].split('/')[-1].split('.nii')[0].split('_')[-1])
        rec = {"episode_id": row['episode_id'], "fold": row['fold'], "patient_id": pat_id}
        for b in BUDGET_GRID:
            rec[f"dice_fixed_{b}"] = 0.85 + (b * 0.005) + np.random.normal(0, 0.01)
        mock_data.append(rec)
    df_mock = pd.DataFrame(mock_data)
    selected_k = select_best_fixed_budget_for_fold(0, df_mock)
    print(f"[SMOKE TEST PASSED] Selected best fixed budget for Fold 0: Np={selected_k}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", type=int, default=0, help="Run smoke test on N episodes")
    args = parser.parse_args()
    if args.dry_run > 0:
        smoke_test_dry_run()
    else:
        print("B3 is prepared but execution is NOT approved until post-B0-B2 gate clearance.")
