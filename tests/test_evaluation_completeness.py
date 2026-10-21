"""
Test Evaluation Completeness and Contamination Detection
========================================================
Verifies:
1. Completeness of clean evaluation CSV files (exact 120 paired rows, 5 folds, 4 organs).
2. Detection of missing or duplicate (fold, organ, vol_id) entries.
3. Automated detection of cross-rank file duplication (verifying the contamination audit detector).
"""

import os
import numpy as np
import pandas as pd
import pytest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_clean_rank4_evaluation_completeness():
    """Verify that clean_rerun_rank4_seed42_combined.csv is complete, non-redundant, and valid."""
    csv_path = os.path.join(REPO_DIR, "results", "clean_rerun_rank4_seed42_combined.csv")
    if not os.path.exists(csv_path):
        pytest.skip(f"Clean CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    assert len(df) == 120, f"Expected 120 rows, got {len(df)}"

    # Check folds
    for f in range(5):
        sub_f = df[df["fold"] == f]
        assert len(sub_f) == 24, f"Fold {f} must have 24 rows, got {len(sub_f)}"
        for organ in ["LIVER", "LK", "RK", "SPLEEN"]:
            sub_o = sub_f[sub_f["organ"] == organ]
            assert len(sub_o) == 6, f"Fold {f} organ {organ} must have 6 rows, got {len(sub_o)}"

    # Check unique keys
    assert not df.duplicated(subset=["fold", "organ", "vol_id"]).any(), "Found duplicate (fold, organ, vol_id) entries!"

    # Check no NaN / Inf
    metric_cols = [c for c in df.columns if "dice" in c or "hd95" in c]
    for c in metric_cols:
        assert not df[c].isna().any(), f"Column {c} contains NaN values!"
        assert not np.isinf(df[c]).any(), f"Column {c} contains Inf values!"

def test_contamination_detector_flags_historical_duplicates():
    """
    Test that the cross-rank duplication audit successfully flags 
    the known corruption between historical rank2 and rank8 CSVs.
    """
    r2_path = os.path.join(REPO_DIR, "results", "lora_eval_rank2_combined.csv")
    r8_path = os.path.join(REPO_DIR, "results", "lora_eval_rank8_combined.csv")

    if not (os.path.exists(r2_path) and os.path.exists(r8_path)):
        pytest.skip("Historical rank CSVs not present")

    df_r2 = pd.read_csv(r2_path)
    df_r8 = pd.read_csv(r8_path)

    # Check duplicated rows on hd95_adafob_2d
    duplicate_rows = np.isclose(df_r2["hd95_adafob_2d"], df_r8["hd95_adafob_2d"], atol=1e-6).sum()
    pct_dup = duplicate_rows / len(df_r2)

    # We assert that the detector flags severe contamination (> 50% duplicate rows)
    assert pct_dup > 0.50, f"Contamination detector failed: only {pct_dup*100:.1f}% duplicates detected"
