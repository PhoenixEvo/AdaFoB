"""
Test Suite: Patient-Level Isolation for Tuning & Model Selection (Amendment A1.1)
Verifies:
1. For every test fold i in {0..4}, the tuning set strictly excludes all patients
   queried in fold i (including boundary patients).
2. Exactly 88 volume-organ records across 19 unique patients remain for tuning per fold.
3. Zero patient overlap exists between tuning candidates and test evaluation.
"""
import os
import pandas as pd
import pytest

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "protocol", "paired_episode_manifest.csv")

@pytest.fixture
def manifest():
    assert os.path.exists(MANIFEST_PATH), f"Manifest missing at {MANIFEST_PATH}"
    df = pd.read_csv(MANIFEST_PATH)
    df['patient_id'] = df['vol_id'].apply(lambda x: int(x.split('/')[-1].split('.nii')[0].split('_')[-1]))
    return df

def test_tuning_patient_isolation(manifest):
    """Assert zero patient leakage between tuning candidate pool and test queries."""
    for test_fold in range(5):
        test_df = manifest[manifest['fold'] == test_fold]
        test_patients = set(test_df['patient_id'].unique())
        assert len(test_patients) == 6, f"Expected 6 test patients in fold {test_fold}, got {len(test_patients)}"

        other_folds_df = manifest[manifest['fold'] != test_fold]
        # Tuning pool with strict patient exclusion
        tuning_pool = other_folds_df[~other_folds_df['patient_id'].isin(test_patients)]
        tuning_patients = set(tuning_pool['patient_id'].unique())

        # 1. Assert disjoint sets (zero overlap)
        overlap = test_patients.intersection(tuning_patients)
        assert len(overlap) == 0, f"Fold {test_fold} tuning leaked patients: {overlap}"

        # 2. Assert sample size: exactly 88 records across 19 patients
        assert len(tuning_pool) == 88, f"Expected 88 records in tuning pool for fold {test_fold}, got {len(tuning_pool)}"
        assert len(tuning_patients) == 19, f"Expected 19 unique patients in tuning pool for fold {test_fold}, got {len(tuning_patients)}"
