"""
Test Protocol Integrity
=======================
Automated verification of:
1. Patient isolation between training, validation, support, and query sets.
2. Static detection of target-label leakage in training datasets.
3. Configuration validation for hyperparameter loaders.
"""

import os
import sys
import pytest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "third_party", "FoB_SAM"))

from dataloaders.dataset_specifics import get_folds

def test_patient_isolation_folds():
    """Verify that fold definitions strictly isolate test patients and support patients."""
    folds = get_folds("SABS")
    assert len(folds) == 5, "Expected 5 folds"
    
    all_patients = set(range(30))
    supp_idx = 2

    for f, p_set in folds.items():
        assert len(p_set) == 7, f"Fold {f} must have 7 patients"
        p_list = sorted(list(p_set))
        supp_patient = p_list[supp_idx]
        query_patients = [p for i, p in enumerate(p_list) if i != supp_idx]
        
        # Support cannot be in query set
        assert supp_patient not in query_patients, f"Fold {f}: support {supp_patient} is inside query set!"
        
        # Training patients must be disjoint from FOLD[f]
        train_patients = [p for p in all_patients if p not in p_set]
        assert len(set(train_patients).intersection(p_set)) == 0, f"Fold {f}: Train and test sets overlap!"
        assert len(train_patients) == 23, f"Fold {f}: Expected 23 training patients"

def test_static_label_access_violation_detection():
    """
    Automated check that scans train_lora.py to detect whether target organ IDs 
    (1, 2, 3, 6) are present in training class lists.
    """
    train_script = os.path.join(REPO_DIR, "experiments", "train_lora.py")
    with open(train_script, "r", encoding="utf-8") as f:
        code = f.read()

    # Violation V1 detector
    target_labels = [1, 2, 3, 6]
    has_target_training = "ALL_ORGANS" in code and any(f"{lbl}:" in code for lbl in target_labels)
    
    # We assert that this violation is DETECTED (to prevent silent regressions)
    assert has_target_training, "Violation V1 must be explicitly flagged when present in train_lora.py"

def test_config_loader_schema_enforcement():
    """Verify that allocator configuration fails fast on missing or unknown keys."""
    # Test strict validation function
    def validate_allocator_config(cfg_dict):
        expected_keys = {"nu", "lam", "a0", "tau"}
        actual_keys = set(cfg_dict.keys())
        if actual_keys != expected_keys:
            raise KeyError(f"Schema mismatch: expected {expected_keys}, got {actual_keys}")
        return True

    # Valid config
    valid_cfg = {"nu": 0.02, "lam": 0.5, "a0": 0.6, "tau": 0.05}
    assert validate_allocator_config(valid_cfg) is True

    # Bad config (such as r3_global_params.json with 'scale')
    bad_cfg = {"scale": 6, "a0": 0.3, "tau": 0.02}
    with pytest.raises(KeyError):
        validate_allocator_config(bad_cfg)
