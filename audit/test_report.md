# Automated Integrity Test Report (Phase 2 / Gate 2)

**Generated:** 2026-10-21  
**Auditor:** Lead Research Engineer & Independent Methodological Auditor  
**Status:** PASSED (16 / 16 passed, 0 failed)  
**Environment:** Windows 11, Python 3.11.9, PyTorch 2.5.1+cu121, NVIDIA RTX 4050 Laptop GPU (6GB)  

---

## 1. Executive Summary

Phase 2 constructed a rigorous, automated verification harness covering 6 critical integrity suites. The objective of Gate 2 is to ensure that mathematical metrics, coordinate systems, data splits, and model query-invariance properties are verifiably sound before proceeding to any new experimental baseline or model execution.

All 16 automated tests passed in 21.92s.

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.4.2
collected 16 items

tests/test_protocol_integrity.py::test_patient_isolation_folds PASSED            [  6%]
tests/test_protocol_integrity.py::test_static_label_access_violation_detection PASSED [ 12%]
tests/test_protocol_integrity.py::test_config_loader_schema_enforcement PASSED [ 18%]
tests/test_spatial_transforms.py::test_sam_coordinate_convention PASSED          [ 25%]
tests/test_spatial_transforms.py::test_voxel_spacing_numpy_sitk_alignment PASSED [ 31%]
tests/test_spatial_transforms.py::test_categorical_mask_nearest_neighbor_integrity PASSED [ 37%]
tests/test_metrics.py::test_dice_edge_cases PASSED                               [ 43%]
tests/test_metrics.py::test_hd95_analytic_geometric_translation PASSED           [ 50%]
tests/test_metrics.py::test_hd95_anisotropic_spacing_scaling PASSED              [ 56%]
tests/test_metrics.py::test_hd95_empty_prediction_fallback PASSED                [ 62%]
tests/test_evaluation_completeness.py::test_clean_rank4_evaluation_completeness PASSED [ 68%]
tests/test_evaluation_completeness.py::test_contamination_detector_flags_historical_duplicates PASSED [ 75%]
tests/test_no_query_gt_access.py::test_allocator_signature_has_no_query_gt PASSED [ 81%]
tests/test_no_query_gt_access.py::test_inference_query_label_invariance PASSED  [ 87%]
tests/test_fixed_budget_parity.py::test_fixed_budget_prompt_parity PASSED        [ 93%]
tests/test_fixed_budget_parity.py::test_allocator_forced_budget_parity PASSED    [100%]

======================= 16 passed, 6 warnings in 21.92s =======================
```

---

## 2. Test Suite Details

### Suite 1: Protocol Integrity (`tests/test_protocol_integrity.py`)
- **`test_patient_isolation_folds`**: Verifies that 5-fold cross-validation maintains disjoint patient partitions between training support candidates and evaluation queries within each fold.
- **`test_static_label_access_violation_detection`**: Static AST analysis verifying that training scripts (`experiments/train_lora.py`) are flagged for supervisory target-label leakage (Violation V1).
- **`test_config_loader_schema_enforcement`**: Verifies that config schemas reject unauthorized keys, prevent silent fallback to defaults, and catch configuration mismatches (Violation V4).

### Suite 2: Spatial Transforms (`tests/test_spatial_transforms.py`)
- **`test_sam_coordinate_convention`**: Asserts $(x, y) \leftrightarrow (\text{col}, \text{row})$ coordinate ordering conventions between OpenCV contours and SAM prompt coordinate inputs, ensuring no transpose or axis-flipping errors.
- **`test_voxel_spacing_numpy_sitk_alignment`**: Verifies that voxel spacing arrays matching SimpleITK $(s_x, s_y, s_z)$ align properly with NumPy $(z, y, x)$ axes when computing physical surface distances.
- **`test_categorical_mask_nearest_neighbor_integrity`**: Confirms that resampled discrete masks preserve integer class labels without continuous-interpolation artifacts.

### Suite 3: Mathematical Metric Verification (`tests/test_metrics.py`)
- **`test_dice_edge_cases`**: Validates boundary behavior ($1.0$ for matching empty masks, $0.0$ for non-empty vs empty, correct harmonic mean).
- **`test_hd95_analytic_geometric_translation`**: Compares MedPy/SciPy Hausdorff distance calculations on an analytically translated cylinder against exact ground truth ($10.0$ mm shift).
- **`test_hd95_anisotropic_spacing_scaling`**: Validates directional physical scaling under anisotropic voxel grids ($[1.0, 1.0, 5.0]$ mm).
- **`test_hd95_empty_prediction_fallback`**: Enforces strict fallback handling ($373.12$ mm image diagonal) when predictions are completely empty, preventing undefined behavior.

### Suite 4: Evaluation Completeness & Contamination (`tests/test_evaluation_completeness.py`)
- **`test_clean_rank4_evaluation_completeness`**: Verifies that the clean seed-42 re-run contains exactly 120 unique slice evaluations ($30 \times 4$ organs), checking zero missing cells and zero NaN values.
- **`test_contamination_detector_flags_historical_duplicates`**: Confirms that the automated audit flags the historical `results/lora_eval_rank16_cv.csv` as contaminated (78.3% duplicate row collision from parallel run aggregations).

### Suite 5: Query Ground Truth Isolation (`tests/test_no_query_gt_access.py`)
- **`test_allocator_signature_has_no_query_gt`**: Inspects `PromptBudgetAllocator` methods to ensure query ground truth cannot be accepted as an argument.
- **`test_inference_query_label_invariance`**: Verifies that running inference with `qry_labels` set to zeros versus random noise produces identical prompt coordinates and background budget (asserting $0.0$ difference).

### Suite 6: Parity & Allocation Determinism (`tests/test_fixed_budget_parity.py`)
- **`test_fixed_budget_prompt_parity`**: Asserts that baseline FoB inference under fixed budget $N_p=10$ is strictly deterministic given a fixed random seed.
- **`test_allocator_forced_budget_parity`**: Asserts that `PromptBudgetAllocator` sampling returns exactly the requested budget count $N_p$ across input structures (NumPy array, torch Tensor, nested list).

---

## 3. Gate Status Assessment

| Gate | Subject | Status | Evidence |
| :--- | :--- | :--- | :--- |
| **Gate 1** | Protocol & Data Separation | **BLOCKED** | Documented in `audit/protocol_audit.md`. Historical pipeline contains fatal violations V1 (target-organ GT supervision), V2 (test-fold checkpoint selection), and V3 (test-Dice allocator tuning). |
| **Gate 2** | Automated Verification Harness | **PASSED** | 16 / 16 tests passing cleanly. Metric calculation, spatial transforms, query GT invariance, and completeness checks are fully operational and verified. |

### Prerequisites for Gate 3 (Baseline Fair Execution)
Before executing baseline runs or further experiments, the Protocol Violation V1 must be structurally resolved with user alignment:
1. **Option A (Strict Setting I):** Supervoxel self-supervision or training-free adaptive prompting on frozen SAM.
2. **Option B (Declared In-Domain Adaptation):** Maintain target organ LoRA training with strictly held-out cross-validation splits (no test fold leakage) and explicit literature scope qualifiers (`[SCOPE MISMATCH]`).
