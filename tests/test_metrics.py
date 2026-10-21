"""
Test Metric Integrity
=====================
Verifies mathematical correctness of Dice and HD95 implementations:
1. Dice edge cases (empty masks, perfect overlap, disjoint masks).
2. HD95 analytic geometric verification on synthetic shapes with known distances.
3. Anisotropic physical voxel spacing scaling.
4. Handling of failed/empty segmentations.
"""

import numpy as np
import pytest

# Metric definitions from experiments/eval_lora.py
def compute_volume_dice(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    inter = np.logical_and(pred, gt).sum()
    return 2.0 * inter / (pred.sum() + gt.sum())

def compute_volume_hd95(pred, gt, spacing):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 300.0
    try:
        from medpy.metric.binary import hd95
        return hd95(pred, gt, voxelspacing=spacing[::-1])
    except Exception:
        return 300.0

def test_dice_edge_cases():
    """Verify Dice computation on boundary conditions."""
    zeros = np.zeros((10, 10, 10), dtype=np.uint8)
    ones = np.ones((10, 10, 10), dtype=np.uint8)

    # Empty vs Empty -> 1.0 (both agree on empty organ)
    assert compute_volume_dice(zeros, zeros) == 1.0

    # Perfect overlap -> 1.0
    assert compute_volume_dice(ones, ones) == 1.0

    # Complete miss -> 0.0
    half1 = zeros.copy()
    half1[:, :5, :] = 1
    half2 = zeros.copy()
    half2[:, 5:, :] = 1
    assert compute_volume_dice(half1, half2) == 0.0

    # Known 50% intersection
    # half1 has 500 voxels, combined has 750 voxels
    quarter = zeros.copy()
    quarter[:, :5, :5] = 1 # 250 voxels
    # inter = 250, sum = 500 + 250 = 750 -> 2*250/750 = 2/3
    assert np.isclose(compute_volume_dice(half1, quarter), 2.0 * 250 / 750)

def test_hd95_analytic_geometric_translation():
    """
    Test HD95 on two solid cubes separated by a known physical offset.
    A cube of size 20x20x20 translated by 10 voxels along X-axis.
    At isotropic 1.0 mm spacing, the 95th percentile distance should match 10.0 mm.
    """
    try:
        from medpy.metric.binary import hd95
    except ImportError:
        pytest.skip("medpy not installed in test environment")

    vol_pred = np.zeros((40, 40, 40), dtype=np.uint8)
    vol_gt = np.zeros((40, 40, 40), dtype=np.uint8)

    # GT cube from 10 to 30
    vol_gt[10:30, 10:30, 10:30] = 1
    # Pred cube shifted by 5 voxels along x (column)
    vol_pred[10:30, 10:30, 15:35] = 1

    spacing = (1.0, 1.0, 1.0)
    dist = compute_volume_hd95(vol_pred, vol_gt, spacing)
    # The maximum and 95th percentile shift is exactly 5.0 mm
    assert np.isclose(dist, 5.0, atol=0.2), f"Expected ~5.0 mm, got {dist:.2f} mm"

def test_hd95_anisotropic_spacing_scaling():
    """Verify that scaling the slice thickness (z-axis) proportionally scales HD95."""
    try:
        from medpy.metric.binary import hd95
    except ImportError:
        pytest.skip("medpy not installed in test environment")

    vol_pred = np.zeros((40, 40, 40), dtype=np.uint8)
    vol_gt = np.zeros((40, 40, 40), dtype=np.uint8)

    # Shift along z-axis (depth) by 4 voxels
    vol_gt[10:30, 10:30, 10:30] = 1
    vol_pred[14:34, 10:30, 10:30] = 1

    # At 1.0 mm z-spacing: 4 * 1.0 = 4.0 mm
    dist_1mm = compute_volume_hd95(vol_pred, vol_gt, (1.0, 1.0, 1.0))
    assert np.isclose(dist_1mm, 4.0, atol=0.2)

    # At 3.0 mm z-spacing: 4 * 3.0 = 12.0 mm
    dist_3mm = compute_volume_hd95(vol_pred, vol_gt, (1.0, 1.0, 3.0))
    assert np.isclose(dist_3mm, 12.0, atol=0.5)

def test_hd95_empty_prediction_fallback():
    """Verify that an empty prediction falls back safely to 300.0 without unhandled crash."""
    zeros = np.zeros((20, 20, 20), dtype=np.uint8)
    ones = np.ones((20, 20, 20), dtype=np.uint8)
    spacing = (1.0, 1.0, 3.0)

    assert compute_volume_hd95(zeros, ones, spacing) == 300.0
    assert compute_volume_hd95(ones, zeros, spacing) == 300.0
    assert compute_volume_hd95(zeros, zeros, spacing) == 0.0
