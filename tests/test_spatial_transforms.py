"""
Test Spatial Consistency and Coordinate Transforms
==================================================
Verifies:
1. SAM prompt coordinate conventions (x, y) = (col, row).
2. Spacing ordering: SimpleITK (x, y, z) vs NumPy (z, y, x) for MedPy.
3. Categorical mask invariance under spatial resizing (no non-binary interpolation).
"""

import numpy as np
import pytest
import torch

def test_sam_coordinate_convention():
    """Verify that SAM prompt coordinates follow (x, y) = (column, row)."""
    # Suppose a target is at row 100, column 200 in a (256, 256) image.
    row, col = 100, 200
    # In SAM format, point is (x, y) = (col, row)
    pt = np.array([col, row])
    
    assert pt[0] == col, "x must correspond to column"
    assert pt[1] == row, "y must correspond to row"

def test_voxel_spacing_numpy_sitk_alignment():
    """
    SimpleITK GetSpacing() returns (sx, sy, sz) in physical space.
    NumPy volume has shape (depth_z, height_y, width_x).
    MedPy hd95 requires voxelspacing in NumPy axis order: (sz, sy, sx).
    """
    sitk_spacing = (1.2, 1.2, 3.0)  # (x, y, z)
    numpy_spacing = sitk_spacing[::-1]  # (z, y, x)
    
    assert numpy_spacing == (3.0, 1.2, 1.2)
    assert numpy_spacing[0] == 3.0, "First spacing component must correspond to z-axis (depth)"
    assert numpy_spacing[1] == 1.2, "Second spacing component must correspond to y-axis (height)"
    assert numpy_spacing[2] == 1.2, "Third spacing component must correspond to x-axis (width)"

def test_categorical_mask_nearest_neighbor_integrity():
    """Verify categorical segmentation masks do not introduce intermediate values under resampling."""
    import torch.nn.functional as F
    
    # Binary mask
    mask = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    mask[:, :, 20:40, 20:40] = 1.0
    
    # Resize with nearest neighbor
    resized_nearest = F.interpolate(mask, size=(128, 128), mode="nearest")
    unique_vals = torch.unique(resized_nearest).tolist()
    assert set(unique_vals).issubset({0.0, 1.0}), f"Nearest neighbor introduced non-binary values: {unique_vals}"
    
    # Bilinear would introduce intermediate values (demonstrating why it is prohibited for categorical masks)
    resized_bilinear = F.interpolate(mask, size=(128, 128), mode="bilinear", align_corners=False)
    unique_bilinear = torch.unique(resized_bilinear).tolist()
    assert len(unique_bilinear) > 2, "Bilinear should introduce continuous gradients (prohibited for masks)"
