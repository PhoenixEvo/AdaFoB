"""
Standardized Post-Processing Module for AdaFoB Research Project
Frozen under Pre-Run Amendment A1.7.
"""
import numpy as np
import cv2

def filter_connected_components(pred_mask: np.ndarray, pos_points: np.ndarray = None) -> np.ndarray:
    """
    Connected-Component (CC) spatial filtering rule:
    1. If pred_mask is empty, return empty mask.
    2. Label connected components using 8-connectivity.
    3. Retain all connected components that contain at least one positive prompt point.
    4. If no component contains a positive prompt point (or if pos_points is None/empty),
       retain the single largest connected component.
    5. Return the binary filtered mask (same dtype as input).

    Args:
        pred_mask: 2D binary numpy array (H, W) with values in {0, 1}.
        pos_points: Optional array of shape (N, 2) with (x, y) = (col, row) coordinates.

    Returns:
        Filtered 2D binary numpy array (H, W).
    """
    if pred_mask is None or pred_mask.sum() == 0:
        return np.zeros_like(pred_mask) if pred_mask is not None else np.zeros((256, 256), dtype=np.uint8)

    mask_u8 = pred_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(pred_mask)

    H, W = pred_mask.shape
    matching_components = set()

    # Check which components contain positive prompt points
    if pos_points is not None and len(pos_points) > 0:
        pos_arr = np.array(pos_points).reshape(-1, 2)
        for pt in pos_arr:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < W and 0 <= y < H:
                c = labels[y, x]
                if c > 0:
                    matching_components.add(c)

    # If components matched positive prompts, retain all of them
    if len(matching_components) > 0:
        filtered = np.isin(labels, list(matching_components)).astype(pred_mask.dtype)
        return filtered

    # Fallback: retain the single largest component by area (excluding background index 0)
    # stats[:, cv2.CC_STAT_AREA] has area for each label
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    filtered = (labels == largest_label).astype(pred_mask.dtype)
    return filtered
