"""Explicit legacy and pixel-centre-restored coordinate variants."""
import numpy as np


def restore_xy(points, source_hw=(256, 256), target_hw=(257, 257)):
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("expected finite N,2 xy coordinates")
    sh, sw = source_hw
    th, tw = target_hw
    if min(sh, sw, th, tw) <= 0:
        raise ValueError("positive grid dimensions required")
    return (p.astype(np.float64) + 0.5) * np.array([tw / sw, th / sh]) - 0.5


def legacy_xy(points):
    return np.asarray(points).copy()
