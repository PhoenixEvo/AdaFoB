"""
Unit Tests for Connected-Component (CC) Filtering (Amendment A1.7)
"""
import numpy as np
import pytest
from models.postprocessing import filter_connected_components

def test_cc_filter_empty():
    empty = np.zeros((100, 100), dtype=np.uint8)
    out = filter_connected_components(empty, np.array([[50, 50]]))
    assert out.sum() == 0

def test_cc_filter_single_component():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 1
    pos = np.array([[30, 30]]) # inside
    out = filter_connected_components(mask, pos)
    assert np.array_equal(out, mask)

def test_cc_filter_removes_distant_island():
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Main organ
    mask[20:40, 20:40] = 1
    # Distant small false-positive speckle (10 pixels)
    mask[80:85, 80:82] = 1
    pos = np.array([[30, 30]]) # inside main organ only

    out = filter_connected_components(mask, pos)
    assert out[80:85, 80:82].sum() == 0, "Distant island was not removed!"
    assert out[20:40, 20:40].sum() == (20 * 20), "Main organ was altered!"

def test_cc_filter_fallback_largest_when_no_pos_point_inside():
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Large component (400 pixels)
    mask[20:40, 20:40] = 1
    # Small component (16 pixels)
    mask[70:74, 70:74] = 1
    # Positive point is completely outside both components
    pos = np.array([[5, 5]])

    out = filter_connected_components(mask, pos)
    assert out.sum() == (20 * 20), "Fallback did not retain largest component!"
    assert out[70:74, 70:74].sum() == 0, "Small component was not removed!"
