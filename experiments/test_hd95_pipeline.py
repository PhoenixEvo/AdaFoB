import numpy as np
import sys
import os

# Import the hd95 function from eval_lora
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.eval_lora import compute_volume_hd95

def test_hd95_pipeline():
    print("Testing HD95 Pipeline...")
    
    # Create 100x200x200 volume (Z, Y, X)
    shape = (100, 200, 200)
    
    # spacing_x=0.7, spacing_y=0.8, spacing_z=3.0
    # This is what SimpleITK GetSpacing() returns
    spacing = (0.7, 0.8, 3.0) 
    
    # Test 1: Offset along X-axis
    gt1 = np.zeros(shape, dtype=np.uint8)
    pred1 = np.zeros(shape, dtype=np.uint8)
    gt1[50, 100, 100] = 1
    pred1[50, 100, 150] = 1 # 50 pixels along X-axis
    
    # Expected distance = 50 * spacing_x = 50 * 0.7 = 35.0 mm
    dist1 = compute_volume_hd95(pred1, gt1, spacing)
    print(f"Test 1 (X-axis offset): Expected 35.0, Got {dist1:.2f}")
    assert np.isclose(dist1, 35.0), f"Test 1 failed! Got {dist1}"
    
    # Test 2: Offset along Z-axis
    gt2 = np.zeros(shape, dtype=np.uint8)
    pred2 = np.zeros(shape, dtype=np.uint8)
    gt2[50, 100, 100] = 1
    pred2[60, 100, 100] = 1 # 10 pixels along Z-axis
    
    # Expected distance = 10 * spacing_z = 10 * 3.0 = 30.0 mm
    dist2 = compute_volume_hd95(pred2, gt2, spacing)
    print(f"Test 2 (Z-axis offset): Expected 30.0, Got {dist2:.2f}")
    assert np.isclose(dist2, 30.0), f"Test 2 failed! Got {dist2}"
    
    # Test 3: Offset along Y-axis
    gt3 = np.zeros(shape, dtype=np.uint8)
    pred3 = np.zeros(shape, dtype=np.uint8)
    gt3[50, 100, 100] = 1
    pred3[50, 120, 100] = 1 # 20 pixels along Y-axis
    
    # Expected distance = 20 * spacing_y = 20 * 0.8 = 16.0 mm
    dist3 = compute_volume_hd95(pred3, gt3, spacing)
    print(f"Test 3 (Y-axis offset): Expected 16.0, Got {dist3:.2f}")
    assert np.isclose(dist3, 16.0), f"Test 3 failed! Got {dist3}"

    print("All regression tests passed!")

if __name__ == '__main__':
    test_hd95_pipeline()
