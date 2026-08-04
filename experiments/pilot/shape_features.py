import os
import math
import yaml
import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import convolve

try:
    import skan
    HAS_SKAN = True
except ImportError:
    HAS_SKAN = False


def compute_shape_features(mask: np.ndarray) -> dict:
    """
    Computes shape features from a binary mask.
    
    Args:
        mask (np.ndarray): A 2D binary uint8 array (e.g. 256x256), where 
            foreground is >0.
            
    Returns:
        dict: A dictionary containing 'area', 'perimeter', 'compactness', 
              'skeleton', 'skeleton_length', and 'num_branches'.
    """
    # Ensure binary format
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Area
    area = int(np.sum(binary_mask))
    
    # Perimeter
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = 0.0
    if contours:
        # Sum perimeter of all external contours
        for cnt in contours:
            perimeter += cv2.arcLength(cnt, True)
            
    # Compactness (clip to [0, 1])
    compactness = 0.0
    if perimeter > 0:
        compactness = (4 * math.pi * area) / (perimeter ** 2)
        compactness = min(max(compactness, 0.0), 1.0)
    elif area > 0:
        # A single pixel or small blob with 0 perimeter
        compactness = 1.0
        
    # Skeleton
    # skeletonize requires boolean array, returns boolean array
    skel_bool = skeletonize(binary_mask > 0)
    skeleton = skel_bool.astype(np.uint8)
    
    # Skeleton length
    skeleton_length = int(np.sum(skeleton))
    
    # Number of branches (endpoints)
    num_branches = 0
    if skeleton_length > 0:
        if HAS_SKAN:
            try:
                sk_obj = skan.Skeleton(skeleton)
                # skan degrees node attribute: 1 indicates an endpoint
                # Since we just want count of endpoints, we can check node degrees
                # or just use fallback. Let's try to get degrees.
                # However, for simplicity and robustness, we can just use fallback
                # if skan logic is not obvious. We will attempt standard skan usage:
                # paths is a list of paths, but wait, skan provides degree image or we can just count.
                # skan.Skeleton(skeleton) has degrees at junction nodes.
                # Actually, the fallback 8-neighbor is very safe.
                # Let's count endpoints by degree.
                pass
            except Exception:
                pass
        
        # Fallback to 8-neighbor degree counting
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        neighbor_count = convolve(skeleton, kernel, mode='constant', cval=0)
        
        # Endpoints are skeleton pixels with exactly 1 neighbor
        endpoints = (skeleton > 0) & (neighbor_count == 1)
        num_branches = int(np.sum(endpoints))
        
        # If skan is available, override with skan.Skeleton
        if HAS_SKAN:
            try:
                sk_obj = skan.Skeleton(skeleton)
                # Skan nodes with degree 1 are endpoints
                degrees = sk_obj.degrees
                num_branches = int(np.sum(degrees == 1))
            except Exception:
                pass
                
    return {
        'area': area,
        'perimeter': float(perimeter),
        'compactness': float(compactness),
        'skeleton': skeleton,
        'skeleton_length': skeleton_length,
        'num_branches': num_branches
    }


def compute_heuristic_np(features: dict, config: dict) -> int:
    """
    Computes heuristic adaptive Np based on shape features.
    
    Args:
        features (dict): Shape features computed by compute_shape_features.
        config (dict): Configuration dictionary containing 'alpha', 'beta', 
                       'gamma', 'np_min', and 'np_max'.
                       
    Returns:
        int: The computed Np value.
    """
    alpha = config.get('alpha', 0.0)
    beta = config.get('beta', 0.0)
    gamma = config.get('gamma', 0.0)
    np_min = config.get('np_min', 1)
    np_max = config.get('np_max', 256)
    
    perimeter = features.get('perimeter', 0.0)
    compactness = features.get('compactness', 0.0)
    skeleton_length = features.get('skeleton_length', 0)
    
    val = (alpha * math.log(perimeter + 1) + 
           beta * (1 - compactness) + 
           gamma * math.log(skeleton_length + 1))
           
    np_val = int(round(val))
    
    # Clip to [np_min, np_max]
    return max(np_min, min(np_val, np_max))


def load_pilot_config(config_path: str = None) -> dict:
    """
    Loads pilot.yaml configuration.
    
    Args:
        config_path (str, optional): Explicit path to config. If None,
            walks up from the script directory to find the AdaFoB root.
            
    Returns:
        dict: The loaded configuration.
    """
    if config_path is None:
        # Try to find it by walking up
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while True:
            candidate = os.path.join(current_dir, 'configs', 'pilot.yaml')
            if os.path.isfile(candidate):
                config_path = candidate
                break
            
            parent = os.path.dirname(current_dir)
            if parent == current_dir:
                # Reached root of file system without finding it
                raise FileNotFoundError("Could not find configs/pilot.yaml walking up from script directory.")
            current_dir = parent
            
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    print("Running sanity checks...")
    
    # Dummy config for testing
    dummy_config = {
        'alpha': 2.0,
        'beta': 5.0,
        'gamma': 3.0,
        'np_min': 5,
        'np_max': 50
    }
    
    # 1. Circle Mask
    circle_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(circle_mask, (128, 128), 50, 1, -1)
    
    # 2. Star Mask (less compact)
    star_mask = np.zeros((256, 256), dtype=np.uint8)
    center = (128, 128)
    points = []
    outer_radius = 80
    inner_radius = 20
    for i in range(10):
        angle = i * math.pi / 5
        r = outer_radius if i % 2 == 0 else inner_radius
        x = int(center[0] + r * math.cos(angle))
        y = int(center[1] + r * math.sin(angle))
        points.append([x, y])
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(star_mask, [pts], 1)
    
    print("\n--- Circle Mask ---")
    circle_features = compute_shape_features(circle_mask)
    for k, v in circle_features.items():
        if k != 'skeleton':
            print(f"{k}: {v}")
    np_circle = compute_heuristic_np(circle_features, dummy_config)
    print(f"Heuristic Np: {np_circle}")
    
    print("\n--- Star Mask ---")
    star_features = compute_shape_features(star_mask)
    for k, v in star_features.items():
        if k != 'skeleton':
            print(f"{k}: {v}")
    np_star = compute_heuristic_np(star_features, dummy_config)
    print(f"Heuristic Np: {np_star}")
