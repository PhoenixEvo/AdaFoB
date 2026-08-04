import os
import math
import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, sobel, label
import matplotlib.pyplot as plt

class PromptGenerator:
    """Base class for prompt generators."""
    def generate(self, mask: np.ndarray, np_count: int, image: np.ndarray = None) -> dict:
        """
        Generates prompt points based on the given mask.
        
        Args:
            mask: Binary mask array of shape (H, W).
            np_count: Number of negative (background) prompt points to generate.
            image: Optional original image array.
            
        Returns:
            dict containing:
                - points: np.ndarray of shape (np_count, 2) in (x, y) format.
                - labels: np.ndarray of shape (np_count,) containing all zeros.
                - debug_info: dict with extra information for visualization.
        """
        raise NotImplementedError
    
    def save_debug_overlay(self, mask: np.ndarray, image: np.ndarray, points: np.ndarray, labels: np.ndarray, save_path: str, title: str = '', debug_info: dict = None):
        """
        Saves a debug visualization overlay.
        
        Args:
            mask: Binary mask.
            image: Grayscale image or None.
            points: Prompt points (N, 2).
            labels: Prompt labels (N,).
            save_path: Path to save the PNG image.
            title: Title for the plot.
            debug_info: Optional debug information dict (e.g., containing 'skeleton').
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        if image is not None:
            if image.ndim == 2:
                ax.imshow(image, cmap='gray')
            else:
                ax.imshow(image)
        else:
            # Create a dark background if no image is provided
            ax.imshow(np.zeros_like(mask), cmap='gray', vmin=0, vmax=1)
        
        # Draw mask contour
        mask_uint8 = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            c = contour.reshape(-1, 2)
            c = np.vstack([c, c[0]])
            ax.plot(c[:, 0], c[:, 1], 'g-', linewidth=2, alpha=0.8)

        # Draw skeleton if present
        if debug_info and 'skeleton' in debug_info:
            skel = debug_info['skeleton']
            skel_y, skel_x = np.where(skel > 0)
            ax.plot(skel_x, skel_y, 'b.', markersize=2, alpha=0.6, label='Skeleton')

        # Separate points by label
        pos_points = points[labels == 1]
        neg_points = points[labels == 0]

        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', s=20, label='Background (0)')
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', s=60, marker='*', label='Foreground (1)')
        
        if title:
            ax.set_title(title)
            
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


class RingPriorGenerator(PromptGenerator):
    """Generates background prompts uniformly sampled from a morphological ring band."""
    def __init__(self, r_outer: int = 15, r_inner: int = 13):
        self.r_outer = r_outer
        self.r_inner = r_inner

    def generate(self, mask: np.ndarray, np_count: int, image: np.ndarray = None) -> dict:
        # Use a fixed seed based on mask hash for reproducibility across calls with the same mask
        rng = np.random.RandomState(hash(mask.tobytes()) % (2**32))
        
        kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.r_outer + 1, 2 * self.r_outer + 1))
        kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.r_inner + 1, 2 * self.r_inner + 1))
        
        mask_uint8 = (mask > 0).astype(np.uint8)
        dilated_outer = cv2.dilate(mask_uint8, kernel_outer, iterations=1)
        dilated_inner = cv2.dilate(mask_uint8, kernel_inner, iterations=1)
        
        band = dilated_outer - dilated_inner
        y_coords, x_coords = np.where(band > 0)
        
        # Fallback to contour if band is empty
        if len(y_coords) == 0:
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                c = np.vstack(contours).reshape(-1, 2)
                x_coords, y_coords = c[:, 0], c[:, 1]
            else:
                x_coords, y_coords = np.array([]), np.array([])
                
        if len(y_coords) > 0:
            indices = rng.choice(len(y_coords), size=min(np_count, len(y_coords)), replace=False)
            if len(indices) < np_count:
                # Pad if not enough unique points
                indices = np.pad(indices, (0, np_count - len(indices)), mode='wrap')
            x_sampled = x_coords[indices]
            y_sampled = y_coords[indices]
        else:
            x_sampled = np.zeros(np_count)
            y_sampled = np.zeros(np_count)
            
        points = np.column_stack((x_sampled, y_sampled))
        labels = np.zeros(np_count, dtype=np.int32)
        
        return {
            'points': points,
            'labels': labels,
            'debug_info': {'band': band}
        }


class SkeletonPriorGenerator(PromptGenerator):
    """Generates background prompts by pushing points outwards along mask skeleton normals."""
    def __init__(self, normal_offset_px: int = 8):
        self.normal_offset_px = normal_offset_px

    def generate(self, mask: np.ndarray, np_count: int, image: np.ndarray = None) -> dict:
        rng = np.random.RandomState(hash(mask.tobytes()) % (2**32))
        
        mask_bool = mask > 0
        skel = skeletonize(mask_bool)
        
        # Compute Signed Distance Field to get robust outward normals
        # Positive values outside (background), negative values inside (foreground)
        sdf = distance_transform_edt(~mask_bool) - distance_transform_edt(mask_bool)
        
        # Smooth SDF to get better gradients around ridges
        smoothed_sdf = cv2.GaussianBlur(sdf.astype(np.float32), (5, 5), 0)
        dy = sobel(smoothed_sdf, axis=0)
        dx = sobel(smoothed_sdf, axis=1)
        
        labeled_skel, num_features = label(skel)
        
        candidates_x = []
        candidates_y = []
        branch_lengths = []
        
        if num_features > 0:
            for i in range(1, num_features + 1):
                branch_lengths.append(np.sum(labeled_skel == i))
            
            branch_lengths_arr = np.array(branch_lengths, dtype=np.float32)
            probs = branch_lengths_arr / branch_lengths_arr.sum()
            
            sampled_branches = rng.choice(np.arange(1, num_features + 1), size=np_count, p=probs, replace=True)
            
            for b in sampled_branches:
                by, bx = np.where(labeled_skel == b)
                idx = rng.choice(len(by))
                sy, sx = by[idx], bx[idx]
                
                # Gradient of our SDF points outward
                gy = dy[sy, sx]
                gx = dx[sy, sx]
                norm = math.sqrt(gy**2 + gx**2)
                
                if norm > 1e-6:
                    gy /= norm
                    gx /= norm
                else:
                    # Fallback random normal
                    angle = rng.uniform(0, 2 * math.pi)
                    gy, gx = math.sin(angle), math.cos(angle)
                
                ty = int(round(sy + self.normal_offset_px * gy))
                tx = int(round(sx + self.normal_offset_px * gx))
                
                # Check validity
                if 0 <= ty < mask.shape[0] and 0 <= tx < mask.shape[1]:
                    if not mask_bool[ty, tx]:
                        candidates_x.append(tx)
                        candidates_y.append(ty)
        
        # Top up from RingPriorGenerator if needed
        if len(candidates_x) < np_count:
            ring_gen = RingPriorGenerator(r_outer=15, r_inner=13)
            ring_res = ring_gen.generate(mask, np_count)
            ring_points = ring_res['points']
            
            needed = np_count - len(candidates_x)
            for i in range(needed):
                idx = rng.choice(len(ring_points))
                candidates_x.append(ring_points[idx, 0])
                candidates_y.append(ring_points[idx, 1])
                
        # Trim to requested count
        candidates_x = candidates_x[:np_count]
        candidates_y = candidates_y[:np_count]
        
        points = np.column_stack((candidates_x, candidates_y))
        labels = np.zeros(np_count, dtype=np.int32)
        
        return {
            'points': points,
            'labels': labels,
            'debug_info': {
                'skeleton': skel, 
                'branch_lengths': branch_lengths
            }
        }


def get_mask_centroid(mask: np.ndarray) -> tuple:
    """
    Returns (x, y) of the mask centroid.
    
    Args:
        mask: Binary mask array of shape (H, W).
        
    Returns:
        Tuple of (x, y) integer coordinates.
    """
    M = cv2.moments((mask > 0).astype(np.uint8))
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            cX = int(np.mean(x_coords))
            cY = int(np.mean(y_coords))
        else:
            cX, cY = 0, 0
    return (cX, cY)


if __name__ == '__main__':
    import tempfile
    
    # Create a synthetic ellipse mask for testing
    test_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.ellipse(test_mask, (50, 50), (30, 15), 30, 0, 360, 1, -1)
    
    cx, cy = get_mask_centroid(test_mask)
    pos_points = np.array([[cx, cy]])
    pos_labels = np.array([1])
    
    ring_gen = RingPriorGenerator(r_outer=15, r_inner=13)
    ring_res = ring_gen.generate(test_mask, 10)
    
    skel_gen = SkeletonPriorGenerator(normal_offset_px=15)
    skel_res = skel_gen.generate(test_mask, 10)
    
    r_pts = np.vstack([pos_points, ring_res['points']])
    r_lbls = np.concatenate([pos_labels, ring_res['labels']])
    
    s_pts = np.vstack([pos_points, skel_res['points']])
    s_lbls = np.concatenate([pos_labels, skel_res['labels']])
    
    temp_dir = tempfile.gettempdir()
    r_path = os.path.join(temp_dir, 'debug_ring.png')
    s_path = os.path.join(temp_dir, 'debug_skel.png')
    
    ring_gen.save_debug_overlay(test_mask, None, r_pts, r_lbls, r_path, title='Ring Prior', debug_info=ring_res['debug_info'])
    skel_gen.save_debug_overlay(test_mask, None, s_pts, s_lbls, s_path, title='Skeleton Prior', debug_info=skel_res['debug_info'])
    
    print(f"Saved ring debug to {r_path}")
    print(f"Saved skel debug to {s_path}")
