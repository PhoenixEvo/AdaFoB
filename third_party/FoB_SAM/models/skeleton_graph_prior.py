import numpy as np
import math
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, sobel
from scipy.ndimage import label
from scipy.spatial.distance import cdist

class GAPGenerator:
    """
    Generation of Adaptive Prompts (GAP) using skeleton-derived priors.
    This replaces FoB's dilation-band sampling and ring topology.
    """
    def __init__(self, normal_offset_px: int = 8, np_count: int = 10, k_neighbors: int = 2):
        self.normal_offset_px = normal_offset_px
        self.np_count = np_count
        self.k_neighbors = k_neighbors

    def get_ring_fallback(self, mask: np.ndarray, num_points: int):
        """Fallback to ring sampling if skeleton fails (e.g. empty or tiny mask)."""
        import torch
        import torch.nn.functional as F
        
        # Simple contour sampling (similar to FoB)
        kernel_size = 21
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        label_dilate_9 = F.max_pool2d(mask_t, kernel_size=9, stride=1, padding=4)
        label_dilate_5 = F.max_pool2d(mask_t, kernel_size=15, stride=1, padding=7)
        ring = (label_dilate_9 - label_dilate_5).squeeze().numpy()
        ring = (ring > 0).astype(np.uint8)

        contours, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return np.zeros((num_points, 2), dtype=int)
            
        contour = max(contours, key=cv2.contourArea)
        contour_length = cv2.arcLength(contour, True)
        
        cumulative_lengths = [0]
        for i in range(1, len(contour)):
            pt1 = contour[i - 1][0]
            pt2 = contour[i][0]
            cumulative_lengths.append(cumulative_lengths[-1] + np.linalg.norm(pt2 - pt1))
            
        cumulative_lengths = np.array(cumulative_lengths)
        total_length = cumulative_lengths[-1]
        
        if total_length == 0:
            return np.zeros((num_points, 2), dtype=int)
            
        desired_lengths = np.linspace(0, total_length, num_points, endpoint=False)
        
        sampled_points = []
        idx = 0
        for d in desired_lengths:
            while idx < len(cumulative_lengths) - 1 and cumulative_lengths[idx + 1] < d:
                idx += 1
            pt1 = contour[idx][0]
            pt2 = contour[min(idx + 1, len(contour)-1)][0]
            if cumulative_lengths[idx + 1] - cumulative_lengths[idx] > 0:
                ratio = (d - cumulative_lengths[idx]) / (cumulative_lengths[idx + 1] - cumulative_lengths[idx])
            else:
                ratio = 0
            sampled_point = pt1 + ratio * (pt2 - pt1)
            sampled_points.append(sampled_point)
            
        return np.round(np.array(sampled_points)).astype(int)

    def generate(self, mask: np.ndarray):
        """
        Generates background prompts and their topology graph.
        Returns:
            points: (Np, 2) array of (x, y) coordinates
            adj_matrix: (Np, Np) float adjacency matrix
        """
        rng = np.random.RandomState(hash(mask.tobytes()) % (2**32))
        mask_bool = mask > 0
        skel = skeletonize(mask_bool)
        
        sdf = distance_transform_edt(~mask_bool) - distance_transform_edt(mask_bool)
        smoothed_sdf = cv2.GaussianBlur(sdf.astype(np.float32), (5, 5), 0)
        dy = sobel(smoothed_sdf, axis=0)
        dx = sobel(smoothed_sdf, axis=1)
        
        labeled_skel, num_features = label(skel)
        
        candidates_x = []
        candidates_y = []
        anchor_x = []
        anchor_y = []
        
        if num_features > 0:
            branch_lengths = [np.sum(labeled_skel == i) for i in range(1, num_features + 1)]
            branch_lengths_arr = np.array(branch_lengths, dtype=np.float32)
            probs = branch_lengths_arr / branch_lengths_arr.sum()
            
            sampled_branches = rng.choice(np.arange(1, num_features + 1), size=self.np_count, p=probs, replace=True)
            
            for b in sampled_branches:
                by, bx = np.where(labeled_skel == b)
                idx = rng.choice(len(by))
                sy, sx = by[idx], bx[idx]
                
                gy, gx = dy[sy, sx], dx[sy, sx]
                norm = math.sqrt(gy**2 + gx**2)
                
                if norm > 1e-6:
                    gy, gx = gy / norm, gx / norm
                else:
                    angle = rng.uniform(0, 2 * math.pi)
                    gy, gx = math.sin(angle), math.cos(angle)
                
                ty = int(round(sy + self.normal_offset_px * gy))
                tx = int(round(sx + self.normal_offset_px * gx))
                
                # Try opposite direction if invalid
                if not (0 <= ty < mask.shape[0] and 0 <= tx < mask.shape[1]) or mask_bool[ty, tx]:
                    ty = int(round(sy - self.normal_offset_px * gy))
                    tx = int(round(sx - self.normal_offset_px * gx))
                    
                if 0 <= ty < mask.shape[0] and 0 <= tx < mask.shape[1] and not mask_bool[ty, tx]:
                    candidates_x.append(tx)
                    candidates_y.append(ty)
                    anchor_x.append(sx)
                    anchor_y.append(sy)
        
        # Top up from fallback ring if needed
        needed = self.np_count - len(candidates_x)
        if needed > 0:
            fallback_pts = self.get_ring_fallback(mask, self.np_count)
            # Pick 'needed' points randomly from fallback
            idxs = rng.choice(len(fallback_pts), size=needed, replace=False)
            for idx in idxs:
                tx, ty = fallback_pts[idx]
                candidates_x.append(tx)
                candidates_y.append(ty)
                # For anchors of fallback points, just use the point itself
                anchor_x.append(tx)
                anchor_y.append(ty)
                
        # Trim (just in case)
        candidates_x = candidates_x[:self.np_count]
        candidates_y = candidates_y[:self.np_count]
        anchor_x = anchor_x[:self.np_count]
        anchor_y = anchor_y[:self.np_count]
        
        points = np.column_stack((candidates_x, candidates_y))
        anchors = np.column_stack((anchor_x, anchor_y))
        
        # Sort points clockwise around centroid to maintain consistent ordering for L2 loss
        centroid_x = np.mean(points[:, 0])
        centroid_y = np.mean(points[:, 1])
        angles = np.arctan2(points[:, 1] - centroid_y, points[:, 0] - centroid_x)
        sort_idx = np.argsort(angles)
        
        points = points[sort_idx]
        anchors = anchors[sort_idx]
        
        # Build Adjacency Matrix A^{skeleton}
        dist_mat = cdist(anchors, anchors, metric='euclidean')
        adj_matrix = np.zeros((self.np_count, self.np_count), dtype=np.float32)
        
        for i in range(self.np_count):
            nearest = np.argsort(dist_mat[i])[1 : self.k_neighbors + 1]
            for j in nearest:
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0 # make it symmetric
                
        return points, adj_matrix
