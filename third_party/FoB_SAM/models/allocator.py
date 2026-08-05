import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class PromptBudgetAllocator(nn.Module):
    def __init__(self, max_points=24, device='cuda', nu=0.05, lam=1.0, gamma=1.0, a0=0.4, tau=0.1, alpha=0.35):
        super().__init__()
        self.max_points = max_points
        self.device = device
        
        # Budget parameters
        self.nu = nu         # Prompts per unit boundary length
        self.lam = lam       # Curvature multiplier
        self.gamma = gamma   # Leak risk multiplier
        self.a0 = a0         # Sigmoid center
        self.tau = tau       # Sigmoid temperature
        self.alpha = alpha   # Scale-adaptive offset multiplier
        
        # Ambiguity weights
        self.w = [0.5, 0.3, 0.2]

    def compute_ambiguity_score(self, qry_img, qry_pred_coarse, spt_fg_proto, supp_mask, model, supp_fts):
        """
        Computes the ambiguity score 'a' based on proto, edge, and conf signals.
        """
        B, C, H, W = qry_pred_coarse.shape
        
        # 1. a_proto
        # Extract 24 background prompts from support mask to estimate p_b_bar
        points_spt = model.uniform_sample_contour(supp_mask[0].float(), num_keypoints=24)
        if len(points_spt) > 0:
            heatmaps_spt = model.generate_keypoint_heatmaps((256, 256), points_spt)
            heatmaps_spt = torch.from_numpy(heatmaps_spt).to(self.device)
            skps = []
            for i in range(len(points_spt)):
                skp = [[model.getFeatures(supp_fts, heatmaps_spt[i])]] #[1, 512]
                skp = model.getPrototype(skp)[0].transpose(0, 1)  # [512, 1]
                skps.append(skp)
            skps = torch.stack(skps).squeeze(2) # [24, 512]
            p_b_bar = skps.mean(dim=0, keepdim=True) # [1, 512]
            
            # cosine similarity between spt_fg_proto [1, 512] and p_b_bar [1, 512]
            cos_sim = F.cosine_similarity(spt_fg_proto, p_b_bar, dim=-1).item()
            a_proto = 0.5 * (1 + cos_sim)
        else:
            a_proto = 0.0
            
        # 2. a_edge
        # a_edge = 1 - norm(mean(grad_I) on contour / mean(grad_I) on body)
        # We need gradient of query image (convert to grayscale first)
        qry_img_np = qry_img[0][0].permute(1, 2, 0).cpu().numpy()
        qry_img_gray = cv2.cvtColor(qry_img_np, cv2.COLOR_RGB2GRAY)
        
        # Compute gradient magnitude
        grad_x = cv2.Sobel(qry_img_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(qry_img_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        # Pre-mask M_tilde
        M_tilde = (qry_pred_coarse[0, 0].cpu().numpy() > 0.9).astype(np.uint8)
        
        # Find contour
        contours, _ = cv2.findContours(M_tilde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) > 0:
            contour_mask = np.zeros_like(M_tilde)
            cv2.drawContours(contour_mask, contours, -1, 1, 1)
            
            grad_contour = grad_mag[contour_mask == 1].mean() if contour_mask.sum() > 0 else 0
            grad_body = grad_mag[M_tilde == 1].mean() if M_tilde.sum() > 0 else 1e-5
            grad_body = max(grad_body, 1e-5) # avoid division by zero
            
            ratio = grad_contour / grad_body
            # norm() function: We can use a simple scaling, e.g. 1 - exp(-ratio) or clip(ratio/5, 0, 1)
            # A common norm for ratio in [0, inf) to [0, 1] is ratio / (1 + ratio)
            # If grad on contour is very low relative to body (ratio -> 0), a_edge -> 1 (high ambiguity)
            # The formula says a_edge = 1 - norm(ratio)
            norm_ratio = min(ratio / 5.0, 1.0) # Assuming ratio > 5 means very sharp edge
            a_edge = 1.0 - norm_ratio
        else:
            a_edge = 0.0
            
        # 3. a_conf
        # a_conf = |{u : 0.5 < C(u) < 0.9}| / (|M_tilde| + eps)
        C_map = qry_pred_coarse[0, 0].cpu().numpy()
        uncertain_pixels = np.logical_and(C_map > 0.5, C_map < 0.9).sum()
        body_pixels = M_tilde.sum() + 1e-5
        a_conf = uncertain_pixels / body_pixels
        a_conf = min(a_conf, 1.0)
        
        a = self.w[0] * a_proto + self.w[1] * a_edge + self.w[2] * a_conf
        return a, contours, M_tilde

    def compute_budget(self, a, contours, M_tilde):
        """
        Computes the budget N_p
        """
        if len(contours) == 0:
            return 0
            
        # Compute contour length L and mean curvature
        L = 0
        total_kappa = 0
        valid_kappa_pts = 0
        
        for contour in contours:
            if len(contour) < 3:
                continue
            L += cv2.arcLength(contour, closed=True)
            
            # Simple curvature estimation: angle between adjacent segments
            # We smooth the contour slightly
            pts = contour[:, 0, :]
            if len(pts) >= 5:
                # Calculate curvature using standard formulas or a discrete approximation
                # kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
                # For simplicity, we use angle variation
                dx = np.gradient(pts[:, 0])
                dy = np.gradient(pts[:, 1])
                ddx = np.gradient(dx)
                ddy = np.gradient(dy)
                
                denom = (dx**2 + dy**2)**1.5
                valid_idx = denom > 1e-5
                if valid_idx.sum() > 0:
                    kappa = np.abs(dx[valid_idx]*ddy[valid_idx] - dy[valid_idx]*ddx[valid_idx]) / denom[valid_idx]
                    total_kappa += kappa.sum()
                    valid_kappa_pts += len(kappa)
        
        mean_kappa = total_kappa / valid_kappa_pts if valid_kappa_pts > 0 else 0
        
        g_a = 1.0 / (1.0 + np.exp(-(a - self.a0) / self.tau))
        
        budget_float = self.nu * L * (1 + self.lam * mean_kappa) * g_a
        budget = int(np.round(budget_float))
        
        budget = np.clip(budget, 0, self.max_points)
        return budget

    def get_scale_adaptive_offset(self, M_tilde):
        """
        r(A) = clip(alpha * sqrt(A/pi), 6, 24)
        """
        A = M_tilde.sum()
        r = self.alpha_r * np.sqrt(A / np.pi)
        return int(np.clip(r, 6, 24))

    def sample_placement(self, qry_img, M_tilde, contours, N_p, r):
        """
        Inverse-CDF sampling along offset contour based on leak risk.
        """
        if N_p == 0:
            return np.zeros((0, 2), dtype=np.float32)
            
        # Dilate M_tilde to get the offset contour
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r*2+1, r*2+1))
        M_dilated = cv2.dilate(M_tilde, kernel)
        
        offset_contours, _ = cv2.findContours(M_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(offset_contours) == 0:
            return np.zeros((0, 2), dtype=np.float32)
            
        qry_img_np = qry_img[0][0].permute(1, 2, 0).cpu().numpy()
        qry_img_gray = cv2.cvtColor(qry_img_np, cv2.COLOR_RGB2GRAY)
        
        # We need to sample N_p points proportional to contour lengths
        # But Opus spec says "Process each connected component independently; allocate proportional to length; guarantee >= 1 per component"
        
        pts_list = []
        lengths = [cv2.arcLength(c, True) for c in offset_contours if len(c) > 2]
        valid_contours = [c for c in offset_contours if len(c) > 2]
        
        if len(lengths) == 0:
            return np.zeros((0, 2), dtype=np.float32)
            
        total_L = sum(lengths)
        
        budget_allocs = []
        remaining_budget = N_p
        
        # Guarantee 1 per component
        for i in range(len(valid_contours)):
            budget_allocs.append(1)
            remaining_budget -= 1
            
        if remaining_budget < 0:
            # If we don't even have 1 per component, just take the largest components
            budget_allocs = [0] * len(valid_contours)
            sorted_idx = np.argsort(lengths)[::-1]
            for i in range(N_p):
                budget_allocs[sorted_idx[i]] = 1
        elif remaining_budget > 0:
            # Distribute proportional to length
            for i in range(len(valid_contours)):
                added = int(np.round(remaining_budget * (lengths[i] / total_L)))
                budget_allocs[i] += added
            
            # Fix rounding errors
            while sum(budget_allocs) > N_p:
                budget_allocs[np.argmax(budget_allocs)] -= 1
            while sum(budget_allocs) < N_p:
                budget_allocs[np.argmax(budget_allocs)] += 1
                
        # Now sample for each contour
        for i, contour in enumerate(valid_contours):
            n_samples = budget_allocs[i]
            if n_samples == 0:
                continue
                
            pts = contour[:, 0, :] # [M, 2]
            M = len(pts)
            
            # Compute leak risk l(s)
            # Local normal is hard to compute efficiently in pure Python, we approximate by
            # dilating and eroding the offset contour to get inner/outer bands
            
            contour_mask = np.zeros_like(M_tilde)
            cv2.drawContours(contour_mask, [contour], -1, 1, thickness=1)
            
            # Small dilation for outer, erosion for inner
            small_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            outer_mask = cv2.dilate(contour_mask, small_k) - contour_mask
            inner_mask = contour_mask - cv2.erode(contour_mask, small_k)
            # Fallback to local average intensity inside/outside the whole M_dilated
            
            # To be robust, we just extract I_in and I_out around each point
            l_s = np.zeros(M)
            for j in range(M):
                pt = pts[j]
                x, y = pt[0], pt[1]
                # sample a 5x5 window
                y1, y2 = max(0, y-2), min(256, y+3)
                x1, x2 = max(0, x-2), min(256, x+3)
                window = qry_img_gray[y1:y2, x1:x2]
                mask_win = M_dilated[y1:y2, x1:x2]
                
                I_in = window[mask_win == 1].mean() if (mask_win == 1).sum() > 0 else 0
                I_out = window[mask_win == 0].mean() if (mask_win == 0).sum() > 0 else 0
                
                l_s[j] = np.exp(- ((I_in - I_out)**2) / (2 * 20.0**2))
                
            # Compute curvature kappa(s)
            dx = np.gradient(pts[:, 0])
            dy = np.gradient(pts[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            denom = (dx**2 + dy**2)**1.5
            kappa = np.zeros(M)
            valid_idx = denom > 1e-5
            kappa[valid_idx] = np.abs(dx[valid_idx]*ddy[valid_idx] - dy[valid_idx]*ddx[valid_idx]) / denom[valid_idx]
            
            density = (1 + self.lam * kappa) * (1 + self.gamma * l_s)
            density = density / density.sum()
            
            # Inverse CDF sampling
            cdf = np.cumsum(density)
            # Uniformly spaced probabilities for stratified sampling
            p = np.linspace(0.5/n_samples, 1 - 0.5/n_samples, n_samples)
            sampled_idx = np.searchsorted(cdf, p)
            sampled_idx = np.clip(sampled_idx, 0, M-1)
            
            pts_list.append(pts[sampled_idx])
            
        if len(pts_list) > 0:
            return np.concatenate(pts_list, axis=0)
        else:
            return np.zeros((0, 2), dtype=np.float32)

    def allocate(self, qry_img, qry_pred_coarse, spt_fg_proto, supp_mask, model, supp_fts):
        """
        Main entry point.
        Returns:
            points: [N_p, 2]
            budget_Np: int
        """
        a, contours, M_tilde = self.compute_ambiguity_score(qry_img, qry_pred_coarse, spt_fg_proto, supp_mask, model, supp_fts)
        budget_Np = self.compute_budget(a, contours, M_tilde)
        r = self.get_scale_adaptive_offset(M_tilde)
        points = self.sample_placement(qry_img, M_tilde, contours, budget_Np, r)
        
        # Double check exact count
        if len(points) > budget_Np:
            points = points[:budget_Np]
        elif len(points) < budget_Np:
            budget_Np = len(points)
            
        return points, budget_Np
