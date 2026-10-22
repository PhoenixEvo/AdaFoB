"""
Rigorous Step D Recomputation (Optimized Vectorized Bootstrap):
- Patient-cluster bootstrap 95% CIs for gradient-vs-leak correlation (B=5000, K=25 patients)
- Partial correlation controlling for organ type, predicted-mask size, and longitudinal slice position
- Localized boundary segment analysis: gradient at leak boundary vs intact boundary
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from scipy import stats

def run_step_d_rigorous():
    pred_dir = "results/baselines/predictions"
    ct_dir = "D:/sabs_CT_normalized"
    manifest_path = "protocol/paired_episode_manifest.csv"
    
    manifest_df = pd.read_csv(manifest_path)
    npz_files = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))
    print(f"Loaded {len(npz_files)} prediction volumes.", flush=True)
    
    slice_records = []
    segment_records = []
    
    for npz_path in npz_files:
        basename = os.path.basename(npz_path).replace(".npz", "")
        parts = basename.split("_")
        ep_id = int(parts[1])
        pat_id = f"{parts[2]}_{parts[3]}"
        organ = "_".join(parts[4:])
        
        data = np.load(npz_path)
        b1_raw = data["b1_raw"] # (Z, H, W)
        gt = data["gt"]         # (Z, H, W)
        
        img_nii_path = os.path.join(ct_dir, f"{pat_id}.nii.gz")
        if not os.path.exists(img_nii_path):
            continue
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_nii_path))
        
        Z, H, W = b1_raw.shape
        
        for z in range(Z):
            pred_sl = b1_raw[z]
            gt_sl = gt[z]
            
            pred_area = int(pred_sl.sum())
            if pred_area == 0:
                continue
                
            img_sl = img_arr[z]
            img_norm = ((img_sl - img_sl.min()) / (img_sl.max() - img_sl.min() + 1e-8) * 255).astype(np.uint8)
            if img_norm.ndim == 2:
                gray = img_norm
            else:
                gray = cv2.cvtColor(img_norm, cv2.COLOR_RGB2GRAY)
                
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            gmag = cv2.magnitude(gx, gy)
            
            contours, _ = cv2.findContours(pred_sl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c_mask = np.zeros_like(pred_sl)
            cv2.drawContours(c_mask, contours, -1, 1, 1)
            
            mean_edge_grad = float(gmag[c_mask == 1].mean()) if c_mask.sum() > 0 else 0.0
            
            fp_mask = np.logical_and(pred_sl == 1, gt_sl == 0).astype(np.uint8)
            fp_area = float(fp_mask.sum())
            
            slice_records.append({
                "ep_id": ep_id,
                "pat_id": pat_id,
                "organ": organ,
                "slice_z": z,
                "rel_z": float(z / Z),
                "pred_area": pred_area,
                "gt_area": int(gt_sl.sum()),
                "fp_area": fp_area,
                "edge_grad": mean_edge_grad,
            })
            
            if fp_area > 0 and gt_sl.sum() > 0:
                kernel = np.ones((3, 3), np.uint8)
                gt_dilated = cv2.dilate(gt_sl, kernel, iterations=2)
                
                intact_boundary = np.logical_and(c_mask == 1, gt_dilated == 1)
                leak_boundary = np.logical_and(c_mask == 1, gt_dilated == 0)
                
                if intact_boundary.sum() >= 5 and leak_boundary.sum() >= 5:
                    grad_intact = float(gmag[intact_boundary].mean())
                    grad_leak = float(gmag[leak_boundary].mean())
                    segment_records.append({
                        "pat_id": pat_id,
                        "organ": organ,
                        "grad_intact": grad_intact,
                        "grad_leak": grad_leak,
                        "grad_diff": grad_leak - grad_intact
                    })

    df_slices = pd.DataFrame(slice_records)
    df_segments = pd.DataFrame(segment_records)
    print(f"Extracted {len(df_slices)} slice records across {df_slices['pat_id'].nunique()} unique patients.", flush=True)
    print(f"Extracted {len(df_segments)} paired boundary segment comparisons.", flush=True)

    # 1. Bivariate Pearson & Spearman
    r_biv, p_r = stats.pearsonr(df_slices["edge_grad"], df_slices["fp_area"])
    rho_biv, p_rho = stats.spearmanr(df_slices["edge_grad"], df_slices["fp_area"])
    
    # Pre-group slice arrays by patient for fast bootstrap
    pat_slice_data = {}
    for p, group in df_slices.groupby("pat_id"):
        pat_slice_data[p] = {
            "edge_grad": group["edge_grad"].values,
            "fp_area": group["fp_area"].values,
            "pred_area": group["pred_area"].values,
            "rel_z": group["rel_z"].values,
            "organ": group["organ"].values
        }
        
    patients = np.array(list(pat_slice_data.keys()))
    K = len(patients)
    B = 5000
    np.random.seed(42)
    
    boot_r = []
    boot_rho = []
    
    for _ in range(B):
        boot_idx = np.random.choice(len(patients), size=K, replace=True)
        chosen_pats = patients[boot_idx]
        b_grad = np.concatenate([pat_slice_data[p]["edge_grad"] for p in chosen_pats])
        b_fp   = np.concatenate([pat_slice_data[p]["fp_area"] for p in chosen_pats])
        
        br, _ = stats.pearsonr(b_grad, b_fp)
        brho, _ = stats.spearmanr(b_grad, b_fp)
        boot_r.append(br)
        boot_rho.append(brho)
        
    ci_r = (float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5)))
    ci_rho = (float(np.percentile(boot_rho, 2.5)), float(np.percentile(boot_rho, 97.5)))

    # 2. Partial Correlation controlling for organ, pred_area, and rel_z
    organ_dummies = pd.get_dummies(df_slices["organ"], drop_first=True, dtype=float)
    X = np.column_stack([
        np.ones(len(df_slices)),
        organ_dummies.values,
        df_slices["pred_area"].values,
        df_slices["rel_z"].values
    ])
    y_grad = df_slices["edge_grad"].values
    y_fp = df_slices["fp_area"].values
    
    beta_grad, _, _, _ = np.linalg.lstsq(X, y_grad, rcond=None)
    resid_grad = y_grad - X @ beta_grad
    
    beta_fp, _, _, _ = np.linalg.lstsq(X, y_fp, rcond=None)
    resid_fp = y_fp - X @ beta_fp
    
    r_partial, p_partial = stats.pearsonr(resid_grad, resid_fp)
    
    # Precompute per-patient X, y for partial correlation bootstrap
    pat_part_data = {}
    for p, group in df_slices.groupby("pat_id"):
        sub_X = X[group.index.values]
        pat_part_data[p] = {
            "X": sub_X,
            "y_grad": y_grad[group.index.values],
            "y_fp": y_fp[group.index.values]
        }
        
    boot_r_partial = []
    for _ in range(B):
        boot_idx = np.random.choice(len(patients), size=K, replace=True)
        chosen_pats = patients[boot_idx]
        bX = np.vstack([pat_part_data[p]["X"] for p in chosen_pats])
        bg = np.concatenate([pat_part_data[p]["y_grad"] for p in chosen_pats])
        bfp = np.concatenate([pat_part_data[p]["y_fp"] for p in chosen_pats])
        
        bg_res = bg - bX @ np.linalg.lstsq(bX, bg, rcond=None)[0]
        bfp_res = bfp - bX @ np.linalg.lstsq(bX, bfp, rcond=None)[0]
        pr, _ = stats.pearsonr(bg_res, bfp_res)
        boot_r_partial.append(pr)
        
    ci_r_partial = (float(np.percentile(boot_r_partial, 2.5)), float(np.percentile(boot_r_partial, 97.5)))

    # 3. Localized Boundary Segment Analysis
    mean_grad_intact = float(df_segments["grad_intact"].mean())
    mean_grad_leak = float(df_segments["grad_leak"].mean())
    mean_diff = float(df_segments["grad_diff"].mean())
    t_stat, p_seg = stats.ttest_rel(df_segments["grad_leak"], df_segments["grad_intact"])
    
    pat_seg_data = {p: group["grad_diff"].values for p, group in df_segments.groupby("pat_id")}
    seg_pats = np.array(list(pat_seg_data.keys()))
    K_seg = len(seg_pats)
    boot_seg_diff = []
    for _ in range(B):
        boot_idx = np.random.choice(len(seg_pats), size=K_seg, replace=True)
        chosen_pats = seg_pats[boot_idx]
        b_diff = np.concatenate([pat_seg_data[p] for p in chosen_pats])
        boot_seg_diff.append(float(np.mean(b_diff)))
        
    ci_seg_diff = (float(np.percentile(boot_seg_diff, 2.5)), float(np.percentile(boot_seg_diff, 97.5)))
    cohen_d = float(mean_diff / (df_segments["grad_diff"].std() + 1e-8))

    summary = {
        "n_slices": len(df_slices),
        "n_patients": int(K),
        "bivariate_pearson_r": float(r_biv),
        "bivariate_pearson_ci": [float(ci_r[0]), float(ci_r[1])],
        "bivariate_pearson_p": float(p_r),
        "bivariate_spearman_rho": float(rho_biv),
        "bivariate_spearman_ci": [float(ci_rho[0]), float(ci_rho[1])],
        "partial_pearson_r": float(r_partial),
        "partial_pearson_ci": [float(ci_r_partial[0]), float(ci_r_partial[1])],
        "partial_pearson_p": float(p_partial),
        "controlled_covariates": ["organ_type", "pred_area", "longitudinal_rel_z"],
        "n_boundary_segments": len(df_segments),
        "mean_grad_intact_boundary": float(mean_grad_intact),
        "mean_grad_leak_boundary": float(mean_grad_leak),
        "mean_segment_diff": float(mean_diff),
        "segment_diff_ci": [float(ci_seg_diff[0]), float(ci_seg_diff[1])],
        "segment_p_value": float(p_seg),
        "cohens_d": float(cohen_d)
    }

    out_path = "results/baselines/step_d_rigorous_analysis.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStep D rigorous analysis complete! Saved to {out_path}:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    run_step_d_rigorous()
