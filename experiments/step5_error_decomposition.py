"""
Step 5 Diagnostic Script: Item B.1 - Fast Vectorized Error Decomposition
Decomposes segmentation errors on B0 (ViT-H, Upstream R3, CC) and B3B (ViT-B, Frozen R1, CC)
across all 120 episodes into:
(a) detached false-positive islands
(b) continuous boundary leakage into neighboring tissue
(c) under-segmentation / missed parts
(d) wrong-organ capture
"""

import os
import glob
import json
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

DATA_DIR = "D:/sabs_CT_normalized"
RESULTS_CSV = "results/step4_baselines/step4_baselines_results.csv"
B0_NPZ_DIR = "results/baselines/predictions"
B3B_NPZ_DIR = "results/step4_baselines/predictions"

ORGAN_NAME_TO_ID = {
    "SPLEEN": 1,
    "RK": 2,
    "LK": 3,
    "LIVER": 6
}

ORGAN_ID_TO_NAME = {
    1: "Spleen", 2: "Right Kidney", 3: "Left Kidney", 4: "Gallbladder",
    5: "Esophagus", 6: "Liver", 7: "Stomach", 8: "Aorta", 9: "IVC",
    10: "Portal & Splenic Veins", 11: "Pancreas", 12: "Right Adrenal", 13: "Left Adrenal"
}

def analyze_episode(ep_row, lbl_cache):
    ep_id = int(ep_row['episode_id'])
    pat_id = ep_row['patient_id']
    organ_str = ep_row['organ']
    organ_id = ORGAN_NAME_TO_ID[organ_str]
    spacing = (float(ep_row['spacing_x']), float(ep_row['spacing_y']), float(ep_row['spacing_z']))
    
    # Cache multi-organ label volumes to avoid repeated disk I/O
    if pat_id not in lbl_cache:
        pat_num = pat_id.split('_')[-1]
        lbl_file = os.path.join(DATA_DIR, f"label_{pat_num}.nii.gz")
        lbl_itk = sitk.ReadImage(lbl_file)
        lbl_cache[pat_id] = sitk.GetArrayFromImage(lbl_itk)
        
    lbl_full = lbl_cache[pat_id]
    
    # Active slices where target organ > 0
    target_mask = (lbl_full == organ_id)
    active_slices = target_mask.sum(axis=(1, 2)) > 0
    lbl_slices = lbl_full[active_slices]
    
    # Load predictions
    npz_b0_path = os.path.join(B0_NPZ_DIR, f"ep_{ep_id:03d}_{pat_id}_{organ_str}.npz")
    npz_b3b_path = os.path.join(B3B_NPZ_DIR, f"ep_{ep_id:03d}_{pat_id}_{organ_str}.npz")
    
    d_b0 = np.load(npz_b0_path) if os.path.exists(npz_b0_path) else None
    d_b3b = np.load(npz_b3b_path) if os.path.exists(npz_b3b_path) else None
    
    if d_b0 is None or d_b3b is None:
        return None
        
    gt = (d_b3b['gt'] > 0).astype(np.uint8)
    gt_bin = gt.astype(bool)
    total_gt_vox = int(gt.sum())
    gt_vol_safe = max(total_gt_vox, 1)
    
    # Precompute GT distance transform and surface for fast directed distance computation
    gt_border = np.logical_xor(gt_bin, ndimage.binary_erosion(gt_bin))
    dt_gt = ndimage.distance_transform_edt(~gt_bin, sampling=spacing[::-1])
    
    ep_results = {}
    for arm_name, pred_raw, pred_cc, hd_col, dice_col in [
        ("B0", d_b0['b0_raw'], d_b0['b0_cc'], 'hd95_b0_cc', 'dice_b0_cc'),
        ("B3B", d_b3b['b3b_raw'], d_b3b['b3b_cc'], 'hd95_b3b_cc', 'dice_b3b_cc')
    ]:
        pred_raw = (pred_raw > 0).astype(np.uint8)
        pred_cc = (pred_cc > 0).astype(np.uint8)
        pred_bin = pred_cc.astype(bool)
        
        # Voxels
        tp_cc = int(np.logical_and(pred_cc == 1, gt == 1).sum())
        fp_cc = int(np.logical_and(pred_cc == 1, gt == 0).sum())
        fn_cc = int(np.logical_and(pred_cc == 0, gt == 1).sum())
        
        # (a) Detached FP islands: removed by CC + residual detached
        detached_filtered = int(np.logical_and(pred_raw == 1, pred_cc == 0).sum())
        labeled_cc, num_cc = ndimage.label(pred_cc)
        residual_detached = 0
        if num_cc > 0:
            indices = np.arange(1, num_cc + 1)
            gt_overlaps = ndimage.sum(gt, labels=labeled_cc, index=indices)
            comp_sizes = ndimage.sum(np.ones_like(gt), labels=labeled_cc, index=indices)
            # Components with 0 overlap with GT
            residual_detached = int(comp_sizes[gt_overlaps == 0].sum())
            
        total_detached_fp = detached_filtered + residual_detached
        
        # (d) Wrong-organ capture: where pred_cc FP overlaps another recognized organ
        other_organ_mask = np.logical_and(lbl_slices > 0, lbl_slices != organ_id)
        wrong_organ_fp = int(np.logical_and(pred_cc == 1, other_organ_mask).sum())
        
        # Invaded organ breakdown
        invaded_organs = {}
        if wrong_organ_fp > 0:
            invaded_voxels = lbl_slices[np.logical_and(pred_cc == 1, other_organ_mask)]
            unique_orgs, counts = np.unique(invaded_voxels, return_counts=True)
            for org_code, cnt in zip(unique_orgs, counts):
                org_label = ORGAN_ID_TO_NAME.get(int(org_code), f"Organ_{org_code}")
                invaded_organs[org_label] = int(cnt)
                
        # (b) Continuous boundary leakage: contiguous FP not overlapping other organs
        boundary_leakage_fp = max(0, int(fp_cc - wrong_organ_fp - residual_detached))
        
        # (c) Under-segmentation: missed parts (FN)
        missed_fn = fn_cc
        
        # Dice
        denom = 2.0 * tp_cc + fp_cc + fn_cc
        dice_cc = float(2.0 * tp_cc / denom) if denom > 0 else 0.0
        
        # Directed distances for HD95 attribution
        pred_border = np.logical_xor(pred_bin, ndimage.binary_erosion(pred_bin))
        dt_pred = ndimage.distance_transform_edt(~pred_bin, sampling=spacing[::-1])
        
        dist_pred_to_gt = dt_gt[pred_border] if pred_border.sum() > 0 else np.array([0.0])
        dist_gt_to_pred = dt_pred[gt_border] if gt_border.sum() > 0 else np.array([0.0])
        
        h95_leakage = float(np.percentile(dist_pred_to_gt, 95)) if len(dist_pred_to_gt) > 0 else 0.0
        h95_missed = float(np.percentile(dist_gt_to_pred, 95)) if len(dist_gt_to_pred) > 0 else 0.0
        
        hd_full = float(ep_row[hd_col])
        
        if h95_leakage > h95_missed + 1.0:
            hd95_driver = "Over-segmentation / Leakage"
        elif h95_missed > h95_leakage + 1.0:
            hd95_driver = "Under-segmentation / Missed parts"
        else:
            hd95_driver = "Balanced"
            
        ep_results[arm_name] = {
            "episode_id": ep_id, "patient_id": pat_id, "organ": organ_str,
            "dice_cc": dice_cc, "hd95_cc": hd_full,
            "h95_pred_to_gt": h95_leakage, "h95_gt_to_pred": h95_missed,
            "hd95_driver": hd95_driver,
            "gt_voxels": total_gt_vox, "tp_voxels": tp_cc,
            "fp_voxels_cc": fp_cc, "fn_voxels_cc": fn_cc,
            "detached_islands_voxels": total_detached_fp,
            "boundary_leakage_voxels": boundary_leakage_fp,
            "wrong_organ_voxels": wrong_organ_fp,
            "missed_parts_voxels": missed_fn,
            "detached_frac_gt": float(total_detached_fp / gt_vol_safe),
            "boundary_leak_frac_gt": float(boundary_leakage_fp / gt_vol_safe),
            "wrong_organ_frac_gt": float(wrong_organ_fp / gt_vol_safe),
            "fn_frac_gt": float(missed_fn / gt_vol_safe),
            "invaded_organs": invaded_organs
        }
        
    return ep_results

def main():
    print("Executing Fast Vectorized Step 5 Error Decomposition across 120 episodes...", flush=True)
    df_manifest = pd.read_csv(RESULTS_CSV)
    
    lbl_cache = {}
    all_b0 = []
    all_b3b = []
    
    t0 = time.time()
    for idx, row in df_manifest.iterrows():
        res = analyze_episode(row, lbl_cache)
        if res is not None:
            all_b0.append(res["B0"])
            all_b3b.append(res["B3B"])
        if (idx + 1) % 30 == 0:
            print(f"Processed {idx + 1}/120 episodes ({time.time() - t0:.1f}s elapsed)...", flush=True)
            
    df_b0 = pd.DataFrame(all_b0)
    df_b3b = pd.DataFrame(all_b3b)
    
    os.makedirs("results/step5_diagnostics", exist_ok=True)
    df_b0.to_csv("results/step5_diagnostics/error_decomposition_b0.csv", index=False)
    df_b3b.to_csv("results/step5_diagnostics/error_decomposition_b3b.csv", index=False)
    
    # Aggregate summary
    summary = {}
    for arm_name, df_arm in [("B0_ViTH_R3_CC", df_b0), ("B3B_ViTB_R1_CC", df_b3b)]:
        summary[arm_name] = {}
        for organ in ["LIVER", "RK", "LK", "SPLEEN", "OVERALL"]:
            sub = df_arm if organ == "OVERALL" else df_arm[df_arm['organ'] == organ]
            summary[arm_name][organ] = {
                "n_episodes": len(sub),
                "dice_mean": float(sub['dice_cc'].mean() * 100.0),
                "dice_median": float(sub['dice_cc'].median() * 100.0),
                "hd95_mean": float(sub['hd95_cc'].mean()),
                "hd95_median": float(sub['hd95_cc'].median()),
                "hd95_p90": float(sub['hd95_cc'].quantile(0.90)),
                "h95_leakage_mean": float(sub['h95_pred_to_gt'].mean()),
                "h95_missed_mean": float(sub['h95_gt_to_pred'].mean()),
                "hd95_driver_counts": sub['hd95_driver'].value_counts().to_dict(),
                "mean_gt_voxels": float(sub['gt_voxels'].mean()),
                "mean_boundary_leakage_vox": float(sub['boundary_leakage_voxels'].mean()),
                "mean_wrong_organ_vox": float(sub['wrong_organ_voxels'].mean()),
                "mean_detached_islands_vox": float(sub['detached_islands_voxels'].mean()),
                "mean_missed_vox": float(sub['missed_parts_voxels'].mean()),
                "boundary_leak_frac_gt_pct": float(sub['boundary_leak_frac_gt'].mean() * 100.0),
                "wrong_organ_frac_gt_pct": float(sub['wrong_organ_frac_gt'].mean() * 100.0),
                "detached_frac_gt_pct": float(sub['detached_frac_gt'].mean() * 100.0),
                "missed_frac_gt_pct": float(sub['fn_frac_gt'].mean() * 100.0),
            }
            
    summary_path = "results/step5_diagnostics/error_decomposition_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Error decomposition complete in {time.time() - t0:.1f}s! Saved to {summary_path}", flush=True)

if __name__ == "__main__":
    main()
