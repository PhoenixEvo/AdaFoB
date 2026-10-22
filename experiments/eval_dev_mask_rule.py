"""
Fast & Rigorous Candidate Mask Selection Rule Evaluation on Disjoint Development Set (N=5 patients):
R1: multimask_output=False (SingleMask mode)
R2: multimask_output=True, argmax(iou_prediction)
R3: upstream mask[0] (reference)

Disjoint dev patients: image_2, image_8, image_14, image_20, image_25
Strictly zero overlap with the 25 evaluation test query patients.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

try:
    from medpy.metric.binary import hd95 as medpy_hd95
except ImportError:
    os.system("pip install -q medpy")
    from medpy.metric.binary import hd95 as medpy_hd95

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, "third_party", "FoB_SAM"))
sys.path.append(os.path.join(REPO_DIR, "third_party", "segment-anything"))

from models.FoB import FewShotSeg
from segment_anything import sam_model_registry, SamPredictor
from models.postprocessing import filter_connected_components

DEV_PATIENTS = ["image_2", "image_8", "image_14", "image_20", "image_25"]
ORGANS = {1: "SPLEEN", 2: "RK", 3: "LK", 6: "LIVER"}

def compute_volume_dice(pred, gt):
    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)
    p_sum = pred_bin.sum()
    g_sum = gt_bin.sum()
    if p_sum == 0 and g_sum == 0:
        return 1.0
    if p_sum == 0 or g_sum == 0:
        return 0.0
    inter = np.logical_and(pred_bin, gt_bin).sum()
    return float(2.0 * inter / (p_sum + g_sum))

def compute_volume_hd95_fast(pred, gt, spacing):
    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)
    shape_z, shape_y, shape_x = pred.shape
    diag = float(np.sqrt((shape_x * spacing[0])**2 + (shape_y * spacing[1])**2 + (shape_z * spacing[2])**2))
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return diag
    try:
        hd = float(medpy_hd95(pred_bin, gt_bin, voxelspacing=spacing[::-1]))
        return min(diag, hd) if (not np.isnan(hd) and not np.isinf(hd)) else diag
    except Exception:
        return diag

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    fob_ckpt_path = "checkpoints/fob_checkpoints/exps_train_on_SABS_FSMIS_FoB/FSMIS_train_SABS_cv0/ckpt.pth"
    class DummyArgs:
        dataset = "SABS"
        n_ways = 1
        n_shots = 1
    fob_model = FewShotSeg(DummyArgs()).to(device).eval()
    ckpt = torch.load(fob_ckpt_path, map_location=device)
    raw_sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    cleaned_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
    fob_model.load_state_dict(cleaned_sd, strict=False)
    fob_model.allocator = None

    sam_b_path = "D:/results/checkpoints/sam_vit_b_01ec64.pth"
    print(f"Loading SAM ViT-B from {sam_b_path}...", flush=True)
    sam_b = sam_model_registry["vit_b"](checkpoint=sam_b_path).eval().to(device)
    predictor_b = SamPredictor(sam_b)

    ct_dir = "D:/sabs_CT_normalized"
    dev_vols = {}
    for pat in DEV_PATIENTS:
        img_p = os.path.join(ct_dir, f"{pat}.nii.gz")
        lbl_p = os.path.join(ct_dir, f"label_{pat.split('_')[-1]}.nii.gz")
        img_itk = sitk.ReadImage(img_p)
        lbl_itk = sitk.ReadImage(lbl_p)
        dev_vols[pat] = {
            "img": sitk.GetArrayFromImage(img_itk),
            "lbl": sitk.GetArrayFromImage(lbl_itk),
            "spacing": img_itk.GetSpacing()
        }

    results = []

    for q_idx, q_pat in enumerate(DEV_PATIENTS):
        s_pat = DEV_PATIENTS[(q_idx + 1) % len(DEV_PATIENTS)]
        q_data = dev_vols[q_pat]
        s_data = dev_vols[s_pat]
        spacing = q_data["spacing"]

        for organ_id, organ_name in ORGANS.items():
            q_gt = (q_data["lbl"] == organ_id).astype(np.uint8)
            s_gt = (s_data["lbl"] == organ_id).astype(np.uint8)
            if q_gt.sum() == 0 or s_gt.sum() == 0:
                continue

            s_z = int(np.argmax([(s_gt[z]).sum() for z in range(len(s_gt))]))
            s_img_sl = torch.from_numpy(np.stack(3 * [(s_data["img"][s_z] - s_data["img"].mean()) / s_data["img"].std()], axis=0)).unsqueeze(0).float().to(device)
            s_lbl_sl = torch.from_numpy(s_gt[s_z].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

            Z, H, W = q_gt.shape
            pred_r1_raw = np.zeros((Z, H, W), dtype=np.uint8)
            pred_r2_raw = np.zeros((Z, H, W), dtype=np.uint8)
            pred_r3_raw = np.zeros((Z, H, W), dtype=np.uint8)

            pred_r1_cc = np.zeros((Z, H, W), dtype=np.uint8)
            pred_r2_cc = np.zeros((Z, H, W), dtype=np.uint8)
            pred_r3_cc = np.zeros((Z, H, W), dtype=np.uint8)

            active_z = np.where(q_gt.sum(axis=(1, 2)) > 0)[0]
            z_min = max(0, active_z.min() - 3)
            z_max = min(Z, active_z.max() + 4)

            for z in range(z_min, z_max):
                q_img_raw = q_data["img"][z]
                q_img_t = torch.from_numpy(np.stack(3 * [(q_img_raw - q_data["img"].mean()) / q_data["img"].std()], axis=0)).unsqueeze(0).float().to(device)
                dummy_lbl = torch.ones_like(q_img_t[:, 0]).long()

                with torch.no_grad():
                    fob_out = fob_model([[s_img_sl]], [[s_lbl_sl]], [q_img_t], dummy_lbl, train=False, use_skeleton=False, budget_Np=10)
                pos_pts = np.array(fob_out[1]).reshape(-1, 2)[:10]
                neg_pts = np.array(fob_out[0]).reshape(-1, 2)[:10]

                img_norm = ((q_img_raw - q_img_raw.min()) / (q_img_raw.max() - q_img_raw.min() + 1e-8) * 255).astype(np.uint8)
                img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
                predictor_b.set_image(img_rgb)

                all_pts = np.concatenate([pos_pts, neg_pts], axis=0) if len(pos_pts) > 0 and len(neg_pts) > 0 else None
                all_lbls = np.concatenate([np.ones(len(pos_pts), dtype=np.int32), np.zeros(len(neg_pts), dtype=np.int32)], axis=0) if all_pts is not None else None

                if all_pts is not None:
                    masks_r1, scores_r1, _ = predictor_b.predict(point_coords=all_pts, point_labels=all_lbls, multimask_output=False)
                    masks_m, scores_m, _ = predictor_b.predict(point_coords=all_pts, point_labels=all_lbls, multimask_output=True)

                    m1 = (masks_r1[0] > 0).astype(np.uint8)
                    m2 = (masks_m[np.argmax(scores_m)] > 0).astype(np.uint8)
                    m3 = (masks_m[0] > 0).astype(np.uint8)

                    pred_r1_raw[z] = m1
                    pred_r2_raw[z] = m2
                    pred_r3_raw[z] = m3

                    pred_r1_cc[z] = filter_connected_components(m1, pos_pts)
                    pred_r2_cc[z] = filter_connected_components(m2, pos_pts)
                    pred_r3_cc[z] = filter_connected_components(m3, pos_pts)

            d_r1_raw = compute_volume_dice(pred_r1_raw, q_gt) * 100.0
            d_r2_raw = compute_volume_dice(pred_r2_raw, q_gt) * 100.0
            d_r3_raw = compute_volume_dice(pred_r3_raw, q_gt) * 100.0

            d_r1_cc = compute_volume_dice(pred_r1_cc, q_gt) * 100.0
            d_r2_cc = compute_volume_dice(pred_r2_cc, q_gt) * 100.0
            d_r3_cc = compute_volume_dice(pred_r3_cc, q_gt) * 100.0

            h_r1_cc = compute_volume_hd95_fast(pred_r1_cc, q_gt, spacing)
            h_r2_cc = compute_volume_hd95_fast(pred_r2_cc, q_gt, spacing)
            h_r3_cc = compute_volume_hd95_fast(pred_r3_cc, q_gt, spacing)

            rec = {
                "query_patient": q_pat, "support_patient": s_pat, "organ": organ_name,
                "dice_r1_raw": d_r1_raw, "dice_r1_cc": d_r1_cc,
                "dice_r2_raw": d_r2_raw, "dice_r2_cc": d_r2_cc,
                "dice_r3_raw": d_r3_raw, "dice_r3_cc": d_r3_cc,
                "hd95_r1_cc": h_r1_cc, "hd95_r2_cc": h_r2_cc, "hd95_r3_cc": h_r3_cc
            }
            results.append(rec)
            print(f"[{q_pat} | {organ_name:6s}] R1 (Single): {d_r1_cc:5.2f}% | R2 (Argmax): {d_r2_cc:5.2f}% | R3 (mask[0]): {d_r3_cc:5.2f}%", flush=True)

    df_res = pd.DataFrame(results)
    out_csv = "results/dev_mask_rule_comparison.csv"
    df_res.to_csv(out_csv, index=False)
    
    print("\n" + "="*70, flush=True)
    print("DEVELOPMENT SET MASK RULE COMPARISON SUMMARY (N=5 disjoint patients):", flush=True)
    print(f"  R1 (SingleMask) Raw Dice : {df_res['dice_r1_raw'].mean():.2f}% | Post-CC Dice: {df_res['dice_r1_cc'].mean():.2f}% | Post-CC HD95: {df_res['hd95_r1_cc'].mean():.2f} mm", flush=True)
    print(f"  R2 (ArgmaxScore) Raw Dice: {df_res['dice_r2_raw'].mean():.2f}% | Post-CC Dice: {df_res['dice_r2_cc'].mean():.2f}% | Post-CC HD95: {df_res['hd95_r2_cc'].mean():.2f} mm", flush=True)
    print(f"  R3 (Upstream m0) Raw Dice: {df_res['dice_r3_raw'].mean():.2f}% | Post-CC Dice: {df_res['dice_r3_cc'].mean():.2f}% | Post-CC HD95: {df_res['hd95_r3_cc'].mean():.2f} mm", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    main()
