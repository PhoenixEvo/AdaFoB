"""
Step 5 Diagnostic Script: Item B.2 - Fast Standalone Prompt Quality & Contamination Audit
Runs FoB prompt generation across all 120 episodes on local GPU (RTX 4050).
Extracts:
- Fraction of negative points falling INSIDE the true organ (contamination).
- Fraction of positive points falling OUTSIDE the true organ.
- Spatial distribution of negative points relative to organ boundary (distances to GT surface in mm).
- Breakdown by organ (LIVER, RK, LK, SPLEEN).
- Per-patient failure cases with high contamination.
"""

import os
import sys
import gc
import glob
import json
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, "third_party", "FoB_SAM"))

from models.FoB import FewShotSeg
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import get_label_names

DATA_DIR = "D:/"
MANIFEST_PATH = os.path.join(REPO_DIR, "protocol", "paired_episode_manifest.csv")
CKPT_DIR = os.path.join(REPO_DIR, "checkpoints", "fob_checkpoints")
OUT_DIR = os.path.join(REPO_DIR, "results", "step5_diagnostics")

TEST_LABELS = [1, 2, 3, 6]
SUPP_IDX = 2
N_PART = 3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Item B.2] Running Standalone Prompt Quality Audit on {device} ({torch.cuda.get_device_name(0)})...", flush=True)

    manifest_df = pd.read_csv(MANIFEST_PATH)
    labels = get_label_names("SABS")
    os.makedirs(OUT_DIR, exist_ok=True)

    records = []
    t0 = time.time()

    for eval_fold in range(5):
        print(f"\nEvaluating Prompt Generation for Fold {eval_fold}...", flush=True)
        cands = glob.glob(os.path.join(CKPT_DIR, f"**/*cv{eval_fold}*/**/*.pth"), recursive=True)
        if not cands:
            cands = glob.glob(os.path.join(CKPT_DIR, f"**/*cv{eval_fold}*.pth"), recursive=True)
        fob_ckpt_path = sorted(cands)[0]

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

        data_config = {
            'data_dir': DATA_DIR, 'dataset': 'SABS', 'n_shot': 1, 'n_way': 1, 'n_query': 1,
            'n_sv': 5000, 'max_iter': 3000, 'eval_fold': eval_fold, 'min_size': 200,
            'max_slices': 3, 'supp_idx': SUPP_IDX,
        }
        test_dataset = TestDataset(data_config)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        for label_val in TEST_LABELS:
            label_name = labels[label_val]
            test_dataset.label = label_val
            support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=N_PART)

            support_image = [support_sample['image'][[i]].float().to(device) for i in range(support_sample['image'].shape[0])]
            support_fg_mask = [support_sample['label'][[i]].float().to(device) for i in range(support_sample['image'].shape[0])]

            for vi, sample in enumerate(test_loader):
                vol_id_str = sample['id'][0].replace('\\', '/')
                pat_id = vol_id_str.split('/')[-1].split('.nii')[0]

                row_match = manifest_df[(manifest_df['fold'] == eval_fold) &
                                        (manifest_df['organ'] == label_name) &
                                        (manifest_df['vol_id'].str.contains(pat_id, regex=False))]
                if len(row_match) == 0:
                    continue
                ep_row = row_match.iloc[0]
                ep_id = int(ep_row['episode_id'])
                spacing = (float(ep_row['spacing_x']), float(ep_row['spacing_y']), float(ep_row['spacing_z']))
                pixel_spacing_mm = float(ep_row['spacing_x'])

                query_images = sample['image'].float()
                query_labels = sample['label'].long()
                C_q = query_images.shape[1]
                H_q, W_q = query_images.shape[-2], query_images.shape[-1]
                gt_vol_np = (query_labels[0].squeeze(1).numpy() > 0).astype(np.uint8)

                ep_pos_total, ep_pos_outside = 0, 0
                ep_neg_total, ep_neg_inside = 0, 0
                ep_distances_mm = []

                idx_ = np.linspace(0, C_q, N_PART + 1).astype(int)

                with torch.no_grad():
                    for sub_chunk in range(N_PART):
                        support_image_s = [support_image[sub_chunk]]
                        support_fg_mask_s = [support_fg_mask[sub_chunk]]
                        query_image_chunk = query_images[0, idx_[sub_chunk]:idx_[sub_chunk+1]]
                        chunk_len = query_image_chunk.shape[0]

                        if chunk_len == 0:
                            continue

                        for j in range(chunk_len):
                            global_z = idx_[sub_chunk] + j
                            q_img_slice = query_image_chunk[[j]].to(device)
                            gt_slice = gt_vol_np[global_z]
                            dummy_lbl = torch.ones_like(q_img_slice[:, 0]).long()

                            if support_image_s[0].max() == 0:
                                pos_arr = np.zeros((0, 2))
                                neg_arr = np.zeros((0, 2))
                            else:
                                fob_out = fob_model(
                                    [support_image_s], [support_fg_mask_s],
                                    [q_img_slice], dummy_lbl,
                                    train=False, use_skeleton=False, budget_Np=10
                                )
                                neg_arr = np.array(fob_out[0]).reshape(-1, 2)[:10]
                                pos_arr = np.array(fob_out[1]).reshape(-1, 2)[:10]

                            gt_slice_bool = (gt_slice > 0)
                            if gt_slice_bool.sum() > 0:
                                dt_outside = ndimage.distance_transform_edt(~gt_slice_bool) * pixel_spacing_mm
                            else:
                                dt_outside = np.zeros_like(gt_slice, dtype=float)

                            for p in pos_arr:
                                x, y = int(round(p[0])), int(round(p[1]))
                                if 0 <= y < H_q and 0 <= x < W_q:
                                    ep_pos_total += 1
                                    if gt_slice[y, x] == 0:
                                        ep_pos_outside += 1

                            for p in neg_arr:
                                x, y = int(round(p[0])), int(round(p[1]))
                                if 0 <= y < H_q and 0 <= x < W_q:
                                    ep_neg_total += 1
                                    if gt_slice[y, x] == 1:
                                        ep_neg_inside += 1
                                        ep_distances_mm.append(-1.0)
                                    else:
                                        dist_val = float(dt_outside[y, x])
                                        ep_distances_mm.append(dist_val)

                dist_arr = np.array(ep_distances_mm)
                neg_inside_cnt = int((dist_arr < 0).sum())
                d_0_5_cnt = int(np.logical_and(dist_arr >= 0, dist_arr < 5.0).sum())
                d_5_10_cnt = int(np.logical_and(dist_arr >= 5.0, dist_arr < 10.0).sum())
                d_10_20_cnt = int(np.logical_and(dist_arr >= 10.0, dist_arr < 20.0).sum())
                d_gt20_cnt = int((dist_arr >= 20.0).sum())

                row = {
                    'episode_id': ep_id, 'fold': eval_fold, 'organ': label_name, 'patient_id': pat_id,
                    'pos_total': ep_pos_total, 'pos_outside': ep_pos_outside,
                    'pos_outside_pct': float(ep_pos_outside / max(ep_pos_total, 1) * 100.0),
                    'neg_total': ep_neg_total, 'neg_inside': ep_neg_inside,
                    'neg_contam_pct': float(ep_neg_inside / max(ep_neg_total, 1) * 100.0),
                    'dist_inside': neg_inside_cnt, 'dist_0_5mm': d_0_5_cnt, 'dist_5_10mm': d_5_10_cnt,
                    'dist_10_20mm': d_10_20_cnt, 'dist_gt20mm': d_gt20_cnt
                }
                records.append(row)
                if (len(records)) % 24 == 0:
                    print(f"Processed {len(records)}/120 episodes ({time.time() - t0:.1f}s elapsed)...", flush=True)

                del query_images, query_labels, gt_vol_np, ep_distances_mm, dist_arr
                gc.collect()

    df_audit = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, "step5_prompt_audit_per_episode.csv")
    df_audit.to_csv(csv_path, index=False)

    # Compute summary
    total_pos = int(df_audit['pos_total'].sum())
    total_pos_out = int(df_audit['pos_outside'].sum())
    total_neg = int(df_audit['neg_total'].sum())
    total_neg_in = int(df_audit['neg_inside'].sum())

    inside_tot = int(df_audit['dist_inside'].sum())
    d_0_5_tot = int(df_audit['dist_0_5mm'].sum())
    d_5_10_tot = int(df_audit['dist_5_10mm'].sum())
    d_10_20_tot = int(df_audit['dist_10_20mm'].sum())
    d_gt20_tot = int(df_audit['dist_gt20mm'].sum())

    audit_summary = {
        "cohort_overview": {
            "total_episodes": len(df_audit),
            "total_positive_points": total_pos,
            "positive_points_outside_gt": total_pos_out,
            "positive_outside_pct": float(total_pos_out / max(total_pos, 1) * 100.0),
            "total_negative_points": total_neg,
            "negative_points_inside_gt": total_neg_in,
            "negative_contamination_pct": float(total_neg_in / max(total_neg, 1) * 100.0),
            "distance_bins_count": {
                "inside_gt_negative": inside_tot,
                "0_to_5mm": d_0_5_tot,
                "5_to_10mm": d_5_10_tot,
                "10_to_20mm": d_10_20_tot,
                "gt_20mm": d_gt20_tot,
            },
            "distance_bins_pct": {
                "inside_gt_negative": float(inside_tot / max(total_neg, 1) * 100.0),
                "0_to_5mm": float(d_0_5_tot / max(total_neg, 1) * 100.0),
                "5_to_10mm": float(d_5_10_tot / max(total_neg, 1) * 100.0),
                "10_to_20mm": float(d_10_20_tot / max(total_neg, 1) * 100.0),
                "gt_20mm": float(d_gt20_tot / max(total_neg, 1) * 100.0),
            }
        },
        "per_organ": {}
    }

    for organ in ["LIVER", "RK", "LK", "SPLEEN"]:
        sub = df_audit[df_audit['organ'] == organ]
        sub_pos = int(sub['pos_total'].sum())
        sub_pos_out = int(sub['pos_outside'].sum())
        sub_neg = int(sub['neg_total'].sum())
        sub_neg_in = int(sub['neg_inside'].sum())
        audit_summary["per_organ"][organ] = {
            "n_episodes": len(sub),
            "positive_points_total": sub_pos,
            "positive_outside_pct": float(sub_pos_out / max(sub_pos, 1) * 100.0),
            "negative_points_total": sub_neg,
            "negative_contamination_pct": float(sub_neg_in / max(sub_neg, 1) * 100.0),
            "dist_inside_cnt": int(sub['dist_inside'].sum()),
            "dist_0_5mm_cnt": int(sub['dist_0_5mm'].sum()),
            "dist_5_10mm_cnt": int(sub['dist_5_10mm'].sum()),
            "dist_10_20mm_cnt": int(sub['dist_10_20mm'].sum()),
            "dist_gt20mm_cnt": int(sub['dist_gt20mm'].sum()),
        }

    # Identify cases with high contamination (>15%)
    high_contam = df_audit[df_audit['neg_contam_pct'] > 15.0].sort_values("neg_contam_pct", ascending=False)
    audit_summary["high_contamination_cases"] = high_contam[['episode_id', 'fold', 'organ', 'patient_id', 'neg_contam_pct', 'pos_outside_pct']].to_dict(orient="records")

    json_path = os.path.join(OUT_DIR, "step5_prompt_audit_summary.json")
    with open(json_path, "w") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"\n[Item B.2] Audit complete in {time.time() - t0:.1f}s! Saved to {json_path}", flush=True)


if __name__ == "__main__":
    main()
