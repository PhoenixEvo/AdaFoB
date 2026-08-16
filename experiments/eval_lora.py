"""
AdaFoB + LoRA Evaluation Script
================================
Evaluates FoB/AdaFoB with LoRA-adapted SAM ViT-B vs frozen SAM ViT-H baseline.

Methods compared:
  1. FoB Baseline (Np=10, frozen ViT-H)
  2. AdaFoB 2D (adaptive Np, frozen ViT-H)
  3. FoB + LoRA (Np=10, LoRA ViT-B)
  4. AdaFoB + LoRA (adaptive Np, LoRA ViT-B)  ← FULL METHOD

Designed for Kaggle T4x2 (split organs across GPUs).

Usage:
    python experiments/eval_lora.py --gpu 0 --fold 0
    python experiments/eval_lora.py --gpu 0 --organs 1 2 &
    python experiments/eval_lora.py --gpu 1 --organs 3 6 & wait
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import cv2

from tqdm import tqdm
import SimpleITK as sitk

try:
    from medpy.metric.binary import hd95
except ImportError:
    os.system("pip install medpy")
    from medpy.metric.binary import hd95

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "/kaggle/working/AdaFoB" not in sys.path:
    sys.path.insert(0, "/kaggle/working/AdaFoB")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "third_party", "FoB_SAM"))
sys.path.insert(0, os.path.join(REPO_DIR, "third_party", "segment-anything"))

from models.FoB import FewShotSeg
from models.allocator import PromptBudgetAllocator
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import get_label_names, get_folds
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry, SamPredictor

# LoRA
from experiments.lora_sam import LoRA_Sam

# Reuse eval.py functions
sys.path.insert(0, os.path.join(REPO_DIR, "experiments"))
try:
    from eval import predict_sam_from_points, sanitize_prompts
except ImportError:
    pass


# ── Auto-detect paths ────────────────────────────────────────────────────────
def find_path(target_name, is_file=False):
    working_path = os.path.join("/kaggle/working", target_name)
    if os.path.exists(working_path):
        return working_path
    for pat in [f"/kaggle/input/**/{target_name}"]:
        candidates = glob.glob(pat, recursive=True)
        if candidates:
            return candidates[0]
    local = os.path.join(REPO_DIR, target_name)
    if os.path.exists(local):
        return local
    return working_path


NORMALIZED_DIR = find_path("sabs_CT_normalized")
CKPT_DIR = find_path("fob_checkpoints")
SAM_H_CKPT = find_path("sam_vit_h.pth", is_file=True)
SAM_B_CKPT = find_path("sam_vit_b_01ec64.pth", is_file=True)
LORA_CKPT_DIR = os.path.join(REPO_DIR, "outputs", "lora_checkpoints")
GLOBAL_PARAMS_PATH = os.path.join(REPO_DIR, "results", "r3_global_params.json")

TEST_LABELS = [1, 2, 3, 6]  # Spleen, RK, LK, Liver
N_PART = 3
SUPP_IDX = 2
VALID_NPS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]


# ── Voxel spacing lookup ─────────────────────────────────────────────────────
def get_voxel_spacing(vol_idx):
    """Get voxel spacing for a SABS volume."""
    norm_dir = NORMALIZED_DIR
    img_path = os.path.join(norm_dir, f"image_{vol_idx}.nii")
    if not os.path.exists(img_path):
        img_path = img_path + ".gz"
    if os.path.exists(img_path):
        img_sitk = sitk.ReadImage(img_path)
        return img_sitk.GetSpacing()
    return (1.0, 1.0, 3.0)


def load_fob_checkpoint(fold, ckpt_dir):
    candidates = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*/**/*.pth"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*.pth"), recursive=True)
    if candidates:
        return sorted(candidates)[0]
    return None


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_volume_dice(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    inter = np.logical_and(pred, gt).sum()
    return 2 * inter / (pred.sum() + gt.sum())


def compute_volume_hd95(pred, gt, spacing):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 300.0
    try:
        return hd95(pred, gt, voxelspacing=spacing)
    except:
        return 300.0


# ── SAM prediction helpers ────────────────────────────────────────────────────
def predict_with_sam(predictor, pts_pos, pts_neg, img_shape):
    """Standard SAM prediction (no mask propagation, for baseline methods)."""
    all_pts = []
    all_lbl = []
    if len(pts_pos) > 0:
        all_pts.extend(pts_pos.tolist() if hasattr(pts_pos, 'tolist') else list(pts_pos))
        all_lbl.extend([1] * len(pts_pos))
    if len(pts_neg) > 0:
        all_pts.extend(pts_neg.tolist() if hasattr(pts_neg, 'tolist') else list(pts_neg))
        all_lbl.extend([0] * len(pts_neg))

    if len(all_pts) > 0:
        masks, scores, logits = predictor.predict(
            point_coords=np.array(all_pts),
            point_labels=np.array(all_lbl),
            multimask_output=True,
        )
        return (masks[0] > 0.5).astype(np.uint8)
    else:
        H, W = img_shape[:2]
        return np.zeros((H, W), dtype=np.uint8)


# ── Main Evaluation ──────────────────────────────────────────────────────────
def evaluate_lora(gpu=0, target_organs=None, target_fold=None):
    """Main evaluation loop comparing baseline vs LoRA methods."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if target_organs is None:
        target_organs = TEST_LABELS

    # Load global params
    global_params = {"nu": 0.02, "lam": 0.5, "a0": 0.6, "tau": 0.05}
    if os.path.exists(GLOBAL_PARAMS_PATH):
        with open(GLOBAL_PARAMS_PATH, "r") as f:
            global_params.update(json.load(f))
        print(f"Loaded Global Params: {global_params}")

    labels = get_label_names("SABS")
    data_dir = os.path.dirname(NORMALIZED_DIR)

    # ── Init SAM ViT-H (frozen baseline) ──────────────────────────────────
    print("Loading SAM ViT-H (frozen baseline)...")
    if os.path.exists(SAM_H_CKPT):
        sam_h = sam_model_registry["vit_h"](checkpoint=SAM_H_CKPT).eval().cuda()
        predictor_h = SamPredictor(sam_h)
        has_vith = True
    else:
        print("  WARNING: sam_vit_h.pth not found, skipping ViT-H baseline")
        has_vith = False

    # ── Init SAM ViT-B + LoRA ─────────────────────────────────────────────
    print("Loading SAM ViT-B + LoRA...")
    if os.path.exists(SAM_B_CKPT):
        sam_b = sam_model_registry["vit_b"](checkpoint=SAM_B_CKPT)
        lora_model = LoRA_Sam(sam_b, r=4, lora_alpha=8)
        has_lora = True
    else:
        sam_b_alt = find_path("sam_vit_b.pth", is_file=True)
        if os.path.exists(sam_b_alt):
            sam_b = sam_model_registry["vit_b"](checkpoint=sam_b_alt)
            lora_model = LoRA_Sam(sam_b, r=4, lora_alpha=8)
            has_lora = True
        else:
            print("  WARNING: sam_vit_b checkpoint not found, skipping LoRA methods")
            has_lora = False

    results_data = []

    fold_list = range(5) if target_fold is None else [target_fold]
    for eval_fold in fold_list:
        print(f"\n{'='*70}")
        print(f"EVAL LoRA: Fold {eval_fold} (GPU {gpu})")
        print(f"{'='*70}")

        # Load LoRA checkpoint for this fold
        if has_lora:
            lora_ckpt = os.path.join(LORA_CKPT_DIR, f"lora_fold{eval_fold}_best.pth")
            if os.path.exists(lora_ckpt):
                lora_model.load_lora_parameters(lora_ckpt)
                print(f"  Loaded LoRA weights: {lora_ckpt}")
            else:
                print(f"  WARNING: LoRA checkpoint not found: {lora_ckpt}")
                print(f"  Using un-trained LoRA (should match ViT-B baseline)")
            lora_model = lora_model.eval().cuda()
            predictor_b = SamPredictor(lora_model.sam)

        # Load FoB model
        ckpt_path = load_fob_checkpoint(eval_fold, CKPT_DIR)

        class DummyArgs:
            pass
        args_fob = DummyArgs()
        args_fob.dataset = "SABS"
        args_fob.max_points = 24

        model = FewShotSeg(args_fob)
        model.cuda()
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        model.eval()

        data_config = {
            'data_dir': data_dir, 'dataset': 'SABS', 'n_shot': 1, 'n_way': 1, 'n_query': 1,
            'n_sv': 5000, 'max_iter': 3000, 'eval_fold': eval_fold, 'min_size': 200,
            'max_slices': 3, 'supp_idx': SUPP_IDX,
        }
        test_dataset = TestDataset(data_config)
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=True, drop_last=False,
        )

        for label_val, label_name in labels.items():
            if label_name == 'BG' or label_val not in target_organs:
                continue

            print(f"\n  Testing: {label_name} (label={label_val})")
            support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=N_PART)
            test_dataset.label = label_val

            support_image = [support_sample['image'][[i]].float().cuda()
                             for i in range(support_sample['image'].shape[0])]
            support_fg_mask = [support_sample['label'][[i]].float().cuda()
                               for i in range(support_sample['image'].shape[0])]

            with torch.no_grad():
                for vi, sample in enumerate(test_loader):
                    vol_id_str = sample.get('id', [f"image_{vi}.nii.gz"])[0]
                    try:
                        vol_idx = int(vol_id_str.split("image_")[-1].split(".nii")[0])
                    except:
                        vol_idx = vi
                    spacing = get_voxel_spacing(vol_idx)

                    query_image = [sample['image'][j].float().cuda()
                                   for j in range(sample['image'].shape[0])]
                    query_label = sample['label'].long()
                    C_q = sample['image'].shape[1]

                    idx_ = np.linspace(0, C_q, N_PART + 1).astype('int')

                    # ── Accumulators ──────────────────────────────────
                    pred_baseline = np.zeros(query_label.shape[-3:], dtype=np.uint8)
                    pred_adafob_2d = np.zeros(query_label.shape[-3:], dtype=np.uint8)
                    pred_lora_fob = np.zeros(query_label.shape[-3:], dtype=np.uint8)
                    pred_lora_ada = np.zeros(query_label.shape[-3:], dtype=np.uint8)

                    # ── Per-slice FoB prompt generation ────────────────
                    for sub_chunk in range(N_PART):
                        support_image_s = [support_image[sub_chunk]]
                        support_fg_mask_s = [support_fg_mask[sub_chunk]]
                        query_image_s = query_image[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]
                        query_label_s = query_label[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]

                        for j in range(query_image_s.shape[0]):
                            global_z = idx_[sub_chunk] + j

                            # FoB uniform prompts (Np=10)
                            model.allocator = None
                            try:
                                uni_neg_p, pos_p = model(
                                    [support_image_s], [support_fg_mask_s],
                                    [query_image_s[[j]]], query_label_s[[j]], None
                                )
                                uni_neg = np.array(uni_neg_p).reshape(-1, 2)
                                pos_arr = np.array(pos_p).reshape(-1, 2)
                            except:
                                uni_neg, pos_arr = np.zeros((0, 2)), np.zeros((0, 2))

                            # AdaFoB adaptive prompts
                            allocator = PromptBudgetAllocator(max_points=24).cuda()
                            allocator.nu = global_params.get("nu", 0.02)
                            allocator.lam = global_params.get("lam", 0.5)
                            allocator.a0 = global_params.get("a0", 0.6)
                            allocator.tau = global_params.get("tau", 0.05)
                            model.allocator = allocator

                            original_allocate = model.allocator.allocate
                            def capturing_allocate(qry_img, qry_pred_coarse, spt_fg_proto,
                                                   supp_m, mdl, supp_fts):
                                a, contours, M_tilde = mdl.allocator.get_ambiguity_score(
                                    qry_img, qry_pred_coarse, spt_fg_proto, supp_m, mdl, supp_fts)
                                budget = mdl.allocator.compute_budget(a, contours, M_tilde)
                                mdl.allocator.last_budget = budget
                                r = mdl.allocator.get_scale_adaptive_offset(M_tilde)
                                pts = mdl.allocator.sample_placement(
                                    qry_img, M_tilde, contours, 24, r)
                                return pts, budget
                            model.allocator.allocate = capturing_allocate

                            try:
                                ada_neg_p, _ = model(
                                    [support_image_s], [support_fg_mask_s],
                                    [query_image_s[[j]]], query_label_s[[j]], None
                                )
                                ada_neg = np.array(ada_neg_p).reshape(-1, 2)
                            except:
                                ada_neg = np.zeros((0, 2))

                            slice_budget = getattr(model.allocator, 'last_budget', 10)
                            budget_np = min(VALID_NPS, key=lambda x: abs(x - slice_budget))

                            # Prepare SAM image
                            img_t = query_image_s[[j]][0].permute(1, 2, 0).cpu().numpy()
                            img_t = ((img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8) * 255).astype(np.uint8)

                            # ── Method 1: FoB Baseline (ViT-H, Np=10) ────
                            if has_vith:
                                predictor_h.set_image(img_t)
                                pred_baseline[global_z] = predict_with_sam(
                                    predictor_h, pos_arr[:10], uni_neg[:10], img_t.shape
                                )

                            # ── Method 2: AdaFoB 2D (ViT-H, adaptive Np) ─
                            if has_vith:
                                pred_adafob_2d[global_z] = predict_with_sam(
                                    predictor_h, pos_arr, ada_neg[:budget_np], img_t.shape
                                )

                            # ── Method 3: FoB + LoRA (ViT-B+LoRA, Np=10) ─
                            if has_lora:
                                predictor_b.set_image(img_t)
                                pred_lora_fob[global_z] = predict_with_sam(
                                    predictor_b, pos_arr[:10], uni_neg[:10], img_t.shape
                                )

                            # ── Method 4: AdaFoB + LoRA (ViT-B+LoRA, adaptive Np)
                            if has_lora:
                                pred_lora_ada[global_z] = predict_with_sam(
                                    predictor_b, pos_arr, ada_neg[:budget_np], img_t.shape
                                )

                    # ── Compute 3D metrics ────────────────────────────
                    gt = (query_label.squeeze(0).cpu().numpy() > 0).astype(np.uint8)

                    row = {
                        "fold": eval_fold,
                        "organ": label_name,
                        "vol_id": vol_id_str,
                        "spacing_x": spacing[0],
                        "spacing_y": spacing[1],
                        "spacing_z": spacing[2],
                    }

                    if has_vith:
                        row.update({
                            "dice_baseline": compute_volume_dice(pred_baseline, gt),
                            "hd95_baseline": compute_volume_hd95(pred_baseline, gt, spacing),
                            "dice_adafob_2d": compute_volume_dice(pred_adafob_2d, gt),
                            "hd95_adafob_2d": compute_volume_hd95(pred_adafob_2d, gt, spacing),
                        })

                    if has_lora:
                        row.update({
                            "dice_lora_fob": compute_volume_dice(pred_lora_fob, gt),
                            "hd95_lora_fob": compute_volume_hd95(pred_lora_fob, gt, spacing),
                            "dice_lora_ada": compute_volume_dice(pred_lora_ada, gt),
                            "hd95_lora_ada": compute_volume_hd95(pred_lora_ada, gt, spacing),
                        })

                    results_data.append(row)

                    # Print progress
                    parts = []
                    if has_vith:
                        parts.append(f"Baseline={row['dice_baseline']*100:.2f}%")
                        parts.append(f"AdaFoB2D={row['dice_adafob_2d']*100:.2f}%")
                    if has_lora:
                        parts.append(f"LoRA+FoB={row['dice_lora_fob']*100:.2f}%")
                        parts.append(f"LoRA+Ada={row['dice_lora_ada']*100:.2f}%")
                    print(f"    Vol {vi}: {', '.join(parts)}")

    # ── Save results ──────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(results_data)
    os.makedirs(os.path.join(REPO_DIR, "results"), exist_ok=True)
    out_csv = os.path.join(REPO_DIR, "results", f"lora_eval_gpu{gpu}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("LORA EVALUATION SUMMARY")
    print(f"{'='*80}")
    for organ in df['organ'].unique():
        odf = df[df['organ'] == organ]
        print(f"\n--- {organ} (n={len(odf)} volumes) ---")
        if has_vith:
            print(f"  FoB Baseline (ViT-H, Np=10):  Dice={odf['dice_baseline'].mean()*100:.2f}%  "
                  f"HD95={odf['hd95_baseline'].median():.1f}mm")
            print(f"  AdaFoB 2D (ViT-H):            Dice={odf['dice_adafob_2d'].mean()*100:.2f}%  "
                  f"HD95={odf['hd95_adafob_2d'].median():.1f}mm")
        if has_lora:
            print(f"  FoB + LoRA (ViT-B+LoRA):      Dice={odf['dice_lora_fob'].mean()*100:.2f}%  "
                  f"HD95={odf['hd95_lora_fob'].median():.1f}mm")
            print(f"  AdaFoB + LoRA (OURS):          Dice={odf['dice_lora_ada'].mean()*100:.2f}%  "
                  f"HD95={odf['hd95_lora_ada'].median():.1f}mm")

    print(f"\n{'='*80}")
    print("OVERALL MEAN")
    if has_vith:
        print(f"  FoB Baseline:       {df['dice_baseline'].mean()*100:.2f}%")
        print(f"  AdaFoB 2D:          {df['dice_adafob_2d'].mean()*100:.2f}%")
    if has_lora:
        print(f"  FoB + LoRA:         {df['dice_lora_fob'].mean()*100:.2f}%")
        print(f"  AdaFoB + LoRA:      {df['dice_lora_ada'].mean()*100:.2f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--organs', type=int, nargs='+', default=None,
                        help='Organ IDs (default: split by GPU)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold (0-4). Runs all if None.')
    args = parser.parse_args()

    gpu_id = int(args.gpu)
    if args.organs is None:
        if gpu_id == 0:
            organs = [1, 2]
        else:
            organs = [3, 6]
    else:
        organs = args.organs

    evaluate_lora(gpu=gpu_id, target_organs=organs, target_fold=args.fold)
