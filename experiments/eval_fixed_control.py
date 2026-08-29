"""
Fixed-Mean-Budget Control Evaluation
======================================
Tests the null hypothesis: "Would a FIXED lower budget achieve the same
result as AdaFoB's adaptive allocation?"

Uses the SAME pipeline as eval_lora.py but replaces the adaptive budget
with a per-organ constant Np = round(mean_adaptive_Np).

Usage:
    python experiments/eval_fixed_control.py --gpu 0 \
        --lora_ckpt_dir /path/to/checkpoints \
        --np_spleen 3 --np_rk 4 --np_lk 3 --np_liver 2
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import torch

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
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import get_label_names, get_folds
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry, SamPredictor

from experiments.lora_sam import LoRA_Sam

# Reuse functions from eval_lora
from experiments.eval_lora import (
    find_path, get_voxel_spacing, load_fob_checkpoint,
    compute_volume_dice, compute_volume_hd95, predict_with_sam,
    NORMALIZED_DIR, CKPT_DIR, GLOBAL_PARAMS_PATH,
    TEST_LABELS, N_PART, SUPP_IDX
)


def evaluate_fixed_control(gpu=0, lora_ckpt_dir=None, fixed_budgets=None):
    """
    Run evaluation with FIXED per-organ negative prompt budget using LoRA SAM.
    Mirrors eval_lora.py exactly, but uses fixed Np instead of adaptive.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    SAM_B_CKPT = find_path("sam_vit_b_01ec64.pth", is_file=True)

    # Load global params (needed for FoB model init)
    global_params = {"nu": 0.02, "lam": 0.5, "a0": 0.6, "tau": 0.05}
    if os.path.exists(GLOBAL_PARAMS_PATH):
        with open(GLOBAL_PARAMS_PATH, "r") as f:
            global_params.update(json.load(f))
        print(f"Loaded Global Params: {global_params}")

    labels = get_label_names("SABS")
    data_dir = os.path.dirname(NORMALIZED_DIR)

    # Init SAM ViT-B + LoRA
    print("Loading SAM ViT-B + LoRA...")
    sam_b = sam_model_registry["vit_b"](checkpoint=SAM_B_CKPT)
    lora_model = LoRA_Sam(sam_b, r=4, lora_alpha=8)

    results_data = []

    for eval_fold in range(5):
        print(f"\n{'='*70}")
        print(f"EVAL Fixed Control: Fold {eval_fold} (GPU {gpu})")
        print(f"{'='*70}")

        # Load LoRA checkpoint for this fold
        ckpt_dir_to_use = lora_ckpt_dir or os.path.join(REPO_DIR, "outputs", "lora_checkpoints")
        lora_ckpt = os.path.join(ckpt_dir_to_use, f"lora_fold{eval_fold}_best.pth")
        if os.path.exists(lora_ckpt):
            lora_model.load_lora_parameters(lora_ckpt)
            print(f"  Loaded LoRA weights: {lora_ckpt}")
        else:
            print(f"  WARNING: LoRA checkpoint not found: {lora_ckpt}")
            print(f"  Using un-trained LoRA")
        lora_model = lora_model.eval().cuda()
        predictor_b = SamPredictor(lora_model.sam)

        # Load FoB model (same as eval_lora.py)
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
            if label_name == 'BG' or label_val not in TEST_LABELS:
                continue

            target_np = fixed_budgets[label_val]
            print(f"\n  Testing: {label_name} (label={label_val}, fixed Np={target_np})")
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

                    pred_fixed = np.zeros(query_label.shape[-3:], dtype=np.uint8)

                    for sub_chunk in range(N_PART):
                        support_image_s = [support_image[sub_chunk]]
                        support_fg_mask_s = [support_fg_mask[sub_chunk]]
                        query_image_s = query_image[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]
                        query_label_s = query_label[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]

                        for j in range(query_image_s.shape[0]):
                            global_z = idx_[sub_chunk] + j

                            # Generate UNIFORM prompts (same as FoB baseline)
                            # No allocator, standard FoB forward pass
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

                            # Prepare SAM image (identical to eval_lora.py)
                            img_t = query_image_s[[j]][0].permute(1, 2, 0).cpu().numpy()
                            img_t = ((img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8) * 255).astype(np.uint8)

                            # Fixed budget: use UNIFORM points, capped at target_np
                            predictor_b.set_image(img_t)
                            pred_fixed[global_z] = predict_with_sam(
                                predictor_b, pos_arr[:10], uni_neg[:target_np], img_t.shape
                            )

                    # Compute 3D metrics
                    gt = (query_label.squeeze(0).cpu().numpy() > 0).astype(np.uint8)

                    row = {
                        "fold": eval_fold,
                        "organ": label_name,
                        "vol_id": vol_id_str,
                        "spacing_x": spacing[0],
                        "spacing_y": spacing[1],
                        "spacing_z": spacing[2],
                        "fixed_np": target_np,
                        "dice_fixed_control": compute_volume_dice(pred_fixed, gt),
                        "hd95_fixed_control": compute_volume_hd95(pred_fixed, gt, spacing),
                    }

                    results_data.append(row)
                    print(f"    Vol {vi}: Fixed(Np={target_np}) Dice={row['dice_fixed_control']*100:.2f}%")

    # Save results
    import pandas as pd
    df = pd.DataFrame(results_data)
    os.makedirs(os.path.join(REPO_DIR, "results"), exist_ok=True)
    out_csv = os.path.join(REPO_DIR, "results", f"control_fixed_mean_budget_gpu{gpu}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    # Print summary
    print(f"\n{'='*80}")
    print("FIXED CONTROL EVALUATION SUMMARY")
    print(f"{'='*80}")
    for organ in df['organ'].unique():
        odf = df[df['organ'] == organ]
        np_used = odf['fixed_np'].iloc[0]
        print(f"\n--- {organ} (n={len(odf)}, Np={np_used}) ---")
        print(f"  Dice: {odf['dice_fixed_control'].mean()*100:.2f}%  "
              f"HD95: {odf['hd95_fixed_control'].mean():.1f}mm")

    print(f"\n{'='*80}")
    print(f"OVERALL MEAN: {df['dice_fixed_control'].mean()*100:.2f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lora_ckpt_dir', type=str, default=None)
    parser.add_argument('--np_spleen', type=int, required=True, help='Fixed Np for Spleen')
    parser.add_argument('--np_rk', type=int, required=True, help='Fixed Np for RK')
    parser.add_argument('--np_lk', type=int, required=True, help='Fixed Np for LK')
    parser.add_argument('--np_liver', type=int, required=True, help='Fixed Np for Liver')
    args = parser.parse_args()

    fixed_budgets = {
        1: args.np_spleen,
        2: args.np_rk,
        3: args.np_lk,
        6: args.np_liver,
    }

    print(f"Fixed budgets: Spleen={args.np_spleen}, RK={args.np_rk}, "
          f"LK={args.np_lk}, Liver={args.np_liver}")

    evaluate_fixed_control(
        gpu=args.gpu,
        lora_ckpt_dir=args.lora_ckpt_dir,
        fixed_budgets=fixed_budgets,
    )
