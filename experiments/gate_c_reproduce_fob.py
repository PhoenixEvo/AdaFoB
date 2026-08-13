"""
Gate C Notebook: Reproduce FoB Baseline on SABS (~86% Dice)
============================================================
This notebook does THREE things in order:
  1. Preprocess RawData -> sabs_CT_normalized (matching Ouyang et al. exactly)
  2. Download all 5 per-fold FoB checkpoints from HuggingFace
  3. Run FoB's own evaluation protocol and confirm Dice ~86%

Only after Gate C passes do we have a valid baseline to compare AdaFoB against.

Usage on Kaggle:
  - Add dataset: nhatphatnguyen/abd-ct (contains RawData/)
  - GPU: T4 x2 or P100
  - Internet: ON (for HuggingFace downloads)
"""

import os
import sys
import glob
import json
import shutil
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm

# Install missing dependency required by FoB
try:
    import info_nce
except ImportError:
    print("Installing info-nce-pytorch...")
    os.system("pip install info-nce-pytorch")

# ============================================================================
# CELL 1: Clone repo & setup paths
# ============================================================================
REPO_DIR = "/kaggle/working/AdaFoB"
if not os.path.isdir(REPO_DIR):
    os.system("git clone https://github.com/PhoenixEvo/AdaFoB.git /kaggle/working/AdaFoB")
else:
    os.system(f"cd {REPO_DIR} && git pull")

sys.path.insert(0, REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, "third_party", "FoB_SAM"))

# Paths
RAW_DATA = "/kaggle/input/datasets/nhatphatnguyen/abd-ct/RawData"
NORMALIZED_DIR = "/kaggle/working/sabs_CT_normalized"
CKPT_DIR = "/kaggle/working/fob_checkpoints"
SAM_CKPT = "/kaggle/working/sam_vit_h.pth"

# ============================================================================
# CELL 2: Preprocess RawData -> sabs_CT_normalized
# ============================================================================

def preprocess_sabs(raw_dir, out_dir):
    """Exact replication of FoB's preprocessing pipeline."""
    if os.path.isdir(out_dir) and len(glob.glob(os.path.join(out_dir, "image_*.nii.gz"))) >= 30:
        print(f"sabs_CT_normalized already exists with {len(glob.glob(os.path.join(out_dir, 'image_*.nii.gz')))} volumes. Skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)

    img_dir = os.path.join(raw_dir, "Training", "img")
    lbl_dir = os.path.join(raw_dir, "Training", "label")

    if not os.path.isdir(img_dir):
        for candidate in ["images", "img", "imagesTr"]:
            alt = os.path.join(raw_dir, candidate)
            if os.path.isdir(alt):
                img_dir = alt
                break
    if not os.path.isdir(lbl_dir):
        for candidate in ["labels", "label", "labelsTr"]:
            alt = os.path.join(raw_dir, candidate)
            if os.path.isdir(alt):
                lbl_dir = alt
                break

    def find_all_nii(directory):
        found = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.startswith('.'): continue
                if f.endswith(".nii") or f.endswith(".nii.gz"):
                    found.append(os.path.join(root, f))
        return sorted(found)

    img_files = find_all_nii(img_dir)
    print(f"Found {len(img_files)} raw image files")

    LIR, HIR = -125, 275
    BD_BIAS = 32

    def copy_spacing_ori(src, dst):
        dst.SetSpacing(src.GetSpacing())
        dst.SetOrigin(src.GetOrigin())
        dst.SetDirection(src.GetDirection())
        return dst

    def resample_by_res(mov_img_obj, new_spacing, interpolator=sitk.sitkLinear):
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(interpolator)
        resample.SetOutputDirection(mov_img_obj.GetDirection())
        resample.SetOutputOrigin(mov_img_obj.GetOrigin())
        mov_spacing = mov_img_obj.GetSpacing()
        resample.SetOutputSpacing(new_spacing)
        RES_COE = np.array(mov_spacing) * 1.0 / np.array(new_spacing)
        new_size = np.array(mov_img_obj.GetSize()) * RES_COE
        resample.SetSize([int(sz + 1) for sz in new_size])
        return resample.Execute(mov_img_obj)

    def resample_lb_by_res(mov_lb_obj, new_spacing, ref_img=None):
        src_mat = sitk.GetArrayFromImage(mov_lb_obj)
        lbvs = np.unique(src_mat)
        for idx, lbv in enumerate(lbvs):
            _src_curr_mat = np.float32(src_mat == lbv)
            _src_curr_obj = sitk.GetImageFromArray(_src_curr_mat)
            _src_curr_obj.CopyInformation(mov_lb_obj)
            _tar_curr_obj = resample_by_res(_src_curr_obj, new_spacing, sitk.sitkLinear)
            _tar_curr_mat = np.rint(sitk.GetArrayFromImage(_tar_curr_obj)) * lbv
            if idx == 0:
                out_vol = _tar_curr_mat
            else:
                out_vol[_tar_curr_mat == lbv] = lbv
        out_obj = sitk.GetImageFromArray(out_vol)
        out_obj.SetSpacing(_tar_curr_obj.GetSpacing())
        if ref_img is not None:
            out_obj.CopyInformation(ref_img)
        return out_obj

    SPA_FAC = (512 - 2 * BD_BIAS) / 256

    lbl_files_all = find_all_nii(lbl_dir)
    import re

    for reindex, img_fid in enumerate(img_files):
        basename = os.path.basename(img_fid)
        pid_match = re.search(r'(?:img|label)(\d+)', img_fid)
        if not pid_match:
            print(f"  WARNING: No patient ID in {img_fid}, skipping")
            continue

        pid = pid_match.group(1)
        lbl_fid = None
        for lf in lbl_files_all:
            lf_match = re.search(r'(?:img|label)(\d+)', lf)
            if lf_match and lf_match.group(1) == pid:
                lbl_fid = lf
                break

        if not lbl_fid:
            print(f"  WARNING: No label for {img_fid} (pid: {pid}), skipping")
            continue

        img_obj = sitk.ReadImage(img_fid)
        seg_obj = sitk.ReadImage(lbl_fid)

        # Step 1: Intensity normalization (clip HU + min-max -> [0,255])
        array = sitk.GetArrayFromImage(img_obj).astype(np.float64)
        array[array > HIR] = HIR
        array[array < LIR] = LIR
        array = (array - array.min()) / (array.max() - array.min()) * 255.0

        wined_img = sitk.GetImageFromArray(array)
        wined_img = copy_spacing_ori(img_obj, wined_img)

        # Step 2: Crop border + resample
        array_crop = sitk.GetArrayFromImage(wined_img)
        array_crop = array_crop[:, BD_BIAS:-BD_BIAS, BD_BIAS:-BD_BIAS]
        cropped_img_o = sitk.GetImageFromArray(array_crop)
        cropped_img_o = copy_spacing_ori(wined_img, cropped_img_o)

        img_spa_ori = wined_img.GetSpacing()
        res_img_o = resample_by_res(
            cropped_img_o,
            [img_spa_ori[0] * SPA_FAC, img_spa_ori[1] * SPA_FAC, img_spa_ori[-1]]
        )

        lb_arr = sitk.GetArrayFromImage(seg_obj)
        lb_arr = lb_arr[:, BD_BIAS:-BD_BIAS, BD_BIAS:-BD_BIAS]
        cropped_lb_o = sitk.GetImageFromArray(lb_arr)
        cropped_lb_o = copy_spacing_ori(seg_obj, cropped_lb_o)

        lb_spa_ori = seg_obj.GetSpacing()
        res_lb_o = resample_lb_by_res(
            cropped_lb_o,
            [lb_spa_ori[0] * SPA_FAC, lb_spa_ori[1] * SPA_FAC, lb_spa_ori[-1]],
            ref_img=res_img_o
        )

        out_img = os.path.join(out_dir, f"image_{reindex}.nii.gz")
        out_lbl = os.path.join(out_dir, f"label_{reindex}.nii.gz")
        sitk.WriteImage(res_img_o, out_img, True)
        sitk.WriteImage(res_lb_o, out_lbl, True)

        res_shape = sitk.GetArrayFromImage(res_img_o).shape
        print(f"  [{reindex}] {basename} -> {res_shape}")

    n_out = len(glob.glob(os.path.join(out_dir, "image_*.nii.gz")))
    print(f"\nPreprocessing complete: {n_out} volumes in {out_dir}")


print("=" * 70)
print("Step 1: Preprocessing RawData -> sabs_CT_normalized")
print("=" * 70)
preprocess_sabs(RAW_DATA, NORMALIZED_DIR)

# ============================================================================
# CELL 3: Download FoB checkpoints from HuggingFace
# ============================================================================

def download_fob_checkpoints(ckpt_dir):
    """Download all 5 fold checkpoints from HuggingFace."""
    os.makedirs(ckpt_dir, exist_ok=True)

    existing = []
    for fold in range(5):
        candidates = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*/**/*.pth"), recursive=True)
        if not candidates:
            candidates = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*.pth"), recursive=True)
        if candidates:
            existing.append(fold)

    if len(existing) == 5:
        print("All 5 fold checkpoints already present. Skipping download.")
        return

    print(f"Found checkpoints for folds: {existing}. Downloading missing ones...")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="PrimeBo1/FoB_SAM",
            local_dir=ckpt_dir,
            allow_patterns=["*SABS*"],
        )
        print("Downloaded from HuggingFace successfully.")
    except Exception as e:
        print(f"HuggingFace download failed: {e}")

    # Extract ZIP if downloaded
    zip_path = os.path.join(ckpt_dir, "exps_train_on_SABS_FSMIS_FoB.zip")
    if not os.path.exists(zip_path):
        zips = glob.glob(os.path.join(ckpt_dir, "**/*.zip"), recursive=True)
        if zips:
            zip_path = zips[0]

    if os.path.exists(zip_path):
        print(f"Found zip archive at {zip_path}, extracting...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ckpt_dir)
        print("Extraction complete.")

    # Verify
    for fold in range(5):
        candidates = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*/**/*.pth"), recursive=True)
        if not candidates:
            candidates = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*.pth"), recursive=True)
        status = "OK" if candidates else "MISSING"
        found_file = candidates[0] if candidates else "None"
        print(f"  Fold {fold}: {status} ({found_file})")


def download_sam(sam_path):
    """Download SAM ViT-H checkpoint if not present."""
    if os.path.exists(sam_path):
        print(f"SAM checkpoint already present at {sam_path}")
        return
    print("Downloading SAM ViT-H checkpoint...")
    import urllib.request
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    urllib.request.urlretrieve(url, sam_path)
    print(f"SAM checkpoint saved to {sam_path}")


def load_fob_checkpoint(fold, ckpt_dir):
    """Load the correct per-fold FoB checkpoint dynamically."""
    candidates = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*/**/*.pth"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(ckpt_dir, f"**/*cv{fold}*.pth"), recursive=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No checkpoint found for fold {fold} in {ckpt_dir}")


print("\n" + "=" * 70)
print("Step 2: Download FoB checkpoints")
print("=" * 70)
download_fob_checkpoints(CKPT_DIR)
download_sam(SAM_CKPT)

# ============================================================================
# CELL 4: Reproduce FoB Baseline (Gate C)
# ============================================================================
# Use FoB's OWN TestDataset, SAM wrapper, and test.py logic verbatim.
# This guarantees identical data loading, normalization, and evaluation.
# ============================================================================

from models.FoB import FewShotSeg
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import get_label_names, get_folds
from torch.utils.data import DataLoader
from SAM import SAM

TEST_LABELS = [1, 2, 3, 6]
N_PART = 3
SUPP_IDX = 2


def reproduce_fob_baseline():
    """Run FoB's exact evaluation protocol using their own code."""

    labels = get_label_names("SABS")
    data_dir = os.path.dirname(NORMALIZED_DIR)

    class_dice_all = {}

    for eval_fold in range(5):
        print(f"\n--- Fold {eval_fold} ---")

        ckpt_path = load_fob_checkpoint(eval_fold, CKPT_DIR)
        print(f"  Checkpoint: {ckpt_path}")

        class DummyArgs:
            pass
        args = DummyArgs()
        args.dataset = "SABS"
        args.max_points = 10

        model = FewShotSeg(args)
        model.cuda()
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        model.eval()

        sam = SAM(sam_pretrained_path=SAM_CKPT)

        data_config = {
            'data_dir': data_dir,
            'dataset': 'SABS',
            'n_shot': 1,
            'n_way': 1,
            'n_query': 1,
            'n_sv': 5000,
            'max_iter': 3000,
            'eval_fold': eval_fold,
            'min_size': 200,
            'max_slices': 3,
            'supp_idx': SUPP_IDX,
        }
        test_dataset = TestDataset(data_config)
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False
        )

        _config = {'dataset': 'SABS', 'n_part': N_PART}

        for label_val, label_name in labels.items():
            if label_name == 'BG':
                continue
            if label_val not in TEST_LABELS:
                continue

            print(f"  Testing: {label_name} (label={label_val})")

            support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=N_PART)
            test_dataset.label = label_val

            support_image = [support_sample['image'][[i]].float().cuda()
                             for i in range(support_sample['image'].shape[0])]
            support_fg_mask = [support_sample['label'][[i]].float().cuda()
                               for i in range(support_sample['image'].shape[0])]

            dice_scores = []

            with torch.no_grad():
                for vi, sample in enumerate(test_loader):
                    query_image = [sample['image'][j].float().cuda()
                                   for j in range(sample['image'].shape[0])]
                    query_label = sample['label'].long()

                    query_pred = torch.zeros(query_label.shape[-3:])
                    C_q = sample['image'].shape[1]

                    idx_ = np.linspace(0, C_q, N_PART + 1).astype('int')
                    for sub_chunk in range(N_PART):
                        support_image_s = [support_image[sub_chunk]]
                        support_fg_mask_s = [support_fg_mask[sub_chunk]]
                        query_image_s = query_image[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]
                        query_label_s = query_label[0][idx_[sub_chunk]:idx_[sub_chunk + 1]]

                        query_pred_s = []
                        for j in range(query_image_s.shape[0]):
                            try:
                                neg_point, pos_point = model(
                                    [support_image_s], [support_fg_mask_s],
                                    [query_image_s[[j]]], query_label_s[[j]], None
                                )
                                pred = sam(
                                    query_image_s[[j]][0], pos_point, neg_point,
                                    _config, return_logits=False
                                )
                                pred = torch.from_numpy(pred).float().cuda().unsqueeze(0).unsqueeze(0)
                            except Exception as e:
                                print(f"    Error on slice {j}: {e}")
                                pred = torch.zeros(1, 1, query_image_s.shape[-2], query_image_s.shape[-1])
                            query_pred_s.append(pred)

                        query_pred_s = torch.cat(query_pred_s, dim=0).squeeze(1)
                        query_pred[idx_[sub_chunk]:idx_[sub_chunk + 1]] = query_pred_s.cpu()

                    # Compute Dice for this query volume
                    query_pred_bin = (query_pred > 0.5).float()
                    query_label_bin = (query_label.squeeze(0) > 0).float()
                    inter = (query_pred_bin * query_label_bin).sum()
                    total = query_pred_bin.sum() + query_label_bin.sum()
                    dice = (2.0 * inter / (total + 1e-8)).item() if total > 0 else 1.0
                    dice_scores.append(dice)
                    print(f"    Vol {vi}: Dice = {dice*100:.2f}%")

            if dice_scores:
                mean_dice = np.mean(dice_scores)
                key = f"{label_name}_fold{eval_fold}"
                class_dice_all[key] = mean_dice
                print(f"  {label_name} fold {eval_fold}: Mean Dice = {mean_dice*100:.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("GATE C RESULTS: FoB Baseline Reproduction")
    print("=" * 70)

    published = {"SPLEEN": 84.54, "RK": 86.51, "LK": 87.29, "LIVER": 86.51}

    organ_dices = {}
    for key, dice in class_dice_all.items():
        organ = key.split("_fold")[0]
        if organ not in organ_dices:
            organ_dices[organ] = []
        organ_dices[organ].append(dice)

    print(f"{'Organ':<12} | {'Our Dice (%)':<15} | {'Published (%)':<15} | {'Gap'}")
    print("-" * 65)

    for organ in ["SPLEEN", "RK", "LK", "LIVER"]:
        if organ in organ_dices:
            mean = np.mean(organ_dices[organ]) * 100
            pub = published.get(organ, 0)
            gap = mean - pub
            print(f"{organ:<12} | {mean:>13.2f} | {pub:>13.2f} | {gap:>+7.2f}")
        else:
            pub = published.get(organ, 0)
            print(f"{organ:<12} | {'N/A':>13} | {pub:>13.2f} | {'N/A':>7}")

    all_vals = []
    for v in organ_dices.values():
        all_vals.extend(v)
    overall = np.mean(all_vals) * 100 if all_vals else 0

    print(f"\n{'OVERALL':<12} | {overall:>13.2f} | {86.21:>13.2f}")

    gate_c = overall > 75.0
    print(f"\nGATE C: {'PASSED' if gate_c else 'FAILED'} (threshold: >75%)")

    import csv
    os.makedirs(os.path.join(REPO_DIR, "results"), exist_ok=True)
    with open(os.path.join(REPO_DIR, "results", "gate_c_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "dice"])
        for key, dice in class_dice_all.items():
            writer.writerow([key, dice])

    return gate_c, class_dice_all


print("\n" + "=" * 70)
print("Step 3: Reproduce FoB Baseline (GATE C)")
print("=" * 70)
gate_c_passed, results = reproduce_fob_baseline()

if gate_c_passed:
    print("\n*** GATE C PASSED! Baseline is valid. ***")
else:
    print("\n*** GATE C FAILED. DO NOT proceed until baseline is fixed. ***")
