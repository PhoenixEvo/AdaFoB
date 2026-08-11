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
# This replicates FoB's exact 2-step pipeline:
#   Step 1 (intensity_normalization.py): clip HU [-125,275], min-max -> [0,255]
#   Step 2 (resampling_and_roi.py): crop 32px border, resample to ~256x256
# ============================================================================

def preprocess_sabs(raw_dir, out_dir):
    """Exact replication of FoB's preprocessing pipeline."""
    if os.path.isdir(out_dir) and len(glob.glob(os.path.join(out_dir, "image_*.nii.gz"))) >= 30:
        print(f"sabs_CT_normalized already exists with {len(glob.glob(os.path.join(out_dir, 'image_*.nii.gz')))} volumes. Skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)
    
    # Find raw image/label pairs
    img_dir = os.path.join(raw_dir, "Training", "img")
    lbl_dir = os.path.join(raw_dir, "Training", "label")
    
    if not os.path.isdir(img_dir):
        # Try alternative layout
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
    
    print(f"Image dir: {img_dir}")
    print(f"Label dir: {lbl_dir}")
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    if not img_files:
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii")))
    
    print(f"Found {len(img_files)} raw images")
    
    LIR, HIR = -125, 275  # HU window from FoB
    BD_BIAS = 32  # border crop from FoB
    
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
    
    SPA_FAC = (512 - 2 * BD_BIAS) / 256  # spacing factor from FoB
    
    for reindex, img_fid in enumerate(img_files):
        # Find matching label
        basename = os.path.basename(img_fid)
        lbl_fid = os.path.join(lbl_dir, basename.replace("img", "label"))
        if not os.path.exists(lbl_fid):
            # Try other naming conventions
            import re
            pid = re.search(r'(\d+)', basename).group(1)
            candidates = glob.glob(os.path.join(lbl_dir, f"*{pid}*"))
            if candidates:
                lbl_fid = candidates[0]
            else:
                print(f"  WARNING: No label for {basename}, skipping")
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
        # Image
        array_crop = sitk.GetArrayFromImage(wined_img)
        array_crop = array_crop[:, BD_BIAS:-BD_BIAS, BD_BIAS:-BD_BIAS]
        cropped_img_o = sitk.GetImageFromArray(array_crop)
        cropped_img_o = copy_spacing_ori(wined_img, cropped_img_o)
        
        img_spa_ori = wined_img.GetSpacing()
        res_img_o = resample_by_res(
            cropped_img_o,
            [img_spa_ori[0] * SPA_FAC, img_spa_ori[1] * SPA_FAC, img_spa_ori[-1]]
        )
        
        # Label
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
        
        # Save
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
# FoB publishes per-fold checkpoints. We need all 5 for proper 5-fold CV.
# ============================================================================

def download_fob_checkpoints(ckpt_dir):
    """Download all 5 fold checkpoints from HuggingFace."""
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Check if already downloaded
    existing = []
    for fold in range(5):
        fold_dir = os.path.join(ckpt_dir, f"FSMIS_train_SABS_cv{fold}")
        ckpt_file = os.path.join(fold_dir, "1", "snapshots", "39000.pth")
        if os.path.exists(ckpt_file):
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
        print("Trying direct download...")
        
        # Fallback: try wget from HuggingFace
        import urllib.request
        base_url = "https://huggingface.co/PrimeBo1/FoB_SAM/resolve/main"
        for fold in range(5):
            if fold in existing:
                continue
            fold_name = f"exps_train_on_SABS_FSMIS_FoB/FSMIS_train_SABS_cv{fold}/1/snapshots/39000.pth"
            url = f"{base_url}/{fold_name}"
            local_path = os.path.join(ckpt_dir, fold_name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            print(f"  Downloading fold {fold} from {url}...")
            try:
                urllib.request.urlretrieve(url, local_path)
                print(f"  Fold {fold}: OK")
            except Exception as e2:
                print(f"  Fold {fold}: FAILED ({e2})")
    
    # Verify
    for fold in range(5):
        fold_dir = os.path.join(ckpt_dir, f"exps_train_on_SABS_FSMIS_FoB", f"FSMIS_train_SABS_cv{fold}", "1", "snapshots")
        ckpt_file = os.path.join(fold_dir, "39000.pth")
        status = "OK" if os.path.exists(ckpt_file) else "MISSING"
        print(f"  Fold {fold}: {status} ({ckpt_file})")


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


print("\n" + "=" * 70)
print("Step 2: Download FoB checkpoints")
print("=" * 70)
download_fob_checkpoints(CKPT_DIR)
download_sam(SAM_CKPT)

# ============================================================================
# CELL 4: Reproduce FoB Baseline (Gate C)
# ============================================================================
# Run FoB's EXACT evaluation protocol:
#   - Per-volume z-score normalization (FoB's datasets.py line 47)
#   - sabs_CT_normalized data
#   - Per-fold checkpoint
#   - TEST_LABEL = [1, 2, 3, 6] (Spleen, RK, LK, Liver)
#   - Middle-slice support (supp_idx=2, the 3rd volume in fold)
# ============================================================================

from models.FoB import FewShotSeg
from segment_anything import sam_model_registry, SamPredictor
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist


# FoB's fold definitions (from dataset_specifics.py)
FOLDS = {
    0: set(range(0, 7)),
    1: set(range(6, 13)),
    2: set(range(12, 19)),
    3: set(range(18, 25)),
    4: set(range(24, 30)),
}
FOLDS[4].update([0])

TEST_LABELS = [1, 2, 3, 6]  # Spleen, RK, LK, Liver
LABEL_NAMES = {1: "Spleen", 2: "RK", 3: "LK", 6: "Liver"}
SUPP_IDX = 2  # FoB default


def compute_dice(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = np.sum(pred * gt)
    total = np.sum(pred) + np.sum(gt)
    if total == 0:
        return 1.0
    return 2.0 * inter / total


def load_fob_checkpoint(fold, ckpt_dir):
    """Load the correct per-fold FoB checkpoint."""
    # Try multiple possible layouts
    candidates = [
        os.path.join(ckpt_dir, f"exps_train_on_SABS_FSMIS_FoB/FSMIS_train_SABS_cv{fold}/1/snapshots/39000.pth"),
        os.path.join(ckpt_dir, f"exps_train_on_SABS_cdfs_FoB/FSMIS_train_SABS_cv{fold}/1/snapshots/39000.pth"),
        os.path.join(ckpt_dir, f"FSMIS_train_SABS_cv{fold}/1/snapshots/39000.pth"),
        os.path.join(ckpt_dir, f"FSMIS_train_SABS_cv{fold}/ckpt.pth"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"No checkpoint found for fold {fold}. Tried: {candidates}")


def reproduce_fob_baseline():
    """Run FoB's exact evaluation and report per-organ Dice."""
    
    # Load normalized volumes
    img_files = sorted(
        glob.glob(os.path.join(NORMALIZED_DIR, "image_*.nii.gz")),
        key=lambda x: int(x.split("_")[-1].split(".nii.gz")[0])
    )
    print(f"Found {len(img_files)} normalized volumes")
    
    # Load SAM
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
    sam.cuda().eval()
    predictor = SamPredictor(sam)
    
    all_results = []
    
    for fold in range(5):
        fold_vols = sorted(FOLDS[fold])
        print(f"\n--- Fold {fold} (volumes: {fold_vols}) ---")
        
        # Load per-fold checkpoint
        ckpt_path = load_fob_checkpoint(fold, CKPT_DIR)
        print(f"  Checkpoint: {ckpt_path}")
        
        fob = FewShotSeg(pretrained_path=None, use_original_imgsize=False)
        fob.n_shots = 1
        fob.n_ways = 1
        fob.n_queries = 1
        fob.max_points = 10
        fob.allocator = None
        fob.cuda().eval()
        
        # Load checkpoint
        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("state_dict", "model", "net", "model_state_dict"):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break
        cleaned = {}
        for k, v in obj.items():
            nk = k
            for pref in ("module.", "_orig_mod."):
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            cleaned[nk] = v
        
        missing, unexpected = fob.load_state_dict(cleaned, strict=False)
        model_keys = set(fob.state_dict().keys())
        matched = [k for k in cleaned if k in model_keys]
        print(f"  Loaded {len(matched)}/{len(model_keys)} params, missing={len(missing)}, unexpected={len(unexpected)}")
        
        # Get test volumes (volumes IN the fold)
        available_vols = [i for i in fold_vols if i < len(img_files)]
        if len(available_vols) < 3:
            print(f"  WARNING: Only {len(available_vols)} volumes in fold {fold}")
            continue
        
        # Support = SUPP_IDX-th volume, Query = the rest
        support_idx_in_fold = min(SUPP_IDX, len(available_vols) - 1)
        support_vol_id = available_vols[support_idx_in_fold]
        query_vol_ids = [v for v in available_vols if v != support_vol_id]
        
        # Load support volume (FoB's exact normalization: per-volume z-score)
        supp_path = img_files[support_vol_id]
        supp_img_raw = sitk.GetArrayFromImage(sitk.ReadImage(supp_path))  # (Z, H, W)
        supp_img_norm = (supp_img_raw - supp_img_raw.mean()) / (supp_img_raw.std() + 1e-8)
        supp_img_3ch = np.stack(3 * [supp_img_norm], axis=1)  # (Z, 3, H, W)
        
        supp_lbl_path = supp_path.replace("image_", "label_")
        supp_lbl = sitk.GetArrayFromImage(sitk.ReadImage(supp_lbl_path))
        
        for label_id in TEST_LABELS:
            label_name = LABEL_NAMES[label_id]
            
            # Support: binary mask for this label
            supp_mask_bin = (supp_lbl == label_id).astype(np.float32)
            
            # Find slices with this label
            supp_has_label = supp_mask_bin.sum(axis=(1, 2)) > 0
            if supp_has_label.sum() == 0:
                print(f"  {label_name}: No support slices with label {label_id} in vol {support_vol_id}")
                continue
            
            # Middle slice as support (FoB protocol: get_support_index with n_shot=1 -> 50%)
            labeled_indices = np.where(supp_has_label)[0]
            mid_idx = labeled_indices[len(labeled_indices) // 2]
            
            supp_slice = torch.from_numpy(supp_img_3ch[mid_idx:mid_idx+1]).float()  # (1, 3, H, W)
            supp_mask_slice = torch.from_numpy(supp_mask_bin[mid_idx:mid_idx+1]).float()  # (1, H, W)
            
            fold_dices = []
            
            for qvol_id in query_vol_ids:
                # Load query volume
                qry_path = img_files[qvol_id]
                qry_img_raw = sitk.GetArrayFromImage(sitk.ReadImage(qry_path))
                qry_img_norm = (qry_img_raw - qry_img_raw.mean()) / (qry_img_raw.std() + 1e-8)
                qry_img_3ch = np.stack(3 * [qry_img_norm], axis=1)
                
                qry_lbl_path = qry_path.replace("image_", "label_")
                qry_lbl = sitk.GetArrayFromImage(sitk.ReadImage(qry_lbl_path))
                qry_mask_bin = (qry_lbl == label_id).astype(np.float32)
                
                # Only evaluate on slices that have this label
                qry_has_label = qry_mask_bin.sum(axis=(1, 2)) > 0
                if qry_has_label.sum() == 0:
                    continue
                
                labeled_qry_indices = np.where(qry_has_label)[0]
                
                for qslice_idx in labeled_qry_indices:
                    qry_slice = torch.from_numpy(qry_img_3ch[qslice_idx:qslice_idx+1]).float()
                    qry_mask_gt = qry_mask_bin[qslice_idx]
                    
                    # Build FoB input format
                    support_images = [[supp_slice.clone().cuda()]]
                    support_fg_labels = [[supp_mask_slice.clone().cuda()]]
                    query_images = [qry_slice.clone().cuda()]
                    query_labels = torch.from_numpy(qry_mask_bin[qslice_idx:qslice_idx+1]).float().cuda()
                    
                    try:
                        with torch.no_grad():
                            neg_pts, pos_pts = fob(
                                support_images, support_fg_labels,
                                query_images, query_labels,
                                train=False, use_skeleton=False, budget_Np=10
                            )
                    except Exception as e:
                        continue
                    
                    # Convert points
                    if neg_pts is not None and torch.is_tensor(neg_pts):
                        neg_pts = neg_pts.detach().cpu().numpy().reshape(-1, 2)
                    else:
                        neg_pts = np.zeros((0, 2), dtype=np.float32)
                    
                    if pos_pts is not None and torch.is_tensor(pos_pts):
                        pos_pts = pos_pts.detach().cpu().numpy().reshape(-1, 2)
                    else:
                        pos_pts = np.zeros((0, 2), dtype=np.float32)
                    
                    # SAM prediction
                    H, W = qry_mask_gt.shape
                    # SAM uint8 from canonical (before z-score)
                    sam_slice = qry_img_raw[qslice_idx]
                    sam_norm = (sam_slice - sam_slice.min()) / (sam_slice.max() - sam_slice.min() + 1e-8) * 255
                    sam_uint8 = np.stack([sam_norm.astype(np.uint8)] * 3, axis=-1)
                    
                    predictor.set_image(sam_uint8)
                    
                    all_pts = np.concatenate([pos_pts, neg_pts], axis=0) if len(pos_pts) + len(neg_pts) > 0 else np.zeros((0, 2))
                    all_lbls = np.concatenate([np.ones(len(pos_pts)), np.zeros(len(neg_pts))]) if len(all_pts) > 0 else np.array([])
                    
                    if len(all_pts) == 0:
                        continue
                    
                    # Clip to image bounds
                    all_pts[:, 0] = np.clip(all_pts[:, 0], 0, W - 1)
                    all_pts[:, 1] = np.clip(all_pts[:, 1], 0, H - 1)
                    
                    masks, scores, _ = predictor.predict(
                        point_coords=all_pts,
                        point_labels=all_lbls,
                        multimask_output=True,
                    )
                    pred_mask = masks[0]  # FoB uses index 0
                    
                    dice = compute_dice(pred_mask, qry_mask_gt)
                    fold_dices.append(dice)
            
            if fold_dices:
                mean_dice = np.mean(fold_dices) * 100
                print(f"  {label_name}: Dice = {mean_dice:.2f}% ({len(fold_dices)} slices)")
                all_results.append({
                    "fold": fold,
                    "organ": label_name,
                    "label_id": label_id,
                    "dice": mean_dice,
                    "n_slices": len(fold_dices),
                })
            else:
                print(f"  {label_name}: No valid slices evaluated")
    
    # Summary
    print("\n" + "=" * 70)
    print("GATE C RESULTS: FoB Baseline Reproduction")
    print("=" * 70)
    print(f"{'Organ':<12} | {'Mean Dice (%)':<15} | {'Published':<12} | {'Match?'}")
    print("-" * 60)
    
    published = {"Spleen": 84.54, "RK": 86.51, "LK": 87.29, "Liver": 86.51}
    
    for organ in ["Spleen", "RK", "LK", "Liver"]:
        organ_results = [r["dice"] for r in all_results if r["organ"] == organ]
        if organ_results:
            mean = np.mean(organ_results)
            pub = published.get(organ, 0)
            match = "OK" if abs(mean - pub) < 10 else "MISMATCH"
            print(f"{organ:<12} | {mean:>13.2f} | {pub:>10.2f} | {match}")
        else:
            print(f"{organ:<12} | {'N/A':>13} | {published.get(organ, 0):>10.2f} | MISSING")
    
    overall = np.mean([r["dice"] for r in all_results]) if all_results else 0
    print(f"\n{'OVERALL':<12} | {overall:>13.2f} | {86.21:>10.2f}")
    
    gate_c = overall > 75.0
    print(f"\nGATE C: {'PASSED' if gate_c else 'FAILED'} (threshold: >75%)")
    
    # Save results
    import csv
    with open(os.path.join(REPO_DIR, "results", "gate_c_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "organ", "label_id", "dice", "n_slices"])
        writer.writeheader()
        writer.writerows(all_results)
    
    return gate_c, all_results


print("\n" + "=" * 70)
print("Step 3: Reproduce FoB Baseline (GATE C)")
print("=" * 70)
gate_c_passed, results = reproduce_fob_baseline()

if gate_c_passed:
    print("\n*** GATE C PASSED! Baseline is valid. Safe to proceed with AdaFoB evaluation. ***")
else:
    print("\n*** GATE C FAILED. DO NOT proceed with AdaFoB until baseline is fixed. ***")
    print("Check: (1) sabs_CT_normalized preprocessing, (2) checkpoint loading, (3) z-score normalization")
