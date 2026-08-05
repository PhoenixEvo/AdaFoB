"""
Phase 4 Evaluation: AdaFoB-GAP vs FoB-pretrained on Abd-CT (SABS).

This script replicates the EXACT same data pipeline as FoB's original test.py:
  - SimpleITK for loading (axis order: Z, H, W)
  - Per-VOLUME z-score normalization (not per-slice)
  - 3-channel replication via np.stack(3 * [img], axis=1)
  - SAM receives FoB-normalized tensor converted to uint8 via min-max
  - Mask selection: masks[0] for non-ISIC (matches SAM.py L74-75)
  - Deep-copy of inputs before each model call (FoB mutates lists in-place)
"""

import os
import sys
import glob
import copy
import csv
import random
import argparse
import urllib.request
import zipfile

import numpy as np
import torch
import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wilcoxon

# Add FoB_SAM to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "FoB_SAM")))
from models.FoB import FewShotSeg
from segment_anything import sam_model_registry, SamPredictor


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_dice(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = np.sum(pred * gt)
    total = np.sum(pred) + np.sum(gt)
    if total == 0:
        return 1.0
    return 2.0 * inter / total


def compute_hd95(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return 256.0
    
    from scipy.spatial.distance import cdist
    pred_pts = np.argwhere(pred)
    gt_pts = np.argwhere(gt)
    
    # Compute all pairwise distances (can be heavy, but we only have 256x256 max)
    # For large point clouds, cdist is fast enough in 2D
    dist_matrix = cdist(pred_pts, gt_pts)
    
    # For each point in pred, find the min distance to gt
    min_dist_pred_to_gt = np.min(dist_matrix, axis=1)
    
    # For each point in gt, find the min distance to pred
    min_dist_gt_to_pred = np.min(dist_matrix, axis=0)
    
    # Combine the minimum distances and find the 95th percentile
    all_min_dists = np.concatenate([min_dist_pred_to_gt, min_dist_gt_to_pred])
    hd95 = np.percentile(all_min_dists, 95)
    
    return float(hd95)


# ---------------------------------------------------------------------------
# SAM preprocessing — replicates SAM.py L92-93 exactly
# ---------------------------------------------------------------------------

def sam_preprocess_tensor(tensor_3hw):
    """Convert FoB-normalized (3,H,W) float tensor to (H,W,3) uint8 for SAM.

    Original SAM.py pre_process:
        image = image.permute(1, 2, 0).cpu().numpy()
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    """
    img = tensor_3hw.permute(1, 2, 0).cpu().numpy()
    vmin, vmax = img.min(), img.max()
    denom = vmax - vmin
    if denom < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - vmin) / denom * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Volume loading — uses SimpleITK only, per-volume normalization
# ---------------------------------------------------------------------------

def load_volumes(data_root):
    """Load all image/label volume pairs.

    Returns list of (img_3ch, label) where
        img_3ch : (Z, 3, 256, 256) float32  -- z-score normalised per volume, resized
        label   : (Z, 256, 256) int          -- resized with nearest interpolation
    """
    import re
    import cv2
    volumes = []

    def _resize_volume(vol, size=(256, 256), is_label=False):
        """Resize each slice of a 3D volume to (size)."""
        out = np.zeros((vol.shape[0], size[1], size[0]), dtype=vol.dtype)
        interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
        for z in range(vol.shape[0]):
            out[z] = cv2.resize(vol[z], size, interpolation=interp)
        return out

    # 1) Try FoB preprocessed layout: sabs_CT_normalized/image_*.nii.gz
    fob_dir = os.path.join(data_root, "sabs_CT_normalized")
    fob_imgs = sorted(glob.glob(os.path.join(fob_dir, "image_*.nii.gz"))) if os.path.isdir(fob_dir) else []

    if fob_imgs:
        print(f"Found FoB preprocessed SABS: {len(fob_imgs)} volumes")
        for ip in fob_imgs:
            lp = ip.replace("image_", "label_")
            if not os.path.exists(lp):
                continue
            img = sitk.GetArrayFromImage(sitk.ReadImage(ip))  # (Z, H, W)
            lbl = sitk.GetArrayFromImage(sitk.ReadImage(lp))
            img = _resize_volume(img, (256, 256), is_label=False)
            lbl = _resize_volume(lbl, (256, 256), is_label=True)
            img = (img.astype(np.float64) - img.mean()) / (img.std() + 1e-8)
            img = np.stack(3 * [img], axis=1).astype(np.float32)  # (Z, 3, 256, 256)
            volumes.append((img, lbl))
    else:
        # 2) Raw BTCV / AbdCT layout — pair by patient number
        img_dict, lbl_dict = {}, {}
        for root, _, files in os.walk(data_root):
            for f in sorted(files):
                if not (f.endswith(".nii") or f.endswith(".nii.gz")):
                    continue
                path = os.path.join(root, f)
                fl = f.lower()
                match = re.search(r"(\d+)", f)
                if not match:
                    continue
                pid = match.group(1)
                if "label" in fl or "seg" in fl:
                    lbl_dict[pid] = path
                elif "image" in fl or "img" in fl or "avg" in fl:
                    img_dict[pid] = path

        common = sorted(set(img_dict.keys()) & set(lbl_dict.keys()))
        print(f"Found raw data: {len(img_dict)} images, {len(lbl_dict)} labels, {len(common)} paired")

        for pid in common:
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_dict[pid]))
            lbl = sitk.GetArrayFromImage(sitk.ReadImage(lbl_dict[pid]))
            if img.shape != lbl.shape:
                print(f"  WARNING: patient {pid} shape mismatch img {img.shape} vs lbl {lbl.shape}, skipping")
                continue
            
            img = img.astype(np.float64)
            img = _resize_volume(img, (256, 256), is_label=False)
            lbl = _resize_volume(lbl, (256, 256), is_label=True)
            img = (img - img.mean()) / (img.std() + 1e-8)
            img = np.stack(3 * [img], axis=1).astype(np.float32)
            volumes.append((img, lbl))

    if not volumes:
        raise ValueError(f"No volumes found in {data_root}")

    for i, (img, lbl) in enumerate(volumes):
        ulbl = np.unique(lbl)
        print(f"  Vol {i}: img {img.shape}, lbl {lbl.shape}, labels: {ulbl[:15]}")

    return volumes


# ---------------------------------------------------------------------------
# Episode sampling
# ---------------------------------------------------------------------------

def sample_episode(volumes, organ_cls, n_shot=1):
    """Sample a 1-way 1-shot episode.  Returns dict or None."""
    MIN_ORGAN_PIXELS = 50

    # Find volumes with enough organ slices
    candidates = []
    for vi, (img, lbl) in enumerate(volumes):
        valid = np.where((lbl == organ_cls).sum(axis=(1, 2)) > MIN_ORGAN_PIXELS)[0]
        if len(valid) >= 2:
            candidates.append((vi, valid))

    if not candidates:
        return None

    # Pick support and query volumes (different when possible)
    if len(candidates) >= 2:
        si, qi = random.sample(range(len(candidates)), 2)
    else:
        si = qi = 0

    sv_idx, s_valid = candidates[si]
    qv_idx, q_valid = candidates[qi]

    # FoB TestDataset.get_support_index always picks the middle slice for 1-shot
    # s_valid is an array of valid slice indices.
    mid_idx = int(0.5 * len(s_valid))
    s_slices = [s_valid[mid_idx]]
    q_pool = [s for s in q_valid if sv_idx != qv_idx or s not in s_slices]
    if not q_pool:
        q_pool = list(q_valid)
    q_slice = random.choice(q_pool)

    sv_img, sv_lbl = volumes[sv_idx]
    qv_img, qv_lbl = volumes[qv_idx]

    # Build tensors in the exact format FoB.forward expects:
    #   supp_imgs  : way x shot x [B x 3 x H x W]
    #   supp_masks : way x shot x [B x H x W]
    #   qry_imgs   : N x [B x 3 x H x W]
    #   qry_labels : B x H x W  (tensor)
    supp_imgs_list = []
    supp_masks_list = []
    for s in s_slices:
        supp_imgs_list.append(
            torch.from_numpy(sv_img[s]).unsqueeze(0).float()        # (1, 3, H, W)
        )
        m = (sv_lbl[s] == organ_cls).astype(np.float32)
        supp_masks_list.append(
            torch.from_numpy(m).unsqueeze(0).float()                # (1, H, W)
        )

    qry_tensor = torch.from_numpy(qv_img[q_slice]).unsqueeze(0).float()  # (1, 3, H, W)
    qry_mask_np = (qv_lbl[q_slice] == organ_cls).astype(np.int64)        # (H, W)
    qry_label  = torch.from_numpy(qry_mask_np).unsqueeze(0).long()       # (1, H, W)

    return {
        "support_images":    [supp_imgs_list],   # [[tensor(1,3,H,W)]]
        "support_fg_labels": [supp_masks_list],   # [[tensor(1,H,W)]]
        "query_images":      [qry_tensor],        # [tensor(1,3,H,W)]
        "query_labels":      qry_label,           # tensor(1,H,W)
        "query_mask_np":     qry_mask_np,          # (H,W) numpy
    }


# ---------------------------------------------------------------------------
# Model inference (deep-copies to avoid mutation)
# ---------------------------------------------------------------------------

def run_model(model, sample, use_skeleton):
    """Run FoB/AdaFoB and return (neg_points, pos_points) numpy arrays."""
    supp_imgs  = [[t.clone().cuda() for t in way] for way in sample["support_images"]]
    supp_masks = [[t.clone().cuda() for t in way] for way in sample["support_fg_labels"]]
    qry_imgs   = [t.clone().cuda() for t in sample["query_images"]]
    qry_labels = sample["query_labels"].clone().cuda()

    with torch.no_grad():
        neg, pos = model(supp_imgs, supp_masks, qry_imgs, qry_labels,
                         train=False, use_skeleton=use_skeleton)
    return neg, pos


# ---------------------------------------------------------------------------
# SAM prediction
# ---------------------------------------------------------------------------

def predict_sam(predictor, qry_tensor_1x3xHxW, pos_pts, neg_pts):
    """Feed SAM with points and return predicted mask.

    qry_tensor_1x3xHxW : original un-mutated query tensor (1,3,H,W)
    pos_pts / neg_pts   : numpy arrays from FoB (shape usually (1, N, 2))
    """
    sam_img = sam_preprocess_tensor(qry_tensor_1x3xHxW[0])   # (H, W, 3) uint8
    predictor.set_image(sam_img)

    pos = pos_pts.reshape(-1, 2)
    neg = neg_pts.reshape(-1, 2)
    all_pts  = np.concatenate([pos, neg], axis=0)
    all_lbls = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))], axis=0)

    masks, scores, _ = predictor.predict(
        point_coords=all_pts,
        point_labels=all_lbls,
        multimask_output=True,
    )
    return masks[0]   # best_pred_idx = 0 for non-ISIC (SAM.py L75)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate():
    parser = argparse.ArgumentParser(description="Phase 4: AdaFoB vs FoB")
    parser.add_argument("--ckpt", type=str,
                        default="outputs/checkpoints/adafob_abdct.pth")
    parser.add_argument("--sam_ckpt", type=str,
                        default="/kaggle/working/checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- baseline checkpoint ---
    fob_dir = "/kaggle/working/baseline_fob"
    os.makedirs(fob_dir, exist_ok=True)
    existing = glob.glob(f"{fob_dir}/**/*.pth", recursive=True)
    if existing:
        baseline_ckpt = existing[0]
    else:
        print("Downloading Baseline FoB SABS checkpoint from HuggingFace...")
        zp = os.path.join(fob_dir, "SABS_FSMIS_FoB.zip")
        urllib.request.urlretrieve(
            "https://huggingface.co/PrimeBo1/FoB_SAM/resolve/main/"
            "exps_train_on_SABS_FSMIS_FoB.zip", zp)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(fob_dir)
        baseline_ckpt = glob.glob(f"{fob_dir}/**/*.pth", recursive=True)[0]
    print(f"Baseline FoB checkpoint: {baseline_ckpt}")
    print(f"AdaFoB checkpoint: {args.ckpt}")

    # --- data root ---
    data_root = args.data_root
    if not data_root:
        for pat in ["/kaggle/input/**/*sabs_CT_normalized*",
                    "/kaggle/input/**/*abd*ct*"]:
            hits = glob.glob(pat, recursive=True)
            for h in hits:
                if os.path.isdir(h):
                    data_root = h
                    break
            if data_root:
                break
        if not data_root:
            data_root = "/kaggle/input/datasets/nhatphatnguyen/abd-ct/abd-ct"
    print(f"Data root: {data_root}")

    # --- load volumes ---
    volumes = load_volumes(data_root)

    # --- SAM ---
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).eval().cuda()
    predictor = SamPredictor(sam)

    # --- models ---
    dummy = type("A", (), {})()
    adafob = FewShotSeg(dummy).cuda().eval()
    if os.path.exists(args.ckpt):
        adafob.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    else:
        print(f"WARNING: AdaFoB ckpt not found: {args.ckpt}")

    fob = FewShotSeg(dummy).cuda().eval()
    fob.load_state_dict(torch.load(baseline_ckpt, map_location="cpu"), strict=False)

    # --- run ---
    organ_map = {1: "spleen", 2: "rk", 3: "lk", 6: "liver"}
    organ_list = list(organ_map.keys())
    results = []
    skipped = 0

    for ep in range(args.n_episodes):
        organ_cls = organ_list[ep % len(organ_list)]
        sample = sample_episode(volumes, organ_cls, n_shot=1)
        if sample is None or sample["support_fg_labels"][0][0].max() == 0 \
                or sample["query_labels"].max() == 0:
            skipped += 1
            continue

        try:
            ada_neg, ada_pos = run_model(adafob, sample, use_skeleton=True)
        except Exception as e:
            print(f"  Ep {ep}: AdaFoB err: {e}")
            skipped += 1
            continue

        try:
            base_neg, base_pos = run_model(fob, sample, use_skeleton=False)
        except Exception as e:
            print(f"  Ep {ep}: FoB err: {e}")
            skipped += 1
            continue

        qry_t = sample["query_images"][0]      # (1,3,H,W) — unmutated
        gt    = sample["query_mask_np"]          # (H,W) numpy

        ada_pred  = predict_sam(predictor, qry_t, ada_pos, ada_neg)
        base_pred = predict_sam(predictor, qry_t, base_pos, base_neg)

        results.append({
            "ep": ep, "organ": organ_map[organ_cls],
            "ada_dice":  compute_dice(ada_pred, gt),
            "ada_hd95":  compute_hd95(ada_pred, gt),
            "base_dice": compute_dice(base_pred, gt),
            "base_hd95": compute_hd95(base_pred, gt),
        })

        if (ep + 1) % 10 == 0:
            last10 = results[-min(10, len(results)):]
            print(f"Episode {ep+1}/{args.n_episodes} | "
                  f"Ada Dice(last10): {np.mean([r['ada_dice'] for r in last10]):.4f} | "
                  f"FoB Dice(last10): {np.mean([r['base_dice'] for r in last10]):.4f}")

    # --- CSV ---
    csv_dir = os.path.abspath("results")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "phase4_validation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ep", "organ", "ada_dice", "ada_hd95", "base_dice", "base_hd95"])
        for r in results:
            w.writerow([r["ep"], r["organ"],
                        f"{r['ada_dice']:.6f}", f"{r['ada_hd95']:.4f}",
                        f"{r['base_dice']:.6f}", f"{r['base_hd95']:.4f}"])
    print(f"\nCSV written to: {csv_path}")
    print(f"Completed {len(results)} episodes ({skipped} skipped)")

    # --- summary ---
    if not results:
        print("No results collected!")
        return

    ada_d  = [r["ada_dice"]  for r in results]
    base_d = [r["base_dice"] for r in results]
    ada_h  = [r["ada_hd95"]  for r in results]
    base_h = [r["base_hd95"] for r in results]

    print(f"\n{'='*60}")
    print(f"AdaFoB  Mean Dice: {np.mean(ada_d):.4f}   Mean HD95: {np.mean(ada_h):.4f}")
    print(f"FoB     Mean Dice: {np.mean(base_d):.4f}   Mean HD95: {np.mean(base_h):.4f}")

    print(f"\nPer-organ breakdown:")
    for organ_name in organ_map.values():
        sub = [r for r in results if r["organ"] == organ_name]
        if sub:
            print(f"  {organ_name:>8s}:  Ada={np.mean([r['ada_dice'] for r in sub]):.4f}  "
                  f"FoB={np.mean([r['base_dice'] for r in sub]):.4f}")

    try:
        _, p_d = wilcoxon(ada_d, base_d)
        _, p_h = wilcoxon(ada_h, base_h)
        print(f"\nWilcoxon  Dice p={p_d:.4e}   HD95 p={p_h:.4e}")
    except ValueError as e:
        print(f"Wilcoxon failed: {e}")
    print(f"{'='*60}")


if __name__ == "__main__":
    evaluate()
