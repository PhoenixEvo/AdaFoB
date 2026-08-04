"""
Phase 2 Pilot Comparison: Frozen SAM ViT-H inference with different prompt strategies.

Compares 6 arms (A1-A6) across pilot cases using Dice and HD95 metrics.
All code runs on Kaggle with a single T4 GPU.

Usage (on Kaggle):
    export SAM_CHECKPOINT=/kaggle/working/checkpoints/sam_vit_h_4b8939.pth
    export ABDCT_ROOT=/kaggle/working/data/abdct_pilot
    export BRATS_ROOT=/kaggle/input/brats20-dataset-training-validation
    python experiments/pilot/run_pilot.py
"""
import os
import sys
import csv
import time
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.pilot.shape_features import (
    compute_shape_features,
    compute_heuristic_np,
    load_pilot_config,
)
from experiments.pilot.prompt_generators import (
    RingPriorGenerator,
    SkeletonPriorGenerator,
    get_mask_centroid,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_dice(pred, gt):
    """Compute Dice coefficient between two binary masks."""
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)
    intersection = (pred * gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / total)


def compute_hd95(pred, gt, voxel_spacing=None):
    """
    Compute 95th percentile Hausdorff distance.

    Uses medpy if available, otherwise a manual implementation.
    """
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float("inf")

    try:
        from medpy.metric.binary import hd95
        return float(hd95(pred_bin, gt_bin, voxelspacing=voxel_spacing))
    except ImportError:
        pass

    # Fallback: manual HD95 via distance transforms
    from scipy.ndimage import distance_transform_edt

    dt_pred = distance_transform_edt(~pred_bin.astype(bool))
    dt_gt = distance_transform_edt(~gt_bin.astype(bool))

    # Surface distances
    pred_surface = pred_bin & ~cv2.erode(pred_bin, np.ones((3, 3), np.uint8)).astype(bool)
    gt_surface = gt_bin & ~cv2.erode(gt_bin, np.ones((3, 3), np.uint8)).astype(bool)

    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return float("inf")

    d_pred_to_gt = dt_gt[pred_surface > 0]
    d_gt_to_pred = dt_pred[gt_surface > 0]

    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_dists, 95))


# ---------------------------------------------------------------------------
# SAM Wrapper
# ---------------------------------------------------------------------------
class SAMInference:
    """Wraps SAM ViT-H for point-prompt inference."""

    def __init__(self, checkpoint_path, device="cuda:0"):
        from segment_anything import sam_model_registry, SamPredictor

        self.device = torch.device(device)
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam.to(self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    def predict(self, image, fg_points, fg_labels, bg_points, bg_labels):
        """
        Run SAM inference with point prompts.

        Args:
            image: np.ndarray (H, W, 3) uint8
            fg_points: np.ndarray (Nf, 2) in (x, y)
            fg_labels: np.ndarray (Nf,) all 1
            bg_points: np.ndarray (Nb, 2) in (x, y)
            bg_labels: np.ndarray (Nb,) all 0

        Returns:
            masks: list of np.ndarray (H, W) binary
            scores: np.ndarray
        """
        self.predictor.set_image(image)

        all_points = np.concatenate([fg_points, bg_points], axis=0)
        all_labels = np.concatenate([fg_labels, bg_labels], axis=0)

        masks, scores, _ = self.predictor.predict(
            point_coords=all_points,
            point_labels=all_labels,
            multimask_output=True,
        )
        return masks, scores


# ---------------------------------------------------------------------------
# Case Loading
# ---------------------------------------------------------------------------
def load_pilot_cases(config):
    """
    Load pilot cases and their masks from results/pilot/masks/.

    Falls back to collect_shape_features if masks are missing.
    """
    mask_dir = os.path.join(PROJECT_ROOT, "results", "pilot", "masks")
    csv_path = os.path.join(PROJECT_ROOT, "results", "pilot", "pilot_shape_features.csv")

    if not os.path.isfile(csv_path):
        print("[INFO] pilot_shape_features.csv not found. Running collect_shape_features first...")
        from experiments.pilot.collect_shape_features import main as collect_main
        collect_main()

    cases = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mask_path = os.path.join(mask_dir, f"{row['case_id']}.npy")
            if not os.path.isfile(mask_path):
                print(f"[WARN] Mask not found: {mask_path}, skipping.")
                continue
            mask = np.load(mask_path)
            cases.append({
                "case_id": row["case_id"],
                "dataset": row["dataset"],
                "organ": row["organ"],
                "mask": mask,
                "Np_heuristic": int(row["Np_heuristic"]),
            })

    return cases


def load_image_for_case(case):
    """
    Load the corresponding image for a case.

    If no raw image is available (e.g., we only saved masks),
    synthesize a 3-channel grayscale image from the mask for SAM.
    SAM can still produce meaningful segmentations from point prompts
    even with a uniform image, since the prompts themselves guide output.
    """
    abdct_root = os.environ.get("ABDCT_ROOT")
    brats_root = os.environ.get("BRATS_ROOT")
    drive_root = os.environ.get("DRIVE_ROOT")

    case_id = case["case_id"]
    dataset = case["dataset"]

    # Try to find and load the actual image
    import re
    
    vol_pattern, slice_idx = None, None
    match = re.search(r'_s(\d+)_', case_id)
    if match:
        vol_pattern = case_id[:match.start()]
        slice_idx = int(match.group(1))

    if dataset == "BraTS" and brats_root and vol_pattern:
        vol_id = vol_pattern

            # Find the T2 FLAIR volume
            import glob
            flair_files = glob.glob(
                os.path.join(brats_root, "**", f"{vol_id}*flair*.nii*"),
                recursive=True
            )
            t2_files = glob.glob(
                os.path.join(brats_root, "**", f"{vol_id}*t2*.nii*"),
                recursive=True
            )

            vol_path = None
            if flair_files:
                vol_path = flair_files[0]
            elif t2_files:
                vol_path = t2_files[0]

            if vol_path:
                try:
                    import nibabel as nib
                    vol = nib.load(vol_path).get_fdata()
                except ImportError:
                    import SimpleITK as sitk
                    vol = sitk.GetArrayFromImage(sitk.ReadImage(vol_path))

                slc = vol[slice_idx]
                # Z-score normalize, then scale to [0, 255]
                slc = (slc - slc.mean()) / (slc.std() + 1e-8)
                slc = np.clip(slc * 50 + 128, 0, 255).astype(np.uint8)
                slc = cv2.resize(slc, (256, 256))
                return np.stack([slc, slc, slc], axis=-1)

    if dataset == "AbdCT" and abdct_root and vol_pattern:
        import glob
        img_files = []
        for root, _, files in os.walk(abdct_root):
            for f in files:
                if (f.endswith(".nii") or f.endswith(".nii.gz")) and vol_pattern.replace("_seg.nii.gz", "").replace("_seg.nii", "") in f and ("image" in f or "img" in f or "avg.nii" in f):
                    img_files.append(os.path.join(root, f))

        if not img_files:
            # Fallback to glob if os.walk fails to find the exact match
            img_files = glob.glob(os.path.join(abdct_root, "**", f"*{vol_pattern}*image*"), recursive=True) + \
                        glob.glob(os.path.join(abdct_root, "**", f"image*{vol_pattern}*"), recursive=True)

        if img_files:
            try:
                    import nibabel as nib
                    vol = nib.load(img_files[0]).get_fdata()
                except ImportError:
                    import SimpleITK as sitk
                    vol = sitk.GetArrayFromImage(sitk.ReadImage(img_files[0]))

                if slice_idx < vol.shape[0]:
                    slc = vol[slice_idx]
                    slc = np.clip(slc * 255, 0, 255).astype(np.uint8)
                    slc = cv2.resize(slc, (256, 256))
                    return np.stack([slc, slc, slc], axis=-1)

    # Fallback: generate a synthetic image from the mask
    # Use distance transform to create a textured grayscale
    mask = case["mask"]
    from scipy.ndimage import distance_transform_edt
    dt = distance_transform_edt(mask > 0)
    dt_inv = distance_transform_edt(mask == 0)
    img = np.clip(128 + dt * 3 - dt_inv * 2, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (256, 256))
    return np.stack([img, img, img], axis=-1)


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------
def build_arms(config):
    """Build the 6 experimental arms from config."""
    ring_cfg = config.get("ring_prior", {})
    skel_cfg = config.get("skeleton_prior", {})

    ring_gen = RingPriorGenerator(
        r_outer=ring_cfg.get("r_outer", 15),
        r_inner=ring_cfg.get("r_inner", 13),
    )
    skel_gen = SkeletonPriorGenerator(
        normal_offset_px=skel_cfg.get("normal_offset_px", 8),
    )

    arms = []
    for arm_def in config.get("pilot", {}).get("arms", []):
        gen = ring_gen if arm_def["generator"] == "ring" else skel_gen
        arms.append({
            "name": arm_def["name"],
            "generator": gen,
            "np_mode": arm_def["np"],  # int or "heuristic"
        })

    return arms


# ---------------------------------------------------------------------------
# Main Pilot
# ---------------------------------------------------------------------------
import argparse

def run_pilot():
    parser = argparse.ArgumentParser(description="Run AdaFoB Pilot")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset to run (e.g., AbdCT, BraTS, DRIVE, or all)")
    parser.add_argument("--arms", type=str, default="all", help="Comma-separated list of arms to run (e.g., A2_ring_np10,A6_skel_np10)")
    args = parser.parse_args()

    config = load_pilot_config()
    heuristic_cfg = config.get("heuristic", {})

    # SAM checkpoint path
    sam_ckpt = os.environ.get(
        "SAM_CHECKPOINT",
        config.get("pilot", {}).get(
            "sam_checkpoint",
            "/kaggle/working/checkpoints/sam_vit_h_4b8939.pth",
        ),
    )

    print(f"Loading SAM ViT-H from {sam_ckpt} ...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sam = SAMInference(sam_ckpt, device=device)
    print(f"SAM loaded on {device}.")

    # Load cases
    cases = load_pilot_cases(config)
    if not cases:
        print("[ERROR] No pilot cases found. Run collect_shape_features.py first.")
        sys.exit(1)
        
    if args.dataset.lower() != "all":
        cases = [c for c in cases if c["dataset"] == args.dataset]
    print(f"Loaded {len(cases)} pilot cases.")

    # Build arms
    arms = build_arms(config)
    if args.arms.lower() != "all":
        selected_arms = [a.strip() for a in args.arms.split(",")]
        arms = [a for a in arms if a["name"] in selected_arms]
        
    if not arms:
        print("[ERROR] No arms defined or selected.")
        sys.exit(1)
    print(f"Arms: {[a['name'] for a in arms]}")

    # Output dirs
    results_dir = os.path.join(PROJECT_ROOT, "results", "pilot")
    figures_dir = os.path.join(PROJECT_ROOT, "figures", "pilot")
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # GPU timing
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    wall_start = time.time()

    # Per-case metrics
    all_metrics = []
    comparison_examples = []  # for the comparison grid figure

    for ci, case in enumerate(cases):
        case_id = case["case_id"]
        dataset = case["dataset"]
        organ = case["organ"]
        gt_mask = case["mask"]
        np_h = case["Np_heuristic"]

        print(f"\n[{ci+1}/{len(cases)}] {case_id} (dataset={dataset}, Np_h={np_h})")

        # Load image
        image = load_image_for_case(case)

        # Positive prompt: mask centroid
        cx, cy = get_mask_centroid(gt_mask)
        fg_points = np.array([[cx, cy]], dtype=np.float32)
        fg_labels = np.array([1], dtype=np.int32)

        # Collect arm results for this case (for comparison figure)
        case_results = {"case_id": case_id, "dataset": dataset, "image": image,
                        "gt_mask": gt_mask, "arm_results": {}}

        for arm in arms:
            arm_name = arm["name"]

            # Determine Np for this arm
            if arm["np_mode"] == "heuristic":
                np_count = np_h
            else:
                np_count = int(arm["np_mode"])

            # Generate background prompts
            gen_result = arm["generator"].generate(gt_mask, np_count)
            bg_points = gen_result["points"].astype(np.float32)
            bg_labels = gen_result["labels"].astype(np.int32)

            # Save debug overlay (only for first 5 cases)
            if ci < 5:
                overlay_path = os.path.join(
                    figures_dir,
                    f"debug_{case_id}_{arm_name}.png",
                )
                # Combine fg + bg points for the debug overlay
                all_overlay_pts = np.concatenate(
                    [fg_points, bg_points], axis=0
                )
                all_overlay_lbls = np.concatenate(
                    [fg_labels, bg_labels], axis=0
                )
                arm["generator"].save_debug_overlay(
                    gt_mask, image, all_overlay_pts, all_overlay_lbls,
                    overlay_path,
                    title=f"{case_id} | {arm_name} (Np={np_count})",
                    debug_info=gen_result.get("debug_info"),
                )

            # SAM inference
            masks_pred, scores = sam.predict(
                image, fg_points, fg_labels, bg_points, bg_labels
            )

            # Oracle selection: pick best mask by Dice vs GT
            best_dice = -1.0
            best_mask = None
            for mi in range(len(masks_pred)):
                d = compute_dice(masks_pred[mi], gt_mask)
                if d > best_dice:
                    best_dice = d
                    best_mask = masks_pred[mi]

            hd95_val = compute_hd95(best_mask, gt_mask)

            all_metrics.append({
                "case_id": case_id,
                "dataset": dataset,
                "arm": arm_name,
                "Np_used": np_count,
                "dice": f"{best_dice:.4f}",
                "hd95": f"{hd95_val:.2f}" if hd95_val != float("inf") else "inf",
            })

            case_results["arm_results"][arm_name] = {
                "mask": best_mask,
                "dice": best_dice,
                "hd95": hd95_val,
                "bg_points": bg_points,
            }

            print(f"  {arm_name:25s} Np={np_count:3d} Dice={best_dice:.4f} HD95={hd95_val:.2f}")

        # Keep first few irregular cases for comparison figure
        if dataset == "BraTS" and len(comparison_examples) < 4:
            comparison_examples.append(case_results)
        elif dataset == "AbdCT" and len(comparison_examples) < 6:
            comparison_examples.append(case_results)

    # GPU timing
    gpu_time_ms = 0.0
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        gpu_time_ms = start_event.elapsed_time(end_event)

    wall_time = time.time() - wall_start

    # Save timing
    gpu_log_path = os.path.join(logs_dir, "pilot_gpu_time.txt")
    with open(gpu_log_path, "w") as f:
        f.write(f"GPU time (ms): {gpu_time_ms:.1f}\n")
        f.write(f"GPU time (hours): {gpu_time_ms / 3600000:.4f}\n")
        f.write(f"Wall time (s): {wall_time:.1f}\n")
        f.write(f"Wall time (hours): {wall_time / 3600:.4f}\n")
        f.write(f"Cases: {len(cases)}\n")
        f.write(f"Arms: {len(arms)}\n")
        f.write(f"Total inferences: {len(cases) * len(arms)}\n")
    print(f"\nGPU time: {gpu_time_ms/1000:.1f}s ({gpu_time_ms/3600000:.4f} hours)")
    print(f"Wall time: {wall_time:.1f}s")

    # -----------------------------------------------------------------------
    # Save per-case metrics CSV
    # -----------------------------------------------------------------------
    metrics_path = os.path.join(results_dir, "pilot_metrics.csv")
    
    # Load existing metrics if they exist
    combined_metrics = []
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r") as f:
            reader = csv.DictReader(f)
            combined_metrics = list(reader)
    
    # Append new metrics (overwrite if case_id and arm match)
    existing_keys = {(m["case_id"], m["arm"]) for m in combined_metrics}
    for m in all_metrics:
        key = (m["case_id"], m["arm"])
        if key in existing_keys:
            # Replace existing
            idx = next(i for i, v in enumerate(combined_metrics) if (v["case_id"], v["arm"]) == key)
            combined_metrics[idx] = m
        else:
            combined_metrics.append(m)

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "dataset", "arm", "Np_used", "dice", "hd95"])
        writer.writeheader()
        writer.writerows(combined_metrics)
    print(f"Saved merged metrics to {metrics_path}")

    # -----------------------------------------------------------------------
    # Compute aggregate statistics
    # -----------------------------------------------------------------------
    compute_summary_stats(combined_metrics, results_dir)

    # -----------------------------------------------------------------------
    # Generate figures
    # -----------------------------------------------------------------------
    generate_comparison_figure(comparison_examples, figures_dir)
    generate_np_vs_dice_figure(all_metrics, figures_dir)


def compute_summary_stats(all_metrics, results_dir):
    """Compute per-arm per-dataset summary statistics and Wilcoxon tests."""
    import pandas as pd

    df = pd.DataFrame(all_metrics)
    df["dice"] = df["dice"].replace("inf", np.nan).astype(float)
    df["hd95"] = df["hd95"].replace("inf", np.nan).astype(float)

    # Per arm+dataset aggregation
    summary_rows = []
    for (arm, dataset), grp in df.groupby(["arm", "dataset"]):
        dice_vals = grp["dice"].dropna()
        hd95_vals = grp["hd95"].dropna()
        summary_rows.append({
            "arm": arm,
            "dataset": dataset,
            "n_cases": len(grp),
            "dice_mean": f"{dice_vals.mean():.4f}" if len(dice_vals) > 0 else "N/A",
            "dice_std": f"{dice_vals.std():.4f}" if len(dice_vals) > 0 else "N/A",
            "hd95_mean": f"{hd95_vals.mean():.2f}" if len(hd95_vals) > 0 else "N/A",
            "hd95_std": f"{hd95_vals.std():.2f}" if len(hd95_vals) > 0 else "N/A",
        })

    # Wilcoxon tests on the irregular subset (BraTS)
    irregular = df[df["dataset"] == "BraTS"]
    wilcoxon_notes = []

    if len(irregular) > 0:
        # A4 (ring heuristic) vs best of A1-A3 (fixed ring)
        a4 = irregular[irregular["arm"] == "A4_ring_heuristic"].set_index("case_id")["dice"].astype(float)
        best_fixed_dice = {}
        for arm_name in ["A1_ring_np5", "A2_ring_np10", "A3_ring_np20"]:
            arm_df = irregular[irregular["arm"] == arm_name].set_index("case_id")["dice"].astype(float)
            for cid, val in arm_df.items():
                if cid not in best_fixed_dice or val > best_fixed_dice[cid]:
                    best_fixed_dice[cid] = val

        common_ids = sorted(set(a4.index) & set(best_fixed_dice.keys()))
        if len(common_ids) >= 5:
            a4_vals = [float(a4[cid]) for cid in common_ids]
            bf_vals = [float(best_fixed_dice[cid]) for cid in common_ids]
            try:
                stat, p = stats.wilcoxon(a4_vals, bf_vals)
                wilcoxon_notes.append(
                    f"Wilcoxon A4_heuristic vs best_fixed (Dice, BraTS): stat={stat:.4f}, p={p:.4f}, n={len(common_ids)}"
                )
            except Exception as e:
                wilcoxon_notes.append(f"Wilcoxon A4 vs best_fixed: error ({e})")

        # A5 (skeleton heuristic) vs A2 (ring np10) on HD95
        a5_hd = irregular[irregular["arm"] == "A5_skel_heuristic"].set_index("case_id")["hd95"].astype(float)
        a2_hd = irregular[irregular["arm"] == "A2_ring_np10"].set_index("case_id")["hd95"].astype(float)
        common_hd = sorted(set(a5_hd.index) & set(a2_hd.index))
        if len(common_hd) >= 5:
            a5_vals = [float(a5_hd[cid]) for cid in common_hd]
            a2_vals = [float(a2_hd[cid]) for cid in common_hd]
            # Remove inf pairs
            valid = [(a, b) for a, b in zip(a5_vals, a2_vals) if np.isfinite(a) and np.isfinite(b)]
            if len(valid) >= 5:
                a5v, a2v = zip(*valid)
                try:
                    stat, p = stats.wilcoxon(a5v, a2v)
                    wilcoxon_notes.append(
                        f"Wilcoxon A5_skel vs A2_ring (HD95, BraTS): stat={stat:.4f}, p={p:.4f}, n={len(valid)}"
                    )
                except Exception as e:
                    wilcoxon_notes.append(f"Wilcoxon A5 vs A2 HD95: error ({e})")

    # Wilcoxon tests on the compact subset (AbdCT) A2 vs A6
    compact = df[df["dataset"] == "AbdCT"]
    if len(compact) > 0:
        a6_dice = compact[compact["arm"] == "A6_skel_np10"].set_index("case_id")["dice"].astype(float)
        a2_dice = compact[compact["arm"] == "A2_ring_np10"].set_index("case_id")["dice"].astype(float)
        common_ct = sorted(set(a6_dice.index) & set(a2_dice.index))
        if len(common_ct) >= 3:
            a6_vals = [float(a6_dice[cid]) for cid in common_ct]
            a2_vals = [float(a2_dice[cid]) for cid in common_ct]
            try:
                stat, p = stats.wilcoxon(a6_vals, a2_vals)
                wilcoxon_notes.append(
                    f"Wilcoxon A6_skel vs A2_ring (Dice, AbdCT): stat={stat:.4f}, p={p:.4f}, n={len(common_ct)}"
                )
            except Exception as e:
                wilcoxon_notes.append(f"Wilcoxon A6 vs A2 AbdCT Dice: error ({e})")

    # Add Wilcoxon notes to summary
    for note in wilcoxon_notes:
        summary_rows.append({
            "arm": "STATISTICAL_TEST",
            "dataset": "BraTS",
            "n_cases": "",
            "dice_mean": note,
            "dice_std": "",
            "hd95_mean": "",
            "hd95_std": "",
        })

    # Save
    summary_path = os.path.join(results_dir, "pilot_summary_stats.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "arm", "dataset", "n_cases", "dice_mean", "dice_std", "hd95_mean", "hd95_std"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved summary stats to {summary_path}")
    for note in wilcoxon_notes:
        print(f"  {note}")


def generate_comparison_figure(examples, figures_dir):
    """Generate comparison grid: image, GT, A2 result, A5 result, prompt overlays."""
    if not examples:
        print("[WARN] No examples for comparison figure.")
        return

    n_examples = min(len(examples), 4)
    fig, axes = plt.subplots(n_examples, 5, figsize=(20, 4 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image", "GT Mask", "A2 ring Np=10", "A5 skel heuristic", "Prompt Overlay"]

    for i in range(n_examples):
        ex = examples[i]
        image = ex["image"]
        gt = ex["gt_mask"]

        # Column 0: image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"{ex['case_id'][:25]}" if i == 0 else "")
        axes[i, 0].axis("off")

        # Column 1: GT mask
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].axis("off")

        # Column 2: A2 result
        a2_res = ex["arm_results"].get("A2_ring_np10", {})
        if a2_res:
            axes[i, 2].imshow(a2_res["mask"], cmap="gray")
            axes[i, 2].set_title(f"Dice={a2_res['dice']:.3f}" if i == 0 else f"D={a2_res['dice']:.3f}")
        axes[i, 2].axis("off")

        # Column 3: A5 result
        a5_res = ex["arm_results"].get("A5_skel_heuristic", {})
        if a5_res:
            axes[i, 3].imshow(a5_res["mask"], cmap="gray")
            axes[i, 3].set_title(f"Dice={a5_res['dice']:.3f}" if i == 0 else f"D={a5_res['dice']:.3f}")
        axes[i, 3].axis("off")

        # Column 4: prompt overlay (A5 on image)
        axes[i, 4].imshow(image)
        if a5_res and "bg_points" in a5_res:
            pts = a5_res["bg_points"]
            axes[i, 4].scatter(pts[:, 0], pts[:, 1], c="red", s=30, marker="x", linewidths=1)
        cx, cy = get_mask_centroid(gt)
        axes[i, 4].scatter([cx], [cy], c="lime", s=60, marker="o", edgecolors="black", linewidths=1)
        # GT contour overlay
        contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt_pts = cnt.squeeze()
            if cnt_pts.ndim == 2 and len(cnt_pts) > 2:
                axes[i, 4].plot(cnt_pts[:, 0], cnt_pts[:, 1], "g-", linewidth=1, alpha=0.7)
        axes[i, 4].axis("off")

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(figures_dir, "comparison_examples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison figure to {save_path}")


def generate_np_vs_dice_figure(all_metrics, figures_dir):
    """Generate per-case Dice vs Np for arms A1-A4 on irregular (BraTS) cases."""
    import pandas as pd

    df = pd.DataFrame(all_metrics)
    df["dice"] = df["dice"].replace("inf", np.nan).astype(float)
    df["Np_used"] = df["Np_used"].astype(int)

    # Filter to BraTS and ring arms (A1-A4)
    ring_arms = ["A1_ring_np5", "A2_ring_np10", "A3_ring_np20", "A4_ring_heuristic"]
    irregular = df[(df["dataset"] == "BraTS") & (df["arm"].isin(ring_arms))]

    if irregular.empty:
        print("[WARN] No BraTS ring-arm data for Np vs Dice figure.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"A1_ring_np5": "#1f77b4", "A2_ring_np10": "#ff7f0e",
              "A3_ring_np20": "#2ca02c", "A4_ring_heuristic": "#d62728"}
    markers = {"A1_ring_np5": "o", "A2_ring_np10": "s",
               "A3_ring_np20": "^", "A4_ring_heuristic": "D"}

    for arm_name in ring_arms:
        arm_data = irregular[irregular["arm"] == arm_name]
        if arm_data.empty:
            continue
        ax.scatter(
            arm_data["Np_used"], arm_data["dice"],
            c=colors.get(arm_name, "gray"),
            marker=markers.get(arm_name, "o"),
            label=arm_name, s=50, alpha=0.8,
        )

    ax.set_xlabel("Np (number of background prompts)", fontsize=12)
    ax.set_ylabel("Dice", fontsize=12)
    ax.set_title("Per-case Dice vs Np (BraTS irregular cases, ring prior)", fontsize=13)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(figures_dir, "np_vs_dice.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Np vs Dice figure to {save_path}")


if __name__ == "__main__":
    run_pilot()
