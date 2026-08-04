"""
Collect shape features for the Phase 2 pilot experiment.

Iterates over pilot cases from ABDCT_ROOT and BRATS_ROOT (env vars),
computes shape features, and writes results/pilot/pilot_shape_features.csv.

Usage (on Kaggle):
    export ABDCT_ROOT=/kaggle/working/data/abdct_pilot
    export BRATS_ROOT=/kaggle/input/brats20-dataset-training-validation
    python experiments/pilot/collect_shape_features.py
"""
import os
import sys
import glob
import csv
import numpy as np

try:
    import nibabel as nib
except ImportError:
    import SimpleITK as sitk
    nib = None

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python is required")

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.pilot.shape_features import (
    compute_shape_features,
    compute_heuristic_np,
    load_pilot_config,
)


def load_nifti_volume(path):
    """Load a NIfTI volume, return numpy array."""
    if nib is not None:
        img = nib.load(path)
        return img.get_fdata()
    else:
        img = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(img)


def collect_abdct_cases(abdct_root, num_cases=10):
    """
    Collect Abd-CT (SABS/BTCV) pilot cases.

    Expects preprocessed slices as .nii.gz volumes in abdct_root with
    structure: image_*.nii.gz and label_*.nii.gz (SSL-ALPNet format),
    OR raw NIfTI volumes with multi-organ labels.

    Returns list of dicts with keys: case_id, dataset, organ, mask (256x256).
    """
    cases = []

    # Try SSL-ALPNet preprocessed format first
    image_files = sorted(glob.glob(os.path.join(abdct_root, "**", "image_*.nii.gz"), recursive=True))
    label_files = sorted(glob.glob(os.path.join(abdct_root, "**", "label_*.nii.gz"), recursive=True))

    if not image_files:
        # Try BTCV raw format: img####.nii.gz + label####.nii.gz
        image_files = sorted(glob.glob(os.path.join(abdct_root, "**", "img*.nii.gz"), recursive=True))
        label_files = sorted(glob.glob(os.path.join(abdct_root, "**", "label*.nii.gz"), recursive=True))

    if not label_files:
        # FUSE / Kaggle compatibility: use os.walk instead of ** glob
        image_files = []
        label_files = []
        for root, _, files in os.walk(abdct_root):
            for f in files:
                if f.endswith(".nii") or f.endswith(".nii.gz"):
                    path = os.path.join(root, f)
                    if "image" in f or "img" in f or "avg.nii" in f:
                        image_files.append(path)
                    elif "label" in f or "seg" in f:
                        label_files.append(path)
        image_files = sorted(image_files)
        label_files = sorted(label_files)

    if not label_files:
        print(f"[WARN] No label files found in {abdct_root}. Skipping Abd-CT.")
        # Debug: list all .nii files found in the root
        all_nii = []
        for root, _, files in os.walk(abdct_root):
            all_nii.extend([f for f in files if ".nii" in f])
        print(f"       Found .nii files overall: {len(all_nii)} (First 10: {all_nii[:10]})")
        return cases

    # SABS/BTCV organ labels: 6=liver, 1=spleen (commonly used in few-shot literature)
    # Some versions use: 1=spleen, 2=right kidney, 3=left kidney, 6=liver
    organ_map = {6: "liver", 1: "spleen"}

    collected = 0
    for label_path in label_files:
        if collected >= num_cases:
            break

        vol = load_nifti_volume(label_path)
        vol_id = os.path.basename(label_path).replace(".nii.gz", "")

        for organ_label, organ_name in organ_map.items():
            if collected >= num_cases:
                break

            # Find slices with this organ
            if vol.ndim == 3:
                for s in range(vol.shape[0]):
                    if collected >= num_cases:
                        break
                    slc = vol[s]
                    organ_mask = (slc == organ_label).astype(np.uint8)
                    if organ_mask.sum() < 200:  # skip tiny masks
                        continue

                    # Resize to 256x256
                    organ_mask = cv2.resize(organ_mask, (256, 256),
                                           interpolation=cv2.INTER_NEAREST)

                    cases.append({
                        "case_id": f"{vol_id}_s{s:03d}_{organ_name}",
                        "dataset": "AbdCT",
                        "organ": organ_name,
                        "mask": organ_mask,
                    })
                    collected += 1
                    break  # one slice per organ per volume

    return cases


def collect_brats_cases(brats_root, num_cases=10, tumor_core_labels=None):
    """
    Collect BraTS2020 pilot cases.

    Expects BraTS2020 NIfTI structure:
      brats_root/BraTS20_Training_*/BraTS20_Training_*_seg.nii.gz

    Tumor core = union of labels {1, 4} (necrotic + enhancing).

    Returns list of dicts with keys: case_id, dataset, organ, mask (256x256).
    """
    if tumor_core_labels is None:
        tumor_core_labels = [1, 4]

    cases = []

    # Find segmentation files
    seg_files = sorted(glob.glob(
        os.path.join(brats_root, "**", "*_seg.nii*"), recursive=True
    ))

    if not seg_files:
        # Try alternate naming
        seg_files = sorted(glob.glob(
            os.path.join(brats_root, "**", "*seg*.nii*"), recursive=True
        ))

    if not seg_files:
        print(f"[WARN] No segmentation files found in {brats_root}. Skipping BraTS.")
        return cases

    collected = 0
    for seg_path in seg_files:
        if collected >= num_cases:
            break

        vol = load_nifti_volume(seg_path)
        vol_id = os.path.basename(os.path.dirname(seg_path))

        # Create tumor core mask (union of specified labels)
        tc_vol = np.zeros_like(vol, dtype=np.uint8)
        for lbl in tumor_core_labels:
            tc_vol = tc_vol | (vol == lbl).astype(np.uint8)

        # Find the axial slice with the largest tumor core area
        if vol.ndim == 3:
            areas = [tc_vol[s].sum() for s in range(tc_vol.shape[0])]
            best_slice = int(np.argmax(areas))

            if areas[best_slice] < 100:  # skip if tumor is too small
                continue

            mask = tc_vol[best_slice]
            # Resize to 256x256
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

            cases.append({
                "case_id": f"{vol_id}_s{best_slice:03d}_tumor",
                "dataset": "BraTS",
                "organ": "tumor_core",
                "mask": mask,
            })
            collected += 1

    return cases


def collect_drive_cases(drive_root, num_cases=5):
    """
    Collect DRIVE retinal vessel cases (optional, cross-domain).

    Expects DRIVE structure with manual annotations.

    Returns list of dicts with keys: case_id, dataset, organ, mask (256x256).
    """
    cases = []

    if drive_root is None:
        return cases

    # Find manual segmentation masks
    mask_files = sorted(glob.glob(
        os.path.join(drive_root, "**", "*_manual1.*"), recursive=True
    ))

    if not mask_files:
        # Try alternate structure
        mask_files = sorted(glob.glob(
            os.path.join(drive_root, "**", "*.gif"), recursive=True
        ))
        mask_files = [f for f in mask_files if "manual" in f.lower() or "1st" in f.lower()]

    if not mask_files:
        print(f"[WARN] No mask files found in {drive_root}. Skipping DRIVE.")
        return cases

    collected = 0
    for mask_path in mask_files:
        if collected >= num_cases:
            break

        from PIL import Image
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.uint8)

        # Resize shorter dimension to 256, then center-crop to 256x256
        h, w = mask.shape
        if h < w:
            new_h = 256
            new_w = int(w * 256 / h)
        else:
            new_w = 256
            new_h = int(h * 256 / w)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Center crop
        ch, cw = mask.shape
        y0 = (ch - 256) // 2
        x0 = (cw - 256) // 2
        mask = mask[y0:y0+256, x0:x0+256]

        case_id = os.path.splitext(os.path.basename(mask_path))[0]
        cases.append({
            "case_id": f"drive_{case_id}",
            "dataset": "DRIVE",
            "organ": "vessel",
            "mask": mask,
        })
        collected += 1

    return cases


def main():
    config = load_pilot_config()
    heuristic_cfg = config.get("heuristic", {})
    pilot_cfg = config.get("pilot", {})
    brats_cfg = config.get("brats", {})

    abdct_root = os.environ.get("ABDCT_ROOT")
    brats_root = os.environ.get("BRATS_ROOT")
    drive_root = os.environ.get("DRIVE_ROOT")

    num_abdct = pilot_cfg.get("num_abdct_cases", 10)
    num_brats = pilot_cfg.get("num_brats_cases", 10)
    num_drive = pilot_cfg.get("num_drive_cases", 5)
    tumor_core_labels = brats_cfg.get("tumor_core_labels", [1, 4])

    all_cases = []

    # Collect from each dataset
    if abdct_root and os.path.isdir(abdct_root):
        print(f"Collecting Abd-CT cases from {abdct_root} ...")
        all_cases.extend(collect_abdct_cases(abdct_root, num_abdct))
        print(f"  Found {sum(1 for c in all_cases if c['dataset']=='AbdCT')} Abd-CT cases.")
    else:
        print(f"[SKIP] ABDCT_ROOT not set or not found: {abdct_root}")

    if brats_root and os.path.isdir(brats_root):
        print(f"Collecting BraTS cases from {brats_root} ...")
        all_cases.extend(collect_brats_cases(brats_root, num_brats, tumor_core_labels))
        print(f"  Found {sum(1 for c in all_cases if c['dataset']=='BraTS')} BraTS cases.")
    else:
        print(f"[SKIP] BRATS_ROOT not set or not found: {brats_root}")

    if drive_root and os.path.isdir(drive_root):
        print(f"Collecting DRIVE cases from {drive_root} ...")
        all_cases.extend(collect_drive_cases(drive_root, num_drive))
        print(f"  Found {sum(1 for c in all_cases if c['dataset']=='DRIVE')} DRIVE cases.")
    else:
        print(f"[SKIP] DRIVE_ROOT not set or not found: {drive_root}")

    if not all_cases:
        print("[ERROR] No cases collected. Check dataset paths.")
        sys.exit(1)

    print(f"\nTotal cases collected: {len(all_cases)}")

    # Compute features and save CSV
    out_dir = os.path.join(PROJECT_ROOT, "results", "pilot")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pilot_shape_features.csv")

    fieldnames = [
        "case_id", "dataset", "organ", "area", "perimeter",
        "compactness", "skeleton_length", "num_branches", "Np_heuristic",
    ]

    # Also save masks as .npy for run_pilot.py to load
    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case in all_cases:
            mask = case["mask"]
            features = compute_shape_features(mask)
            np_h = compute_heuristic_np(features, heuristic_cfg)

            row = {
                "case_id": case["case_id"],
                "dataset": case["dataset"],
                "organ": case["organ"],
                "area": features["area"],
                "perimeter": f"{features['perimeter']:.2f}",
                "compactness": f"{features['compactness']:.4f}",
                "skeleton_length": features["skeleton_length"],
                "num_branches": features["num_branches"],
                "Np_heuristic": np_h,
            }
            writer.writerow(row)

            # Save mask for run_pilot.py
            np.save(os.path.join(mask_dir, f"{case['case_id']}.npy"), mask)

            print(f"  {case['case_id']:40s} | area={features['area']:6d} | "
                  f"compact={features['compactness']:.3f} | "
                  f"skel_len={features['skeleton_length']:4d} | "
                  f"branches={features['num_branches']:3d} | Np={np_h:2d}")

    print(f"\nSaved shape features to {csv_path}")
    print(f"Saved masks to {mask_dir}")


if __name__ == "__main__":
    main()
