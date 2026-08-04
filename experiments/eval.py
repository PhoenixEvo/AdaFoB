import os
import yaml
import glob
import random
import torch
import numpy as np
import cv2
import csv
import sys
import argparse
from scipy.spatial.distance import directed_hausdorff

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "FoB_SAM")))
from models.FoB import FewShotSeg
from segment_anything import sam_model_registry, SamPredictor

class DummyArgs:
    pass

def load_slice(vol_path, slice_idx):
    try:
        import nibabel as nib
        vol = nib.load(vol_path).get_fdata()
    except ImportError:
        import SimpleITK as sitk
        vol = sitk.GetArrayFromImage(sitk.ReadImage(vol_path))
        
    slc = vol[slice_idx]
    if "label" not in vol_path and "seg" not in vol_path:
        slc = (slc - slc.mean()) / (slc.std() + 1e-8)
        slc = np.clip(slc * 50 + 128, 0, 255).astype(np.uint8)
        slc = cv2.resize(slc, (256, 256))
        slc = np.stack([slc, slc, slc], axis=0) 
        slc = slc.astype(np.float32) / 255.0
    else:
        slc = cv2.resize(slc, (256, 256), interpolation=cv2.INTER_NEAREST)
        slc = slc.astype(np.int64)
        
    return slc

def compute_dice(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:
        return 1.0
    return 2.0 * intersection / union

def compute_hd95(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return 256.0 # Max distance for 256x256 image
        
    pred_pts = np.argwhere(pred > 0)
    gt_pts = np.argwhere(gt > 0)
    
    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return max(d1, d2)

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/adafob_abdct.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/adafob_abdct.pth")
    parser.add_argument("--sam_ckpt", type=str, default="/kaggle/working/checkpoints/sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    abdct_root = os.environ.get("ABDCT_ROOT")
    if not abdct_root:
        abdct_candidates = glob.glob('/kaggle/input/**/*averaged-testing-images*', recursive=True)
        if abdct_candidates:
            abdct_root = os.path.dirname(abdct_candidates[0])
        else:
            print("Warning: ABDCT_ROOT not found, evaluation will fail if run.")
            abdct_root = "."
            
    # Load SAM
    if os.path.exists(args.sam_ckpt):
        sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).eval().cuda()
        predictor = SamPredictor(sam)
    else:
        print(f"Warning: SAM checkpoint not found at {args.sam_ckpt}. Will skip SAM forward pass.")
        predictor = None

    # Load FoB
    fob_args = DummyArgs()
    model = FewShotSeg(fob_args)
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt))
    model = model.cuda()
    model.eval()
    
    # In a real evaluation, we would find test cases, construct 1-shot pairs, run the model,
    # collect neg_point and pos_point, pass them to SAM predictor, and compute metrics.
    # We will log the results to results/phase4_validation.csv.
    os.makedirs("results", exist_ok=True)
    with open("results/phase4_validation.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "organ", "dice", "hd95"])
    print("Evaluation skeleton complete. results/phase4_validation.csv initialized.")

if __name__ == "__main__":
    evaluate()
