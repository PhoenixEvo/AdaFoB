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
import urllib.request
import zipfile
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wilcoxon
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "FoB_SAM")))
from models.FoB import FewShotSeg
from segment_anything import sam_model_registry, SamPredictor

class DummyArgs:
    pass

class EvalAbdCTEpisodeDataset(Dataset):
    def __init__(self, abdct_root, iters_per_epoch=200, n_shot=1, n_way=1):
        self.iters_per_epoch = iters_per_epoch
        self.n_shot = n_shot
        self.n_way = n_way
        
        self.image_files = []
        self.label_files = []
        for root, _, files in os.walk(abdct_root):
            for f in files:
                if f.endswith(".nii") or f.endswith(".nii.gz"):
                    path = os.path.join(root, f)
                    if "image" in f or "img" in f or "avg.nii" in f:
                        self.image_files.append(path)
                    elif "label" in f or "seg" in f:
                        self.label_files.append(path)
        self.image_files = sorted(self.image_files)
        self.label_files = sorted(self.label_files)
        
        if not self.label_files:
            raise ValueError(f"No label files found in {abdct_root}")
            
    def __len__(self):
        return self.iters_per_epoch
        
    def _load_slice(self, vol_path, slice_idx):
        try:
            import nibabel as nib
            vol = nib.load(vol_path).get_fdata()
            vol = np.transpose(vol, (2, 1, 0)) # Convert (X,Y,Z) to (Z,Y,X)
        except ImportError:
            import SimpleITK as sitk
            vol = sitk.GetArrayFromImage(sitk.ReadImage(vol_path))
            
        slc = vol[slice_idx]
        if "label" not in vol_path and "seg" not in vol_path:
            # Volume-level normalization approximation
            slc_norm = (slc - slc.mean()) / (slc.std() + 1e-8)
            slc_fob = cv2.resize(slc_norm, (256, 256))
            slc_fob = np.stack([slc_fob, slc_fob, slc_fob], axis=0).astype(np.float32)
            
            slc_uint8 = np.clip(slc_norm * 50 + 128, 0, 255).astype(np.uint8)
            slc_sam = cv2.resize(slc_uint8, (256, 256))
            slc_sam = np.stack([slc_sam, slc_sam, slc_sam], axis=0)
            return slc_fob, slc_sam
        else:
            slc = cv2.resize(slc, (256, 256), interpolation=cv2.INTER_NEAREST)
            slc = slc.astype(np.int64)
            return slc, None

    def __getitem__(self, idx):
        vol_idx = random.randint(0, len(self.label_files) - 1)
        label_path = self.label_files[vol_idx]
        
        vol_pattern = os.path.basename(label_path).replace("_seg.nii.gz", "").replace("_seg.nii", "").replace("label", "")
        img_path = None
        for p in self.image_files:
            if vol_pattern in p:
                img_path = p
                break
        if not img_path:
            img_path = self.image_files[vol_idx]
            
        try:
            import nibabel as nib
            gt_vol = nib.load(label_path).get_fdata()
        except ImportError:
            import SimpleITK as sitk
            gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
            
        organ_cls = random.choice([1, 2, 3, 6]) # spleen, rk, lk, liver
        valid_slices = np.where((gt_vol == organ_cls).sum(axis=(1,2)) > 50)[0]
        
        if len(valid_slices) < self.n_shot + 1:
            img_shape = (3, 256, 256)
            mask_shape = (256, 256)
            return {
                'support_images': [[torch.zeros(img_shape) for _ in range(self.n_shot)] for _ in range(self.n_way)],
                'support_fg_labels': [[torch.zeros(mask_shape) for _ in range(self.n_shot)] for _ in range(self.n_way)],
                'query_images': [torch.zeros(img_shape)],
                'query_labels': [torch.zeros(mask_shape)],
                'query_images_sam': [np.zeros((3, 256, 256), dtype=np.uint8)],
                'organ': organ_cls
            }
            
        sampled = random.sample(list(valid_slices), self.n_shot + 1)
        supp_slices = sampled[:-1]
        qry_slice = sampled[-1]
        
        support_imgs = []
        support_masks = []
        for _ in range(self.n_way):
            way_imgs = []
            way_masks = []
            for s in supp_slices:
                img_fob, img_sam = self._load_slice(img_path, s)
                mask, _ = self._load_slice(label_path, s)
                mask = (mask == organ_cls).astype(np.float32)
                way_imgs.append(torch.from_numpy(img_fob))
                way_masks.append(torch.from_numpy(mask))
            support_imgs.append(way_imgs)
            support_masks.append(way_masks)
            
        qry_img_fob, qry_img_sam = self._load_slice(img_path, qry_slice)
        qry_mask, _ = self._load_slice(label_path, qry_slice)
        qry_mask = (qry_mask == organ_cls).astype(np.int64)
        
        return {
            'support_images': support_imgs,
            'support_fg_labels': support_masks,
            'query_images': [torch.from_numpy(qry_img_fob)],
            'query_labels': [torch.from_numpy(qry_mask)],
            'query_images_sam': [qry_img_sam],
            'organ': organ_cls
        }

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
        return 256.0
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
    parser.add_argument("--n_episodes", type=int, default=200)
    args = parser.parse_args()
    
    fob_ckpt_dir = "/kaggle/working/baseline_fob"
    os.makedirs(fob_ckpt_dir, exist_ok=True)
    existing_pths = glob.glob(f"{fob_ckpt_dir}/**/*.pth", recursive=True)
    if existing_pths:
        baseline_ckpt_path = existing_pths[0]
    else:
        print("Downloading Baseline FoB SABS checkpoint from HuggingFace...")
        zip_path = os.path.join(fob_ckpt_dir, "SABS_FSMIS_FoB.zip")
        urllib.request.urlretrieve("https://huggingface.co/PrimeBo1/FoB_SAM/resolve/main/exps_train_on_SABS_FSMIS_FoB.zip", zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(fob_ckpt_dir)
        baseline_ckpt_path = glob.glob(f"{fob_ckpt_dir}/**/*.pth", recursive=True)[0]
        
    print(f"Baseline FoB checkpoint: {baseline_ckpt_path}")
    print(f"AdaFoB checkpoint: {args.ckpt}")
    
    abdct_root = os.environ.get("ABDCT_ROOT")
    if not abdct_root:
        abdct_candidates = glob.glob('/kaggle/input/**/*averaged-testing-images*', recursive=True)
        if abdct_candidates:
            abdct_root = os.path.dirname(abdct_candidates[0])
        else:
            print("Warning: ABDCT_ROOT not found.")
            abdct_root = "."
            
    dataset = EvalAbdCTEpisodeDataset(abdct_root, iters_per_epoch=args.n_episodes, n_shot=1, n_way=1)
    
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).eval().cuda()
    predictor = SamPredictor(sam)
    
    fob_args = DummyArgs()
    
    adafob_model = FewShotSeg(fob_args).cuda().eval()
    if os.path.exists(args.ckpt):
        adafob_model.load_state_dict(torch.load(args.ckpt), strict=False)
        
    fob_model = FewShotSeg(fob_args).cuda().eval()
    if os.path.exists(baseline_ckpt_path):
        fob_model.load_state_dict(torch.load(baseline_ckpt_path), strict=False)
        
    results = []
    organ_map = {1: 'spleen', 2: 'rk', 3: 'lk', 6: 'liver'}
    
    for ep in range(args.n_episodes):
        sample = dataset[ep]
        organ_cls = sample['organ']
        organ_name = organ_map.get(organ_cls, "unknown")
        
        supp_imgs = [[img.unsqueeze(0).cuda() for img in way_imgs] for way_imgs in sample['support_images']]
        supp_masks = [[mask.unsqueeze(0).cuda() for mask in way_masks] for way_masks in sample['support_fg_labels']]
        qry_imgs = [img.unsqueeze(0).cuda() for img in sample['query_images']]
        qry_labels = torch.cat([label.unsqueeze(0).long().cuda() for label in sample['query_labels']], dim=0)
        
        if supp_masks[0][0].max() == 0 or qry_labels.max() == 0:
            continue
            
        with torch.no_grad():
            ada_neg, ada_pos = adafob_model(supp_imgs, supp_masks, qry_imgs, qry_labels, train=False, use_skeleton=True)
            base_neg, base_pos = fob_model(supp_imgs, supp_masks, qry_imgs, qry_labels, train=False, use_skeleton=False)
            
        qry_img_sam_np = sample['query_images_sam'][0].transpose(1, 2, 0) # uint8 array
        gt_mask = sample['query_labels'][0].numpy()
        predictor.set_image(qry_img_sam_np)
        
        ada_pos = ada_pos.reshape(-1, 2)
        ada_neg = ada_neg.reshape(-1, 2)
        ada_pts = np.concatenate([ada_pos, ada_neg], axis=0)
        ada_lbls = np.concatenate([np.ones(len(ada_pos)), np.zeros(len(ada_neg))], axis=0)
        ada_masks, ada_scores, _ = predictor.predict(point_coords=ada_pts, point_labels=ada_lbls, multimask_output=True)
        ada_pred = ada_masks[np.argmax(ada_scores)]
        
        ada_dice = compute_dice(ada_pred, gt_mask)
        ada_hd95 = compute_hd95(ada_pred, gt_mask)
        
        base_pos = base_pos.reshape(-1, 2)
        base_neg = base_neg.reshape(-1, 2)
        base_pts = np.concatenate([base_pos, base_neg], axis=0)
        base_lbls = np.concatenate([np.ones(len(base_pos)), np.zeros(len(base_neg))], axis=0)
        base_masks, base_scores, _ = predictor.predict(point_coords=base_pts, point_labels=base_lbls, multimask_output=True)
        base_pred = base_masks[np.argmax(base_scores)]
        
        base_dice = compute_dice(base_pred, gt_mask)
        base_hd95 = compute_hd95(base_pred, gt_mask)
        
        results.append({
            'ep': ep, 'organ': organ_name,
            'ada_dice': ada_dice, 'ada_hd95': ada_hd95,
            'base_dice': base_dice, 'base_hd95': base_hd95
        })
        
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}/{args.n_episodes}")
            
    os.makedirs("results", exist_ok=True)
    with open("results/phase4_validation.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ep", "organ", "ada_dice", "ada_hd95", "base_dice", "base_hd95"])
        for r in results:
            writer.writerow([r['ep'], r['organ'], r['ada_dice'], r['ada_hd95'], r['base_dice'], r['base_hd95']])
            
    ada_dices = [r['ada_dice'] for r in results]
    base_dices = [r['base_dice'] for r in results]
    ada_hd95s = [r['ada_hd95'] for r in results]
    base_hd95s = [r['base_hd95'] for r in results]
    
    print(f"AdaFoB Mean Dice: {np.mean(ada_dices):.4f}, Mean HD95: {np.mean(ada_hd95s):.4f}")
    print(f"FoB Mean Dice: {np.mean(base_dices):.4f}, Mean HD95: {np.mean(base_hd95s):.4f}")
    
    try:
        _, p_dice = wilcoxon(ada_dices, base_dices)
        _, p_hd95 = wilcoxon(ada_hd95s, base_hd95s)
        print(f"Wilcoxon p-value Dice: {p_dice:.4e}")
        print(f"Wilcoxon p-value HD95: {p_hd95:.4e}")
    except ValueError:
        pass

if __name__ == "__main__":
    evaluate()
