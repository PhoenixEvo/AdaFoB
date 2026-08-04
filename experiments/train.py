import os
import argparse
import yaml
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import sys

# Append third_party path so we can import FoB
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "FoB_SAM")))
from models.FoB import FewShotSeg

class DummyArgs:
    pass

class AbdCTEpisodeDataset(Dataset):
    def __init__(self, abdct_root, iters_per_epoch=100, n_shot=1, n_way=1):
        self.iters_per_epoch = iters_per_epoch
        self.n_shot = n_shot
        self.n_way = n_way
        
        # Collect image and label files
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
        except ImportError:
            import SimpleITK as sitk
            vol = sitk.GetArrayFromImage(sitk.ReadImage(vol_path))
            
        slc = vol[slice_idx]
        if "label" not in vol_path and "seg" not in vol_path:
            # image normalization
            slc = (slc - slc.mean()) / (slc.std() + 1e-8)
            slc = np.clip(slc * 50 + 128, 0, 255).astype(np.uint8)
            slc = cv2.resize(slc, (256, 256))
            slc = np.stack([slc, slc, slc], axis=0) # 3 channels (3, 256, 256)
            slc = slc.astype(np.float32) / 255.0
        else:
            slc = cv2.resize(slc, (256, 256), interpolation=cv2.INTER_NEAREST)
            slc = slc.astype(np.int64)
            
        return slc

    def __getitem__(self, idx):
        # Randomly select a volume
        vol_idx = random.randint(0, len(self.label_files) - 1)
        label_path = self.label_files[vol_idx]
        
        # Find matching image
        vol_pattern = os.path.basename(label_path).replace("_seg.nii.gz", "").replace("_seg.nii", "").replace("label", "")
        img_path = None
        for p in self.image_files:
            if vol_pattern in p:
                img_path = p
                break
        if not img_path:
            img_path = self.image_files[vol_idx] # fallback
            
        # load volume to find valid slices
        try:
            import nibabel as nib
            gt_vol = nib.load(label_path).get_fdata()
        except ImportError:
            import SimpleITK as sitk
            gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
            
        # SABS organs: 6=liver, 1=spleen
        organ_cls = random.choice([1, 6])
        valid_slices = np.where((gt_vol == organ_cls).sum(axis=(1,2)) > 50)[0]
        
        if len(valid_slices) < self.n_shot + 1:
            # Fallback to zero arrays if volume has too few slices
            img_shape = (3, 256, 256)
            mask_shape = (256, 256)
            support_imgs = [[torch.zeros(img_shape) for _ in range(self.n_shot)] for _ in range(self.n_way)]
            support_masks = [[torch.zeros(mask_shape) for _ in range(self.n_shot)] for _ in range(self.n_way)]
            query_imgs = [torch.zeros(img_shape)]
            query_masks = [torch.zeros(mask_shape)]
            return {
                'support_images': support_imgs,
                'support_fg_labels': support_masks,
                'query_images': query_imgs,
                'query_labels': query_masks
            }
            
        # Sample slices
        sampled = random.sample(list(valid_slices), self.n_shot + 1)
        supp_slices = sampled[:-1]
        qry_slice = sampled[-1]
        
        support_imgs = []
        support_masks = []
        for _ in range(self.n_way):
            way_imgs = []
            way_masks = []
            for s in supp_slices:
                img = self._load_slice(img_path, s)
                mask = self._load_slice(label_path, s)
                mask = (mask == organ_cls).astype(np.float32)
                way_imgs.append(torch.from_numpy(img))
                way_masks.append(torch.from_numpy(mask))
            support_imgs.append(way_imgs)
            support_masks.append(way_masks)
            
        qry_img = self._load_slice(img_path, qry_slice)
        qry_mask = self._load_slice(label_path, qry_slice)
        qry_mask = (qry_mask == organ_cls).astype(np.int64)
        
        return {
            'support_images': support_imgs,
            'support_fg_labels': support_masks,
            'query_images': [torch.from_numpy(qry_img)],
            'query_labels': [torch.from_numpy(qry_mask)]
        }

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/adafob_abdct.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    abdct_root = os.environ.get("ABDCT_ROOT")
    if not abdct_root:
        # Fallback to local Kaggle path if environment var is not set
        import glob
        abdct_candidates = glob.glob('/kaggle/input/**/*averaged-training-images*', recursive=True)
        if abdct_candidates:
            abdct_root = os.path.dirname(abdct_candidates[0])
        else:
            raise ValueError("ABDCT_ROOT environment variable must be set!")
            
    print(f"Loading AbdCT dataset from {abdct_root}...")
    dataset = AbdCTEpisodeDataset(abdct_root, iters_per_epoch=config["iters_per_epoch"], n_shot=config["n_shot"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    # Initialize model
    print("Initializing FoB model with GAP Module...")
    fob_args = DummyArgs()
    model = FewShotSeg(fob_args)
    model = model.cuda()
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    print("Starting training...")
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        for i, batch in enumerate(dataloader):
            # Format inputs for FoB
            # way x shot x [B x 3 x H x W]
            supp_imgs = [[shot.float().cuda() for shot in way] for way in batch['support_images']]
            supp_mask = [[shot.float().cuda() for shot in way] for way in batch['support_fg_labels']]
            qry_imgs = [q.float().cuda() for q in batch['query_images']]
            qry_labels = batch['query_labels'][0].long().cuda()
            
            # Forward pass
            optimizer.zero_grad()
            prompt_loss, rac_loss, foreground_loss = model(supp_imgs, supp_mask, qry_imgs, qry_labels, train=True)
            loss = prompt_loss + rac_loss + foreground_loss
            
            # Backward pass
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{config['epochs']}, Iter {i}/{config['iters_per_epoch']}, Loss: {loss.item():.4f}")
                
        print(f"==> Epoch {epoch+1} Average Loss: {epoch_loss/len(dataloader):.4f}")
        
    # Save checkpoint
    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "outputs/checkpoints/adafob_abdct.pth")
    print("Training complete! Model saved to outputs/checkpoints/adafob_abdct.pth")

if __name__ == "__main__":
    train()
