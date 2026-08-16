"""
SAM LoRA Fine-Tuning Script for SABS CT
========================================
Trains LoRA adapters on SAM ViT-B's image encoder using supervised
segmentation on the SABS (BTCV) abdominal CT dataset.

Per-fold cross-validation: trains on ~24 volumes, validates on ~6.
Uses simulated point prompts from ground truth masks.

Designed for Kaggle T4 GPU (16GB VRAM).

Usage:
    python experiments/train_lora.py --fold 0 --gpu 0 --epochs 100
    python experiments/train_lora.py --fold 1 --gpu 1 --epochs 100
"""

import os
import sys
import glob
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import SimpleITK as sitk

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "third_party", "segment-anything"))

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from experiments.lora_sam import LoRA_Sam


# ── Auto-detect paths (Kaggle-compatible) ─────────────────────────────────────
def find_path(target_name, is_file=False):
    """Search for target in /kaggle/working/ and /kaggle/input/."""
    working_path = os.path.join("/kaggle/working", target_name)
    if os.path.exists(working_path):
        return working_path
    for pat in [f"/kaggle/input/**/{target_name}"]:
        candidates = glob.glob(pat, recursive=True)
        if candidates:
            return candidates[0]
    # Local fallback
    local = os.path.join(REPO_DIR, target_name)
    if os.path.exists(local):
        return local
    return working_path


# ── Dataset ───────────────────────────────────────────────────────────────────
def get_folds():
    """SABS 5-fold cross-validation splits (30 volumes)."""
    FOLD = {}
    FOLD[0] = set(range(0, 7))
    FOLD[1] = set(range(6, 13))
    FOLD[2] = set(range(12, 19))
    FOLD[3] = set(range(18, 25))
    FOLD[4] = set(range(24, 30))
    FOLD[4].update([0])
    return FOLD


# All 13 SABS organ labels for maximum training diversity
ALL_ORGANS = {
    1: "SPLEEN", 2: "RK", 3: "LK", 4: "GALLBLADDER", 5: "ESOPHAGUS",
    6: "LIVER", 7: "STOMACH", 8: "AORTA", 9: "IVC", 10: "PS_VEIN",
    11: "PANCREAS", 12: "AG_R", 13: "AG_L",
}


class SABSLoRADataset(Dataset):
    """Slice-level dataset for SAM LoRA training on SABS CT.

    Each sample returns:
        - image: (3, 1024, 1024) float tensor (SAM-preprocessed)
        - mask: (1, 1024, 1024) binary float tensor (GT mask)
        - point_coords: (N, 2) float tensor (simulated prompt coords in 1024 space)
        - point_labels: (N,) int tensor (1=foreground, 0=background)
        - original_size: (H, W) tuple
    """

    def __init__(
        self,
        data_dir: str,
        fold: int,
        is_train: bool = True,
        min_fg_pixels: int = 50,
        max_points: int = 20,
        transform_prob: float = 0.5,
    ):
        super().__init__()
        self.fold = fold
        self.is_train = is_train
        self.min_fg_pixels = min_fg_pixels
        self.max_points = max_points
        self.transform_prob = transform_prob

        # SAM preprocessing
        self.sam_transform = ResizeLongestSide(1024)
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

        # Load volumes
        FOLDS = get_folds()
        test_indices = FOLDS[fold]
        norm_dir = os.path.join(data_dir, "sabs_CT_normalized")

        all_image_paths = sorted(
            glob.glob(os.path.join(norm_dir, "image_*.nii*")),
            key=lambda x: int(x.split("image_")[-1].split(".nii")[0]),
        )
        all_label_paths = sorted(
            glob.glob(os.path.join(norm_dir, "label_*.nii*")),
            key=lambda x: int(x.split("label_")[-1].split(".nii")[0]),
        )

        if is_train:
            # Training: use volumes NOT in the test fold
            self.image_paths = [p for i, p in enumerate(all_image_paths) if i not in test_indices]
            self.label_paths = [p for i, p in enumerate(all_label_paths) if i not in test_indices]
        else:
            # Validation: use volumes IN the test fold
            self.image_paths = [p for i, p in enumerate(all_image_paths) if i in test_indices]
            self.label_paths = [p for i, p in enumerate(all_label_paths) if i in test_indices]

        # Pre-extract all valid (slice, organ) pairs
        self.samples = []  # List of (vol_idx, slice_idx, organ_id)
        print(f"[Dataset] Loading {'train' if is_train else 'val'} set for fold {fold}...")
        for vol_idx, (img_path, lbl_path) in enumerate(zip(self.image_paths, self.label_paths)):
            lbl_vol = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path))
            # SABS label remapping
            lbl_vol_remap = lbl_vol.copy()
            lbl_vol_remap[lbl_vol == 200] = 1
            lbl_vol_remap[lbl_vol == 500] = 2
            lbl_vol_remap[lbl_vol == 600] = 3

            for z in range(lbl_vol_remap.shape[0]):
                for organ_id in ALL_ORGANS:
                    fg_pixels = np.sum(lbl_vol_remap[z] == organ_id)
                    if fg_pixels >= min_fg_pixels:
                        self.samples.append((vol_idx, z, organ_id))

        print(f"[Dataset] Found {len(self.samples)} valid (slice, organ) pairs "
              f"from {len(self.image_paths)} volumes")

        # Cache loaded volumes (memory efficient for ~24 volumes)
        self._vol_cache = {}

    def _load_volume(self, vol_idx):
        """Lazy-load and cache a volume pair."""
        if vol_idx not in self._vol_cache:
            img = sitk.GetArrayFromImage(sitk.ReadImage(self.image_paths[vol_idx]))
            lbl = sitk.GetArrayFromImage(sitk.ReadImage(self.label_paths[vol_idx]))
            # SABS label remapping
            lbl[lbl == 200] = 1
            lbl[lbl == 500] = 2
            lbl[lbl == 600] = 3
            self._vol_cache[vol_idx] = (img, lbl)
        return self._vol_cache[vol_idx]

    def __len__(self):
        return len(self.samples)

    def _simulate_prompts(self, binary_mask: np.ndarray, original_size):
        """Generate simulated point prompts from a binary GT mask.

        During training, randomly varies the number of points (1-max_points)
        to teach SAM to work with any prompt budget.
        """
        H, W = binary_mask.shape
        fg_coords = np.argwhere(binary_mask > 0)  # (N, 2) in (row, col) format
        bg_coords = np.argwhere(binary_mask == 0)

        if self.is_train:
            # Random number of positive points: 1 to min(10, available)
            n_pos = random.randint(1, min(10, len(fg_coords)))
            # Random number of negative points: 0 to min(10, available)
            n_neg = random.randint(0, min(10, len(bg_coords)))
        else:
            # Fixed for validation reproducibility
            n_pos = min(5, len(fg_coords))
            n_neg = min(5, len(bg_coords))

        # Sample foreground points
        if len(fg_coords) > 0:
            fg_idx = np.random.choice(len(fg_coords), n_pos, replace=n_pos > len(fg_coords))
            pos_points = fg_coords[fg_idx]  # (n_pos, 2) in (row, col)
        else:
            pos_points = np.zeros((0, 2), dtype=np.float32)

        # Sample background points (preferring near-boundary for harder training)
        if n_neg > 0 and len(bg_coords) > 0:
            if self.is_train and random.random() < 0.7:
                # 70% chance: sample near the boundary for harder training
                from scipy.ndimage import distance_transform_edt
                dist = distance_transform_edt(binary_mask == 0)
                # Prefer points close to foreground (distance 5-30 pixels)
                near_boundary = (dist > 2) & (dist < 30)
                near_coords = np.argwhere(near_boundary)
                if len(near_coords) >= n_neg:
                    bg_idx = np.random.choice(len(near_coords), n_neg, replace=False)
                    neg_points = near_coords[bg_idx]
                else:
                    bg_idx = np.random.choice(len(bg_coords), n_neg, replace=n_neg > len(bg_coords))
                    neg_points = bg_coords[bg_idx]
            else:
                bg_idx = np.random.choice(len(bg_coords), n_neg, replace=n_neg > len(bg_coords))
                neg_points = bg_coords[bg_idx]
        else:
            neg_points = np.zeros((0, 2), dtype=np.float32)

        # Convert from (row, col) to (x, y) = (col, row) for SAM
        all_points = []
        all_labels = []
        if len(pos_points) > 0:
            all_points.append(pos_points[:, ::-1].copy().astype(np.float64))  # (row,col) → (x,y)
            all_labels.extend([1] * len(pos_points))
        if len(neg_points) > 0:
            all_points.append(neg_points[:, ::-1].copy().astype(np.float64))
            all_labels.extend([0] * len(neg_points))

        if len(all_points) == 0:
            # Fallback: single center point
            cy, cx = H // 2, W // 2
            all_points = [np.array([[cx, cy]], dtype=np.float64)]
            all_labels = [1]

        point_coords = np.concatenate(all_points, axis=0)  # (N, 2) in (x, y)
        point_labels = np.array(all_labels, dtype=np.int64)

        # Transform coordinates to 1024x1024 space
        point_coords_1024 = self.sam_transform.apply_coords(point_coords, original_size)

        return (
            torch.as_tensor(point_coords_1024, dtype=torch.float32),
            torch.as_tensor(point_labels, dtype=torch.int64),
        )

    def _apply_augmentation(self, img_slice, mask_slice):
        """Apply CT-appropriate augmentations."""
        if not self.is_train or random.random() > self.transform_prob:
            return img_slice, mask_slice

        # Gamma transform (50% chance)
        if random.random() > 0.5:
            gamma = random.uniform(0.7, 1.5)
            cmin = img_slice.min()
            irange = img_slice.max() - cmin + 1e-5
            img_slice = irange * np.power((img_slice - cmin + 1e-5) / irange, gamma) + cmin

        # Horizontal flip (50% chance)
        if random.random() > 0.5:
            img_slice = np.flip(img_slice, axis=1).copy()
            mask_slice = np.flip(mask_slice, axis=1).copy()

        # Gaussian noise (30% chance)
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.02 * (img_slice.max() - img_slice.min()),
                                      img_slice.shape)
            img_slice = img_slice + noise

        return img_slice, mask_slice

    def __getitem__(self, idx):
        vol_idx, z, organ_id = self.samples[idx]
        img_vol, lbl_vol = self._load_volume(vol_idx)

        # Extract slice
        img_slice = img_vol[z].astype(np.float64)
        mask_slice = (lbl_vol[z] == organ_id).astype(np.uint8)

        # Augmentation
        img_slice, mask_slice = self._apply_augmentation(img_slice, mask_slice)

        # Convert to SAM uint8 RGB format
        img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        img_rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)  # (H, W, 3)

        original_size = img_rgb.shape[:2]  # (H, W)

        # Resize image to 1024
        img_resized = self.sam_transform.apply_image(img_rgb)  # (1024, 1024, 3)

        # Resize mask to 1024
        import cv2
        mask_1024 = cv2.resize(
            mask_slice.astype(np.float32),
            (1024, 1024),
            interpolation=cv2.INTER_NEAREST,
        )

        # Simulate prompts from original-resolution mask
        point_coords, point_labels = self._simulate_prompts(mask_slice, original_size)

        # Convert image to tensor and normalize (SAM preprocessing)
        img_tensor = torch.as_tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)  # (3, 1024, 1024)
        img_tensor = (img_tensor - self.pixel_mean) / self.pixel_std

        # Mask tensor
        mask_tensor = torch.as_tensor(mask_1024, dtype=torch.float32).unsqueeze(0)  # (1, 1024, 1024)

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'original_size': original_size,
        }


def collate_fn(batch):
    """Custom collate for variable-length point prompts."""
    return batch  # Return list of dicts (not stacked)


# ── Loss Functions ────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) sigmoid-activated predictions
            target: (B, 1, H, W) binary ground truth
        """
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class CompoundLoss(nn.Module):
    """Combined Dice + BCE loss (SAMed-style)."""

    def __init__(self, dice_weight: float = 0.8, bce_weight: float = 0.2):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw mask logits (before sigmoid)
            target: (B, 1, H, W) binary ground truth
        """
        pred_sigmoid = torch.sigmoid(logits)
        dice = self.dice_loss(pred_sigmoid, target)
        bce = self.bce_loss(logits, target)
        return self.dice_weight * dice + self.bce_weight * bce


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, grad_accum=4):
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    # Keep image encoder in eval mode (BatchNorm, Dropout behavior)
    model.sam.image_encoder.eval()

    running_loss = 0.0
    running_dice = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # batch is a list of dicts (from collate_fn)
        for sample in batch:
            image = sample['image'].to(device)
            mask_gt = sample['mask'].to(device)
            point_coords = sample['point_coords'].to(device)
            point_labels = sample['point_labels'].to(device)

            with autocast(dtype=torch.float16):
                # Forward through LoRA-SAM
                batched_input = [{
                    'image': image,
                    'point_coords': point_coords,
                    'point_labels': point_labels,
                    'original_size': sample['original_size'],
                }]
                outputs = model(batched_input, multimask_output=False)

                # Get best mask (index 0 for single-mask output)
                pred_logits = outputs[0]['masks']  # (1, 1024, 1024)
                pred_logits = pred_logits.unsqueeze(0)  # (1, 1, 1024, 1024)
                mask_gt_batch = mask_gt.unsqueeze(0)  # (1, 1, 1024, 1024)

                loss = criterion(pred_logits, mask_gt_batch) / grad_accum

            scaler.scale(loss).backward()

            # Compute Dice for logging
            with torch.no_grad():
                pred_binary = (torch.sigmoid(pred_logits) > 0.5).float()
                inter = (pred_binary * mask_gt_batch).sum()
                dice = (2 * inter + 1e-5) / (pred_binary.sum() + mask_gt_batch.sum() + 1e-5)
                running_dice += dice.item()

            running_loss += loss.item() * grad_accum
            n_batches += 1

        # Gradient accumulation step
        if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    avg_loss = running_loss / max(n_batches, 1)
    avg_dice = running_dice / max(n_batches, 1)
    return avg_loss, avg_dice


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate and compute average Dice score."""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    n_batches = 0

    for batch in dataloader:
        for sample in batch:
            image = sample['image'].to(device)
            mask_gt = sample['mask'].to(device)
            point_coords = sample['point_coords'].to(device)
            point_labels = sample['point_labels'].to(device)

            with autocast(dtype=torch.float16):
                batched_input = [{
                    'image': image,
                    'point_coords': point_coords,
                    'point_labels': point_labels,
                    'original_size': sample['original_size'],
                }]
                outputs = model(batched_input, multimask_output=False)

                pred_logits = outputs[0]['masks'].unsqueeze(0)
                mask_gt_batch = mask_gt.unsqueeze(0)
                loss = criterion(pred_logits, mask_gt_batch)

            pred_binary = (torch.sigmoid(pred_logits) > 0.5).float()
            inter = (pred_binary * mask_gt_batch).sum()
            dice = (2 * inter + 1e-5) / (pred_binary.sum() + mask_gt_batch.sum() + 1e-5)

            running_loss += loss.item()
            running_dice += dice.item()
            n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    avg_dice = running_dice / max(n_batches, 1)
    return avg_loss, avg_dice


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SAM LoRA Fine-Tuning on SABS CT")
    parser.add_argument('--fold', type=int, required=True, help='CV fold (0-4)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Peak learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--grad_accum', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--rank', type=int, default=4, help='LoRA rank')
    parser.add_argument('--val_every', type=int, default=5, help='Validate every N epochs')
    parser.add_argument('--sam_ckpt', type=str, default=None,
                        help='Path to sam_vit_b checkpoint')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory containing sabs_CT_normalized/')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"SAM LoRA Training: Fold {args.fold} (GPU {args.gpu})")
    print(f"{'='*70}")

    # ── Find paths ────────────────────────────────────────────────────────
    if args.sam_ckpt:
        sam_ckpt = args.sam_ckpt
    else:
        sam_ckpt = find_path("sam_vit_b_01ec64.pth", is_file=True)
        if not os.path.exists(sam_ckpt):
            sam_ckpt = find_path("sam_vit_b.pth", is_file=True)
    print(f"SAM checkpoint: {sam_ckpt}")

    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.dirname(find_path("sabs_CT_normalized"))
    print(f"Data directory: {data_dir}")

    # ── Build model ───────────────────────────────────────────────────────
    print("\nLoading SAM ViT-B...")
    sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt)
    lora_model = LoRA_Sam(sam, r=args.rank, lora_alpha=args.rank * 2)
    lora_model = lora_model.to(device)

    # ── Build datasets ────────────────────────────────────────────────────
    train_dataset = SABSLoRADataset(data_dir, fold=args.fold, is_train=True)
    val_dataset = SABSLoRADataset(data_dir, fold=args.fold, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )

    print(f"\nTrain samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Train batches/epoch: {len(train_loader)} | Effective batch: {args.batch_size * args.grad_accum}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    param_groups = lora_model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        [{'params': g['params'], 'lr': args.lr} for g in param_groups],
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # Cosine annealing with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loss & Scaler ─────────────────────────────────────────────────────
    criterion = CompoundLoss(dice_weight=0.8, bce_weight=0.2)
    scaler = GradScaler()

    # ── Output directory ──────────────────────────────────────────────────
    out_dir = os.path.join(REPO_DIR, "outputs", "lora_checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    best_dice = 0.0
    best_epoch = 0

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_dice = train_one_epoch(
            lora_model, train_loader, optimizer, criterion, scaler,
            device, grad_accum=args.grad_accum,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]['lr']

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_loss, val_dice = validate(lora_model, val_loader, criterion, device)

            is_best = val_dice > best_dice
            if is_best:
                best_dice = val_dice
                best_epoch = epoch + 1
                ckpt_path = os.path.join(out_dir, f"lora_fold{args.fold}_best.pth")
                lora_model.save_lora_parameters(ckpt_path)

            print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss={train_loss:.4f} Dice={train_dice:.4f} | "
                  f"Val Loss={val_loss:.4f} Dice={val_dice:.4f} {'★' if is_best else ''} | "
                  f"LR={lr_current:.2e} | {elapsed:.0f}s")
        else:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss={train_loss:.4f} Dice={train_dice:.4f} | "
                  f"LR={lr_current:.2e} | {elapsed:.0f}s")

    # ── Save final checkpoint ─────────────────────────────────────────────
    final_path = os.path.join(out_dir, f"lora_fold{args.fold}_final.pth")
    lora_model.save_lora_parameters(final_path)

    print(f"\n{'='*70}")
    print(f"Training complete! Best val Dice: {best_dice:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {os.path.join(out_dir, f'lora_fold{args.fold}_best.pth')}")
    print(f"Final checkpoint: {final_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
