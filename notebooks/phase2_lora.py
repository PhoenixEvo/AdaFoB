"""
Phase 2: SAM LoRA Fine-Tuning for AdaFoB
==========================================
Kaggle Notebook Cells for training and evaluating LoRA-adapted SAM.

Prerequisite Dataset:
  - nhatphatnguyen/adafob-env (SABS CT normalized volumes)
  - FoB checkpoints
  - sam_vit_b_01ec64.pth (will be downloaded if not present)

GPU: T4 x 2
Runtime: ~3-4 hours for full 5-fold training + evaluation
"""

# ==============================================================================
# CELL 1: Setup — Git Pull + Dependencies
# ==============================================================================
"""
!cd /kaggle/working && git clone https://github.com/PhoenixEvo/AdaFoB.git 2>/dev/null; \
 cd AdaFoB && git pull origin main

!pip install medpy segment-anything 2>/dev/null

# Install segment-anything from local third_party if pip fails
!cd /kaggle/working/AdaFoB/third_party/segment-anything && pip install -e . 2>/dev/null
"""

# ==============================================================================
# CELL 2: Download SAM ViT-B Checkpoint
# ==============================================================================
"""
import os
sam_b_path = "/kaggle/working/checkpoints/sam_vit_b_01ec64.pth"
if not os.path.exists(sam_b_path):
    os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
    !wget -q -O {sam_b_path} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    print(f"Downloaded SAM ViT-B: {os.path.getsize(sam_b_path) / 1e6:.0f} MB")
else:
    print(f"SAM ViT-B already exists: {sam_b_path}")
"""

# ==============================================================================
# CELL 3: Train LoRA — Fold 0 & 1 (Parallel on 2 GPUs)
# ==============================================================================
"""
# Training Fold 0 on GPU 0, Fold 1 on GPU 1 simultaneously
!cd /kaggle/working/AdaFoB && \
 python experiments/train_lora.py \
   --fold 0 --gpu 0 --epochs 60 --lr 5e-4 \
   --batch_size 1 --grad_accum 16 --rank 4 \
   --sam_ckpt /kaggle/working/checkpoints/sam_vit_b_01ec64.pth & \
 cd /kaggle/working/AdaFoB && \
 python experiments/train_lora.py \
   --fold 1 --gpu 1 --epochs 60 --lr 5e-4 \
   --batch_size 1 --grad_accum 16 --rank 4 \
   --sam_ckpt /kaggle/working/checkpoints/sam_vit_b_01ec64.pth & \
 wait
"""

# ==============================================================================
# CELL 4: Train LoRA — Fold 2 & 3 (Parallel on 2 GPUs)
# ==============================================================================
"""
!cd /kaggle/working/AdaFoB && \
 python experiments/train_lora.py \
   --fold 2 --gpu 0 --epochs 60 --lr 5e-4 \
   --batch_size 1 --grad_accum 16 --rank 4 \
   --sam_ckpt /kaggle/working/checkpoints/sam_vit_b_01ec64.pth & \
 cd /kaggle/working/AdaFoB && \
 python experiments/train_lora.py \
   --fold 3 --gpu 1 --epochs 60 --lr 5e-4 \
   --batch_size 1 --grad_accum 16 --rank 4 \
   --sam_ckpt /kaggle/working/checkpoints/sam_vit_b_01ec64.pth & \
 wait
"""

# ==============================================================================
# CELL 5: Train LoRA — Fold 4 (Single GPU)
# ==============================================================================
"""
!cd /kaggle/working/AdaFoB && \
 python experiments/train_lora.py \
   --fold 4 --gpu 0 --epochs 60 --lr 5e-4 \
   --batch_size 1 --grad_accum 16 --rank 4 \
   --sam_ckpt /kaggle/working/checkpoints/sam_vit_b_01ec64.pth
"""

# ==============================================================================
# CELL 6: Verify LoRA Checkpoints
# ==============================================================================
"""
import os, glob
ckpt_dir = "/kaggle/working/AdaFoB/outputs/lora_checkpoints/"
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
print(f"Found {len(ckpts)} LoRA checkpoints:")
for c in ckpts:
    size = os.path.getsize(c) / (1024*1024)
    print(f"  {os.path.basename(c)}: {size:.1f} MB")
"""

# ==============================================================================
# CELL 7: Evaluate — All Folds (Parallel on 2 GPUs)
# ==============================================================================
"""
# GPU 0: Spleen (1) + RK (2)
# GPU 1: LK (3) + Liver (6)
!cd /kaggle/working/AdaFoB && \
 python experiments/eval_lora.py --gpu 0 --organs 1 2 & \
 cd /kaggle/working/AdaFoB && \
 python experiments/eval_lora.py --gpu 1 --organs 3 6 & \
 wait
"""

# ==============================================================================
# CELL 8: Analyze Results
# ==============================================================================
"""
import pandas as pd

df0 = pd.read_csv("/kaggle/working/AdaFoB/results/lora_eval_gpu0.csv")
df1 = pd.read_csv("/kaggle/working/AdaFoB/results/lora_eval_gpu1.csv")
df = pd.concat([df0, df1], ignore_index=True)

print("=" * 80)
print("FULL RESULTS: BASELINE vs LORA")
print("=" * 80)

for organ in df['organ'].unique():
    dfo = df[df['organ'] == organ]
    print(f"\n--- {organ} ---")
    if 'dice_baseline' in df.columns:
        print(f"  FoB Baseline (ViT-H):  {dfo['dice_baseline'].mean()*100:.2f}%")
        print(f"  AdaFoB 2D (ViT-H):     {dfo['dice_adafob_2d'].mean()*100:.2f}%")
    if 'dice_lora_fob' in df.columns:
        print(f"  FoB + LoRA (ViT-B):    {dfo['dice_lora_fob'].mean()*100:.2f}%")
        print(f"  AdaFoB + LoRA (OURS):  {dfo['dice_lora_ada'].mean()*100:.2f}%")

print(f"\n{'='*80}")
print("OVERALL")
if 'dice_baseline' in df.columns:
    print(f"  FoB Baseline:    {df['dice_baseline'].mean()*100:.2f}%")
    print(f"  AdaFoB 2D:       {df['dice_adafob_2d'].mean()*100:.2f}%")
if 'dice_lora_fob' in df.columns:
    print(f"  FoB + LoRA:      {df['dice_lora_fob'].mean()*100:.2f}%")
    print(f"  AdaFoB + LoRA:   {df['dice_lora_ada'].mean()*100:.2f}%")
print("=" * 80)
"""

# ==============================================================================
# CELL 9 (Optional): Quick Smoke Test — Train 2 Epochs on Fold 0
# ==============================================================================
"""
# Run this first to verify everything works before committing to full training
!cd /kaggle/working/AdaFoB && \
 python experiments/train_lora.py \
   --fold 0 --gpu 0 --epochs 2 --lr 5e-4 \
   --batch_size 1 --grad_accum 16 --rank 4 --val_every 1 \
   --sam_ckpt /kaggle/working/checkpoints/sam_vit_b_01ec64.pth
"""
