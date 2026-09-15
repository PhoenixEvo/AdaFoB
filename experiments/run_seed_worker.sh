#!/bin/bash
# Worker script to run full 5-fold training and evaluation for a single seed (r=4 and r=16)
# Usage: bash experiments/run_seed_worker.sh <SEED>
set -e

SEED=${1:-42}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

echo "========================================================================"
echo ">> STARTING FULL MULTI-SEED RUN FOR SEED: $SEED"
echo ">> Repository: $REPO_DIR"
echo "========================================================================"

# Check / download SAM ViT-B if needed
if [ ! -f /kaggle/working/checkpoints/sam_vit_b_01ec64.pth ] && [ -z "$(find /kaggle/input -name "sam_vit_b*.pth" 2>/dev/null)" ]; then
    echo ">> Downloading SAM ViT-B checkpoint..."
    mkdir -p /kaggle/working/checkpoints
    wget -q -O /kaggle/working/checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth || true
fi

CKPT_R4="/kaggle/working/checkpoints_r4_seed${SEED}"
CKPT_R16="/kaggle/working/checkpoints_r16_seed${SEED}"
mkdir -p "$CKPT_R4" "$CKPT_R16"

# ==============================================================================
# PHASE 1: TRAIN RANK 4 (5 FOLDS)
# ==============================================================================
echo ""
echo "========================================================================"
echo ">> [1/4] TRAINING RANK 4 (SEED $SEED) — 5 FOLDS"
echo "========================================================================"
echo "-> Training Folds 0 & 1 in parallel on GPU 0 & 1..."
python experiments/train_lora.py --fold 0 --gpu 0 --rank 4 --seed "$SEED" --output_dir "$CKPT_R4" > train_r4_s${SEED}_f0.log 2>&1 &
python experiments/train_lora.py --fold 1 --gpu 1 --rank 4 --seed "$SEED" --output_dir "$CKPT_R4" > train_r4_s${SEED}_f1.log 2>&1 &
wait

echo "-> Training Folds 2 & 3 in parallel on GPU 0 & 1..."
python experiments/train_lora.py --fold 2 --gpu 0 --rank 4 --seed "$SEED" --output_dir "$CKPT_R4" > train_r4_s${SEED}_f2.log 2>&1 &
python experiments/train_lora.py --fold 3 --gpu 1 --rank 4 --seed "$SEED" --output_dir "$CKPT_R4" > train_r4_s${SEED}_f3.log 2>&1 &
wait

echo "-> Training Fold 4 on GPU 0..."
python experiments/train_lora.py --fold 4 --gpu 0 --rank 4 --seed "$SEED" --output_dir "$CKPT_R4" > train_r4_s${SEED}_f4.log 2>&1

echo ">> Rank 4 training complete for all 5 folds."

# ==============================================================================
# PHASE 2: EVALUATE RANK 4
# ==============================================================================
echo ""
echo "========================================================================"
echo ">> [2/4] EVALUATING RANK 4 (SEED $SEED)"
echo "========================================================================"
rm -f results/lora_eval*.csv

python experiments/eval_lora.py --gpu 0 --rank 4 --seed "$SEED" --lora_ckpt_dir "$CKPT_R4" > eval_r4_s${SEED}_gpu0.log 2>&1 &
python experiments/eval_lora.py --gpu 1 --rank 4 --seed "$SEED" --lora_ckpt_dir "$CKPT_R4" > eval_r4_s${SEED}_gpu1.log 2>&1 &
wait

python experiments/summarize_results.py
cp results/lora_eval_final_combined.csv /kaggle/working/lora_eval_rank4_seed${SEED}_combined.csv 2>/dev/null || true
cp results/lora_eval_final_combined.csv results/lora_eval_rank4_seed${SEED}_combined.csv 2>/dev/null || true
echo ">> Saved: /kaggle/working/lora_eval_rank4_seed${SEED}_combined.csv"

# ==============================================================================
# PHASE 3: TRAIN RANK 16 (5 FOLDS)
# ==============================================================================
echo ""
echo "========================================================================"
echo ">> [3/4] TRAINING RANK 16 (SEED $SEED) — 5 FOLDS"
echo "========================================================================"
echo "-> Training Folds 0 & 1 in parallel on GPU 0 & 1..."
python experiments/train_lora.py --fold 0 --gpu 0 --rank 16 --seed "$SEED" --output_dir "$CKPT_R16" > train_r16_s${SEED}_f0.log 2>&1 &
python experiments/train_lora.py --fold 1 --gpu 1 --rank 16 --seed "$SEED" --output_dir "$CKPT_R16" > train_r16_s${SEED}_f1.log 2>&1 &
wait

echo "-> Training Folds 2 & 3 in parallel on GPU 0 & 1..."
python experiments/train_lora.py --fold 2 --gpu 0 --rank 16 --seed "$SEED" --output_dir "$CKPT_R16" > train_r16_s${SEED}_f2.log 2>&1 &
python experiments/train_lora.py --fold 3 --gpu 1 --rank 16 --seed "$SEED" --output_dir "$CKPT_R16" > train_r16_s${SEED}_f3.log 2>&1 &
wait

echo "-> Training Fold 4 on GPU 0..."
python experiments/train_lora.py --fold 4 --gpu 0 --rank 16 --seed "$SEED" --output_dir "$CKPT_R16" > train_r16_s${SEED}_f4.log 2>&1

echo ">> Rank 16 training complete for all 5 folds."

# ==============================================================================
# PHASE 4: EVALUATE RANK 16
# ==============================================================================
echo ""
echo "========================================================================"
echo ">> [4/4] EVALUATING RANK 16 (SEED $SEED)"
echo "========================================================================"
rm -f results/lora_eval*.csv

python experiments/eval_lora.py --gpu 0 --rank 16 --seed "$SEED" --lora_ckpt_dir "$CKPT_R16" > eval_r16_s${SEED}_gpu0.log 2>&1 &
python experiments/eval_lora.py --gpu 1 --rank 16 --seed "$SEED" --lora_ckpt_dir "$CKPT_R16" > eval_r16_s${SEED}_gpu1.log 2>&1 &
wait

python experiments/summarize_results.py
cp results/lora_eval_final_combined.csv /kaggle/working/lora_eval_rank16_seed${SEED}_combined.csv 2>/dev/null || true
cp results/lora_eval_final_combined.csv results/lora_eval_rank16_seed${SEED}_combined.csv 2>/dev/null || true
echo ">> Saved: /kaggle/working/lora_eval_rank16_seed${SEED}_combined.csv"

echo ""
echo "========================================================================"
echo ">> HOÀN TẤT TOÀN BỘ PIPELINE CHO SEED $SEED!"
echo ">> Hai file kết quả cần nộp nằm tại:"
echo "   1. /kaggle/working/lora_eval_rank4_seed${SEED}_combined.csv"
echo "   2. /kaggle/working/lora_eval_rank16_seed${SEED}_combined.csv"
echo "========================================================================"
