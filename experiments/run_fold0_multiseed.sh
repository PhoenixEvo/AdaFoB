#!/bin/bash
# Representative-Fold Multi-Seed Runner (Fold 0 only)
# Trains r=4 on GPU 0 and r=16 on GPU 1 in parallel, then evaluates both.
# Usage: bash experiments/run_fold0_multiseed.sh <SEED>
set -e

SEED=${1:-42}
FOLD=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

echo "========================================================================"
echo ">> REPRESENTATIVE-FOLD MULTI-SEED RUN (FOLD $FOLD, SEED $SEED)"
echo ">> Repository: $REPO_DIR"
echo "========================================================================"

# Auto-download SAM ViT-B if not present
if [ ! -f /kaggle/working/checkpoints/sam_vit_b_01ec64.pth ] && [ -z "$(find /kaggle/input -name "sam_vit_b*.pth" 2>/dev/null)" ]; then
    echo ">> Downloading SAM ViT-B checkpoint..."
    mkdir -p /kaggle/working/checkpoints
    wget -q -O /kaggle/working/checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth || true
fi

CKPT_R4="/kaggle/working/checkpoints_r4_s${SEED}"
CKPT_R16="/kaggle/working/checkpoints_r16_s${SEED}"
mkdir -p "$CKPT_R4" "$CKPT_R16"

# ==============================================================================
# 1. PARALLEL TRAINING: Rank 4 on GPU 0 & Rank 16 on GPU 1
# ==============================================================================
echo ""
echo "========================================================================"
echo ">> TRAINING FOLD $FOLD (SEED $SEED)"
echo "   GPU 0: Rank 4"
echo "   GPU 1: Rank 16"
echo "========================================================================"

python experiments/train_lora.py --fold "$FOLD" --gpu 0 --rank 4 --seed "$SEED" --output_dir "$CKPT_R4" > train_r4_s${SEED}_f0.log 2>&1 &
python experiments/train_lora.py --fold "$FOLD" --gpu 1 --rank 16 --seed "$SEED" --output_dir "$CKPT_R16" > train_r16_s${SEED}_f0.log 2>&1 &
wait

echo ">> Training complete for both ranks on Fold $FOLD."

# ==============================================================================
# 2. EVALUATE RANK 4 ON FOLD 0
# ==============================================================================
echo ""
echo "========================================================================"
echo ">> EVALUATING RANK 4 (FOLD $FOLD, SEED $SEED)"
echo "========================================================================"
rm -f results/lora_eval*.csv

python experiments/eval_lora.py --gpu 0 --organs 1 2 --fold "$FOLD" --rank 4 --seed "$SEED" --lora_ckpt_dir "$CKPT_R4" > eval_r4_s${SEED}_gpu0.log 2>&1 &
python experiments/eval_lora.py --gpu 1 --organs 3 6 --fold "$FOLD" --rank 4 --seed "$SEED" --lora_ckpt_dir "$CKPT_R4" > eval_r4_s${SEED}_gpu1.log 2>&1 &
wait

python experiments/summarize_results.py
OUT_R4="/kaggle/working/lora_eval_fold0_rank4_seed${SEED}_combined.csv"
cp results/lora_eval_final_combined.csv "$OUT_R4" 2>/dev/null || true
cp results/lora_eval_final_combined.csv results/lora_eval_fold0_rank4_seed${SEED}_combined.csv 2>/dev/null || true
echo ">> Saved: $OUT_R4"

# ==============================================================================
# 3. EVALUATE RANK 16 ON FOLD 0
# ==============================================================================
echo ""
echo "========================================================================"
echo ">> EVALUATING RANK 16 (FOLD $FOLD, SEED $SEED)"
echo "========================================================================"
rm -f results/lora_eval*.csv

python experiments/eval_lora.py --gpu 0 --organs 1 2 --fold "$FOLD" --rank 16 --seed "$SEED" --lora_ckpt_dir "$CKPT_R16" > eval_r16_s${SEED}_gpu0.log 2>&1 &
python experiments/eval_lora.py --gpu 1 --organs 3 6 --fold "$FOLD" --rank 16 --seed "$SEED" --lora_ckpt_dir "$CKPT_R16" > eval_r16_s${SEED}_gpu1.log 2>&1 &
wait

python experiments/summarize_results.py
OUT_R16="/kaggle/working/lora_eval_fold0_rank16_seed${SEED}_combined.csv"
cp results/lora_eval_final_combined.csv "$OUT_R16" 2>/dev/null || true
cp results/lora_eval_final_combined.csv results/lora_eval_fold0_rank16_seed${SEED}_combined.csv 2>/dev/null || true
echo ">> Saved: $OUT_R16"

echo ""
echo "========================================================================"
echo ">> SUCCESS! HOÀN THÀNH FOLD 0 CHO SEED $SEED"
echo ">> 2 file kết quả cần tải về:"
echo "   1. $OUT_R4"
echo "   2. $OUT_R16"
echo "========================================================================"
