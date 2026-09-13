#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

echo "Repository directory: $REPO_DIR"

RANK8_DIR=$(dirname $(find /kaggle/input -name "lora_fold0_best.pth" 2>/dev/null | grep -i "rank8" | head -n 1 || true))
if [ -z "$RANK8_DIR" ]; then 
    RANK8_DIR="/kaggle/input/datasets/nhatphatnguyen/adafob-rank8-ckpts"
fi

RANK16_DIR=$(dirname $(find /kaggle/input -name "lora_fold0_best.pth" 2>/dev/null | grep -i "rank16" | head -n 1 || true))
if [ -z "$RANK16_DIR" ]; then 
    RANK16_DIR="/kaggle/input/datasets/nhatphatnguyen/adafob-rank16-ckpts"
fi

echo "========================================================================"
echo ">> [1/2] B?T ??U ??NH GI? RANK 8..."
echo "Checkpoint directory: $RANK8_DIR"
echo "========================================================================"

rm -f results/lora_eval*.csv

python experiments/eval_lora.py --gpu 0 --rank 8 --lora_ckpt_dir "$RANK8_DIR" &
python experiments/eval_lora.py --gpu 1 --rank 8 --lora_ckpt_dir "$RANK8_DIR" &
wait

python experiments/summarize_results.py
cp results/lora_eval_final_combined.csv /kaggle/working/lora_eval_rank8_combined.csv 2>/dev/null || true

echo "========================================================================"
echo ">> [2/2] B?T ??U ??NH GI? RANK 16..."
echo "Checkpoint directory: $RANK16_DIR"
echo "========================================================================"

rm -f results/lora_eval*.csv

python experiments/eval_lora.py --gpu 0 --rank 16 --lora_ckpt_dir "$RANK16_DIR" &
python experiments/eval_lora.py --gpu 1 --rank 16 --lora_ckpt_dir "$RANK16_DIR" &
wait

python experiments/summarize_results.py
cp results/lora_eval_final_combined.csv /kaggle/working/lora_eval_rank16_combined.csv 2>/dev/null || true

echo "========================================================================"
echo ">> HO?N T?T ??NH GI? C? 2 RANK!"
echo "Files ?? l?u t?i:"
echo "  - /kaggle/working/lora_eval_rank8_combined.csv"
echo "  - /kaggle/working/lora_eval_rank16_combined.csv"
echo "========================================================================"
