#!/bin/bash
# Recompute all HD95 numbers (Priority 2)
# Since masks are not cached, this requires re-inference on Kaggle.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

echo "================================================================"
echo " Starting Full HD95 Recompute for All Ranks (Seed 42)"
echo " Repository directory: $REPO_DIR"
echo "================================================================"

# Check / download SAM ViT-B if needed
if [ ! -f /kaggle/working/sam_vit_b_01ec64.pth ] && [ -z "$(find /kaggle/input -name "sam_vit_b*.pth" 2>/dev/null)" ]; then
    echo ">> Downloading SAM ViT-B checkpoint to /kaggle/working/sam_vit_b_01ec64.pth..."
    wget -q -O /kaggle/working/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth || true
fi

find_ckpt_dir() {
    local r=$1
    local dir=""
    # Search for explicit rank (e.g. rank2, rank4, rank8, rank16)
    dir=$(dirname $(find /kaggle/input -name "lora_fold0_best.pth" 2>/dev/null | grep -i "rank${r}" | head -n 1 || true))
    if [ -z "$dir" ]; then
        dir=$(dirname $(find /kaggle/input -name "lora_fold0_best.pth" 2>/dev/null | grep -i "r${r}" | head -n 1 || true))
    fi
    # If rank 4 and not found, look for fallback without "rank" in the name
    if [ "$r" -eq 4 ] && [ -z "$dir" ]; then
        dir=$(dirname $(find /kaggle/input -name "lora_fold0_best.pth" 2>/dev/null | grep -v -i "rank" | head -n 1 || true))
    fi
    if [ -z "$dir" ]; then
        dir="/kaggle/input/datasets/nhatphatnguyen/adafob-rank${r}-ckpts"
    fi
    echo "$dir"
}

for rank in 2 4 8 16; do
    CKPT_DIR=$(find_ckpt_dir $rank)
    echo "================================================================"
    echo " Evaluating Rank $rank (Seed 42)"
    echo " Checkpoint directory: $CKPT_DIR"
    echo "================================================================"
    
    rm -f results/lora_eval*.csv
    
    python experiments/eval_lora.py --gpu 0 --rank $rank --seed 42 --lora_ckpt_dir "$CKPT_DIR" &
    python experiments/eval_lora.py --gpu 1 --rank $rank --seed 42 --lora_ckpt_dir "$CKPT_DIR" &
    wait
    
    python experiments/summarize_results.py
    cp results/lora_eval_final_combined.csv /kaggle/working/lora_eval_rank${rank}_combined.csv 2>/dev/null || true
    cp results/lora_eval_final_combined.csv results/lora_eval_rank${rank}_combined.csv 2>/dev/null || true
done

echo "================================================================"
echo " Recompute complete! Download the new lora_eval*.csv files"
echo " from /kaggle/working/ or results/."
echo "================================================================"
