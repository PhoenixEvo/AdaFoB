#!/bin/bash
# Multi-seed execution script for Rank 4 vs Rank 16 ablation
# To be executed in a Kaggle Notebook or similar environment with 2 T4 GPUs

echo "================================================================"
echo " Starting Multi-seed LoRA Rank Ablation (r=4 vs r=16)"
echo " Seeds: 43, 44 (Seed 42 was previously executed unseeded,"
echo " but we will re-run it for full consistency if desired)."
echo "================================================================"

# Make sure we are in the AdaFoB root
cd /kaggle/working/AdaFoB || cd .

for seed in 43 44; do
    echo "================================================================"
    echo " Training Seed $seed"
    echo "================================================================"
    
    # Train Rank 4
    echo ">>> Training Rank 4 (Seed $seed)"
    python experiments/train_lora.py --fold 0 --gpu 0 --epochs 50 --rank 4 --seed $seed > train_r4_seed${seed}_gpu0.log 2>&1 &
    python experiments/train_lora.py --fold 1 --gpu 1 --epochs 50 --rank 4 --seed $seed > train_r4_seed${seed}_gpu1.log 2>&1 &
    wait
    
    # Train Rank 16
    echo ">>> Training Rank 16 (Seed $seed)"
    python experiments/train_lora.py --fold 0 --gpu 0 --epochs 50 --rank 16 --seed $seed > train_r16_seed${seed}_gpu0.log 2>&1 &
    python experiments/train_lora.py --fold 1 --gpu 1 --epochs 50 --rank 16 --seed $seed > train_r16_seed${seed}_gpu1.log 2>&1 &
    wait
    
    echo "================================================================"
    echo " Evaluating Seed $seed"
    echo "================================================================"
    
    # Eval Rank 4
    echo ">>> Evaluating Rank 4 (Seed $seed)"
    python experiments/eval_lora.py --gpu 0 --organs 1 2 --rank 4 --seed $seed > eval_r4_seed${seed}_gpu0.log 2>&1 &
    python experiments/eval_lora.py --gpu 1 --organs 3 6 --rank 4 --seed $seed > eval_r4_seed${seed}_gpu1.log 2>&1 &
    wait
    
    # Eval Rank 16
    echo ">>> Evaluating Rank 16 (Seed $seed)"
    python experiments/eval_lora.py --gpu 0 --organs 1 2 --rank 16 --seed $seed > eval_r16_seed${seed}_gpu0.log 2>&1 &
    python experiments/eval_lora.py --gpu 1 --organs 3 6 --rank 16 --seed $seed > eval_r16_seed${seed}_gpu1.log 2>&1 &
    wait
done

echo "================================================================"
echo " Multi-seed ablation complete!"
echo " Please combine the CSVs and run experiments/analyze_rank_ablation.py"
echo "================================================================"
