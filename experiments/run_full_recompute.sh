#!/bin/bash
# Recompute all HD95 numbers (Priority 2)
# Since masks are not cached, this requires re-inference on Kaggle.

echo "================================================================"
echo " Starting Full HD95 Recompute for All Ranks (Seed 42)"
echo " This will overwrite the buggy CSVs with the corrected HD95s."
echo "================================================================"

cd /kaggle/working/AdaFoB || cd .

for rank in 2 4 8 16; do
    echo "================================================================"
    echo " Evaluating Rank $rank (Seed 42)"
    echo "================================================================"
    
    python experiments/eval_lora.py --gpu 0 --organs 1 2 --rank $rank --seed 42 > eval_recompute_r${rank}_gpu0.log 2>&1 &
    python experiments/eval_lora.py --gpu 1 --organs 3 6 --rank $rank --seed 42 > eval_recompute_r${rank}_gpu1.log 2>&1 &
    wait
done

echo "================================================================"
echo " Recompute complete! Download the new lora_eval*.csv files"
echo " and we will re-run the downstream analyses locally."
echo "================================================================"
