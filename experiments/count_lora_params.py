"""Count LoRA / trainable parameters of LoRA_Sam for a list of ranks.

Counts are obtained by instantiating SAM ViT-B (random init, no checkpoint
download needed because parameter counts do not depend on weight values) and
wrapping it with LoRA_Sam exactly as train_lora.py does. Optionally the LoRA
tensors stored inside a saved checkpoint are counted as an independent check.

Usage:
    python experiments/count_lora_params.py --ranks 2 4 8 16 \
        --out_csv results/lora_param_counts.csv \
        --ckpt outputs/rank4_fold0_1_results/lora_fold0_best.pth --ckpt_rank 4
"""
import os
import sys
import argparse
from typing import Dict, List, Optional

import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "third_party", "segment-anything"))

from segment_anything import sam_model_registry
from experiments.lora_sam import LoRA_Sam


def count_for_rank(rank: int, alpha_multiplier: int) -> Dict[str, int]:
    """Instantiate LoRA_Sam for `rank` and count parameter groups.

    Args:
        rank: LoRA rank r.
        alpha_multiplier: lora_alpha = alpha_multiplier * rank (train_lora.py uses 2).

    Returns:
        Dict with lora_params (A and B matrices of q and v in every ViT block),
        mask_decoder_params, prompt_encoder_params, trainable_params, total_params.
    """
    sam = sam_model_registry["vit_b"](checkpoint=None)
    model = LoRA_Sam(sam, r=rank, lora_alpha=alpha_multiplier * rank)
    lora = sum(p.numel() for layer in model.lora_layers for n, p in layer.named_parameters()
               if "linear_a" in n or "linear_b" in n)
    md = sum(p.numel() for p in model.sam.mask_decoder.parameters())
    pe = sum(p.numel() for p in model.sam.prompt_encoder.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "rank": rank,
        "n_blocks": len(model.lora_layers),
        "lora_params": lora,
        "mask_decoder_params": md,
        "prompt_encoder_params": pe,
        "trainable_params": trainable,
        "total_params": total,
    }


def count_in_checkpoint(path: str) -> Optional[int]:
    """Count numel of all LoRA tensors (keys starting with 'lora.') stored in a checkpoint."""
    state = torch.load(path, map_location="cpu")
    total = 0
    for key, value in state.items():
        if str(key).startswith("lora."):
            total += sum(t.numel() for t in value.values())
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Count LoRA parameters per rank")
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--alpha_multiplier", type=int, default=2)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional saved checkpoint to count LoRA tensors in")
    parser.add_argument("--ckpt_rank", type=int, default=None, help="Rank that --ckpt was trained with")
    args = parser.parse_args()

    rows: List[Dict[str, int]] = [count_for_rank(r, args.alpha_multiplier) for r in args.ranks]
    import pandas as pd
    df = pd.DataFrame(rows)
    # Closed-form check: 4 matrices (A_q, B_q, A_v, B_v) of size dim*r per block.
    dim = 768
    df["formula_lora_params"] = 4 * dim * df["rank"] * df["n_blocks"]
    df["formula_matches"] = df["formula_lora_params"] == df["lora_params"]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))

    if args.ckpt:
        n = count_in_checkpoint(args.ckpt)
        print(f"\nLoRA tensors stored in checkpoint {args.ckpt}: {n:,}")
        if args.ckpt_rank is not None:
            expected = int(df.loc[df["rank"] == args.ckpt_rank, "lora_params"].iloc[0])
            print(f"Expected for rank {args.ckpt_rank}: {expected:,}  match={n == expected}")


if __name__ == "__main__":
    main()
