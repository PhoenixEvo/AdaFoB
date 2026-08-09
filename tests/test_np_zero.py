"""Unit test: FoB forward pass with budget_Np=0 must not crash.

Verifies:
1. Forward pass completes without error
2. Returns neg_point with shape (0, 2) or (1, 0, 2)
3. Returns valid pos_point
4. All loss terms are finite during training with Np=0

Usage:
    python tests/test_np_zero.py --ckpt /path/to/ckpt.pth --data_root /path/to/abd-ct
"""
import sys
import os
import argparse
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.append(os.path.join(_ROOT, "third_party", "FoB_SAM"))

from models.FoB import FewShotSeg
from experiments.eval import (
    load_volumes, available_organs, sample_episode,
    build_inputs, load_checkpoint
)
from data.preprocess import norm_zscore


def make_baseline_norm(vol, dataset_mean=35.577, dataset_std=59.635):
    return lambda sl: norm_zscore(sl, dataset_mean, dataset_std)


def main():
    parser = argparse.ArgumentParser(description="Np=0 unit test")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to FoB checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Path to Abd-CT data")
    args = parser.parse_args()

    print("=" * 60)
    print("UNIT TEST: FoB forward pass with budget_Np=0")
    print("=" * 60)

    # Load data
    volumes, stats = load_volumes(args.data_root, (-1024, 3072), None)
    organ_map = {1: "spleen", 6: "liver"}
    organs = available_organs(volumes, organ_map, 200)
    if not organs:
        print("FAIL: No organs available")
        sys.exit(1)

    dummy = type("A", (), {"n_ways": 1, "n_shots": 1})()
    model = FewShotSeg(dummy).cuda().eval()
    load_checkpoint(model, args.ckpt, "FoB", strict=False)
    model.allocator = None

    cls = organs[0]
    ep = sample_episode(volumes, cls, 200)
    if ep is None:
        print("FAIL: Could not sample episode")
        sys.exit(1)

    sv = volumes[ep["support_vol"]]
    sample = build_inputs(volumes, ep, make_baseline_norm(sv))

    supp_imgs = [[t.clone().cuda() for t in way] for way in sample['support_images']]
    supp_masks = [[t.clone().cuda() for t in way] for way in sample['support_fg_labels']]
    qry_imgs = [t.clone().cuda() for t in sample['query_images']]
    qry_labels = sample['query_labels'].clone().cuda()

    all_passed = True

    # Test 1: Inference with Np=0
    print("\nTest 1: Inference with budget_Np=0")
    try:
        with torch.no_grad():
            neg_point, pos_point = model(
                supp_imgs, supp_masks, qry_imgs, qry_labels,
                train=False, use_skeleton=False, budget_Np=0
            )
        neg_arr = np.array(neg_point)
        pos_arr = np.array(pos_point)
        # neg_point should be empty
        neg_empty = (neg_arr.size == 0) or (neg_arr.shape[-2] == 0 if neg_arr.ndim >= 2 else neg_arr.shape[0] == 0)
        if neg_empty:
            print(f"  PASS: neg_point is empty (shape={neg_arr.shape})")
        else:
            print(f"  FAIL: neg_point is NOT empty (shape={neg_arr.shape})")
            all_passed = False
        if pos_arr.shape[-1] == 2 and pos_arr.size > 0:
            print(f"  PASS: pos_point is valid (shape={pos_arr.shape})")
        else:
            print(f"  FAIL: pos_point is invalid (shape={pos_arr.shape})")
            all_passed = False
    except Exception as e:
        print(f"  FAIL: Exception during inference: {e}")
        all_passed = False

    # Test 2: Training with Np=0 (check losses are finite)
    print("\nTest 2: Training with budget_Np=0")
    try:
        model.train()
        loss_tuple = model(
            supp_imgs, supp_masks, qry_imgs, qry_labels,
            train=True, use_skeleton=False, budget_Np=0
        )
        for i, loss_val in enumerate(loss_tuple):
            if torch.is_tensor(loss_val):
                finite = torch.isfinite(loss_val).all().item()
                print(f"  Loss[{i}] = {loss_val.item():.6f} {'PASS' if finite else 'FAIL (non-finite)'}")
                if not finite:
                    all_passed = False
            else:
                print(f"  Loss[{i}] = {loss_val} (non-tensor, PASS)")
        model.eval()
    except Exception as e:
        print(f"  FAIL: Exception during training: {e}")
        all_passed = False
        model.eval()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
