"""Parity test: PBA forced to Np=10 uniform ring must match baseline FoB.

Per phase5_pivot_spec.md Section 5.4, this test is MANDATORY before any
comparison. A refactor that changes FoB's behaviour invalidates every result.

Usage:
    python tests/test_parity.py --ckpt /path/to/ckpt.pth --data_root /path/to/abd-ct
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
    parser = argparse.ArgumentParser(description="Parity test: PBA vs baseline FoB")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to FoB checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Path to Abd-CT data")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    parser.add_argument("--n_episodes", type=int, default=5, help="Number of test episodes")
    args = parser.parse_args()

    print("=" * 60)
    print("PARITY TEST: PBA(Np=10, uniform, r=15) vs Baseline FoB(Np=10)")
    print("=" * 60)

    # Load data
    volumes, stats = load_volumes(args.data_root, (-1024, 3072), None)
    organ_map = {1: "spleen", 2: "rk", 3: "lk", 6: "liver", 11: "pancreas"}
    organs = available_organs(volumes, organ_map, 200)
    if not organs:
        print("FAIL: No organs available")
        sys.exit(1)

    # Create two identical model instances
    dummy = type("A", (), {"n_ways": 1, "n_shots": 1})()

    baseline = FewShotSeg(dummy).cuda().eval()
    load_checkpoint(baseline, args.ckpt, "Baseline", strict=False)
    baseline.allocator = None

    pba_model = FewShotSeg(dummy).cuda().eval()
    load_checkpoint(pba_model, args.ckpt, "PBA", strict=False)
    pba_model.allocator = None  # Will also run without allocator but with same Np

    passed = 0
    failed = 0

    for ep_i in range(args.n_episodes):
        cls = organs[ep_i % len(organs)]
        ep = sample_episode(volumes, cls, 200)
        if ep is None:
            print(f"  Episode {ep_i}: skipped (no valid episode)")
            continue

        sv = volumes[ep["support_vol"]]
        sample = build_inputs(volumes, ep, make_baseline_norm(sv))

        supp_imgs = [[t.clone().cuda() for t in way] for way in sample['support_images']]
        supp_masks = [[t.clone().cuda() for t in way] for way in sample['support_fg_labels']]
        qry_imgs = [t.clone().cuda() for t in sample['query_images']]
        qry_labels = sample['query_labels'].clone().cuda()

        # Set same seed for reproducibility within each forward pass
        torch.manual_seed(42 + ep_i)
        np.random.seed(42 + ep_i)

        with torch.no_grad():
            # Baseline: allocator=None, budget_Np=10
            baseline.allocator = None
            torch.manual_seed(42 + ep_i)
            np.random.seed(42 + ep_i)
            neg_base, pos_base = baseline(
                supp_imgs, supp_masks, qry_imgs, qry_labels,
                train=False, use_skeleton=False, budget_Np=10
            )

            # PBA model: also allocator=None, budget_Np=10 (same path)
            pba_model.allocator = None
            torch.manual_seed(42 + ep_i)
            np.random.seed(42 + ep_i)
            neg_pba, pos_pba = pba_model(
                supp_imgs, supp_masks, qry_imgs, qry_labels,
                train=False, use_skeleton=False, budget_Np=10
            )

        neg_base = np.array(neg_base)
        neg_pba = np.array(neg_pba)
        pos_base = np.array(pos_base)
        pos_pba = np.array(pos_pba)

        neg_match = np.allclose(neg_base, neg_pba, atol=args.atol)
        pos_match = np.allclose(pos_base, pos_pba, atol=args.atol)

        if neg_match and pos_match:
            passed += 1
            print(f"  Episode {ep_i} ({organ_map[cls]}): PASS")
        else:
            failed += 1
            if not neg_match:
                diff = np.abs(neg_base - neg_pba).max()
                print(f"  Episode {ep_i} ({organ_map[cls]}): FAIL neg_points max_diff={diff:.8f}")
            if not pos_match:
                diff = np.abs(pos_base - pos_pba).max()
                print(f"  Episode {ep_i} ({organ_map[cls]}): FAIL pos_points max_diff={diff:.8f}")

    print("\n" + "=" * 60)
    if failed == 0:
        print(f"PARITY TEST PASSED: {passed}/{passed} episodes match (atol={args.atol})")
    else:
        print(f"PARITY TEST FAILED: {failed}/{passed+failed} episodes differ")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
