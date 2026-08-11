"""R0: Per-organ pipeline validation (oracle gate).

Per Opus remediation spec (e3_review_and_remediation.md, Section R0):
- Run the oracle gate SEPARATELY for every organ we intend to report.
- Use positives-only (n_pos=10, n_neg=0) to measure the true pipeline ceiling.
- Report the with-negatives value separately as data.
- Acceptance: every reported organ has positives-only oracle >= 0.85.
"""
import sys
import os
import csv
import random
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.append(os.path.join(_ROOT, "third_party", "FoB_SAM"))

from segment_anything import sam_model_registry, SamPredictor

from experiments.eval import (
    load_volumes, available_organs, sample_episode,
    predict_sam_from_points, compute_dice
)
from data.preprocess import oracle_prompts, sam_uint8_from_canonical


def run_oracle_per_organ(predictor, volumes, organ_id, organ_name,
                         n_cases=30, n_pos=10, seed=42):
    """Run oracle gate for one organ, both positives-only and with negatives."""
    rng = random.Random(seed)
    results_pos_only = []
    results_with_neg = []

    for i in range(n_cases):
        ep = sample_episode(volumes, organ_id, min_pixels=50)
        if ep is None:
            continue
        qv = volumes[ep["query_vol"]]
        q = ep["query_slice"]
        gt = (qv["label"][q] == organ_id).astype(np.uint8)
        if gt.sum() < 20:
            continue

        sam_img = sam_uint8_from_canonical(qv["canon"][q])
        predictor.set_image(sam_img)
        H, W = gt.shape

        # Positives-only (the true pipeline ceiling)
        pos, _ = oracle_prompts(gt, n_pos=n_pos, n_neg=0, rng=random.Random(seed + i))
        pred_pos, _ = predict_sam_from_points(predictor, pos, np.zeros((0, 2), np.float32),
                                              H, W, mask_select="fixed0")
        d_pos = compute_dice(pred_pos, gt) if pred_pos is not None else 0.0
        results_pos_only.append(d_pos)

        # With negatives (to measure the depression effect)
        pos2, neg2 = oracle_prompts(gt, n_pos=n_pos, n_neg=10, rng=random.Random(seed + i))
        pred_neg, _ = predict_sam_from_points(predictor, pos2, neg2,
                                              H, W, mask_select="fixed0")
        d_neg = compute_dice(pred_neg, gt) if pred_neg is not None else 0.0
        results_with_neg.append(d_neg)

    return results_pos_only, results_with_neg


def main():
    print("R0: Per-Organ Pipeline Validation (Oracle Gate)")
    print("=" * 60)

    data_root = "/kaggle/input/datasets/nhatphatnguyen/abd-ct"
    sam_ckpt = "/kaggle/working/checkpoints/sam_vit_h_4b8939.pth"

    # Auto-detect data root
    import glob
    for pat in ["/kaggle/input/**/*sabs_CT_normalized*", "/kaggle/input/**/*abd*ct*"]:
        for h in glob.glob(pat, recursive=True):
            if os.path.isdir(h):
                data_root = h
                break

    volumes, stats = load_volumes(data_root, (-1024, 3072), None)

    # The four competent organs (FoB was trained on these)
    organ_map = {1: "spleen", 2: "rk", 3: "lk", 6: "liver"}
    available = available_organs(volumes, organ_map, 50)

    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
    predictor = SamPredictor(sam)

    rows = []
    gate_pass = True

    for organ_id in [1, 2, 3, 6]:
        if organ_id not in available:
            print(f"\n  organ {organ_id} ({organ_map[organ_id]}): SKIPPED (not available)")
            continue

        organ_name = organ_map[organ_id]
        print(f"\nRunning oracle gate for {organ_name} (ID={organ_id})...")

        pos_only, with_neg = run_oracle_per_organ(
            predictor, volumes, organ_id, organ_name, n_cases=30, seed=42
        )

        mean_pos = float(np.mean(pos_only)) if pos_only else 0.0
        std_pos = float(np.std(pos_only)) if pos_only else 0.0
        mean_neg = float(np.mean(with_neg)) if with_neg else 0.0
        std_neg = float(np.std(with_neg)) if with_neg else 0.0
        depression = mean_pos - mean_neg

        passed = mean_pos >= 0.85
        status = "PASS" if passed else "FAIL"
        if not passed:
            gate_pass = False

        print(f"  Positives-only:   {mean_pos:.4f} +/- {std_pos:.4f}  [{status}]")
        print(f"  With 10 neg:      {mean_neg:.4f} +/- {std_neg:.4f}")
        print(f"  Neg depression:   {depression:+.4f}")

        rows.append({
            "organ_id": organ_id,
            "organ_name": organ_name,
            "n_cases": len(pos_only),
            "oracle_pos_only_mean": round(mean_pos, 4),
            "oracle_pos_only_std": round(std_pos, 4),
            "oracle_with_neg_mean": round(mean_neg, 4),
            "oracle_with_neg_std": round(std_neg, 4),
            "neg_depression": round(depression, 4),
            "gate_status": status
        })

    # Write results
    os.makedirs("results", exist_ok=True)
    out_path = "results/r0_oracle_gate.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out_path}")

    print("\n" + "=" * 60)
    if gate_pass:
        print("ORACLE GATE: ALL ORGANS PASS (>= 0.85). Pipeline is valid.")
        print("Proceed to R1 (E1 budget sweep on organs 1, 2, 3, 6).")
    else:
        failed = [r["organ_name"] for r in rows if r["gate_status"] == "FAIL"]
        print(f"ORACLE GATE: FAILED for {failed}.")
        print("These organs may only appear in the limitations section.")
    print("=" * 60)


if __name__ == "__main__":
    main()
