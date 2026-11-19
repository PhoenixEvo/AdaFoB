"""CPU unit tests for AdaFoB v2 Stage 1 semantic-gated selectors and decision rules."""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.adafob_v2_stage1.metrics import (
    compute_estimands_v2,
    evaluate_stage1_decision,
    paired_bootstrap_contrasts_v2,
)
from experiments.adafob_v2_stage1.selectors import (
    ARMS,
    place_pba_fixed_k2,
    place_semgate_pba,
)

ROOT = Path(__file__).resolve().parents[1]
OPTION_A_CSV = ROOT / "experiments/runs/option_a_reassessment_20261119_v2/kaggle_outputs/measurements.csv"


class TestAdaFoBV2Stage1(unittest.TestCase):
    def test_liver_interior_cut_safety_vs_unconditional_pba(self) -> None:
        h, w = 257, 257
        yy, xx = np.ogrid[:h, :w]
        cx, cy, r = 128.0, 128.0, 40.0
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r**2)

        # Entire right side is missed organ tissue (iso-intense +1.0 inside & outside, e_local = +0.6);
        # Left side borders an iso-intense adjacent organ (+0.95) that is background (e_local = -0.6).
        img = np.broadcast_to(np.where(xx >= cx, 1.0, 0.95), (h, w)).astype(np.float64)
        e_local = np.broadcast_to(np.where(xx >= cx, 0.6, -0.6), (h, w)).astype(np.float64)

        pts_pba, _ = place_pba_fixed_k2(mask, img, n_samples=2, offset_px=5.0)
        self.assertIsNotNone(pts_pba)
        assert pts_pba is not None
        # Unconditional PBA places at least one point into the missed right half (x > cx).
        self.assertTrue(bool((pts_pba[:, 0] > cx).any()))

        pts_sg, meta_sg = place_semgate_pba(mask, img, e_local, n_samples=2, offsets=(5.0,), use_intensity=True)
        self.assertIsNotNone(pts_sg)
        assert pts_sg is not None
        self.assertIsNone(meta_sg["fallback"])
        # Semantic-gated PBA puts 100% of placed prompts in true exterior background (left side, e_local < 0).
        self.assertTrue(bool((pts_sg[:, 0] < cx).all()))
        for x, y in pts_sg:
            self.assertLess(float(e_local[int(round(y)), int(round(x))]), 0.0)

    def test_all_undersegmented_triggers_zero_negative_gate(self) -> None:
        h, w = 257, 257
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - 128.0) ** 2 + (yy - 128.0) ** 2) <= (35.0**2)
        img = np.ones((h, w), dtype=np.float64)
        e_local = np.full((h, w), 0.5, dtype=np.float64)

        pts, meta = place_semgate_pba(mask, img, e_local, n_samples=2, offsets=(5.0, 8.0, 11.0, 14.0), use_intensity=True)
        self.assertIsNotNone(pts)
        assert pts is not None
        self.assertEqual(pts.shape, (0, 2))
        self.assertEqual(meta["fallback"], "no_bg_arc")
        self.assertEqual(meta["k_placed"], 0)

    def test_outward_semantic_raymarch_finds_clean_background(self) -> None:
        h, w = 257, 257
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - 128.0) ** 2 + (yy - 128.0) ** 2)
        r = 35.0
        mask = dist <= r
        img = np.where(dist <= r + 6.5, 1.0, 0.2).astype(np.float64)
        # Blurred organ rim extends 6.5 px outside coarse mask (e_local > 0 at r=5, < 0 at r>=8).
        e_local = np.where(dist <= r + 6.5, 0.2, -0.4).astype(np.float64)

        pts_r5, meta_r5 = place_semgate_pba(mask, img, e_local, n_samples=2, offsets=(5.0,), use_intensity=True)
        self.assertEqual(pts_r5.shape, (0, 2))
        self.assertEqual(meta_r5["fallback"], "no_bg_arc")

        pts_rm, meta_rm = place_semgate_pba(mask, img, e_local, n_samples=2, offsets=(5.0, 8.0, 11.0, 14.0), use_intensity=True)
        self.assertEqual(meta_rm["fallback"], None)
        self.assertEqual(pts_rm.shape, (2, 2))
        self.assertGreaterEqual(meta_rm["mean_offset_px"], 8.0)

        pts_safe, meta_safe = place_semgate_pba(mask, img, e_local, n_samples=2, offsets=(8.0, 11.0, 14.0), use_intensity=True)
        self.assertEqual(meta_safe["fallback"], None)
        self.assertEqual(pts_safe.shape, (2, 2))
        for x, y in pts_safe:
            d_pt = math.hypot(x - 128.0, y - 128.0)
            self.assertGreaterEqual(d_pt, r + 6.8)
            self.assertLess(float(e_local[int(round(y)), int(round(x))]), 0.0)

    def test_stage1_decision_bar_and_option_a_anchors(self) -> None:
        if OPTION_A_CSV.is_file():
            df = pd.read_csv(OPTION_A_CSV).replace({"arm": {"pba_fixed_k2": "pba_f10_int_k2"}})
            sub = df[df["arm"].isin(["full10", "positive0", "boundary_near2", "pba_f10_int_k2"])].copy()
            est = compute_estimands_v2(sub)
            cc = est[est["post"] == "cc"].set_index("arm")
            self.assertAlmostEqual(float(cc.loc["pba_f10_int_k2", "patient_dice_pct"]), 84.718990, places=4)

        # Build synthetic 4-organ 5-patient DataFrame covering all 11 ARMS to check decision bar logic.
        rows = []
        rng = np.random.default_rng(7)
        for p in range(5):
            pid = f"image_{p}"
            for organ in ("LIVER", "LK", "RK", "SPLEEN"):
                for arm in ARMS:
                    base_dice = 0.82
                    base_rec = 0.94 if organ != "LIVER" else 0.92
                    base_hd = 30.0
                    if arm == "boundary_near2":
                        base_dice = 0.83
                    elif arm == "semgate_pos0_k2_r5":
                        base_dice = 0.855  # +2.5 pp over near2
                        base_hd = 25.0
                        base_rec = 0.945 if organ != "LIVER" else 0.925
                    elif arm == "semgate_f10_k2_r5":
                        base_dice = 0.855  # fails liver recall bar
                        base_hd = 25.0
                        base_rec = 0.945 if organ != "LIVER" else 0.880
                    jitter = float(rng.normal(0.0, 0.001))
                    for post in ("raw", "cc"):
                        rows.append(
                            {
                                "episode_id": p * 4,
                                "fold": 0,
                                "patient_id": pid,
                                "organ": organ,
                                "arm": arm,
                                "post": post,
                                "removed_tp": 0,
                                "removed_fp": 0,
                                "dice": base_dice + jitter,
                                "hd95": base_hd,
                                "recall": base_rec,
                                "detached_fp": 0,
                                "fp": 10,
                                "fn": 10,
                                "empty_prediction": False,
                            }
                        )
        syn = pd.DataFrame(rows)
        est_s = compute_estimands_v2(syn)
        con_s = paired_bootstrap_contrasts_v2(syn, replicates=200, seed=42)
        dec = evaluate_stage1_decision(est_s, con_s)
        self.assertEqual(dec["decision"], "STAGE1_PASS_PROCEED_MULTIDATASET")
        self.assertIn("semgate_pos0_k2_r5", dec["passing_arms"])
        self.assertNotIn("semgate_f10_k2_r5", dec["passing_arms"])


if __name__ == "__main__":
    unittest.main()
