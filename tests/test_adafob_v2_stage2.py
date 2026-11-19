"""CPU unit tests for AdaFoB v2 Stage 2 selectors, query calibration, CBV, and decision rules."""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.adafob_v2_stage2.metrics import (
    compute_estimands_stage2,
    evaluate_stage2_decision,
    paired_bootstrap_contrasts_stage2,
)
from experiments.adafob_v2_stage2.selectors import (
    ARMS,
    compute_equivalent_radius,
    compute_query_anchored_evidence,
    place_pba_scale_adaptive,
    scale_adaptive_offset_px,
    select_best_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE1_CSV = ROOT / "experiments/runs/adafob_v2_stage1_20261119/kaggle_outputs/measurements.csv"


class TestAdaFoBV2Stage2(unittest.TestCase):
    def test_scale_adaptive_radius_and_placement(self) -> None:
        h, w = 257, 257
        yy, xx = np.ogrid[:h, :w]
        # Kidney-sized disk (r = 18 px -> 0.25 * 18 = 4.5 -> clipped to 5.0 px)
        kidney_mask = ((xx - 128.0) ** 2 + (yy - 128.0) ** 2) <= (18.0**2)
        self.assertAlmostEqual(compute_equivalent_radius(kidney_mask), 18.0, delta=0.5)
        self.assertAlmostEqual(scale_adaptive_offset_px(kidney_mask), 5.0, places=5)

        # Liver-sized disk (r = 56 px -> 0.25 * 56 = 14.0 px)
        liver_mask = ((xx - 128.0) ** 2 + (yy - 128.0) ** 2) <= (56.0**2)
        r_off_liver = scale_adaptive_offset_px(liver_mask)
        self.assertGreater(r_off_liver, 13.0)
        self.assertLessEqual(r_off_liver, 16.0)

        img = np.where(liver_mask, 1.0, -1.5).astype(np.float64)
        pts, meta = place_pba_scale_adaptive(liver_mask, img, n_samples=2)
        self.assertIsNotNone(pts)
        assert pts is not None
        self.assertIsNone(meta["fallback"])
        self.assertEqual(pts.shape, (2, 2))
        for x, y in pts:
            d = math.hypot(x - 128.0, y - 128.0)
            self.assertGreaterEqual(d, 56.0 + r_off_liver - 1.5)
            self.assertFalse(bool(liver_mask[int(round(y)), int(round(x))]))

    def test_query_anchored_evidence_and_midpoint_calibration(self) -> None:
        h, w = 257, 257
        fh, fw = 64, 64
        rng = np.random.default_rng(123)

        # Create clean synthetic 3-channel features: channel 0 = organ, channel 1 = tissue bg, channel 2 = air
        qf = np.zeros((3, fh, fw), dtype=np.float64)
        qf[2, :, :] = 1.0  # air outside torso
        qf[:, 12:52, 12:52] = 0.0
        qf[1, 12:52, 12:52] = 1.0  # surrounding tissue
        qf[:, 22:42, 22:42] = 0.0
        qf[0, 22:42, 22:42] = 1.0  # target organ in center

        img = np.full((h, w), -2.0, dtype=np.float64)
        img[48:209, 48:209] = 0.2
        img[88:169, 88:169] = 1.0

        pos0_cc = np.zeros((h, w), dtype=bool)
        pos0_cc[90:167, 90:167] = True
        pos0_logits = np.full((h, w), -3.0, dtype=np.float64)
        pos0_logits[90:167, 90:167] = 1.0
        pos0_logits[104:153, 104:153] = 5.0  # high-confidence interior core

        # Simulate support domain shift where e_local has global negative offset -0.25:
        # organ foreground has e_local ~ -0.05 (erroneously < 0!), tissue bg has e_local ~ -0.55.
        e_local = np.full((h, w), -0.55, dtype=np.float64)
        e_local[88:169, 88:169] = -0.05
        e_local += rng.normal(0.0, 0.01, size=(h, w))

        e_cal, e_qry, e_dual, meta = compute_query_anchored_evidence(
            e_local=e_local,
            qry_feat=qf,
            pos0_cc=pos0_cc,
            pos0_logits_hw=pos0_logits,
            qry_img_hw=img,
        )
        assert e_cal is not None and e_qry is not None and e_dual is not None
        self.assertAlmostEqual(meta["tau_cal"], -0.30, delta=0.05)
        self.assertGreater(float(meta["sep_margin"]), 0.40)
        # After calibration, organ interior becomes strongly positive and adjacent bg stays negative.
        self.assertGreater(float(e_cal[128, 128]), 0.20)
        self.assertLess(float(e_cal[65, 128]), -0.20)
        self.assertGreater(float(e_qry[128, 128]), 0.40)
        self.assertLess(float(e_qry[65, 128]), -0.40)
        self.assertGreater(float(e_dual[128, 128]), 0.30)
        self.assertLess(float(e_dual[65, 128]), -0.30)

        # Verify no-separation guard falls back to tau_cal = 0.0 cleanly.
        e_inverted = -e_local
        _, _, _, meta_inv = compute_query_anchored_evidence(e_inverted, qf, pos0_cc, pos0_logits, img)
        self.assertEqual(meta_inv["cal_note"], "no_sep")
        self.assertEqual(meta_inv["tau_cal"], 0.0)

    def test_counterfactual_boundary_verification_cases(self) -> None:
        h, w = 64, 64
        ev = np.full((h, w), -0.5, dtype=np.float64)
        ev[16:40, 16:40] = 0.8  # True organ core (24x24 = 576 px)

        gt_region = np.zeros((h, w), dtype=bool)
        gt_region[16:40, 16:40] = True

        # Positive0 over-segments 2 leaked lobes (left + top) -> c1 removes top leak, c2 removes left leak
        pos0 = gt_region.copy()
        pos0[8:16, 20:36] = True   # top leak
        pos0[20:36, 8:16] = True   # left leak

        c1_clean = gt_region.copy()
        c1_clean[20:36, 8:16] = True  # only top leak removed

        c2_clean = gt_region.copy()
        c2_clean[8:16, 20:36] = True  # only left leak removed

        # Case 1: Both c1 and c2 help -> c12 (exact GT) must win
        chosen, _ = select_best_candidate(
            {"pos0": pos0, "c1": c1_clean, "c2": c2_clean, "c12": gt_region},
            ev,
            fallback_key="c12",
        )
        self.assertEqual(chosen, "c12")

        # Case 2: c1 removes leak, c2 over-erodes true organ -> c1 (partial k=1) must win!
        c12_eroded = gt_region.copy()
        c12_eroded[16:32, 16:40] = False
        chosen2, _ = select_best_candidate(
            {"pos0": pos0, "c1": c1_clean, "c2": c12_eroded, "c12": c12_eroded},
            ev,
            fallback_key="c12",
        )
        self.assertEqual(chosen2, "c1")

        # Case 3: Pure under-segmentation (no leak, c1/c2/c12 all erode true organ) -> abstain to pos0!
        c1_cut = gt_region.copy()
        c1_cut[16:28, 16:40] = False
        chosen3, _ = select_best_candidate(
            {"pos0": gt_region, "c1": c1_cut, "c2": c12_eroded, "c12": c12_eroded},
            ev,
            fallback_key="c12",
        )
        self.assertEqual(chosen3, "pos0")

        # Case 4: Empty mask guard -> never select an empty mask when non-empty exists
        empty = np.zeros((h, w), dtype=bool)
        ev_all_neg = np.full((h, w), -0.2, dtype=np.float64)
        chosen4, _ = select_best_candidate(
            {"pos0": gt_region, "c12": empty},
            ev_all_neg,
            fallback_key="pos0",
        )
        self.assertEqual(chosen4, "pos0")

    def test_stage2_decision_rules_and_stage1_anchor_load(self) -> None:
        if STAGE1_CSV.is_file():
            df = pd.read_csv(STAGE1_CSV)
            sub = df[df["arm"].isin(["full10", "positive0", "boundary_near2", "pba_f10_int_k2"])].copy()
            est = compute_estimands_stage2(sub)
            cc = est[est["post"] == "cc"].set_index("arm")
            self.assertAlmostEqual(float(cc.loc["pba_f10_int_k2", "patient_dice_pct"]), 84.718990, places=4)

        rows = []
        rng = np.random.default_rng(11)
        for p in range(5):
            pid = f"image_{p}"
            for organ in ("LIVER", "LK", "RK", "SPLEEN"):
                for arm in ARMS:
                    base_dice = 0.82
                    base_rec = 0.94 if organ != "LIVER" else 0.918
                    base_hd = 32.0
                    if arm == "boundary_near2":
                        base_dice = 0.83
                    elif arm == "cbv_f10_scale_dual":
                        base_dice = 0.855
                        base_hd = 27.0
                        base_rec = 0.945 if organ != "LIVER" else 0.920
                    elif arm == "cbv_hybrid_scale_dual":
                        base_dice = 0.850
                        base_hd = 28.0
                        base_rec = 0.935 if organ != "LIVER" else 0.903
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
        est_s = compute_estimands_stage2(syn)
        con_s = paired_bootstrap_contrasts_stage2(syn, replicates=200, seed=42)
        dec = evaluate_stage2_decision(est_s, con_s)
        self.assertEqual(dec["decision"], "STAGE2_PASS_PROCEED_MULTIDATASET")
        self.assertIn("cbv_f10_scale_dual", dec["passing_arms"])
        self.assertIn("cbv_hybrid_scale_dual", dec["pareto_strong_arms"])


if __name__ == "__main__":
    unittest.main()
