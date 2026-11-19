"""CPU unit tests verifying Option A selectors, bug fixes, 3D metrics, and Step 9 parity."""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from medpy.metric.binary import hd95 as medpy_hd95

from experiments.option_a_reassessment_v1.metrics import (
    compute_estimands,
    independent_hd95_mm,
    paired_bootstrap_contrasts,
)
from experiments.option_a_reassessment_v1.selectors import (
    compute_support_evidence,
    place_pba_fixed_k2,
    select_local_contrast_near2,
    select_response_arms,
)

ROOT = Path(__file__).resolve().parents[1]
STEP9_CSV = (
    ROOT / "results/step9_controls_20261024T034851Z/cpu_analysis_v3_20261025T024935616Z_1e6a9939/measurements.csv"
)


class TestOptionAReassessment(unittest.TestCase):
    def test_flaw_b3_localring_vs_global_evidence(self) -> None:
        # Build synthetic 64x64 features and 257x257 image with three orthogonal unit embeddings:
        # organ (+1,0,0), adjacent soft-tissue bg (0,+1,0), and dominant extracorporeal air (0,0,+1).
        fh, fw = 64, 64
        nh, nw = 257, 257
        sf = np.zeros((3, fh, fw), dtype=np.float64)
        qf = np.zeros((3, fh, fw), dtype=np.float64)
        # Background air everywhere initially (z-score = -2.0).
        sf[2, :, :] = 1.0
        qf[2, :, :] = 1.0
        s_img = np.full((nh, nw), -2.0, dtype=np.float64)

        # Adjacent abdominal tissue shares moderate channel-0 organ similarity (0.6, 0.8, 0),
        # while extracorporeal CT air dominates the full slice as orthogonal channel 2 (0, 0, 1).
        sf[:, 16:48, 16:48] = 0.0
        sf[0, 16:48, 16:48] = 0.6
        sf[1, 16:48, 16:48] = 0.8
        qf[:, 16:48, 16:48] = 0.0
        qf[0, 16:48, 16:48] = 0.6
        qf[1, 16:48, 16:48] = 0.8
        s_img[64:193, 64:193] = 0.1

        # Organ core at center (1, 0, 0).
        s_mask = np.zeros((fh, fw), dtype=bool)
        s_mask[24:40, 24:40] = True
        sf[:, 24:40, 24:40] = 0.0
        sf[0, 24:40, 24:40] = 1.0
        qf[:, 24:40, 24:40] = 0.0
        qf[0, 24:40, 24:40] = 1.0
        s_img[96:161, 96:161] = 1.0

        e_glob, _ = compute_support_evidence(sf, s_mask, qf, (nh, nw), mode="global")
        e_loc, _ = compute_support_evidence(sf, s_mask, qf, (nh, nw), mode="localring", support_image_hw=s_img)
        assert e_glob is not None and e_loc is not None

        # On adjacent tissue strip inside [64:96, 64:193], global-air bg leaks positive score, whereas localring < 0.
        adj_y, adj_x = 76, 128
        self.assertGreater(float(e_glob[adj_y, adj_x]), 0.0)
        self.assertLess(float(e_loc[adj_y, adj_x]), 0.0)
        self.assertGreater(float(e_loc[128, 128]), 0.0)

    def test_flaw_b2_no_asymmetric_veto_and_adaptive(self) -> None:
        h, w = 64, 64
        full = np.zeros((h, w), dtype=bool)
        full[16:32, 16:32] = True  # true organ area 256

        e_local = np.full((h, w), -0.4, dtype=np.float64)
        e_local[16:32, 16:32] = 0.8
        # Inject tiny positive speck in added region (1% area) to recreate high af > lf vs strongly negative net score.
        e_local[32:34, 16:20] = 0.5

        deletions = np.repeat(full[None, :, :], 10, axis=0)
        # Candidate 3 & 4 strongly suppress negative background (good to keep: added is background, score > 0).
        deletions[3, 32:48, 16:32] = True
        deletions[4, 32:46, 16:32] = True
        # Candidate 6 & 7 falsely delete adjacent background or suppress true foreground (score < 0).
        e_local[12:16, 16:20] = 0.8
        deletions[6, 12:16, 16:20] = True
        deletions[7, 13:16, 16:20] = True

        out = select_response_arms(full, deletions, e_global=e_local, e_local=e_local, fallback_ids=[0, 1])
        self.assertEqual(out["resp_signed_localring_k2"], [3, 4])
        self.assertEqual(out["resp_signed_global_k2"], [3, 4])
        self.assertEqual(out["resp_signed_adaptive"], [3, 4])

        # Verify adaptive returns empty (k=0) when all scores are <= 0.
        e_all_pos = np.full((h, w), 0.5, dtype=np.float64)
        out_zero = select_response_arms(full, deletions, e_global=e_all_pos, e_local=e_all_pos, fallback_ids=[0, 1])
        self.assertEqual(out_zero["resp_signed_adaptive"], [])
        self.assertEqual(out_zero["notes"]["resp_signed_adaptive"], "zero_negatives")

    def test_bug_a3_local_contrast_near2(self) -> None:
        h, w = 100, 100
        full = np.zeros((h, w), dtype=bool)
        full[40:60, 40:60] = True
        img = np.full((h, w), -2.0, dtype=np.float64)
        img[40:60, 40:60] = 1.0  # organ intensity ~ 1.0

        # Candidate points equidistant from mask boundary (~3 px away):
        # idx 0, 1 sit on iso-intense ambiguous tissue (I=0.95 -> high l_s);
        # idx 2..9 sit on sharp air boundary (I=-2.0 -> low l_s) or farther away.
        pts = np.array(
            [
                [37.0, 50.0],
                [63.0, 50.0],
                [50.0, 37.0],
                [50.0, 63.0],
                [10.0, 10.0],
                [90.0, 90.0],
                [10.0, 90.0],
                [90.0, 10.0],
                [20.0, 50.0],
                [80.0, 50.0],
            ],
            dtype=np.float64,
        )
        img[45:55, 34:40] = 0.95
        img[45:55, 60:66] = 0.95

        sel, meta = select_local_contrast_near2(pts, full, img)
        self.assertEqual(sorted(sel), [0, 1])
        self.assertGreater(meta["l_s"][0], 0.9)
        self.assertLess(meta["l_s"][2], 0.05)

    def test_bug_a1_pba_fixed_k2_spacing_and_outward(self) -> None:
        h, w = 257, 257
        yy, xx = np.ogrid[:h, :w]
        cx, cy, r = 128.0, 128.0, 40.0
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r**2)
        img = np.where(mask, 1.0, -1.5).astype(np.float64)

        pts, meta = place_pba_fixed_k2(mask, img, n_samples=2, offset_px=5.0)
        self.assertIsNotNone(pts)
        assert pts is not None
        self.assertEqual(pts.shape, (2, 2))
        self.assertIsNone(meta["fallback"])

        # Both placed points must lie outside mask and inside [0, w-1] x [0, h-1].
        for x, y in pts:
            self.assertFalse(bool(mask[int(round(y)), int(round(x))]))
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, w - 1.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, h - 1.0)

        # Angular separation around center must be ~180 deg (not Bug A1's 13.5 deg cluster!).
        v0 = pts[0] - np.array([cx, cy])
        v1 = pts[1] - np.array([cx, cy])
        cos_ang = np.clip(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)), -1.0, 1.0)
        ang_deg = float(np.degrees(np.arccos(cos_ang)))
        self.assertGreater(ang_deg, 150.0)

    def test_hd95_matches_medpy_and_step9_parity(self) -> None:
        rng = np.random.default_rng(42)
        gt = np.zeros((16, 48, 48), dtype=bool)
        pred = np.zeros((16, 48, 48), dtype=bool)
        gt[4:12, 14:30, 16:32] = True
        pred[5:13, 15:31, 14:30] = True
        # Add noise protrusion.
        pred[7:9, 31:35, 20:24] = True
        spacing_xyz = (1.2, 1.2, 3.0)
        spacing_zyx = spacing_xyz[::-1]

        hd_ck, _, _ = independent_hd95_mm(pred, gt, spacing_zyx)
        hd_medpy = float(medpy_hd95(pred, gt, voxelspacing=spacing_zyx))
        self.assertLess(abs(hd_ck - hd_medpy), 1e-6)

        if STEP9_CSV.is_file():
            df = pd.read_csv(STEP9_CSV)
            anchors = ("full10", "positive0", "boundary_near2")
            sub = df[df["arm"].isin(anchors)].copy()
            est = compute_estimands(sub)
            cc = est[est["post"] == "cc"].set_index("arm")
            self.assertAlmostEqual(float(cc.loc["full10", "patient_dice_pct"]), 81.821210, places=5)
            self.assertAlmostEqual(float(cc.loc["positive0", "patient_dice_pct"]), 80.640488, places=5)
            self.assertAlmostEqual(float(cc.loc["boundary_near2", "patient_dice_pct"]), 83.164001, places=5)

            con = paired_bootstrap_contrasts(sub, baselines=("full10",), replicates=5000, seed=42)
            near_row = next(c for c in con["contrasts"] if c["arm"] == "boundary_near2" and c["baseline"] == "full10")
            self.assertAlmostEqual(float(near_row["cc_dice"]["point"]), 1.342792, places=5)
            self.assertAlmostEqual(float(near_row["cc_dice"]["ci95"][0]), 0.637372, places=5)
            self.assertAlmostEqual(float(near_row["cc_dice"]["ci95"][1]), 2.104989, places=5)


if __name__ == "__main__":
    unittest.main()
