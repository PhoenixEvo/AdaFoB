"""CPU unit tests for Stage-3 CHAOST2 manifest, orientation handling, CBV no0, and metrics."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from experiments.adafob_v2_stage2.selectors import select_best_candidate
from experiments.adafob_v2_stage3_chaost2.dataset import (
    build_chaost2_manifest,
    episode_inputs_chaost2,
    load_chaost2_volume,
)
from experiments.adafob_v2_stage3_chaost2.metrics import (
    compute_estimands_stage3,
    evaluate_stage3_decision,
    measure_episode_stage3,
    paired_bootstrap_contrasts_stage3,
)
from experiments.adafob_v2_stage3_chaost2.selectors import RAW_ARMS


class TestAdaFoBV2Stage3CHAOST2(unittest.TestCase):
    def test_chaost2_manifest_structure(self) -> None:
        df = build_chaost2_manifest()
        self.assertEqual(len(df), 80)
        self.assertEqual(df["patient_id"].nunique(), 15)
        self.assertEqual(df["support_patient"].nunique(), 5)
        self.assertEqual(set(df["organ"].unique()), {"LIVER", "RK", "LK", "SPLEEN"})
        self.assertEqual(
            df.groupby("fold")["support_patient"].first().to_dict(),
            {0: "image_3", 1: "image_13", 2: "image_21", 3: "image_33", 4: "image_37"},
        )
        # Support patient must never appear as query patient inside its own fold.
        self.assertFalse(bool((df["patient_id"] == df["support_patient"]).any()))

    def test_chaost2_volume_orientation_and_episode_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            z, h, w = 6, 256, 256
            for pid, y0 in (("1", 30), ("3", 40)):
                img = np.zeros((z, h, w), dtype=np.float32)
                lbl = np.zeros((z, h, w), dtype=np.int16)
                img[:, y0 : y0 + 20, 50:80] = 120.0
                lbl[1:5, y0 : y0 + 20, 50:80] = 1  # LIVER = 1
                lbl[2:4, 100:115, 160:180] = 4     # SPLEEN = 4
                for arr, prefix in ((img, "image"), (lbl, "label")):
                    obj = sitk.GetImageFromArray(arr)
                    obj.SetSpacing((1.25, 1.25, 7.70))
                    sitk.WriteImage(obj, str(data_dir / f"{prefix}_{pid}.nii.gz"))

            lbl_id, _ = load_chaost2_volume(data_dir, "label_1", is_label=True, orientation="identity")
            lbl_fy, _ = load_chaost2_volume(data_dir, "label_1", is_label=True, orientation="flip_y")
            self.assertEqual(int((lbl_id[2, 30:50, 50:80] == 1).sum()), 600)
            self.assertEqual(int((lbl_fy[2, 256 - 50 : 256 - 30, 50:80] == 1).sum()), 600)

            row = {
                "episode_id": 0,
                "fold": 0,
                "patient_id": "image_1",
                "support_patient": "image_3",
                "organ": "LIVER",
                "spacing_x": 1.25,
                "spacing_y": 1.25,
                "spacing_z": 7.70,
            }
            slices = list(episode_inputs_chaost2(row, data_dir, orientation="identity", support_parts=3))
            self.assertEqual(len(slices), 4)
            self.assertEqual(slices[0]["query"].shape, (1, 3, 256, 256))
            self.assertEqual(slices[0]["support_mask"].shape, (1, 256, 256))
            self.assertEqual(int(slices[1]["other"].sum()), 300)  # SPLEEN present as other organ

    def test_cbv_no0_vs_cbv_fixed_on_spleen_abstention(self) -> None:
        h, w = 32, 32
        ev = np.full((h, w), -0.4, dtype=np.float64)
        ev[8:20, 8:20] = 0.6

        gt = np.zeros((h, w), dtype=bool)
        gt[8:20, 8:20] = True
        pos0 = gt.copy()
        pos0[4:8, 8:20] = True  # leaked lobe in negative evidence

        # Suppose noisy local evidence shifts pos0 slightly higher than c12 when foreground eroded slightly
        c1 = gt.copy()
        c1[8:11, 8:20] = False
        c12 = c1.copy()
        ev[4:8, 8:20] = -0.05
        ev[8:11, 8:20] = 0.4

        k_fixed, _ = select_best_candidate({"pos0": pos0, "c1": c1, "c12": c12}, ev, fallback_key="c12")
        k_no0, _ = select_best_candidate({"c1": c1, "c12": c12}, ev, fallback_key="c12")
        self.assertEqual(k_fixed, "pos0")
        self.assertNotEqual(k_no0, "pos0")

    def test_stage3_metrics_random_mean5_and_backbone_scaling(self) -> None:
        gt = np.zeros((4, 32, 32), dtype=bool)
        gt[1:3, 8:24, 8:24] = True
        other = np.zeros_like(gt)

        all_rows = []
        for bb, bonus in (("vit_b", 0.0), ("vit_h", 1.0)):
            for p_idx in range(4):
                pid = f"image_{p_idx + 1}"
                for org in ("LIVER", "RK"):
                    raw_by_arm = {}
                    cc_by_arm = {}
                    for arm in RAW_ARMS:
                        pred = gt.copy()
                        if arm == "boundary_near2":
                            pred[:, 8:10, 8:24] = False
                        elif arm == "full10":
                            pred[:, 8:11, 8:24] = False
                        elif arm.startswith("random2_s"):
                            pred[:, 8:10, 8:22] = False
                        elif arm == "cbv_f10_fixed_supp" and bonus > 0:
                            pass  # perfect match on ViT-H
                        elif arm == "cbv_f10_fixed_supp":
                            pred[:, 8:9, 8:24] = False
                        raw_by_arm[arm] = pred
                        cc_by_arm[arm] = pred

                    all_rows.extend(
                        measure_episode_stage3(
                            backbone=bb,
                            orientation="identity",
                            episode_id=p_idx * 2 + (0 if org == "LIVER" else 1),
                            fold=0,
                            patient_id=pid,
                            organ=org,
                            spacing_xyz=(1.25, 1.25, 7.70),
                            gt_3d=gt,
                            other_3d=other,
                            raw_by_arm=raw_by_arm,
                            cc_by_arm=cc_by_arm,
                        )
                    )

        df = __import__("pandas").DataFrame(all_rows)
        est = compute_estimands_stage3(df)
        con = paired_bootstrap_contrasts_stage3(df, replicates=100, seed=42)
        dec = evaluate_stage3_decision(est, con)
        self.assertIn("vit_b", con["backbones"])
        self.assertIn("vit_h", con["backbones"])
        cbv_scale = next(x for x in con["backbone_scaling"] if x["arm"] == "cbv_f10_fixed_supp")
        self.assertGreater(cbv_scale["cc_dice"]["point"], 0.0)
        self.assertEqual(len(dec["arms"]), 8)


if __name__ == "__main__":
    unittest.main()
