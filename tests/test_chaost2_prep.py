"""CPU unit tests for automated CHAOST2 DICOM/PNG preprocessing and alignment checks."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

from experiments.data_prep.chaost2_prep import (
    map_chaos_png_labels,
    prepare_chaost2_dataset,
    verify_and_align_label_volume,
)


class TestChaost2Prep(unittest.TestCase):
    def test_chaos_png_intensity_mapping_exact(self) -> None:
        raw = np.array([[0, 63, 126], [189, 252, 62]], dtype=np.uint8)
        mapped = map_chaos_png_labels(raw)
        expected = np.array([[0, 1, 2], [3, 4, 1]], dtype=np.int16)
        np.testing.assert_array_equal(mapped, expected)

        with self.assertRaises(ValueError):
            map_chaos_png_labels(np.array([[0, 30]], dtype=np.uint8))

    def test_orientation_alignment_catches_flips(self) -> None:
        z, h, w = 10, 64, 64
        yy, xx = np.ogrid[:h, :w]
        # Build asymmetric upper-body oval shifted toward upper-half rows (y=18..42, x=14..50)
        # so flip_y pushes organs outside the body mask, and make body wider at high z so flip_z hurts edges.
        img = np.zeros((z, h, w), dtype=np.float32)
        lbl_true = np.zeros((z, h, w), dtype=np.int16)
        for zi in range(z):
            cy = 24.0 + 0.3 * zi
            ry = 12.0 + 0.4 * zi
            rx = 20.0
            torso = (((yy - cy) / ry) ** 2 + ((xx - 32.0) / rx) ** 2) <= 1.0
            img[zi, torso] = 100.0
            liv = (((yy - (cy - 2.0)) / 5.0) ** 2 + ((xx - 22.0) / 5.0) ** 2) <= 1.0
            spl = (((yy - (cy + 3.0)) / 4.0) ** 2 + ((xx - 43.0) / 4.0) ** 2) <= 1.0
            img[zi, liv] = 180.0
            img[zi, spl] = 240.0
            if zi >= 2:
                lbl_true[zi, liv] = 1  # LIVER on smaller x
                lbl_true[zi, spl] = 4  # SPLEEN on larger x

        # Corrupt ground truth with BOTH flip_y and flip_z
        lbl_flipped = np.flip(np.flip(lbl_true, axis=0), axis=1).copy()
        recovered, audit = verify_and_align_label_volume(img, lbl_flipped)
        self.assertEqual(audit["best_mode"], "flip_y_and_z")
        self.assertGreater(audit["margin"], 0.05)
        self.assertTrue(audit["liver_left_of_spleen"])
        np.testing.assert_array_equal(recovered, lbl_true)

    def test_end_to_end_chaost2_prep_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            in_root = root / "raw/CHAOS/Train_Sets/MR/1/T2SPIR"
            dcm_dir = in_root / "DICOM_anon"
            gnd_dir = in_root / "Ground"
            dcm_dir.mkdir(parents=True)
            gnd_dir.mkdir(parents=True)

            z, h, w = 8, 128, 128
            yy, xx = np.ogrid[:h, :w]
            vol_u16 = np.zeros((z, h, w), dtype=np.uint16)
            for zi in range(z):
                cy = 48.0 + 1.0 * zi
                torso = (((yy - cy) / 28.0) ** 2 + ((xx - 64.0) / 42.0) ** 2) <= 1.0
                liv = (((yy - (cy - 4.0)) / 10.0) ** 2 + ((xx - 40.0) / 10.0) ** 2) <= 1.0
                spl = (((yy - (cy + 4.0)) / 9.0) ** 2 + ((xx - 90.0) / 8.0) ** 2) <= 1.0
                vol_u16[zi, torso] = 150
                vol_u16[zi, liv] = 280
                vol_u16[zi, spl] = 400
                vol_u16[0, 0, 0] = 5000  # top 0.5% off-resonance spike to clip

                png = np.zeros((h, w), dtype=np.uint8)
                png[liv] = 63
                png[spl] = 252
                Image.fromarray(png).save(gnd_dir / f"IMG-0001-{zi+1:05d}.png")

            # Write valid DICOM series using SimpleITK GDCM
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
            series_uid = "1.2.826.0.1.3680043.2.1125.1"
            for zi in range(z):
                slc = sitk.GetImageFromArray(vol_u16[zi : zi + 1])
                slc.SetSpacing((1.5, 1.5, 8.0))
                slc.SetMetaData("0020|000e", series_uid)
                slc.SetMetaData("0020|0013", str(zi + 1))
                slc.SetMetaData("0020|0032", f"0\\0\\{zi * 8.0:.1f}")
                writer.SetFileName(str(dcm_dir / f"IMG-0001-{zi+1:05d}.dcm"))
                writer.Execute(slc)

            out_dir = root / "out_chaost2"
            summary = prepare_chaost2_dataset(root / "raw", out_dir, expected_pids=("1",))
            self.assertEqual(summary["patients"], 1)
            self.assertEqual(summary["consistent_align_mode"], "identity")

            out_img = sitk.ReadImage(str(out_dir / "chaos_MR_T2_normalized/image_1.nii.gz"))
            out_lbl = sitk.ReadImage(str(out_dir / "chaos_MR_T2_normalized/label_1.nii.gz"))
            self.assertEqual(out_img.GetSize()[:2], (256, 256))
            self.assertEqual(out_lbl.GetSize(), out_img.GetSize())
            np.testing.assert_allclose(out_img.GetSpacing(), [1.25, 1.25, 7.70], atol=1e-5)

            img_arr = sitk.GetArrayFromImage(out_img)
            self.assertLess(float(img_arr.max()), 1000.0)  # off-resonance spike clipped
            cmap = json.loads((out_dir / "chaos_MR_T2_normalized/classmap_1.json").read_text(encoding="utf-8"))
            self.assertGreater(len(cmap["LIVER"]["1"]), 0)
            self.assertTrue((out_dir / "alignment_audit.csv").is_file())


if __name__ == "__main__":
    unittest.main()
