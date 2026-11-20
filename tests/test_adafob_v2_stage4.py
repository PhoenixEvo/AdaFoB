"""CPU unit tests for Stage-4 dual-background verification, FoB encoder hook, LPT sharding, and metrics."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.adafob_v2_stage2.selectors import select_best_candidate
from experiments.adafob_v2_stage3_chaost2.dataset import build_chaost2_manifest
from experiments.adafob_v2_stage4.fob_hook import generate_with_fob_prior
from experiments.adafob_v2_stage4.metrics import (
    compute_estimands_stage4,
    evaluate_stage4_decision,
    measure_episode_stage4,
    paired_bootstrap_contrasts_stage4,
)
from experiments.adafob_v2_stage4.package import STEP9_LOCK
from experiments.adafob_v2_stage4.runner import run_stage4_slice, shard_manifest_lpt
from experiments.adafob_v2_stage4.selectors import (
    CBV_ARMS,
    RAW_ARMS,
    compute_dualbg_evidence,
    compute_fob_prob_evidence,
    select_best_candidate_logit_weighted,
)
from experiments.option_a_reassessment_v1.runner import SupportFeatureCache
from experiments.option_a_reassessment_v1.selectors import compute_support_evidence


class _MockEncoder(nn.Module):
    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        fts = torch.zeros((b, 8, 16, 16), dtype=torch.float32, device=x.device)
        # Channel 0 encodes organ core (+1) vs exterior (-1); support & query agree in organ region
        fts[0, 0, 4:8, 4:8] = 2.0
        fts[0, 1, :, :] = 1.0
        fts[1, 0, 5:9, 5:9] = 2.5
        fts[1, 1, :, :] = 0.5
        tao = torch.tensor([[0.1], [-0.4]], dtype=torch.float32, device=x.device)
        return fts, tao


class _MockFoB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _MockEncoder()
        self.scaler = 20.0
        self.feature_dim = 8

    def pre_process(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x.unsqueeze(0), size=(32, 32), mode="nearest").squeeze(0)

    def getFeatures(self, fts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        fts_up = F.interpolate(fts, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sum(fts_up * mask[None, ...], dim=(-2, -1)) / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

    def getPrototype(self, fg_fts):
        return [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) for way in fg_fts]

    def getPred(self, fts: torch.Tensor, prototype: torch.Tensor, thresh: torch.Tensor) -> torch.Tensor:
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        return 1.0 - torch.sigmoid(0.5 * (sim - thresh))

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_labels, train=False, use_skeleton=False, budget_Np=10):
        imgs_concat = torch.cat([supp_imgs[0][0], qry_imgs[0]], dim=0)
        img_fts, _ = self.encoder(imgs_concat)
        neg = np.tile(np.array([[2.0, 3.0]], dtype=np.float32), (10, 1))
        pos = np.array([[10.0, 10.0]], dtype=np.float32)
        return neg, pos


class TestAdaFoBV2Stage4(unittest.TestCase):
    def test_dualbg_blocks_distant_fp_explosion(self) -> None:
        # 16x16 feature grid matching 64x64 native image; support organ sits at [3:7, 3:7].
        c, fh, fw = 4, 16, 16
        h, w = 64, 64
        supp_f = np.zeros((c, fh, fw), dtype=np.float64)
        qry_f = np.zeros((c, fh, fw), dtype=np.float64)
        supp_mask_fg = np.zeros((fh, fw), dtype=bool)
        supp_mask_fg[3:7, 3:7] = True

        # Support organ vector e0 at [3:7, 3:7]; 1..5 ring is [0:12, 0:12] \ [2:8, 2:8].
        supp_f[2, :, :] = 1.0  # broad exterior outside local ring dominated by channel 2
        supp_f[2, 0:12, 0:12] = 0.0
        supp_f[1, 0:12, 0:12] = 1.0  # immediate 1..5 ring around [3:7, 3:7] pure channel 1
        supp_f[0, 3:7, 3:7] = 1.0
        supp_f[1, 3:7, 3:7] = 0.0

        # Query matches organ cleanly at [3:7, 3:7], AND has distant channel 2 + slight channel 0 overlap at [12:16, 12:16].
        qry_f[3, :, :] = 0.05  # strictly nonzero feature norm everywhere
        qry_f[0, 3:7, 3:7] = 1.0
        qry_f[1, 0:12, 0:12] = 0.8
        qry_f[1, 3:7, 3:7] = 0.0
        qry_f[2, 12:16, 12:16] = 1.0
        qry_f[0, 12:16, 12:16] = 0.25  # positive cos(fg, q) ~ 0.24, zero cos(ring, q) -> fooled localring!

        e_local, _ = compute_support_evidence(supp_f, supp_mask_fg, qry_f, (h, w), mode="localring")
        e_dual, _ = compute_dualbg_evidence(supp_f, supp_mask_fg, qry_f, (h, w))
        self.assertIsNotNone(e_local)
        self.assertIsNotNone(e_dual)

        clean_mask = np.zeros((h, w), dtype=bool)
        clean_mask[12:28, 12:28] = True
        exploded_mask = clean_mask.copy()
        exploded_mask[48:64, 48:64] = True  # simulates FP spillover into distant abdominal organ

        choice_local, _ = select_best_candidate({"clean": clean_mask, "exploded": exploded_mask}, e_local, "clean")
        choice_dual, _ = select_best_candidate({"clean": clean_mask, "exploded": exploded_mask}, e_dual, "clean")
        self.assertEqual(choice_local, "exploded")
        self.assertEqual(choice_dual, "clean")

    def test_fob_hook_matches_direct_fob_pred_and_keeps_coords_exact(self) -> None:
        model = _MockFoB().eval()
        supp = np.ones((1, 3, 32, 32), dtype=np.float32)
        qry = np.ones((1, 3, 32, 32), dtype=np.float32) * 0.5
        supp_mask = np.zeros((1, 32, 32), dtype=np.float32)
        supp_mask[:, 8:16, 8:16] = 1.0
        case = {"supp": supp, "query": qry, "support_mask": supp_mask}

        orig_cuda = torch.Tensor.cuda
        cpu_cuda = (lambda self, *args, **kwargs: self) if not torch.cuda.is_available() else orig_cuda
        with patch.object(torch.Tensor, "cuda", cpu_cuda):
            neg, pos, p_fob, f_supp, f_qry = generate_with_fob_prior(model, case, native_hw=(32, 32))

        expected_neg = np.tile(np.array([[2.0, 3.0]], dtype=np.float32), (10, 1))
        expected_pos = np.array([[10.0, 10.0]], dtype=np.float32)
        np.testing.assert_array_equal(neg, expected_neg)
        np.testing.assert_array_equal(pos, expected_pos)
        self.assertEqual(p_fob.shape, (32, 32))
        self.assertTrue(np.isfinite(p_fob).all())
        self.assertGreaterEqual(float(p_fob.min()), 0.0)
        self.assertLessEqual(float(p_fob.max()), 1.0)
        self.assertEqual(f_supp.shape, (8, 16, 16))
        self.assertEqual(f_qry.shape, (8, 16, 16))

        # Verify exact match against direct encoder + getFeatures + getPred calculation.
        with torch.inference_mode():
            enc_fts, enc_tao = model.encoder(torch.cat([torch.from_numpy(supp), torch.from_numpy(qry)], dim=0))
            sm = model.pre_process(torch.from_numpy(supp_mask))
            sf = model.getFeatures(enc_fts[0:1], sm)
            proto = model.getPrototype([[sf]])[0]
            pred = model.getPred(enc_fts[1:2], proto[0], enc_tao[1:])
            p_direct = F.interpolate(pred.unsqueeze(1), size=(32, 32), mode="bilinear", align_corners=True)[0, 0].numpy()
        np.testing.assert_allclose(p_fob, p_direct, atol=1e-6, rtol=0)

    def test_logit_weighted_selection_protects_high_confidence_fg(self) -> None:
        h, w = 32, 32
        p_fob = np.full((h, w), 0.1, dtype=np.float64)
        p_fob[8:20, 8:20] = 0.85
        p_fob[20:24, 8:20] = 0.60  # moderately positive uncertain border strip
        ev = compute_fob_prob_evidence(p_fob)

        # Clean mask retains the true foreground; eroded cuts [20:24] while adding confident exterior spillover.
        clean = np.zeros((h, w), dtype=bool)
        clean[8:24, 8:20] = True
        spilled = np.zeros((h, w), dtype=bool)
        spilled[8:20, 8:20] = True
        spilled[2:8, 22:30] = True  # confident SAM spillover where p_fob == 0.1 (ev == -0.8)

        logits_clean = np.full((h, w), -5.0, dtype=np.float32)
        logits_clean[clean] = 4.0
        logits_spilled = np.full((h, w), -5.0, dtype=np.float32)
        logits_spilled[spilled] = 6.0

        chosen, _ = select_best_candidate_logit_weighted(
            {"clean": clean, "spilled": spilled},
            {"clean": logits_clean, "spilled": logits_spilled},
            ev,
            fallback_key="spilled",
        )
        self.assertEqual(chosen, "clean")

    def test_lpt_sharding_exact_partition_and_balance(self) -> None:
        m_chaos = build_chaost2_manifest()
        rng = np.random.default_rng(42)
        counts_c = {int(eid): int(rng.integers(6, 35)) for eid in m_chaos["episode_id"]}
        w0 = shard_manifest_lpt(m_chaos, counts_c, worker_id=0, num_workers=2)
        w1 = shard_manifest_lpt(m_chaos, counts_c, worker_id=1, num_workers=2)
        ids0 = set(w0["episode_id"].tolist())
        ids1 = set(w1["episode_id"].tolist())
        self.assertEqual(len(ids0 & ids1), 0)
        self.assertEqual(ids0 | ids1, set(m_chaos["episode_id"].tolist()))
        s0 = sum(counts_c[i] for i in ids0)
        s1 = sum(counts_c[i] for i in ids1)
        self.assertLessEqual(abs(s0 - s1), 20)

        m_sabs = pd.read_csv(STEP9_LOCK / "paired_episode_manifest.csv")
        counts_s = {int(eid): int(rng.integers(8, 80)) for eid in m_sabs["episode_id"]}
        sw0 = shard_manifest_lpt(m_sabs, counts_s, worker_id=0, num_workers=2)
        sw1 = shard_manifest_lpt(m_sabs, counts_s, worker_id=1, num_workers=2)
        self.assertEqual(len(set(sw0["episode_id"]) & set(sw1["episode_id"])), 0)
        self.assertEqual(set(sw0["episode_id"]) | set(sw1["episode_id"]), set(m_sabs["episode_id"]))

    def test_stage4_metrics_and_decision_rules(self) -> None:
        gt = np.zeros((4, 32, 32), dtype=bool)
        gt[1:3, 8:24, 8:24] = True
        other = np.zeros_like(gt)

        all_rows = []
        for bb in ("vit_b", "vit_h"):
            for p_idx in range(4):
                pid = f"image_{p_idx + 1}"
                for org in ("LIVER", "SPLEEN"):
                    raw_by_arm = {}
                    cc_by_arm = {}
                    for arm in RAW_ARMS:
                        pred = gt.copy()
                        if arm in ("full10", "fob_coarse_prior"):
                            pred[:, 8:12, 8:24] = False
                        elif arm in ("boundary_near2", "cbv_sam_localring_no0"):
                            pred[:, 8:11, 8:24] = False
                        elif arm == "pba_f10_int_k2":
                            pred[:, 8:10, 8:24] = False
                        elif arm.startswith("random2_s"):
                            pred[:, 8:11, 8:22] = False
                        elif arm == "cbv_fob_dualbg_no0":
                            pass  # wins across both ViT-B and ViT-H
                        raw_by_arm[arm] = pred
                        cc_by_arm[arm] = pred

                    all_rows.extend(
                        measure_episode_stage4(
                            backbone=bb,
                            orientation="flip_y",
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

        df = pd.DataFrame(all_rows)
        est = compute_estimands_stage4(df)
        con = paired_bootstrap_contrasts_stage4(df, replicates=100, seed=42)
        dec = evaluate_stage4_decision(est, con)
        self.assertEqual(dec["decision"], "STAGE4_BACKBONE_INVARIANT_WIN")
        self.assertIn("cbv_fob_dualbg_no0", dec["backbone_invariant_passing"])

    def test_run_stage4_slice_end_to_end_mock(self) -> None:
        class _SliceFoB(_MockFoB):
            def forward(self, supp_imgs, supp_mask, qry_imgs, qry_labels, train=False, use_skeleton=False, budget_Np=10):
                imgs_concat = torch.cat([supp_imgs[0][0], qry_imgs[0]], dim=0)
                _img_fts, _ = self.encoder(imgs_concat)
                angles = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
                neg = np.stack([128.0 + 70.0 * np.cos(angles), 128.0 + 70.0 * np.sin(angles)], axis=1).astype(np.float32)
                pos = np.array([[128.0, 128.0]], dtype=np.float32)
                return neg, pos

        class _MockPredictor:
            def __init__(self) -> None:
                self.model = SimpleNamespace(mask_threshold=0.0)
                self.audit_encoder_calls = 0
                self.audit_decoder_calls = 0
                self.features = None

            def set_image(self, rgb: np.ndarray) -> None:
                self.audit_encoder_calls += 1
                f = torch.zeros((1, 8, 16, 16), dtype=torch.float32)
                f[0, 1, :, :] = 1.0
                f[0, 1, 4:12, 4:12] = 0.0
                f[0, 0, 4:12, 4:12] = 1.0
                self.features = f

            def predict(self, point_coords, point_labels, multimask_output=False, return_logits=True):
                self.audit_decoder_calls += 1
                yy, xx = np.mgrid[0:64, 0:64]
                logits = np.where((yy >= 18) & (yy < 46) & (xx >= 18) & (xx < 46), 4.0, -4.0).astype(np.float32)
                # Slightly shrink/perturb boundary near negative points so subsets produce varied masks.
                neg_pts = point_coords[point_labels == 0]
                if len(neg_pts) == 0:
                    logits[14:18, 22:42] = 1.5
                elif len(neg_pts) == 10:
                    logits[18:20, 18:46] = -1.0
                for pt in neg_pts:
                    dist2 = (xx - pt[0]) ** 2 + (yy - pt[1]) ** 2
                    logits[dist2 < 25.0] = -3.0
                return logits[None, ...], np.array([0.9], dtype=np.float32), None

        qry_img = np.full((1, 3, 64, 64), -0.5, dtype=np.float32)
        qry_img[:, :, 18:46, 18:46] = 1.0
        supp_img = qry_img.copy()
        supp_mask = np.zeros((1, 64, 64), dtype=np.float32)
        supp_mask[:, 18:46, 18:46] = 1.0
        case = {
            "episode_id": 7,
            "fold": 0,
            "patient_id": "image_1",
            "organ": "LIVER",
            "native_z": 4,
            "compact_z": 1,
            "support_patient": "image_3",
            "support_native_z": 2,
            "query": qry_img,
            "supp": supp_img,
            "support_mask": supp_mask,
        }

        model = _SliceFoB().eval()
        predictor = _MockPredictor()
        cache = SupportFeatureCache()
        orig_cuda = torch.Tensor.cuda
        cpu_cuda = (lambda self, *args, **kwargs: self) if not torch.cuda.is_available() else orig_cuda
        with patch.object(torch.Tensor, "cuda", cpu_cuda):
            raw_m, cc_m, aud, coords = run_stage4_slice(model, predictor, cache, case, expected_hw=(64, 64))

        self.assertEqual(set(raw_m.keys()), set(RAW_ARMS))
        self.assertEqual(set(cc_m.keys()), set(RAW_ARMS))
        for arm in RAW_ARMS:
            self.assertEqual(raw_m[arm].shape, (64, 64))
            self.assertEqual(cc_m[arm].shape, (64, 64))
        self.assertEqual(aud["query_encoder_calls"], 1)
        self.assertGreaterEqual(aud["decoder_calls"], 8)
        self.assertLessEqual(aud["decoder_calls"], 11)
        for arm in CBV_ARMS:
            self.assertIn(aud[f"{arm}__choice"], {"pos0", "c1", "c2", "c12"})
        self.assertEqual(coords["p_fob_hw"].shape, (64, 64))


if __name__ == "__main__":
    unittest.main()
