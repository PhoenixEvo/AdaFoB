# AdaFoB v2 Dataset & Checkpoint Preparation Guide (MICCAI 2027)

Last updated: 2026-11-19 (branch `miccai2027/adafob-v2`)

## 1. Automated CHAOST2 Preparation Completed on Kaggle (Zero Manual Upload)

We ran a CPU-only Kaggle kernel (`nhatphatnguyen/adafob-prep-chaost2-20261119`, `0.0` GPU-hours, `~170s` runtime) that automatically pulled the raw 20-patient CHAOS MRI dataset from a public Kaggle mirror (`oumaymabejaoui/chaos-ct-mri-data-downloading`) and official FoB checkpoints from HuggingFace (`PrimeBo1/FoB_SAM`). Both are now persistently stored in Kaggle kernel output and ready to attach directly via `kernel_sources` for Stage 3 evaluation.

Verified local audit receipts (full 1.1 GB NIfTI + `.pth` payload retained in Kaggle cloud output):
- Manifest: [prep_manifest.json](experiments/runs/chaost2_prep_20261119_v2/kaggle_outputs/prep_manifest.json)
- Orientation & spacing audit: [alignment_audit.csv](experiments/runs/chaost2_prep_20261119_v2/kaggle_outputs/alignment_audit.csv)
- Slice index: [classmap_1.json](experiments/runs/chaost2_prep_20261119_v2/kaggle_outputs/classmap_1.json)
- Remote log: [adafob-prep-chaost2-20261119.log](experiments/runs/chaost2_prep_20261119_v2/kaggle_outputs/adafob-prep-chaost2-20261119.log)

### Summary of Automated Verification
- **Patients converted:** `20/20` (`{1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 38, 39}`), output layout `CHAOST2/chaos_MR_T2_normalized/{image,label}_{pid}.nii.gz`.
- **DICOM-to-PNG correspondence:** `matched_by_stem = True` across `20/20` scans.
- **Physical orientation & anatomical alignment:** Consistent `best_mode = identity` across all `20/20` scans (`min_in_body_frac = 0.999952`, `min_margin = +0.9501`, `liver_left_of_spleen = True` on all 20 scans).
- **Resampling & ROI crop:** Clipped top `0.5%` intensity, resampled to `[1.25, 1.25, 7.70]` mm spacing, center-cropped axial plane to `[256, 256]`, and verified upstream `third_party/FoB_SAM/dataloaders/datasets.py:TestDataset` instantiation across all 5 folds (`cv0..cv4`, 4 query + 1 support per fold).
- **FoB CHAOST2 Checkpoints:** Extracted `checkpoints_chaost2/exps_train_on_CHAOST2_FSMIS_FoB/FSMIS_train_CHAOST2_cv{0..4}/ckpt.pth` (`218,845,482` bytes each, `688` state_dict keys verified with `torch.load` and SHA256-recorded in `prep_manifest.json`).

### Mandatory Stage-3 Orientation Parity Check (Before Full CHAOST2 Run)
`SimpleITK.ImageSeriesReader` keeps raw DICOM/PNG orientation (`identity`), whereas legacy `dcm2nii` + `png_gth_to_nii.ipynb` flipped Y vertically (`np.flip(..., axis=1)`). If upstream `PrimeBo1/FoB_SAM` CHAOST2 checkpoints were trained on `dcm2nii`-flipped NIfTIs while our clean NIfTIs are unflipped (or vice versa), spatial convolutions / keypoint sorting in `FoB` could see Y-inverted images. Before running full 5-fold Stage 3 on CHAOST2, we MUST run a fast 1-fold (`cv0`) parity check comparing `full10` Dice on:
1. Unflipped `(image, label)` as saved (`identity`), vs.
2. Jointly Y-flipped `(np.flip(image, axis=1), np.flip(label, axis=1))`,
verifying which matches FoB's published ~84.5% Abd-MRI benchmark without any label leakage.

---

## 2. Master Dataset & Checkpoint Reference (Kaggle Sources vs Official Links)

Most required datasets and foundation-model weights are already available directly on Kaggle (zero local download/re-upload required). Official external URLs are also listed below for backup/private mirrors or non-Kaggle use.

| Asset / Role | Ready Kaggle Source (Recommended) | Kaggle Size / Verified Contents | Official External Source / Download Link | Notes & Preprocessing Status |
| :--- | :--- | :--- | :--- | :--- |
| **Abd-CT (`SABS`)** | `nhatphatnguyen/adafob-env` (private dataset) | Pre-normalized CT + FoB SABS `cv0..cv4` + `sam_vit_b` (`3.91 GB`) | Synapse BTCV/SABS: https://www.synapse.org/#!Synapse:syn3193805/wiki/218292 | Already locked & SHA256-verified in Stage 1 / Stage 2 (`image_0..29`). |
| **Abd-MRI (`CHAOST2`)** | `nhatphatnguyen/adafob-prep-chaost2-20261119` (`kernel_sources`) from public `oumaymabejaoui/chaos-ct-mri-data-downloading` | `20/20` pre-processed `.nii.gz` scans + `classmap_1.json` + FoB `cv0..cv4` | Zenodo CHAOS: https://zenodo.org/records/3431873 (Direct zip: https://zenodo.org/record/3431873/files/CHAOS_Train_Sets.zip?download=1) | Auto-prepared on Kaggle; ready for Stage 3 `kernel_sources` mount. |
| **Cardiac MRI (`MS-CMRSeg` / bSSFP)** | `anhoangvo/mscmrseg` (public Kaggle) | `143.6 MB`, `402` files (`MSCMR_preprocessed/`) | Fudan ZMIC: https://zmiclab.github.io/zxh/0/mscmrseg19/ (requires signed DUA for raw) | Verify train/test HDF5 keys & label conventions before evaluation; FoB did not release CMR weights (either zero-shot transfer from CT/MRI or FoB supervoxel training required). |
| **Cardiac MRI (`ACDC`)** | `anhoangvo/acdc-dataset` (public Kaggle) | `299.2 MB`, `2,212` files (`ACDC_preprocessed/`) | Official ACDC: https://www.creatis.insa-lyon.fr/Challenge/acdc/ | Candidate secondary cardiac cohort if needed. |
| **Skin Lesion (`ISIC 2018` Task 1)** | `tschandl/isic2018-challenge-task1-data-segmentation` (public Kaggle) | `13.8 GB` (`Training_Input`, `Validation_Input`, `Test_Input`, `Training_GroundTruth`) | Official ISIC Archive: https://challenge.isic-archive.com/data/#2018 | Requires running `third_party/FoB_SAM/data/isic/{split.py,prepare_setting1_dataset.py}`; HuggingFace `PrimeBo1/FoB_SAM` has not published Skin-DS FoB weights yet. |
| **FoB Pretrained Weights (`SABS` & `CHAOST2`)** | SABS in `nhatphatnguyen/adafob-env`; CHAOST2 in `nhatphatnguyen/adafob-prep-chaost2-20261119` | `~995 MB` (SABS) + `~1.01 GB` (CHAOST2), `cv0..cv4` | HuggingFace: https://huggingface.co/PrimeBo1/FoB_SAM | Both zip archives (`exps_train_on_SABS_FSMIS_FoB.zip` and `exps_train_on_CHAOST2_FSMIS_FoB.zip`) are already hosted on Kaggle. |
| **SAM ViT-B (`sam_vit_b_01ec64.pth`)** | Auto-fetched & SHA256-verified at runtime in internet-enabled Kaggle kernels (`nhatphatnguyen/adafob-env` backup) | `~375 MB` | Official Meta: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth | Used across Step 9, Option A v2, Stage 1, and Stage 2. |
| **SAM ViT-H (`sam_vit_h_4b8939.pth`)** | `simayyamuruysal/sam-vit-h-4b8939-pth` (public Kaggle) | `2.38 GB` (`2,380,884,548` bytes) | Official Meta: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth | Ready on Kaggle for Stage-3 backbone scaling; SHA256 must be verified at runtime before loading. |
| **SAM 2.1 Hiera Large** | `hm0mahmud/sam2-1-hiera-large` (public Kaggle) | `834.5 MB` (`834,506,053` bytes) | Official Meta GitHub / HF (`facebookresearch/sam2`) | Optional backbone scaling after Stage-3 ViT-B & ViT-H completion (`sam2` wheel install needed on Kaggle). |

## 3. Recommended Practical Actions for the Owner

- **Right now (nothing manual needed for SABS-CT & CHAOST2):** Both abdominal datasets, both official FoB 5-fold checkpoint sets, SAM ViT-B, SAM ViT-H, MS-CMRSeg, ACDC, and ISIC 2018 are already accessible inside Kaggle.
- **If you want a private Kaggle backup of raw MS-CMRSeg / Synapse CT external test data:** Check if you want to register at ZMIC (`mscmrseg19`) or Synapse for extra external-CT cohorts (e.g., BTCV vs AMOS22/TotalSegmentator subset) beyond the public Kaggle mirrors listed above.
