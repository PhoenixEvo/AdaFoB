# AdaFoB v2 Stage 4 Results: CHAOST2 (Abd-MRI) Multi-Backbone CBV Verification (Setting I)

- **Run Directory:** `experiments\runs\adafob_v2_stage4_chaost2_20261119_v2`
- **Kaggle Kernel Status:** `COMPLETED` (`episodes_per_backbone=80`, `patients=15`, `slices=1354`, `aggregate_gpu_hours=1.9232`, `wall=3461.8s`)
- **Anchor Parity vs Prior Locked Stage:** `max_abs_diff_pp=0.0000` (`checked_rows=28`)
- **Pre-Registered Stage-4 Decision:** **`STAGE4_BACKBONE_INVARIANT_WIN`** (`passing_vit_b=['cbv_sam_localring_all', 'cbv_sam_localring_no0', 'cbv_sam_dualbg_all', 'cbv_sam_dualbg_no0', 'cbv_fob_prob_all', 'cbv_fob_prob_no0', 'cbv_fob_dualbg_all', 'cbv_fob_dualbg_no0', 'cbv_joint_prob_sam_all', 'cbv_joint_prob_sam_no0', 'cbv_logit_fob_prob_all', 'cbv_logit_fob_prob_no0']`, `passing_vit_h=['cbv_fob_prob_all', 'cbv_fob_prob_no0', 'cbv_fob_dualbg_no0', 'cbv_joint_prob_sam_all', 'cbv_joint_prob_sam_no0', 'cbv_logit_fob_prob_all', 'cbv_logit_fob_prob_no0']`, `backbone_invariant_passing=['cbv_fob_prob_all', 'cbv_fob_prob_no0', 'cbv_fob_dualbg_no0', 'cbv_joint_prob_sam_all', 'cbv_joint_prob_sam_no0', 'cbv_logit_fob_prob_all', 'cbv_logit_fob_prob_no0']`)

## Patient-Balanced 3D Metrics (`vit_b`, 80 Episodes / 15 Patients / 1354 Slices)

| Arm | Pat Raw Dice (%) | Pat CC Dice (%) [SD] | Ep CC Dice (%) | Pat HD95 (mm) | Pat Recall (%) | Detached FP / pat |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fob_coarse_prior` | 82.95 | 84.26 [7.02] | 84.22 | 19.24 | 85.94 | 956.6 |
| `full10` | 77.76 | 80.22 [6.24] | 80.43 | 44.56 | 92.92 | 4195.3 |
| `positive0` | 78.61 | 79.09 [8.84] | 79.47 | 49.44 | 94.75 | 2097.1 |
| `random2_mean5` | 80.64 | 81.26 [6.76] | 81.61 | 41.34 | 93.16 | 2342.8 |
| `boundary_near2` | 82.29 | 82.74 [6.51] | 82.92 | 36.65 | 92.45 | 1950.6 |
| `pba_f10_int_k2` | 83.24 | 83.57 [6.43] | 83.78 | 34.65 | 91.72 | 1922.2 |
| `cbv_sam_localring_all` | 83.25 | 83.50 [6.01] | 83.60 | 34.38 | 92.17 | 1769.6 |
| `cbv_sam_localring_no0` | 83.49 | 83.74 [6.11] | 83.93 | 33.50 | 91.98 | 1725.9 |
| `cbv_sam_dualbg_all` | 83.29 | 83.54 [5.97] | 83.65 | 34.19 | 92.06 | 1786.4 |
| `cbv_sam_dualbg_no0` | 83.54 | 83.79 [6.09] | 83.97 | 33.37 | 91.90 | 1745.8 |
| `cbv_fob_prob_all` | 83.90 | 84.19 [5.99] | 84.44 | 32.30 | 92.61 | 1666.4 |
| `cbv_fob_prob_no0` | 83.94 | 84.23 [6.08] | 84.46 | 32.27 | 92.24 | 1660.6 |
| `cbv_fob_dualbg_all` | 83.20 | 83.50 [5.87] | 83.65 | 35.37 | 93.29 | 1787.3 |
| `cbv_fob_dualbg_no0` | 83.56 | 83.84 [6.12] | 84.07 | 33.96 | 92.70 | 1741.5 |
| `cbv_joint_prob_sam_all` | 83.90 | 84.19 [5.99] | 84.44 | 32.30 | 92.59 | 1667.0 |
| `cbv_joint_prob_sam_no0` | 83.94 | 84.23 [6.08] | 84.45 | 32.28 | 92.24 | 1660.5 |
| `cbv_logit_fob_prob_all` | 83.89 | 84.19 [6.02] | 84.42 | 32.33 | 92.75 | 1684.9 |
| `cbv_logit_fob_prob_no0` | 83.92 | 84.21 [6.09] | 84.43 | 32.32 | 92.36 | 1669.9 |

### Per-Organ Patient-Balanced Breakdown (`vit_b`, `CC Dice % / CC Recall % / CC HD95 mm`)

| Arm | LIVER | RK | LK | SPLEEN |
| :--- | :---: | :---: | :---: | :---: |
| `fob_coarse_prior` | 83.60 / 84.51 / 25.2 | 90.33 / 89.68 / 6.9 | 86.48 / 87.87 / 18.8 | 76.61 / 81.71 / 26.0 |
| `full10` | 71.66 / 89.65 / 63.4 | 87.88 / 95.60 / 26.3 | 82.63 / 95.72 / 39.3 | 78.70 / 90.72 / 49.2 |
| `positive0` | 76.52 / 93.24 / 55.0 | 85.52 / 95.94 / 34.3 | 80.55 / 95.99 / 46.4 | 73.75 / 93.84 / 62.2 |
| `random2_mean5` | 77.20 / 89.84 / 48.2 | 87.54 / 95.72 / 28.0 | 83.44 / 95.65 / 37.5 | 76.87 / 91.42 / 51.7 |
| `boundary_near2` | 80.16 / 88.30 / 40.1 | 88.52 / 95.52 / 23.4 | 84.21 / 95.32 / 35.3 | 78.07 / 90.69 / 47.8 |
| `pba_f10_int_k2` | 81.44 / 86.76 / 37.0 | 89.58 / 94.78 / 20.5 | 84.22 / 94.78 / 34.5 | 79.03 / 90.57 / 46.6 |
| `cbv_sam_localring_all` | 81.75 / 88.00 / 36.1 | 90.05 / 95.11 / 17.8 | 84.01 / 95.24 / 36.0 | 78.18 / 90.35 / 47.6 |
| `cbv_sam_localring_no0` | 81.94 / 87.53 / 34.9 | 90.07 / 95.02 / 18.0 | 84.21 / 95.07 / 35.1 | 78.74 / 90.29 / 45.9 |
| `cbv_sam_dualbg_all` | 81.96 / 87.75 / 35.3 | 90.00 / 95.01 / 17.9 | 84.03 / 95.20 / 36.0 | 78.18 / 90.28 / 47.6 |
| `cbv_sam_dualbg_no0` | 82.10 / 87.30 / 34.5 | 90.07 / 95.01 / 18.0 | 84.23 / 95.03 / 35.1 | 78.78 / 90.26 / 45.9 |
| `cbv_fob_prob_all` | 83.37 / 89.09 / 33.5 | 89.98 / 95.42 / 17.4 | 84.39 / 95.28 / 34.2 | 79.01 / 90.64 / 44.1 |
| `cbv_fob_prob_no0` | 83.18 / 88.23 / 33.6 | 89.99 / 95.16 / 18.0 | 84.55 / 95.11 / 33.6 | 79.20 / 90.47 / 43.9 |
| `cbv_fob_dualbg_all` | 83.23 / 91.14 / 34.7 | 89.59 / 95.52 / 19.2 | 83.14 / 95.63 / 39.3 | 78.06 / 90.86 / 48.4 |
| `cbv_fob_dualbg_no0` | 83.21 / 89.57 / 34.0 | 89.66 / 95.27 / 19.3 | 83.96 / 95.37 / 36.0 | 78.56 / 90.61 / 46.6 |
| `cbv_joint_prob_sam_all` | 83.38 / 89.07 / 33.5 | 89.98 / 95.41 / 17.4 | 84.38 / 95.26 / 34.2 | 79.01 / 90.61 / 44.1 |
| `cbv_joint_prob_sam_no0` | 83.19 / 88.22 / 33.5 | 89.99 / 95.16 / 18.1 | 84.54 / 95.11 / 33.6 | 79.20 / 90.46 / 43.9 |
| `cbv_logit_fob_prob_all` | 83.46 / 89.52 / 33.5 | 90.01 / 95.50 / 17.4 | 84.33 / 95.28 / 34.3 | 78.95 / 90.68 / 44.1 |
| `cbv_logit_fob_prob_no0` | 83.21 / 88.61 / 33.7 | 90.01 / 95.23 / 18.0 | 84.50 / 95.08 / 33.7 | 79.14 / 90.51 / 43.9 |

## Patient-Balanced 3D Metrics (`vit_h`, 80 Episodes / 15 Patients / 1354 Slices)

| Arm | Pat Raw Dice (%) | Pat CC Dice (%) [SD] | Ep CC Dice (%) | Pat HD95 (mm) | Pat Recall (%) | Detached FP / pat |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fob_coarse_prior` | 82.95 | 84.26 [7.02] | 84.22 | 19.24 | 85.94 | 956.6 |
| `full10` | 77.28 | 79.62 [7.50] | 79.76 | 45.28 | 93.42 | 3340.5 |
| `positive0` | 72.89 | 73.25 [12.13] | 73.84 | 70.02 | 94.57 | 1782.9 |
| `random2_mean5` | 78.54 | 79.19 [7.51] | 79.64 | 52.36 | 93.74 | 2012.1 |
| `boundary_near2` | 81.60 | 82.18 [6.45] | 82.48 | 42.51 | 92.71 | 2069.1 |
| `pba_f10_int_k2` | 81.91 | 82.35 [7.02] | 82.86 | 40.37 | 92.55 | 1993.9 |
| `cbv_sam_localring_all` | 81.45 | 81.76 [7.09] | 82.39 | 43.17 | 92.85 | 1902.7 |
| `cbv_sam_localring_no0` | 81.67 | 82.00 [7.18] | 82.61 | 42.52 | 92.77 | 1861.3 |
| `cbv_sam_dualbg_all` | 81.64 | 81.98 [7.09] | 82.56 | 42.08 | 92.78 | 1980.2 |
| `cbv_sam_dualbg_no0` | 81.85 | 82.20 [7.17] | 82.78 | 41.40 | 92.71 | 1938.2 |
| `cbv_fob_prob_all` | 82.51 | 82.94 [7.18] | 83.40 | 38.75 | 92.91 | 1901.1 |
| `cbv_fob_prob_no0` | 82.52 | 82.93 [7.16] | 83.40 | 39.00 | 92.76 | 1883.8 |
| `cbv_fob_dualbg_all` | 81.68 | 82.10 [7.15] | 82.68 | 41.75 | 93.41 | 1990.1 |
| `cbv_fob_dualbg_no0` | 81.86 | 82.28 [7.30] | 82.86 | 41.14 | 93.13 | 1955.2 |
| `cbv_joint_prob_sam_all` | 82.50 | 82.90 [7.11] | 83.37 | 38.93 | 92.93 | 1897.8 |
| `cbv_joint_prob_sam_no0` | 82.52 | 82.93 [7.17] | 83.40 | 38.99 | 92.75 | 1882.9 |
| `cbv_logit_fob_prob_all` | 82.48 | 82.92 [7.16] | 83.38 | 38.78 | 93.02 | 1895.3 |
| `cbv_logit_fob_prob_no0` | 82.47 | 82.91 [7.14] | 83.38 | 39.01 | 92.85 | 1902.4 |

### Per-Organ Patient-Balanced Breakdown (`vit_h`, `CC Dice % / CC Recall % / CC HD95 mm`)

| Arm | LIVER | RK | LK | SPLEEN |
| :--- | :---: | :---: | :---: | :---: |
| `fob_coarse_prior` | 83.60 / 84.51 / 25.2 | 90.33 / 89.68 / 6.9 | 86.48 / 87.87 / 18.8 | 76.61 / 81.71 / 26.0 |
| `full10` | 77.72 / 91.60 / 55.3 | 86.48 / 95.26 / 31.9 | 81.19 / 95.89 / 42.9 | 73.10 / 90.93 / 51.0 |
| `positive0` | 74.59 / 93.47 / 76.5 | 75.51 / 95.33 / 66.1 | 75.64 / 95.19 / 59.4 | 67.27 / 94.28 / 78.1 |
| `random2_mean5` | 78.50 / 92.15 / 60.5 | 84.18 / 95.31 / 38.6 | 81.25 / 95.29 / 45.7 | 72.84 / 92.20 / 64.7 |
| `boundary_near2` | 82.45 / 88.85 / 43.0 | 87.11 / 95.20 / 29.4 | 84.10 / 95.01 / 36.3 | 75.05 / 91.78 / 61.3 |
| `pba_f10_int_k2` | 82.97 / 89.09 / 40.0 | 87.62 / 94.88 / 27.6 | 83.31 / 94.33 / 36.9 | 75.50 / 91.90 / 57.0 |
| `cbv_sam_localring_all` | 82.11 / 90.00 / 45.9 | 87.58 / 94.79 / 27.3 | 83.14 / 94.60 / 39.5 | 74.24 / 92.02 / 59.9 |
| `cbv_sam_localring_no0` | 82.31 / 89.79 / 45.6 | 87.57 / 94.77 / 27.4 | 83.24 / 94.56 / 39.3 | 74.86 / 91.95 / 57.8 |
| `cbv_sam_dualbg_all` | 82.61 / 89.82 / 42.5 | 87.58 / 94.77 / 27.3 | 83.12 / 94.52 / 39.5 | 74.60 / 92.01 / 59.0 |
| `cbv_sam_dualbg_no0` | 82.76 / 89.65 / 42.2 | 87.58 / 94.76 / 27.4 | 83.23 / 94.48 / 39.2 | 75.24 / 91.94 / 56.8 |
| `cbv_fob_prob_all` | 84.10 / 90.34 / 38.2 | 87.68 / 95.07 / 27.3 | 83.98 / 94.50 / 35.4 | 76.00 / 91.74 / 54.1 |
| `cbv_fob_prob_no0` | 83.93 / 89.84 / 38.4 | 87.68 / 94.94 / 27.3 | 84.04 / 94.46 / 35.1 | 76.08 / 91.81 / 55.2 |
| `cbv_fob_dualbg_all` | 83.61 / 91.59 / 39.5 | 87.33 / 95.12 / 28.6 | 83.02 / 94.84 / 40.4 | 74.43 / 92.11 / 58.5 |
| `cbv_fob_dualbg_no0` | 83.63 / 90.92 / 39.6 | 87.35 / 94.99 / 28.6 | 83.04 / 94.68 / 40.0 | 75.12 / 91.94 / 56.4 |
| `cbv_joint_prob_sam_all` | 84.13 / 90.36 / 38.2 | 87.69 / 95.04 / 27.3 | 83.81 / 94.50 / 36.1 | 75.96 / 91.79 / 54.1 |
| `cbv_joint_prob_sam_no0` | 83.93 / 89.83 / 38.4 | 87.67 / 94.92 / 27.3 | 84.04 / 94.45 / 35.1 | 76.08 / 91.80 / 55.2 |
| `cbv_logit_fob_prob_all` | 84.08 / 90.60 / 38.3 | 87.66 / 95.09 / 27.3 | 83.95 / 94.59 / 35.5 | 75.97 / 91.80 / 54.0 |
| `cbv_logit_fob_prob_no0` | 83.92 / 90.11 / 38.4 | 87.66 / 94.94 / 27.3 | 84.02 / 94.49 / 35.1 | 76.04 / 91.85 / 55.2 |

## Paired Patient-Cluster Bootstrap Contrasts (5,000 Replicates, `seed=42`)

| Backbone | Arm | vs `near2` (pp) [95% CI] | vs `full10` (pp) [95% CI] | vs `pba_f10` (pp) [95% CI] | vs `legacy_cbv_no0` (pp) [95% CI] | Liver Rec vs `pba` (pp) | Passes BB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| `vit_b` | `pba_f10_int_k2` | 0.83 [0.17, 1.45] | 3.35 [2.51, 4.18] | N/A | -0.17 [-0.72, 0.34] | 0.00 | `True` |
| `vit_b` | `cbv_sam_localring_all` | 0.76 [0.32, 1.22] | 3.28 [2.38, 4.19] | -0.07 [-0.74, 0.62] | -0.24 [-0.50, -0.03] | 1.24 | `True` |
| `vit_b` | `cbv_sam_localring_no0` | 1.00 [0.59, 1.44] | 3.52 [2.71, 4.37] | 0.17 [-0.34, 0.72] | 0.00 [0.00, 0.00] | 0.77 | `True` |
| `vit_b` | `cbv_sam_dualbg_all` | 0.80 [0.35, 1.30] | 3.33 [2.43, 4.25] | -0.02 [-0.67, 0.64] | -0.20 [-0.45, 0.00] | 1.00 | `True` |
| `vit_b` | `cbv_sam_dualbg_no0` | 1.05 [0.63, 1.50] | 3.58 [2.76, 4.42] | 0.22 [-0.29, 0.77] | 0.05 [0.01, 0.10] | 0.55 | `True` |
| `vit_b` | `cbv_fob_prob_all` | 1.45 [0.99, 1.94] | 3.97 [3.15, 4.79] | 0.62 [0.09, 1.29] | 0.45 [0.20, 0.71] | 2.33 | `True` |
| `vit_b` | `cbv_fob_prob_no0` | 1.49 [1.05, 1.97] | 4.01 [3.24, 4.82] | 0.66 [0.21, 1.22] | 0.49 [0.30, 0.70] | 1.48 | `True` |
| `vit_b` | `cbv_fob_dualbg_all` | 0.76 [0.18, 1.36] | 3.29 [2.22, 4.32] | -0.06 [-0.88, 0.87] | -0.24 [-0.72, 0.23] | 4.39 | `True` |
| `vit_b` | `cbv_fob_dualbg_no0` | 1.11 [0.71, 1.55] | 3.63 [2.75, 4.51] | 0.28 [-0.29, 0.96] | 0.11 [-0.14, 0.33] | 2.81 | `True` |
| `vit_b` | `cbv_joint_prob_sam_all` | 1.45 [1.00, 1.94] | 3.97 [3.15, 4.79] | 0.62 [0.09, 1.30] | 0.45 [0.20, 0.71] | 2.32 | `True` |
| `vit_b` | `cbv_joint_prob_sam_no0` | 1.49 [1.05, 1.97] | 4.01 [3.24, 4.81] | 0.66 [0.21, 1.22] | 0.49 [0.30, 0.70] | 1.47 | `True` |
| `vit_b` | `cbv_logit_fob_prob_all` | 1.45 [1.01, 1.92] | 3.97 [3.14, 4.81] | 0.62 [0.07, 1.31] | 0.45 [0.19, 0.71] | 2.77 | `True` |
| `vit_b` | `cbv_logit_fob_prob_no0` | 1.47 [1.05, 1.94] | 4.00 [3.22, 4.79] | 0.65 [0.18, 1.21] | 0.47 [0.28, 0.69] | 1.85 | `True` |
| `vit_h` | `pba_f10_int_k2` | 0.17 [-0.56, 0.93] | 2.73 [1.27, 4.20] | N/A | 0.35 [-0.20, 1.17] | 0.00 | `False` |
| `vit_h` | `cbv_sam_localring_all` | -0.41 [-1.17, 0.36] | 2.14 [0.87, 3.55] | -0.59 [-1.37, 0.00] | -0.23 [-0.38, -0.12] | 0.90 | `False` |
| `vit_h` | `cbv_sam_localring_no0` | -0.18 [-0.99, 0.63] | 2.38 [1.08, 3.79] | -0.35 [-1.17, 0.20] | 0.00 [0.00, 0.00] | 0.70 | `False` |
| `vit_h` | `cbv_sam_dualbg_all` | -0.20 [-0.93, 0.54] | 2.36 [1.04, 3.79] | -0.37 [-1.06, 0.15] | -0.02 [-0.24, 0.21] | 0.73 | `False` |
| `vit_h` | `cbv_sam_dualbg_no0` | 0.02 [-0.76, 0.80] | 2.58 [1.24, 4.03] | -0.15 [-0.87, 0.35] | 0.20 [0.06, 0.38] | 0.56 | `False` |
| `vit_h` | `cbv_fob_prob_all` | 0.76 [0.05, 1.52] | 3.32 [1.83, 4.85] | 0.59 [0.33, 0.88] | 0.95 [0.41, 1.77] | 1.25 | `True` |
| `vit_h` | `cbv_fob_prob_no0` | 0.75 [0.07, 1.47] | 3.31 [1.84, 4.80] | 0.58 [0.33, 0.86] | 0.93 [0.43, 1.72] | 0.75 | `True` |
| `vit_h` | `cbv_fob_dualbg_all` | -0.08 [-0.87, 0.67] | 2.47 [1.15, 3.90] | -0.25 [-0.92, 0.26] | 0.10 [-0.20, 0.42] | 2.49 | `False` |
| `vit_h` | `cbv_fob_dualbg_no0` | 0.11 [-0.72, 0.91] | 2.66 [1.33, 4.11] | -0.07 [-0.75, 0.42] | 0.29 [0.02, 0.57] | 1.82 | `True` |
| `vit_h` | `cbv_joint_prob_sam_all` | 0.72 [0.04, 1.43] | 3.28 [1.81, 4.80] | 0.55 [0.31, 0.77] | 0.90 [0.39, 1.72] | 1.27 | `True` |
| `vit_h` | `cbv_joint_prob_sam_no0` | 0.75 [0.07, 1.47] | 3.31 [1.85, 4.80] | 0.58 [0.33, 0.86] | 0.93 [0.43, 1.72] | 0.74 | `True` |
| `vit_h` | `cbv_logit_fob_prob_all` | 0.74 [0.02, 1.49] | 3.29 [1.80, 4.81] | 0.57 [0.32, 0.85] | 0.92 [0.40, 1.73] | 1.51 | `True` |
| `vit_h` | `cbv_logit_fob_prob_no0` | 0.73 [0.04, 1.44] | 3.29 [1.81, 4.78] | 0.56 [0.32, 0.82] | 0.91 [0.41, 1.71] | 1.01 | `True` |

## CBV Candidate Selection Audit

- `vit_b`: mean decoder calls/slice `10.83`, `pba` fallbacks `0`, mean FoB prior `0.0507`
  - `cbv_sam_localring_all`: mean `k=1.341`, choices `{'c1': 303, 'c12': 585, 'c2': 343, 'pos0': 123}`
  - `cbv_sam_localring_no0`: mean `k=1.442`, choices `{'c1': 350, 'c12': 598, 'c2': 406}`
  - `cbv_sam_dualbg_all`: mean `k=1.354`, choices `{'c1': 295, 'c12': 596, 'c2': 346, 'pos0': 117}`
  - `cbv_sam_dualbg_no0`: mean `k=1.453`, choices `{'c1': 340, 'c12': 614, 'c2': 400}`
  - `cbv_fob_prob_all`: mean `k=1.165`, choices `{'c1': 346, 'c12': 465, 'c2': 302, 'pos0': 241}`
  - `cbv_fob_prob_no0`: mean `k=1.358`, choices `{'c1': 452, 'c12': 485, 'c2': 417}`
  - `cbv_fob_dualbg_all`: mean `k=1.053`, choices `{'c1': 341, 'c12': 397, 'c2': 291, 'pos0': 325}`
  - `cbv_fob_dualbg_no0`: mean `k=1.313`, choices `{'c1': 489, 'c12': 424, 'c2': 441}`
  - `cbv_joint_prob_sam_all`: mean `k=1.191`, choices `{'c1': 346, 'c12': 483, 'c2': 301, 'pos0': 224}`
  - `cbv_joint_prob_sam_no0`: mean `k=1.369`, choices `{'c1': 443, 'c12': 499, 'c2': 412}`
  - `cbv_logit_fob_prob_all`: mean `k=1.148`, choices `{'c1': 350, 'c12': 447, 'c2': 310, 'pos0': 247}`
  - `cbv_logit_fob_prob_no0`: mean `k=1.343`, choices `{'c1': 456, 'c12': 465, 'c2': 433}`
- `vit_h`: mean decoder calls/slice `10.82`, `pba` fallbacks `0`, mean FoB prior `0.0507`
  - `cbv_sam_localring_all`: mean `k=1.229`, choices `{'c1': 315, 'c12': 509, 'c2': 331, 'pos0': 199}`
  - `cbv_sam_localring_no0`: mean `k=1.386`, choices `{'c1': 416, 'c12': 522, 'c2': 416}`
  - `cbv_sam_dualbg_all`: mean `k=1.242`, choices `{'c1': 313, 'c12': 517, 'c2': 335, 'pos0': 189}`
  - `cbv_sam_dualbg_no0`: mean `k=1.391`, choices `{'c1': 408, 'c12': 529, 'c2': 417}`
  - `cbv_fob_prob_all`: mean `k=1.123`, choices `{'c1': 302, 'c12': 437, 'c2': 344, 'pos0': 271}`
  - `cbv_fob_prob_no0`: mean `k=1.341`, choices `{'c1': 441, 'c12': 462, 'c2': 451}`
  - `cbv_fob_dualbg_all`: mean `k=0.945`, choices `{'c1': 321, 'c12': 329, 'c2': 301, 'pos0': 403}`
  - `cbv_fob_dualbg_no0`: mean `k=1.261`, choices `{'c1': 541, 'c12': 354, 'c2': 459}`
  - `cbv_joint_prob_sam_all`: mean `k=1.138`, choices `{'c1': 306, 'c12': 448, 'c2': 339, 'pos0': 261}`
  - `cbv_joint_prob_sam_no0`: mean `k=1.348`, choices `{'c1': 443, 'c12': 471, 'c2': 440}`
  - `cbv_logit_fob_prob_all`: mean `k=1.063`, choices `{'c1': 290, 'c12': 408, 'c2': 333, 'pos0': 323}`
  - `cbv_logit_fob_prob_no0`: mean `k=1.317`, choices `{'c1': 468, 'c12': 429, 'c2': 457}`
