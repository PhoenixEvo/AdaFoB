# AdaFoB v2 Stage 3 Results: Cross-Modality CHAOST2 (Abd-MRI, Frozen SAM ViT-B & ViT-H, Setting I)

- **Run Directory:** `experiments\runs\adafob_v2_stage3_chaost2_20261119`
- **Kaggle Kernel Status:** `COMPLETED` (`episodes_per_backbone=80`, `patients=15`, `slices=1354`, `aggregate_gpu_hours=2.4301`, `wall=4374.2s`)
- **Stage-0 Locked Orientation:** `flip_y` (`best_fold0_r1_full10_pat_cc_dice_pct=81.17%`, `intra_run_exact_5folds=True`)
- **Pre-Registered Stage-3 Decision:** **`STAGE3_CHAOST2_OOD_GAIN_WITH_RECALL_TRADEOFF`** (OOD pass: `[{'backbone': 'vit_b', 'arm': 'pba_f10_int_k2'}, {'backbone': 'vit_b', 'arm': 'cbv_f10_fixed_supp'}, {'backbone': 'vit_b', 'arm': 'cbv_f10_no0_supp'}]`, Recall-protected pass: `[]`)

## 1. Stage-0 Orientation & Multimask Parity Check (Fold 0, SAM ViT-B)

| Orientation | Multimask Output | `full10` Raw Dice (%) | `full10` CC Dice (%) | `positive0` CC Dice (%) |
| :--- | :---: | ---: | ---: | ---: |
| `identity` | `False` | 54.31 | 62.05 | 64.02 |
| `identity` | `True` | 30.93 | 35.27 | 11.63 |
| `flip_y` | `False` | 78.92 | 81.17 | 83.31 |
| `flip_y` | `True` | 56.98 | 60.93 | 19.09 |

## 2. Patient-Balanced 3D Metrics on CHAOST2 (`vit_b`, 80 Episodes / 15 Patients / 1,354 Slices)

| Arm | Pat Raw Dice (%) | Pat CC Dice (%) [SD] | Ep CC Dice (%) | Pat HD95 (mm) | Pat Recall (%) | Detached FP / pat |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full10` | 77.76 | 80.22 [6.24] | 80.43 | 44.56 | 92.92 | 4195.3 |
| `positive0` | 78.61 | 79.09 [8.84] | 79.47 | 49.44 | 94.75 | 2097.1 |
| `random2_mean5` | 80.64 | 81.26 [6.76] | 81.61 | 41.34 | 93.16 | 2342.8 |
| `boundary_near2` | 82.29 | 82.74 [6.51] | 82.92 | 36.65 | 92.45 | 1950.6 |
| `resp_signed_localring_k2` | 82.47 | 83.02 [6.41] | 83.17 | 36.14 | 92.47 | 1907.1 |
| `pba_f10_int_k2` | 83.24 | 83.57 [6.43] | 83.78 | 34.65 | 91.72 | 1922.2 |
| `cbv_f10_fixed_supp` | 83.25 | 83.50 [6.01] | 83.60 | 34.38 | 92.17 | 1769.6 |
| `cbv_f10_no0_supp` | 83.49 | 83.74 [6.11] | 83.93 | 33.50 | 91.98 | 1725.9 |

### Per-Organ Patient-Balanced Breakdown (`vit_b`, `CC Dice % / CC Recall % / CC HD95 mm`)

| Arm | LIVER | RK | LK | SPLEEN |
| :--- | :---: | :---: | :---: | :---: |
| `full10` | 71.66 / 89.65 / 63.4 | 87.88 / 95.60 / 26.3 | 82.63 / 95.72 / 39.3 | 78.70 / 90.72 / 49.2 |
| `positive0` | 76.52 / 93.24 / 55.0 | 85.52 / 95.94 / 34.3 | 80.55 / 95.99 / 46.4 | 73.75 / 93.84 / 62.2 |
| `random2_mean5` | 77.20 / 89.84 / 48.2 | 87.54 / 95.72 / 28.0 | 83.44 / 95.65 / 37.5 | 76.87 / 91.42 / 51.7 |
| `boundary_near2` | 80.16 / 88.30 / 40.1 | 88.52 / 95.52 / 23.4 | 84.21 / 95.32 / 35.3 | 78.07 / 90.69 / 47.8 |
| `resp_signed_localring_k2` | 80.50 / 88.73 / 40.7 | 89.42 / 95.42 / 20.9 | 84.18 / 95.39 / 36.3 | 77.97 / 90.37 / 46.7 |
| `pba_f10_int_k2` | 81.44 / 86.76 / 37.0 | 89.58 / 94.78 / 20.5 | 84.22 / 94.78 / 34.5 | 79.03 / 90.57 / 46.6 |
| `cbv_f10_fixed_supp` | 81.75 / 88.00 / 36.1 | 90.05 / 95.11 / 17.8 | 84.01 / 95.24 / 36.0 | 78.18 / 90.35 / 47.6 |
| `cbv_f10_no0_supp` | 81.94 / 87.53 / 34.9 | 90.07 / 95.02 / 18.0 | 84.21 / 95.07 / 35.1 | 78.74 / 90.29 / 45.9 |

## 2. Patient-Balanced 3D Metrics on CHAOST2 (`vit_h`, 80 Episodes / 15 Patients / 1,354 Slices)

| Arm | Pat Raw Dice (%) | Pat CC Dice (%) [SD] | Ep CC Dice (%) | Pat HD95 (mm) | Pat Recall (%) | Detached FP / pat |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full10` | 77.28 | 79.62 [7.50] | 79.76 | 45.28 | 93.42 | 3340.5 |
| `positive0` | 72.89 | 73.25 [12.13] | 73.84 | 70.02 | 94.57 | 1782.9 |
| `random2_mean5` | 78.54 | 79.19 [7.51] | 79.64 | 52.36 | 93.74 | 2012.1 |
| `boundary_near2` | 81.60 | 82.18 [6.45] | 82.48 | 42.51 | 92.71 | 2069.1 |
| `resp_signed_localring_k2` | 81.47 | 82.22 [7.04] | 82.63 | 42.60 | 92.55 | 2244.7 |
| `pba_f10_int_k2` | 81.91 | 82.35 [7.02] | 82.86 | 40.37 | 92.55 | 1993.9 |
| `cbv_f10_fixed_supp` | 81.45 | 81.76 [7.09] | 82.39 | 43.17 | 92.85 | 1902.7 |
| `cbv_f10_no0_supp` | 81.67 | 82.00 [7.18] | 82.61 | 42.52 | 92.77 | 1861.3 |

### Per-Organ Patient-Balanced Breakdown (`vit_h`, `CC Dice % / CC Recall % / CC HD95 mm`)

| Arm | LIVER | RK | LK | SPLEEN |
| :--- | :---: | :---: | :---: | :---: |
| `full10` | 77.72 / 91.60 / 55.3 | 86.48 / 95.26 / 31.9 | 81.19 / 95.89 / 42.9 | 73.10 / 90.93 / 51.0 |
| `positive0` | 74.59 / 93.47 / 76.5 | 75.51 / 95.33 / 66.1 | 75.64 / 95.19 / 59.4 | 67.27 / 94.28 / 78.1 |
| `random2_mean5` | 78.50 / 92.15 / 60.5 | 84.18 / 95.31 / 38.6 | 81.25 / 95.29 / 45.7 | 72.84 / 92.20 / 64.7 |
| `boundary_near2` | 82.45 / 88.85 / 43.0 | 87.11 / 95.20 / 29.4 | 84.10 / 95.01 / 36.3 | 75.05 / 91.78 / 61.3 |
| `resp_signed_localring_k2` | 81.76 / 89.92 / 45.1 | 87.98 / 95.00 / 27.1 | 83.73 / 94.87 / 38.7 | 75.41 / 90.39 / 59.4 |
| `pba_f10_int_k2` | 82.97 / 89.09 / 40.0 | 87.62 / 94.88 / 27.6 | 83.31 / 94.33 / 36.9 | 75.50 / 91.90 / 57.0 |
| `cbv_f10_fixed_supp` | 82.11 / 90.00 / 45.9 | 87.58 / 94.79 / 27.3 | 83.14 / 94.60 / 39.5 | 74.24 / 92.02 / 59.9 |
| `cbv_f10_no0_supp` | 82.31 / 89.79 / 45.6 | 87.57 / 94.77 / 27.4 | 83.24 / 94.56 / 39.3 | 74.86 / 91.95 / 57.8 |

## 3. Paired Patient-Cluster Bootstrap Contrasts & Backbone Scaling (`vit_h - vit_b`)

| Backbone | Method Arm | Delta vs `boundary_near2` (pp) [95% CI] | Delta vs `random2_mean5` (pp) | Delta vs `full10` (pp) | Delta HD95 vs `near2` (mm) | Rec vs `full10` (pp) | OOD / Rec-Prot |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| `vit_b` | `resp_signed_localring_k2` | 0.28 [-0.07, 0.66] | 1.76 | 2.80 | -0.50 | -0.45 | `False` / `False` |
| `vit_b` | `pba_f10_int_k2` | 0.83 [0.17, 1.45] | 2.31 | 3.35 | -1.99 | -1.20 | `True` / `False` |
| `vit_b` | `cbv_f10_fixed_supp` | 0.76 [0.32, 1.22] | 2.24 | 3.28 | -2.27 | -0.75 | `True` / `False` |
| `vit_b` | `cbv_f10_no0_supp` | 1.00 [0.59, 1.44] | 2.48 | 3.52 | -3.15 | -0.95 | `True` / `False` |
| `vit_h` | `resp_signed_localring_k2` | 0.04 [-0.59, 0.63] | 3.03 | 2.60 | 0.09 | -0.87 | `False` / `False` |
| `vit_h` | `pba_f10_int_k2` | 0.17 [-0.56, 0.93] | 3.16 | 2.73 | -2.14 | -0.87 | `False` / `False` |
| `vit_h` | `cbv_f10_fixed_supp` | -0.41 [-1.17, 0.36] | 2.57 | 2.14 | 0.66 | -0.57 | `False` / `False` |
| `vit_h` | `cbv_f10_no0_supp` | -0.18 [-0.99, 0.63] | 2.80 | 2.38 | 0.02 | -0.65 | `False` / `False` |

### Backbone Scaling (`SAM ViT-H` minus `SAM ViT-B` on Same 15 Patients)

| Arm | Delta Pat CC Dice (pp) [95% CI] | Delta Pat HD95 (mm) [95% CI] | Delta Pat Recall (pp) |
| :--- | ---: | ---: | ---: |
| `full10` | -0.60 [-2.34, 1.08] | 0.72 [-6.67, 7.50] | 0.50 |
| `positive0` | -5.83 [-8.44, -3.44] | 20.58 [13.23, 28.41] | -0.19 |
| `random2_mean5` | -2.07 [-2.93, -1.28] | 11.02 [7.96, 14.31] | 0.58 |
| `boundary_near2` | -0.56 [-1.28, 0.06] | 5.86 [2.62, 9.23] | 0.26 |
| `resp_signed_localring_k2` | -0.80 [-1.88, 0.19] | 6.45 [1.84, 10.97] | 0.07 |
| `pba_f10_int_k2` | -1.22 [-2.12, -0.25] | 5.71 [1.95, 9.39] | 0.83 |
| `cbv_f10_fixed_supp` | -1.73 [-3.00, -0.59] | 8.79 [4.97, 13.07] | 0.68 |
| `cbv_f10_no0_supp` | -1.74 [-3.02, -0.60] | 9.03 [4.95, 13.36] | 0.79 |

## 4. CBV Choice Audit

- `vit_b`: mean decoder calls/slice `21.74`, `pba` fallbacks `0`, `cbv_f10_fixed_supp` mean `k=1.341` choices `{'c1': 303, 'c12': 585, 'c2': 343, 'pos0': 123}`, `cbv_f10_no0_supp` mean `k=1.442` choices `{'c1': 350, 'c12': 598, 'c2': 406}`.
- `vit_h`: mean decoder calls/slice `21.74`, `pba` fallbacks `0`, `cbv_f10_fixed_supp` mean `k=1.229` choices `{'c1': 315, 'c12': 509, 'c2': 331, 'pos0': 199}`, `cbv_f10_no0_supp` mean `k=1.386` choices `{'c1': 416, 'c12': 522, 'c2': 416}`.
