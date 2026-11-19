# AdaFoB v2 Stage 1 Results: Semantic-Gated Boundary Negative Prompting (Frozen SAM ViT-B, Setting I)

- **Run Directory:** `experiments\runs\adafob_v2_stage1_20261119`
- **Kaggle Kernel Status:** `COMPLETED` (`episodes=120`, `patients=25`, `slices=4009`)
- **Pre-Registered Stage-1 Decision:** **`STAGE1_ITERATE_OR_STOP`** (Passing arms: `[]`)
- **Anchor Agreement vs Option A v2 (`cu130`):** `max_patient_cc_dice_diff_pp=0.0000` across `full10`, `positive0`, `boundary_near2`, `pba_f10_int_k2`
- **Cross-CUDA Agreement vs Step 9 (`cu128 -> cu130`):** `max_patient_cc_dice_diff_pp=0.0025`, `max_patient_cc_hd95_diff_mm=0.0002`

## 1. Stage-1 Criteria Evaluation (`NEW_ARMS` vs `boundary_near2` & `full10`)

Pass criteria: `delta_vs_boundary_near2 >= +1.00 pp` (`95% CI low > 0`), `recall_vs_full10 >= -0.15 pp` (`95% CI low >= -0.50 pp`), `LIVER recall >= 91.0%`, and `delta_hd95_vs_boundary_near2 < 0`.

| Arm | Pat CC Dice (%) | Delta vs `boundary_near2` (pp) [95% CI] | Delta vs `pba_f10` (pp) | Pat HD95 (mm) | Recall vs `full10` (pp) [95% CI] | LIVER Rec (%) | Passes Bar |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `pba_pos0_int_k2` | 84.338 | 1.171 [0.636, 1.725] | -0.381 | 31.366 | -1.050 [-1.443, -0.694] | 88.484 | `False` |
| `semgate_f10_k2_r5` | 84.251 | 1.085 [0.665, 1.524] | -0.468 | 30.081 | -1.045 [-1.467, -0.697] | 88.612 | `False` |
| `semgate_pos0_k2_r5` | 83.685 | 0.519 [0.012, 1.034] | -1.034 | 33.893 | -0.957 [-1.365, -0.584] | 89.010 | `False` |
| `semgate_pos0_only_k2_r5` | 83.687 | 0.521 [0.023, 1.037] | -1.032 | 33.940 | -0.754 [-1.146, -0.403] | 89.674 | `False` |
| `semgate_pos0_raymarch_k2` | 83.751 | 0.584 [0.062, 1.114] | -0.968 | 33.712 | -0.994 [-1.417, -0.614] | 88.898 | `False` |
| `semgate_pos0_safe_k2` | 83.604 | 0.437 [-0.125, 0.992] | -1.115 | 34.065 | -0.868 [-1.269, -0.504] | 89.210 | `False` |
| `semgate_f10_safe_k2` | 84.032 | 0.866 [0.499, 1.245] | -0.687 | 31.191 | -0.865 [-1.241, -0.548] | 89.011 | `False` |

## 2. Full 11-Arm Comparison Table (120 Episodes / 25 Patients / 4,009 Slices)

| Arm | Pat Raw Dice (%) | Pat CC Dice (%) [SD] | Ep CC Dice (%) | Pat CC HD95 (mm) | Ep Med/p90 HD95 (mm) | Pat CC Recall (%) | Empty Preds |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full10` | 78.866 | 81.821 [9.185] | 82.482 | 36.220 | 28.625 / 80.920 | 94.428 | 0 |
| `positive0` | 80.287 | 80.641 [10.252] | 81.463 | 43.925 | 31.757 / 91.788 | 95.008 | 0 |
| `boundary_near2` | 82.558 | 83.166 [9.366] | 83.789 | 32.796 | 25.234 / 67.764 | 94.288 | 0 |
| `pba_f10_int_k2` | 84.460 | 84.719 [9.198] | 85.336 | 27.934 | 18.966 / 61.012 | 92.950 | 0 |
| `pba_pos0_int_k2` | 84.138 | 84.338 [8.993] | 85.034 | 31.366 | 21.202 / 70.285 | 93.378 | 0 |
| `semgate_f10_k2_r5` | 84.048 | 84.251 [9.312] | 84.941 | 30.081 | 21.543 / 65.279 | 93.383 | 0 |
| `semgate_pos0_k2_r5` | 83.450 | 83.685 [9.460] | 84.465 | 33.893 | 25.198 / 72.651 | 93.472 | 0 |
| `semgate_pos0_only_k2_r5` | 83.420 | 83.687 [9.625] | 84.389 | 33.940 | 23.572 / 77.001 | 93.674 | 0 |
| `semgate_pos0_raymarch_k2` | 83.525 | 83.751 [9.489] | 84.552 | 33.712 | 24.285 / 76.350 | 93.434 | 0 |
| `semgate_pos0_safe_k2` | 83.332 | 83.604 [9.532] | 84.367 | 34.065 | 23.887 / 84.310 | 93.560 | 0 |
| `semgate_f10_safe_k2` | 83.765 | 84.032 [9.271] | 84.698 | 31.191 | 22.572 / 67.602 | 93.564 | 0 |

## 3. Per-Organ Patient-Balanced Breakdown (`CC Dice % / CC Recall % / CC HD95 mm`)

| Arm | LIVER | LK | RK | SPLEEN |
| :--- | :---: | :---: | :---: | :---: |
| `full10` | 84.26 / 91.66 / 41.3 | 82.86 / 95.64 / 26.2 | 83.25 / 97.01 / 29.0 | 76.91 / 93.40 / 48.4 |
| `positive0` | 84.40 / 93.45 / 39.8 | 83.20 / 95.73 / 27.9 | 80.69 / 97.05 / 45.9 | 74.27 / 93.79 / 62.1 |
| `boundary_near2` | 86.26 / 91.27 / 29.5 | 84.00 / 95.55 / 24.6 | 83.83 / 96.78 / 29.1 | 78.57 / 93.55 / 48.0 |
| `pba_f10_int_k2` | 87.26 / 87.78 / 23.1 | 84.95 / 95.32 / 23.6 | 84.91 / 96.04 / 26.9 | 81.76 / 92.66 / 38.2 |
| `pba_pos0_int_k2` | 86.99 / 88.48 / 30.8 | 84.98 / 95.38 / 23.6 | 84.88 / 96.72 / 25.7 | 80.50 / 92.93 / 45.3 |
| `semgate_f10_k2_r5` | 87.27 / 88.61 / 25.1 | 84.55 / 95.43 / 24.2 | 84.94 / 96.65 / 25.7 | 80.24 / 92.83 / 45.3 |
| `semgate_pos0_k2_r5` | 86.87 / 89.01 / 32.8 | 84.70 / 95.41 / 24.0 | 83.95 / 96.56 / 29.2 | 79.21 / 92.91 / 49.6 |
| `semgate_pos0_only_k2_r5` | 87.05 / 89.67 / 33.2 | 84.37 / 95.48 / 25.2 | 83.81 / 96.58 / 28.6 | 79.53 / 92.95 / 48.8 |
| `semgate_pos0_raymarch_k2` | 86.82 / 88.90 / 33.3 | 84.76 / 95.38 / 24.0 | 83.99 / 96.56 / 29.8 | 79.43 / 92.91 / 47.8 |
| `semgate_pos0_safe_k2` | 86.92 / 89.21 / 32.2 | 84.47 / 95.44 / 24.1 | 83.94 / 96.57 / 28.9 | 79.08 / 93.02 / 51.1 |
| `semgate_f10_safe_k2` | 87.08 / 89.01 / 27.7 | 84.39 / 95.54 / 24.4 | 84.77 / 96.80 / 26.5 | 79.89 / 92.91 / 46.2 |

## 4. Semantic-Gate Behavior & Compute Accounting

- Audited slices: `4009`, Mean SAM decoder calls per slice: `10.29`
- Remote aggregate T4 GPU-hours: `1.3542` (wall `2437.5s`)

| Semantic-Gated Arm | Mean `k` | `k=0` Slices | `k=1` Slices | `k=2` Slices | Mean Valid Arc (%) | Mean Offset (px) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `semgate_f10_k2_r5` | 1.991 | 0 | 38 | 3971 | 89.51 | 5.00 |
| `semgate_pos0_k2_r5` | 1.991 | 0 | 36 | 3973 | 91.24 | 5.00 |
| `semgate_pos0_only_k2_r5` | 2.000 | 0 | 0 | 4009 | 91.24 | 5.00 |
| `semgate_pos0_raymarch_k2` | 1.992 | 0 | 34 | 3975 | 94.63 | 5.26 |
| `semgate_pos0_safe_k2` | 1.991 | 1 | 35 | 3973 | 90.47 | 8.19 |
| `semgate_f10_safe_k2` | 1.990 | 1 | 39 | 3969 | 89.28 | 8.24 |

- Stage-0 Pilot intra-run exact determinism: `True`, Step 9 tolerance passed: `True` (`max_neg_coord_diff_px=0.00011`)
