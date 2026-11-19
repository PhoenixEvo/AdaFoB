# AdaFoB v2 Stage 2 Results: Scale-Adaptive Counterfactual Boundary Verification (Frozen SAM ViT-B, Setting I)

- **Run Directory:** `experiments\runs\adafob_v2_stage2_20261119`
- **Kaggle Kernel Status:** `COMPLETED` (`episodes=120`, `patients=25`, `slices=4009`)
- **Pre-Registered Stage-2 Decision:** **`STAGE2_PARETO_STRONG_RECALL_REVIEW`** (Primary pass: `[]`, Secondary strong Pareto: `['cbv_f10_fixed_supp', 'cbv_f10_scale_supp']`)
- **Anchor Agreement vs Stage 1 (`cu130`):** `max_patient_cc_dice_diff_pp=0.0000` across `full10`, `positive0`, `boundary_near2`, `pba_f10_int_k2`
- **Cross-CUDA Agreement vs Step 9 (`cu128 -> cu130`):** `max_patient_cc_dice_diff_pp=0.0025`, `max_patient_cc_hd95_diff_mm=0.0002`

## 1. Stage-2 Criteria Evaluation (`NEW_ARMS` vs `boundary_near2` & `full10`)

Primary criteria: `delta_vs_boundary_near2 >= +1.00 pp` (`95% CI low > 0`), `recall_vs_full10 >= -0.15 pp` (`95% CI low >= -0.50 pp`), `LIVER recall >= 91.0%`, `delta_hd95_vs_boundary_near2 < 0`. Secondary Pareto diagnostic: `delta >= +1.00 pp`, `LIVER recall >= 90.0%`, `delta_hd95 <= -2.0 mm`.

| Arm | Pat CC Dice (%) | Delta vs `boundary_near2` (pp) [95% CI] | Delta vs `pba_f10` (pp) | Pat HD95 (mm) | Recall vs `full10` (pp) [95% CI] | LIVER Rec (%) | Primary / Pareto |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `pba_f10_scale_k2` | 84.571 | 1.404 [0.971, 1.857] | -0.148 | 29.114 | -1.428 [-1.797, -1.082] | 88.015 | `False` / `False` |
| `cbv_f10_fixed_supp` | 84.426 | 1.259 [0.829, 1.723] | -0.293 | 29.793 | -0.784 [-1.204, -0.400] | 90.755 | `False` / `True` |
| `cbv_f10_scale_supp` | 84.412 | 1.245 [0.823, 1.683] | -0.307 | 29.812 | -0.806 [-1.262, -0.397] | 90.816 | `False` / `True` |
| `cbv_f10_scale_cal` | 82.227 | -0.939 [-1.749, -0.196] | -2.492 | 39.761 | -0.381 [-0.810, 0.018] | 91.210 | `False` / `False` |
| `cbv_f10_scale_dual` | 82.168 | -0.998 [-1.816, -0.257] | -2.551 | 40.817 | -0.452 [-0.851, -0.074] | 91.093 | `False` / `False` |
| `cbv_hybrid_scale_dual` | 82.189 | -0.977 [-1.806, -0.246] | -2.530 | 40.970 | -0.432 [-0.847, -0.037] | 91.239 | `False` / `False` |
| `cbv_fast3_scale_dual` | 81.932 | -1.234 [-2.037, -0.497] | -2.787 | 40.931 | -0.503 [-0.934, -0.100] | 90.501 | `False` / `False` |

## 2. Full 11-Arm Comparison Table (120 Episodes / 25 Patients / 4,009 Slices)

| Arm | Pat Raw Dice (%) | Pat CC Dice (%) [SD] | Ep CC Dice (%) | Pat CC HD95 (mm) | Ep Med/p90 HD95 (mm) | Pat CC Recall (%) | Empty Preds |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full10` | 78.866 | 81.821 [9.185] | 82.482 | 36.220 | 28.625 / 80.920 | 94.428 | 0 |
| `positive0` | 80.287 | 80.641 [10.252] | 81.463 | 43.925 | 31.757 / 91.788 | 95.008 | 0 |
| `boundary_near2` | 82.558 | 83.166 [9.366] | 83.789 | 32.796 | 25.234 / 67.764 | 94.288 | 0 |
| `pba_f10_int_k2` | 84.460 | 84.719 [9.198] | 85.336 | 27.934 | 18.966 / 61.012 | 92.950 | 0 |
| `pba_f10_scale_k2` | 84.339 | 84.571 [9.195] | 85.217 | 29.114 | 20.281 / 62.679 | 93.000 | 0 |
| `cbv_f10_fixed_supp` | 84.175 | 84.426 [9.767] | 85.117 | 29.793 | 22.986 / 63.890 | 93.645 | 0 |
| `cbv_f10_scale_supp` | 84.176 | 84.412 [9.783] | 85.123 | 29.812 | 21.148 / 60.589 | 93.622 | 0 |
| `cbv_f10_scale_cal` | 81.991 | 82.227 [10.095] | 83.119 | 39.761 | 28.134 / 85.364 | 94.047 | 0 |
| `cbv_f10_scale_dual` | 81.952 | 82.168 [10.051] | 83.056 | 40.817 | 28.395 / 85.902 | 93.976 | 0 |
| `cbv_hybrid_scale_dual` | 81.934 | 82.189 [10.063] | 83.076 | 40.970 | 28.617 / 85.790 | 93.996 | 0 |
| `cbv_fast3_scale_dual` | 81.725 | 81.932 [10.015] | 82.828 | 40.931 | 28.735 / 88.326 | 93.925 | 0 |

## 3. Per-Organ Patient-Balanced Breakdown (`CC Dice % / CC Recall % / CC HD95 mm`)

| Arm | LIVER | LK | RK | SPLEEN |
| :--- | :---: | :---: | :---: | :---: |
| `full10` | 84.26 / 91.66 / 41.3 | 82.86 / 95.64 / 26.2 | 83.25 / 97.01 / 29.0 | 76.91 / 93.40 / 48.4 |
| `positive0` | 84.40 / 93.45 / 39.8 | 83.20 / 95.73 / 27.9 | 80.69 / 97.05 / 45.9 | 74.27 / 93.79 / 62.1 |
| `boundary_near2` | 86.26 / 91.27 / 29.5 | 84.00 / 95.55 / 24.6 | 83.83 / 96.78 / 29.1 | 78.57 / 93.55 / 48.0 |
| `pba_f10_int_k2` | 87.26 / 87.78 / 23.1 | 84.95 / 95.32 / 23.6 | 84.91 / 96.04 / 26.9 | 81.76 / 92.66 / 38.2 |
| `pba_f10_scale_k2` | 87.13 / 88.01 / 23.5 | 84.96 / 95.31 / 23.6 | 84.83 / 95.99 / 26.3 | 81.35 / 92.68 / 43.0 |
| `cbv_f10_fixed_supp` | 88.23 / 90.76 / 23.5 | 84.99 / 95.25 / 23.9 | 84.82 / 95.79 / 26.2 | 79.66 / 92.79 / 45.6 |
| `cbv_f10_scale_supp` | 88.08 / 90.82 / 24.5 | 85.00 / 95.24 / 23.9 | 84.92 / 95.70 / 24.9 | 79.65 / 92.73 / 45.9 |
| `cbv_f10_scale_cal` | 87.47 / 91.21 / 28.9 | 83.81 / 95.44 / 28.6 | 81.94 / 96.59 / 42.8 | 75.69 / 92.95 / 58.8 |
| `cbv_f10_scale_dual` | 87.07 / 91.09 / 35.3 | 83.98 / 95.40 / 27.3 | 81.97 / 96.53 / 41.9 | 75.66 / 92.88 / 58.7 |
| `cbv_hybrid_scale_dual` | 87.05 / 91.24 / 35.2 | 83.98 / 95.38 / 27.2 | 81.95 / 96.52 / 42.6 | 75.77 / 92.84 / 58.9 |
| `cbv_fast3_scale_dual` | 86.51 / 90.50 / 35.1 | 83.87 / 95.47 / 27.4 | 81.92 / 96.61 / 42.7 | 75.43 / 93.12 / 58.6 |

## 4. CBV Subset Selection, Calibration & Compute Accounting

- Audited slices: `4009`, Mean SAM decoder calls per slice: `8.10`
- Mean `tau_cal`: `-0.0141`, `no_sep` slices: `24`
- Remote aggregate T4 GPU-hours: `1.3751` (wall `2475.2s`)

| CBV Arm | Mean `k` | Choice Distribution | Fallbacks |
| :--- | ---: | :--- | ---: |
| `cbv_f10_fixed_supp` | 1.385 | `{'c1': 944, 'c12': 1865, 'c2': 880, 'pos0': 320}` | 0 |
| `cbv_f10_scale_supp` | 1.376 | `{'c1': 933, 'c12': 1828, 'c2': 927, 'pos0': 321}` | 0 |
| `cbv_f10_scale_cal` | 1.279 | `{'c1': 941, 'c12': 1636, 'c2': 916, 'pos0': 516}` | 0 |
| `cbv_f10_scale_dual` | 1.298 | `{'c1': 969, 'c12': 1659, 'c2': 915, 'pos0': 466}` | 0 |
| `cbv_hybrid_scale_dual` | 1.466 | `{'c1': 789, 'c12': 1403, 'c2': 665, 'near2': 808, 'pos0': 344}` | 0 |
| `cbv_fast3_scale_dual` | 1.493 | `{'c12': 2993, 'pos0': 1016}` | 0 |

- Stage-0 Pilot intra-run exact determinism: `True`, Step 9 tolerance passed: `True` (`max_neg_coord_diff_px=0.00011`)
