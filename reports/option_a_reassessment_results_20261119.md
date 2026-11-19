# Option A Bounded Setting I Reassessment Results (Frozen SAM ViT-B)

- **Run Directory:** `experiments\runs\option_a_reassessment_20261119_v2`
- **Kaggle Kernel Status:** `COMPLETED` (`episodes=120`, `patients=25`, `slices=4009`)
- **Runtime Drift Note:** Step 9 ran on `torch 2.11.0+cu128` (`CUDA 12.8`); Option A ran on `torch 2.11.0+cu130` (`CUDA 13.0`). All 9 Option A arms ran inside the same matched session.
- **Pre-Registered Decision:** **`STOP_BOUNDARY_NEAR2_SUFFICIENT`** (Passing arms: `[]`)
- **Cross-CUDA Anchor Agreement vs Step 9 (720 rows):** `exact_mask_sha_matches=519/720`, `max_patient_cc_dice_diff_pp=0.0025`, `max_patient_cc_hd95_diff_mm=0.0002`, `max_episode_dice_diff=4.8988e-03`

## 1. Pre-Registered Criteria Evaluation

Criteria require beating matched `boundary_near2` by `>= +0.50 pp` Patient-balanced CC Dice with `95% CI lower bound > 0`, while keeping Recall delta vs `full10` `>= -0.15 pp` (`95% CI lower bound >= -0.50 pp`).

| New Arm | Pat CC Dice (%) | Delta vs `boundary_near2` (pp) [95% CI] | Delta vs `full10` (pp) [95% CI] | Pat HD95 (mm) | Recall vs `full10` (pp) [95% CI] | Passes Bar |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| `resp_magnitude_k2` | 82.315 | -0.852 [-1.361, -0.449] | 0.494 [-0.175, 1.238] | 38.185 | 0.028 [-0.353, 0.382] | `False` |
| `resp_signed_global_k2` | 82.986 | -0.181 [-0.669, 0.292] | 1.164 [0.438, 2.005] | 33.660 | 0.221 [-0.131, 0.600] | `False` |
| `resp_signed_localring_k2` | 83.854 | 0.688 [0.344, 1.072] | 2.033 [1.301, 2.820] | 32.274 | -0.233 [-0.578, 0.127] | `False` |
| `resp_signed_adaptive` | 83.769 | 0.602 [0.125, 1.078] | 1.948 [1.279, 2.639] | 32.517 | -0.471 [-0.788, -0.122] | `False` |
| `local_contrast_near2` | 83.056 | -0.111 [-0.411, 0.232] | 1.235 [0.571, 1.904] | 34.476 | -0.058 [-0.403, 0.265] | `False` |
| `pba_fixed_k2` | 84.719 | 1.553 [1.100, 2.024] | 2.898 [2.027, 3.868] | 27.934 | -1.479 [-1.867, -1.117] | `False` |

## 2. Full 9-Arm Comparison Table (120 Episodes / 25 Patients / 4,009 Slices)

| Arm | Pat Raw Dice (%) | Pat CC Dice (%) [SD] | Ep CC Dice (%) [SD] | Pat CC HD95 (mm) | Ep Med/p90 HD95 (mm) | Pat CC Recall (%) | Empty Preds |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full10` | 78.866 | 81.821 [9.185] | 82.482 [13.652] | 36.220 | 28.625 / 80.920 | 94.428 | 0 |
| `positive0` | 80.287 | 80.641 [10.252] | 81.463 [15.160] | 43.925 | 31.757 / 91.788 | 95.008 | 0 |
| `boundary_near2` | 82.558 | 83.166 [9.366] | 83.789 [13.587] | 32.796 | 25.234 / 67.764 | 94.288 | 0 |
| `resp_magnitude_k2` | 81.263 | 82.315 [9.364] | 82.989 [13.824] | 38.185 | 29.438 / 77.343 | 94.457 | 0 |
| `resp_signed_global_k2` | 82.150 | 82.986 [9.455] | 83.649 [13.559] | 33.660 | 26.102 / 71.203 | 94.649 | 0 |
| `resp_signed_localring_k2` | 83.256 | 83.854 [8.972] | 84.505 [13.248] | 32.274 | 24.546 / 70.553 | 94.195 | 0 |
| `resp_signed_adaptive` | 82.756 | 83.769 [9.285] | 84.450 [13.308] | 32.517 | 25.167 / 68.228 | 93.957 | 0 |
| `local_contrast_near2` | 82.398 | 83.056 [9.378] | 83.691 [13.504] | 34.476 | 26.872 / 73.723 | 94.371 | 0 |
| `pba_fixed_k2` | 84.460 | 84.719 [9.198] | 85.336 [12.982] | 27.934 | 18.966 / 61.012 | 92.950 | 0 |

## 3. Per-Organ Patient-Balanced CC Dice (%)

| Arm | LIVER (%) | LK (%) | RK (%) | SPLEEN (%) |
| :--- | ---: | ---: | ---: | ---: |
| `full10` | 84.264 | 82.862 | 83.248 | 76.911 |
| `positive0` | 84.403 | 83.195 | 80.694 | 74.272 |
| `boundary_near2` | 86.262 | 84.001 | 83.834 | 78.570 |
| `resp_magnitude_k2` | 85.061 | 83.186 | 83.554 | 77.458 |
| `resp_signed_global_k2` | 86.222 | 83.980 | 84.014 | 77.725 |
| `resp_signed_localring_k2` | 87.330 | 84.416 | 84.480 | 79.191 |
| `resp_signed_adaptive` | 87.435 | 84.166 | 84.225 | 79.250 |
| `local_contrast_near2` | 86.216 | 83.674 | 83.230 | 79.103 |
| `pba_fixed_k2` | 87.262 | 84.953 | 84.905 | 81.755 |

## 4. Prompt Behavior and Parity Summary

- Total audited slices: `4009`
- Mean SAM decoder calls per slice: `17.70`
- `resp_signed_localring_k2 == resp_magnitude_k2` slices: `690/4009` (vs `resp_signed_global_k2 == resp_magnitude_k2`: `762/4009`)
- `local_contrast_near2 == boundary_near2` slices: `1293/4009`
- `resp_signed_adaptive` mean negative points `k`: `4.397` (histogram: `{'0': 164, '1': 236, '2': 362, '3': 455, '4': 463, '5': 524, '6': 1805}`)
- `pba_fixed_k2` fallbacks: `0/4009`
- Remote aggregate T4 GPU-hours: `1.4055` (wall `2529.8s`)
- Stage-0 Pilot intra-run exact determinism: `True`, Step 9 CUDA tolerance passed: `True` (`max_neg_coord_diff_px=0.00011`, `min_full10_cc_dice_vs_step9=1.00000`, `projected_total_gpu_hours=1.3218`)
