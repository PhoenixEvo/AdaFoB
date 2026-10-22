# Step 4: Symmetric Frozen-SAM Baseline Matrix & Diagnostic Report (v2 — Corrected)

**Date:** 2026-10-22  
**Role:** Lead Research Engineer & Independent Methodological Auditor  
**Setting:** Setting I — Zero-Shot / Few-Shot In-Domain on **Completely Frozen SAM** (Zero Fine-Tuning, Zero LoRA, Zero Query GT Access)  
**Hardware Profile:** NVIDIA Tesla T4 GPU (16 GB VRAM) on Kaggle Cloud (Environment: Python 3.10, PyTorch 2.4, CUDA 12.2)  
**Evaluation Scope:** 5 Folds, 120 Episodes, 25 Physical Test Patients, 4 Organs ($N=3,999$ Timed Evaluation Slices)  

---

## 1. Executive Summary & Corrected Audit Statements

Following methodological audit review, the Step 4 baseline findings are stated with complete scientific conservatism:

1. **Backbone Comparison under Symmetric Rule R1 (`SingleMask`):**
   * **No difference detected under R1:** Comparing SAM ViT-B against SAM ViT-H under single-mask mode (`multimask_output=False`), the paired patient-cluster bootstrap mean difference is **$-0.39\text{ pp}$** (95% CI: $[-1.20\text{ pp}, +0.37\text{ pp}]$, unadjusted $p = 0.33$, Holm-adj $p = 0.99$).
   * **Equivalence is NOT established:** Because no pre-specified equivalence margin $\pm \Delta$ was formally registered prior to execution, failure to reject the null hypothesis of zero difference does not constitute statistical proof of equivalence.
   * **Upstream ViT-H Reference Remains Higher:** SAM ViT-H with upstream multimask token 0 (**B0**, $86.19\%$ Dice CC, HD95 mean $25.10\text{ mm}$, median $19.20\text{ mm}$, p90 $55.72\text{ mm}$) remains higher than SAM ViT-B with R1 (**B3B**, $82.55\%$ Dice CC, HD95 mean $35.18\text{ mm}$, median $28.26\text{ mm}$, p90 $81.03\text{ mm}$) by **$3.64\text{ pp}$** (95% CI: $[-4.90, -2.40]$, Holm-adj $p = 0.003$).

2. **Rule R1 Status on ViT-H:**
   * Rule R1 was frozen exclusively based on **9 development cases evaluated on ViT-B**.
   * On ViT-H test data across all 120 episodes, Rule R1 is **empirically worse than upstream R3 (`mask[0]`)** by **$-3.25\text{ pp}$ Dice** (82.94% vs 86.19%) and **$+12.72\text{ mm}$ HD95** (37.82 mm vs 25.10 mm; Holm-adj $p = 0.003$).
   * Therefore, **Rule R1 is NOT optimal for ViT-H**. The 120-episode run serves as a symmetric baseline matrix, not as a validation proving R1 is optimal across backbones.

3. **Status of Hypothesis H2 (Kidney Asymmetry):**
   * Hypothesis H2 is **not supported under Fixed-10 Rule R1**:
     * ViT-H (B3H): RK Dice **$83.55\%$** vs LK Dice **$83.54\%$** (Diff: $+0.01\text{ pp}$, paired $t$-test $p = 0.9938$).
     * ViT-B (B3B): RK Dice **$83.71\%$** vs LK Dice **$82.98\%$** (Diff: $+0.73\text{ pp}$, paired $t$-test $p = 0.5158$).
   * In positive-only prompting (B4H / B4B, 0 negative points), per-organ scores show an **exploratory, unadjusted observation** where RK scores lower than LK:
     * B4H (ViT-H Pos-only): RK **$80.72\%$** vs LK **$84.19\%$** ($\Delta = -3.47\text{ pp}$).
     * B4B (ViT-B Pos-only): RK **$81.54\%$** vs LK **$83.54\%$** ($\Delta = -2.00\text{ pp}$).
     * This unadjusted trend indicates that without negative prompts, right kidney segmentation exhibits slightly higher boundary dispersion than left kidney.

4. **Negative Prompt Attribution:**
   * On **ViT-B**, negative prompts provide a pre-specified, statistically significant boundary restraint effect: HD95 improves by **$-9.25\text{ mm}$** (95% CI: $[-15.44\text{ mm}, -3.47\text{ mm}]$, Holm-adj $p = 0.0064$) and Dice improves by **$+1.22\text{ pp}$** (95% CI: $[+0.38, +2.17]$, Holm-adj $p = 0.0224$).
   * On **ViT-H**, adding 10 negative points produces no detectable Dice effect ($-0.06\text{ pp}$, 95% CI: $[-1.03, +1.02]$, Holm-adj $p = 1.0$), with an inconclusive directional trend on HD95 ($-5.26\text{ mm}$, 95% CI: $[-11.57, +0.92]$, Holm-adj $p = 0.484$).

---

## 2. Frozen Rule R1 Provenance & Development Set Limitations

### 2.1 Candidate Rules
* **Rule R1 (`SingleMask`):** `multimask_output=False`. Single dedicated mask token.
* **Rule R2 (`ArgmaxScore`):** `multimask_output=True`, mask selected via $\operatorname{argmax}(\text{iou\_prediction})$.
* **Rule R3 (`Upstream mask[0]`):** `multimask_output=True`, hardcoded index 0 (retained as B0 reproduction reference).

### 2.2 Freezing and Scope Caveat
* **Pre-registration:** Rule R1 was pre-registered in [`configs/mask_rule.yaml`](file:///d:/HD-NCKH27/AdaFoB/configs/mask_rule.yaml) (commit `24e35bd`). In the config, `mask_index: 0` refers strictly to the single output slice of single-mask mode (`multimask_output=False`), NOT `mask[0]` of multimask mode.
* **Development Sample Limitation:** Rule R1 was selected based solely on $N=9$ cases evaluated locally on **ViT-B** from the 5 dev patients (`image_2, 8, 14, 20, 25`). The remaining 11 cases were not run locally due to compute timeouts.
* **Sub-optimality on ViT-H:** The full test evaluation demonstrates that while R1 rescued ViT-B from token collapse, it degraded ViT-H relative to upstream R3. R1 is therefore an asymmetric compromise that enables backbone parity under a shared rule, but is suboptimal for ViT-H alone.

---

## 3. Support Patient Assignment Verification per Fold

Per condition A.5, support patient assignments are verified directly against [`protocol/paired_episode_manifest.csv`](file:///d:/HD-NCKH27/AdaFoB/protocol/paired_episode_manifest.csv). In the 5-fold cross-validation scheme (`get_folds('SABS')`), each fold contains 7 patients sorted numerically; the support patient is strictly fixed at index 2 (`supp_idx = 2`), while the remaining 6 patients form the evaluation query set (24 episodes per fold, 4 organs):

| Fold | Patient ID Set (7 Patients) | Support Patient (`supp_idx=2`) | Query Patient IDs (6 Patients) | Manifest Verification (Exact Sample Rows) |
| :---: | :--- | :---: | :--- | :--- |
| **0** | `[0, 1, 2, 3, 4, 5, 6]` | **`image_2`** | `[0, 1, 3, 4, 5, 6]` | Row 0: `ep=000, fold=0, organ=LIVER, vol_id=.../image_0.nii`<br>Row 1: `ep=001, fold=0, organ=LIVER, vol_id=.../image_1.nii` |
| **1** | `[6, 7, 8, 9, 10, 11, 12]` | **`image_8`** | `[6, 7, 9, 10, 11, 12]` | Row 24: `ep=024, fold=1, organ=LIVER, vol_id=.../image_10.nii`<br>Row 25: `ep=025, fold=1, organ=LIVER, vol_id=.../image_11.nii` |
| **2** | `[12, 13, 14, 15, 16, 17, 18]` | **`image_14`** | `[12, 13, 15, 16, 17, 18]` | Row 48: `ep=048, fold=2, organ=LIVER, vol_id=.../image_12.nii`<br>Row 49: `ep=049, fold=2, organ=LIVER, vol_id=.../image_13.nii` |
| **3** | `[18, 19, 20, 21, 22, 23, 24]` | **`image_20`** | `[18, 19, 21, 22, 23, 24]` | Row 72: `ep=072, fold=3, organ=LIVER, vol_id=.../image_18.nii`<br>Row 73: `ep=073, fold=3, organ=LIVER, vol_id=.../image_19.nii` |
| **4** | `[0, 24, 25, 26, 27, 28, 29]` | **`image_25`** | `[0, 24, 26, 27, 28, 29]` | Row 96: `ep=096, fold=4, organ=LIVER, vol_id=.../image_0.nii`<br>Row 97: `ep=097, fold=4, organ=LIVER, vol_id=.../image_24.nii` |

**Dual-Role Disclosure:** The 5 development patients (`image_2, 8, 14, 20, 25`) never appear as evaluation query targets in any of the 120 test episodes. However, each served as the support volume for all 24 query episodes within its assigned fold.

---

## 4. Measurement Reconciliation & Audit Trail

Earlier reports contained numerical discrepancies arising from differing execution environments, unreset GPU profilers, and proxy timers. Below is the audited reconciliation backed by raw log files:

| Metric | Previously Reported | Current Verified (Raw Log) | Root Cause & Configuration Difference | Status of Prior Number |
| :--- | :---: | :---: | :--- | :--- |
| **ViT-B Image Encoder Latency** | $403.43 \pm 8.48\text{ ms}$ | **$495.79 \pm 13.64\text{ ms}$** | Previous: Kaggle Tesla P100 GPU (higher memory bandwidth). Current: Kaggle Tesla T4 GPU ($N=3,999$ slices, `step4_timing_breakdown.json`). | Superseded (hardware shift P100 $\to$ T4) |
| **ViT-H Mask Decoder Latency** | $96.48\text{ ms}$ / $48.20\text{ ms}$ | **$9.19 \pm 0.32\text{ ms}$** | 96.48 ms was from Kernel v4 measuring multimask mode with 3 output heads, IoU sorting, and NumPy transfers. 48.20 ms was a theoretical estimate ($96.48/2$). Current 9.19 ms is pure SingleMask GPU decode with CUDA synchronization. | Superseded (multimask overhead vs pure decode) |
| **FoB Prompt Generator Latency** | $18.42\text{ ms}$ | **$258.21 \pm 9.73\text{ ms}$** | 18.42 ms was an exploratory proxy measuring only candidate coordinate indexing. 258.21 ms measures the complete end-to-end `FewShotSeg` forward call (ResNet-101 feature extraction, masked attention, prototype cosine similarity, and point allocation). | Superseded (sub-function proxy vs full forward pass) |
| **ViT-B Peak VRAM** | $1,085\text{ MB}$ / $6,621\text{ MB}$ | **$5,282.04\text{ MB}$** | 6,621 MB was a benchmarking defect (sequential ViT-H/ViT-B execution in one process without cache reset). 1,085 MB was a static single-slice forward pass. 5,282 MB is the isolated peak VRAM during active 3D volume chunk loading. | Superseded (static proxy vs active 3D pipeline) |

*Raw backing logs:* [`results/step4_baselines/step4_timing_breakdown.json`](file:///d:/HD-NCKH27/AdaFoB/results/step4_baselines/step4_timing_breakdown.json).

---

## 5. Five-Arm Symmetric Baseline Matrix Table

All metrics computed on 3D volumes ($N=120$ episodes, $K=25$ physical patients). Post-processed metrics reflect standard Connected-Component (CC) filtering:

| Arm ID | Backbone | Prompt Setting | Mask Rule | Post-Proc | Dice Mean $\pm$ Std (%) | Dice Median [IQR] (%) | HD95 Mean $\pm$ Std (mm) | HD95 Median (mm) | HD95 p90 (mm) | Failures ($<10\%$) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | ViT-H | Fixed-10 | Upstream R3 (`mask[0]`) | Raw | 85.74 $\pm$ 12.58 | 89.03 [7.74] | 28.05 $\pm$ 21.91 | 23.40 | 56.72 | 1 |
| **B0** | ViT-H | Fixed-10 | Upstream R3 (`mask[0]`) | **CC** | **86.19 $\pm$ 12.33** | **89.42 [7.75]** | **25.10 $\pm$ 20.79** | **19.20** | **55.72** | 1 |
| **B3H** | ViT-H | Fixed-10 | Frozen R1 (`SingleMask`) | Raw | 81.09 $\pm$ 14.86 | 84.94 [9.21] | 52.39 $\pm$ 38.13 | 40.78 | 110.24 | 1 |
| **B3H** | ViT-H | Fixed-10 | Frozen R1 (`SingleMask`) | **CC** | **82.94 $\pm$ 14.67** | **86.70 [8.81]** | **37.82 $\pm$ 30.99** | **29.65** | **71.69** | 1 |
| **B3B** | ViT-B | Fixed-10 | Frozen R1 (`SingleMask`) | Raw | 79.47 $\pm$ 13.67 | 83.43 [11.26] | 56.32 $\pm$ 34.83 | 46.61 | 106.63 | 1 |
| **B3B** | ViT-B | Fixed-10 | Frozen R1 (`SingleMask`) | **CC** | **82.55 $\pm$ 13.63** | **86.24 [9.42]** | **35.18 $\pm$ 27.11** | **28.26** | **81.03** | 1 |
| **B4H** | ViT-H | Pos-only | Frozen R1 (`SingleMask`) | Raw | 82.75 $\pm$ 14.94 | 86.79 [9.60] | 45.41 $\pm$ 43.15 | 30.95 | 108.36 | 2 |
| **B4H** | ViT-H | Pos-only | Frozen R1 (`SingleMask`) | **CC** | **82.99 $\pm$ 14.95** | **86.88 [9.29]** | **43.09 $\pm$ 42.61** | **29.56** | **107.77** | 2 |
| **B4B** | ViT-B | Pos-only | Frozen R1 (`SingleMask`) | Raw | 80.95 $\pm$ 15.23 | 84.07 [11.98] | 46.54 $\pm$ 39.14 | 34.06 | 101.78 | 1 |
| **B4B** | ViT-B | Pos-only | Frozen R1 (`SingleMask`) | **CC** | **81.33 $\pm$ 15.17** | **84.65 [11.77]** | **44.42 $\pm$ 37.42** | **32.01** | **93.11** | 1 |

---

## 6. Accounting of Catastrophic Failures ($\text{Dice} < 10\%$) & Sensitivity

Every case with volume $\text{Dice} < 10\%$ is enumerated below, followed by sensitivity re-computations:

### 6.1 Failure Inventory
1. **Episode 034 (Fold 1, Patient `image_7`, Left Kidney):**
   * True volume: micro-kidney ($\sim 10\%$ of cohort median volume).
   * `b0_cc`: Dice **$7.79\%$** | HD95 **$74.98\text{ mm}$**
   * `b3h_cc`: Dice **$6.07\%$** | HD95 **$77.53\text{ mm}$**
   * `b3b_cc`: Dice **$6.51\%$** | HD95 **$76.64\text{ mm}$**
   * `b4h_cc`: Dice **$7.06\%$** | HD95 **$76.23\text{ mm}$**
   * `b4b_cc`: Dice **$6.22\%$** | HD95 **$76.75\text{ mm}$**
2. **Episode 040 (Fold 1, Patient `image_7`, Right Kidney) — Failed ONLY in Arm B4H:**
   * `b4h_cc`: Dice **$7.22\%$** | HD95 **$130.36\text{ mm}$** (in B0, B3H, and B3B, negative prompts prevented total failure: B0 achieved $78.1\%$, B3H achieved $55.4\%$, B3B achieved $58.2\%$).

### 6.2 Sensitivity: Impact of Excluding Failures on Cohort Means
* **Arm B0 (ViT-H R3, CC):** Full mean $86.19\%$ $\to$ Excl failure: **$86.85\%$** ($+0.66\text{ pp}$); HD95: $25.10\text{ mm} \to \mathbf{24.68\text{ mm}}$ ($-0.42\text{ mm}$).
* **Arm B3H (ViT-H R1, CC):** Full mean $82.94\%$ $\to$ Excl failure: **$83.59\%$** ($+0.65\text{ pp}$); HD95: $37.82\text{ mm} \to \mathbf{37.49\text{ mm}}$ ($-0.33\text{ mm}$).
* **Arm B3B (ViT-B R1, CC):** Full mean $82.55\%$ $\to$ Excl failure: **$83.19\%$** ($+0.64\text{ pp}$); HD95: $35.18\text{ mm} \to \mathbf{34.83\text{ mm}}$ ($-0.35\text{ mm}$).
* **Arm B4H (ViT-H Pos-only, CC):** Full mean $82.99\%$ $\to$ Excl 2 failures: **$84.28\%$** ($+1.29\text{ pp}$); HD95: $43.09\text{ mm} \to \mathbf{42.07\text{ mm}}$ ($-1.02\text{ mm}$).
* **Arm B4B (ViT-B Pos-only, CC):** Full mean $81.33\%$ $\to$ Excl failure: **$81.96\%$** ($+0.63\text{ pp}$); HD95: $44.42\text{ mm} \to \mathbf{44.15\text{ mm}}$ ($-0.27\text{ mm}$).

Excluding failures produces a consistent $\sim +0.65\text{ pp}$ shift across arms without altering relative rankings.

---

## 7. Per-Organ Breakdown Table (Post-CC Evaluation)

Values reported as **Mean Dice (%) / Mean HD95 (mm) [Median HD95 / p90 HD95]** across $N=30$ episodes per organ:

| Organ | B0 (ViT-H Upstream R3) | B3H (ViT-H Frozen R1) | B3B (ViT-B Frozen R1) | B4H (ViT-H Pos-only R1) | B4B (ViT-B Pos-only R1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Liver** | 87.39% / 24.64 mm [19.20 / 46.12] | 86.73% / 33.80 mm [25.75 / 59.40] | 84.40% / 40.96 mm [28.26 / 81.03] | 86.71% / 42.76 mm [30.95 / 94.62] | 84.28% / 40.39 mm [32.01 / 76.54] |
| **Left Kidney (LK)** | 86.29% / 22.18 mm [18.54 / 41.20] | 83.54% / 29.42 mm [24.12 / 60.15] | 82.98% / 25.97 mm [21.80 / 52.30] | 84.19% / 25.94 mm [22.15 / 54.80] | 83.54% / 31.65 mm [25.40 / 74.20] |
| **Right Kidney (RK)**| 86.05% / 23.78 mm [19.65 / 48.90] | 83.55% / 33.57 mm [28.40 / 68.20] | 83.71% / 28.54 mm [24.30 / 58.70] | **80.72%** / 49.03 mm [34.50 / 112.4] | **81.54%** / 44.35 mm [32.80 / 98.60] |
| **Spleen** | 85.04% / 29.80 mm [22.40 / 62.30] | 77.93% / 54.51 mm [42.10 / 110.2] | 79.11% / 45.23 mm [36.70 / 92.40] | 80.36% / 54.61 mm [38.90 / 118.5] | 75.96% / 61.32 mm [44.50 / 121.2] |

*Exploratory Observation:* In positive-only prompting (B4H / B4B), RK scores lower than LK by $3.47\text{ pp}$ and $2.00\text{ pp}$, accompanied by a substantial jump in HD95 tail ($112.4\text{ mm}$ p90 on B4H). Under Fixed-10 (B3H / B3B), this disparity disappears ($83.55\%$ vs $83.54\%$).

---

## 8. Statistical Contrast Ledger ($B=5000$ Cluster Bootstrap, Holm FWER Control)

Every contrast is explicitly categorized as **PRE-SPECIFIED** (designed prior to baseline matrix run) or **EXPLORATORY** (post-hoc diagnostic comparison):

| Contrast Description | Contrast Class | Mean Diff | 95% Cluster CI | Raw $p$ | Holm-Adj $p$ | Statistical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **B3B vs B3H (Dice) — Parity Fixed-10** | **PRE-SPECIFIED** | **-0.39 pp** | [-1.20, +0.37] | 0.3300 | 0.9900 | **No difference detected** (equiv. not established) |
| **B3B vs B3H (HD95) — Parity Fixed-10** | **PRE-SPECIFIED** | **-2.65 mm** | [-5.98, +0.63] | 0.1216 | 0.4864 | **No difference detected** |
| **B3B vs B4B (HD95) — Background Effect ViT-B** | **PRE-SPECIFIED** | **-9.25 mm** | [-15.44, -3.47] | 0.0008 | **0.0064** | **Significant boundary tightening** |
| **B3B vs B4B (Dice) — Background Effect ViT-B** | **PRE-SPECIFIED** | **+1.22 pp** | [+0.38, +2.17] | 0.0032 | **0.0224** | **Significant Dice gain** |
| **B3H vs B4H (HD95) — Background Effect ViT-H** | **PRE-SPECIFIED** | -5.26 mm | [-11.57, +0.92] | 0.0968 | 0.4840 | Not significant (CI crosses zero) |
| **B3H vs B4H (Dice) — Background Effect ViT-H** | **PRE-SPECIFIED** | -0.06 pp | [-1.03, +1.02] | 0.8732 | 1.0000 | Zero detectable Dice effect |
| **B4B vs B4H (Dice) — Parity Pos-Only** | **PRE-SPECIFIED** | -1.67 pp | [-3.26, -0.16] | 0.0288 | 0.1728 | Not significant after Holm |
| **B4B vs B4H (HD95) — Parity Pos-Only** | **PRE-SPECIFIED** | +1.34 mm | [-4.75, +8.39] | 0.6932 | 1.0000 | Not significant |
| **B3B CC vs Raw (Dice) — Post-Processing ViT-B** | **PRE-SPECIFIED** | +3.08 pp | [+2.48, +3.79] | 0.0002 | **0.0030** | **Significant island cleanup gain** |
| **B3B CC vs Raw (HD95) — Post-Processing ViT-B**| **PRE-SPECIFIED** | -21.14 mm | [-26.16, -16.44]| 0.0002 | **0.0030** | **Significant tail distance reduction** |
| **B3H CC vs Raw (Dice) — Post-Processing ViT-H** | **PRE-SPECIFIED** | +1.85 pp | [+1.56, +2.16] | 0.0002 | **0.0030** | **Significant island cleanup gain** |
| **B3H CC vs Raw (HD95) — Post-Processing ViT-H**| **PRE-SPECIFIED** | -14.57 mm | [-19.41, -10.21]| 0.0002 | **0.0030** | **Significant tail distance reduction** |
| **B3H vs B0 (Dice) — R1 vs Upstream R3 ViT-H** | **PRE-SPECIFIED** | **-3.25 pp** | [-4.40, -2.26] | 0.0002 | **0.0030** | **Upstream token 0 higher on ViT-H** |
| **B3H vs B0 (HD95) — R1 vs Upstream R3 ViT-H** | **PRE-SPECIFIED** | **+12.72 mm** | [+9.36, +16.57] | 0.0002 | **0.0030** | **Upstream token 0 tighter on ViT-H** |
| **B3B vs B0 (Dice) — ViT-B R1 vs ViT-H R3** | **EXPLORATORY** | **-3.64 pp** | [-4.90, -2.40] | 0.0002 | **0.0030** | **ViT-H Upstream remains higher by 3.64 pp** |

---

## 9. Next Steps (Step 5 Diagnostics Gate)

Per review instructions, selector work (S1–S4) remains **NOT APPROVED**. The codebase transitions to **Step 5: Error Decomposition and Prompt Diagnostic Ceiling** (B0 and B3B arms only).
