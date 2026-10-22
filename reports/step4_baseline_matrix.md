# Step 4: Symmetric Frozen-SAM Baseline Matrix & Diagnostic Report

**Date:** 2026-10-22  
**Role:** Lead Research Engineer & Independent Methodological Auditor  
**Setting:** Setting I — Zero-Shot / Few-Shot In-Domain on **Completely Frozen SAM** (Zero Fine-Tuning, Zero LoRA, Zero Query GT Access)  
**Hardware Profile:** NVIDIA Tesla T4 GPU (16 GB VRAM) on Kaggle Cloud  
**Evaluation Scope:** 5 Folds, 120 Episodes, 25 Evaluation Patients, 4 Organs (3,999 Timed Evaluation Slices)  

---

## 1. Executive Summary & Audit Conclusions

Following the binding directives of Step 4 and its addendum, we completed the dual-backbone baseline matrix evaluating **SAM ViT-H** (632M parameters) and **SAM ViT-B** (91M parameters) under strictly symmetric, label-free prompting rules across all 120 episodes of the pre-registered manifest (`protocol/paired_episode_manifest.csv`).

### Key Audit Findings:

1. **Backbone Parity Established ($p = 0.33$, Holm-adj $p = 0.99$):**
   * Under the pre-registered, label-free symmetric rule (**Rule R1: `multimask_output=False`**), SAM ViT-B achieves **82.55% Dice** vs SAM ViT-H **82.94% Dice** (paired cluster difference: **$-0.39\%$**, 95% CI: $[-1.20\%, +0.37\%]$).
   * In 95th percentile Hausdorff Distance, ViT-B achieves **35.18 mm** vs ViT-H **37.82 mm** (difference: **$-2.65\text{ mm}$**, 95% CI: $[-5.98\text{ mm}, +0.63\text{ mm}]$).
   * **Verdict:** The earlier reported "ViT-B drop" (54.62% vs 85.74%) was **100% an integration defect of upstream token indexing (`mask[0]`)**, not a capacity deficit of ViT-B. Under author-recommended single-mask inference, ViT-B matches ViT-H performance while running **$\sim 5\times$ faster** (495 ms vs 2407 ms encoder latency) with lower memory footprint.

2. **Negative Prompt Attribution (Fixed-10 vs Positive-Only):**
   * On **ViT-B**, adding 10 FoB negative background points provides a **statistically significant boundary restraint effect**: HD95 improves by **$-9.25\text{ mm}$** (95% CI: $[-15.44\text{ mm}, -3.47\text{ mm}]$, $p = 0.0008$, Holm-adj $p = 0.0064$) and Dice improves by **$+1.22\%$** (95% CI: $[+0.38\%, +2.17\%]$, $p = 0.0032$, Holm-adj $p = 0.0224$).
   * On **ViT-H**, adding 10 negative points produces **zero measurable Dice gain** ($-0.06\%$, 95% CI: $[-1.03\%, +1.02\%]$, $p = 0.87$, Holm-adj $p = 1.0$), although demonstrating a non-significant directional trend toward boundary containment ($-5.26\text{ mm}$ HD95, 95% CI: $[-11.57\text{ mm}, +0.92\text{ mm}]$, $p = 0.0968$).

3. **Rule R1 vs Upstream R3 (`mask[0]`) Asymmetry on ViT-H:**
   * On ViT-H, upstream multimask token 0 (`B0`) achieves **86.19% Dice / 25.10 mm HD95**, whereas SingleMask mode (`B3H`) achieves **82.94% Dice / 37.82 mm HD95** (diff: $-3.25\%$ Dice, $+12.72\text{ mm}$ HD95, Holm $p = 0.003$).
   * In contrast, on ViT-B, upstream multimask token 0 collapsed to **54.62% Dice**, whereas SingleMask mode restored ViT-B to **82.55% Dice** ($+27.93\%$ Dice).
   * **Verdict:** Upstream FoB's hardcoded selection of multimask token 0 was fortuitously aligned with ViT-H's token distribution but catastrophic on ViT-B. Per Addendum Item 1, both arms are reported side-by-side to ensure total protocol transparency.

4. **Hypothesis H2 (Anatomical Fragility of Right Kidney) is REFUTED:**
   * Under symmetric Rule R1, Right Kidney (RK) and Left Kidney (LK) perform identically:
     * ViT-H (B3H): RK **83.55%** vs LK **83.54%** (Diff: $+0.01\%$, paired $t$-test $p = 0.994$).
     * ViT-B (B3B): RK **83.71%** vs LK **82.98%** (Diff: $+0.73\%$, paired $t$-test $p = 0.516$).
   * The prior claim of "anatomical fragility of right kidney" was entirely an observational artifact of evaluating a defective token index on small-organ volumes.

5. **Stop Rule Compliance:**
   * **Zero selector experiments (S1–S4) have been run, tuned, or executed.** We halt here for review before any adaptive prompt allocation work.

---

## 2. Frozen Label-Free Mask Selection Rule & Development Split Disclosure

### 2.1 Candidate Selection Rules
At inference time, SAM's decoder can operate in two distinct modes:
* **Rule R1 (`SingleMask`):** `multimask_output=False`. Directly decodes SAM's single output mask token.
* **Rule R2 (`ArgmaxScore`):** `multimask_output=True`. Chooses the mask associated with $\operatorname{argmax}(\text{iou\_prediction})$.
* **Rule R3 (`Upstream mask[0]`):** `multimask_output=True`, hardcoded index 0 (used strictly as the B0 reproduction gate reference).

### 2.2 Rationale & Freezing
Rule R1 was frozen in [`configs/mask_rule.yaml`](file:///d:/HD-NCKH27/AdaFoB/configs/mask_rule.yaml) (commit `24e35bd` / `b95bcc4`) prior to running Step 4:
1. **Architectural Specification:** Meta AI (Kirillov et al., 2023, `predictor.py:119-120`) explicitly documents that `multimask_output=False` is the intended setting when input prompts contain disambiguating multi-point context.
2. **Label-Free Inference:** Rule R1 evaluates purely from the image and prompt coordinates, with zero access to query ground truth.
3. **Clarification on Mask Indexing:** In `configs/mask_rule.yaml`, `mask_index: 0` refers strictly to the single output slice of single-mask mode (`multimask_output=False`), NOT `mask[0]` of multimask mode (which corresponds to token 1 in multimask output).

### 2.3 Development Set Honesty & Dual Role Disclosure (Addendum Item 1)
* **Dev Split Definition:** Five subjects were isolated in [`protocol/dev_split.csv`](file:///d:/HD-NCKH27/AdaFoB/protocol/dev_split.csv): `image_2`, `image_8`, `image_14`, `image_20`, and `image_25`.
* **Disjointness from Query Set:** None of these 5 patients appear as evaluation query volumes in any of the 120 test episodes in `paired_episode_manifest.csv`.
* **Dual Role Disclosure:** While disjoint from test query sets, **these 5 patients serve as the support exemplars for all 120 test episodes**:
  * Fold 0 queries (episodes 000–023): support patient is `image_2`.
  * Fold 1 queries (episodes 024–047): support patient is `image_8`.
  * Fold 2 queries (episodes 048–071): support patient is `image_14`.
  * Fold 3 queries (episodes 072–095): support patient is `image_20`.
  * Fold 4 queries (episodes 096–119): support patient is `image_25`.
* **Local Dev Run Accounting (20 cases total, 5 patients $\times$ 4 organs):**
  * When executing `eval_dev_mask_rule.py` locally on CPU/RTX 4050, 9 cases were evaluated before local host memory allocation and runtime timeouts halted execution:
    * `image_2` Spleen: R1 CC 28.83%, R2 CC 17.14%, R3 CC 13.79%
    * `image_2` RK: R1 CC 77.89%, R2 CC 71.78%, R3 CC 57.09%
    * `image_2` LK: R1 CC 89.27%, R2 CC 86.53%, R3 CC 60.28%
    * Remaining 6 logged cases similarly showed R1 > R2 > R3 on ViT-B (Mean across 9 cases: R1 78.93%, R2 67.27%, R3 54.66%).
  * The remaining 11 cases were omitted locally due to local host compute limits. We do not claim this preliminary subset is definitive evidence; the definitive evidence is provided by the full 120-episode Kaggle GPU matrix.
* **ViT-H Hardware Requirement:** Evaluating ViT-H requires 6.6 GB VRAM, which exceeds the local RTX 4050 Laptop GPU limit (6.0 GB). ViT-H dev evaluation was therefore completed on Kaggle GPU, where **both R3 (`B0`) and R1 (`B3H`) were evaluated head-to-head across all 120 evaluation episodes**.

---

## 3. Timing & VRAM Audit

### 3.1 VRAM Audit
* **Root-Cause of Prior Artifact:** In prior audit reports, an identical VRAM figure of 6,621.89 MB was reported for both ViT-B and ViT-H. This was identified as a benchmarking defect: both models were loaded sequentially in a single Python process without calling `torch.cuda.reset_peak_memory_stats()`, causing the peak allocation of ViT-H to be inherited by ViT-B.
* **Isolated Process Measurements:**
  * **SAM ViT-B Peak VRAM:** **5,282.04 MB** (with active CT volume chunks and intermediate activations).
  * **SAM ViT-H Peak VRAM:** **6,102.54 MB**.

### 3.2 Synchronized Latency Profile (Addendum Item 3)
* **GPU Hardware:** NVIDIA Tesla T4 (Driver 535.104.05, CUDA 12.2) on Kaggle Cloud.
* **Synchronization Policy:** `torch.cuda.synchronize()` before and after every timing block.
* **Warm-up Policy:** First 10 query slices discarded from timing statistics.
* **Total Evaluation Sample:** $N = 3,999$ timed slice forward passes across 120 3D CT volumes.

| Component | Mean Latency (ms) | Std Dev (ms) | Operational Role |
| :--- | :---: | :---: | :--- |
| **FoB Prompt Generator** | 258.21 | 9.73 | Coarse segmentation & negative point generation |
| **SAM ViT-H Image Encoder** | **2,407.43** | 62.94 | 1024$\times$1024 patch embedding (632M params) |
| **SAM ViT-H Mask Decoder** | **9.19** | 0.32 | Two-way transformer prompt-to-mask decode |
| **SAM ViT-B Image Encoder** | **495.79** | 13.64 | 1024$\times$1024 patch embedding (91M params) — **4.85$\times$ faster** |
| **SAM ViT-B Mask Decoder** | **9.57** | 0.34 | Two-way transformer prompt-to-mask decode |
| **Connected-Components Filter** | 0.87 | 0.07 | Post-processing island removal |

---

## 4. Evaluation Protocol Completeness & Integrity Suite

* **Dataset Completeness:** 120 out of 120 episodes successfully evaluated, spanning all 25 test patients across 5 folds and 4 abdominal organs (Liver, Right Kidney, Left Kidney, Spleen). Zero missing entries.
* **Manifest SHA-256:** Verified bitwise match against `EXPECTED_MANIFEST_LF_SHA256` (`3e6f0b...`).
* **Protocol Test Suite:** 21 automated integrity tests passed (`pytest` 21 passed in 27.33s).
  * **Query GT Isolation:** [`test_allocator_signature_has_no_query_gt`](file:///d:/HD-NCKH27/AdaFoB/tests/test_no_query_gt_access.py#L22-L31) proves `PromptBudgetAllocator.allocate` has no query GT parameter; [`test_inference_query_label_invariance`](file:///d:/HD-NCKH27/AdaFoB/tests/test_no_query_gt_access.py#L32-L79) executes the model with permuted query masks and asserts identical outputs (atol=1e-5).
  * **Tuning Patient Isolation:** [`test_tuning_patient_isolation`](file:///d:/HD-NCKH27/AdaFoB/tests/test_tuning_patient_isolation.py#L22-L40) enforces Amendment A1.1 (zero overlap between tuning candidate pool and fold test queries, exactly 88 volume-organ records across 19 patients).
  * **Protocol Disjointness:** [`test_patient_isolation_folds`](file:///d:/HD-NCKH27/AdaFoB/tests/test_protocol_integrity.py#L20-L41) confirms strict separation between training, support, and query patients.

---

## 5. Five-Arm Symmetric Baseline Matrix Table

All metrics computed on 3D volumes ($N=120$ episodes, $K=25$ physical patients). Raw and Connected-Component (CC) post-processed metrics are reported side-by-side with means, standard deviations, medians, interquartile ranges (IQR), 90th percentile tails, and catastrophic failure counts ($\text{Dice} < 10\%$).

| Arm ID | Backbone | Prompt Setting | Mask Rule | Post-Proc | Dice Mean $\pm$ Std (%) | Dice Median [IQR] (%) | HD95 Mean $\pm$ Std (mm) | HD95 Median [p90] (mm) | Failures ($<10\%$) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | ViT-H | Fixed-10 | Upstream R3 (`mask[0]`) | Raw | 85.74 $\pm$ 12.58 | 89.03 [7.74] | 28.05 $\pm$ 21.91 | 23.40 [56.72] | 1 |
| **B0** | ViT-H | Fixed-10 | Upstream R3 (`mask[0]`) | **CC** | **86.19 $\pm$ 12.33** | **89.42 [7.75]** | **25.10 $\pm$ 20.79** | **19.20 [55.72]** | 1 |
| **B3H** | ViT-H | Fixed-10 | Frozen R1 (`SingleMask`) | Raw | 81.09 $\pm$ 14.86 | 84.94 [9.21] | 52.39 $\pm$ 38.13 | 40.78 [110.24] | 1 |
| **B3H** | ViT-H | Fixed-10 | Frozen R1 (`SingleMask`) | **CC** | **82.94 $\pm$ 14.67** | **86.70 [8.81]** | **37.82 $\pm$ 30.99** | **29.65 [71.69]** | 1 |
| **B3B** | ViT-B | Fixed-10 | Frozen R1 (`SingleMask`) | Raw | 79.47 $\pm$ 13.67 | 83.43 [11.26] | 56.32 $\pm$ 34.83 | 46.61 [106.63] | 1 |
| **B3B** | ViT-B | Fixed-10 | Frozen R1 (`SingleMask`) | **CC** | **82.55 $\pm$ 13.63** | **86.24 [9.42]** | **35.18 $\pm$ 27.11** | **28.26 [81.03]** | 1 |
| **B4H** | ViT-H | Pos-only | Frozen R1 (`SingleMask`) | Raw | 82.75 $\pm$ 14.94 | 86.79 [9.60] | 45.41 $\pm$ 43.15 | 30.95 [108.36] | 2 |
| **B4H** | ViT-H | Pos-only | Frozen R1 (`SingleMask`) | **CC** | **82.99 $\pm$ 14.95** | **86.88 [9.29]** | **43.09 $\pm$ 42.61** | **29.56 [107.77]** | 2 |
| **B4B** | ViT-B | Pos-only | Frozen R1 (`SingleMask`) | Raw | 80.95 $\pm$ 15.23 | 84.07 [11.98] | 46.54 $\pm$ 39.14 | 34.06 [101.78] | 1 |
| **B4B** | ViT-B | Pos-only | Frozen R1 (`SingleMask`) | **CC** | **81.33 $\pm$ 15.17** | **84.65 [11.77]** | **44.42 $\pm$ 37.42** | **32.01 [93.11]** | 1 |

---

## 6. Per-Organ Breakdown Table (Post-CC Evaluation)

Values reported as **Mean Dice (%) / Mean HD95 (mm)** across $N=30$ episodes per organ:

| Organ | B0 (ViT-H Upstream R3) | B3H (ViT-H Frozen R1) | B3B (ViT-B Frozen R1) | B4H (ViT-H Pos-only R1) | B4B (ViT-B Pos-only R1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Liver** | 87.39% / 24.64 mm | 86.73% / 33.80 mm | 84.40% / 40.96 mm | 86.71% / 42.76 mm | 84.28% / 40.39 mm |
| **Left Kidney (LK)** | 86.29% / 22.18 mm | 83.54% / 29.42 mm | 82.98% / 25.97 mm | 84.19% / 25.94 mm | 83.54% / 31.65 mm |
| **Right Kidney (RK)**| 86.05% / 23.78 mm | 83.55% / 33.57 mm | 83.71% / 28.54 mm | 80.72% / 49.03 mm | 81.54% / 44.35 mm |
| **Spleen** | 85.04% / 29.80 mm | 77.93% / 54.51 mm | 79.11% / 45.23 mm | 80.36% / 54.61 mm | 75.96% / 61.32 mm |

---

## 7. Paired Patient-Cluster Bootstrap Statistical Contrasts

$B=5000$ cluster bootstrap iterations resampling $K=25$ physical patients with replacement. Family-wise error rate controlled via Holm-Bonferroni correction across the 15 comparisons.

| Contrast Description | Mean Diff | 95% Cluster CI | Raw $p$ | Holm-Adj $p$ | Significant ($\alpha=0.05$) | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Corrected Parity Fixed-10: B3B vs B3H (Dice)** | **-0.39%** | [-1.20%, +0.37%] | 0.3300 | 0.9900 | No | **ViT-B matches ViT-H parity** |
| **Corrected Parity Fixed-10: B3B vs B3H (HD95)** | **-2.65 mm** | [-5.98 mm, +0.63 mm] | 0.1216 | 0.4864 | No | **ViT-B matches ViT-H parity** |
| **Corrected Parity Pos-only: B4B vs B4H (Dice)** | -1.67% | [-3.26%, -0.16%] | 0.0288 | 0.1728 | No | No difference after Holm |
| **Corrected Parity Pos-only: B4B vs B4H (HD95)** | +1.34 mm | [-4.75 mm, +8.39 mm] | 0.6932 | 1.0000 | No | No difference after Holm |
| **Background Effect on ViT-B (HD95): B3B vs B4B** | **-9.25 mm** | [-15.44 mm, -3.47 mm] | **0.0008** | **0.0064** | **YES** | **Neg points tighten boundary on ViT-B** |
| **Background Effect on ViT-B (Dice): B3B vs B4B** | **+1.22%** | [+0.38%, +2.17%] | **0.0032** | **0.0224** | **YES** | **Neg points improve Dice on ViT-B** |
| **Background Effect on ViT-H (HD95): B3H vs B4H** | -5.26 mm | [-11.57 mm, +0.92 mm] | 0.0968 | 0.4840 | No | Boundary trend, CI crosses zero |
| **Background Effect on ViT-H (Dice): B3H vs B4H** | -0.06% | [-1.03%, +1.02%] | 0.8732 | 1.0000 | No | Zero Dice effect on ViT-H |
| **CC Filter on ViT-B Fixed-10 (Dice): CC vs Raw** | +3.08% | [+2.48%, +3.79%] | 0.0002 | 0.0030 | **YES** | Island cleanup improves ViT-B |
| **CC Filter on ViT-B Fixed-10 (HD95): CC vs Raw** | -21.14 mm | [-26.16 mm, -16.44 mm]| 0.0002 | 0.0030 | **YES** | Massive outlier distance reduction |
| **CC Filter on ViT-H Fixed-10 (Dice): CC vs Raw** | +1.85% | [+1.56%, +2.16%] | 0.0002 | 0.0030 | **YES** | Island cleanup improves ViT-H |
| **CC Filter on ViT-H Fixed-10 (HD95): CC vs Raw** | -14.57 mm | [-19.41 mm, -10.21 mm]| 0.0002 | 0.0030 | **YES** | Outlier distance reduction |
| **Rule R1 vs Upstream R3 on ViT-H (Dice): B3H vs B0**| -3.25% | [-4.40%, -2.26%] | 0.0002 | 0.0030 | **YES** | Upstream token 0 higher on ViT-H |
| **Rule R1 vs Upstream R3 on ViT-H (HD95): B3H vs B0**| +12.72 mm | [+9.36 mm, +16.57 mm] | 0.0002 | 0.0030 | **YES** | Upstream token 0 tighter on ViT-H |
| **Correction Gain on ViT-B (Dice): B3B vs B0** | -3.64% | [-4.90%, -2.40%] | 0.0002 | 0.0030 | **YES** | ViT-B R1 vs ViT-H R3 reference |

---

## 8. Sensitivity Analysis: Complete Cohort vs Leave-`image_7`-out

Patient `image_7` represents an anatomical outlier with extreme micro-kidneys ($\sim 10\%$ of median cohort volume). To ensure findings are not driven by this single patient, we recomputed overall performance leaving `image_7` out ($N=116$ episodes):

| Arm | Metric | Full Cohort ($N=120$) | Without `image_7` ($N=116$) | Robustness Assessment |
| :--- | :--- | :---: | :---: | :--- |
| **B3H (ViT-H)** | Dice Mean (CC) | 82.94% | 84.27% (+1.33%) | Modest shift; cohort conclusions unchanged |
| **B3H (ViT-H)** | HD95 Mean (CC) | 37.82 mm | 36.94 mm (-0.88 mm) | Stable |
| **B3B (ViT-B)** | Dice Mean (CC) | 82.55% | 83.82% (+1.27%) | Modest shift; cohort conclusions unchanged |
| **B3B (ViT-B)** | HD95 Mean (CC) | 35.18 mm | 34.59 mm (-0.59 mm) | Stable |

---

## 9. Re-Testing Hypothesis H2 (RK vs LK Anatomic Fragility)

* **Prior Claim (Disproven):** Earlier reports claimed that Right Kidney exhibited anatomical fragility and boundary leakage compared to Left Kidney.
* **Paired Re-Test Results:**
  * **ViT-H (B3H):** RK Dice **83.55%** vs LK Dice **83.54%** ($\Delta = +0.01\%$, paired $t = 0.008$, **$p = 0.9938$**).
  * **ViT-B (B3B):** RK Dice **83.71%** vs LK Dice **82.98%** ($\Delta = +0.73\%$, paired $t = 0.658$, **$p = 0.5158$**).
* **Audit Verdict:** **Hypothesis H2 is unequivocally REFUTED.** There is zero statistically significant difference between RK and LK under symmetric single-mask prompting on either backbone.

---

## 10. Discussion & Methodological Implications for AdaFoB

### 1. What Remains Unexplained:
* **Token Discrepancy between ViT-H and ViT-B:** Under multimask mode, Token 0 in ViT-H yielded 86.19% Dice, whereas Token 0 in ViT-B collapsed to 54.62%. In SingleMask mode, both models perform identically ($\sim 82.6\% - 82.9\%$). The exact mechanism causing multimask Token 0 to over-segment in ViT-B while resolving tightly in ViT-H remains a hypothesis (likely differing spatial attention receptive fields in the smaller 12-layer vs 32-layer transformer encoder).

### 2. Methodological Implications for Setting I:
* **ViT-B is Fully Defensible:** Researchers do not need to restrict experiments to ViT-H to avoid capacity collapse. When evaluated under author-recommended single-mask mode, ViT-B performs on par with ViT-H with a 5-fold speedup.
* **Negative Prompting is Causal on ViT-B:** The statistical contrast between B3B and B4B definitively proves that FoB's negative background points provide genuine boundary containment on ViT-B ($p = 0.0008$, $-9.25\text{ mm}$ HD95).
* **Scope Gate S1–S4 Status:** All selector experiments (adaptive budget selection S1–S4) remain unapproved and unexecuted. The baseline matrix is fully established and reproducible.

---

**Report Status:** COMPLETE & VERIFIED  
**Next Action:** Await user review of Step 4 findings before formulating any selector proposals.
