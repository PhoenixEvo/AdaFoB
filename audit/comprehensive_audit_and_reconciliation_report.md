# Comprehensive Methodological Audit, Statistical Reconciliation, and Strategic Roadmap for AdaFoB

**Lead Research Engineer & Independent Methodological Auditor**  
**Repository Commit:** `88914bc` | **Date:** 2026-10-21  
**Project:** AdaFoB (Few-Shot Medical Image Segmentation with SAM)

---

## 1. Executive Summary & Audit Gate Scorecard

An exhaustive audit of the AdaFoB repository, git history, data loaders, training routines, evaluation scripts, and historical result CSVs was conducted. The findings reveal critical structural violations in prior protocols alongside rigorous statistical truths about the method's behavior.

### Audit Gate Decisions

| Gate | Focus | Verdict | Core Justification |
| :--- | :--- | :--- | :--- |
| **Gate 1** | Protocol & Data Separation | **BLOCKED** | Fatal supervisory leakage (Violation V1): `experiments/train_lora.py` supervised LoRA and mask decoder directly on target organ ground truth masks, turning few-shot novel-class segmentation into standard supervised in-domain fine-tuning. Validation was performed directly on test-fold query patients (Violation V2), and allocator hyper-parameters were tuned to maximize test-fold Dice (Violation V3). |
| **Gate 2** | Automated Verification Harness | **PASSED** | 16 of 16 automated tests pass cleanly across 6 suites (`tests/test_protocol_integrity.py`, `tests/test_spatial_transforms.py`, `tests/test_metrics.py`, `tests/test_evaluation_completeness.py`, `tests/test_no_query_gt_access.py`, `tests/test_fixed_budget_parity.py`). Coordinate systems, metric definitions, anisotropic voxel scaling, and inference query-label invariance are verifiably sound. |
| **Gate 3** | Baseline & Methodological Validity | **CONDITIONAL** | Pre-specified TOST proves Dice equivalence between AdaFoB and FoB+LoRA within $\pm 1.0\%$. Under Holm-Bonferroni correction, HD95 difference is not statistically significant ($p_{\text{adj}} = 0.0660$). The ambiguity score has zero correlation with optimal budget ($r=0.035, p=0.58$). Main-track MICCAI submission is **NOT viable** in current form without protocol pivot. |

---

## 2. Reconciliation of the 7 Legacy Issues

### 2.1 Rank 16 Discrepancy & CSV Contamination
- **Issue:** Historical reports cited Rank 16 as Dice $89.91\%$, HD95 $27.92$ mm, whereas the clean unseeded re-run yielded Dice $89.10\%$, HD95 $48.59$ mm.
- **Root Cause Verified:** Historical CSV `results/lora_eval_rank16_cv.csv` suffered from parallel Kaggle script output file collisions. 78.3% of rows were identical duplicates copied across folds. The file has been formally marked `[DO NOT CITE - CONTAMINATED]`.
- **True Clean Performance:** Under the verified clean unseeded run (`results/clean_rerun_rank16_unseeded_combined.csv`), Rank 16 achieves Dice $89.10\%$ and mean HD95 $48.59$ mm (median $10.14$ mm, IQR $45.61$ mm). The extreme jump in mean HD95 is driven by isolated false-positive outliers (>100 mm).

### 2.2 LoRA Parameter Count Reconciliation
- **Formula & Layer Scope:**
  - SAM ViT-B has $L=12$ transformer blocks, hidden dimension $d=768$.
  - In `experiments/lora_sam.py`, LoRA is applied to the merged `qkv` projection:
    $$\Delta W = B \cdot A, \quad A \in \mathbb{R}^{r \times d}, \; B \in \mathbb{R}^{2d \times r} \quad (\text{adapting } q \text{ and } v)$$
    $$\text{Params}_{\text{LoRA}}(r) = 12 \times (768 \times r + 2 \times 768 \times r) = 12 \times 3 \times 768 \times r = 36,864 \times r$$
- **Exact Counts:**
  - **Rank 2:** $36,864 \times 2 = 73,728$ LoRA params ($0.074$M). Trainable total: $4,138,288$ ($4.41\%$ of SAM).
  - **Rank 4:** $36,864 \times 4 = 147,456$ LoRA params ($0.147$M). Trainable total: $4,212,016$ ($4.49\%$ of SAM).
  - **Rank 8:** $36,864 \times 8 = 294,912$ LoRA params ($0.295$M). Trainable total: $4,359,472$ ($4.64\%$ of SAM).
  - **Rank 16:** $36,864 \times 16 = 589,824$ LoRA params ($0.590$M). Trainable total: $4,654,384$ ($4.93\%$ of SAM).
- **Reconciliation with Earlier Reports:** An earlier table reported $0.037$M for $r=4$ because it counted only $q$-adaptation ($768 \times 4 \times 12 = 36,864$). The actual implementation adapts both $q$ and $v$, yielding exactly $147,456$. Crucially, the fine-tuned SAM Mask Decoder adds $4,058,340$ parameters (~$4.06$M). LoRA parameters represent only $0.16\%$ of total SAM parameters, whereas decoder tuning represents $4.32\%$.

### 2.3 Comprehensive Descriptive Statistics ($N=120$)
Evaluated on the clean seed-42 Rank 4 run (`results/clean_rerun_rank4_seed42_combined.csv`):

| Method | Backbone | Mean Dice (%) | Median Dice (%) | IQR Dice (%) | Mean HD95 (mm) | Median HD95 (mm) | IQR HD95 (mm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (FoB)** | ViT-H (Frozen) | $85.74 \pm 12.58$ | $89.02$ | $7.74$ | $28.05 \pm 21.91$ | $23.40$ | $26.71$ |
| **FoB + LoRA (Fixed 10)** | ViT-B (LoRA) | $88.97 \pm 13.91$ | $93.32$ | $7.50$ | $32.13 \pm 36.07$ | $19.62$ | $40.61$ |
| **AdaFoB + LoRA (Adaptive)** | ViT-B (LoRA) | $89.03 \pm 14.03$ | $93.20$ | $7.27$ | $30.12 \pm 34.31$ | $19.16$ | $36.16$ |

- **Explicit Median Definitions:**
  - *Difference of medians (Dice):* $93.20\% - 93.32\% = -0.118\%$
  - *Median of paired differences (Dice):* $+0.023\%$
  - *Difference of medians (HD95):* $19.16 - 19.62 = -0.466$ mm
  - *Median of paired differences (HD95):* $0.000$ mm (exactly zero difference for over 35% of slices)

### 2.4 Volume-Level Cluster Bootstrap & Multiple Testing Correction
Because 4 organ slices belong to the same physical patient volume, individual slices are non-independent. A volume-level cluster bootstrap was executed ($B=2000$ resamples of the 30 volume clusters):

- **Dice Difference (AdaFoB vs FoB+LoRA):**
  $$\Delta \text{Dice} = +0.064\% \quad [95\% \text{ CI}: -0.248\%, +0.316\%], \quad p = 0.6210$$
- **HD95 Difference (AdaFoB vs FoB+LoRA):**
  $$\Delta \text{HD95} = -2.064 \text{ mm} \quad [95\% \text{ CI}: -4.407 \text{ mm}, -0.296 \text{ mm}], \quad p_{\text{raw}} = 0.0220$$

#### Holm-Bonferroni Family-Wise Error Rate Control

| Rank | Hypothesis Test | Raw $p$-value | Multiplier | Adjusted $p$-value | Significance ($\alpha=0.05$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | AdaFoB vs Baseline Dice | $<0.0001$ | 4 | $<0.0001$ | **SIGNIFICANT** |
| 2 | AdaFoB vs FoB+LoRA HD95 | $0.0220$ | 3 | $0.0660$ | **NOT SIGNIFICANT** |
| 3 | AdaFoB vs Baseline HD95 | $0.3690$ | 2 | $0.7380$ | **NOT SIGNIFICANT** |
| 4 | AdaFoB vs FoB+LoRA Dice | $0.6210$ | 1 | $0.6210$ | **NOT SIGNIFICANT** |

> [!IMPORTANT]
> The previously cited HD95 advantage ($p=0.0220$) **does not survive Holm-Bonferroni multiple testing correction** ($p_{\text{adj}} = 0.0660 > 0.05$). Scientifically, we cannot claim that AdaFoB significantly improves HD95 over FoB+LoRA.

### 2.5 TOST Equivalence Testing
- **Dice Equivalence (Pre-specified Margin $\pm 1.0\%$):**
  - Cluster bootstrap $90\%$ CI: $[-0.189\%, +0.276\%]$.
  - The $90\%$ CI falls completely within $[-1.0\%, +1.0\%]$.
  - **Result: EQUIVALENT.** AdaFoB and FoB+LoRA achieve equivalent segmentation Dice.
- **HD95 Equivalence:**
  - Cluster bootstrap $90\%$ CI: $[-3.919 \text{ mm}, -0.535 \text{ mm}]$.
  - Under pre-specified margin $\pm 3.0$ mm: The lower bound ($-3.919$ mm) crosses the margin.
    - **Result: NOT EQUIVALENT.** Non-equivalence cannot be rejected under $\pm 3.0$ mm.
  - Under post-hoc sensitivity margin $\pm 5.0$ mm: The interval is contained within $[-5.0, +5.0]$ mm.
    - **Result: Post-hoc EQUIVALENT only.**

### 2.6 Patient Cluster Reconciliation ($K=25$ vs $N=30$)
- In SABS/Abd-CT, exactly 30 physical CT volumes exist (`image_0` to `image_29`).
- In standard 5-fold cross-validation with 7 patients per fold and support index 2:
  - Exactly 5 support patients (`image_2, 8, 14, 20, 25`) are used exclusively as support and never tested as queries.
  - 20 patients appear as queries in exactly 1 fold.
  - 5 boundary patients (`image_0, 6, 12, 18, 24`) appear as queries in 2 adjacent folds.
  - Total queries evaluated across 5 folds: $20 \times 1 + 5 \times 2 = 30$ query instances ($30 \times 4 = 120$ slices) derived from $K=25$ physical patients.

---

## 3. Retrospective Error Diagnosis & Hard Cases

### 3.1 Hard Cases Analysis (HD95 > 50 mm)
Across the 120 evaluations, exactly 24 cases (20.0%) exhibit HD95 > 50 mm in AdaFoB+LoRA:
- **Extreme Outliers:**
  - Fold 1, SPLEEN, `image_12.nii`: Dice = $67.5\%$, HD95 = **$165.7$ mm**
  - Fold 1, SPLEEN, `image_6.nii`: Dice = $92.5\%$, HD95 = **$139.3$ mm**
  - Fold 3, LK, `image_18.nii`: Dice = $90.1\%$, HD95 = **$130.4$ mm**
- **Root Cause Mechanism (Isolated False-Positive Islands):**
  Notice that for LK `image_18.nii`, the Dice score is $90.1\%$, yet HD95 is $130.4$ mm. This divergence proves that the primary organ body is segmented almost perfectly, but SAM generates a tiny isolated false-positive island in remote anatomy (e.g. abdominal wall or bowel). Under Hausdorff distance, a single pixel at the opposite edge of the CT field of view expands HD95 from 4 mm to over 130 mm.
  *Remedy:* Standard connected-component filtering (retaining only components overlapping positive prompt points) eliminates these spurious islands without altering the model.

### 3.2 The Case of `image_7` (Bilateral Renal Failure)
In Fold 1, patient `image_7.nii` presents a catastrophic bilateral kidney collapse:
- **Left Kidney:** Baseline Dice = $7.63\%$ (HD95 $76.8$ mm) $\rightarrow$ AdaFoB Dice = $7.67\%$ (HD95 $77.2$ mm).
- **Right Kidney:** Baseline Dice = $30.56\%$ (HD95 $55.3$ mm) $\rightarrow$ FoB+LoRA Dice = $3.73\%$ (HD95 $94.5$ mm) $\rightarrow$ AdaFoB Dice = $4.97\%$ (HD95 $95.2$ mm).
- **Liver & Spleen:** Liver Dice = $96.1\%$, Spleen Dice = $74.6\%$.
- **Diagnosis:** `image_7` has severe anatomical atrophy or abnormal renal attenuation. Baseline frozen SAM achieved $30.6\%$ on RK, but LoRA fine-tuning caused a complete representation collapse ($4.97\%$), because the fine-tuned mask decoder overfitted to typical kidney morphologies and suppressed the atypical renal signals.

### 3.3 Right Kidney (RK) vs Left Kidney (LK) Fragility
- Right Kidney performance across 30 evaluations:
  - Baseline ViT-H: Dice $85.62\%$, mean HD95 $27.55$ mm (median $18.60$ mm)
  - FoB + LoRA ViT-B: Dice $85.97\%$, mean HD95 $35.70$ mm (median $10.21$ mm)
  - AdaFoB + LoRA ViT-B: Dice $85.53\%$, mean HD95 $34.60$ mm (median $15.02$ mm)
- **Anatomical Cause:** RK borders the inferior surface of the liver with virtually identical Hounsfield Unit (HU) attenuation. SAM prompts along the upper renal pole frequently leak into hepatic parenchyma, creating severe boundary errors. LoRA fine-tuning did not improve RK Dice and actually degraded mean HD95 by $+7.05$ mm.

### 3.4 Ambiguity Score Validation (Critical Methodological Finding)
Analysis of the 600 episodes in `results/e1_budget_sweep.csv` reveals:
- **Correlation with Baseline Dice:** Pearson $r = -0.2357$ ($p = 1.69 \times 10^{-4}$), Spearman $\rho = -0.3523$ ($p = 1.02 \times 10^{-8}$). Higher ambiguity weakly corresponds to lower Dice (harder cases).
- **Correlation with Optimal Negative Budget ($N^*$):**
  $$\text{Pearson } r = 0.0351 \quad (p = 0.58), \quad \text{Spearman } \rho = 0.0206 \quad (p = 0.75)$$

> [!CAUTION]
> The ambiguity score has **near-zero correlation with the empirically optimal budget**. The allocator dynamically adjusts prompt numbers based on a metric that does not predict how many negative prompts are actually beneficial. This explains why AdaFoB's dynamic allocation yields virtually identical Dice ($+0.06\%$) compared to a fixed budget.

---

## 4. Literature Comparison & Protocol Violation Scope Mismatch

In historical tables, AdaFoB was benchmarked against SOTA few-shot methods:

| Method | Venue | Supervision / Setting | Setting I Compliant? | Reported SABS Dice (%) |
| :--- | :---: | :---: | :---: | :---: |
| **ALPNet** (Ouyang et al.) | TMI 2022 | Supervoxels only | Yes | $60.8$ |
| **SSL-ALPNet** | MICCAI 2022 | Self-supervised supervoxels | Yes | $68.4$ |
| **AM-SAM** | arXiv 2024 | Frozen SAM + prototype prompts | Yes | $76.2$ |
| **FoB** (Bo et al.) | CVPR 2026 | Supervoxels on base folds | Yes | $85.7$ |
| **Historical AdaFoB+LoRA** | Internal | **Target organ GT masks on seen folds** | **NO [SCOPE MISMATCH]** | $88.9$ |

### The Setting I Invalidation (Violation V1)
Setting I in few-shot medical image segmentation strictly stipulates that **test organ classes are unseen during training**. Methods may only train on pseudo-labels (supervoxels) or disjoint base organ classes.  
`experiments/train_lora.py` trained on `ALL_ORGANS = {1: SPLEEN, 2: RK, 3: LK, 6: LIVER}`. Consequently:
1. The model was fine-tuned on the exact same anatomical classes it was tested on.
2. The comparison of AdaFoB ($88.9\%$) against FoB ($85.7\%$) or ALPNet ($60.8\%$) is an uncalibrated scope mismatch. The $+3.2\%$ gain over FoB stems from in-domain target-organ supervision on the SAM Mask Decoder, not from few-shot generalization.

---

## 5. Strategic Roadmap: Path Forward to MICCAI / Top-Tier Publication

Given the audit findings, continuing down the path of "fine-tuning LoRA with target organ labels" for few-shot Setting I is scientifically indefensible and will lead to an immediate rejection or desk-reject at MICCAI main track.

Two viable strategic paths exist:

```
                            AdaFoB Strategy Decision
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        ▼                                                             ▼
    [PATH 1: RECOMMENDED]                                     [PATH 2: ALTERNATIVE]
 Training-Free Spatial Selector                             Declared In-Domain Adaptation
   on Frozen SAM (ViT-B/ViT-H)                             with Strict Cross-Validation
        │                                                             │
 ├─ Zero fine-tuning (No LoRA)                                ├─ Supervised target-organ LoRA
 ├─ 100% Setting I compliant                                  ├─ Re-split: held-out validation
 ├─ Target: Efficient background                              ├─ Label all tables: [SCOPE MISMATCH]
 │  prompting (3-4 pts vs 10 pts)                             ├─ Benchmark against MedSAM/SAMed
 ├─ Latency & compute reduction                               │  (supervised adaptation papers)
 └─ Target venue: MICCAI / MIDL / TMI                         └─ Target venue: BME / JBHI / EMBC
```

### Path 1 (Recommended): Training-Free Spatial Prompt Selector on Frozen SAM
- **Core Premise:** Eliminate LoRA entirely. Keep SAM completely frozen (both ViT-B and ViT-H).
- **Novel Contribution:**
  1. Replace FoB's brute-force fixed 10-point ring with an intelligent, geometry-driven selector:
     - Curvature-adaptive offset distance ($r$ scaled to organ radius and contour curvature).
     - Steeper-gradient leak-risk placement (prioritizing regions where coarse foreground prediction meets adjacent organ boundaries).
     - Connected-component suppression of false-positive prompt targets.
  2. Prove that this selector matches or exceeds FoB's accuracy using only $3\text{--}4$ points instead of $10$ points, drastically reducing SAM decoder prompt embedding latency.
- **Scientific Strengths:**
  - Zero risk of supervisory leakage (100% compliant with Setting I).
  - Clean, defensible, and reproducible experimental design.
  - Fair attribution: no confound between backbone fine-tuning and prompt engineering.

### Path 2: Declared In-Domain Adaptation
- If LoRA fine-tuning must be retained:
  - Formally redefine the paper as *"Parameter-Efficient In-Domain Adaptation of SAM for Abdominal CT Segmentation"*.
  - Remove all claims of few-shot unseen-class segmentation.
  - Fix Violation V2: implement nested cross-validation where validation patients are strictly separated from test folds.
  - Compare exclusively against supervised medical SAM adaptation methods (MedSAM, SAMed, SAM-Med2D) rather than few-shot methods (FoB, ALPNet).
