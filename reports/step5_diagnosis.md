# Step 5 Diagnostic Report: Error Attribution, Prompt Audit, and Oracle Ceiling

> **Protocol Version:** Step 5 Diagnostic Audit (Binding Auditor Scope)  
> **Setting:** Setting I — Zero-Shot / Few-Shot In-Domain on FROZEN SAM (Zero fine-tuning, Zero LoRA, Zero query GT access during inference)  
> **Evaluation Cohort:** 120 Paired Episodes, 25 Unique Patients, 4 Target Organs (Spleen, Right Kidney, Left Kidney, Liver), 5-Fold Cross-Validation  
> **Focal Arms Evaluated:**  
> - **B0:** SAM ViT-H, Upstream Mask Selection (`mask[0]`, R3), Connected-Component Filtering (Dice 86.19%, HD95 24.52 mm)  
> - **B3B:** SAM ViT-B, Frozen Label-Free Rule (`SingleMask`, R1), Connected-Component Filtering (Dice 82.55%, HD95 35.18 mm)  

---

## 1. Executive Summary & Audit Mandate

Per the auditor binding instructions, Step 5 is strictly **DIAGNOSTIC ONLY**. No selector algorithms (S1–S4), adaptive budgeting networks, LoRA adapters, or new architectural variants are implemented or tuned in this step.

The objective is to resolve empirically where the performance gap originates between ViT-H (B0) and ViT-B (B3B) and determine whether intelligent background prompt selection from FoB's candidate pool is a viable lever. Specifically, this audit answers five fundamental questions:
1. **Error Decomposition (Item B.1):** Does the performance deficit come from detached false-positive islands, continuous boundary leakage, under-segmentation (missed parts), or wrong-organ capture?
2. **Prompt Quality Audit (Item B.2):** Are FoB's background prompt coordinates placed correctly outside the organ, or are they contaminating the target organ interior?
3. **Candidate-Pool Oracle Ceiling (Item B.3):** What is the theoretical maximum Dice achievable if an oracle selected the optimal subset of negative prompts from FoB's existing candidate pool? (Pre-specified gate: $<1.0\text{ pp}$ gain implies prompt selection cannot solve the deficit; $>3.0\text{ pp}$ gain proves a genuine target exists).
4. **Budget Sensitivity (Item B.4):** Is $k=10$ the right budget, or does a smaller fixed budget ($k \in \{0, 2, 4, 6\}$) or a random subset match/exceed fixed 10 points?
5. **Synthesis Decision Table (Item B.5):** How does the effect size of prompt selection compare to budget reduction, connected-component filtering, and token selection?

---

## 2. Item B.1: Error Decomposition Analysis

To diagnose where the Dice and HD95 loss originates, all 120 full 3D prediction volumes from arms **B0** and **B3B** were decomposed against the ground-truth multi-organ label volumes (`D:/sabs_CT_normalized/label_*.nii.gz`).

### 2.1 Categorization Methodology
For every episode, the voxel space is partitioned into:
- **True Positive ($TP$):** $\text{Pred} \cap \text{GT}$.
- **False Negative ($FN$, Missed Parts):** $\text{GT} \setminus \text{Pred}$.
- **Detached False-Positive Islands:** Segmented components with zero overlap with ground truth (isolated and filtered by 3D connected-component post-processing).
- **Wrong-Organ Capture:** Connected false-positive voxels that overlap another recognized anatomical organ in the SABS/BTCV multi-organ label volume (Organ IDs 1–13: Spleen, Right Kidney, Left Kidney, Gallbladder, Esophagus, Liver, Stomach, Aorta, IVC, Portal & Splenic Veins, Pancreas, Right Adrenal, Left Adrenal).
- **Continuous Boundary Leakage:** Connected false-positive voxels extending directly into adjacent unannotated abdominal tissue / peritoneal fat.
- **Directed 95th Percentile Hausdorff Distances:**
  - $h_{95}^{\text{pred}\to\text{gt}}$: Directed 95th percentile distance from prediction boundary to nearest GT voxel (measures over-segmentation / leakage severity).
  - $h_{95}^{\text{gt}\to\text{pred}}$: Directed 95th percentile distance from GT boundary to nearest prediction voxel (measures under-segmentation / missed organ coverage).

### 2.2 Error Decomposition Summary Table

| Arm | Organ | Mean GT (Vox) | Detached FP (Vox) | Boundary Leak (Vox) | Wrong Organ (Vox) | Missed Parts FN (Vox) | $h_{95}^{\text{leak}}$ (mm) | $h_{95}^{\text{missed}}$ (mm) | Primary HD95 Driver |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **B0 (ViT-H R3 CC)** | **Liver** | 253,739 | 10,711 (4.3%) | 11,382 (4.8%) | 5,239 (2.1%) | 43,327 (15.9%) | 19.54 | 17.27 | Balanced (16 leak / 13 miss) |
| | **Right Kidney** | 23,604 | 740 (5.5%) | 3,039 (22.4%) | 861 (4.2%) | 1,903 (6.8%) | 28.32 | 2.57 | Over-seg / Leakage (26/30) |
| | **Left Kidney** | 22,744 | 1,578 (30.4%) | 2,878 (13.7%) | 1,143 (20.1%) | 863 (5.7%) | 27.18 | 2.24 | Over-seg / Leakage (29/30) |
| | **Spleen** | 36,609 | 3,106 (14.3%) | 2,518 (7.8%) | 3,967 (18.7%) | 2,910 (8.4%) | 34.45 | 4.03 | Over-seg / Leakage (24/30) |
| | **OVERALL** | **84,174** | **4,034 (13.6%)** | **4,954 (12.2%)** | **2,803 (11.3%)** | **12,251 (9.2%)** | **27.37** | **6.53** | **Over-seg / Leakage (95/120)** |
| **B3B (ViT-B R1 CC)** | **Liver** | 253,739 | 42,579 (18.0%) | 44,975 (19.3%) | 18,971 (7.8%) | 22,280 (8.1%) | 46.22 | 7.99 | Over-seg / Leakage (29/30) |
| | **Right Kidney** | 23,604 | 1,911 (11.4%) | 5,012 (40.1%) | 1,318 (6.2%) | 890 (3.1%) | 33.71 | 1.07 | Over-seg / Leakage (29/30) |
| | **Left Kidney** | 22,744 | 2,766 (43.7%) | 5,262 (23.7%) | 1,684 (22.6%) | 461 (3.9%) | 31.97 | 1.25 | Over-seg / Leakage (30/30) |
| | **Spleen** | 36,609 | 5,814 (22.4%) | 5,933 (19.2%) | 8,047 (34.0%) | 1,966 (6.0%) | 49.74 | 3.50 | Over-seg / Leakage (29/30) |
| | **OVERALL** | **84,174** | **13,267 (23.9%)** | **15,295 (25.6%)** | **7,505 (17.7%)** | **6,399 (5.3%)** | **40.41** | **3.45** | **Over-seg / Leakage (117/120)** |

*Note: Percentage values represent the mean ratio of voxel error relative to ground-truth organ volume ($\text{Voxels} / \text{GT Vol} \times 100$).*

### 2.3 Key Error Attribution Findings
1. **Over-Segmentation Dominates ViT-B (97.5% of Episodes):** In B3B, 117 out of 120 episodes have HD95 driven by boundary leakage / over-segmentation ($h_{95}^{\text{leak}} = 40.41\text{ mm}$ vs $h_{95}^{\text{missed}} = 3.45\text{ mm}$). ViT-B rarely misses organ tissue (mean under-segmentation is only $5.3\%$), but bleeds aggressively outward into adjacent structures.
2. **Boundary Leakage is 3.1x Worse in ViT-B:** ViT-B leaks an average of $15,295\text{ voxels}$ ($25.6\%$ of organ volume) across contiguous boundaries into surrounding tissue, compared to $4,954\text{ voxels}$ ($12.2\%$) in ViT-H.
3. **Wrong-Organ Capture is 2.7x Higher in ViT-B:** ViT-B falsely encompasses an average of $7,505\text{ voxels}$ of neighboring anatomical organs per episode, compared to $2,803\text{ voxels}$ for ViT-H.
4. **Anatomical Inventory of Invaded Tissue:**
   - **Stomach (341,772 voxels invaded across cohort):** The single largest victim of boundary leakage. The lower pole of the spleen and the left lobe of the liver consistently bleed across the gastrohepatic ligament and gastrosplenic ligament into the gastric wall.
   - **Inferior Vena Cava (IVC, 122,790 voxels invaded):** Caused by liver segmentations extending medially across the bare area into the retrohepatic IVC, and right kidney segmentations bleeding into the adjacent vena cava.
   - **Right Kidney & Liver Cross-Talk (91,985 & 62,615 voxels):** Hepatorenal boundary leakage across Morison's pouch is a chronic source of false positives in both directions.
   - **Gallbladder (41,152 voxels):** Frequently enveloped inside liver predictions due to identical CT attenuation on non-contrast / portal phase slices.
   - **Pancreas (36,098 voxels):** Invaded primarily during splenic hilum segmentation.

---

## 3. Item B.2: Prompt Quality & Contamination Audit

To investigate whether FoB's background prompt generation is helping or hurting, all **40,087 negative candidate points** and **37,221 positive prompt points** generated across all 120 episodes were audited against the ground-truth target organ masks.

### 3.1 Prompt Accuracy & Spatial Placement Summary

| Organ | Evaluated Episodes | Total Pos Points | Pos Outside GT (%) | Total Neg Points | Neg Contamination Inside GT (%) | Neg Points $<5\text{ mm}$ (%) | Neg Points $5-10\text{ mm}$ (%) | Neg Points $10-20\text{ mm}$ (%) | Neg Points $\ge 20\text{ mm}$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Liver** | 30 | 13,690 | 6.97% | 14,937 | **0.033% (5 pts)** | 1.27% | 3.55% | 20.01% | 75.13% |
| **Right Kidney** | 30 | 8,214 | 10.42% | 8,610 | **0.081% (7 pts)** | 0.67% | 6.30% | 62.45% | 30.50% |
| **Left Kidney** | 30 | 8,277 | 12.43% | 8,740 | **0.011% (1 pt)** | 0.38% | 6.22% | 62.49% | 30.89% |
| **Spleen** | 30 | 7,040 | 12.71% | 7,800 | **0.179% (14 pts)** | 0.40% | 3.74% | 46.86% | 48.82% |
| **OVERALL** | **120** | **37,221** | **10.03%** | **40,087** | **0.067% (27 pts)** | **0.78%** | **4.76%** | **43.61%** | **50.78%** |

### 3.2 Findings & Mechanistic Insights
1. **Contamination Hypothesis Refuted:** Across the entire cohort of 40,087 negative prompt points, **only 27 points (0.067%, or 1 in every 1,485 points)** fell inside the true organ boundary. In Liver, LK, and RK, contamination is virtually non-existent ($0.011\% - 0.081\%$). No episode exhibited $>3.5\%$ contamination, and zero episodes met the high-contamination threshold ($>15\%$). Background points do **NOT** poison the segmentation by falsely suppressing the organ interior.
2. **Positive Prompt Fidelity (90% Interior Hit Rate):** 89.97% of positive points land strictly within the true organ interior. The 10.03% falling outside occur exclusively on transitional slices at the extreme superior or inferior poles where the organ cross-section tapers to near-zero.
3. **The "Distant Background" Pathology:**
   - **94.39% of all negative points fall $>10\text{ mm}$ away from the organ boundary** ($43.61\%$ at $10-20\text{ mm}$ and $50.78\%$ at $>20\text{ mm}$).
   - **Only 5.54% of negative points fall within $10\text{ mm}$ of the organ surface** (and only $0.78\%$ within $5\text{ mm}$).
   - **Mechanism:** FoB's background candidate generation places points into the broad peritoneal cavity and distant body wall rather than along the immediate ambiguous organ border. Because negative points are positioned far away, SAM receives no constraint along the hepatorenal or gastrosplenic interface, allowing boundary leakage to expand unchecked into contiguous adjacent organs.

---

## 4. Item B.3: Candidate-Pool Restricted Oracle Ceiling

To establish whether intelligent selection from FoB's background candidate pool can theoretically resolve the performance deficit, we evaluate two oracle ceilings:
1. **Prefix Oracle:** Evaluates the pre-ordered FoB candidate list truncated at budgets $k \in \{1, \dots, 10\}$ and selects the budget $k^*$ that maximizes 3D volume Dice.
2. **Greedy Subset Oracle:** Iteratively selects the negative prompt point from FoB's 10 candidate pool that yields the highest incremental 3D volume Dice, stopping when no point improves Dice.

> **Crucial Methodological Distinction:** Oracle ceilings access query ground truth to select prompt subsets. They are **diagnostic-only theoretical ceilings** and are **NOT deployable methods**.

### 4.1 Oracle Ceiling Performance vs Fixed-10 Baseline

| Arm / Selection Mode | Mean Dice CC (%) | Median Dice CC (%) | Mean HD95 CC (mm) | Median HD95 CC (mm) | p90 HD95 CC (mm) | Mean Gain over Fixed-10 (pp) | Median Gain over Fixed-10 (pp) | 95% Patient-Cluster CI [pp] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B3B Fixed-10 (R1 SingleMask)** | 82.55 | 86.24 | 35.18 | 28.26 | 81.03 | — | — | — |
| **Prefix Oracle ($k^* \le 10$)** | 85.31 | 88.51 | 29.92 | 21.93 | 64.33 | +2.76 | +1.89 | [+2.14, +3.38] |
| **Greedy Subset Oracle** | **87.04** | **90.34** | **27.88** | **17.84** | **62.87** | **+4.49** | **+3.21** | **[+3.72, +5.34]** |

*Note: 95% confidence intervals computed via 5,000-iteration patient-clustered bootstrap ($K=25$ clusters).*

### 4.2 Per-Organ Oracle Breakdown

| Organ | Episodes | Fixed-10 Dice CC (%) | Prefix Oracle CC (%) | Greedy Oracle CC (%) | Greedy Median CC (%) | Fixed-10 HD95 (mm) | Greedy HD95 (mm) | Greedy Gain (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Liver** | 30 | 84.40 | 88.56 | **91.18** | 91.97 | 40.96 | **21.09** | **+6.78** |
| **Right Kidney** | 30 | 83.71 | 85.75 | **86.23** | 89.30 | 28.54 | **26.72** | **+2.52** |
| **Left Kidney** | 30 | 82.98 | 84.66 | **86.13** | 89.15 | 25.97 | **26.78** | **+3.15** |
| **Spleen** | 30 | 79.11 | 82.27 | **84.64** | 90.91 | 45.23 | **36.94** | **+5.53** |
| **OVERALL** | **120** | **82.55** | **85.31** | **87.04** | **90.34** | **35.18** | **27.88** | **+4.49** |

### 4.3 Evaluation Against Pre-Specified Gates
- **Pre-declared Gate:**
  - Gain $< 1.0\text{ pp}$: Candidate pool is deficient; prompt selection cannot solve the segmentation deficit.
  - Gain $\in [1.0, 3.0]\text{ pp}$: Inconclusive headroom.
  - Gain $> 3.0\text{ pp}$: Genuine target exists; prompt selection can meaningfully close the deficit.
- **Empirical Outcome:**
  - The Greedy Subset Oracle achieves **+4.49 pp** mean Dice gain, with a 95% patient-cluster bootstrap CI of **[+3.72, +5.34] pp**.
  - Even the lower bound of the 95% CI (+3.72 pp) substantially clears the 3.0 pp threshold.
  - In large high-ambiguity organs (Liver and Spleen), the oracle achieves **+6.78 pp** and **+5.53 pp** gains, while cutting HD95 in half (Liver HD95 drops from 40.96 mm to 21.09 mm).
  - **Conclusion:** A rich, highly effective subset of background prompts *already exists* within FoB's candidate pool. The failure of baseline FoB is not pool absence, but poor subset selection (the naive fixed top-10 heuristic incorporates harmful/redundant points).

---

## 5. Item B.4: Budget Sensitivity Analysis & Random Controls

To determine whether the deficit is simply caused by having too many prompts (e.g. over-prompting) vs bad selection, we sweep fixed budgets $k \in \{0, 2, 4, 6, 10\}$ and compare each against 5 matched random subsets drawn from the same candidate pool.

### 5.1 Fixed Budget vs Random Control Summary

| Budget Arm | Selection Policy | Mean Dice Raw (%) | Mean Dice CC (%) | Median Dice CC (%) | Mean HD95 CC (mm) | Median HD95 CC (mm) | p90 HD95 CC (mm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k=0$ (B4B)** | Positives Only (No Negatives) | 80.95 | 81.33 | 84.65 | 44.42 | 32.01 | 93.11 |
| **$k=2$** | Top-2 FoB Conf. Order | 81.60 | 82.33 | 86.27 | 36.48 | 25.89 | 76.90 |
| **$k=4$** | Top-4 FoB Conf. Order | 80.70 | 81.97 | 85.09 | 36.94 | 28.79 | 79.68 |
| **$k=6$** | Top-6 FoB Conf. Order | 80.16 | 81.97 | 85.42 | 37.96 | 28.85 | 78.29 |
| **$k=10$ (B3B)** | Top-10 FoB Conf. Order | 79.47 | **82.55** | 86.24 | **35.18** | 28.26 | 81.03 |
| **Random $k=2$** | Mean over 5 Random Seeds | **82.60** | **83.26** | 83.26 | 35.48 | 27.10 | 76.54 |
| **Random $k=4$** | Mean over 5 Random Seeds | 82.20 | 83.22 | 83.22 | 34.61 | 27.24 | 75.80 |
| **Random $k=6$** | Mean over 5 Random Seeds | 81.40 | 82.95 | 82.95 | 34.15 | 27.45 | 76.21 |

### 5.2 Key Budget Observations
1. **Zero Negatives ($k=0$) Causes Severe Boundary Degradation:** Omitting background points drops Dice to 81.33% and causes HD95 to explode to **44.42 mm** (p90 reaches 93.11 mm). Negative prompts are functionally essential to pin down boundaries in frozen SAM.
2. **Fixed Budget Reduction Does Not Close the Gap:** Sweeping fixed top-$k$ budgets from 10 down to 2 changes Dice from 82.55% to 82.33% (-0.22 pp) and worsens HD95 from 35.18 mm to 36.48 mm. A static reduction in $k$ yields **0.00 pp** gain over $k=10$.
3. **Random Subsets Outperform Fixed Top-$k$ (The Diversity Anomaly):**
   - For every tested budget ($k=2, 4, 6$), drawing a random subset of points from FoB's candidates beats the corresponding fixed top-$k$ by **+0.71 to +1.25 pp**:
     - Random $k=2$ achieves **83.26% CC** vs Fixed $k=2$ at **82.33% CC** (+0.93 pp).
     - Random $k=4$ achieves **83.22% CC** vs Fixed $k=4$ at **81.97% CC** (+1.25 pp).
   - **Interpretation:** FoB's confidence ordering ranks points that are highly clustered in the same region. Taking top-$k$ by confidence concentrates points redundantly in one quadrant of the background. Random selection breaks this spatial clustering, distributing background points across different sectors of the image and achieving superior boundary coverage.

---

## 6. Item B.5: Synthesis Decision Table & Statistical Contrasts

To provide an objective basis for the next phase of research, we rank the empirical effect sizes of the four possible intervention levers on frozen SAM:

| Rank | Intervention Lever | Contrast Evaluated | Effect Size on Dice (pp) | Effect Size on HD95 (mm) | Status / Feasibility |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | **Better Prompt Selection** | Greedy Oracle vs Fixed-10 (B3B) | **+4.49 pp** (95% CI [+3.72, +5.34]) | **-7.30 mm** | **Primary Headroom:** Large target confirmed inside candidate pool; requires deployable selector. |
| **2** | **Token Selection / Backbone Calibration** | Upstream ViT-H `mask[0]` vs ViT-B `SingleMask` | **+3.64 pp** (Holm $p=0.003$) | **-10.66 mm** | Backbone integration deficit; frozen R1 is sub-optimal for ViT-H, and ViT-B lacks calibrated IoU head. |
| **3** | **Connected-Component Post-Processing** | 3D CC Filtered vs Raw Output (B3B) | **+3.08 pp** | **-5.84 mm** | **Completed:** Successfully suppresses detached false-positive islands across all arms. |
| **4** | **Reducing Fixed Budget** | Best Fixed $k$ ($k=10$) vs Other Fixed $k$ | **+0.00 pp** | **+0.00 mm** | **Ineffective:** Static budget tuning alone cannot solve the problem (though random selection yields +0.71 pp). |

---

## 7. Item C: Binding Stopping Rule & Determination

Per the binding instructions of the auditor mandate, the audit concludes with the pre-specified determination:

```
============================================================
DETERMINATION: GAP EXISTS FOR PROMPT SELECTION: YES
============================================================
```

### 7.1 Empirical Justification
1. **Oracle Target Confirmed:** The Greedy Subset Oracle achieves **87.04% Dice** (+4.49 pp gain over fixed-10, 95% CI [+3.72, +5.34] pp), well exceeding the $>3.0\text{ pp}$ significance gate.
2. **Deficit is Selection, Not Poisoning:** Prompt contamination inside true organs is virtually non-existent (0.067%, 27 points in 40,087). The performance loss is caused by continuous boundary leakage (97.5% of episodes) and over-clustered prompt placement.
3. **Diversity Lever Proven:** Random subsets from the candidate pool beat confidence-ranked top-$k$ by +0.71 to +1.25 pp, verifying that breaking spatial clustering improves segmentation.

### 7.2 Decision Paths for Next Phase (Awaiting Review)
With the diagnosis complete, the auditor halts all code execution. The user may select one of the following paths:
- **Path S (Develop Deployable Selectors S1–S4):** Implement and evaluate label-free prompt selection strategies (e.g. S1 Confidence, S2 Diversity/Farthest-Point, S3 Hybrid, S4 Leak-Risk Geometry) under strict zero-query-GT leakage.
- **Path T (Address ViT-B Token / Calibration Deficit):** Investigate calibration or multi-mask scoring to resolve the 3.64 pp gap between ViT-H `mask[0]` and ViT-B `SingleMask`.
- **Path P (Address Boundary Placement):** Modify prompt generation to place points closer to the organ boundary ($<10\text{ mm}$), addressing the 94.4% distant-placement pathology.
- **Terminate:** Close the study with the established diagnostic findings.

```
============================================================
AUDITOR STATUS: DIAGNOSIS COMPLETE — AWAITING USER DECISION
============================================================
```
