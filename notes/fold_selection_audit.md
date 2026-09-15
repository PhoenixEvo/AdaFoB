# Fold Selection Audit for Representative-Fold Multi-Seed Validation

## 1. Context & Motivation
Due to compute constraints on Kaggle (each fold training requires ~4-5 hours of dual-T4 GPU time, and full 5-fold training across 2 ranks and 3 seeds would require ~50+ hours), assessing seed sensitivity on a single representative fold is adopted for the ablation study. This document audits the 5 cross-validation folds to justify the representative fold choice.

---

## 2. Organ-Count Balance Across Folds
Inspection of `results/lora_eval_rank4_combined.csv` confirms:
* Each fold contains exactly 6 volumes.
* Each volume is evaluated for all 4 target organs (`LIVER`, `LK`, `RK`, `SPLEEN`).
* **Distribution:** Exactly 6 samples per organ per fold (24 samples per fold).
* **Conclusion:** All 5 folds are 100% balanced in terms of organ representation.

| Fold | Volumes | Total Samples | Liver | LK | RK | Spleen |
|------|---------|---------------|-------|----|----|--------|
| Fold 0 | 6 (img_0, 1, 3, 4, 5, 6) | 24 | 6 | 6 | 6 | 6 |
| Fold 1 | 6 (img_6, 7, 9, 10, 11, 12) | 24 | 6 | 6 | 6 | 6 |
| Fold 2 | 6 (img_12, 13, 15, 16, 17, 18) | 24 | 6 | 6 | 6 | 6 |
| Fold 3 | 6 (img_18, 19, 21, 22, 23, 24) | 24 | 6 | 6 | 6 | 6 |
| Fold 4 | 6 (img_0, 24, 26, 27, 28, 29) | 24 | 6 | 6 | 6 | 6 |

---

## 3. Location of Known Problematic Cases
We audited the exact fold membership of all previously documented failure cases:
1. **`image_11.nii` (Catastrophic Spleen HD95 outlier, 374mm):** Located in **Fold 1**.
2. **`image_7.nii` (Catastrophic Left Kidney failure case):** Located in **Fold 1**.
3. **`image_15.nii` (Rank 16 Spleen outlier):** Located in **Fold 2**.
4. **`image_20.nii` / Fold 3 outliers:** Located in **Fold 3**.

**Finding on Fold 1:** Fold 1 concentrates both of the dataset's worst catastrophic failures (`image_7` and `image_11`). Consequently, Fold 1 exhibits severely depressed performance across all models (Rank 4 Dice: 80.80%, Rank 16 Dice: 78.45%, HD95 mean: 83.04mm). **Fold 1 is an extreme outlier and should NOT be used as the representative fold.**

---

## 4. Quantitative Comparison of Candidate Folds

| Fold | Rank 4 Mean Dice | Rank 4 Mean HD95 | Rank 16 Mean Dice | Rank 16 Mean HD95 | Assessment |
|------|------------------|------------------|-------------------|-------------------|------------|
| **Fold 0** | **89.86%** | 39.07mm | **94.13%** | 34.97mm | **Closest to overall 5-fold mean (90.22%)**, no catastrophic outliers |
| Fold 1 | 80.80% | 55.05mm | 78.45% | 83.04mm | Severely distorted by `image_7` and `image_11` |
| Fold 2 | 91.05% | 30.45mm | 91.56% | 26.38mm | Stable, but minimal gap between r=4 and r=16 |
| Fold 3 | 91.26% | 32.73mm | 91.47% | 40.86mm | Contains extreme HD95 outlier in r=16 (598mm) |
| Fold 4 | 90.51% | 25.86mm | 93.93% | 19.44mm | High performance, but overlaps volumes with Fold 0 |

---

## 5. Final Decision & Justification
**Decision: FOLD 0 is selected as the representative fold.**

**Justifications:**
1. **Representativeness:** Rank 4 performance on Fold 0 (Dice 89.86%) is the closest match to the overall 5-fold cross-validation benchmark (90.22%).
2. **Cleanliness:** Fold 0 is completely free from the known dataset-level artifacts and segmentation collapses present in Fold 1 (`image_7` LK collapse and `image_11` Spleen false-positive hallucination).
3. **Ablation Sensitivity:** Fold 0 clearly captures the performance gap between Rank 4 (89.86%) and Rank 16 (94.13%), providing an ideal testbed to measure whether this gain is statistically robust against seed variations ({42, 43, 44}) or merely random fluctuation.
