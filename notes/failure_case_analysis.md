# Failure Case Analysis

> [!WARNING]
> **PENDING HD95 RECOMPUTATION:** The HD95 values discussed in this document were generated prior to fixing the `medpy` coordinate spacing bug. The RK plateau and LK outlier hypotheses will be re-verified once the corrected Kaggle inference completes.

Analysis of 120 volume-organ evaluations from 5-fold cross-validation.

---

## 1. Right Kidney (RK) -- Low Headroom Hypothesis

### 1.1 Per-Organ Baseline Dice and LoRA Improvement

| Organ | n | Baseline Dice | Std | FoB+LoRA Dice | Std | Delta (LoRA-Base) | HD95 Base | HD95 LoRA | HD95 Delta |
|---|---|---|---|---|---|---|---|---|---|
| LIVER | 30 | 87.08% | 4.76% | 94.09% | 3.90% | +7.01% | 30.1mm | 11.8mm | -18.3mm |
| LK | 30 | 85.91% | 15.27% | 87.99% | 15.81% | +2.08% | 25.8mm | 37.7mm | +11.8mm |
| RK | 30 | 85.62% | 11.36% | 84.90% | 16.61% | -0.72% | 32.7mm | 40.2mm | +7.5mm |
| SPLEEN | 30 | 84.34% | 16.14% | 88.22% | 12.24% | +3.88% | 33.3mm | 49.8mm | +16.5mm |

### 1.2 Correlation: Baseline Dice vs LoRA Improvement

- Pearson r = -0.154 (p = 0.0921)
- Spearman rho = -0.443 (p = 0.0000)

**Interpretation:** The correlation is weak (|r| < 0.2), suggesting that baseline Dice alone does not fully explain the variable improvement across organs. Other factors -- organ morphology, boundary contrast with adjacent tissues, typical HU similarity -- likely contribute. An alternative explanation is that the FoB prompt generator already provides strong guidance for kidneys, limiting LoRA's marginal value.

### 1.3 Direct Cross-Organ Baseline Comparison

Organs ranked by baseline Dice (descending):

1. **LIVER**: 87.08% baseline, +7.01% delta
1. **LK**: 85.91% baseline, +2.08% delta
1. **RK**: 85.62% baseline, -0.72% delta
1. **SPLEEN**: 84.34% baseline, +3.88% delta

## 2. Outlier Detection

### 2.1 Cases with Dice < 20% in Any Method (2 found)

| Fold | Organ | Vol ID | Baseline Dice | FoB+LoRA Dice | AdaFoB+LoRA Dice | HD95 Base | HD95 FoB+LoRA | HD95 AdaFoB |
|---|---|---|---|---|---|---|---|---|
| 1 | LK | image_7.nii | 7.6% | 7.3% | 8.1% | 153.9mm | 162.7mm | 160.9mm |
| 1 | RK | image_7.nii | 30.6% | 9.5% | 12.0% | 76.7mm | 153.2mm | 150.8mm |

### 2.2 Cases with HD95 > 200mm After LoRA (4 found)

| Fold | Organ | Vol ID | HD95 Base | HD95 FoB+LoRA | HD95 AdaFoB |
|---|---|---|---|---|---|
| 0 | SPLEEN | image_3.nii | 41.0mm | 211.4mm | 179.6mm |
| 0 | LK | image_0.nii | 20.2mm | 3.3mm | 328.2mm |
| 1 | SPLEEN | image_11.nii | 20.1mm | 376.8mm | 374.1mm |
| 3 | LK | image_18.nii | 7.7mm | 246.5mm | 264.6mm |

## 3. HD95 Regression After LoRA

### 3.1 Per-Organ HD95 Summary

| Organ | HD95 Baseline | HD95 FoB+LoRA | HD95 AdaFoB+LoRA | Delta (LoRA-Base) |
|---|---|---|---|---|
| LIVER | 30.1mm | 11.8mm | 12.0mm | -18.3mm |
| LK | 25.8mm | 37.7mm | 48.3mm | +11.8mm |
| RK | 32.7mm | 40.2mm | 38.7mm | +7.5mm |
| SPLEEN | 33.3mm | 49.8mm | 47.5mm | +16.5mm |

### 3.2 Top 10 Cases Where HD95 Worsened Most (LoRA vs Baseline)

| Fold | Organ | Vol ID | HD95 Base | HD95 LoRA | Delta | Dice Base | Dice LoRA |
|---|---|---|---|---|---|---|---|
| 1 | SPLEEN | image_11.nii | 20.1mm | 376.8mm | +356.7mm | 93.1% | 89.5% |
| 3 | LK | image_18.nii | 7.7mm | 246.5mm | +238.8mm | 89.4% | 91.1% |
| 0 | SPLEEN | image_3.nii | 41.0mm | 211.4mm | +170.4mm | 87.9% | 85.1% |
| 4 | RK | image_29.nii | 22.1mm | 144.9mm | +122.8mm | 88.6% | 81.1% |
| 4 | SPLEEN | image_27.nii | 105.8mm | 194.8mm | +89.0mm | 79.0% | 88.7% |
| 2 | RK | image_12.nii | 4.9mm | 92.4mm | +87.4mm | 90.3% | 89.4% |
| 1 | RK | image_12.nii | 2.8mm | 85.5mm | +82.7mm | 90.0% | 90.7% |
| 1 | RK | image_7.nii | 76.7mm | 153.2mm | +76.5mm | 30.6% | 9.5% |
| 3 | RK | image_24.nii | 40.9mm | 116.7mm | +75.8mm | 87.6% | 75.4% |
| 3 | LK | image_24.nii | 10.8mm | 82.3mm | +71.5mm | 90.5% | 89.0% |

## 4. image_7 (LK) Specific Analysis

| Fold | Organ | Dice Base | Dice FoB+LoRA | Dice AdaFoB | HD95 Base | HD95 FoB+LoRA | HD95 AdaFoB |
|---|---|---|---|---|---|---|---|
| 1 | SPLEEN | 70.6% | 74.7% | 74.5% | 67.3mm | 53.1mm | 50.3mm |
| 1 | RK | 30.6% | 9.5% | 12.0% | 76.7mm | 153.2mm | 150.8mm |
| 1 | LK | 7.6% | 7.3% | 8.1% | 153.9mm | 162.7mm | 160.9mm |
| 1 | LIVER | 92.4% | 96.1% | 95.8% | 9.5mm | 3.0mm | 4.0mm |

**Key finding:** image_7 shows severe segmentation failure for the following organ(s):

- RK: Dice = 12.0% (all methods fail, suggesting a data-level issue such as atypical anatomy, motion artifact, or partial volume at organ boundary)
- LK: Dice = 8.1% (all methods fail, suggesting a data-level issue such as atypical anatomy, motion artifact, or partial volume at organ boundary)

**Note:** A visual overlay of predicted vs ground-truth mask is needed to identify the concrete failure mode. This requires running the visualization script on Kaggle with access to the NIfTI volumes.
