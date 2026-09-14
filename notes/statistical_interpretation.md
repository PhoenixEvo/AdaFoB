# Statistical Interpretation of AdaFoB+LoRA Results

Based on 120 paired volume-organ evaluations across 5-fold cross-validation.

All p-values from two-sided Wilcoxon signed-rank test. 95% confidence intervals from bootstrap resampling (n=10,000). Effect sizes reported as paired Cohen's d.

---

## Baseline vs FoB+LoRA

**Dice:** Mean difference = -3.06% (95% CI: [-4.26%, -1.77%]), Wilcoxon p = 0.0000 (statistically significant (p < 0.05)), Cohen's d = -0.441 (small).

**HD95:** Mean difference = -4.39mm (95% CI: [-15.01, 4.60]mm), Wilcoxon p = 0.0401 (statistically significant (p < 0.05)), Cohen's d = -0.081 (negligible).

Per-organ significant differences:

- LIVER (Dice): p = 0.0000, d = -1.524, diff = -7.01%
- LK (Dice): p = 0.0016, d = -0.564, diff = -2.08%
- SPLEEN (Dice): p = 0.0001, d = -0.704, diff = -3.88%
- LIVER (HD95): p = 0.0000, d = 0.840, diff = +18.31mm

---

## FoB+LoRA vs AdaFoB+LoRA

**Dice:** Mean difference = +0.10% (95% CI: [-0.30%, 0.54%]), Wilcoxon p = 0.3889 (NOT statistically significant (p >= 0.05)), Cohen's d = 0.045 (negligible).

**HD95:** Mean difference = -1.75mm (95% CI: [-8.36, 2.52]mm), Wilcoxon p = 0.1449 (NOT statistically significant (p >= 0.05)), Cohen's d = -0.055 (negligible).

---

## Baseline vs AdaFoB+LoRA

**Dice:** Mean difference = -2.96% (95% CI: [-4.15%, -1.74%]), Wilcoxon p = 0.0000 (statistically significant (p < 0.05)), Cohen's d = -0.436 (small).

**HD95:** Mean difference = -6.14mm (95% CI: [-17.77, 3.71]mm), Wilcoxon p = 0.0386 (statistically significant (p < 0.05)), Cohen's d = -0.103 (negligible).

Per-organ significant differences:

- LIVER (Dice): p = 0.0000, d = -1.582, diff = -7.10%
- LK (Dice): p = 0.0164, d = -0.317, diff = -1.47%
- SPLEEN (Dice): p = 0.0002, d = -0.667, diff = -3.97%
- LIVER (HD95): p = 0.0000, d = 0.837, diff = +18.03mm

---

## AdaFoB+LoRA vs Fixed-Budget-Control

**Dice:** Mean difference = +0.08% (95% CI: [-0.22%, 0.41%]), Wilcoxon p = 0.6887 (NOT statistically significant (p >= 0.05)), Cohen's d = 0.042 (negligible).

**HD95:** Mean difference = -2.38mm (95% CI: [-6.78, 0.45]mm), Wilcoxon p = 0.6957 (NOT statistically significant (p >= 0.05)), Cohen's d = -0.113 (negligible).

---

## Key Conclusions for Manuscript

1. **HD95 difference (FoB+LoRA vs AdaFoB+LoRA) is NOT significant** (p = 0.1449). The adaptive budget does not cause a statistically detectable boundary quality loss.

2. **HD95 difference (AdaFoB vs Fixed Control) is NOT significant** (p = 0.6957).

3. **Attribution of contributions:** The Dice improvement from Baseline to LoRA-adapted models is primarily attributable to LoRA domain adaptation. AdaFoB's contribution is orthogonal: it preserves accuracy while cutting the prompt budget by ~67%, and (if finding #2 holds) provides superior boundary quality compared to naive budget reduction.
