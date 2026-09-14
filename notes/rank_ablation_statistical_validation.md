# LoRA Rank Ablation Statistical Validation & Model Selection

## 1. Statistical Significance (Wilcoxon Signed-Rank + Cohen's d)
We ran the two-sided Wilcoxon signed-rank test and standard Cohen's d on the paired 5-fold CV outputs (4 organs * 20 volumes * 5 folds = 400 cases) across all rank pairs.

### Overall Results (All Organs)
| Metric | Pair | Mean r1 | Mean r2 | P-value | Cohen's d | Conclusion |
|--------|------|---------|---------|---------|-----------|------------|
| Dice | r2 vs r4 | 90.86% | 90.21% | **6.77e-07** | -0.071 | r4 is significantly **worse** than r2 |
| Dice | r4 vs r8 | 90.21% | 90.36% | **2.81e-08** | +0.015 | r8 is significantly better than r4 |
| Dice | r8 vs r16 | 90.36% | 91.36% | **1.22e-04** | +0.100 | r16 is significantly better than r8 |
| Dice | r2 vs r16 | 90.86% | 91.36% | **9.36e-06** | +0.053 | r16 is significantly better than r2 |
| HD95 | r2 vs r4 | 39.22mm | 41.10mm | **0.0019** | +0.024 | r4 is significantly **worse** than r2 |
| HD95 | r4 vs r8 | 41.10mm | 48.21mm | 0.792 | +0.082 | Not significant (driven by outliers) |
| HD95 | r8 vs r16 | 48.21mm | 40.08mm | **0.0147** | -0.082 | r16 is significantly better than r8 |
| HD95 | r2 vs r16 | 39.22mm | 40.08mm | 0.482 | +0.009 | Not significantly different |

**Takeaway:** The non-monotonic Dice trend is statistically verified. The drop from `r=2` to `r=4` is highly significant (p < 0.001), despite small effect size. The gain of `r=16` over all lower ranks is also highly significant for Dice.

### Spleen HD95 Volatility
When isolating just the Spleen HD95 (where values swung from 71mm (r2) -> 25mm (r4) -> 85mm (r8) -> 70mm (r16)):
* **None of the differences between ranks for Spleen HD95 are statistically significant** (p > 0.13 for all pairs).
* This indicates the wild swings in mean Spleen HD95 are driven entirely by a few extreme outliers rather than a systematic degradation of the model's capabilities.

## 2. Spleen HD95 Outlier Investigation
We ran `experiments/visualize_spleen_outliers.py` to identify cases where Spleen HD95 > 100mm.
* **Rank 4 Outliers (4 cases):** Worst is `image_11.nii` (HD95 = 374.1mm).
* **Rank 16 Outliers (5 cases):** Worst are `image_15.nii` (HD95 = 580.3mm) and `image_20.nii` (HD95 = 598.5mm).

*Diagnosis:* An HD95 of ~600mm inside a standard CT volume means the model predicted a false positive component in the absolute opposite corner of the 3D volume. To visualize exactly what anatomical structure it hallucinated, we need to run an inference script that saves the 3D NIfTI predictions for these specific volumes, as Kaggle currently only saves the CSV metrics.

## 3. Trainable Parameter Breakdown
Using SAM ViT-B (hidden dim 768) with 12 attention blocks. The Prompt Encoder and Mask Decoder are fully unfrozen (4,064,560 parameters). LoRA is applied to Q and V matrices, adding `36,864 * rank` parameters.

* **Original SAM ViT-B**: 93,735,472 parameters
* **Rank 2**: Total = 93,809,200 | Trainable = 4,138,288 (4.41%) | LoRA = 73,728
* **Rank 4**: Total = 93,882,928 | Trainable = 4,212,016 (4.48%) | LoRA = 147,456
* **Rank 8**: Total = 94,030,384 | Trainable = 4,359,472 (4.64%) | LoRA = 294,912
* **Rank 16**: Total = 94,325,296 | Trainable = 4,654,384 (4.93%) | LoRA = 589,824

## 4. Multi-seed Variance Check (Critical Issue)
An audit of `experiments/train_lora.py` and `experiments/eval_lora.py` reveals that **no random seeds are set**. The training data loaders, prompt simulations (which use `random.randint` and `np.random.choice`), and PyTorch initialization are currently running completely unseeded.
* **Violation:** Per `AGENTS.md` Section 5.2, an unseeded result cannot be used to make final decisions.
* **Action Required:** We must inject `torch.manual_seed(42)` and `np.random.seed(42)` into the scripts and re-run `r=4` and `r=16` multi-seed to prove that the 1% Dice gap (90.2% vs 91.3%) is true rank superiority and not just a lucky unseeded prompt simulation draw.

## 5. Headline Model Decision
**Decision: PENDING SEED CONTROL.**
While `r=16` is statistically superior to `r=4` in the current unseeded outputs, we cannot definitively switch the headline model until we prove this gap survives multi-seed variance. 

**Immediate next step:** Add seed control to `train_lora.py` and `eval_lora.py`, then run 3 seeds of `r=4` and 3 seeds of `r=16` on Kaggle.
