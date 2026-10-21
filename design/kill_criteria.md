# Pre-Registered Kill Criteria for Training-Free Prompt Selector (Path 1)

**Status:** FROZEN BEFORE CONFIRMATORY EXECUTION  
**Date:** 2026-10-21  
**Project:** AdaFoB (Few-Shot Medical Image Segmentation with SAM)

---

## 1. Decision Rule & Primary Kill Criteria

The training-free spatial prompt selector (Arms S1–S4) operates on a completely frozen SAM backbone (no LoRA, no fine-tuned decoder). Its objective is to achieve accuracy non-inferiority to upstream FoB while reducing prompt count and latency.

The project shall be declared **STOP / PIVOT** if any of the following pre-registered conditions occur:

### Kill Criterion 1: Failure to Outperform Matched-Budget Random Subset (B4)
- **Metric:** Patient-level paired Hausdorff Distance ($\Delta \text{HD95} = \text{HD95}_{\text{selector}} - \text{HD95}_{\text{B4}}$).
- **Threshold:** At matched budget $k \in \{3, 4\}$, the selector must achieve a patient-level paired HD95 improvement over the random subset baseline (B4) such that the 95% cluster-bootstrap confidence interval **strictly excludes zero in the favorable direction**:
  $$\text{Upper bound of } 95\% \text{ CI of } (\text{HD95}_{\text{selector}} - \text{HD95}_{\text{B4}}) < 0.0 \text{ mm}$$
- **Kill Condition:** If the 95% CI includes zero or is positive, the hypothesis that spatial/geometry selection outperforms random sampling at matched budget is refuted $\rightarrow$ **STOP / PIVOT**.

### Kill Criterion 2: Failure to Beat Best Fixed Budget (B3)
- **Metric:** Patient-level paired HD95 and Dice relative to the best validation-chosen fixed budget arm (B3).
- **Threshold:** The selector ($k \le 4$) must achieve statistically superior HD95 over B3 (95% CI of difference strictly $< 0.0$ mm) without a statistically significant Dice degradation.
- **Kill Condition:** If the selector does not outperform B3 with an interval excluding zero $\rightarrow$ **STOP / PIVOT** (the complexity of dynamic/spatial selection is unjustified over a trivial constant budget).

### Kill Criterion 3: Inferiority to Upstream FoB ViT-H (B0) on Dice
- **Metric:** Non-inferiority margin on Dice ($\Delta \text{Dice} = \text{Dice}_{\text{selector}} - \text{Dice}_{\text{B0}}$).
- **Pre-Registered Non-Inferiority Margin:** $\delta = -1.5\%$ ($-0.015$ Dice points).
- **Threshold:** The lower bound of the one-sided 95% cluster bootstrap CI (or two-sided 90% CI) must exceed $\delta$:
  $$\text{Lower bound of } 90\% \text{ CI of } (\text{Dice}_{\text{selector}} - \text{Dice}_{\text{B0}}) > -1.5\%$$
- **Kill Condition:** If the lower bound is $\le -1.5\%$, the selector fails non-inferiority against the published upstream baseline $\rightarrow$ **STOP / PIVOT**.

### Kill Criterion 4: Latency & Compute Overhead Violation
- **Metric:** Wall-clock selector execution latency per slice.
- **Threshold:** The selector overhead must not exceed $15$ ms per slice on GPU (or $25$ ms on CPU), and must yield a net latency reduction relative to B1 (fixed 10 points) due to fewer prompt tokens passed to the SAM Mask Decoder.
- **Kill Condition:** If selector overhead offsets the decoder latency savings, net efficiency is zero $\rightarrow$ **STOP / PIVOT**.

---

## 2. Enforcement Protocol

1. **No Ad-Hoc Modifications:** If any kill criterion triggers, no new modules, auxiliary losses, or heuristic patches may be added to rescue the result.
2. **Transparent Negative Reporting:** A triggered kill criterion will be documented and reported as a definitive scientific conclusion (e.g. *"Empirical Demonstration that Geometry-Guided Selection Does Not Surpass Random Subsets at Low Budgets"*).
3. **Freezing Rule:** These criteria are committed to the repository and may not be adjusted post-hoc.
