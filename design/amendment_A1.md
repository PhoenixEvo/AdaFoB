# Pre-Run Protocol Amendment A1

**Status:** APPROVED & FROZEN PRIOR TO BASELINE RUNS  
**Date:** 2026-10-21  
**Project:** AdaFoB (Few-Shot Medical Image Segmentation with SAM)  
**Supersedes by Reference:** `design/kill_criteria.md`

---

## 1. Summary of Amendments

This amendment establishes the experimental constitution for Path 1 (training-free background-prompt selection on a frozen SAM). It supersedes earlier binary decision rules by reference while preserving scientific integrity.

### A1.1 Patient-Level Isolation for Tuning & Selection
- **Rule:** For test fold $i \in \{0, \dots, 4\}$, all hyperparameter tuning, validation, and model selection (such as selecting the best fixed budget in B3 or tuning spatial selectors) must strictly use **ONLY patients that do not appear in test fold $i$'s query set**.
- **Data Availability:**
  - In each test fold $i$, 6 patients form the query set $Q_i$.
  - The remaining 4 folds provide 96 volume-organ records. Excluding records from boundary patients in $Q_i$ leaves **exactly 88 volume-organ records** across **19 independent physical patients** for tuning.
  - This sample size ($N=88$, $K=19$) is confirmed to be sufficient for discrete 1-parameter tuning ($k^* \in \{2, 3, 4, 5, 6, 10\}$).
- **Verification:** Enforced via explicit patient masking in `tests/test_tuning_patient_isolation.py`.

### A1.2 Upstream Checkpoint Provenance Verification
- **Rule:** The official FoB checkpoints (`PrimeBo1/FoB_SAM`, `exps_train_on_SABS_FSMIS_FoB.zip`) were audited against upstream source code (`third_party/FoB_SAM/dataloaders/datasets.py:143-149` and `config.py:50-51`).
- **Audit Outcome:** For each fold $k$, training strictly excluded patients in `FOLD[k]` (`if idx not in self.FOLD[args['eval_fold']]`) and used supervoxels (`use_gt=False`), ensuring zero overlap with test query patients. Documented in `audit/fob_checkpoint_split_check.csv`.

### A1.3 Formal Hypothesis Table & Multiplicity Control
- Abstract designations ($T_1\text{--}T_4$) are replaced by `design/hypothesis_table.csv`.
- **Primary Comparator:** The primary comparator for selector evaluation is **B1 (Frozen SAM ViT-B, fixed 10 points)**, not B0. B0 (ViT-H) serves strictly as an upstream literature-reproduction anchor.
- **Family-Wise Error Control:** Multiplicity correction follows the Holm-Bonferroni step-down procedure within the pre-registered confirmatory family. Screened kill criteria are explicitly labeled as unadjusted screening gates.

### A1.4 Margin Justification (Dice Non-Inferiority at $\delta = -1.5\%$)
- **Clinical & Empirical Basis:**
  1. The inter-patient standard deviation on SABS CT segmentation is $10\text{--}14\%$. A margin of $\delta = -1.5\%$ corresponds to $\approx 0.1\text{--}0.15$ standard deviations, well within inter-radiologist contouring variation ($1.5\text{--}2.0\%$).
  2. For a method whose primary goal is efficiency (reducing prompt count by $60\text{--}70\%$ and decreasing latency), non-inferiority within $1.5\%$ Dice represents a clinically and scientifically acceptable trade-off.
  3. The earlier exploratory $\pm 1.0\%$ equivalence analysis was post-hoc; it is not reused as confirmatory evidence.

### A1.5 Three-Way Decision Outcomes (PASS / FAIL / INCONCLUSIVE)
- **PASS:** The cluster-bootstrap CI satisfies the target criterion (e.g. HD95 upper bound $< 0.0$ mm vs B4, or Dice lower bound $> -1.5\%$ vs B1).
- **FAIL:** The CI strictly excludes a practically meaningful effect:
  - For HD95 superiority vs B4: Lower bound of $(\text{HD95}_{\text{sel}} - \text{HD95}_{\text{B4}}) > -0.5$ mm (rules out an effect $>0.5$ mm).
  - For Dice non-inferiority vs B1: Upper bound of $(\text{Dice}_{\text{sel}} - \text{Dice}_{\text{B1}}) < -1.5\%$ (conclusively inferior).
  - **Action on FAIL:** Declare **STOP / PIVOT** on the primary hypothesis.
- **INCONCLUSIVE:** The point estimate is favorable but the CI is wide:
  - HD95 CI width $> 6.0$ mm, or Dice CI width $> 2.0\%$.
  - **Action on INCONCLUSIVE:** Extend evidence (increase bootstrap resamples $B \ge 10000$, run multi-seed sweeps) rather than prematurely abandoning the direction.

### A1.6 Latency & Computational Efficiency Protocol
- **Pinned Hardware Configuration:**
  - Accelerator: NVIDIA Tesla T4 GPU (16GB VRAM) / NVIDIA RTX 4050 Laptop GPU (6GB VRAM).
  - Framework: PyTorch 2.5.1 + CUDA 12.1/12.2, FP32 / TF32 mode, batch size = 1.
  - Image Resolution: $256 \times 256 \times 3$.
  - Embedding Cache Policy: SAM image-encoder feature map computed ONCE per query slice and cached in GPU memory; prompt decoder calls operate on cached embeddings.
- **Breakdown Reporting:**
  - SAM image-encoder time ($T_{\text{enc}}$).
  - Prompt selector execution time ($T_{\text{sel}}$).
  - SAM mask-decoder time ($T_{\text{dec}}$).
  - Post-processing time ($T_{\text{post}}$).
  - Peak VRAM allocation (`torch.cuda.max_memory_allocated()`).
- **Decision Rule:** Net end-to-end latency must be non-inferior to B1. If the image encoder completely dominates runtime ($>90\%$), efficiency claims shall be strictly bounded to prompt decoding / interactive latency, and the primary paper contribution shall focus on boundary precision and false-positive suppression.

### A1.7 Frozen Connected-Component (CC) Post-Processing Rule
- **Rule Formulation:**
  ```python
  def filter_connected_components(pred_mask: np.ndarray, pos_points: np.ndarray) -> np.ndarray:
      # Retain components containing at least one positive prompt.
      # If none contain a positive prompt, retain the single largest component.
  ```
- **Scope:** Implemented in `models/postprocessing.py`. All arms (B0, B1, B2, B3, B4, S1–S4) report metrics both **WITHOUT CC** and **WITH CC**.

### A1.8 Per-Volume CT Normalization & Reproduction Tolerance
- **Normalization:** Enforce the official FoB per-volume z-score normalization:
  $$\text{img}_{\text{norm}} = \frac{\text{img} - \mu_{\text{vol}}}{\sigma_{\text{vol}} + \epsilon}$$
  as implemented in `third_party/FoB_SAM/dataloaders/datasets.py:47`.
- **B0 Reproduction Target:** Upstream FoB reported $86.21\%$ on SABS (CVPR 2026 Table 1). Reproduction tolerance is $\pm 1.0\%$ ($85.21\% \le \text{Dice}_{\text{B0}} \le 87.21\%$). If outside this window, audit slice alignment before proceeding.

### A1.9 Experimental Terminology & Primary Unit of Analysis
- **Patient ($K=25$):** Unique physical human subject in query test set.
- **Volume-Fold Instance ($N=30$):** Query evaluation episode of a patient in a specific fold.
- **Volume-Organ Record ($N=120$):** Evaluation of one organ in a volume-fold instance ($30 \times 4$).
- **Primary Endpoint:** The **patient-level paired difference**, computed by averaging each patient's metrics across organs and folds, then calculating paired differences against the comparator ($K=25$). Per-organ results are secondary endpoints.
