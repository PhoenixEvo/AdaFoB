# AdaFoB: Adaptive Focus on Background for Few-Shot Medical Image Segmentation

**Training-Free Boundary-Guided Negative Prompting and Counterfactual Verification for Frozen SAM.**

AdaFoB is a strictly **Setting I** (zero target-organ supervision), **training-free** test-time prompting framework built on top of frozen Segment Anything Model (SAM) and Focus-on-Background (FoB) coarse priors for 3D few-shot medical image segmentation.

Rather than passing fixed heuristic negative prompts to SAM—which frequently cause boundary leakage or foreground over-suppression across modalities and encoder scales—AdaFoB combines:
1. **Boundary-Guided Negative Prompt Placement**: places sparse (`<= 2`) negative prompts along outward boundary normals of the coarse organ contour using local appearance contrast.
2. **Counterfactual Boundary Verification**: evaluates candidate negative prompt subsets using cached single-pass SAM image embeddings (zero extra encoder forward passes, zero fine-tuning, zero additional hyperparameters) and selects the mask whose counterfactual edits best agree with the medical prior.

---

## Verified Setting I Results (Stage 4: 2 Modalities x 2 SAM Backbones)

All numbers below are verified 3D **Patient-Balanced Connected-Component (CC) Dice (%)**, **HD95 (mm)**, and **Recall (%)** under strict 5-fold Setting I evaluation (`single-seed (seed=42)` for deterministic methods; `5-seed mean (seeds=42..46)` for Random 2-Point; paired 95% CIs from `5,000`-replicate patient-cluster bootstrap, `seed=42`). Every Stage 4 run was locally re-verified from raw Kaggle artifacts with `0.0000 pp` anchor drift against prior locked stages.

### 1. SABS-CT (`120` episodes, `25` query patients, `4,009` slices)
Full report: [`reports/adafob_v2_stage4_sabs_results_20261119.md`](reports/adafob_v2_stage4_sabs_results_20261119.md)

| Backbone | Method | Neg. Pts | Pat CC Dice (%) | HD95 (mm) | Recall (%) | Delta vs FoB Baseline (95% CI) | Delta vs Boundary-Nearest (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|
| SAM ViT-B | FoB Coarse Prior (No SAM) | 0 | 81.97 | 22.24 | 90.58 | +0.15 pp | -1.20 pp |
| SAM ViT-B | FoB Baseline (`multimask=False`) | 10 | 81.82 | 36.22 | 94.43 | — | -1.35 pp |
| SAM ViT-B | Positive-Only | 0 | 80.64 | 43.93 | 95.01 | -1.18 pp | -2.53 pp |
| SAM ViT-B | Random 2-Point (5-seed, `seeds=42..46`) | 2 | 82.66 | 35.57 | 94.51 | +0.84 pp | -0.51 pp |
| SAM ViT-B | Boundary-Nearest | 2 | 83.17 | 32.80 | 94.29 | +1.35 pp | — |
| SAM ViT-B | Boundary Placement | 2 | 84.72 | 27.93 | 92.95 | +2.90 pp | +1.55 pp |
| SAM ViT-B | **AdaFoB (Verified Placement)** | **1-2** | **84.79** | **27.67** | **93.51** | **+2.97 pp [+2.07, +3.96]** | **+1.63 pp [+1.13, +2.12]** |
| SAM ViT-H | FoB Baseline (`multimask=False`) | 10 | 82.18 | 39.87 | 92.98 | — | -2.09 pp |
| SAM ViT-H | Positive-Only | 0 | 82.60 | 43.38 | 92.84 | +0.42 pp | -1.67 pp |
| SAM ViT-H | Random 2-Point (5-seed, `seeds=42..46`) | 2 | 83.54 | 35.23 | 92.87 | +1.36 pp | -0.73 pp |
| SAM ViT-H | Boundary-Nearest | 2 | 84.27 | 32.59 | 92.50 | +2.09 pp | — |
| SAM ViT-H | Boundary Placement | 2 | 84.67 | 28.22 | 91.22 | +2.49 pp | +0.40 pp |
| SAM ViT-H | **AdaFoB (Verified Placement)** | **1-2** | **85.11** | **27.63** | **91.96** | **+2.93 pp [+2.24, +3.71]** | **+0.83 pp [+0.42, +1.25]** |

### 2. CHAOST2 Abdominal MRI (`80` episodes, `15` query patients, `1,354` slices)
Full report: [`reports/adafob_v2_stage4_chaost2_results_20261119.md`](reports/adafob_v2_stage4_chaost2_results_20261119.md)

| Backbone | Method | Neg. Pts | Pat CC Dice (%) | HD95 (mm) | Recall (%) | Delta vs FoB Baseline (95% CI) | Delta vs Boundary-Nearest (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|
| SAM ViT-B | FoB Coarse Prior (No SAM) | 0 | 84.26 | 19.24 | 85.94 | +4.04 pp | +1.52 pp |
| SAM ViT-B | FoB Baseline (`multimask=False`) | 10 | 80.22 | 44.56 | 92.92 | — | -2.52 pp |
| SAM ViT-B | Positive-Only | 0 | 79.09 | 49.44 | 94.75 | -1.13 pp | -3.65 pp |
| SAM ViT-B | Random 2-Point (5-seed, `seeds=42..46`) | 2 | 81.26 | 41.34 | 93.16 | +1.04 pp | -1.48 pp |
| SAM ViT-B | Boundary-Nearest | 2 | 82.74 | 36.65 | 92.45 | +2.52 pp | — |
| SAM ViT-B | Boundary Placement | 2 | 83.57 | 34.65 | 91.72 | +3.35 pp | +0.83 pp |
| SAM ViT-B | **AdaFoB (Verified Placement)** | **1-2** | **84.23** | **32.27** | **92.24** | **+4.01 pp [+3.24, +4.82]** | **+1.49 pp [+1.05, +1.97]** |
| SAM ViT-H | FoB Baseline (`multimask=False`) | 10 | 79.62 | 45.28 | 93.42 | — | -2.56 pp |
| SAM ViT-H | Positive-Only | 0 | 73.25 | 70.02 | 94.57 | -6.37 pp | -8.93 pp |
| SAM ViT-H | Random 2-Point (5-seed, `seeds=42..46`) | 2 | 79.19 | 52.36 | 93.74 | -0.43 pp | -2.99 pp |
| SAM ViT-H | Boundary-Nearest | 2 | 82.18 | 42.51 | 92.71 | +2.56 pp | — |
| SAM ViT-H | Boundary Placement | 2 | 82.35 | 40.37 | 92.55 | +2.73 pp | +0.17 pp |
| SAM ViT-H | **AdaFoB (Verified Placement)** | **1-2** | **82.93** | **39.00** | **92.76** | **+3.31 pp [+1.84, +4.80]** | **+0.75 pp [+0.07, +1.47]** |

*Reproducibility arm ID mapping in experiment logs and CSVs: FoB Coarse Prior = `fob_coarse_prior`; FoB Baseline = `full10`; Positive-Only = `positive0`; Random 2-Point = `random2_mean5`; Boundary-Nearest = `boundary_near2`; Boundary Placement = `pba_f10_int_k2`; AdaFoB (Verified Placement) = `cbv_fob_prob_no0`.*

> **Ongoing Research Note (Stage 5A):** Stage 4 evaluated frozen SAM under `multimask_output=False` (mask token `t0`) for exact continuity with our locked Step 9 controls. Active Stage 5A experiments extend AdaFoB to **Joint Prompt-and-Granularity Counterfactual Verification** across all 4 SAM mask tokens (`t0..t3`, including official FoB's `multimask_output=True` token `t1`) alongside box and mask prior prompt controls.

---

## Repository Structure

```text
AdaFoB/
├── analysis/
│   ├── controls_v1/                # Locked Step 9 Setting I control definitions & bootstrap stats
│   ├── gate_a/                     # Exact coordinate transforms, metrics, and runtime utilities
│   ├── rebaseline_v1/              # Step 8 paired rebaseline utilities
│   ├── patient_balanced_summary.py # Patient-balanced 3D metric aggregation
│   └── independent_reassessment_20261118/
├── data/
│   ├── download_instructions.md    # Dataset setup instructions (SABS-CT and CHAOST2)
│   └── preprocess.py               # Preprocessing utilities
├── experiments/
│   ├── CHANGELOG.md                # Append-only chronological experiment ledger
│   ├── data_prep/                  # CHAOST2 NIfTI conversion and Kaggle dataset packaging
│   ├── option_a_reassessment_v1/   # Stage 0 bug-fixed PBA & signed local-ring verification (SABS ViT-B)
│   ├── adafob_v2_stage1/           # Stage 1 semantic-gated boundary prompting (SABS ViT-B)
│   ├── adafob_v2_stage2/           # Stage 2 counterfactual boundary verification (SABS ViT-B)
│   ├── adafob_v2_stage3_chaost2/   # Stage 3 cross-modality & backbone scaling (CHAOST2 ViT-B/H)
│   ├── adafob_v2_stage4/           # Stage 4 backbone-invariant CBV v2 (SABS & CHAOST2, ViT-B/H)
│   ├── run_step4_baselines.py      # Step 9 frozen reference script
│   └── run_step5_diagnostics.py    # Step 9 frozen reference script
├── models/
│   └── postprocessing.py           # Step 9 frozen 2D largest connected-component postprocessing
├── protocol/
│   ├── setting_i_spec.md           # Setting I evaluation specification
│   ├── paired_episode_manifest.csv # Locked 120-episode SABS-CT manifest
│   └── dev_split.csv               # Support split reference
├── reports/                        # Verified stage reports and independent audit reports
├── results/
│   └── QUARANTINE_INDEX.md         # Explicit quarantine record of invalidated historical artifacts
├── tests/                          # Unit & integration test suite (CPU, no dataset download required)
└── third_party/                    # Vendored FoB_SAM reference code
```

---

## Setup and Verification

### 1. Environment Installation
```bash
conda env create -f environment.yml
conda activate adafob
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2. Run the Unit Test Suite (CPU)
```bash
python -m pytest tests/ -q
```
All tracked tests run deterministically on CPU without requiring local medical NIfTI volumes or SAM checkpoints.

### 3. Reproducing Kaggle GPU Runs and Local Verification
Each stage package under `experiments/adafob_v2_stage*/` contains a self-contained Kaggle T4x2 packager (`package.py`) and local SHA256 + estimand verifier (`verify_local.py`). For example, for Stage 4 (`SABS-CT` or `CHAOST2`):

```bash
# Package submission bundle for Kaggle T4x2
python -m experiments.adafob_v2_stage4.package \
    --dataset sabs \
    --run-dir experiments/runs/adafob_v2_stage4_sabs \
    --kernel-id <kaggle-user>/adafob-v2-stage4-sabs

# Push to Kaggle
kaggle kernels push -p experiments/runs/adafob_v2_stage4_sabs/submission

# Download outputs and locally verify SHA256, estimands, and anchor parity
kaggle kernels output <kaggle-user>/adafob-v2-stage4-sabs \
    -p experiments/runs/adafob_v2_stage4_sabs/kaggle_outputs
python -m experiments.adafob_v2_stage4.verify_local \
    --dataset sabs \
    --run-dir experiments/runs/adafob_v2_stage4_sabs
```

See [`reports/dataset_preparation_guide_20261119.md`](reports/dataset_preparation_guide_20261119.md) for CHAOST2 NIfTI preparation and checkpoint verification.

---

## Scientific Integrity and Audit Trail

This repository maintains a strict, transparent audit history:
- **[`experiments/CHANGELOG.md`](experiments/CHANGELOG.md)**: Chronological ledger of every verified run, bundle SHA256, anchor drift check, and negative/positive finding.
- **[`reports/independent_reassessment_20261118.md`](reports/independent_reassessment_20261118.md)**: First-principles methodological audit documenting historical bugs (`A1-A5`, `B1-B3`) and Setting I violations (`V1-V7`) discovered and eliminated prior to AdaFoB v2.
- **[`results/QUARANTINE_INDEX.md`](results/QUARANTINE_INDEX.md)**: Complete record of quarantined legacy artifacts removed from the active tree (permanently preserved in Git at tags `backup/pre-cleanup-20261120` and `backup/legacy-main-063192a`).
