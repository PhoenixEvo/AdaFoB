# Legacy Claims Audit & Lineage Registry

Date: 2026-10-21  
Auditor: Lead Research Engineer & Independent Methodological Auditor

This document catalogs every scientific and numerical claim generated during historical exploratory phases of the AdaFoB project, assigning each an immutable audit verdict: **VERIFIED**, **UNVERIFIED**, **INVALID**, or **NOT REPRODUCIBLE**.

---

## 1. Summary of Legacy Claims

| Claim ID | Claimed Statement / Metric | Historical Source | Status | Exact Scientific Rationale |
| :--- | :--- | :--- | :---: | :--- |
| **CLM-01** | FoB Frozen SAM ViT-H ($N_p=10$): Dice 85.74% | `results/lora_eval_final_combined.csv` | **UNVERIFIED** | Arithmetic re-evaluation on `clean_rerun_rank4_seed42_combined.csv` yields 85.74% (macro-mean), and training protocol matches Setting I (supervoxels). However, checkpoint SHA256 was never recorded locally or cryptographically locked. |
| **CLM-02** | FoB + LoRA ViT-B ($N_p=10$): Dice 88.80% (old) / 88.97% (clean) | `adafob_lora_final_results.md` / `clean_rerun_rank4` | **INVALID** | Model was trained with ground-truth masks of test organs (Violation V1) and checkpoints were selected on test-fold validation Dice (Violation V2). |
| **CLM-03** | AdaFoB + LoRA ViT-B (Adaptive $N_p$): Dice 88.70% (old) / 89.03% (clean) | `adafob_lora_final_results.md` / `clean_rerun_rank4` | **INVALID** | Same training contamination (V1, V2) plus allocator budget parameters were tuned on query test-fold Dice (Violation V3) and suffered from config key mismatch (Violation V4). |
| **CLM-04** | Restricted oracle result called "theoretical ceiling": Dice 86.98% | `results/r3_global_params.json` | **INVALID AS CEILING** | Derived from retrospective grid search selecting the optimal per-volume constant $N_p \in [0, 24]$ on test query GT using frozen ViT-H. It is a retrospective query-GT oracle over a restricted discrete grid, not a global theoretical upper bound for arbitrary continuous spatial prompting. |
| **CLM-05** | Mean adaptive prompt budget is approximately 3.25 | `results/e4_efficiency.csv` | **UNVERIFIED** | Derived from the corrupted V3/V4 parameter set. While empirical mean $N_p$ is ~3.3, it reflects a misconfigured parameter deployment. |
| **CLM-06** | Historical mean HD95: FoB ViT-H = 30.0 mm; FoB+LoRA = 28.4 mm; AdaFoB+LoRA = 30.5 mm; Fixed-low-budget = 39.0 mm | `notes/hd95_bug_investigation.md` | **INVALID / NOT REPRODUCIBLE** | Historical HD95 was computed before voxel spacing orientation bugs were fixed (MedPy array indexing inversion). Clean re-run yields 28.05 mm, 32.13 mm, and 30.12 mm respectively under the invalid training protocol. |
| **CLM-07** | Fixed-low-budget settings (Liver 6, Spleen 3, RK 2, LK 2) | `experiments/extract_mean_budget.py` | **INVALID AS TUNED BASELINE** | Extracted directly by averaging adaptive budget allocations across all 5 evaluation test folds (Violation V5). It is an exploratory descriptive post-hoc control, not a legitimate validation-tuned baseline. |
| **CLM-08** | Two-sample test p = 0.965 proved statistical equivalence | Exploratory chat notes | **INVALID (WITHDRAWN)** | Interpreted non-significance ($p > 0.05$ on a superiority test) as proof of equivalence. Formally withdrawn. Equivalence requires pre-specified two-sided TOST margins. |
| **CLM-09** | 30 volumes x 4 organs = 120 independent experimental units | Early draft notes | **INVALID** | Repeated organs within physical patient volumes exhibit correlated anatomy; 5 boundary patients are repeated across overlapping folds. Primary statistical unit is the physical patient volume ($K=25$), not slice or organ observations. |
| **CLM-10** | Right Kidney performance regression & extreme HD95 | `notes/failure_case_analysis.md` | **VERIFIED** | Confirmed on clean data: RK exhibits zero measurable Dice improvement with LoRA ($\Delta = -0.09\%$) and large HD95 spikes driven by false-positive boundary leaks. |
| **CLM-11** | Case `image_7` failure in Fold 1 | `notes/failure_case_analysis.md` | **VERIFIED** | Confirmed: `image_7.nii` suffered bilateral renal collapse (LK Dice 7.6% across all methods; RK Dice 30.6% baseline collapsing to 3.7% in FoB+LoRA and 5.0% in AdaFoB). Section header mislabel in historical notes caused confusion between LK and RK. |
| **CLM-12** | Setting I protocol violation | Pre-audit suspicion | **VERIFIED** | Confirmed by direct code audit (`experiments/train_lora.py` L88). Ground-truth labels of evaluated organs were used as training targets. |
| **CLM-13** | LoRA Rank Ablation trends (r=2 vs r=4 vs r=8 vs r=16) | `results/rank_ablation_stats.csv` | **INVALID** | Audit revealed 78% to 83% of evaluation rows were identical duplicates across rank CSVs due to Kaggle parallel aggregation race conditions; training was unseeded. |
| **CLM-14** | LoRA trainable parameter count formula ($N_{\text{LoRA}} = 36,864 \times r$) | `results/lora_param_counts.csv` | **VERIFIED** | Confirmed by inspecting model state dicts. Earlier report of 37k params for r=4 was a divide-by-4 error in drafting notes. |
