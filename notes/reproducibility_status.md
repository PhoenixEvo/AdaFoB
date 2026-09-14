# Reproducibility Retrofit Status

Because the training and evaluation scripts (`train_lora.py`, `eval_lora.py`) lacked random seed controls prior to this audit, **all existing inference results are subject to unseeded random variance** (data loading order, point simulation). Furthermore, they suffer from the MedPy HD95 coordinate bug.

All result files are tracked here. They must be re-generated with the newly injected `set_seed(42)` logic.

| Artifact / File | Status | Notes |
|----------------|--------|-------|
| `adafob_lora_final_results.md` | ⚠️ pre-seed-fix, needs re-verification | Original r=4 headline results. |
| `results/lora_eval_final_combined.csv` | ⚠️ pre-seed-fix, needs re-verification | Raw r=4 inference metrics (contains HD95 bug). |
| `results/lora_eval_rank2_combined.csv` | ⚠️ pre-seed-fix, needs re-verification | Rank ablation inference (contains HD95 bug). |
| `results/lora_eval_rank8_combined.csv` | ⚠️ pre-seed-fix, needs re-verification | Rank ablation inference (contains HD95 bug). |
| `results/lora_eval_rank16_combined.csv` | ⚠️ pre-seed-fix, needs re-verification | Rank ablation inference (contains HD95 bug). |
| `results/delta_reconciliation_table.csv` | ⚠️ pre-seed-fix, needs re-verification | Derived from unseeded data. |
| `results/hd95_statistical_report.csv` | ⚠️ pre-seed-fix, needs re-verification | Derived from unseeded data (contains HD95 bug). |
| `results/full_statistical_report.csv` | ⚠️ pre-seed-fix, needs re-verification | Derived from unseeded data. |
| `results/rank_ablation_stats.csv` | ⚠️ pre-seed-fix, needs re-verification | Derived from unseeded data. |
| `notes/hd95_bug_investigation.md` | ✅ seed-locked (doc only) | Investigation report. |
| `notes/rank_ablation_statistical_validation.md` | ⚠️ pre-seed-fix, needs re-verification | Derived from unseeded data. |
