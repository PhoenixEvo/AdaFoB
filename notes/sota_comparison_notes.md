# SOTA Comparison Notes (v2)

## Source Documentation

All literature numbers are taken directly from their original papers or from the
FoB paper (CVPR 2026, arXiv:2603.21287) Table 1, which compiles prior work under
a consistent evaluation protocol.

Every row in `sota_comparison_table_v2.csv` has an explicit `source` column:
- `[literature; not re-run]`: number quoted from the original paper, not
  independently verified
- `[this work; re-run]`: number produced by our pipeline, subject to the
  preprocessing deviation described below

### Preprocessing Deviation (*)

Our pipeline uses whole-corpus z-score normalization on the SABS dataset, which
differs from the Ouyang et al. original preprocessing artifact used by all
literature baselines. Consequences:

1. Our re-run baseline (85.74%) differs from FoB's published number (86.21%)
   by approximately 0.5%.
2. All internal comparisons (e.g., FoB+LoRA vs AdaFoB+LoRA) are valid because
   both sides use the same preprocessing.
3. Cross-pipeline comparisons (our numbers vs literature numbers) carry a
   systematic uncertainty of ~0.5% and must be interpreted with this caveat.

### Language Policy

Per the task specification's global constraints, this project does NOT claim
unqualified "state-of-the-art" or "SOTA tier" status due to the preprocessing
deviation. Instead, the following scope-limited phrasing is used:

> "Achieves the best Dice among methods compared here under a modified
> preprocessing protocol relative to Ouyang et al."

### Directly Comparable Methods (Setting I, 1-shot, Ouyang et al. protocol)

| Method | Citation | Source |
|---|---|---|
| ALPNet / SSL-ALPNet | Ouyang et al., ECCV 2020 / TMI 2022 | Literature |
| ADNet | Hansen et al., CVPR 2022 | Literature |
| CRAPNet | MICCAI 2022 | Literature |
| RP-Net | Tang et al., ICCV 2021 | Literature |
| ProtoSAM | 2024 | Literature |
| PGRNet | TMI 2023 | Literature |
| RPT | Zhu et al., MICCAI 2023 | Literature |
| CAT-Net | ICCV 2023 | Literature |
| GMRD | Cheng et al., TMI 2024 | Literature |
| FoB + SAM-Med2D | Bo et al., CVPR 2026 | Literature |
| AM-SAM | MICCAI 2025 | Literature |
| FoB + SAM | Bo et al., CVPR 2026 | Literature |

### Scope Mismatches (Listed separately, NOT directly comparable)

| Method | Why Not Comparable | Source |
|---|---|---|
| SAMed | Fully supervised training on 18/12 Synapse split | Literature |
| MedSAM | Requires ground-truth bounding box prompts at test time | Literature |
| SAM-Med2D | Interactive point/box prompts, not automated few-shot | Literature |

No compatible public checkpoints were found for SAMed, MedSAM, or SAM-Med2D
that could be re-run under our few-shot protocol. Their numbers remain
literature-only with explicit scope mismatch labels.

### Adjacent SAM-Adaptation Baselines (Experiment 6)

Re-running SAMed or MedSAM under our exact 5-fold few-shot protocol would
require significant reimplementation (different training paradigm, different
prompt mechanism). This is not feasible within the current submission timeline.
Their published numbers are retained as literature references with scope
mismatch labels.

### HD95 Reporting

No prior FSMIS paper (ALPNet, RPT, GMRD, FoB, AM-SAM, etc.) reports HD95 on
SABS/BTCV. Our work is the first to systematically report HD95 across all folds
and organs in this benchmark family. This limits cross-method HD95 comparison to
our own internal methods only.
