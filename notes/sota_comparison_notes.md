# SOTA Comparison Notes

## Source Documentation

All literature numbers are taken directly from their original papers or from the
FoB paper (CVPR 2026, arXiv:2603.21287) Table 1, which compiles prior work under
a consistent evaluation protocol.

### Directly Comparable Methods (Setting I, 1-shot, Ouyang et al. protocol)

| Method | Citation |
|---|---|
| ALPNet / SSL-ALPNet | Ouyang et al., ECCV 2020 / TMI 2022 |
| ADNet | Hansen et al., CVPR 2022 |
| CRAPNet | MICCAI 2022 |
| RP-Net | Tang et al., ICCV 2021 |
| ProtoSAM | 2024 |
| PGRNet | TMI 2023 |
| RPT | Zhu et al., MICCAI 2023 |
| CAT-Net | ICCV 2023 |
| GMRD | Cheng et al., TMI 2024 |
| FoB + SAM-Med2D | Bo et al., CVPR 2026 |
| AM-SAM | MICCAI 2025 |
| FoB + SAM | Bo et al., CVPR 2026 |

### Scope Mismatches (Listed separately, NOT directly comparable)

| Method | Why Not Comparable |
|---|---|
| SAMed | Fully supervised training on 18/12 Synapse split, not episodic few-shot |
| MedSAM | Requires ground-truth bounding box prompts at test time |
| SAM-Med2D | Interactive point/box prompts, not automated few-shot |

### Pipeline Deviation Note (*)

Our re-run numbers (FoB Baseline 85.74%, FoB+LoRA 88.80%, AdaFoB+LoRA 88.70%)
use a preprocessing pipeline with whole-corpus z-score normalization, which
differs from the Ouyang et al. original preprocessing artifact used by all
literature baselines. The FoB paper reports 86.21% under Ouyang's protocol.

This means:
1. Our re-run baseline (85.74%) is internally consistent with our LoRA results
2. Direct numerical comparison with literature numbers carries a ~0.5% systematic
   uncertainty due to preprocessing differences
3. All relative improvements within our pipeline (e.g., +2.96% from LoRA) are
   valid internal comparisons

### HD95 Reporting

No prior FSMIS paper (ALPNet, RPT, GMRD, FoB, AM-SAM, etc.) reports HD95 on
SABS/BTCV. Our work is the first to systematically report HD95 across all folds
and organs in this benchmark family.

### Key Observation

Our AdaFoB+LoRA (88.70% under our pipeline) exceeds even AM-SAM (86.19% under
Ouyang's pipeline), the current SOTA for episodic few-shot medical segmentation.
However, due to the preprocessing deviation, we use the phrasing "competitive
with or exceeding recent SOTA" rather than claiming outright state-of-the-art,
unless re-run under the exact Ouyang artifact confirms the advantage.
