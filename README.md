# AdaFoB: Adaptive Number of Background Prompts for Few-shot Medical Image Segmentation of Irregular Structures

**Status:** Research in progress (private until publication). Target venue: FAIR 2026 / MAPR 2027 (IEEE 6-page, double-blind).

AdaFoB extends **FoB (Focus on Background, CVPR 2026)** — a SAM-based few-shot medical image segmentation (FSMIS) method that constrains SAM's over-segmentation with negative background prompts. FoB uses a **fixed number of background prompts (Np = 10)** and a **ring-shaped spatial prior** (a differential dilation band, r = 15, eps = 2). FoB's own Appendix E states these as limitations for irregular/thin structures.

**Central claim:** The optimal number and placement of background prompts is a *geometric* property of the target structure. AdaFoB derives the prompt budget and prompt placement from the support mask's shape (perimeter, compactness, skeleton), improving boundary quality (HD95) on irregular and thin structures.

## Repository layout

```
AdaFoB/
  data/               # dataset download + preprocessing (FoB protocol); raw data NOT committed
  models/             # FoB base modules + new APC / GAP / CPF modules
  losses/             # loss functions
  experiments/        # pilot/ + train/eval/ablation scripts
  configs/            # YAML configs (all hyperparameters live here, not in code)
  results/            # CSV outputs (gitignored except small summary tables)
  figures/            # generated figures
  logs/               # run logs (gitignored)
  notes/              # research notes, audits, decision gates (committed)
  paper/              # LaTeX source
  third_party/        # cloned external repos (FoB_SAM, segment-anything) — gitignored
```

## Phase gates (do not skip)

1. **Phase 1** — repo + environment + FoB code audit (`notes/fob_code_audit.md`)
2. **Phase 2** — training-free pilot: heuristic adaptive Np + skeleton prior vs FoB ring prior (`results/pilot/pilot_metrics.csv`)
3. **Phase 3** — GO / NO-GO decision (`notes/go_no_go_decision.md`)
4. **Phase 4** — minimal trainable AdaFoB (only if pilot passes)
5. **Phase 5** — full evaluation + ablations
6. **Phase 6** — paper drafting

## References

- FoB paper: https://arxiv.org/abs/2603.21287
- FoB code: https://github.com/primebo1/FoB_SAM
- Preprocessing protocol (Ouyang et al. / SSL-ALPNet): https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation
