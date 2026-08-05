# Manuscript claims audit

Record of claims proposed for the paper that were changed before drafting, with
the evidence. Keep this file: if a claim is reinstated later, it needs to come
back with data attached.

## C1. "We swapped our dataset from SABS to Abd-CT (BTCV)" — REJECTED (false)

Abd-CT, SABS and BTCV are three names for one dataset: Synapse `syn3193805`,
Multi-Atlas Labeling Beyond the Cranial Vault.

Evidence — FoB_SAM README, verbatim:
> **Abd-CT (SABS)**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)

FoB paper, Sec. 4: *"Abd-CT includes 30 3D CT scans"* — the same 30 volumes we use.

There was no dataset change. What changed is the **preprocessing pipeline**: we
moved off Ouyang et al.'s `sabs_CT_normalized` artefact and preprocess the source
volumes ourselves.

Consequence that must appear in the paper: our numbers are **not** directly
comparable to published Abd-CT tables. Every baseline, FoB included, must be
re-run under our pipeline; literature figures may only appear if labelled
"as reported". This is handled in `implementation.tex`, "declared protocol
deviation".

## C2. "AdaFoB introduces a new SPR module" — REJECTED (attribution error)

SPR = **Structure-guided Prompt Refinement**, FoB paper Sec. 3.5, with graphs
`A^ada` and `A^ring`, ablated in Table 3 and Appendix B.4. It is published work
by Bo et al., CVPR 2026.

Describing SPR as our contribution would claim a published module. This is the
single highest-risk item in the draft; at a venue with FSMIS reviewers it is
likely to be caught and is not recoverable in rebuttal.

Our actual new module is `refine.*` in `experiments/train.py` (confirmed in the
freezing commit). It is named **GAP (Geometry-Adaptive Prompting)** throughout
the draft. BPPC / BCM / SPR are credited to FoB in `method.tex` Sec. 3.2.

Also corrected: FoB's prior is a **mask-conformal differential dilation band**
(Eq. 1, r=15, eps=2), not a circular ring. The ring assumption enters only via
`A^ring` in SPR's feature space. A strawman "FoB assumes a circle" framing
invites rejection; the draft attacks the residual rigidity (R1–R3) instead.

## C3. "We encountered catastrophic forgetting and solved it by freezing" — DOWNGRADED

The freezing fix itself is good and is kept. The **narrative** is not supported.

What the evidence shows: the low-scoring AdaFoB runs came from `train.py`
building `FewShotSeg(args)` with random heads and **never calling
`load_state_dict`**, trained for `epochs(10) x iters_per_epoch(100) = 1,000`
iterations against FoB's 36,000 (2.8%). There were no pretrained weights loaded,
therefore nothing to forget. Diagnostics also showed the checkpoint loading at
100% only *after* `--ckpt` was passed, on a network that had been trained from
scratch.

Writing that we diagnosed and cured catastrophic forgetting describes an event
that did not occur, and the ablation a reviewer would request would contradict it.

Kept in the draft, in three tiers by evidential strength:
1. compute allocation — established;
2. preserving a converged prompt-localisation prior, making FoB-vs-AdaFoB a
   clean sampler ablation — design rationale, defensible now;
3. stability under a randomly initialised head — flagged as pending, with the
   matched-initialisation frozen-vs-full run specified as ablation (iv).

If that run shows full fine-tuning collapsing from an identical initialisation
and schedule, the forgetting claim becomes reportable, with the loss curve as
evidence. It is a legitimate and interesting table row — it just has to be run.

## C4. "Variance Splatting hypothesis (Phase 2)" — UNRECOGNISED

This term appears nowhere in the repository, the project plan, the FoB paper, or
any prior discussion in this project. The Phase 2 hypotheses on record are:

- Claim A: a shape-adaptive prompt budget beats any single fixed `Np`.
- Claim B: skeleton/curvature-guided placement beats uniform band sampling,
  especially in HD95 on irregular structures.

The draft uses these. If "Variance Splatting" refers to something real that was
developed outside this repository, send the definition and it will be
incorporated; otherwise it should not enter the manuscript.

## C5. Blocking status — not a claim, a gate

`results/` contains only `README.md`. No `diagnosis.json`, no
`alignment_check.json`, no `phase4_validation.csv`.

Last measured state: oracle-prompt Dice **0.4971**, flat across `masks[0..2]`.
Positives land inside GT 30.8% of the time; predicted negatives sit 69.7 px from
the organ against a ~15 px target.

Until the oracle gate clears ~0.85, no FoB-vs-AdaFoB number is interpretable,
and the prompt-attribution ladder additionally showed the background pathway
contributing +0.03 Dice against +0.33 for the foreground pathway. If that ratio
survives the alignment fix, there is no headroom for a background-prompt
contribution to demonstrate anything, which is a Phase 3 kill-criterion
conversation rather than a Phase 5 one.

## Correction to earlier assistant output

A previous message in this project reported commit `5e6a4f8c8d0b...` for the
alignment checker. That hash was not produced by any commit; the alignment
checker landed as `4056b14`. Recorded here so the commit trail in the notes
stays accurate.
