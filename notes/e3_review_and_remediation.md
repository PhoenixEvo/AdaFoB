# E3 review and remediation plan (pre-submission)

Reviewed: E0–E4 / M1–M2 results reported 2026-08-11.
Target: MICCAI 2027 (deadline ~late Feb 2027 — verify on the official site).
Audience: the coding agent and the research team.

**Verdict: do not compile a submission from the current results.** One defect is
fatal, three are serious, four are deviations from spec that must be reconciled.
All are fixable inside the available time. The core idea is sound; it is currently
being measured on the wrong organs with a protocol that leaks.

---

## 1. What is solid — keep it, do not rework

| Item | Why it counts |
|---|---|
| Alignment gate cleared (oracle 0.8944, edge-z 5.93, no flips) | The pipeline-validity blocker is genuinely gone |
| \(N_p{=}10\) parity test passes exactly | Guarantees measured differences come from the allocator, not the refactor. This is the most valuable thing you built |
| Zero-budget bypass + epsilon guards | The \(N_p{=}0\) path is the novel part; hardening it was correct |
| Training-free grid-search design | Keep. Justify it as reproducibility under a hard compute budget, not as elegance |
| Multi-component handling, scale-adaptive offset | Correct implementations of the spec |

---

## 2. Blocking defects

### B1 (FATAL) — the baseline is broken, so the gains measure nothing

| Organ | FoB | AdaFoB | Abs gain | Oracle | Headroom recovered |
|---|---|---|---|---|---|
| Aorta | 8.07 | 12.51 | +4.44 | ~89 | 5.5% |
| Gallbladder | 2.46 | 2.58 | +0.12 | ~89 | 0.1% |

Published Abd-CT FSMIS Dice: ALPNet 73.4, RPT 77.8, GMRD 78.5, AM-SAM 86.2,
FoB+SAM 86.2.

Dice below ~0.30 means the structure is not being found. The gap between 8.07 and
12.51 is the gap between two failures. Reporting "+55% relative" on a base of 8.07
is the single most reliable way to get desk-rejected: the reviewer writes
*"improvements on a broken baseline are not evidence"* and stops reading.

Gallbladder's +0.12 Dice points is below noise. With 250 episodes and a
per-episode Dice sd of 15–25 points (routine in FSMIS), the standard error is
~0.9–1.6 points. The observed difference is ~0.1σ.

**Root cause to confirm:** the FoB checkpoint was trained with liver, right kidney,
left kidney and spleen as the label set. Aorta and gallbladder are small
cross-sectional structures outside that set. A prompt generator that cannot locate
the organ produces near-zero Dice regardless of how its background budget is
allocated.

### B2 (SERIOUS) — per-organ hyperparameter tuning is test-class leakage

λ = 0.0 for aorta, λ = 1.0 for gallbladder. In FSMIS the test class is novel by
construction. Selecting a hyperparameter using the identity of the test organ uses
information the protocol forbids. A reviewer will name this directly, and it
invalidates both regimes as reported.

The underlying observation (different geometry wants different curvature weighting)
is interesting and worth keeping — but it must be *predicted at test time from
observable quantities*, never *selected by organ name*.

### B3 (SERIOUS) — the control that decides the paper is missing

AdaFoB throttles aorta to a mean budget of 2.70 and wins. The first question any
reviewer asks: **would a fixed \(N_p{=}3\) do the same?**

Without the equal-mean-budget control the contribution reduces to "spend fewer
negative prompts", which is one global scalar, not an adaptive method. This is
sharpened by the fact that λ=0 on aorta switches curvature-aware placement **off
entirely**, so on your headline regime the method *is* only budget throttling.

### B4 (SERIOUS) — the results are not auditable

`results/` on `main` contains only `README.md`. No per-episode CSV, no variance, no
significance test, no HD95 — despite HD95 being the stated differentiator of the
whole project. Nothing in the report can currently be checked or reproduced.

### B5 (SERIOUS) — Setting II skipped

Setting II tests generalisation to entirely unseen classes. Skipping it while
simultaneously tuning hyperparameters per test organ (B2) is the worst available
combination: the one protocol that would expose the leak is the one omitted.

---

## 3. Spec deviations to reconcile

**D1 — the ambiguity gate sign is inverted.**
Spec: \(g(a)=\sigma\!\big((a-a_0)/\tau\big)\), increasing in ambiguity.
Implemented: \(e^{-a/a_0}\), decreasing in ambiguity.

The specified form encodes the mechanism: negatives exist to stop SAM leaking, and
leaking happens when foreground and background are hard to separate (high \(a\)),
so budget should *rise* with \(a\). Your form spends *fewer* negatives when
separation is hard.

That may be empirically correct — if negatives placed near an ambiguous boundary get
absorbed as foreground, suppressing them can help. But you cannot ship a method
whose functional form contradicts its own stated motivation. Either:
(a) justify the inversion with a mechanism and evidence (plot negative-prompt gain
vs \(a\); if the trend is genuinely negative, that *is* a finding and becomes
Figure 2), or
(b) revert to the specified sigmoid and let the fit decide.
Also: τ appears in your tuned-parameter list but not in your formula. Reconcile.

**D2 — regime labels are inverted.** Aorta preferring \(N^*{=}0\) in 24% of cases
is *low* over-segmentation susceptibility. Gallbladder preferring larger budgets is
the *high*-susceptibility regime. Fix the terminology everywhere before it reaches
a reviewer.

**D3 — λ=0 on aorta disables placement.** State plainly in the paper that on this
regime the method reduces to budget allocation. Do not let the architecture figure
imply both components are active.

**D4 — efficiency framing.** 257 → 278/311 ms is 8–21% *slower*. Do not describe
this as "saving massive redundant processing". Report the overhead, note it is small
and dominated by Python-side OpenCV morphology, and move on.

---

## 4. Remediation tasks, in execution order

### R0 — per-organ pipeline validation (~1 GPU-h)

Run the oracle gate **separately for every organ you intend to report**. The 0.8944
figure was measured on spleen/liver and does not license aorta or gallbladder.

**Code change required:** the current gate in `gt_prompt_sanity_check` uses 10
positives *and* 10 negatives. Stage 5 showed negatives can *reduce* oracle Dice, so
the gate can now fail for a reason unrelated to pipeline validity. Change the gate
to **positives-only** (`n_pos=10, n_neg=0`), which measures the true pipeline
ceiling, and report the with-negatives value separately as data.

*Acceptance:* every reported organ has positives-only oracle ≥ 0.85. Any organ
below that is reported only in the limitations section, never in the main table.

### R1 — E1 on the organs where the baseline is competent (~4–6 GPU-h)

This is the redirect. Your own Stage 5 already measured **liver: 0.8434 → 0.7071
(−13.63) when negatives are added at an 84% operating point.** That is the paper.

Run the per-episode budget sweep on **spleen (1), right kidney (2), left kidney (3),
liver (6)** — the four organs FoB was trained on, where it scores ~86%, and where
published baselines exist for comparison.

Protocol:
- Use FoB's **predicted** foreground prompts (not oracle) — the realistic regime.
- Sweep \(N_p \in \{0,1,2,3,4,6,8,10,12,16,20\}\), taking the top-\(N_p\) predicted
  background prompts by the model's own confidence ordering.
- **Cache the SAM image embedding once per query slice** and reuse across the whole
  sweep. The encoder dominates; the decoder is milliseconds. Non-negotiable.
- Record Dice and HD95 at every \(N_p\), per episode.

Report: histogram of \(N^*\) per organ; per-case-oracle vs best-single-global \(N_p\);
fraction of episodes with \(N^*{=}0\); mean Dice cost of forcing \(N_p{=}10\) on that
subset.

*Gate G1 — proceed only if either:*
- per-case oracle exceeds best-global \(N_p\) by ≥ 1.5 Dice on average, **or**
- ≥ 20% of episodes have \(N^*{=}0\) **and** forcing \(N_p{=}10\) costs ≥ 3 Dice on
  that subset.

If G1 fails on all four organs, stop and go to §7.

### R2 — the full control set (~6 GPU-h)

Every regime table must contain all of these rows:

| Row | Purpose |
|---|---|
| FoB, \(N_p{=}10\) | published configuration |
| Fixed \(N_p = \text{round(mean adaptive)}\) | **the equal-mean-budget control (B3)** |
| Fixed \(N_p\) = best global, chosen on fold 0 | strongest non-adaptive competitor |
| Adaptive budget + uniform placement | isolates the budget contribution |
| Fixed budget + curvature/leak placement | isolates the placement contribution |
| AdaFoB (adaptive budget + placement) | the method |
| Per-case oracle \(N_p\) | upper bound |

Rows 4–6 form the 2×2 factorial the method claims. Row 2 is the row that decides
whether adaptivity is a contribution.

### R3 — fix the hyperparameter protocol (no extra GPU)

Choose one, and document it in `implementation.tex`:

- **(a) Preferred — one global parameter set.** Fit \((\nu, \lambda, \gamma, a_0,
  \tau, \alpha)\) once on fold 0 pooled across *training* organs, apply unchanged to
  every test organ.
- **(b) If regime adaptation is a claim** — make λ a function of observable
  quantities available at test time, e.g. \(\lambda(a, \overline{|\kappa|})\) with
  *globally* fitted parameters. The adaptation then happens through the model, not
  through the experimenter.

*Hard rule:* no hyperparameter may be indexed by test-class identity. Fit on fold 0,
report on folds 1–4, and state this explicitly.

### R4 — Setting II (~8–10 GPU-h)

Run it. FoB reports both settings; omitting the harder one while doing per-class
tuning is indefensible. If compute is tight, run Setting II on the two strongest
organs rather than skipping it entirely, and say so.

### R5 — statistics, HD95 and provenance (~2 GPU-h)

- Paired **Wilcoxon signed-rank** over per-episode scores for every claimed gain.
- **mean ± std across folds**, plus median difference and IQR as effect size.
- Bonferroni (or Holm) correction across the organs compared.
- **HD95** for every row, surface-based, with the empty-prediction rate reported
  alongside so a degenerate predictor cannot be flattered by the sentinel.
- **Commit every per-episode CSV** plus the run-metadata JSON. Add
  `notes/claims_evidence_map.md` mapping each claim in the paper to a CSV path and
  row. No number enters the PDF without a traceable source.

### R6 — reposition aorta and gallbladder (writing only)

Move them to a **limitations / small-structure analysis** subsection. State plainly
that both the baseline and the method fail there (<15 Dice), give the reason (small
cross-sectional area, 1-shot, prompt generator trained on a different label set),
and use it to bound the claim. Honest, useful, and it costs nothing — but it must
not be the headline.

### R7 — figures

1. Histogram of \(N^*\) per organ, with the mass at zero visible. One figure, whole
   argument.
2. Negative-prompt gain vs ambiguity score \(a\), scatter + fitted trend. This is
   the saturation law, and it is also where D1 gets settled.
3. Dice cost of the fixed budget: FoB \(N_p{=}10\) minus per-case oracle, stratified
   by \(a\).
4. Budget-vs-Dice curves per organ, with \(N^*\) marked.

---

## 5. What the main table must look like

Rows: ALPNet / RPT / GMRD / ProtoSAM / AM-SAM (cited, marked "as reported"), FoB+SAM
(re-run by you), AdaFoB (ours), plus the controls from R2.
Columns: per-organ Dice for spleen, RK, LK, liver; mean Dice; mean HD95; Wilcoxon
*p* vs FoB.
Footnotes: fold count, episodes per fold, Setting, `mask_select` mode, and the
declared preprocessing deviation from Ouyang et al.

If the AdaFoB mean Dice sits in the low-to-mid 80s with a significant gain over FoB,
you have a MICCAI paper. If it matches FoB but with a materially smaller prompt
budget, that is also a paper — an efficiency-and-analysis contribution — provided
the equal-mean-budget control shows adaptivity is doing the work.

---

## 6. Acceptance gates before compiling the PDF

Every one of these must be true:

- [ ] Every organ in the main table has positives-only oracle ≥ 0.85 (R0)
- [ ] G1 passed on at least one competent organ (R1)
- [ ] Equal-mean-budget control present in every table (R2)
- [ ] No hyperparameter indexed by test-class identity (R3)
- [ ] Setting II reported, or its absence justified in the text (R4)
- [ ] Wilcoxon *p* and std accompany every claimed gain (R5)
- [ ] HD95 reported for every row, with empty-prediction rate (R5)
- [ ] Every table cell traces to a committed CSV (R5)
- [ ] Aorta/gallbladder appear only in limitations (R6)
- [ ] D1 resolved: gate sign justified or reverted; τ reconciled
- [ ] D2 resolved: regime labels corrected throughout
- [ ] No `\needsnum` / `\needsrun` macros remain unresolved

---

## 7. If G1 fails on all four competent organs

Then adaptive background budgeting has no headroom anywhere you can measure, and the
method genuinely has nothing to recover. Fall back to the scoped analysis paper:

*"When do negative prompts help SAM in medical imaging? A saturation analysis."*

Requirements to be credible: ≥ 3 datasets, ≥ 2 SAM variants (ViT-B and ViT-H, SAM2
if it fits), the ambiguity score validated as a *predictor* of negative-prompt
utility, and a practical prescription. Target a MICCAI workshop or a journal —
not the main track. This is a real contribution and a respectable outcome; it is
simply a different paper, and it must be planned as one rather than reached by
default.

---

## 8. Reporting rules (apply to every draft)

- Never lead with relative gain when the absolute Dice is below ~30.
- Never report a gain without a significance test and a variance estimate.
- Never mix `mask_select` modes within a table.
- Never report a number produced with `--force`.
- Never describe a slower system as saving processing.
- Never rename BPPC / BCM / SPR — they are FoB's published modules.
- Never tune on a fold used for reporting.

---

## 9. Schedule (today 2026-08-11 → MICCAI 2027)

| Weeks | Work | GPU-h |
|---|---|---|
| 1 | R0 + R1 (per-organ gate, E1 on organs 1/2/3/6), evaluate G1 | 6 |
| 2–3 | R3 protocol fix, refit globally, R2 control set on 2 organs | 8 |
| 4–6 | Full matrix Setting I, 5 folds, 4 organs | 15 |
| 7–8 | R4 Setting II | 10 |
| 9 | R5 statistics, HD95, CSV provenance; R7 figures | 3 |
| 10–13 | Writing, internal review against §6 | 0 |
| 14+ | Buffer, extra ablations, reproducibility package | — |

≈ 42 GPU-hours of real compute — under two weeks of your weekly quota, spread over
roughly three months, leaving substantial slack before a late-February deadline.
