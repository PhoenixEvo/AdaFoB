# AdaFoB — Phase 5 strategy and implementation specification

Status: post-alignment-gate. Oracle Dice 0.8944, edge-z 5.93, no flips.
Audience: the coding agent implementing in `models/FoB.py` and the experiment pipeline.
This document contains **no Python**. It specifies mathematics, interfaces, logic
and decision gates.

---

## 0. Decision

**None of Options A, B or C as stated. Take the reframe: keep the original thesis,
extend the budget range to include zero, and make the allocation ambiguity-aware.**

Rejected, with reasons:

- **Option A (refine positive prompts).** Abandons the FoB extension and lands in
  the most crowded corner of the field: Self-Prompt SAM, AutoProSAM, P2SAM,
  ProtoSAM and AM-SAM all do automated foreground prompting. You would be
  competing against fine-tuned methods on their home benchmark with ~30 GPU-h/week.
  Low novelty, unwinnable resource asymmetry.
- **Option B alone (change dataset until negatives matter).** Defensible only if
  the premise is verified *before* committing. Done blind it reads to a reviewer as
  benchmark shopping. It becomes principled when the dataset is chosen by a
  measured susceptibility criterion — which is Step E2 below, and which is itself
  reportable as method.
- **Option C alone (pure analysis paper).** Publishable, but a main-track analysis
  paper needs many datasets, several SAM variants and a mechanism. With your
  compute and team it lands at a workshop. Analysis *plus* a derived method is the
  standard winning structure, and you can have both.

**The reframe.** Your thesis was always "the background-prompt budget should
adapt." You have now measured that the correct budget is frequently **zero**, and
that FoB's fixed \(N_p{=}10\) costs up to **13.6 Dice points** when it is wrong.
FoB cannot represent zero. An allocator whose range includes zero beats a fixed
budget on exactly those cases — on Abd-CT, with the pipeline you already have.

The negative result is not a refutation. It is Figure 1.

**Working title.** *When Do Negative Prompts Help SAM? Saturation Analysis and
Adaptive Background-Prompt Budgeting for Few-Shot Medical Segmentation.*

**Two contributions, mutually reinforcing:**
1. **Analysis.** A saturation law: negative-prompt utility is governed by
   over-segmentation susceptibility, not by organ identity. Includes the
   counter-intuitive result that negatives *hurt* when positives are reliable.
2. **Method.** A prompt-budget allocator (PBA) that predicts, per episode, whether
   to spend any negative budget and how to place it. It must win on both ends of
   the spectrum: match-or-beat FoB where negatives are useless (by suppressing
   them) and beat it where they matter (by placing them better).

---

## 1. Mandatory precondition: re-measure the stale rows

Every ladder row containing model prompts was measured when the oracle was 0.4971,
i.e. on a broken pipeline. Those numbers are void. Stage 5 is fresh and stands.

**E0 — re-run the full attribution ladder** on the fixed pipeline, both organs,
both models. Do not begin implementation until `results/diagnosis.json` is
regenerated. Specifically required before any architectural work:

- `pos_inside_gt` for the FoB baseline (was 30.8% on the broken pipeline).
- `oracle_pos + model_neg` vs `oracle_pos + oracle_neg` — the true quality gap of
  FoB's predicted negatives.
- `model_pos + model_neg` — the real operating point.

If FoB's baseline Dice on Abd-CT now lands near its published ~0.85, the pipeline
is trustworthy end to end and the project is unblocked. If it is still far below,
stop and report — nothing below is meaningful.

---

## 2. E1 — the gating experiment (decides whether the method exists)

**Question.** Per episode, in the *realistic* regime (FoB's predicted foreground
prompts, not oracle), what is the optimal number of background prompts, and how
much is left on the table by fixing it at 10?

**Protocol.**
- For each evaluation episode, obtain FoB's predicted foreground prompts and its
  ranked predicted background prompts.
- Sweep \(N_p \in \{0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20\}\), feeding the top-\(N_p\)
  background prompts by the model's own confidence ordering.
- Record Dice and HD95 for every \(N_p\).
- **Compute the SAM image embedding once per query slice and reuse it across the
  whole sweep.** The encoder dominates cost; the mask decoder is milliseconds. This
  turns an 11-point sweep into roughly the cost of a single evaluation pass. This is
  non-negotiable for the compute budget.

**Report.**
- \(N^*\) per episode (argmax Dice; ties broken toward the smaller budget).
- Histogram of \(N^*\), per organ and per dataset. **Expected headline: a spike at 0.**
- Oracle-per-case Dice vs best-single-global-\(N_p\) Dice — the achievable headroom.
- Fraction of episodes where \(N^*=0\) and the mean Dice cost of forcing \(N_p{=}10\)
  on that subset. This is the number the method is designed to recover.

**Gate G1.** Proceed only if **either**
- per-case oracle exceeds best-global-\(N_p\) by ≥ 1.5 Dice on average, **or**
- ≥ 20% of episodes have \(N^*=0\) and forcing \(N_p{=}10\) costs ≥ 3 Dice on that subset.

If neither holds, the allocator has nothing to recover on this data; go to §7.

---

## 3. E2 — evidence-gated dataset selection

Do not choose a second dataset by intuition. Run the Stage-5 probe (positive-budget
sweep × {no negatives, oracle negatives}) on each candidate and rank by measured
negative-prompt headroom.

**Candidates, in order of cost-to-value:**

| Candidate | Why | Cost |
|---|---|---|
| **Pancreas, aorta, IVC, stomach, gallbladder from the SAME BTCV volumes** (labels 11, 8, 9, 7, 4) | Ambiguous boundaries against adjacent iso-intense tissue; over-segmentation is the dominant failure. Zero new data, zero new preprocessing, same protocol. | ~free |
| **Skin-DS / ISIC2018** | FoB publishes numbers on it, so comparability is free. Lesion borders are diffuse by definition — border irregularity is a diagnostic criterion. | low |
| **CMR myocardium** (ALPNet lineage) | Thin annulus against an iso-intense blood pool; classic leak case. Established FSMIS protocol. | medium |
| BraTS | Multi-modal MRI, own preprocessing, own sampling protocol. | high — defer |

**Strong recommendation: start with the extra BTCV labels.** You get a
high-ambiguity regime from volumes already on disk, and the resulting claim —
"the same method adapts across susceptibility regimes *within one dataset*" — is
harder to attack than a cross-dataset claim confounded by domain shift.

**Gate G2.** Select the two regimes with the largest and smallest measured
negative headroom. The paper needs both ends of the spectrum: suppression must be
shown to be as valuable as allocation.

---

## 4. Architecture specification — Prompt Budget Allocator (PBA)

Replaces the current `refine.*` module. Keep BPPC, BCM and SPR **unchanged**; they
are FoB's published modules.

### 4.1 Ambiguity score

Compute a scalar \(a \in [0,1]\) per episode from quantities already available at
inference, with **no ground truth**.

Primary signal, essentially free — FoB already computes both prototypes for its RAC
loss:
\[
a_{\text{proto}} \;=\; \frac{1}{2}\Bigl(1 + \cos\bigl(\mathbf{p}_{fg},\; \bar{\mathbf{p}}_{b}\bigr)\Bigr)
\]
where \(\mathbf{p}_{fg}\) is the foreground prototype and \(\bar{\mathbf{p}}_{b}\)
the mean background prompt prototype. High cosine similarity ⇒ foreground and
surrounding background are hard to separate in feature space ⇒ SAM is likely to leak
⇒ negatives are worth spending.

Secondary signals, both cheap:
\[
a_{\text{edge}} = 1 - \operatorname{norm}\!\left(\frac{\overline{|\nabla I^{q}|}\big|_{\partial \tilde{M}}}{\overline{|\nabla I^{q}|}\big|_{\text{body}}}\right),
\qquad
a_{\text{conf}} = \frac{\bigl|\{u : \tau_{lo} < C(u) < \tau_{hi}\}\bigr|}{|\tilde{M}| + \varepsilon}
\]
with \(C\) FoB's foreground similarity map, \(\tilde{M} = \mathbb{1}[C > \mathcal{T}]\)
the pre-mask, and \(\partial\tilde M\) its contour. Suggested \(\tau_{lo}{=}0.5\),
\(\tau_{hi}{=}0.9\) (\(\mathcal{T}{=}0.9\) is FoB's foreground threshold).

Combine with fixed weights, normalised on a held-out fold:
\[
a = w_1 a_{\text{proto}} + w_2 a_{\text{edge}} + w_3 a_{\text{conf}}, \qquad \textstyle\sum_i w_i = 1 .
\]
Start with \(w = (0.5, 0.3, 0.2)\) and treat \(w\) as an ablation, not a claim.

### 4.2 Budget

\[
N_p \;=\; \operatorname{clip}\Bigl(\bigl\lfloor \nu \, L \,\bigl(1 + \lambda\,\overline{|\kappa|}\bigr)\, g(a) \bigr\rceil,\; 0,\; N_{\max}\Bigr),
\qquad
g(a) = \sigma\!\left(\frac{a - a_0}{\tau}\right)
\]

- \(L\): support-mask contour length in pixels at 256×256.
- \(\overline{|\kappa|}\): mean absolute contour curvature, Gaussian-smoothed, scale-normalised.
- \(\nu\): prompts per unit boundary length — the single interpretable budget knob.
- \(g(a)\): the gate that makes **zero reachable**. This is the part FoB structurally cannot express and the part your data says is needed.
- \(N_{\max} = 24\). Note \(N_{\min} = 0\), deliberately.

### 4.3 Placement

Sample along the offset contour by inverse-CDF of an arc-length density that is
both curvature- and leak-aware:
\[
d(s) \;\propto\; \bigl(1 + \lambda|\hat{\kappa}(s)|\bigr)\bigl(1 + \gamma\,\hat{\ell}(s)\bigr)
\]
with a minimum geodesic spacing \(\delta\) between selected points, and
\[
\ell(s) \;=\; \exp\!\left(-\frac{\bigl(I_{\text{in}}(s) - I_{\text{out}}(s)\bigr)^{2}}{2\sigma_{I}^{2}}\right),
\]
where \(I_{\text{in}}(s), I_{\text{out}}(s)\) are local intensity means sampled a few
pixels inside and outside the boundary along the local normal at arc position \(s\).
\(\ell\) is high exactly where interior and exterior are indistinguishable — where
SAM leaks. **Spending negatives where leakage is possible rather than uniformly is
the core methodological novelty**; it is what distinguishes this from FoB's uniform
band sampling and from VesSAM's skeleton prompts.

### 4.4 Scale-adaptive offset

Replace FoB's constant \(r{=}15\) with \(r(A) = \operatorname{clip}(\alpha\sqrt{A/\pi},\, r_{\min}, r_{\max})\),
\(A\) the support-mask area. Suggested \(\alpha \approx 0.35\), \(r_{\min}{=}6\),
\(r_{\max}{=}24\); fit on fold 0.

### 4.5 Multi-component handling

Process each connected component of the offset region independently; allocate
budget proportional to per-component contour length; guarantee ≥ 1 prompt per
component whenever \(N_p > 0\). This removes the topological degeneracy of the
dilation band on thin or branching targets.

---

## 5. Integration requirements in `models/FoB.py`

1. **Verify \(N_p\)-agnosticism first, before anything else.** Locate every tensor
   whose shape depends on \(N_p\): the heatmap stack \(\mathbf{G} \in
   \mathbb{R}^{N_p \times H \times W}\), BCM's token count, SPR's graph
   \(\mathbf{A}^{ada}/\mathbf{A}^{ring}\), and the \(1/N_p\) normalisations in
   \(\mathcal{L}_{heat}\), \(\mathcal{L}_{coor}\), \(\mathcal{L}_{rac}\).
   Report which are shape-parametric. Batch size is 1, which makes variable
   \(N_p\) far easier than it looks.
2. **Support the \(N_p = 0\) path explicitly.** When the allocator returns 0, SAM
   receives foreground prompts only. Every downstream consumer — BCM, SPR, the loss
   terms, the SAM call — must handle an empty background set without a division by
   zero or a degenerate graph. Add an explicit unit test for \(N_p = 0\).
3. **Pad-and-mask, not per-episode loops,** for variable \(N_p\) during training.
   Pad to \(N_{\max}\) and mask in every loss normalisation and in graph construction.
4. **Parity test.** With the allocator forced to \(N_p{=}10\), uniform placement and
   \(r{=}15\), outputs must match unmodified FoB to `atol=1e-5`. This test is the
   guarantee that any measured difference comes from the allocator and not from a
   refactor. Do not proceed without it passing.
5. **Keep the frozen-backbone policy.** Gradients flow only to the allocator.
6. **Alignment policy must match `eval.py`** exactly; the detector is already shared
   through `data/preprocess.py`.

---

## 6. Fitting and training protocol

**Start with zero learned parameters.** \((\nu, \lambda, \gamma, a_0, \tau, \alpha)\)
is six scalars. Fit by grid search on fold 0 with cached SAM embeddings, evaluate on
folds 1–4. This gives a training-free method that:
- costs almost no GPU time,
- cannot suffer the instabilities that have already cost this project weeks,
- is trivially reproducible,
- and is a legitimate contribution — several 2025–26 FSMIS methods are training-free.

**Then, as an ablation only,** train a small learned allocator: input
\([a_{\text{proto}}, a_{\text{edge}}, a_{\text{conf}}, \log L, \log A, \overline{|\kappa|}]\)
plus a 128-d projection of the masked-average-pooled support feature; output a
budget logit. Supervise against \(N^*\) from E1 with an ordinal or L1 loss.
Report learned vs fitted vs oracle. If learned ≈ fitted, **say so and keep the
fitted version as the method** — that is a stronger result, not a weaker one.

---

## 7. Kill criteria and fallbacks

- **G1 fails** (no per-case headroom, few zeros): the allocator cannot win. Fall
  back to §0 Option C, but scoped properly — a saturation study across ≥ 3 datasets
  and ≥ 2 SAM variants (ViT-B/ViT-H, and SAM2 if it fits), with the susceptibility
  predictor as the deliverable. Target a MICCAI workshop or a journal, not a main track.
- **G2 fails** (no candidate regime shows negative headroom): the finding becomes
  "negative prompting is saturated in medical SAM across the board." That is a
  genuinely publishable negative result, and stronger than a weak positive one, but
  it must be demonstrated broadly to be believed.
- **Parity test (§5.4) fails:** stop. A refactor that changes FoB's behaviour
  invalidates every comparison.

---

## 8. Compute and timeline

Today is 2026-08-05. **FAIR 2026 (15 Aug) is 10 days away and is not achievable;
retire it.**

| Stage | Content | GPU-h | Calendar |
|---|---|---|---|
| E0 | Re-run ladder on fixed pipeline | 2 | 2 days |
| E1 | Budget sweep with cached embeddings, gate G1 | 4–6 | 1 week |
| E2 | Susceptibility probe on candidate regimes, gate G2 | 3 | 3 days |
| M1 | PBA implementation + parity test + \(N_p{=}0\) test | 2 | 1.5 weeks |
| M2 | Grid-fit on fold 0, validate folds 1–4 | 6 | 1 week |
| E3 | Full matrix: 2 regimes × Settings I/II × 5 folds | 15–20 | 2 weeks |
| E4 | Ablations, efficiency, statistics | 8 | 1 week |
| W | Writing, figures | ~0 | 3 weeks |

Roughly **40–45 GPU-hours of real compute** — inside two weeks of your quota,
spread over about ten weeks of calendar.

**Venue.** MICCAI 2027 (deadline ~Feb 2027) is the realistic top-tier target and the
better cultural fit: MICCAI reviewers value a rigorous negative-plus-method result.
CVPR 2027 (~Nov 2026) is possible only if E0–E2 all clear by early October. Be
candid with the team: for a prompt-engineering paper on two regimes with beginner
researchers and free-tier compute, MICCAI is the right ambition and a strong outcome.

---

## 9. The three figures that carry the paper

1. **Histogram of \(N^*\)** per regime, with the spike at zero on Abd-CT and mass at
   8–16 on the ambiguous regime. One figure, whole argument.
2. **Negative-prompt gain vs ambiguity score \(a\)**, scatter over all episodes with
   a fitted trend. This is the saturation law and it makes the analysis contribution
   quantitative rather than anecdotal.
3. **Dice cost of a fixed budget**: FoB \(N_p{=}10\) minus per-case oracle, stratified
   by \(a\). Shows precisely where and how much the fixed budget wastes.

---

## 10. What the coding agent must not do

- Do not rename or re-describe BPPC, BCM or SPR. They are FoB's published modules.
- Do not touch the alignment detector or the oracle gate thresholds to make results
  look better.
- Do not report any number produced with `--force`.
- Do not skip the \(N_p{=}10\) parity test (§5.4).
- Do not re-implement preprocessing anywhere; import `data/preprocess.py`.
- Do not tune \((\nu, \lambda, \gamma, a_0, \tau, \alpha)\) on any fold used for
  reporting. Fold 0 only.
