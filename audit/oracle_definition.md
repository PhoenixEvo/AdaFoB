# GT-guided references, not exact ceilings

Source: experiments/run_step5_diagnostics.py at starting commit 063192a.
All references use ViT-B, R1 (underlying token 0), fixed positive prompts and one frozen ten-negative FoB pool per slice. Query GT is diagnostic-only here.

| Item | GT-guided prefix reference | GT-guided greedy reference |
|---|---|---|
| Decision unit | Each 2D query slice | Each 2D query slice |
| Initialization | Positive-only mask (zero negatives) | Positive-only mask (zero negatives) |
| Available candidates | Prefixes k=0..10 | Remaining original negative channels |
| Trajectory | Test k=1..10; replace on strictly greater Dice | Try every one-point addition to current subset; retain best improving candidate |
| Objective | Raw slice Dice | Raw slice Dice |
| Stopping | Exhaust all eleven prefix budgets | No addition exceeds running best by >1e-4, or all selected |
| Ties | Earlier/smaller prefix retained | Original channel traversal; update when >current step best +1e-4 |
| Output | Assemble selected raw masks into a volume, then apply frozen 2D CC | Same |
| Negative counts / zero-choice frequency | NOT LOGGED | NOT LOGGED |
| Search log / selected indices | NOT LOGGED | NOT LOGGED |

The greedy comparison includes an incremental 1e-4 rule during scanning, so ties and near-ties depend on channel traversal. This is neither exhaustive search nor a guaranteed volume-level optimizer. Slice Dice weights differ from volume Dice; subsequent CC can change the ordering again. Greedy raw volume Dice can even be worse than positive-only despite nondecreasing raw slice objectives.

Decoder accounting from source, for a ten-point pool: five fixed-budget calls + fifteen random calls + ten redecoded prefixes + up to 55 greedy trial calls = 30..85 predict invocations per slice. The greedy search itself reuses m_k0 and costs 10..55 additional trials. No-prompt calls return zero without a decoder invocation; exceptions are caught and converted to zeros. Thus these are source-derived call bounds, not measured successful-call counts. Prefix reuses k0 but redundantly decodes k2/4/6/10. Image embedding is cached once per query slice.

Saved CSV comparison (no new inference):
- Greedy CC Dice is worse than prefix CC in episodes 20, 40, 63; largest deficit 16.7152 pp.
- Greedy raw Dice is worse than prefix raw in the same three episodes.
- It is worse than fixed-10 CC in episodes 40, 44, 63 and 93 (largest deficit 9.1834 pp).
- Full counterexamples, including k0/k2/k4/k6, are in audit/step6_20261023/artifacts/greedy_counterexamples.json.

Episode-weighted CC Dice: full-set 82.5492%, prefix reference 85.3095%, greedy reference 87.0440%. The +4.4948 pp greedy gain is privileged-label diagnostic opportunity, not a deployable benefit or guaranteed headroom.

There is no need for an exhaustive 2^10 experiment to establish this limitation. Recovering historical point counts would require rerunning the diagnostic search; it is not justified in this batch. Future bounded diagnostic runs must log counts, zero-choice frequency, path, selected IDs and actual decoder calls.
