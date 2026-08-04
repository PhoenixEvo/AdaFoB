# Phase 2 Pilot Summary

**Status**: TEMPLATE - fill after running `notebooks/phase2_pilot_kaggle.ipynb`

## Claim A: Shape-adaptive Np vs Fixed Np on Irregular Structures

**Hypothesis**: Heuristic Np (A4) beats the best fixed Np (best of A1/A2/A3) on irregular (BraTS) cases.

### Results (fill after experiment)

| Arm | Dataset | Dice (mean +/- std) | HD95 (mean +/- std) |
|-----|---------|---------------------|---------------------|
| A1 ring Np=5   | BraTS | | |
| A2 ring Np=10  | BraTS | | |
| A3 ring Np=20  | BraTS | | |
| A4 ring heuristic | BraTS | | |

**Wilcoxon A4 vs best_fixed (Dice, BraTS)**: stat=___, p=___

**Verdict**: ___

---

## Claim B: Skeleton Prior vs Ring Prior on HD95

**Hypothesis**: Skeleton-derived placement (A5) beats ring prior (A2) on HD95, especially on irregular structures.

### Results (fill after experiment)

| Arm | Dataset | Dice (mean +/- std) | HD95 (mean +/- std) |
|-----|---------|---------------------|---------------------|
| A2 ring Np=10       | BraTS | | |
| A5 skeleton heuristic | BraTS | | |
| A6 skeleton Np=10   | BraTS | | |

**Wilcoxon A5 vs A2 (HD95, BraTS)**: stat=___, p=___

**Verdict**: ___

---

## Regression Check: Compact Abd-CT Controls

| Arm | Dataset | Dice (mean +/- std) | HD95 (mean +/- std) |
|-----|---------|---------------------|---------------------|
| A2 ring Np=10       | AbdCT | | |
| A4 ring heuristic   | AbdCT | | |
| A5 skeleton heuristic | AbdCT | | |

**Verdict**: Does skeleton/heuristic hurt on compact organs? ___

---

## Failure Cases

(List any cases where skeleton prior or heuristic Np performed significantly worse)

---

## GPU Time

- Total GPU time: ___ hours
- Total wall time: ___ hours
- Total inferences: ___ (cases x arms)

## Decision

- [ ] Proceed to Phase 3 (both claims supported)
- [ ] Proceed to Phase 3 with modifications (partial support)
- [ ] Abandon project (both claims refuted)
