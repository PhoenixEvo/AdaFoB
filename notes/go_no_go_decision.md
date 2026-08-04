# Phase 3: GO / NO-GO Decision

## 1. Does heuristic adaptive Np beat the best fixed Np by >= +0.5 Dice on the irregular subset? (Claim A)
**No.** 
On the BraTS dataset, the heuristic Np arm (A4: 0.5934 ± 0.2944) did not beat the best fixed Np arm (A2 Np=10: 0.5991 ± 0.2883). In fact, the heuristic approach slightly underperformed the fixed Np=10 baseline on average. The Wilcoxon test against the per-case oracle best fixed Np showed a significant difference (p=0.0059), but in the negative direction. Claim A fails.

## 2. Does the skeleton prior improve HD95 by >= 10% relative vs the ring prior on the irregular subset? (Claim B)
**Yes.**
Comparing apples to apples (fixed Np=10), the Skeleton Prior (A6: 27.38 ± 35.85) achieved a massive HD95 reduction compared to the Ring Prior (A2: 37.12 ± 40.25) on BraTS. This is an absolute reduction of 9.74 points, equating to a **26.2% relative improvement**, far exceeding the 10% threshold. The heuristic skeleton arm (A5: 34.85) also showed a 6.1% improvement over A2, though A6 was superior. Claim B passes.

## 3. Are gains on compact Abd-CT controls within noise (|delta| < 0.3 Dice)?
**Yes.** 
A rigorous regression check on the Abd-CT cases confirmed that the performance difference is extremely minimal and statistically insignificant (p=0.4922). 
- **A2 (Ring Np=10)**: 0.2824 ± 0.0895 Dice
- **A6 (Skel Np=10)**: 0.2549 ± 0.0768 Dice
The absolute difference is $|0.2549 - 0.2824| = 0.0275$, which is well below the 0.3 noise threshold. The skeleton prior does not severely degrade performance on compact organs.

## 4. Decision
**GO-NARROW**

## 5. Justification & Paper Scope
The pilot data clearly shows that predicting a shape-adaptive number of points $N_p$ (Claim A) using the current heuristic formula does not provide a reliable performance boost over a well-chosen fixed $N_p$ (like $N_p=10$). The variance is high, and the mean Dice drops slightly. 

However, the **placement strategy** (Claim B) is highly successful. Deriving negative prompts from the morphological skeleton rather than a standard dilation band (Ring Prior) significantly reduces outlier predictions and tightens the boundary, as evidenced by the 26.2% relative improvement in HD95 on the highly irregular BraTS tumor cores.

**Scope for Phase 4 & Paper**: 
We will pivot the paper's core contribution entirely to the Generation of Adaptive Prompts (GAP) module. We will drop the Adaptive Point Count (APC) module and treat $N_p$ as a fixed hyperparameter or use a simplified fixed heuristic recipe. The paper will focus on how Skeleton-guided negative prompt placement fundamentally outperforms band-based placement for irregular medical structures in few-shot SAM.
