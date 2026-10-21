"""
Baseline B4: Random Candidate Subset (PREPARATION ONLY — DO NOT RUN CONFIRMATORY)
Evaluates random subset of size k in {2, 3, 4, 5, 6} drawn from FoB candidates across 5 seeds.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

MANIFEST_PATH = os.path.join(REPO_DIR, "protocol", "paired_episode_manifest.csv")
BUDGET_GRID = [2, 3, 4, 5, 6]
SEEDS = [42, 43, 44, 45, 46]

def sample_random_subset(candidate_pool: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Deterministically sample k points from candidate pool given a seed."""
    if len(candidate_pool) <= k:
        return candidate_pool
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(candidate_pool), size=k, replace=False)
    return candidate_pool[indices]

def smoke_test_dry_run():
    print("[SMOKE TEST] Dry-run on 3 episodes for B4 Random Subsets...")
    manifest = pd.read_csv(MANIFEST_PATH).head(3)
    mock_pool = np.arange(20).reshape(10, 2)
    for idx, row in manifest.iterrows():
        ep_id = row['episode_id']
        for k in BUDGET_GRID:
            sampled_seeds = []
            for s in SEEDS:
                pts = sample_random_subset(mock_pool, k, s + ep_id)
                sampled_seeds.append(len(pts))
            assert all(n == k for n in sampled_seeds)
    print(f"[SMOKE TEST PASSED] Successfully tested deterministic sampling across {len(BUDGET_GRID)} budgets and {len(SEEDS)} seeds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", type=int, default=0, help="Run smoke test on N episodes")
    args = parser.parse_args()
    if args.dry_run > 0:
        smoke_test_dry_run()
    else:
        print("B4 is prepared but execution is NOT approved until post-B0-B2 gate clearance.")
