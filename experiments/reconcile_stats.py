"""Reconciliation statistics for a clean 5-fold AdaFoB + LoRA evaluation CSV.

Input is one per-(fold, organ, volume) CSV produced by experiments/eval_lora.py
(+ summarize_results.py) with columns dice_*/hd95_* for the frozen baseline,
FoB+LoRA and AdaFoB+LoRA. The script reports:

  1. Descriptives (mean, median, IQR) per method, metric and scope (overall / organ).
  2. Pairwise paired tests for three contrasts and two metrics:
       - Wilcoxon signed-rank (observation level, ignores clustering),
       - cluster bootstrap CI and p-value (resampling whole volumes, keeping all
         organs and all folds of a physical volume together),
       - TOST equivalence (cluster bootstrap primary, paired t-test for contrast).
  3. Holm adjustment of the reported difference tests.
  4. Sensitivity of TOST conclusions to the equivalence margin.
  5. Failure-case tables (worst HD95, right kidney) and heavy-tail diagnostics.

Notes on the sampling unit: one CSV row is one (fold, organ, volume) triple. The
same physical volume can be a test volume in two folds (SABS fold index sets
overlap), so physical-volume clusters (vid) are the primary cluster unit and
fold-volume instances are reported as a sensitivity analysis.

Usage:
    python experiments/reconcile_stats.py \
        --csv results/clean_rerun_rank4_seed42_combined.csv \
        --out_dir results/reconcile_rank4_seed42
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Schema definitions (column names in the evaluation CSV), not hyperparameters.
METHOD_COLUMNS: Dict[str, Dict[str, str]] = {
    "baseline": {"dice": "dice_baseline", "hd95": "hd95_baseline"},
    "fob_lora": {"dice": "dice_lora_fob", "hd95": "hd95_lora_fob"},
    "ada_lora": {"dice": "dice_lora_ada", "hd95": "hd95_lora_ada"},
}
CONTRASTS: List[Tuple[str, str]] = [
    ("ada_lora", "baseline"),
    ("fob_lora", "baseline"),
    ("ada_lora", "fob_lora"),
]
METRICS: Tuple[str, str] = ("dice", "hd95")
ZERO_VARIANCE_TOL = 1e-12  # numerical tolerance below which paired differences are treated as constant
REQUIRED_COLUMNS = ["fold", "organ", "vol_id"] + [c for m in METHOD_COLUMNS.values() for c in m.values()]


def load_results(path: str, expected_rows: int) -> pd.DataFrame:
    """Load and validate the evaluation CSV.

    Returns a DataFrame with extra columns `vid` (int physical volume id parsed from
    vol_id) and `inst` (str fold-volume instance id). Raises ValueError on any
    structural problem (missing columns, unparsable volume id, duplicate keys,
    unexpected row count, unbalanced fold-organ cells).
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    vid = df["vol_id"].astype(str).str.extract(r"image_(\d+)")[0]
    if vid.isna().any():
        raise ValueError("Could not parse volume id from vol_id for some rows")
    df["vid"] = vid.astype(int)
    df["inst"] = df["fold"].astype(str) + "_" + df["vid"].astype(str)
    if df.duplicated(subset=["fold", "organ", "vid"]).any():
        raise ValueError("Duplicate (fold, organ, volume) rows present")
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(df)}")
    cell_counts = df.groupby(["fold", "organ"]).size()
    if cell_counts.nunique() != 1:
        raise ValueError(f"Unbalanced fold-organ cells:\n{cell_counts}")
    return df


def describe(x: np.ndarray) -> Dict[str, float]:
    """Mean, median, quartiles (linear interpolation), IQR, min, max of a 1-D array."""
    x = np.asarray(x, dtype=float)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {
        "n": int(x.size), "mean": float(x.mean()), "median": float(med),
        "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
        "min": float(x.min()), "max": float(x.max()),
    }


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def cluster_bootstrap_means(values: np.ndarray, clusters: np.ndarray, n_boot: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the mean of `values`, resampling whole clusters.

    Each replicate draws K cluster ids with replacement (K = number of clusters)
    and computes the row-weighted mean of the drawn clusters.
    """
    codes, uniques = pd.factorize(pd.Series(clusters))
    k = len(uniques)
    sums = np.bincount(codes, weights=np.asarray(values, dtype=float), minlength=k)
    counts = np.bincount(codes, minlength=k).astype(float)
    idx = rng.integers(0, k, size=(n_boot, k))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def boot_p_two_sided(boot: np.ndarray) -> float:
    """Two-sided percentile-bootstrap p-value for H0: mean difference = 0."""
    b = boot.size
    le = (np.sum(boot <= 0) + 1) / (b + 1)
    ge = (np.sum(boot >= 0) + 1) / (b + 1)
    return float(min(1.0, 2.0 * min(le, ge)))


def tost_from_boot(boot: np.ndarray, margin: float) -> Dict[str, float]:
    """TOST on a bootstrap distribution of the mean difference (equivalence bound +/- margin).

    p_tost = max(P(boot <= -margin), P(boot >= +margin)) with +1 smoothing. Equivalence
    at alpha=0.05 is equivalent to the 90% percentile interval lying inside (-margin, margin).
    """
    b = boot.size
    p_lo = (np.sum(boot <= -margin) + 1) / (b + 1)
    p_hi = (np.sum(boot >= margin) + 1) / (b + 1)
    lo, hi = np.percentile(boot, [5, 95])
    return {"p_tost": float(max(p_lo, p_hi)), "ci90_lo": float(lo), "ci90_hi": float(hi)}


def tost_paired_t(diff: np.ndarray, margin: float) -> float:
    """Observation-level paired-t TOST p-value (does NOT account for clustering)."""
    sd = float(np.std(diff, ddof=1))
    if sd < ZERO_VARIANCE_TOL:
        return 0.0 if abs(float(np.mean(diff))) < margin else 1.0
    p_lo = stats.ttest_1samp(diff, -margin, alternative="greater").pvalue
    p_hi = stats.ttest_1samp(diff, margin, alternative="less").pvalue
    return float(max(p_lo, p_hi))


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 when all paired differences are zero."""
    if np.all(a == b):
        return 1.0
    return float(stats.wilcoxon(a, b, alternative="two-sided").pvalue)


def git_state(repo_dir: str) -> Dict[str, object]:
    """Current git commit and whether tracked files are modified."""
    def run(args: List[str]) -> str:
        return subprocess.check_output(["git"] + args, cwd=repo_dir).decode("utf-8").strip()
    return {
        "git_commit": run(["rev-parse", "HEAD"]),
        "git_dirty_tracked": bool(run(["status", "--porcelain", "--untracked-files=no"])),
    }


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pairwise_table(df: pd.DataFrame, args: argparse.Namespace, rng: np.random.Generator) -> pd.DataFrame:
    """All contrast x metric x scope tests with Holm adjustment."""
    margins = {"dice": args.dice_margin, "hd95": args.hd95_margin}
    scopes = [("overall", df)] + [(o, df[df["organ"] == o]) for o in sorted(df["organ"].unique())]
    rows: List[Dict[str, object]] = []
    for scope, sub in scopes:
        for (a_name, b_name), metric in [(c, m) for c in CONTRASTS for m in METRICS]:
            a = sub[METHOD_COLUMNS[a_name][metric]].to_numpy(float)
            b = sub[METHOD_COLUMNS[b_name][metric]].to_numpy(float)
            diff = a - b
            boot_vid = cluster_bootstrap_means(diff, sub["vid"].to_numpy(), args.n_boot, rng)
            boot_inst = cluster_bootstrap_means(diff, sub["inst"].to_numpy(), args.n_boot, rng)
            tost = tost_from_boot(boot_vid, margins[metric])
            rows.append({
                "scope": scope, "contrast": f"{a_name}_minus_{b_name}", "metric": metric,
                "n_rows": int(diff.size), "n_clusters_vid": int(sub["vid"].nunique()),
                "n_clusters_inst": int(sub["inst"].nunique()),
                "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
                "wilcoxon_p_obs": wilcoxon_p(a, b),
                "boot_vid_ci95_lo": float(np.percentile(boot_vid, 2.5)),
                "boot_vid_ci95_hi": float(np.percentile(boot_vid, 97.5)),
                "boot_vid_p": boot_p_two_sided(boot_vid),
                "boot_inst_ci95_lo": float(np.percentile(boot_inst, 2.5)),
                "boot_inst_ci95_hi": float(np.percentile(boot_inst, 97.5)),
                "boot_inst_p": boot_p_two_sided(boot_inst),
                "tost_margin": margins[metric],
                "tost_vid_p": tost["p_tost"], "tost_vid_ci90_lo": tost["ci90_lo"],
                "tost_vid_ci90_hi": tost["ci90_hi"],
                "tost_vid_equivalent": bool(tost["p_tost"] < args.alpha),
                "tost_naive_t_p": tost_paired_t(diff, margins[metric]),
            })
    out = pd.DataFrame(rows)
    for col, name in [("wilcoxon_p_obs", "wilcoxon"), ("boot_vid_p", "bootvid")]:
        out[f"holm_all_{name}"] = holm_adjust(out[col].to_numpy())
        out[f"holm_scope_{name}"] = np.nan
        for is_overall in (True, False):
            mask = (out["scope"] == "overall") == is_overall
            out.loc[mask, f"holm_scope_{name}"] = holm_adjust(out.loc[mask, col].to_numpy())
    overall = out["scope"] == "overall"
    out["tost_vid_holm_overall"] = np.nan
    out.loc[overall, "tost_vid_holm_overall"] = holm_adjust(out.loc[overall, "tost_vid_p"].to_numpy())
    return out


def margin_sensitivity(df: pd.DataFrame, args: argparse.Namespace, rng: np.random.Generator) -> pd.DataFrame:
    """TOST (overall, physical-volume cluster bootstrap) over a grid of margins."""
    rows: List[Dict[str, object]] = []
    grids = {"dice": args.dice_margin_grid, "hd95": args.hd95_margin_grid}
    for (a_name, b_name), metric in [(c, m) for c in CONTRASTS for m in METRICS]:
        diff = (df[METHOD_COLUMNS[a_name][metric]] - df[METHOD_COLUMNS[b_name][metric]]).to_numpy(float)
        boot = cluster_bootstrap_means(diff, df["vid"].to_numpy(), args.n_boot, rng)
        for margin in grids[metric]:
            t = tost_from_boot(boot, margin)
            rows.append({"contrast": f"{a_name}_minus_{b_name}", "metric": metric, "margin": margin,
                         "mean_diff": float(diff.mean()), "ci90_lo": t["ci90_lo"], "ci90_hi": t["ci90_hi"],
                         "p_tost": t["p_tost"], "equivalent": bool(t["p_tost"] < args.alpha)})
    return pd.DataFrame(rows)


def descriptives_table(df: pd.DataFrame, tail_mm: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    scopes = [("overall", df)] + [(o, df[df["organ"] == o]) for o in sorted(df["organ"].unique())]
    for scope, sub in scopes:
        for method, cols in METHOD_COLUMNS.items():
            for metric in METRICS:
                x = sub[cols[metric]].to_numpy(float)
                d = describe(x)
                d.update({"scope": scope, "method": method, "metric": metric,
                          "n_above_tail_mm": int(np.sum(x > tail_mm)) if metric == "hd95" else np.nan})
                rows.append(d)
    return pd.DataFrame(rows)[["scope", "method", "metric", "n", "mean", "median", "q1", "q3", "iqr",
                               "min", "max", "n_above_tail_mm"]]


def failure_cases(df: pd.DataFrame, top_n: int, organ: str) -> pd.DataFrame:
    """Worst-case tables: worst HD95 overall, and worst HD95 / lowest Dice for `organ` (AdaFoB+LoRA)."""
    keep = ["fold", "organ", "vid"] + [c for m in METHOD_COLUMNS.values() for c in m.values()]
    parts = []
    t = df.sort_values("hd95_lora_ada", ascending=False).head(top_n)[keep].copy()
    t.insert(0, "table", "worst_hd95_ada_overall")
    parts.append(t)
    sub = df[df["organ"] == organ]
    t = sub.sort_values("hd95_lora_ada", ascending=False).head(top_n)[keep].copy()
    t.insert(0, "table", f"worst_hd95_ada_{organ}")
    parts.append(t)
    t = sub.sort_values("dice_lora_ada", ascending=True).head(top_n)[keep].copy()
    t.insert(0, "table", f"lowest_dice_ada_{organ}")
    parts.append(t)
    return pd.concat(parts, ignore_index=True)


def tail_diagnostics(df: pd.DataFrame, tail_mm: float) -> pd.DataFrame:
    """Share of the mean HD95 contributed by rows above `tail_mm`, per method."""
    rows = []
    for method, cols in METHOD_COLUMNS.items():
        x = df[cols["hd95"]].to_numpy(float)
        tail = x > tail_mm
        rows.append({
            "method": method, "tail_threshold_mm": tail_mm, "n_rows": int(x.size),
            "n_tail": int(tail.sum()), "mean_all": float(x.mean()),
            "mean_without_tail": float(x[~tail].mean()) if (~tail).any() else np.nan,
            "share_of_total_sum_from_tail": float(x[tail].sum() / x.sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconciliation statistics for a clean eval CSV")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--expected_rows", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dice_margin", type=float, default=0.01, help="TOST margin in Dice fraction (0.01 = 1.0 point)")
    parser.add_argument("--hd95_margin", type=float, default=3.0, help="TOST margin in mm")
    parser.add_argument("--dice_margin_grid", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--hd95_margin_grid", type=float, nargs="+", default=[1.5, 3.0, 5.0])
    parser.add_argument("--hd95_tail_mm", type=float, default=100.0)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--failure_organ", type=str, default="RK")
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(args.out_dir, exist_ok=False)
    df = load_results(args.csv, args.expected_rows)
    rng = np.random.default_rng(args.seed)

    desc = descriptives_table(df, args.hd95_tail_mm)
    pairs = pairwise_table(df, args, rng)
    sens = margin_sensitivity(df, args, rng)
    fail = failure_cases(df, args.top_n, args.failure_organ)
    tail = tail_diagnostics(df, args.hd95_tail_mm)

    desc.to_csv(os.path.join(args.out_dir, "descriptives.csv"), index=False)
    pairs.to_csv(os.path.join(args.out_dir, "pairwise_tests.csv"), index=False)
    sens.to_csv(os.path.join(args.out_dir, "tost_margin_sensitivity.csv"), index=False)
    fail.to_csv(os.path.join(args.out_dir, "failure_cases.csv"), index=False)
    tail.to_csv(os.path.join(args.out_dir, "tail_diagnostics.csv"), index=False)

    config = dict(vars(args))
    config.update({
        "input_csv_sha256": sha256_file(args.csv),
        "n_rows": int(len(df)),
        "n_clusters_vid": int(df["vid"].nunique()),
        "n_clusters_inst": int(df["inst"].nunique()),
        "volumes_in_multiple_folds": sorted(int(v) for v, g in df.groupby("vid") if g["fold"].nunique() > 1),
        "numpy": np.__version__, "pandas": pd.__version__,
    })
    config.update(git_state(repo_dir))
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote reconciliation outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
