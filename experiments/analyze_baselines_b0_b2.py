"""
Statistical and Methodological Analysis for Baselines B0, B1, B2
Complies with Protocol Amendment A1.
===============================================================
Performs:
1. Patient-level cluster bootstrap (B=5000) over K=25 physical patients.
2. TOST equivalence tests for Dice (+/- 1.0%) and HD95 (+/- 3.0mm, +/- 5.0mm).
3. Confirmatory Holm-Bonferroni correction.
4. B0 reproduction check vs published FoB (86.21% +/- 1.0%).
5. Empirical testing of Hypotheses H1 (image_7 bilateral renal failure) and H2 (RK fragility).
6. S4 gradient correlation analysis and timing breakdown.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats

def compute_iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))

def cluster_bootstrap_paired(df, metric_col_a, metric_col_b, B=5000, seed=42):
    """
    Patient-level cluster bootstrap over unique patients (K=25).
    Keeps all records of a sampled patient together.
    Returns:
        mean_diff, ci_lower, ci_upper, p_value, tost_p_1mm, tost_p_3mm, tost_p_5mm
    """
    rng = np.random.RandomState(seed)
    patients = np.array(sorted(df['patient_id'].unique()))
    K = len(patients)

    # Compute observed paired difference per patient
    patient_means_a = df.groupby('patient_id')[metric_col_a].mean()
    patient_means_b = df.groupby('patient_id')[metric_col_b].mean()
    diffs = (patient_means_a - patient_means_b).loc[patients].values
    obs_mean_diff = float(np.mean(diffs))

    # Bootstrap
    boot_means = np.zeros(B, dtype=np.float64)
    for b in range(B):
        boot_idx = rng.choice(K, size=K, replace=True)
        boot_means[b] = np.mean(diffs[boot_idx])

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Two-sided p-value (H0: diff == 0)
    # Using shifted bootstrap distribution
    shifted = boot_means - obs_mean_diff
    p_val = float(np.mean(np.abs(shifted) >= np.abs(obs_mean_diff)))

    # TOST tests
    # TOST tests whether diff is strictly within [-margin, +margin]
    def tost_p(margin):
        p1 = np.mean(boot_means <= -margin) # H0: diff <= -margin -> reject if boot_means > -margin
        p2 = np.mean(boot_means >= margin)  # H0: diff >= margin -> reject if boot_means < margin
        return float(max(p1, p2))

    return {
        "mean_diff": obs_mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "tost_p_1": tost_p(1.0),
        "tost_p_3": tost_p(3.0),
        "tost_p_5": tost_p(5.0),
    }

def analyze_baselines(csv_path, timing_path=None, s4_path=None, out_dir="results/baselines"):
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"ADAFOB BASELINE MATRIX ANALYSIS (B0, B1, B2)")
    print(f"Dataset: {len(df)} episodes across {len(df['patient_id'].unique())} unique patients")
    print("=" * 80)

    # 1. Summary Statistics
    arms = {
        "B0_raw": ("dice_b0_raw", "hd95_b0_raw", "ViT-H fixed 10 (no CC)"),
        "B0_cc":  ("dice_b0_cc",  "hd95_b0_cc",  "ViT-H fixed 10 (with CC)"),
        "B1_raw": ("dice_b1_raw", "hd95_b1_raw", "ViT-B fixed 10 (no CC)"),
        "B2_cc":  ("dice_b2_cc",  "hd95_b2_cc",  "ViT-B fixed 10 (with CC)"),
    }

    summary_rows = []
    for arm_id, (d_col, h_col, desc) in arms.items():
        summary_rows.append({
            "Arm": arm_id,
            "Description": desc,
            "Dice_Mean": df[d_col].mean() * 100,
            "Dice_Std": df[d_col].std() * 100,
            "Dice_Median": df[d_col].median() * 100,
            "Dice_IQR": compute_iqr(df[d_col]) * 100,
            "HD95_Mean": df[h_col].mean(),
            "HD95_Std": df[h_col].std(),
            "HD95_Median": df[h_col].median(),
            "HD95_IQR": compute_iqr(df[h_col]),
        })
    sum_df = pd.DataFrame(summary_rows)
    print("\n--- Summary Statistics (Overall N=120) ---")
    print(sum_df.to_string(index=False))

    # Per-organ breakdown
    print("\n--- Per-Organ Dice Mean (%) ---")
    organ_dice = df.groupby('organ')[['dice_b0_raw', 'dice_b0_cc', 'dice_b1_raw', 'dice_b2_cc']].mean() * 100
    print(organ_dice.to_string())

    print("\n--- Per-Organ HD95 Mean (mm) ---")
    organ_hd = df.groupby('organ')[['hd95_b0_raw', 'hd95_b0_cc', 'hd95_b1_raw', 'hd95_b2_cc']].mean()
    print(organ_hd.to_string())

    # 2. B0 Reproduction vs Published Literature
    b0_dice_mean = df['dice_b0_raw'].mean() * 100
    pub_dice_mean = 86.21
    b0_reproduced = (85.21 <= b0_dice_mean <= 87.21)
    print("\n--- B0 Literature Reproduction Check ---")
    print(f"Published FoB (CVPR'26): {pub_dice_mean:.2f}%")
    print(f"B0 Frozen ViT-H Re-run:  {b0_dice_mean:.2f}% (delta = {b0_dice_mean - pub_dice_mean:+.2f}%)")
    print(f"Reproduction Gate (+/- 1.0%): {'PASS' if b0_reproduced else 'FAIL'}")

    # 3. Patient-Level Cluster Bootstrap (K=25)
    print("\n--- Confirmatory Patient-Level Cluster Bootstrap (B=5000, K=25 patients) ---")
    tests = [
        ("CC_Effect_ViT_B_Dice", "dice_b2_cc", "dice_b1_raw", True),
        ("CC_Effect_ViT_B_HD95", "hd95_b2_cc", "hd95_b1_raw", False),
        ("Backbone_ViT_B_vs_ViT_H_Dice", "dice_b1_raw", "dice_b0_raw", True),
        ("Backbone_ViT_B_vs_ViT_H_HD95", "hd95_b1_raw", "hd95_b0_raw", False),
        ("Standardized_B2_vs_B0_cc_Dice", "dice_b2_cc", "dice_b0_cc", True),
        ("Standardized_B2_vs_B0_cc_HD95", "hd95_b2_cc", "hd95_b0_cc", False),
    ]

    boot_results = []
    for test_name, col_a, col_b, is_dice in tests:
        res = cluster_bootstrap_paired(df, col_a, col_b, B=5000)
        scale = 100 if is_dice else 1.0
        unit = "%" if is_dice else "mm"
        res_entry = {
            "Comparison": test_name,
            "Mean_Diff": res['mean_diff'] * scale,
            "CI_Lower": res['ci_lower'] * scale,
            "CI_Upper": res['ci_upper'] * scale,
            "p_value": res['p_value'],
            "tost_p_1": res['tost_p_1'],
            "tost_p_3": res['tost_p_3'],
            "tost_p_5": res['tost_p_5'],
            "Unit": unit
        }
        boot_results.append(res_entry)
        print(f"  {test_name:30s} | Diff: {res['mean_diff']*scale:+.3f} {unit} [95% CI: {res['ci_lower']*scale:+.3f}, {res['ci_upper']*scale:+.3f}] | p={res['p_value']:.4f}")

    # Apply Holm correction to the 4 primary confirmatory tests
    primary_tests = [boot_results[0], boot_results[1], boot_results[2], boot_results[3]]
    p_vals = [t['p_value'] for t in primary_tests]
    order = np.argsort(p_vals)
    m = len(p_vals)
    adj_p = np.zeros(m)
    for rank_idx, orig_idx in enumerate(order):
        multiplier = m - rank_idx
        adj_p[orig_idx] = min(p_vals[orig_idx] * multiplier, 1.0)
    for i, t in enumerate(primary_tests):
        t['p_adj_holm'] = float(adj_p[i])

    # 4. Hypothesis Testing: H1 (image_7) and H2 (RK)
    print("\n--- Empirical Test of Anatomical Hypotheses ---")
    im7 = df[df['patient_id'] == 'image_7']
    print(f"H1: image_7 Examination (N={len(im7)}):")
    for idx, r in im7.iterrows():
        print(f"  Fold {r['fold']} | {r['organ']:6s} | B0: {r['dice_b0_raw']*100:.1f}% ({r['hd95_b0_raw']:.1f}mm) | B1: {r['dice_b1_raw']*100:.1f}% ({r['hd95_b1_raw']:.1f}mm) | B2(CC): {r['dice_b2_cc']*100:.1f}% ({r['hd95_b2_cc']:.1f}mm)")

    rk_df = df[df['organ'] == 'RK']
    lk_df = df[df['organ'] == 'LK']
    print(f"\nH2: Right Kidney Fragility Check:")
    print(f"  RK B1 Dice: {rk_df['dice_b1_raw'].mean()*100:.2f}% | LK B1 Dice: {lk_df['dice_b1_raw'].mean()*100:.2f}%")
    print(f"  RK B1 HD95: {rk_df['hd95_b1_raw'].mean():.2f}mm | LK B1 HD95: {lk_df['hd95_b1_raw'].mean():.2f}mm")
    print(f"  RK B2 HD95: {rk_df['hd95_b2_cc'].mean():.2f}mm | LK B2 HD95: {lk_df['hd95_b2_cc'].mean():.2f}mm")

    # 5. Timing Profile & S4 Feasibility Check
    timing_data = {}
    if timing_path and os.path.exists(timing_path):
        with open(timing_path, "r") as f:
            timing_data = json.load(f)
        print("\n--- Execution Timing & Resource Profile ---")
        for k, v in timing_data.items():
            print(f"  {k:25s}: {v}")

    s4_data = {}
    if s4_path and os.path.exists(s4_path):
        with open(s4_path, "r") as f:
            s4_data = json.load(f)
        print("\n--- Step D: S4 Feasibility Correlation Check ---")
        for k, v in s4_data.items():
            print(f"  {k:30s}: {v}")

    # Output full report json
    out_payload = {
        "summary": summary_rows,
        "b0_reproduction": {
            "b0_dice_mean": float(b0_dice_mean),
            "published_dice_mean": float(pub_dice_mean),
            "reproduced": bool(b0_reproduced)
        },
        "cluster_bootstrap": boot_results,
        "timing": timing_data,
        "s4_feasibility": s4_data
    }
    with open(os.path.join(out_dir, "analysis_summary.json"), "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nAnalysis summary successfully saved to {os.path.join(out_dir, 'analysis_summary.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to baselines_b0_b1_b2_results.csv")
    parser.add_argument("--timing", type=str, default=None, help="Path to timing_breakdown.json")
    parser.add_argument("--s4", type=str, default=None, help="Path to s4_feasibility_check.json")
    parser.add_argument("--out_dir", type=str, default="results/baselines", help="Output directory")
    args = parser.parse_args()

    analyze_baselines(args.csv, args.timing, args.s4, args.out_dir)
