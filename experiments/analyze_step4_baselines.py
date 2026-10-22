"""
Rigorous Statistical Analysis of Step 4 Symmetric Baseline Matrix
- Patient-level cluster bootstrap (B=5000 iterations, resampling K=25 patients)
- Pre-specified vs exploratory contrast labeling
- Holm-Bonferroni family-wise error rate control
- Per-organ breakdowns, medians, IQR, 90th percentile tails, failure counts
- Sensitivity analysis: complete cohort vs leave-image_7-out cohort
- Re-testing of Hypothesis H2 (RK vs LK) under corrected symmetric configuration
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

def compute_cluster_bootstrap(df_eval, col_a, col_b, b_iters=5000, seed=42):
    patients = df_eval['patient_id'].unique()
    K = len(patients)
    np.random.seed(seed)
    
    # Pre-extract paired patient diffs
    pat_diffs = {}
    for p, group in df_eval.groupby('patient_id'):
        pat_diffs[p] = (group[col_a] - group[col_b]).values
        
    p_keys = np.array(list(pat_diffs.keys()))
    boot_means = []
    
    for _ in range(b_iters):
        boot_pats = np.random.choice(p_keys, size=K, replace=True)
        sample_diffs = np.concatenate([pat_diffs[p] for p in boot_pats])
        boot_means.append(float(np.mean(sample_diffs)))
        
    boot_means = np.array(boot_means)
    mean_diff = float(np.mean(df_eval[col_a] - df_eval[col_b]))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    
    # Two-sided empirical p-value
    if mean_diff > 0:
        p_val = 2.0 * float(np.mean(boot_means <= 0))
    else:
        p_val = 2.0 * float(np.mean(boot_means >= 0))
    p_val = min(1.0, max(1.0 / b_iters, p_val))
    
    return mean_diff, ci_lower, ci_upper, p_val


def analyze_step4():
    csv_path = "results/step4_baselines/step4_baselines_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} episodes across {df['patient_id'].nunique()} unique patients.")
    assert len(df) == 120, f"Expected 120 episodes, found {len(df)}"

    # 1. Primary Pre-specified & Exploratory Contrasts
    contrasts = [
        # (name, col_a, col_b, contrast_type, description)
        ("CC Effect on ViT-H Fixed-10 (Dice)", "dice_b3h_cc", "dice_b3h_raw", "Pre-specified", "B3H CC vs B3H raw"),
        ("CC Effect on ViT-H Fixed-10 (HD95)", "hd95_b3h_cc", "hd95_b3h_raw", "Pre-specified", "B3H CC vs B3H raw"),
        ("CC Effect on ViT-B Fixed-10 (Dice)", "dice_b3b_cc", "dice_b3b_raw", "Pre-specified", "B3B CC vs B3B raw"),
        ("CC Effect on ViT-B Fixed-10 (HD95)", "hd95_b3b_cc", "hd95_b3b_raw", "Pre-specified", "B3B CC vs B3B raw"),
        
        ("Background Effect on ViT-H (Dice)", "dice_b3h_cc", "dice_b4h_cc", "Pre-specified", "B3H CC vs B4H CC (Fixed-10 vs Pos-only)"),
        ("Background Effect on ViT-H (HD95)", "hd95_b3h_cc", "hd95_b4h_cc", "Pre-specified", "B3H CC vs B4H CC (Fixed-10 vs Pos-only)"),
        ("Background Effect on ViT-B (Dice)", "dice_b3b_cc", "dice_b4b_cc", "Pre-specified", "B3B CC vs B4B CC (Fixed-10 vs Pos-only)"),
        ("Background Effect on ViT-B (HD95)", "hd95_b3b_cc", "hd95_b4b_cc", "Pre-specified", "B3B CC vs B4B CC (Fixed-10 vs Pos-only)"),

        ("Corrected Architecture Parity Fixed-10 (Dice)", "dice_b3b_cc", "dice_b3h_cc", "Pre-specified", "B3B CC vs B3H CC"),
        ("Corrected Architecture Parity Fixed-10 (HD95)", "hd95_b3b_cc", "hd95_b3h_cc", "Pre-specified", "B3B CC vs B3H CC"),
        ("Corrected Architecture Parity Pos-only (Dice)", "dice_b4b_cc", "dice_b4h_cc", "Pre-specified", "B4B CC vs B4H CC"),
        ("Corrected Architecture Parity Pos-only (HD95)", "hd95_b4b_cc", "hd95_b4h_cc", "Pre-specified", "B4B CC vs B4H CC"),

        ("Rule R1 vs Upstream R3 on ViT-H (Dice)", "dice_b3h_cc", "dice_b0_cc", "Pre-specified", "B3H CC vs B0 CC (ViT-H SingleMask vs Upstream mask[0])"),
        ("Rule R1 vs Upstream R3 on ViT-H (HD95)", "hd95_b3h_cc", "hd95_b0_cc", "Pre-specified", "B3H CC vs B0 CC (ViT-H SingleMask vs Upstream mask[0])"),

        ("Rule R1 Correction Gain on ViT-B (Dice)", "dice_b3b_cc", "dice_b0_cc", "Exploratory", "B3B CC vs B0 CC (ViT-B corrected vs ViT-H upstream)"),
    ]

    contrast_results = []
    p_vals = []

    for name, c_a, c_b, c_type, desc in contrasts:
        mean_diff, ci_l, ci_u, p_val = compute_cluster_bootstrap(df, c_a, c_b)
        p_vals.append(p_val)
        contrast_results.append({
            "contrast_name": name,
            "col_a": c_a, "col_b": c_b,
            "contrast_type": c_type,
            "description": desc,
            "mean_diff": mean_diff,
            "ci_95": [ci_l, ci_u],
            "unadjusted_p": p_val,
        })

    # Holm-Bonferroni correction
    order = np.argsort(p_vals)
    n_tests = len(p_vals)
    holm_p = [0.0] * n_tests
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = p_vals[idx] * (n_tests - rank)
        running_max = max(running_max, adj)
        holm_p[idx] = min(1.0, running_max)

    for i in range(n_tests):
        contrast_results[i]["holm_adjusted_p"] = holm_p[i]

    # 2. Overall Summary Table (Mean, Std, Median, IQR, 90th Tail, Failures)
    arms = [
        ("B0 (ViT-H mask0)", "dice_b0_raw", "hd95_b0_raw", "dice_b0_cc", "hd95_b0_cc"),
        ("B3H (ViT-H R1 Fixed-10)", "dice_b3h_raw", "hd95_b3h_raw", "dice_b3h_cc", "hd95_b3h_cc"),
        ("B3B (ViT-B R1 Fixed-10)", "dice_b3b_raw", "hd95_b3b_raw", "dice_b3b_cc", "hd95_b3b_cc"),
        ("B4H (ViT-H R1 Pos-only)", "dice_b4h_raw", "hd95_b4h_raw", "dice_b4h_cc", "hd95_b4h_cc"),
        ("B4B (ViT-B R1 Pos-only)", "dice_b4b_raw", "hd95_b4b_raw", "dice_b4b_cc", "hd95_b4b_cc"),
    ]

    summary_rows = []
    for arm_label, dr, hr, dc, hc in arms:
        # Raw
        d_raw = df[dr] * 100.0 if df[dr].max() <= 1.0 else df[dr]
        h_raw = df[hr]
        # CC
        d_cc = df[dc] * 100.0 if df[dc].max() <= 1.0 else df[dc]
        h_cc = df[hc]

        summary_rows.append({
            "arm": arm_label,
            "dice_raw_mean": float(d_raw.mean()), "dice_raw_std": float(d_raw.std()),
            "dice_raw_median": float(d_raw.median()), "dice_raw_iqr": float(d_raw.quantile(0.75) - d_raw.quantile(0.25)),
            "hd95_raw_mean": float(h_raw.mean()), "hd95_raw_std": float(h_raw.std()),
            "hd95_raw_median": float(h_raw.median()), "hd95_raw_p90": float(h_raw.quantile(0.90)),
            
            "dice_cc_mean": float(d_cc.mean()), "dice_cc_std": float(d_cc.std()),
            "dice_cc_median": float(d_cc.median()), "dice_cc_iqr": float(d_cc.quantile(0.75) - d_cc.quantile(0.25)),
            "hd95_cc_mean": float(h_cc.mean()), "hd95_cc_std": float(h_cc.std()),
            "hd95_cc_median": float(h_cc.median()), "hd95_cc_p90": float(h_cc.quantile(0.90)),
            
            "n_failures_raw": int((d_raw < 10.0).sum()),
            "n_failures_cc": int((d_cc < 10.0).sum()),
        })

    # 3. Per-Organ Breakdown Table
    per_organ_breakdown = {}
    for organ in df['organ'].unique():
        sub = df[df['organ'] == organ]
        per_organ_breakdown[organ] = {
            "n_episodes": len(sub),
            "b0_cc_dice": float((sub['dice_b0_cc'] * 100.0 if sub['dice_b0_cc'].max() <= 1.0 else sub['dice_b0_cc']).mean()),
            "b3h_cc_dice": float((sub['dice_b3h_cc'] * 100.0 if sub['dice_b3h_cc'].max() <= 1.0 else sub['dice_b3h_cc']).mean()),
            "b3b_cc_dice": float((sub['dice_b3b_cc'] * 100.0 if sub['dice_b3b_cc'].max() <= 1.0 else sub['dice_b3b_cc']).mean()),
            "b4h_cc_dice": float((sub['dice_b4h_cc'] * 100.0 if sub['dice_b4h_cc'].max() <= 1.0 else sub['dice_b4h_cc']).mean()),
            "b4b_cc_dice": float((sub['dice_b4b_cc'] * 100.0 if sub['dice_b4b_cc'].max() <= 1.0 else sub['dice_b4b_cc']).mean()),
            
            "b0_cc_hd95": float(sub['hd95_b0_cc'].mean()),
            "b3h_cc_hd95": float(sub['hd95_b3h_cc'].mean()),
            "b3b_cc_hd95": float(sub['hd95_b3b_cc'].mean()),
            "b4h_cc_hd95": float(sub['hd95_b4h_cc'].mean()),
            "b4b_cc_hd95": float(sub['hd95_b4b_cc'].mean()),
        }

    # 4. Sensitivity Analysis: Complete Cohort vs Leave image_7 Out
    df_no7 = df[~df['patient_id'].str.contains('image_7')]
    sensitivity_summary = {
        "full_cohort_n": len(df),
        "leave_image7_out_n": len(df_no7),
        "b3h_cc_dice_full": float((df['dice_b3h_cc'] * 100.0 if df['dice_b3h_cc'].max() <= 1.0 else df['dice_b3h_cc']).mean()),
        "b3h_cc_dice_no7": float((df_no7['dice_b3h_cc'] * 100.0 if df_no7['dice_b3h_cc'].max() <= 1.0 else df_no7['dice_b3h_cc']).mean()),
        "b3b_cc_dice_full": float((df['dice_b3b_cc'] * 100.0 if df['dice_b3b_cc'].max() <= 1.0 else df['dice_b3b_cc']).mean()),
        "b3b_cc_dice_no7": float((df_no7['dice_b3b_cc'] * 100.0 if df_no7['dice_b3b_cc'].max() <= 1.0 else df_no7['dice_b3b_cc']).mean()),
        
        "b3h_cc_hd95_full": float(df['hd95_b3h_cc'].mean()),
        "b3h_cc_hd95_no7": float(df_no7['hd95_b3h_cc'].mean()),
        "b3b_cc_hd95_full": float(df['hd95_b3b_cc'].mean()),
        "b3b_cc_hd95_no7": float(df_no7['hd95_b3b_cc'].mean()),
    }

    # 5. Re-test of Hypothesis H2 (RK vs LK) under corrected symmetric configurations
    rk_sub = df[df['organ'] == 'RK'].sort_values('patient_id')
    lk_sub = df[df['organ'] == 'LK'].sort_values('patient_id')
    
    # Paired comparisons per patient
    rk_b3h_dice = rk_sub['dice_b3h_cc'].values * 100.0 if rk_sub['dice_b3h_cc'].max() <= 1.0 else rk_sub['dice_b3h_cc'].values
    lk_b3h_dice = lk_sub['dice_b3h_cc'].values * 100.0 if lk_sub['dice_b3h_cc'].max() <= 1.0 else lk_sub['dice_b3h_cc'].values
    rk_b3b_dice = rk_sub['dice_b3b_cc'].values * 100.0 if rk_sub['dice_b3b_cc'].max() <= 1.0 else rk_sub['dice_b3b_cc'].values
    lk_b3b_dice = lk_sub['dice_b3b_cc'].values * 100.0 if lk_sub['dice_b3b_cc'].max() <= 1.0 else lk_sub['dice_b3b_cc'].values

    h2_retest = {
        "vit_h_b3h_rk_mean_dice": float(np.mean(rk_b3h_dice)),
        "vit_h_b3h_lk_mean_dice": float(np.mean(lk_b3h_dice)),
        "vit_h_b3h_rk_minus_lk_diff": float(np.mean(rk_b3h_dice - lk_b3h_dice)),
        "vit_h_b3h_paired_p": float(stats.ttest_rel(rk_b3h_dice, lk_b3h_dice)[1]),
        
        "vit_b_b3b_rk_mean_dice": float(np.mean(rk_b3b_dice)),
        "vit_b_b3b_lk_mean_dice": float(np.mean(lk_b3b_dice)),
        "vit_b_b3b_rk_minus_lk_diff": float(np.mean(rk_b3b_dice - lk_b3b_dice)),
        "vit_b_b3b_paired_p": float(stats.ttest_rel(rk_b3b_dice, lk_b3b_dice)[1]),
    }

    full_analysis = {
        "contrasts": contrast_results,
        "summary_table": summary_rows,
        "per_organ_breakdown": per_organ_breakdown,
        "sensitivity_image7": sensitivity_summary,
        "hypothesis_h2_retest": h2_retest
    }

    out_json = "results/step4_baselines/step4_analysis_summary.json"
    with open(out_json, "w") as f:
        json.dump(full_analysis, f, indent=2)
    print(f"\n[Analysis Complete] Wrote statistical summary to {out_json}", flush=True)


if __name__ == "__main__":
    analyze_step4()
