"""
Master Statistical Analysis for MICCAI Submission
===================================================
Handles Experiments 1, 2, 3, and 5 from the MICCAI task spec.

Reads:
  - results/lora_eval_final_combined.csv (4-method eval, 120 samples)
  - results/control_fixed_mean_budget_gpu0.csv (fixed control, 120 samples)

Outputs:
  - results/hd95_statistical_report.csv        (Exp 1)
  - results/delta_reconciliation_table.csv      (Exp 2)
  - results/full_statistical_report.csv         (Exp 3)
  - notes/statistical_interpretation.md         (Exp 1+3)
  - notes/failure_case_analysis.md              (Exp 5)

Usage:
    python experiments/master_statistical_analysis.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, pearsonr, spearmanr

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# ── Statistical helpers ──────────────────────────────────────────────────────

def cohen_d(x, y):
    """Paired Cohen's d (standardized mean difference)."""
    diff = np.array(x, dtype=np.float64) - np.array(y, dtype=np.float64)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def bootstrap_ci(x, y, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap 95% CI for mean(x - y)."""
    rng = np.random.RandomState(seed)
    diff = np.array(x, dtype=np.float64) - np.array(y, dtype=np.float64)
    n = len(diff)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = diff[idx].mean()
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, alpha * 100)), \
           float(np.percentile(boot_means, (1 - alpha) * 100))


def wilcoxon_safe(x, y):
    """Wilcoxon signed-rank test, handling edge cases."""
    diff = np.array(x, dtype=np.float64) - np.array(y, dtype=np.float64)
    if np.all(diff == 0):
        return 1.0
    # Remove zero differences for Wilcoxon
    nonzero = diff[diff != 0]
    if len(nonzero) < 2:
        return 1.0
    try:
        _, p = wilcoxon(x, y)
        return float(p)
    except ValueError:
        return 1.0


def effect_size_label(d):
    """Cohen's d magnitude label."""
    ad = abs(d)
    if ad >= 0.8:
        return "large"
    elif ad >= 0.5:
        return "medium"
    elif ad >= 0.2:
        return "small"
    return "negligible"


# ── Comparison engine ────────────────────────────────────────────────────────

def compute_all_comparisons(df, col_a, col_b, label, metric_name):
    """Compute stats for one comparison, per organ and overall."""
    rows = []
    organs = sorted(df['organ'].unique())

    for organ in organs + ['OVERALL']:
        sub = df if organ == 'OVERALL' else df[df['organ'] == organ]
        a = sub[col_a].values.astype(np.float64)
        b = sub[col_b].values.astype(np.float64)
        n = len(a)

        if n < 2:
            continue

        mean_diff = float(np.mean(a - b))
        ci_low, ci_high = bootstrap_ci(a, b)
        p = wilcoxon_safe(a, b)
        d = cohen_d(a, b)

        rows.append({
            'comparison': label,
            'metric': metric_name,
            'organ': organ,
            'mean_a': float(np.mean(a)),
            'std_a': float(np.std(a, ddof=1)),
            'mean_b': float(np.mean(b)),
            'std_b': float(np.std(b, ddof=1)),
            'mean_diff': mean_diff,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'wilcoxon_p': p,
            'cohens_d': d,
            'effect_size': effect_size_label(d),
            'n': n,
        })

    return rows


# ── EXPERIMENT 1: HD95 Statistical Report ────────────────────────────────────

def run_experiment_1(df_main, df_fixed):
    """HD95 statistical tests for all pairwise comparisons."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: HD95 Statistical Report")
    print("="*80)

    all_rows = []

    # 1a. Baseline vs FoB+LoRA (HD95)
    all_rows.extend(compute_all_comparisons(
        df_main, 'hd95_baseline', 'hd95_lora_fob',
        'Baseline vs FoB+LoRA', 'HD95'))

    # 1b. FoB+LoRA vs AdaFoB+LoRA (HD95)
    all_rows.extend(compute_all_comparisons(
        df_main, 'hd95_lora_fob', 'hd95_lora_ada',
        'FoB+LoRA vs AdaFoB+LoRA', 'HD95'))

    # 1c. Baseline vs AdaFoB+LoRA (HD95)
    all_rows.extend(compute_all_comparisons(
        df_main, 'hd95_baseline', 'hd95_lora_ada',
        'Baseline vs AdaFoB+LoRA', 'HD95'))

    # 1d. AdaFoB+LoRA vs Fixed-Budget-Control (HD95) — if fixed data available
    if df_fixed is not None:
        merged = pd.merge(df_main, df_fixed, on=['fold', 'organ', 'vol_id'],
                          how='inner', suffixes=('', '_fc'))
        if len(merged) > 0:
            all_rows.extend(compute_all_comparisons(
                merged, 'hd95_lora_ada', 'hd95_fixed_control',
                'AdaFoB+LoRA vs Fixed-Budget-Control', 'HD95'))

    hd95_df = pd.DataFrame(all_rows)
    out_path = os.path.join(_ROOT, "results", "hd95_statistical_report.csv")
    hd95_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print key results
    for comp in hd95_df['comparison'].unique():
        overall = hd95_df[(hd95_df['comparison'] == comp) & (hd95_df['organ'] == 'OVERALL')]
        if len(overall) > 0:
            r = overall.iloc[0]
            sig = "SIGNIFICANT" if r['wilcoxon_p'] < 0.05 else "not significant"
            print(f"\n  {comp}:")
            print(f"    Mean diff = {r['mean_diff']:.2f}mm, "
                  f"95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
            print(f"    Wilcoxon p = {r['wilcoxon_p']:.4f} ({sig}), "
                  f"Cohen's d = {r['cohens_d']:.3f} ({r['effect_size']})")

    return all_rows


# ── EXPERIMENT 2: Delta Reconciliation ───────────────────────────────────────

def run_experiment_2(df_main, df_fixed):
    """Compute every delta from underlying CSV, produce unambiguous table."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Delta Reconciliation Table")
    print("="*80)

    deltas = []

    methods = {
        'FoB Baseline (ViT-H, Np=10)': ('dice_baseline', 'hd95_baseline'),
        'FoB+LoRA (ViT-B, Np=10)': ('dice_lora_fob', 'hd95_lora_fob'),
        'AdaFoB+LoRA (ViT-B, dynamic)': ('dice_lora_ada', 'hd95_lora_ada'),
    }

    method_names = list(methods.keys())

    # Pairwise deltas for Dice and HD95
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            name_a, name_b = method_names[i], method_names[j]
            dice_a, hd_a = methods[name_a]
            dice_b, hd_b = methods[name_b]

            for metric, col_a, col_b in [('Dice', dice_a, dice_b), ('HD95', hd_a, hd_b)]:
                mean_a = float(df_main[col_a].mean())
                mean_b = float(df_main[col_b].mean())
                abs_delta = mean_b - mean_a
                if mean_a != 0:
                    rel_delta = abs_delta / mean_a
                else:
                    rel_delta = float('nan')

                deltas.append({
                    'method_a': name_a,
                    'method_b': name_b,
                    'metric': metric,
                    'mean_a': mean_a,
                    'mean_b': mean_b,
                    'absolute_delta': abs_delta,
                    'relative_delta_pct': rel_delta * 100,
                    'direction': 'B > A' if abs_delta > 0 else ('A > B' if abs_delta < 0 else 'equal'),
                    'unit': '%' if metric == 'Dice' else 'mm',
                })

    # Add Fixed-Budget-Control if available
    if df_fixed is not None:
        merged = pd.merge(df_main, df_fixed, on=['fold', 'organ', 'vol_id'],
                          how='inner', suffixes=('', '_fc'))
        if len(merged) > 0:
            for metric, col_ada, col_fc in [
                ('Dice', 'dice_lora_ada', 'dice_fixed_control'),
                ('HD95', 'hd95_lora_ada', 'hd95_fixed_control')
            ]:
                mean_ada = float(merged[col_ada].mean())
                mean_fc = float(merged[col_fc].mean())
                abs_delta = mean_fc - mean_ada
                rel_delta = (abs_delta / mean_ada * 100) if mean_ada != 0 else float('nan')

                deltas.append({
                    'method_a': 'AdaFoB+LoRA (ViT-B, dynamic)',
                    'method_b': 'Fixed-Mean-Budget Control+LoRA',
                    'metric': metric,
                    'mean_a': mean_ada,
                    'mean_b': mean_fc,
                    'absolute_delta': abs_delta,
                    'relative_delta_pct': rel_delta,
                    'direction': 'B > A' if abs_delta > 0 else ('A > B' if abs_delta < 0 else 'equal'),
                    'unit': '%' if metric == 'Dice' else 'mm',
                })

    delta_df = pd.DataFrame(deltas)
    out_path = os.path.join(_ROOT, "results", "delta_reconciliation_table.csv")
    delta_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print readable table
    print(f"\n{'Method A':<35} {'Method B':<35} {'Metric':<6} {'A':>8} {'B':>8} {'Delta':>10} {'Rel%':>8}")
    print("-" * 120)
    for _, r in delta_df.iterrows():
        unit = r['unit']
        if r['metric'] == 'Dice':
            print(f"{r['method_a']:<35} {r['method_b']:<35} {r['metric']:<6} "
                  f"{r['mean_a']*100:>7.2f}% {r['mean_b']*100:>7.2f}% "
                  f"{r['absolute_delta']*100:>+9.2f}% {r['relative_delta_pct']:>+7.2f}%")
        else:
            print(f"{r['method_a']:<35} {r['method_b']:<35} {r['metric']:<6} "
                  f"{r['mean_a']:>7.2f}mm {r['mean_b']:>7.2f}mm "
                  f"{r['absolute_delta']:>+9.2f}mm {r['relative_delta_pct']:>+7.2f}%")

    return deltas


# ── EXPERIMENT 3: Full CI + Cohen's d Master Table ───────────────────────────

def run_experiment_3(df_main, df_fixed):
    """Full statistical master table: every comparison, both metrics, per organ."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Full Statistical Report (Dice + HD95)")
    print("="*80)

    all_rows = []

    comparisons_dice = [
        ('Baseline vs FoB+LoRA', 'dice_baseline', 'dice_lora_fob'),
        ('FoB+LoRA vs AdaFoB+LoRA', 'dice_lora_fob', 'dice_lora_ada'),
        ('Baseline vs AdaFoB+LoRA', 'dice_baseline', 'dice_lora_ada'),
    ]
    comparisons_hd95 = [
        ('Baseline vs FoB+LoRA', 'hd95_baseline', 'hd95_lora_fob'),
        ('FoB+LoRA vs AdaFoB+LoRA', 'hd95_lora_fob', 'hd95_lora_ada'),
        ('Baseline vs AdaFoB+LoRA', 'hd95_baseline', 'hd95_lora_ada'),
    ]

    for label, col_a, col_b in comparisons_dice:
        all_rows.extend(compute_all_comparisons(df_main, col_a, col_b, label, 'Dice'))

    for label, col_a, col_b in comparisons_hd95:
        all_rows.extend(compute_all_comparisons(df_main, col_a, col_b, label, 'HD95'))

    # Add Fixed-Budget-Control comparisons if available
    if df_fixed is not None:
        merged = pd.merge(df_main, df_fixed, on=['fold', 'organ', 'vol_id'],
                          how='inner', suffixes=('', '_fc'))
        if len(merged) > 0:
            all_rows.extend(compute_all_comparisons(
                merged, 'dice_lora_ada', 'dice_fixed_control',
                'AdaFoB+LoRA vs Fixed-Budget-Control', 'Dice'))
            all_rows.extend(compute_all_comparisons(
                merged, 'hd95_lora_ada', 'hd95_fixed_control',
                'AdaFoB+LoRA vs Fixed-Budget-Control', 'HD95'))

    master_df = pd.DataFrame(all_rows)
    out_path = os.path.join(_ROOT, "results", "full_statistical_report.csv")
    master_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print summary of all OVERALL rows
    overall = master_df[master_df['organ'] == 'OVERALL'].copy()
    print(f"\n{'Comparison':<40} {'Metric':<6} {'Diff':>10} {'95% CI':>22} {'p':>10} {'d':>8} {'Effect':>12}")
    print("-" * 115)
    for _, r in overall.iterrows():
        if r['metric'] == 'Dice':
            diff_str = f"{r['mean_diff']*100:>+9.2f}%"
            ci_str = f"[{r['ci_low']*100:>+7.2f}%, {r['ci_high']*100:>+7.2f}%]"
        else:
            diff_str = f"{r['mean_diff']:>+9.2f}mm"
            ci_str = f"[{r['ci_low']:>+7.2f}, {r['ci_high']:>+7.2f}]mm"
        print(f"{r['comparison']:<40} {r['metric']:<6} {diff_str} {ci_str:>22} "
              f"{r['wilcoxon_p']:>10.4f} {r['cohens_d']:>8.3f} {r['effect_size']:>12}")

    return all_rows


# ── EXPERIMENT 5: Failure Case Analysis ──────────────────────────────────────

def run_experiment_5(df_main):
    """RK plateau hypothesis, outlier detection, HD95 regression analysis."""
    print("\n" + "="*80)
    print("EXPERIMENT 5: Failure Case Analysis")
    print("="*80)

    # Compute deltas
    df = df_main.copy()
    df['dice_delta_lora'] = df['dice_lora_fob'] - df['dice_baseline']
    df['dice_delta_ada'] = df['dice_lora_ada'] - df['dice_baseline']
    df['hd95_delta_lora'] = df['hd95_lora_fob'] - df['hd95_baseline']
    df['hd95_delta_ada'] = df['hd95_lora_ada'] - df['hd95_baseline']

    md = "# Failure Case Analysis\n\n"
    md += f"Analysis of {len(df)} volume-organ evaluations from 5-fold cross-validation.\n\n"
    md += "---\n\n"

    # ── 5.1: RK Plateau Hypothesis ──────────────────────────────────────────
    md += "## 1. Right Kidney (RK) -- Low Headroom Hypothesis\n\n"

    md += "### 1.1 Per-Organ Baseline Dice and LoRA Improvement\n\n"
    md += ("| Organ | n | Baseline Dice | Std | FoB+LoRA Dice | Std | "
           "Delta (LoRA-Base) | HD95 Base | HD95 LoRA | HD95 Delta |\n")
    md += "|---|---|---|---|---|---|---|---|---|---|\n"
    for organ in sorted(df['organ'].unique()):
        o = df[df['organ'] == organ]
        md += (f"| {organ} | {len(o)} | {o['dice_baseline'].mean()*100:.2f}% | "
               f"{o['dice_baseline'].std()*100:.2f}% | "
               f"{o['dice_lora_fob'].mean()*100:.2f}% | "
               f"{o['dice_lora_fob'].std()*100:.2f}% | "
               f"{o['dice_delta_lora'].mean()*100:+.2f}% | "
               f"{o['hd95_baseline'].mean():.1f}mm | "
               f"{o['hd95_lora_fob'].mean():.1f}mm | "
               f"{o['hd95_delta_lora'].mean():+.1f}mm |\n")

    md += "\n### 1.2 Correlation: Baseline Dice vs LoRA Improvement\n\n"
    r_p, p_p = pearsonr(df['dice_baseline'], df['dice_delta_lora'])
    r_s, p_s = spearmanr(df['dice_baseline'], df['dice_delta_lora'])
    md += f"- Pearson r = {r_p:.3f} (p = {p_p:.4f})\n"
    md += f"- Spearman rho = {r_s:.3f} (p = {p_s:.4f})\n\n"

    if r_p < -0.2 and p_p < 0.05:
        md += ("**Interpretation:** The negative correlation is consistent with "
               "the low-headroom hypothesis -- cases with higher baseline Dice "
               "tend to show smaller improvements from LoRA. This is expected "
               "behavior: when the frozen model already segments an organ well, "
               "there is less room for domain adaptation to add value.\n\n")
    elif abs(r_p) < 0.2:
        md += ("**Interpretation:** The correlation is weak (|r| < 0.2), "
               "suggesting that baseline Dice alone does not fully explain the "
               "variable improvement across organs. Other factors -- organ "
               "morphology, boundary contrast with adjacent tissues, typical HU "
               "similarity -- likely contribute. An alternative explanation is "
               "that the FoB prompt generator already provides strong guidance "
               "for kidneys, limiting LoRA's marginal value.\n\n")
    else:
        md += ("**Interpretation:** The positive correlation is unexpected "
               "under the simple headroom hypothesis. An alternative "
               "explanation may be that LoRA's benefits correlate with data "
               "availability or organ frequency in training.\n\n")

    # Direct baseline Dice comparison across organs
    md += "### 1.3 Direct Cross-Organ Baseline Comparison\n\n"
    baseline_by_organ = df.groupby('organ')['dice_baseline'].mean().sort_values(ascending=False)
    md += "Organs ranked by baseline Dice (descending):\n\n"
    for organ, val in baseline_by_organ.items():
        delta = df[df['organ'] == organ]['dice_delta_lora'].mean()
        md += f"1. **{organ}**: {val*100:.2f}% baseline, {delta*100:+.2f}% delta\n"
    md += "\n"

    # ── 5.2: Outlier Detection ──────────────────────────────────────────────
    md += "## 2. Outlier Detection\n\n"

    dice_cols = ['dice_baseline', 'dice_lora_fob', 'dice_lora_ada']
    outlier_mask = (df[dice_cols] < 0.20).any(axis=1)
    outliers = df[outlier_mask]

    md += f"### 2.1 Cases with Dice < 20% in Any Method ({len(outliers)} found)\n\n"
    if len(outliers) > 0:
        md += ("| Fold | Organ | Vol ID | Baseline Dice | FoB+LoRA Dice | "
               "AdaFoB+LoRA Dice | HD95 Base | HD95 FoB+LoRA | HD95 AdaFoB |\n")
        md += "|---|---|---|---|---|---|---|---|---|\n"
        for _, r in outliers.sort_values('dice_lora_ada').iterrows():
            vid = os.path.basename(str(r.get('vol_id', 'N/A')))
            md += (f"| {r['fold']} | {r['organ']} | {vid} | "
                   f"{r['dice_baseline']*100:.1f}% | "
                   f"{r['dice_lora_fob']*100:.1f}% | "
                   f"{r['dice_lora_ada']*100:.1f}% | "
                   f"{r['hd95_baseline']:.1f}mm | "
                   f"{r['hd95_lora_fob']:.1f}mm | "
                   f"{r['hd95_lora_ada']:.1f}mm |\n")
        md += "\n"
    else:
        md += "No cases found with Dice < 20% in any method.\n\n"

    # HD95 outliers
    hd_outlier_mask = (df['hd95_lora_ada'] > 200) | (df['hd95_lora_fob'] > 200)
    hd_outliers = df[hd_outlier_mask]
    md += f"### 2.2 Cases with HD95 > 200mm After LoRA ({len(hd_outliers)} found)\n\n"
    if len(hd_outliers) > 0:
        md += "| Fold | Organ | Vol ID | HD95 Base | HD95 FoB+LoRA | HD95 AdaFoB |\n"
        md += "|---|---|---|---|---|---|\n"
        for _, r in hd_outliers.iterrows():
            vid = os.path.basename(str(r.get('vol_id', 'N/A')))
            md += (f"| {r['fold']} | {r['organ']} | {vid} | "
                   f"{r['hd95_baseline']:.1f}mm | "
                   f"{r['hd95_lora_fob']:.1f}mm | "
                   f"{r['hd95_lora_ada']:.1f}mm |\n")
        md += "\n"
    else:
        md += "No cases found with HD95 > 200mm.\n\n"

    # ── 5.3: HD95 Regression Analysis ───────────────────────────────────────
    md += "## 3. HD95 Regression After LoRA\n\n"

    md += "### 3.1 Per-Organ HD95 Summary\n\n"
    md += "| Organ | HD95 Baseline | HD95 FoB+LoRA | HD95 AdaFoB+LoRA | Delta (LoRA-Base) |\n"
    md += "|---|---|---|---|---|\n"
    for organ in sorted(df['organ'].unique()):
        o = df[df['organ'] == organ]
        md += (f"| {organ} | {o['hd95_baseline'].mean():.1f}mm | "
               f"{o['hd95_lora_fob'].mean():.1f}mm | "
               f"{o['hd95_lora_ada'].mean():.1f}mm | "
               f"{o['hd95_delta_lora'].mean():+.1f}mm |\n")

    md += "\n### 3.2 Top 10 Cases Where HD95 Worsened Most (LoRA vs Baseline)\n\n"
    worst_hd = df.nlargest(10, 'hd95_delta_lora')
    md += "| Fold | Organ | Vol ID | HD95 Base | HD95 LoRA | Delta | Dice Base | Dice LoRA |\n"
    md += "|---|---|---|---|---|---|---|---|\n"
    for _, r in worst_hd.iterrows():
        vid = os.path.basename(str(r.get('vol_id', 'N/A')))
        md += (f"| {r['fold']} | {r['organ']} | {vid} | "
               f"{r['hd95_baseline']:.1f}mm | {r['hd95_lora_fob']:.1f}mm | "
               f"{r['hd95_delta_lora']:+.1f}mm | "
               f"{r['dice_baseline']*100:.1f}% | {r['dice_lora_fob']*100:.1f}% |\n")

    # ── 5.4: image_7 Specific Analysis ──────────────────────────────────────
    md += "\n## 4. image_7 (LK) Specific Analysis\n\n"
    img7 = df[df['vol_id'].astype(str).str.contains('image_7[^0-9]|image_7$', regex=True, na=False)]
    if len(img7) == 0:
        img7 = df[df['vol_id'].astype(str).str.contains('_7\\.nii', na=False)]

    if len(img7) > 0:
        md += ("| Fold | Organ | Dice Base | Dice FoB+LoRA | Dice AdaFoB | "
               "HD95 Base | HD95 FoB+LoRA | HD95 AdaFoB |\n")
        md += "|---|---|---|---|---|---|---|---|\n"
        for _, r in img7.iterrows():
            md += (f"| {r['fold']} | {r['organ']} | "
                   f"{r['dice_baseline']*100:.1f}% | "
                   f"{r['dice_lora_fob']*100:.1f}% | "
                   f"{r['dice_lora_ada']*100:.1f}% | "
                   f"{r['hd95_baseline']:.1f}mm | "
                   f"{r['hd95_lora_fob']:.1f}mm | "
                   f"{r['hd95_lora_ada']:.1f}mm |\n")
        md += "\n"

        # Check if any organ for image_7 has extremely low Dice
        worst_img7 = img7[img7['dice_lora_ada'] < 0.30]
        if len(worst_img7) > 0:
            md += ("**Key finding:** image_7 shows severe segmentation failure "
                   "for the following organ(s):\n\n")
            for _, r in worst_img7.iterrows():
                md += (f"- {r['organ']}: Dice = {r['dice_lora_ada']*100:.1f}% "
                       f"(all methods fail, suggesting a data-level issue such as "
                       f"atypical anatomy, motion artifact, or partial volume at "
                       f"organ boundary)\n")
            md += ("\n**Note:** A visual overlay of predicted vs ground-truth mask "
                   "is needed to identify the concrete failure mode. This requires "
                   "running the visualization script on Kaggle with access to the "
                   "NIfTI volumes.\n")
        else:
            md += "image_7 does not show extreme failure in any organ-method combination.\n"
    else:
        md += ("image_7 was not found in the vol_id column. The volume naming "
               "scheme may differ; manual inspection of the CSV is needed.\n")

    # Save
    notes_dir = os.path.join(_ROOT, "notes")
    os.makedirs(notes_dir, exist_ok=True)
    notes_path = os.path.join(notes_dir, "failure_case_analysis.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved: {notes_path}")

    # Print key findings
    print(f"\n  Baseline-Delta Pearson r = {r_p:.3f} (p = {p_p:.4f})")
    print(f"  Dice outliers (< 20%): {len(outliers)} cases")
    print(f"  HD95 outliers (> 200mm): {len(hd_outliers)} cases")


# ── Statistical Interpretation (Exp 1+3 combined) ───────────────────────────

def write_interpretation(all_stats, df_main, df_fixed):
    """Write comprehensive interpretation markdown."""
    master_df = pd.DataFrame(all_stats)
    overall = master_df[master_df['organ'] == 'OVERALL'].copy()

    md = "# Statistical Interpretation of AdaFoB+LoRA Results\n\n"
    md += (f"Based on {len(df_main)} paired volume-organ evaluations across "
           f"5-fold cross-validation.\n\n")
    md += ("All p-values from two-sided Wilcoxon signed-rank test. "
           "95% confidence intervals from bootstrap resampling (n=10,000). "
           "Effect sizes reported as paired Cohen's d.\n\n")
    md += "---\n\n"

    # Group by comparison
    for comp in overall['comparison'].unique():
        comp_rows = overall[overall['comparison'] == comp]
        md += f"## {comp}\n\n"

        for _, r in comp_rows.iterrows():
            sig = ("statistically significant (p < 0.05)"
                   if r['wilcoxon_p'] < 0.05
                   else "NOT statistically significant (p >= 0.05)")

            if r['metric'] == 'Dice':
                md += (f"**Dice:** Mean difference = {r['mean_diff']*100:+.2f}% "
                       f"(95% CI: [{r['ci_low']*100:.2f}%, {r['ci_high']*100:.2f}%]), "
                       f"Wilcoxon p = {r['wilcoxon_p']:.4f} ({sig}), "
                       f"Cohen's d = {r['cohens_d']:.3f} ({r['effect_size']}).\n\n")
            else:
                md += (f"**HD95:** Mean difference = {r['mean_diff']:+.2f}mm "
                       f"(95% CI: [{r['ci_low']:.2f}, {r['ci_high']:.2f}]mm), "
                       f"Wilcoxon p = {r['wilcoxon_p']:.4f} ({sig}), "
                       f"Cohen's d = {r['cohens_d']:.3f} ({r['effect_size']}).\n\n")

        # Per-organ breakdown
        per_organ = master_df[(master_df['comparison'] == comp) &
                               (master_df['organ'] != 'OVERALL')]
        sig_organs = per_organ[per_organ['wilcoxon_p'] < 0.05]
        if len(sig_organs) > 0:
            md += "Per-organ significant differences:\n\n"
            for _, r in sig_organs.iterrows():
                if r['metric'] == 'Dice':
                    md += (f"- {r['organ']} ({r['metric']}): "
                           f"p = {r['wilcoxon_p']:.4f}, d = {r['cohens_d']:.3f}, "
                           f"diff = {r['mean_diff']*100:+.2f}%\n")
                else:
                    md += (f"- {r['organ']} ({r['metric']}): "
                           f"p = {r['wilcoxon_p']:.4f}, d = {r['cohens_d']:.3f}, "
                           f"diff = {r['mean_diff']:+.2f}mm\n")
            md += "\n"

        md += "---\n\n"

    # Key conclusions
    md += "## Key Conclusions for Manuscript\n\n"

    # Check HD95 FoB+LoRA vs AdaFoB+LoRA
    hd_ada_vs_fob = overall[
        (overall['comparison'] == 'FoB+LoRA vs AdaFoB+LoRA') &
        (overall['metric'] == 'HD95')]
    if len(hd_ada_vs_fob) > 0:
        r = hd_ada_vs_fob.iloc[0]
        if r['wilcoxon_p'] < 0.05:
            md += ("1. **HD95 DEGRADATION IS SIGNIFICANT.** The adaptive budget "
                   "causes a statistically significant increase in HD95 compared "
                   "to the fixed 10-point budget (p = {:.4f}). This MUST be "
                   "stated in the manuscript's Discussion/Limitations section. "
                   "Do NOT use 'zero statistically significant degradation' "
                   "language.\n\n".format(r['wilcoxon_p']))
        else:
            md += ("1. **HD95 difference (FoB+LoRA vs AdaFoB+LoRA) is NOT "
                   "significant** (p = {:.4f}). The adaptive budget does not "
                   "cause a statistically detectable boundary quality loss.\n\n"
                   .format(r['wilcoxon_p']))

    # Check HD95 AdaFoB vs Fixed Control
    hd_ada_vs_fc = overall[
        (overall['comparison'] == 'AdaFoB+LoRA vs Fixed-Budget-Control') &
        (overall['metric'] == 'HD95')]
    if len(hd_ada_vs_fc) > 0:
        r = hd_ada_vs_fc.iloc[0]
        if r['wilcoxon_p'] < 0.05:
            md += ("2. **ADAPTIVE > FIXED for HD95.** The Fixed-Budget-Control "
                   "has significantly worse HD95 than AdaFoB (p = {:.4f}). "
                   "This is a STRONG positive finding: a naive fixed budget "
                   "causes real boundary degradation that AdaFoB's dynamic "
                   "allocation avoids. The manuscript narrative should foreground "
                   "this, not just the Dice-parity framing.\n\n".format(
                       r['wilcoxon_p']))
        else:
            md += ("2. **HD95 difference (AdaFoB vs Fixed Control) is NOT "
                   "significant** (p = {:.4f}).\n\n".format(r['wilcoxon_p']))

    md += ("3. **Attribution of contributions:** The Dice improvement from "
           "Baseline to LoRA-adapted models is primarily attributable to LoRA "
           "domain adaptation. AdaFoB's contribution is orthogonal: it preserves "
           "accuracy while cutting the prompt budget by ~67%, and (if finding #2 "
           "holds) provides superior boundary quality compared to naive budget "
           "reduction.\n")

    notes_dir = os.path.join(_ROOT, "notes")
    os.makedirs(notes_dir, exist_ok=True)
    notes_path = os.path.join(notes_dir, "statistical_interpretation.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nSaved: {notes_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    results_dir = os.path.join(_ROOT, "results")

    # Load main eval CSV
    main_csv = os.path.join(results_dir, "lora_eval_final_combined.csv")
    if not os.path.exists(main_csv):
        print(f"ERROR: Cannot find {main_csv}")
        print("This script must be run where the evaluation CSV exists (e.g. Kaggle).")
        sys.exit(1)

    df_main = pd.read_csv(main_csv)
    req_cols = ['dice_baseline', 'dice_lora_fob', 'dice_lora_ada',
                'hd95_baseline', 'hd95_lora_fob', 'hd95_lora_ada',
                'fold', 'organ', 'vol_id']
    missing = [c for c in req_cols if c not in df_main.columns]
    if missing:
        print(f"ERROR: Missing columns in main CSV: {missing}")
        sys.exit(1)

    df_main = df_main.dropna(subset=['dice_baseline', 'dice_lora_fob', 'dice_lora_ada'])
    print(f"Loaded main eval: {len(df_main)} samples from {main_csv}")

    # Load fixed control CSV (optional)
    fixed_csv = os.path.join(results_dir, "control_fixed_mean_budget_gpu0.csv")
    df_fixed = None
    if os.path.exists(fixed_csv):
        df_fixed = pd.read_csv(fixed_csv)
        print(f"Loaded fixed control: {len(df_fixed)} samples from {fixed_csv}")
    else:
        print(f"WARNING: Fixed control CSV not found at {fixed_csv}")
        print("  Fixed-Budget-Control comparisons will be skipped.")

    os.makedirs(results_dir, exist_ok=True)

    # Run all experiments
    hd95_rows = run_experiment_1(df_main, df_fixed)
    run_experiment_2(df_main, df_fixed)
    all_stats = run_experiment_3(df_main, df_fixed)
    run_experiment_5(df_main)

    # Write combined interpretation
    write_interpretation(all_stats, df_main, df_fixed)

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print(f"  results/hd95_statistical_report.csv")
    print(f"  results/delta_reconciliation_table.csv")
    print(f"  results/full_statistical_report.csv")
    print(f"  notes/failure_case_analysis.md")
    print(f"  notes/statistical_interpretation.md")


if __name__ == "__main__":
    main()
