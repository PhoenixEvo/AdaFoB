"""
Failure Case Analysis for AdaFoB + LoRA
========================================
Analyzes RK plateau, LK outlier (image_7), and HD95 regressions
from lora_eval_final_combined.csv.

Output: notes/failure_case_analysis.md
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    csv_path = os.path.join(_ROOT, "results", "lora_eval_final_combined.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: Cannot find {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['dice_baseline', 'dice_lora_fob', 'dice_lora_ada'])
    print(f"Loaded {len(df)} samples")

    # Compute deltas
    df['dice_delta_lora'] = df['dice_lora_fob'] - df['dice_baseline']
    df['dice_delta_ada'] = df['dice_lora_ada'] - df['dice_baseline']
    df['hd95_delta_lora'] = df['hd95_lora_fob'] - df['hd95_baseline']
    df['hd95_delta_ada'] = df['hd95_lora_ada'] - df['hd95_baseline']

    md = "# Failure Case Analysis\n\n"
    md += f"Analysis of {len(df)} volume-organ evaluations from 5-fold cross-validation.\n\n"

    # ── Section 1: RK Plateau Analysis ──────────────────────────────────────
    md += "## 1. Right Kidney (RK) — Low Headroom Hypothesis\n\n"
    md += "**Observation:** RK shows minimal Dice improvement after LoRA adaptation "
    md += "(85.62% baseline -> 84.92% AdaFoB+LoRA), while LIVER improves dramatically "
    md += "(87.08% -> 94.18%).\n\n"

    md += "**Hypothesis:** Organs with high baseline Dice leave less room for improvement.\n\n"

    # Per-organ baseline stats
    md += "### Baseline Dice Distribution by Organ\n\n"
    md += "| Organ | Mean Baseline | Std | Min | Max | Mean Delta (LoRA) | Mean Delta (AdaFoB) |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for organ in sorted(df['organ'].unique()):
        o = df[df['organ'] == organ]
        md += f"| {organ} | {o['dice_baseline'].mean()*100:.2f}% | "
        md += f"{o['dice_baseline'].std()*100:.2f}% | "
        md += f"{o['dice_baseline'].min()*100:.2f}% | "
        md += f"{o['dice_baseline'].max()*100:.2f}% | "
        md += f"{o['dice_delta_lora'].mean()*100:+.2f}% | "
        md += f"{o['dice_delta_ada'].mean()*100:+.2f}% |\n"

    # Correlation: baseline Dice vs delta
    r_pearson, p_pearson = pearsonr(df['dice_baseline'], df['dice_delta_lora'])
    r_spearman, p_spearman = spearmanr(df['dice_baseline'], df['dice_delta_lora'])

    md += f"\n**Correlation between baseline Dice and LoRA improvement (delta):**\n"
    md += f"- Pearson r = {r_pearson:.3f} (p = {p_pearson:.4f})\n"
    md += f"- Spearman rho = {r_spearman:.3f} (p = {p_spearman:.4f})\n\n"

    if r_pearson < -0.2 and p_pearson < 0.05:
        md += "This is consistent with the low-headroom hypothesis: cases with higher "
        md += "baseline Dice tend to show smaller improvements from LoRA, suggesting "
        md += "diminishing returns when the frozen model already performs well.\n\n"
    elif abs(r_pearson) < 0.2:
        md += "The correlation is weak, suggesting that baseline Dice alone does not "
        md += "fully explain the variable improvement across organs. Other factors "
        md += "(organ morphology, boundary contrast, typical HU similarity with "
        md += "adjacent tissues) likely contribute.\n\n"
    else:
        md += "The correlation is positive, which contradicts the simple headroom "
        md += "hypothesis. An alternative explanation may be needed.\n\n"

    # ── Section 2: Outlier Detection ────────────────────────────────────────
    md += "## 2. Outlier Detection\n\n"

    # Dice outliers (< 20% in any method)
    dice_cols = ['dice_baseline', 'dice_lora_fob', 'dice_lora_ada']
    outlier_mask = (df[dice_cols] < 0.20).any(axis=1)
    outliers = df[outlier_mask]

    md += f"**Cases with Dice < 20% in any method:** {len(outliers)} cases\n\n"
    if len(outliers) > 0:
        md += "| Fold | Organ | Vol ID | Baseline | FoB+LoRA | AdaFoB+LoRA | HD95 Base | HD95 LoRA |\n"
        md += "|---|---|---|---|---|---|---|---|\n"
        for _, r in outliers.iterrows():
            vol_short = os.path.basename(str(r['vol_id'])) if 'vol_id' in r else 'N/A'
            md += f"| {r['fold']} | {r['organ']} | {vol_short} | "
            md += f"{r['dice_baseline']*100:.1f}% | {r['dice_lora_fob']*100:.1f}% | "
            md += f"{r['dice_lora_ada']*100:.1f}% | "
            md += f"{r['hd95_baseline']:.1f}mm | {r['hd95_lora_fob']:.1f}mm |\n"
        md += "\n"

    # HD95 outliers (> 200mm in any LoRA method)
    hd_outlier_mask = (df['hd95_lora_fob'] > 200) | (df['hd95_lora_ada'] > 200)
    hd_outliers = df[hd_outlier_mask]

    md += f"**Cases with HD95 > 200mm after LoRA:** {len(hd_outliers)} cases\n\n"
    if len(hd_outliers) > 0:
        md += "| Fold | Organ | Vol ID | HD95 Base | HD95 FoB+LoRA | HD95 AdaFoB |\n"
        md += "|---|---|---|---|---|---|\n"
        for _, r in hd_outliers.iterrows():
            vol_short = os.path.basename(str(r['vol_id'])) if 'vol_id' in r else 'N/A'
            md += f"| {r['fold']} | {r['organ']} | {vol_short} | "
            md += f"{r['hd95_baseline']:.1f}mm | {r['hd95_lora_fob']:.1f}mm | "
            md += f"{r['hd95_lora_ada']:.1f}mm |\n"
        md += "\n"

    # ── Section 3: HD95 Regression Analysis ─────────────────────────────────
    md += "## 3. HD95 Regression After LoRA\n\n"
    md += "**Observation:** HD95 worsens for LK (34.7 -> 37.7mm) and RK (32.7 -> 39.7mm) "
    md += "after LoRA, while improving for LIVER (30.1 -> 11.8mm) and SPLEEN (33.3 -> 49.8mm*).\n\n"
    md += "*Note: SPLEEN HD95 appears to worsen in the aggregate, warranting case-level inspection.\n\n"

    md += "### Per-Organ HD95 Summary\n\n"
    md += "| Organ | HD95 Baseline | HD95 FoB+LoRA | HD95 AdaFoB | Delta (LoRA) |\n"
    md += "|---|---|---|---|---|\n"
    for organ in sorted(df['organ'].unique()):
        o = df[df['organ'] == organ]
        md += f"| {organ} | {o['hd95_baseline'].mean():.1f}mm | "
        md += f"{o['hd95_lora_fob'].mean():.1f}mm | "
        md += f"{o['hd95_lora_ada'].mean():.1f}mm | "
        md += f"{o['hd95_delta_lora'].mean():+.1f}mm |\n"

    md += "\n### Cases Where HD95 Worsened Most (LoRA vs Baseline)\n\n"
    worst_hd = df.nlargest(10, 'hd95_delta_lora')
    md += "| Fold | Organ | Vol ID | HD95 Base | HD95 LoRA | Delta |\n"
    md += "|---|---|---|---|---|---|\n"
    for _, r in worst_hd.iterrows():
        vol_short = os.path.basename(str(r['vol_id'])) if 'vol_id' in r else 'N/A'
        md += f"| {r['fold']} | {r['organ']} | {vol_short} | "
        md += f"{r['hd95_baseline']:.1f}mm | {r['hd95_lora_fob']:.1f}mm | "
        md += f"{r['hd95_delta_lora']:+.1f}mm |\n"

    # ── Section 4: image_7 Specific Analysis ────────────────────────────────
    md += "\n## 4. image_7 (LK) Specific Analysis\n\n"
    img7_rows = df[df['vol_id'].str.contains('image_7', na=False)]
    if len(img7_rows) == 0:
        # Try matching by pattern
        img7_rows = df[df['vol_id'].str.contains('_7', na=False)]

    if len(img7_rows) > 0:
        md += "| Fold | Organ | Dice Base | Dice LoRA | Dice Ada | HD95 Base | HD95 LoRA | HD95 Ada |\n"
        md += "|---|---|---|---|---|---|---|---|\n"
        for _, r in img7_rows.iterrows():
            md += f"| {r['fold']} | {r['organ']} | "
            md += f"{r['dice_baseline']*100:.1f}% | {r['dice_lora_fob']*100:.1f}% | "
            md += f"{r['dice_lora_ada']*100:.1f}% | "
            md += f"{r['hd95_baseline']:.1f}mm | {r['hd95_lora_fob']:.1f}mm | "
            md += f"{r['hd95_lora_ada']:.1f}mm |\n"
        md += "\n"
        md += "**Note:** To complete this analysis, a visual overlay of the predicted vs "
        md += "ground-truth mask for image_7 (LK) should be generated on Kaggle using the "
        md += "visualization script below.\n"
    else:
        md += "image_7 was not found in the vol_id column. Manual inspection of the CSV "
        md += "is needed to identify the correct volume identifier.\n"

    md += "\n## 5. Visualization Script (Run on Kaggle)\n\n"
    md += "```python\n"
    md += "# Generate overlay PNG for failure cases\n"
    md += "# Add this as a cell in the Kaggle notebook after evaluation\n"
    md += "import matplotlib.pyplot as plt\n"
    md += "import nibabel as nib\n"
    md += "# Load image_7 volume and its predictions\n"
    md += "# [Implementation depends on which slice shows the failure]\n"
    md += "```\n"

    # Save
    notes_dir = os.path.join(_ROOT, "notes")
    os.makedirs(notes_dir, exist_ok=True)
    notes_path = os.path.join(notes_dir, "failure_case_analysis.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved failure case analysis to {notes_path}")

    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    print(f"Baseline-Delta correlation: r={r_pearson:.3f}, p={p_pearson:.4f}")
    print(f"Dice outliers (< 20%): {len(outliers)} cases")
    print(f"HD95 outliers (> 200mm): {len(hd_outliers)} cases")


if __name__ == "__main__":
    main()
