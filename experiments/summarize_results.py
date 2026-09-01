import os
import glob
import pandas as pd
import numpy as np

def main():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(repo_dir, "results")
    
    csv_files = glob.glob(os.path.join(results_dir, "lora_eval_fold*.csv"))
    if not csv_files:
        print("No evaluation CSVs found in results/ directory!")
        return

    print(f"Found {len(csv_files)} evaluation files. Aggregating...")
    
    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f))
    
    df_all = pd.concat(dfs, ignore_index=True)

    # ── Data completeness check ──────────────────────────────────────────
    df_all = df_all.drop_duplicates(subset=['fold', 'organ', 'vol_id'], keep='last')
    expected_organs = {'SPLEEN', 'RK', 'LK', 'LIVER'}
    expected_folds = {0, 1, 2, 3, 4}
    actual_combos = set(zip(df_all['fold'], df_all['organ']))
    missing = []
    for fold in expected_folds:
        for organ in expected_organs:
            if (fold, organ) not in actual_combos:
                missing.append(f"fold={fold}, organ={organ}")
    if missing:
        print("\n" + "!"*80)
        print("WARNING: INCOMPLETE DATA DETECTED")
        print("The following fold-organ combinations are MISSING:")
        for m in missing:
            print(f"  - {m}")
        print(f"Total samples: {len(df_all)} (expected 120 = 5 folds x 4 organs x 6 vols)")
        print("Results below are BIASED. Do NOT use for final reporting.")
        print("!"*80 + "\n")

    cross = df_all.groupby(['fold', 'organ']).size().unstack(fill_value=0)
    print(f"\nFold-Organ sample counts:\n{cross}\n")

    organs = df_all['organ'].unique()
    
    metrics = ['dice_baseline', 'hd95_baseline', 'dice_adafob_2d', 'hd95_adafob_2d', 
               'dice_lora_fob', 'hd95_lora_fob', 'dice_lora_ada', 'hd95_lora_ada']
    
    for m in metrics:
        if m not in df_all.columns:
            df_all[m] = np.nan

    print("\n" + "="*80)
    print("FINAL 5-FOLD CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    overall_means = {m: [] for m in metrics}
    
    for organ in sorted(organs):
        df_organ = df_all[df_all['organ'] == organ]
        print(f"\n--- {organ} (n={len(df_organ)} volumes) ---")
        
        d_base = df_organ['dice_baseline'].mean() * 100
        h_base = df_organ['hd95_baseline'].mean()
        d_ada2d = df_organ['dice_adafob_2d'].mean() * 100
        h_ada2d = df_organ['hd95_adafob_2d'].mean()
        
        d_lora = df_organ['dice_lora_fob'].mean() * 100
        h_lora = df_organ['hd95_lora_fob'].mean()
        d_ada_lora = df_organ['dice_lora_ada'].mean() * 100
        h_ada_lora = df_organ['hd95_lora_ada'].mean()
        
        print(f"  FoB Baseline (ViT-H, Np=10):  Dice={d_base:.2f}%  HD95={h_base:.1f}mm")
        print(f"  AdaFoB 2D (ViT-H):            Dice={d_ada2d:.2f}%  HD95={h_ada2d:.1f}mm")
        print(f"  FoB + LoRA (ViT-B+LoRA):      Dice={d_lora:.2f}%  HD95={h_lora:.1f}mm")
        print(f"  AdaFoB + LoRA (OURS):          Dice={d_ada_lora:.2f}%  HD95={h_ada_lora:.1f}mm")
        
        overall_means['dice_baseline'].append(d_base)
        overall_means['dice_adafob_2d'].append(d_ada2d)
        overall_means['dice_lora_fob'].append(d_lora)
        overall_means['dice_lora_ada'].append(d_ada_lora)

    print("\n" + "="*80)
    print("OVERALL MEAN (Macro-average across organs)")
    print("  FoB Baseline:       {:.2f}%".format(np.mean(overall_means['dice_baseline'])))
    print("  AdaFoB 2D:          {:.2f}%".format(np.mean(overall_means['dice_adafob_2d'])))
    print("  FoB + LoRA:         {:.2f}%".format(np.mean(overall_means['dice_lora_fob'])))
    print("  AdaFoB + LoRA:      {:.2f}%".format(np.mean(overall_means['dice_lora_ada'])))
    print("="*80)
    
    out_path = os.path.join(results_dir, "lora_eval_final_combined.csv")
    df_all.to_csv(out_path, index=False)
    print(f"\nSaved combined raw results to: {out_path}")

if __name__ == "__main__":
    main()
