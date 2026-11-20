"""Summarize executed saved-mask audit; preserve all historical sources."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.patient_balanced_summary import validate,summarize,patient_means,paired_bootstrap

def table(frame):
    return frame.to_markdown(index=False,floatfmt=".6f")

def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args()
 out=a.run_dir/"corrected_analysis";out.mkdir(exist_ok=False)
 df=pd.read_csv(a.run_dir/"metrics_full/corrected_measurements.csv")
 assert len(df)==1200 and not df.duplicated(["episode_id","arm","post"]).any()
 assert df.groupby(["arm","post"]).size().eq(120).all()
 assert (df[["detached_fp","other_annotated_organ_fp","residual_fp"]].sum(axis=1)==df.fp).all()
 manifest=pd.read_csv("protocol/paired_episode_manifest.csv")
 source=pd.read_csv("results/step4_baselines/step4_baselines_results.csv")
 wide=source[["episode_id","fold","organ","patient_id","vol_id","spacing_x","spacing_y","spacing_z"]].copy().set_index("episode_id")
 for (arm,post),g in df.groupby(["arm","post"]):
  wide[f"dice_{arm}_{post}"]=g.set_index("episode_id").dice
  wide[f"hd95_{arm}_{post}"]=g.set_index("episode_id").production_hd95
 wide=wide.reset_index()
 cols=[c for c in wide if c.startswith(("dice_","hd95_"))]
 wide=validate(wide,manifest,cols)
 wide.to_csv(out/"corrected_primary_metrics_wide.csv",index=False)
 summarize(wide,cols).to_csv(out/"step4_both_estimands.csv",index=False)
 patient_means(wide,cols).to_csv(out/"step4_patient_means.csv")
 paired=[paired_bootstrap(wide,"dice_b3b_cc","dice_b0_cc",5000,42)]
 (out/"step4_paired_exploratory.json").write_text(json.dumps(paired,indent=2))
 # Step5 CSV-only reanalysis is not a mask-level independent verification.
 s5=pd.read_csv("results/step5_diagnostics/step5_diagnostics_results.csv")
 s5cols=[c for c in s5 if c.startswith(("dice_","hd95_")) and "_std_" not in c]
 s5=validate(s5,manifest,s5cols)
 summarize(s5,s5cols).to_csv(out/"step5_csv_only_both_estimands.csv",index=False)
 patient_means(s5,s5cols).to_csv(out/"step5_csv_only_patient_means.csv")
 contrasts=[paired_bootstrap(s5,a,b,5000,42) for a,b in [("dice_greedy_oracle_cc","dice_k10_cc"),("dice_prefix_oracle_cc","dice_k10_cc")]]
 (out/"step5_csv_only_paired_exploratory.json").write_text(json.dumps({"status":"UNVERIFIED saved masks except k0 and k10; no new selection inference","contrasts":contrasts},indent=2))
 # Reconcile raw -> CC counts independently of the disjoint residual partition.
 reconciliation=[]
 for (eid,arm),g in df.groupby(["episode_id","arm"]):
  raw=g[g.post=="raw"].iloc[0];cc=g[g.post=="cc"].iloc[0]
  assert raw.fp==cc.fp+cc.removed_fp and raw.tp==cc.tp+cc.removed_tp
  reconciliation.append({"episode_id":eid,"arm":arm,"removed_tp":cc.removed_tp,"removed_fp":cc.removed_fp,"raw_fp":raw.fp,"cc_fp":cc.fp,
                         "cc_detached_fp":cc.detached_fp,"cc_other_annotated_organ_fp":cc.other_annotated_organ_fp,"cc_residual_fp":cc.residual_fp})
 pd.DataFrame(reconciliation).to_csv(out/"raw_to_cc_reconciliation.csv",index=False)
 changes=[];consistency=[]
 for arm in ["b0","b3b"]:
  old=pd.read_csv(f"results/step5_diagnostics/error_decomposition_{arm}.csv").set_index("episode_id")
  new=df[(df.arm==arm)&(df.post=="cc")].set_index("episode_id")
  for legacy,newcol in [("h95_pred_to_gt","legacy_surface_to_fg_pred95"),("h95_gt_to_pred","legacy_surface_to_fg_gt95"),
                        ("tp_voxels","tp"),("fp_voxels_cc","fp"),("fn_voxels_cc","fn")]:
   delta=new[newcol]-old[legacy]
   assert np.nanmax(np.abs(delta))<1e-6
   consistency.append({"arm":arm,"legacy_field":legacy,"new_reconstruction":newcol,"max_abs_delta":float(np.nanmax(np.abs(delta)))})
  mappings=[("detached_islands_voxels","detached_fp"),("wrong_organ_voxels","other_annotated_organ_fp"),
            ("boundary_leakage_voxels","residual_fp"),("h95_pred_to_gt","directed_surface_pred_to_gt95"),("h95_gt_to_pred","directed_surface_gt_to_pred95")]
  for oldcol,newcol in mappings:
   for eid,row in new.iterrows():
    changes.append({"episode_id":eid,"arm":arm,"legacy_field":oldcol,"replacement_field":newcol,"old_value":old.loc[eid,oldcol],"new_value":row[newcol],"change":row[newcol]-old.loc[eid,oldcol]})
 change=pd.DataFrame(changes);change.to_csv(out/"historical_field_replacements.csv",index=False)
 pd.DataFrame(consistency).to_csv(out/"legacy_operator_reconstruction_checks.csv",index=False)
 cols=["tp","fp","fn","removed_tp","removed_fp","detached_fp","other_annotated_organ_fp","residual_fp",
       "legacy_surface_to_fg_pred95","legacy_surface_to_fg_gt95","directed_surface_pred_to_gt95","directed_surface_gt_to_pred95","surface_hd95","recall","fp_mm3","fn_mm3"]
 measures=[]
 for (arm,post),g in df.groupby(["arm","post"]):
  patient=g.groupby(["patient_id","organ"])[cols].agg(lambda x:np.mean(x.to_numpy())).groupby("patient_id").agg(lambda x:np.mean(x.to_numpy()))
  for c in cols:
   values=g[c].to_numpy()
   measures.append({"arm":arm,"post":post,"measure":c,"episode_weighted":float(np.mean(values)),"patient_balanced":float(np.mean(patient[c].to_numpy())),"median":float(np.median(values)),"p90":float(np.quantile(values,.9)),"nonfinite":int((~np.isfinite(values)).sum())})
 pd.DataFrame(measures).to_csv(out/"decomposition_both_estimands.csv",index=False)
 flags={"primary_rows":1200,"raw_CC_count_assertions":600,"disjoint_partition_assertions":1200,
        "dice_max_csv_delta":float(df.dice_minus_source.abs().max()),"production_hd95_max_csv_delta":float(df.hd95_minus_source.abs().max()),
        "empty_prediction_rows":int(df.empty_prediction.sum()),"nonfinite_independent_surface_rows":int((~np.isfinite(df.surface_hd95)).sum()),
        "full_miss_rows":int((df.fn==df.gt_voxels).sum()),"step5_independent_mask_verification":"k0/k10 only",
        "replaced_fields_with_changed_rows":change.groupby(["arm","legacy_field"]).apply(lambda g:int((g.change.abs()>1e-9).sum()),include_groups=False).to_dict().__str__(),
        "uncertainty":"exploratory pointwise paired query-patient bootstrap of fixed predictions; conditions on support selection and checkpoints; no support/training uncertainty; no multiplicity-adjusted confirmation claims"}
 (out/"summary.json").write_text(json.dumps(flags,indent=2))
 print(json.dumps(flags),flush=True)
if __name__=="__main__":main()
