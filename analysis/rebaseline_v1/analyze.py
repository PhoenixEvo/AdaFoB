"""Saved-mask-only corrected rebaseline analysis, frozen before inference."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from analysis.gate_a.runtime import production_metrics,sha256
from analysis.gate_a.measurements import disjoint_fp,independent_surfaces
from analysis.patient_balanced_summary import validate,patient_means,paired_bootstrap
from .common import read_config,read_manifest,label_volume,nifti_path,write_json,ORGANS,restore_xy

ARMS=("R0","R1","R2","R3","B4B")
CONTRASTS=(("R1","R0"),("R2","R1"),("R2","R0"),("R3","B4B"))

def one_episode(task):
    row,root,data_dir,c=task
    root=Path(root);eid=int(row["episode_id"])
    candidates=list(root.glob(f"worker*/episode_{eid:03d}"))
    assert len(candidates)==1,f"episode location {eid}: {candidates}"
    folder=candidates[0];marker=json.loads((folder/"episode_complete.json").read_text())
    labels,spacing=label_volume(str(nifti_path(data_dir,row["patient_id"].replace("image_","label_"))))
    native_z=np.flatnonzero((labels==ORGANS[row["organ"]]).any(axis=(1,2)))
    np.testing.assert_allclose(spacing,[row["spacing_x"],row["spacing_y"],row["spacing_z"]],atol=1e-6,rtol=0)
    assert marker["slices"]==len(native_z)
    label=labels[native_z];gt=label==ORGANS[row["organ"]];other=(label>0)&~gt
    masks={arm:{post:[] for post in ["raw","cc"]} for arm in ["R1","R2","R3"]}
    slices=[]
    commits={int(p.name.split("_")[1].split(".")[0]):json.loads(p.read_text()) for p in folder.glob("slice_*.commit.json")}
    assert set(commits)==set(range(len(native_z)))
    for z in range(len(native_z)):
        commit=commits[z];p=folder/commit["file"];assert sha256(p)==commit["sha256"]
        with np.load(p,allow_pickle=False) as saved:
            record=json.loads(str(saved["record_json"]))
            assert record["episode_id"]==eid and record["native_z"]==int(native_z[z]) and record["compact_z"]==z
            assert record["seed"]==c["seed"] and record["source_grid_hw"]==[256,256]
            assert np.array_equal(saved["gt"],gt[z])
            np.testing.assert_array_equal(saved["negative_restored_xy"],restore_xy(saved["negative_source_xy"],(256,256),(257,257)))
            np.testing.assert_array_equal(saved["positive_restored_xy"],restore_xy(saved["positive_source_xy"],(256,256),(257,257)))
            assert record["generation_order_ids"]==list(range(len(saved["negative_source_xy"])))
            for arm in ["R1","R2","R3"]:
                for post in ["raw","cc"]:
                    arr=saved[arm+"_"+post]>0;assert arr.shape==gt[z].shape;masks[arm][post].append(arr.copy())
                event=record["arms"][arm]
                slices.append({"episode_id":eid,"fold":row["fold"],"patient_id":row["patient_id"],"organ":row["organ"],"arm":arm,
                               "compact_z":z,"native_z":int(native_z[z]),"support_native_z":record["support_native_z"],"seed":record["seed"],
                               "generation_order_ids":json.dumps(record["generation_order_ids"]),"selected_ids":json.dumps(event["selected_ids"]),
                               "positive_count":event["positive_count"],"negative_count":event["negative_count"],"total_count":event["total_count"],
                               "emitted_candidates":len(saved["negative_source_xy"]),"negative_source_xy":json.dumps(saved["negative_source_xy"].tolist()),
                               "negative_restored_xy":json.dumps(saved["negative_restored_xy"].tolist()),
                               "positive_source_xy":json.dumps(saved["positive_source_xy"].tolist()),
                               "positive_restored_xy":json.dumps(saved["positive_restored_xy"].tolist()),
                               "negative_oob":json.dumps(event["negative_oob"]),"positive_oob":json.dumps(event["positive_oob"]),
                               "decoder_calls":event["decoder_calls"],"failure":event["failure"],"slice_failures":json.dumps(record["failures"]),
                               "decoder_seconds":event["decoder_seconds"],"cc_seconds":event["cc_seconds"],**record["runtime"]})
    for arm in masks:
        for post in masks[arm]:masks[arm][post]=np.stack(masks[arm][post])
    legacy=Path("results/step4_baselines/predictions")/f"ep_{eid:03d}_{row['patient_id']}_{row['organ']}.npz"
    assert sha256(legacy)==c["legacy_prediction_sha256"][legacy.as_posix()]
    with np.load(legacy) as old:
        assert np.array_equal(old["gt"]>0,gt)
        for arm,key in [("R0","b3b"),("B4B","b4b")]:masks[arm]={post:old[key+"_"+post]>0 for post in ["raw","cc"]}
    metric,_=production_metrics();rows=[]
    for arm in ARMS:
        raw,cc=masks[arm]["raw"],masks[arm]["cc"]
        assert not (cc&~raw).any()
        removed_tp=int((raw&~cc&gt).sum());removed_fp=int((raw&~cc&~gt).sum())
        for post,pred in [("raw",raw),("cc",cc)]:
            dice,production_hd=metric(pred,gt,tuple(spacing))
            denominator=int(pred.sum()+gt.sum())
            refdice=2*int((pred&gt).sum())/denominator if denominator else 1.
            hd,dp,dg=independent_surfaces(pred,gt,spacing[::-1])
            np.testing.assert_allclose(dice,refdice,atol=c["dice_atol"],rtol=0)
            agreement=bool(np.isfinite(hd) and abs(hd-production_hd)<=c["hd95_atol_mm"])
            if np.isfinite(hd):np.testing.assert_allclose(hd,production_hd,atol=c["hd95_atol_mm"],rtol=0)
            partition=disjoint_fp(pred,gt,other)
            fp,fn,tp=int((pred&~gt).sum()),int((~pred&gt).sum()),int((pred&gt).sum())
            assert sum(partition.values())==fp
            rows.append({"episode_id":eid,"fold":int(row["fold"]),"patient_id":row["patient_id"],"organ":row["organ"],"arm":arm,"post":post,
                         "dice":dice,"hd95":hd,"production_hd95":production_hd,"hd95_difference":hd-production_hd,
                         "reference_dice_difference":refdice-dice,"metric_agreement":agreement,"empty_prediction":not pred.any(),
                         "tp":tp,"fp":fp,"fn":fn,"recall":tp/int(gt.sum()),"fp_mm3":fp*float(np.prod(spacing)),"fn_mm3":fn*float(np.prod(spacing)),
                         **partition,"removed_tp":removed_tp,"removed_fp":removed_fp,
                         "directed_pred_to_gt95":dp,"directed_gt_to_pred95":dg})
        assert rows[-2]["fp"]==rows[-1]["fp"]+removed_fp
        assert rows[-2]["tp"]==rows[-1]["tp"]+removed_tp
    return rows,slices

def summary(df):
    result=[]
    values=["dice","hd95","production_hd95","fp","fn","recall","fp_mm3","fn_mm3","detached_fp","other_annotated_organ_fp","residual_fp","removed_tp","removed_fp"]
    for organ in ["ALL","LIVER","LK","RK","SPLEEN"]:
        data=df if organ=="ALL" else df[df.organ==organ]
        for (arm,post),g in data.groupby(["arm","post"]):
            pt=g.groupby(["patient_id","organ"])[values].agg(lambda v:np.mean(v.to_numpy())).groupby("patient_id").agg(lambda v:np.mean(v.to_numpy()))
            for metric in values:
                x=g[metric].to_numpy()
                result.append({"organ":organ,"arm":arm,"post":post,"metric":metric,"episode_weighted":np.mean(x),
                               "patient_balanced":np.mean(pt[metric].to_numpy()),"median":np.median(x),"p90":np.quantile(x,.9),
                               "episodes":len(g),"patients":len(pt),"nonfinite":int((~np.isfinite(x)).sum())})
    return pd.DataFrame(result)

def main():
    p=argparse.ArgumentParser();p.add_argument("--lock-dir",type=Path,required=True);p.add_argument("--data-dir",required=True)
    p.add_argument("--inference-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=2)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);c=read_config(a.lock_dir)
    manifest=read_manifest(a.lock_dir/"paired_episode_manifest.csv");start=time.perf_counter();allrows=[];allslices=[]
    tasks=[(row,str(a.inference_root),a.data_dir,c) for row in manifest.to_dict("records")]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,(rows,slices) in enumerate(pool.map(one_episode,tasks)):
            allrows.extend(rows);allslices.extend(slices)
            print(f"metrics {i+1}/120 episodes elapsed={time.perf_counter()-start:.2f}s",flush=True)
    df=pd.DataFrame(allrows);slices=pd.DataFrame(allslices)
    assert len(df)==120*len(ARMS)*2 and not df.duplicated(["episode_id","arm","post"]).any()
    assert len(slices)==c["total_slices"]*3 and not slices.duplicated(["episode_id","compact_z","arm"]).any()
    assert df.groupby(["arm","post"]).size().eq(120).all()
    df.to_csv(a.output/"measurements.csv",index=False);slices.to_csv(a.output/"slice_records.csv",index=False)
    summary(df).to_csv(a.output/"both_estimands.csv",index=False)
    invalid=int((~df.metric_agreement).sum());failures=int(slices.failure.notna().sum())+int((slices.slice_failures!="[]").sum())
    contrasts=[]
    wide=manifest.copy().set_index("episode_id")
    for (arm,post),g in df.groupby(["arm","post"]):
        for metric in ["dice","hd95"]:wide[f"{metric}_{arm}_{post}"]=g.set_index("episode_id")[metric]
    wide=wide.reset_index()
    if not invalid:
        columns=[x for x in wide if x.startswith(("dice_","hd95_"))];wide=validate(wide,manifest,columns)
        for organ in ["ALL","LIVER","LK","RK","SPLEEN"]:
            data=wide if organ=="ALL" else wide[wide.organ==organ]
            for first,second in CONTRASTS:
                for post in ["raw","cc"]:
                    for metric in ["dice","hd95"]:
                        result=paired_bootstrap(data,f"{metric}_{first}_{post}",f"{metric}_{second}_{post}",c["bootstrap_replicates"],c["seed"])
                        contrasts.append({"organ":organ,"post":post,"metric":metric,"first_arm":first,"second_arm":second,**result})
        write_json(a.output/"paired_contrasts.json",contrasts)
    wide.to_csv(a.output/"metrics_wide.csv",index=False)
    primary=[v for v in contrasts if v["organ"]=="ALL" and v["post"]=="cc" and v["first_arm"]=="R2" and v["second_arm"]=="R0"]
    matches=not invalid and len(primary)==2
    interval_checks=[]
    for item in primary:
        margin=c["equivalence_style_margins"][item["metric"]]
        lo,hi=item["patient_balanced_ci95"];inside=bool(lo>-margin and hi<margin)
        matches=matches and inside
        interval_checks.append({"metric":item["metric"],"margin":margin,"ci95":[lo,hi],"inside":inside})
    status="PASS" if not invalid and not failures else "FAIL"
    decision={"corrected_baseline_gate":status,"decision_branch":"A" if matches else "B","legacy_matching_established":bool(matches),
              "margin_checks":interval_checks,"metric_disagreements":invalid,"slice_failures":failures,"measurements":len(df),
              "new_arm_slice_records":len(slices),"episodes":len(manifest),"patients":df.patient_id.nunique(),
              "elapsed_cpu_analysis_seconds":time.perf_counter()-start,
              "recommendation":"PROCEED to simple controls on the corrected pipeline" if status=="PASS" else "FIX again",
              "branch_B_interpretation":"A CI crossing a margin is failure to establish matching, not proof of material difference",
              "uncertainty":"Pointwise paired query-patient bootstrap, conditional on fixed supports/checkpoints; no training/support or simultaneous interval claim"}
    write_json(a.output/"decision.json",decision);print(json.dumps(decision),flush=True)
if __name__=="__main__":main()
