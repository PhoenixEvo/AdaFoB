"""Three-repeat T4 timing on eight locked slices, shared versus separate passes."""
from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path
import numpy as np,pandas as pd
from .common import *

def separate(model,predictor,case):
    records={};arrays={};logits={}
    start=time.perf_counter()
    for arm in ARMS:
        (neg,pos),ft=timed(lambda:generate(model,case))
        configs,_,_=configurations(neg,pos);n,p=configs[arm]
        rgb,it=timed(lambda:rgb_image(case))
        _,et=timed(lambda:predictor.set_image(rgb))
        (raw,log,q,calls),dt=timed(lambda:predict(predictor,n,p))
        cc,ct=timed(lambda:cc_mask(raw,p))
        arrays[arm+"_raw"]=raw;arrays[arm+"_cc"]=cc;logits[arm]=(log,q)
        records[arm]={"fob_seconds":ft,"image_conversion_seconds":it,"encoder_api_seconds":et,
                      "decoder_seconds":dt,"cc_seconds":ct,"decoder_calls":calls}
    return arrays,{"arms":records,"total_seconds":time.perf_counter()-start},logits

def main():
    p=argparse.ArgumentParser();p.add_argument("--lock-dir",type=Path,required=True);p.add_argument("--data-dir",required=True)
    p.add_argument("--checkpoint-root",required=True);p.add_argument("--sam-checkpoint",required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=read_config(a.lock_dir);import torch
    seed_all(c["seed"]);assert "T4" in torch.cuda.get_device_name(0)
    rows=pd.read_csv(a.lock_dir/"diagnostic_manifest.csv");manifest=read_manifest(a.lock_dir/"paired_episode_manifest.csv")
    predictor,sm=tracked_sam(a.sam_checkpoint)
    records=[];differences=[];loads=[];calls=0;encoders=0;start=time.perf_counter()
    for fold in sorted(rows.fold.unique()):
        model,meta=load_models(c,a.checkpoint_root,a.sam_checkpoint,int(fold));loads.append(meta)
        for diag in rows[rows.fold==fold].to_dict("records"):
            row=manifest[manifest.episode_id==diag["episode_id"]].iloc[0].to_dict()
            case=next(x for x in episode_inputs(row,a.data_dir,c) if x["native_z"]==diag["native_z"])
            assert case["support_native_z"]==diag["support_native_z"] and case["support_part"]==diag["support_part"]
            for repeat in range(c["pilot_repeats"]+1):
                torch.cuda.reset_peak_memory_stats()
                shared,record,sl=run_shared(model,predictor,case,capture_logits=True)
                assert not record["failures"]
                calls+=record["decoder_calls"];encoders+=1
                peak_shared=torch.cuda.max_memory_allocated()
                serialize_start=time.perf_counter()
                payload=serialized_slice(shared,record)
                serialize_seconds=time.perf_counter()-serialize_start
                sep,sr,pl=separate(model,predictor,case);calls+=sum(v["decoder_calls"] for v in sr["arms"].values());encoders+=3
                max_delta=0.
                for arm in ARMS:
                    for post in ["raw","cc"]:assert np.array_equal(shared[arm+"_"+post],sep[arm+"_"+post]),"cache changes mask"
                    for x,y in zip(sl[arm],pl[arm]):
                        delta=float(np.max(np.abs(x-y)));max_delta=max(max_delta,delta)
                        np.testing.assert_allclose(x,y,atol=c["parity_atol"],rtol=0)
                if repeat:
                    rec={"episode_id":case["episode_id"],"repeat":repeat,"shared_seconds":record["runtime"]["slice_compute_seconds"],
                         "separate_seconds":sr["total_seconds"],"fob_seconds":record["runtime"]["fob_seconds"],
                         "encoder_api_seconds":record["runtime"]["encoder_api_seconds"],
                         "decoder_seconds_per_call":sum(v["decoder_seconds"] for v in record["arms"].values())/3,
                         "cc_seconds_per_arm":sum(v["cc_seconds"] for v in record["arms"].values())/3,
                         "peak_allocated_bytes":peak_shared,"serialized_bytes":len(payload),"max_logit_quality_delta":max_delta}
                    t=time.perf_counter()
                    with (a.output/f"ep{case['episode_id']:03d}_repeat{repeat}.npz").open("xb") as f:f.write(payload)
                    rec["serialize_write_seconds"]=serialize_seconds+time.perf_counter()-t
                    records.append(rec)
                print(f"pilot ep={case['episode_id']} repeat={repeat} shared={record['runtime']['slice_compute_seconds']:.4f}s separate={sr['total_seconds']:.4f}s",flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    df=pd.DataFrame(records);df.to_csv(a.output/"timing_repeats.csv",index=False)
    conservative_slice=float((df.shared_seconds+df.serialize_write_seconds).quantile(.9))
    # Two allocated T4s counted for the entire notebook wall time. Balanced episodes,
    # sequential folds per worker; 25% overhead plus 180s preflight/loading allowance.
    projected_wall=conservative_slice*c["total_slices"]/2*c["projection_overhead_factor"]+c["projection_startup_seconds"]
    result={"hardware":torch.cuda.get_device_name(0),"timing_gpu":0,"warmup_per_slice":1,"repeats":3,"unique_slices":len(rows),
            "decoder_calls":calls,"encoder_calls":encoders,"all_shared_separate_masks_equal":True,
            "max_logit_quality_difference":float(df.max_logit_quality_delta.max()),
            "mean_shared_seconds":float(df.shared_seconds.mean()),"sd_shared_seconds":float(df.shared_seconds.std()),
            "mean_separate_seconds":float(df.separate_seconds.mean()),"sd_separate_seconds":float(df.separate_seconds.std()),
            "p90_shared_plus_write_seconds":conservative_slice,"mean_storage_bytes_per_slice":float(df.serialized_bytes.mean()),
            "estimated_slice_archive_bytes":float(df.serialized_bytes.mean()*c["total_slices"]),
            "uncompressed_three_arms_raw_CC_bytes":c["total_slices"]*257*257*6,
            "peak_allocated_bytes":int(df.peak_allocated_bytes.max()),
            "projection_wall_seconds":projected_wall,"projection_two_T4_gpu_hours":2*projected_wall/3600,
            "pilot_stage_wall_seconds":time.perf_counter()-start,"loads":loads,"sam_load":sm}
    write_json(a.output/"pilot_summary.json",result);print(json.dumps(result),flush=True)
if __name__=="__main__":main()
