"""Resumable corrected three-arm full run. One embedding, three decoder calls."""
from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path
import numpy as np,pandas as pd
from .common import *

def load_completed(folder):
    done={}
    for path in folder.glob("slice_*.commit.json"):
        row=json.loads(path.read_text());p=folder/row["file"]
        assert sha256(p)==row["sha256"],f"invalid completed slice {p}"
        assert row["compact_z"] not in done
        done[row["compact_z"]]=row
    return done

def save_slice(folder,arrays,record):
    z=record["compact_z"];attempt=len(list(folder.glob(f"slice_{z:03d}_attempt*.npz")))
    name=f"slice_{z:03d}_attempt{attempt:03d}.npz";payload=serialized_slice(arrays,record)
    start=time.perf_counter()
    with (folder/name).open("xb") as f:f.write(payload)
    meta={"compact_z":z,"native_z":record["native_z"],"file":name,"sha256":sha256(folder/name),
          "bytes":len(payload),"write_seconds":time.perf_counter()-start,"episode_id":record["episode_id"]}
    write_json(folder/f"slice_{z:03d}.commit.json",meta)
    return meta

def main():
    p=argparse.ArgumentParser();p.add_argument("--lock-dir",type=Path,required=True);p.add_argument("--data-dir",required=True)
    p.add_argument("--checkpoint-root",required=True);p.add_argument("--sam-checkpoint",required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--worker",type=int,required=True);p.add_argument("--deadline-epoch",type=float,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);c=read_config(a.lock_dir)
    import torch
    seed_all(c["seed"]);assert "T4" in torch.cuda.get_device_name(0)
    start=time.perf_counter();manifest=read_manifest(a.lock_dir/"paired_episode_manifest.csv")
    assignment=pd.read_csv(a.lock_dir/"worker_assignment.csv")
    ids=assignment[assignment.worker==a.worker].episode_id.to_list()
    manifest=manifest[manifest.episode_id.isin(ids)]
    predictor,sm=tracked_sam(a.sam_checkpoint)
    write_json(a.output/f"sam_load_attempt{len(list(a.output.glob('sam_load_attempt*.json'))):03d}.json",sm)
    slice_count=0;completed_eps=0;failure_count=0;loads=[]
    for fold in sorted(manifest.fold.unique()):
        model,meta=load_models(c,a.checkpoint_root,a.sam_checkpoint,int(fold));loads.append(meta)
        write_json(a.output/f"fold{fold}_load_attempt{len(list(a.output.glob(f'fold{fold}_load_attempt*.json'))):03d}.json",meta)
        for row in manifest[manifest.fold==fold].to_dict("records"):
            folder=a.output/f"episode_{row['episode_id']:03d}";folder.mkdir(exist_ok=True);done=load_completed(folder)
            expected=None
            for case in episode_inputs(row,a.data_dir,c):
                expected=case["active_slices"];z=case["compact_z"]
                if z in done:
                    assert done[z]["native_z"]==case["native_z"];slice_count+=1;continue
                if time.time()>a.deadline_epoch-30:
                    write_json(a.output/f"budget_stop_{time.time_ns()}.json",{"completed_slices":slice_count,"episode_id":row["episode_id"]})
                    raise SystemExit(75)
                arrays,record,_=run_shared(model,predictor,case)
                record.update(worker=a.worker,physical_gpu=a.worker,checkpoint_sha256=meta["checkpoint_sha256"],variant="width10_valid_channels")
                meta_slice=save_slice(folder,arrays,record);done[z]=meta_slice;slice_count+=1
                failure_count+=len(record["failures"])
                if z%10==0 or z==expected-1:
                    print(f"worker={a.worker} fold={fold} ep={row['episode_id']} slice={z+1}/{expected} total={slice_count} elapsed={time.perf_counter()-start:.2f}s failures={failure_count}",flush=True)
            assert len(done)==expected
            ep_path=folder/"episode_complete.json"
            if not ep_path.exists():write_json(ep_path,{"episode_id":int(row["episode_id"]),"slices":expected,"arms":list(ARMS),"posts":["raw","cc"],"commits":len(done)})
            completed_eps+=1
        del model;gc.collect();torch.cuda.empty_cache()
    result={"worker":a.worker,"hardware":torch.cuda.get_device_name(0),"episodes":completed_eps,"slices":slice_count,
            "failures_observed_this_attempt":failure_count,"wall_seconds":time.perf_counter()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "seed":c["seed"],"loads":loads}
    write_json(a.output/f"worker_complete_attempt{len(list(a.output.glob('worker_complete_attempt*.json'))):03d}.json",result)
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()
