"""Eight committed diagnostic slices, warm-up/three repeats and fail-closed cost."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np,pandas as pd
from analysis.rebaseline_v1.common import read_config,read_manifest,tracked_sam,load_models,episode_inputs,serialized_slice,write_json,seed_all
from .core import run_slice,ARMS,BASE,ZERO

def main():
 p=argparse.ArgumentParser();p.add_argument('--lock-dir',type=Path,required=True);p.add_argument('--data-dir',required=True);p.add_argument('--checkpoint-root',required=True);p.add_argument('--sam-checkpoint',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);c=read_config(a.lock_dir);import torch
 seed_all(c['seed']);assert 'T4' in torch.cuda.get_device_name(0)
 expected=json.loads((a.lock_dir/'step8_fingerprints.json').read_text());diag=pd.read_csv(a.lock_dir/'diagnostic_manifest.csv');manifest=read_manifest(a.lock_dir/'paired_episode_manifest.csv')
 predictor,sm=tracked_sam(a.sam_checkpoint);records=[];loads=[];all_calls=0;all_encoders=0;started=time.perf_counter()
 for fold in sorted(diag.fold.unique()):
  model,meta=load_models(c,a.checkpoint_root,a.sam_checkpoint,int(fold));loads.append(meta)
  for row in diag[diag.fold==fold].to_dict('records'):
   episode=manifest[manifest.episode_id==row['episode_id']].iloc[0].to_dict()
   case=next(v for v in episode_inputs(episode,a.data_dir,c) if v['native_z']==row['native_z'])
   assert case['support_native_z']==row['support_native_z'] and case['support_part']==row['support_part']
   with np.load(a.lock_dir/'baseline_diagnostic'/f"{case['episode_id']}_{case['native_z']}.npz") as old:old_arrays={k:old[k].copy() for k in old.files}
   for repeat in range(c['pilot_repeats']+1):
    torch.cuda.reset_peak_memory_stats();arrays,record=run_slice(model,predictor,case,c,expected[f"{case['episode_id']}:{case['native_z']}"])
    for name,old in old_arrays.items():np.testing.assert_array_equal(arrays[name],old)
    all_calls+=record['decoder_calls'];all_encoders+=record['image_encoder_calls']
    peak=torch.cuda.max_memory_allocated();t=time.perf_counter();payload=serialized_slice(arrays,record)
    path=a.output/f"ep{case['episode_id']:03d}_repeat{repeat}.npz"
    with path.open('xb') as f:f.write(payload)
    write_seconds=time.perf_counter()-t
    if repeat:
     r=record['runtime'];upper=r['slice_compute_seconds']-r['decoder_seconds']+90*r['decoder_seconds_per_call']+write_seconds
     records.append({'episode_id':case['episode_id'],'native_z':case['native_z'],'repeat':repeat,**r,'serialize_write_seconds':write_seconds,
                     'actual_decoder_calls':record['decoder_calls'],'max90_slice_seconds':upper,'serialized_bytes':len(payload),
                     'bytes_per_arm':len(payload)/len(ARMS),'peak_allocated_bytes':peak,'reference_actual_calls':record['reference_actual_decoder_calls']})
    print(f"pilot ep={case['episode_id']} repeat={repeat} total={record['runtime']['slice_compute_seconds']:.4f}s calls={record['decoder_calls']} C0_EXACT=True",flush=True)
  del model;gc.collect();torch.cuda.empty_cache()
 df=pd.DataFrame(records);df.to_csv(a.output/'timing_repeats.csv',index=False)
 actual=float((df.slice_compute_seconds+df.serialize_write_seconds).quantile(.9));upper=float(df.max90_slice_seconds.quantile(.9))
 conservative=max(actual,upper);wall=conservative*c['total_slices']/2*c['projection_overhead_factor']+c['projection_startup_seconds']
 result={'hardware':torch.cuda.get_device_name(0),'unique_slices':8,'repeats':3,'warmup_per_slice':1,'arms':len(ARMS),'encoder_calls':all_encoders,
         'decoder_calls':all_calls,'c0_step8_exact_all_repeats':True,
         'comparison_note':'C0 compared with Step8 saved raw/CC; no new separate-pass strategy in this batch',
         'p90_observed_slice_seconds':actual,'p90_max90_slice_seconds':upper,'projection_uses_worst90_calls':True,
         'max_logit_quality_difference':None,'peak_allocated_bytes':int(df.peak_allocated_bytes.max()),
         'mean_storage_bytes_per_slice':float(df.serialized_bytes.mean()),'estimated_slice_archive_bytes':float(df.serialized_bytes.mean()*c['total_slices']),
         'projection_wall_seconds':wall,'projection_two_T4_gpu_hours':2*wall/3600,'pilot_stage_wall_seconds':time.perf_counter()-started,'loads':loads,'sam_load':sm}
 write_json(a.output/'pilot_summary.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()