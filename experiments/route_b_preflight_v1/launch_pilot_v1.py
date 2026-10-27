"""Bounded research launch wrapper. Default mode is CPU synthetic smoke only.
Actual inference requires the explicit GPU flag and separate owner authorization.
"""
import argparse,json,time,sys,os,threading,io
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]

def write(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)

class Clock:
    def __init__(self,sync=lambda:None):self.sync=sync;self.events=[]
    def call(self,name,fn,*args,**kwargs):
        self.sync();start=time.perf_counter()
        result=fn(*args,**kwargs)
        self.sync();self.events.append({'component':name,'seconds':time.perf_counter()-start})
        return result

class MeasuredBackend:
    def __init__(self,backend,clock):self.base=backend;self.clock=clock;self.total_decodes=0;self.case_decodes=0
    def __getattr__(self,name):return getattr(self.base,name)
    def prepare_support(self,*a):return self.clock.call('support_encoding_feature_copy_and_cache',self.base.prepare_support,*a)
    def generate(self,*a):return self.clock.call('FoB_generation_including_transfers',self.base.generate,*a)
    def encode(self,*a):return self.clock.call('query_encoding_including_feature_copy',self.base.encode,*a)
    def decode(self,*a):
        if self.total_decodes>=160 or self.case_decodes>=20:raise RuntimeError('Decoder configuration cap')
        self.total_decodes+=1;self.case_decodes+=1
        return self.clock.call('decoder_configuration_including_transfers',self.base.decode,*a)
    def cc(self,*a):return self.clock.call('connected_components',self.base.cc,*a)

def run_cases(cases,backend,clock,writer):
    import numpy as np
    import experiments.route_b_preflight_v1.runner as core
    # Wrappers preserve the frozen function and all outputs. No scoring change.
    with ExitStack() as stack:
        for name in ('evidence','score_subsets','proximity','mask_on_feature_grid'):
            old=getattr(core,name)
            def measured(*a,_name=name,_old=old,**kw):return clock.call('scoring_selection_'+_name,_old,*a,**kw)
            stack.enter_context(patch.object(core,name,measured))
        for case_id,legitimate,query_gt in cases:
            backend.case_decodes=0;start=time.perf_counter();before=len(clock.events)
            arrays,record=core.pilot_predict(**legitimate,backend=backend)
            if backend.case_decodes!=record['configurations_decoded']:raise RuntimeError('Backend/cache accounting mismatch')
            record['actual_positive_count']=int(len(arrays['positive_source_xy']))
            record['evaluation']=core.evaluate_saved(arrays,query_gt) # Outside prediction/selection boundary.
            record['physical_area_mm2_per_pixel']=float(__import__('math').prod(legitimate['geometry'].spacing_xy))
            for metric in record['evaluation'].values():
                metric['fp_area_mm2']=metric['fp_pixels']*record['physical_area_mm2_per_pixel']
                metric['fn_area_mm2']=metric['fn_pixels']*record['physical_area_mm2_per_pixel']
            # Retrospective raw-Dice deletion utility only after prediction completes.
            def dice(mask):
                n=int(mask.sum()+query_gt.sum());return 2*int((mask&query_gt).sum())/n if n else 1.
            baseline=dice(arrays['full10_raw'])
            record['retrospective_deletion_benefit']=[dice(m)-baseline for m in arrays['single_deletion_raw']]
            record['components']=clock.events[before:].copy()
            record['slice_prediction_and_retrospective_metrics_seconds']=time.perf_counter()-start
            record['warmup_policy']='first full10 decode retained/counts toward cap; no extra decoder repetitions'
            # Three buffer-compression repetitions; not disk-latency estimates.
            compression=[]
            for _ in range(3):
                buf=io.BytesIO();t=time.perf_counter();np.savez_compressed(buf,**arrays)
                compression.append({'seconds':time.perf_counter()-t,'bytes':buf.tell()})
            record['compression_repeats']=compression
            clock.call('disk_serialization_and_case_commit',writer.save,case_id,arrays,record)
    writer.complete()
    if backend.total_decodes!=writer.state['configurations_decoded']:raise RuntimeError('Aggregate decoder accounting')
    return {'actual_decoder_configurations':backend.total_decodes,'logical_requests':160,
            'api_decoder_invocations':backend.total_decodes,'support_encoder_calls':backend.support_encoder_calls,
            'timing_events':clock.events,'stored_bytes':sum(p.stat().st_size for p in writer.path.rglob('*') if p.is_file())}

def synthetic_cases():
    import numpy as np
    from experiments.route_b_preflight_v1.test_integration import IntegrationTests
    for i in range(8):
        inputs=IntegrationTests().inputs()
        # Four distinct allowed support surrogates, reused for the second synthetic patient.
        inputs['support_mask']=np.roll(inputs['support_mask'],(i%4)*3,axis=1)
        yield 'case'+str(i),inputs,np.zeros((257,257),bool)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--authorize-gpu-pilot',action='store_true')
    ap.add_argument('--allocation-start-unix',type=float)
    ap.add_argument('--data-dir',default='D:/sabs_CT_normalized')
    ap.add_argument('--fob',default=str(ROOT/'checkpoints/fob_checkpoints/exps_train_on_SABS_FSMIS_FoB/FSMIS_train_SABS_cv0/ckpt.pth'))
    ap.add_argument('--sam',default='D:/results/checkpoints/sam_vit_b_01ec64.pth')
    args=ap.parse_args();start=time.perf_counter();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    smoke=not args.authorize_gpu_pilot
    status={'status':'partial','mode':'CPU_SYNTHETIC_SMOKE' if smoke else 'GPU_PILOT','real_model_forward':not smoke}
    write(out/'initial_status.json',status)
    timer=None
    try:
        config=json.loads((ROOT/'audit/route_b_launch_20261026/launch_config.json').read_text())
        lock=json.loads((ROOT/'audit/route_b_launch_20261026/package_lock.json').read_text())
        from experiments.route_b_preflight_v1.inputs import sha
        for f in lock['files']:
            if sha(ROOT/f['path']).lower()!=f['sha256'].lower():raise RuntimeError('Package hash mismatch: '+f['path'])
        from experiments.route_b_preflight_v1.runner import RunWriter,PilotBackend
        if smoke:
            from experiments.route_b_preflight_v1.test_integration import Backend
            base=Backend();clock=Clock();cases=synthetic_cases();setup={'synthetic_only':True,'no_gpu_import_or_allocation':True}
        else:
            if args.allocation_start_unix is None:raise RuntimeError('Actual allocation start timestamp required')
            elapsed=time.time()-args.allocation_start_unix
            if elapsed<0 or elapsed>=config['hard_allocation_seconds']:raise RuntimeError('Allocation already outside cap')
            def expire():
                try:write(out/'timeout.json',{'status':'aborted','reason':'hard allocation cap','cap_seconds':config['hard_allocation_seconds']})
                finally:os._exit(124)
            timer=threading.Timer(config['hard_allocation_seconds']-elapsed,expire);timer.daemon=True;timer.start()
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible authorized GPU required')
            name=torch.cuda.get_device_name(0)
            if 'T4' not in name:raise RuntimeError('Locked GPU type is NVIDIA T4')
            torch.cuda.reset_peak_memory_stats()
            from analysis.gate_a.runtime import load_fob,official_sam_b,seed_all
            seed_all(42);torch.set_num_threads(1);clock=Clock(torch.cuda.synchronize)
            if sha(args.fob)!=config['fob_sha256'] or sha(args.sam)!=config['sam_b_sha256']:raise RuntimeError('Checkpoint hash mismatch')
            model,fo=clock.call('FoB_setup_and_strict_loading',load_fob,'local',10,args.fob,canonical_unused=True)
            predictor,sa=clock.call('SAM_setup_and_strict_loading',official_sam_b,args.sam)
            from experiments.route_b_preflight_v1.inputs import verify_manifest,prepared_cases
            checked,rows,volumes=clock.call('standard_reader_and_manifest',verify_manifest,ROOT/'protocol/pilot_manifest_v1.csv',args.data_dir,
                                           ROOT/'audit/route_b_preflight_20261025/development_entitlement_repaired_v1.csv')
            base=PilotBackend(model,predictor);cases=prepared_cases(rows,volumes)
            setup={'hardware':name,'precision':'float32;TF32off;noautocast','fob_load':fo,'sam_load':sa,
                   'allocation_start_unix':args.allocation_start_unix,'hard_allocation_seconds':config['hard_allocation_seconds']}
        write(out/'setup.json',setup)
        backend=MeasuredBackend(base,clock);writer=RunWriter(out/'predictions')
        result=run_cases(cases,backend,clock,writer)
        result['total_process_seconds_including_setup']=time.perf_counter()-start
        if not smoke:
            result['peak_allocated_gpu_bytes']=torch.cuda.max_memory_allocated()
            result['peak_reserved_gpu_bytes']=torch.cuda.max_memory_reserved()
            result['observed_allocation_seconds']=time.time()-args.allocation_start_unix
        else:
            if result['actual_decoder_configurations']>160 or result['support_encoder_calls']!=4 or len(writer.state['completed_cases'])!=8:raise RuntimeError('Smoke accounting failed')
            if result['actual_decoder_configurations']>=result['logical_requests']:raise RuntimeError('Smoke did not exercise cache reuse')
        write(out/'runtime.json',result)
        status['status']='CPU-SMOKE-COMPLETE' if smoke else 'GPU-PILOT-COMPLETE'
        status['no_segmentation_quality_claim_from_smoke']=smoke
        code=0
    except Exception:
        status['status']='FAIL';status['error']=__import__('traceback').format_exc();print(status['error'],flush=True);code=1
    finally:
        if timer:timer.cancel()
        status['total_process_seconds']=time.perf_counter()-start;write(out/'final_status.json',status)
    print(json.dumps(status),flush=True);return code

if __name__=='__main__':sys.exit(main())