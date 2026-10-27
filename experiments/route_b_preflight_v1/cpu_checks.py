"""CPU-only standard-reader and strict-load preflight. Never calls model forward."""
import argparse,sys,json,platform,importlib,traceback,gc,types,ast,time,os
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def memory():
    import ctypes
    if sys.platform!='win32':return {'note':'Windows process counters unavailable'}
    class Mem(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('peak_rss','rss','peak_pagepool','pagepool','peak_nonpage','nonpage','pagefile','peak_pagefile')]
    m=Mem();m.cb=ctypes.sizeof(m)
    get_handle=ctypes.windll.kernel32.GetCurrentProcess;get_handle.restype=ctypes.c_void_p
    handle=get_handle()
    get_memory=ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes=[ctypes.c_void_p,ctypes.POINTER(Mem),ctypes.c_ulong]
    ok=get_memory(handle,ctypes.byref(m),m.cb)
    return {'peak_rss_bytes':m.peak_rss,'rss_bytes':m.rss} if ok else {'error':'GetProcessMemoryInfo failed'}

def load_fob_cpu(implementation,width,path):
    import torch
    from experiments.route_b_preflight_v1.inputs import sha
    relative='FoB_SAM_upstream' if implementation=='upstream' else 'FoB_SAM'
    label=f'_route_b_cpu_{implementation}_{width}'
    package=types.ModuleType(label);package.__path__=[str(ROOT/'third_party'/relative/'models')];sys.modules[label]=package
    if 'info_nce' not in sys.modules and importlib.util.find_spec('info_nce') is None:
        loss=types.ModuleType('info_nce')
        class DisabledTrainingLoss:
            def __init__(self,*a,**k):pass
            def __call__(self,*a,**k):raise RuntimeError('Training forbidden')
        loss.InfoNCE=DisabledTrainingLoss;sys.modules['info_nce']=loss
    enc=importlib.import_module(label+'.encoder');cls=importlib.import_module(label+'.FoB').FewShotSeg
    original=enc.Res101Encoder.__init__
    def offline_init(self,replace_stride_with_dilation=None,pretrained_weights=None):
        original(self,replace_stride_with_dilation=replace_stride_with_dilation,pretrained_weights=None)
    with patch.object(enc.Res101Encoder,'__init__',offline_init):
        model=cls(types.SimpleNamespace(dataset='SABS',max_points=width,n_ways=1,n_shots=1)).cpu().eval()
    raw=torch.load(path,map_location='cpu',weights_only=False)
    state={k.replace('module.','').replace('_orig_mod.',''):v for k,v in raw.get('state_dict',raw).items()}
    before={k:list(v.shape) for k,v in state.items()};raw_error=None
    try:model.load_state_dict(dict(state),strict=True)
    except RuntimeError as e:
        raw_error=str(e);missing=set(model.state_dict())-set(state);extra=set(state)-set(model.state_dict())
        if missing!={'encoder.reduce2.weight'} or extra:raise
        tree=ast.parse((ROOT/'third_party'/relative/'models/encoder.py').read_text())
        forward=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        if any(isinstance(n,ast.Attribute) and n.attr=='reduce2' for n in ast.walk(forward)):raise RuntimeError('Cannot remove used layer')
        del model.encoder.reduce2
    result=model.load_state_dict(dict(state),strict=True)
    if result.missing_keys or result.unexpected_keys:raise RuntimeError('Strict load differences')
    after={k:list(v.shape) for k,v in model.state_dict().items()}
    changed={k:{'checkpoint':v,'model':after.get(k)} for k,v in before.items() if v!=after.get(k)}
    allowed={'head.head.3.weight','head.head.3.bias','masked_attention.conv.weight'} if width==24 else set()
    if set(changed)!=allowed:raise RuntimeError('Unapproved shape adaptation')
    for k in changed:
        tensor=model.state_dict()[k];source=state[k];slices=tuple(slice(0,n) for n in source.shape)
        if not torch.equal(tensor[slices],source):raise RuntimeError('Padded source changed')
        tail=tensor.clone();tail[slices]=0
        if torch.count_nonzero(tail).item()!=0:raise RuntimeError('Nonzero padding')
    if set(after)!=set(before) or any(v.device.type!='cpu' for v in model.state_dict().values()):raise RuntimeError('Schema/device mismatch')
    record={'implementation':implementation,'width':width,'checkpoint':str(path),'sha256':sha(path),
            'raw_strict_load_error':raw_error,'documented_adapter':'unused_reduce2_removed' if raw_error else 'none',
            'strict_missing_keys':result.missing_keys,'strict_unexpected_keys':result.unexpected_keys,'shape_adaptations':changed,
            'checkpoint_tensor_bytes':sum(v.numel()*v.element_size() for v in state.values()),
            'model_tensor_bytes':sum(v.numel()*v.element_size() for v in model.state_dict().values()),
            'memory':memory(),'forward_calls':0,'load_is_not_numerical_parity':True}
    del model,state,raw;gc.collect();return record

def checkpoints(out):
    import torch
    from experiments.route_b_preflight_v1.inputs import sha
    records=json.loads((ROOT/'audit/route_b_preflight_20261025/checkpoint_hashes.json').read_text())
    for r in records:
        if sha(r['path']).lower()!=r['sha256'].lower():raise ValueError('Checkpoint changed')
    fob=next(Path(r['path']) for r in records if r['type']=='FoB' and r['fold']==0)
    sam=next(Path(r['path']) for r in records if r['type']=='SAM-B')
    torch.set_num_threads(1)
    def forbidden(*a,**k):raise RuntimeError('Model forward / GPU initialization forbidden in preflight')
    loaded=[]
    with patch.object(torch.nn.Module,'_call_impl',forbidden),patch.object(torch.cuda,'_lazy_init',forbidden):
        for implementation,width in [('upstream',10),('local',10),('local',24)]:
            print(f'CPU strict loading {implementation} width{width}',flush=True)
            loaded.append(load_fob_cpu(implementation,width,fob));write(out/f'load_{implementation}_{width}.json',loaded[-1])
        sys.path.insert(0,str(ROOT/'third_party/segment-anything'))
        from segment_anything import sam_model_registry
        model=sam_model_registry['vit_b'](checkpoint=None).cpu().eval()
        state=torch.load(sam,map_location='cpu',weights_only=False)
        result=model.load_state_dict(state,strict=True)
        if result.missing_keys or result.unexpected_keys:raise RuntimeError('SAM strict load mismatch')
        write(out/'load_sam_b.json',{'path':str(sam),'sha256':sha(sam),'missing_keys':result.missing_keys,
             'unexpected_keys':result.unexpected_keys,'model_tensor_bytes':sum(t.numel()*t.element_size() for t in model.state_dict().values()),
             'checkpoint_tensor_bytes':sum(t.numel()*t.element_size() for t in state.values()),'memory':memory(),'forward_calls':0})
    write(out/'memory_requirements.json',{'policy':'One model and its checkpoint tensors resident at a time; gc between variants',
          'live_model_plus_checkpoint_tensor_bytes':[r['model_tensor_bytes']+r['checkpoint_tensor_bytes'] for r in loaded],
          'not_a_total_RAM_bound':'Framework/import overhead, allocator and temporary load/padding storage are additional; process peaks are recorded per load.',
          'peak_process_memory':memory()})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--data-dir',default='D:/sabs_CT_normalized')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    state={'status':'partial','no_model_forward':True,'phases':{}}
    write(out/'initial_status.json',state)
    imports={}
    for mod in ('numpy','scipy','SimpleITK','torch','torchvision','cv2','pandas','medpy'):
        try:
            m=importlib.import_module(mod);imports[mod]={'status':'IMPORTED','version':getattr(m,'__version__',None)}
        except Exception as e:imports[mod]={'status':'FAIL','error':repr(e)}
        print('IMPORT',mod,imports[mod],flush=True)
    write(out/'environment.json',{'interpreter':sys.executable,'python':sys.version,'platform':platform.platform(),'packages':imports,
          'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'memory':memory(),'no_network_or_install':True})
    def phase(name,fn):
        print('START',name,flush=True);start=time.perf_counter()
        try:fn();state['phases'][name]={'status':'PASS','seconds':time.perf_counter()-start}
        except Exception:
            state['phases'][name]={'status':'FAIL','error':traceback.format_exc(),'seconds':time.perf_counter()-start};print(state['phases'][name]['error'],flush=True)
        print('END',name,state['phases'][name]['status'],flush=True)
    def manifest():
        from experiments.route_b_preflight_v1.inputs import verify_manifest
        results,_,_=verify_manifest(ROOT/'protocol/pilot_manifest_v1.csv',args.data_dir,ROOT/'audit/route_b_preflight_20261025/development_entitlement_repaired_v1.csv')
        write(out/'manifest_standard_reader.json',{'reader':'SimpleITK','cases':results,'hash_scope':'separate file and decoded array hashes; locked slice raw bytes and uint8 masks','geometry':'SimpleITK LPS to NIfTI RAS; manifest rounded decimals atol1e-6, rtol0'})
    def integration():
        import unittest
        from experiments.route_b_preflight_v1.test_integration import IntegrationTests
        r=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(IntegrationTests))
        write(out/'integration_status.json',{'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'synthetic_only':True})
        if not r.wasSuccessful() or r.skipped:raise RuntimeError('Integration suite did not pass fully')
    phase('manifest',manifest);phase('synthetic_integration',integration);phase('strict_cpu_loads',lambda:checkpoints(out))
    ok=all(p['status']=='PASS' for p in state['phases'].values()) and all(v['status']=='IMPORTED' for v in imports.values())
    state['status']='CPU-CHECKS-COMPLETE' if ok else 'PREFLIGHT-BLOCKED'
    state['scope']='CPU reader/strict-load/synthetic integration evidence only; no real predictions or GPU approval'
    write(out/'final_status.json',state);return 0 if ok else 1

if __name__=='__main__':sys.exit(main())