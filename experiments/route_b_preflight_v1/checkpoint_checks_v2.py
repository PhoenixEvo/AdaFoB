"""Strict checkpoint-only recovery: retain two constructor tensors on CPU.
No model/source/schema edit; no forward, GPU initialization or numerical inference.
"""
from pathlib import Path
import ast,inspect,json,sys,traceback,time,platform,importlib
from unittest.mock import patch
from experiments.route_b_preflight_v1 import cpu_checks as previous
from experiments.route_b_preflight_v1.inputs import sha

ROOT=Path(__file__).resolve().parents[2]
ORIGINAL_LOAD=previous.load_fob_cpu

def load_with_cpu_constructor(implementation,width,path):
    import torch
    source=ROOT/'third_party'/('FoB_SAM_upstream' if implementation=='upstream' else 'FoB_SAM')/'models/FoB.py'
    tree=ast.parse(source.read_text());sites={}
    for cls in tree.body:
        if not isinstance(cls,ast.ClassDef):continue
        for fn in cls.body:
            if not isinstance(fn,ast.FunctionDef) or fn.name!='__init__':continue
            for statement in ast.walk(fn):
                if not isinstance(statement,ast.Assign):continue
                fields=[n.attr for n in statement.targets if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self']
                for field in fields:
                    if (cls.name,field) not in [('MaskedAttention','sin_pos'),('FewShotSeg','nllloss')]:continue
                    calls=[n for n in ast.walk(statement.value) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='cuda']
                    if len(calls)!=1:raise RuntimeError('Unexpected constructor .cuda source')
                    sites[calls[0].lineno]=(cls.name,field)
    if set(sites.values())!={('MaskedAttention','sin_pos'),('FewShotSeg','nllloss')}:raise RuntimeError('Two exact known constructor sites required')
    events=[]
    def retain_cpu(tensor,*args,**kwargs):
        caller=inspect.currentframe().f_back
        try:
            site=sites.get(caller.f_lineno)
            owner=caller.f_locals.get('self')
            if Path(caller.f_code.co_filename).resolve()!=source.resolve() or caller.f_code.co_name!='__init__' or site is None or type(owner).__name__!=site[0]:
                raise RuntimeError('Unapproved Tensor.cuda call; not a known FoB constructor site')
            if args or kwargs or tensor.device.type!='cpu' or tensor.dtype!=torch.float32:raise RuntimeError('Unexpected constructor tensor/device arguments')
            if site[1]=='sin_pos':
                if tuple(tensor.shape)!=(10 if implementation=='upstream' else width,512):raise RuntimeError('Positional tensor shape')
            else:
                if tuple(tensor.shape)!=(2,) or not torch.equal(tensor,torch.tensor([.1,1.],dtype=torch.float32)):raise RuntimeError('Loss tensor values')
            events.append({'class':site[0],'field':site[1],'line':caller.f_lineno,'shape':list(tensor.shape),
                           'placement':'CPU tensor returned unchanged; no device transfer',
                           'value_sha256':__import__('hashlib').sha256(tensor.numpy().tobytes()).hexdigest()})
            return tensor
        finally:del caller
    # The original loader's only cuda calls are in its constructor. Restrict this
    # shim by source, line, owning class, tensor shape, dtype, and zero arguments.
    with patch.object(torch.Tensor,'cuda',retain_cpu):
        record=ORIGINAL_LOAD(implementation,width,path)
    if len(events)!=2 or {e['field'] for e in events}!={'sin_pos','nllloss'}:raise RuntimeError('Constructor shim call count')
    record['constructor_cpu_placement']={'source_sha256':sha(source),'events':events,
        'scope':'Only known positional/loss constructor tensors remain on CPU; no schema/weight/forward change'}
    return record

def main():
    import argparse,os
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    previous.write(out/'initial_status.json',{'status':'partial','scope':'strict checkpoint-only CPU recovery; no model forward'})
    status={'status':'PREFLIGHT-BLOCKED','no_model_forward':True,'no_gpu_allocation':True}
    started=time.perf_counter();imports={}
    try:
        for name in ('numpy','torch','torchvision','cv2'):
            try:
                m=importlib.import_module(name);imports[name]={'status':'IMPORTED','version':getattr(m,'__version__',None)}
            except Exception as e:imports[name]={'status':'FAIL','error':repr(e)}
            print('IMPORT',name,imports[name],flush=True)
        previous.write(out/'environment.json',{'interpreter':sys.executable,'python':sys.version,'platform':platform.platform(),
            'packages':imports,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'memory':previous.memory()})
        if any(v['status']!='IMPORTED' for v in imports.values()):raise RuntimeError('Required package import failed; no install authorized')
        # Original checkpoints() still prohibits every module forward and CUDA
        # lazy initialization, and retains strict=True and all schema checks.
        with patch.object(previous,'load_fob_cpu',load_with_cpu_constructor):previous.checkpoints(out)
        status['status']='CHECKPOINT-CPU-CHECKS-COMPLETE';status['strict_cpu_loads']='PASS';code=0
    except Exception:
        status['strict_cpu_loads']='FAIL';status['error']=traceback.format_exc();print(status['error'],flush=True);code=1
    status['seconds']=time.perf_counter()-started
    previous.write(out/'final_status.json',status);return code

if __name__=='__main__':sys.exit(main())