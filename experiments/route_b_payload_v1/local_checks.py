"""Local prerequisite only: six-volume parity and persistence synthetic test.
No real model forward, GPU allocation, installs or network requests.
"""
import argparse,json,sys,time,traceback,hashlib
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def parity(original,payload,identity_file,out):
    import numpy as np
    from experiments.route_b_preflight_v1 import inputs as frozen
    from experiments.route_b_payload_v1.adapter import load_identities,verify_payload_manifest
    ids=load_identities(identity_file)
    for stem,r in ids.items():
        if frozen.sha(Path(original)/r['original_gzip_name'])!=r['original_gzip_sha256']:raise ValueError('Original locked gzip changed: '+stem)
    manifest=ROOT/'protocol/pilot_manifest_v1.csv'
    ent=ROOT/'audit/route_b_preflight_20261025/development_entitlement_repaired_v1.csv'
    old_cases,old_rows,old_vols=frozen.verify_manifest(manifest,original,ent)
    new,new_rows,new_vols=verify_payload_manifest(manifest,payload,ent,identity_file)
    if old_rows!=new_rows:raise ValueError('Scientific manifest rows changed')
    results=[]
    for patient in ('image_8','image_14','image_20'):
        for index,kind in ((0,'image'),(1,'label')):
            a=old_vols[patient][index];b=new_vols[patient][index]
            stem=patient if kind=='image' else patient.replace('image_','label_')
            np.testing.assert_array_equal(a['array'],b['array'])
            np.testing.assert_array_equal(a['ras'],b['ras'])
            if a['dtype']!=b['dtype'] or a['decoded_array_sha256']!=b['decoded_array_sha256']:raise ValueError('Array serialization mismatch')
            geometry={}
            for name in ('GetSize','GetSpacing','GetOrigin','GetDirection'):
                av=getattr(a['itk'],name)();bv=getattr(b['itk'],name)()
                if av!=bv:raise ValueError('Geometry parity: '+stem+' '+name)
                geometry[name]=list(bv)
            if a['file_sha256']!=ids[stem]['original_gzip_sha256'] or b['file_sha256']!=ids[stem]['payload_sha256']:raise ValueError('Wrong identity basis')
            results.append({'stem':stem,'status':'PASS','arrays_equal':True,'ras_equal':True,'geometry':geometry,
                'original_gzip_sha256':a['file_sha256'],'payload_sha256':b['file_sha256'],
                'decoded_array_sha256':b['decoded_array_sha256'],'dtype':b['dtype'],'shape_zyx':list(b['array'].shape)})
            print('VOLUME PARITY PASS',stem,flush=True)
    slices=[]
    for old,new_case in zip(old_cases,new['cases']):
        if old['patient']!=new_case['patient'] or old['class_id']!=new_case['class_id']:raise ValueError('Case ordering changed')
        checks={}
        for prefix in ('query','support'):
            for key in (prefix+'_image_slice_sha256',prefix+'_base_mask_sha256'):
                if old[prefix]['checks'][key]!=new_case[prefix]['checks'][key]:raise ValueError('Slice/mask hash parity')
                checks[key]=new_case[prefix]['checks'][key]
            for field in ('z','decoded_image_array_sha256','decoded_label_array_sha256'):
                if old[prefix][field]!=new_case[prefix][field]:raise ValueError('Slice or decoded identity parity')
        slices.append({'patient':old['patient'],'class_id':old['class_id'],'status':'PASS','slice_mask_checks':checks})
    write(out/'local_parity.json',{'status':'PASS','volumes':results,'cases':slices,'identity_basis_change_only':True,
                                 'remote_gzip_container_check':'NOT PERFORMED; NOT CLAIMED'})

def persistence_test(out):
    import numpy as np
    from experiments.route_b_payload_v1.persistence import persistence_wrapper,array_hashes,sha
    from experiments.route_b_preflight_v1 import runner as core
    from experiments.route_b_preflight_v1 import launch_pilot_v1 as launch
    from experiments.route_b_preflight_v1.test_integration import Backend
    root=out/'persistence_before_gt';reads=[]
    class GuardedLabel:
        def __init__(self,i):self.i=i
        def check(self):
            seal=root/('case'+str(self.i)+'.seal.json')
            if not seal.is_file():raise AssertionError('Retrospective label accessed before prediction seal')
            s=json.loads(seal.read_text())
            if s['retrospective_evaluation']!='NOT STARTED' or s['npz_sha256']!=sha(root/('case'+str(self.i)+'.npz')):raise AssertionError('Persistence evidence invalid')
            reads.append({'case':self.i,'read_unix':time.time(),'sealed_unix':s['sealed_unix']})
            return np.zeros((257,257),dtype=bool)
        def __array__(self,dtype=None,copy=None):return np.asarray(self.check(),dtype=dtype)
        def sum(self):return self.check().sum()
    def cases():
        for i,(name,inputs,unused_gt) in enumerate(launch.synthetic_cases()):yield name,inputs,GuardedLabel(i)
    clock=launch.Clock();backend=launch.MeasuredBackend(Backend(),clock);writer=core.RunWriter(out/'synthetic_final_predictions')
    wrapped=persistence_wrapper(core.pilot_predict,root,clock)
    with patch.object(core,'pilot_predict',wrapped):result=launch.run_cases(cases(),backend,clock,writer)
    if writer.state['status']!='completed' or len(writer.state['completed_cases'])!=8 or backend.total_decodes>160:raise AssertionError('Synthetic accounting')
    for i in range(8):
        seal=json.loads((root/('case'+str(i)+'.seal.json')).read_text())
        if not any(e['case']==i for e in reads):raise AssertionError('Label guard was never exercised')
        if any(e['sealed_unix']>e['read_unix'] for e in reads if e['case']==i):raise AssertionError('Persistence ordering')
        with np.load(out/'synthetic_final_predictions'/('case'+str(i)+'.npz'),allow_pickle=False) as z:
            if array_hashes({k:z[k] for k in z.files})!=seal['array_hashes']:raise AssertionError('Pre-evaluation/final arrays differ')
    write(out/'persistence_test.json',{'status':'PASS','synthetic_only':True,'cases':8,'label_read_events':reads,
          'all_reads_after_persist_and_hash':True,'pre_evaluation_arrays_equal_final_arrays':True,
          'configurations':result['actual_decoder_configurations'],'logical_requests':result['logical_requests']})
    print('PERSISTENCE SYNTHETIC PASS: eight cases; all metric label reads after persisted/reloaded prediction seal',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--original-dir',required=True);ap.add_argument('--payload-dir',required=True);ap.add_argument('--identities',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    state={'status':'partial','scope':'CPU parity and synthetic persistence; no real model inference'}
    write(out/'initial_status.json',state)
    try:
        import numpy,scipy,SimpleITK
        write(out/'environment.json',{'interpreter':sys.executable,'python':sys.version,'numpy':numpy.__version__,'scipy':scipy.__version__,'SimpleITK':SimpleITK.__version__})
        parity(args.original_dir,args.payload_dir,args.identities,out)
        persistence_test(out)
        state['status']='LOCAL-PARITY-AND-PERSISTENCE-PASS';code=0
    except Exception:
        state['status']='PILOT-BLOCKED';state['error']=traceback.format_exc();print(state['error'],flush=True);code=1
    write(out/'final_status.json',state);return code

if __name__=='__main__':sys.exit(main())