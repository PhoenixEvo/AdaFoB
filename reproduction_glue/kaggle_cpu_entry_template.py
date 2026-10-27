"""CPU-only Kaggle payload probe. No images/masks/checkpoints are uploaded."""
import base64,hashlib,io,json,os,sys,time,zipfile,subprocess,importlib.util,traceback
from pathlib import Path

FIRST_LOG_UNIX=time.time()
print(json.dumps({'event':'entry_started','unix':FIRST_LOG_UNIX,'mode':'CPU_ONLY_PAYLOAD_PROBE'}),flush=True)
OUT=Path('/kaggle/working/payload_probe');OUT.mkdir(exist_ok=False)
def save(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
status={'status':'partial','gpu_requested':False,'first_log_unix':FIRST_LOG_UNIX}
save('initial_status.json',status)
try:
    payload=base64.b64decode('__BUNDLE__',validate=True)
    if hashlib.sha256(payload).hexdigest()!='__BUNDLE_SHA__':raise ValueError('Uploaded source bundle mismatch')
    source=OUT/'source';source.mkdir()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if z.testzip() is not None:raise ValueError('Source ZIP CRC')
        for name in z.namelist():
            if name.startswith('/') or '..' in Path(name).parts:raise ValueError('Unsafe source path')
        z.extractall(source)
    inv=json.loads((source/'probe_inventory.json').read_text())
    for row in inv:
        if sha(source/row['path'])!=row['sha256']:raise ValueError('Member mismatch '+row['path'])
    save('uploaded_member_checks.json',{'status':'PASS','files':inv})
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    installed=False
    if importlib.util.find_spec('SimpleITK') is None:
        print('Installing only locked SimpleITK==2.4.1 from official PyPI',flush=True)
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--index-url','https://pypi.org/simple','SimpleITK==2.4.1'],check=True,timeout=120)
        installed=True
    import numpy,SimpleITK
    save('environment.json',{'python':sys.version,'executable':sys.executable,'numpy':numpy.__version__,
          'SimpleITK':SimpleITK.__version__,'SimpleITK_install_ran':installed,'gpu_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
          'torch_not_imported':('torch' not in sys.modules)})
    sys.path.insert(0,str(source))
    from experiments.route_b_payload_v1.adapter import verify_payload_manifest
    roots=[p for p in Path('/kaggle/input').glob('**/sabs_CT_normalized') if (p/'image_8.nii').is_file()]
    if len(roots)!=1:raise ValueError('Ambiguous/missing approved data mount')
    manifest=source/'protocol/pilot_manifest_v1.csv';ent=source/'protocol/development_entitlement.csv';ids=source/'payload_identities.json'
    report,rows,volumes=verify_payload_manifest(manifest,roots[0],ent,ids)
    report['status']='PASS';report['payload_match_proves']='Complete decompressed NIfTI header+voxel bytes match locked local payloads exactly'
    report['original_gzip_check_on_kaggle']='NOT PASSED / NOT CLAIMED; owner amended to payload identity'
    save('payload_probe_report.json',report)
    fobs=list(Path('/kaggle/input').glob('**/FSMIS_train_SABS_cv0/ckpt.pth'))
    if len(fobs)!=1:raise ValueError('Ambiguous/missing cv0 checkpoint')
    fob=sha(fobs[0])
    if fob!='e3c9ccc47f74fa562e20c56ba11768aa8661b4d69fd170eb815943df3ff63913':raise ValueError('FoB checkpoint mismatch')
    sams=[]
    for p in Path('/kaggle/input').glob('**/sam_vit_b*.pth'):
        actual=sha(p);sams.append({'path':str(p),'sha256':actual})
        if actual!='ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912':raise ValueError('Mounted SAM-B hash mismatch')
    save('checkpoint_probe.json',{'status':'PASS','fob_sha256':fob,'sam_b_candidates':sams,'sam_b_present':bool(sams),
                                  'sam_h_substituted':False,'checkpoint_loaded':False})
    status['status']='KAGGLE-CPU-PAYLOAD-PROBE-PASS';code=0
except Exception:
    status['status']='PILOT-BLOCKED';status['error']=traceback.format_exc();print(status['error'],flush=True);code=1
status['wall_seconds_from_first_log']=time.time()-FIRST_LOG_UNIX
save('final_status.json',status)
manifest=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file():manifest.append({'path':str(p.relative_to(OUT)),'sha256':sha(p),'bytes':p.stat().st_size})
save('output_hashes.json',manifest)
print(json.dumps(status),flush=True)
sys.exit(code)