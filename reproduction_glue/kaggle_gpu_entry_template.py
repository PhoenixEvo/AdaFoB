"""Private Kaggle Route B GPU execution glue. Scientific files are unchanged."""
import os,sys,time,json,hashlib,base64,io,zipfile,traceback,subprocess,threading,re,importlib.util
from pathlib import Path
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
FIRST=time.time();PUSH_START=__PUSH_START__
OUT=Path('/kaggle/working/route_b_gpu_attempt_01');OUT.mkdir(exist_ok=False)
def save(name,obj):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def event(name,**data):print(json.dumps({'event':name,'unix':time.time(),**data}),flush=True)
event('entry_started',first_log_unix=FIRST,push_start_unix=PUSH_START)
status={'status':'partial','first_log_unix':FIRST,'push_start_unix':PUSH_START,'real_model_forward_started':False}
save('initial_status.json',status)
timer=None;created_sam=None
try:
    container=os.environ.get('KAGGLE_CONTAINER_NAME','')
    match=re.search(r'-([0-9]+)-webtier$',container)
    if not match:raise RuntimeError('INFRASTRUCTURE: platform runtime session identifier unavailable')
    session_id=int(match.group(1))
    save('session_identity.json',{'kernel_session_id':session_id,'source':'KAGGLE_CONTAINER_NAME numeric runtime id; validated via task-owned CPU cancellation API',
                                'captured_before_model_loading':True,'container':container})
    event('session_identified_before_models',kernel_session_id=session_id)
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip().splitlines()
    names=[line.split(',')[0].strip() for line in hardware]
    allocated=len(names)
    if allocated not in (1,2) or any('T4' not in n for n in names):raise RuntimeError('HARDWARE: expected allocated T4 or T4x2; other accelerators blocked')
    deadline=min(PUSH_START+1800,PUSH_START+(7200-__PRIOR_AGG_SECONDS__)/allocated)
    if time.time()>=deadline:raise TimeoutError('Allocation budget expired before setup')
    def expire():
        try:save('timeout.json',{'status':'aborted','reason':'allocation deadline','kernel_session_id':session_id,'allocated_devices':allocated})
        finally:os._exit(124)
    timer=threading.Timer(deadline-time.time(),expire);timer.daemon=True;timer.start()
    save('hardware_allocation.json',{'allocated_devices':allocated,'device_inventory':hardware,'compute_device':0,
       'session_timeout_requested_seconds':1800,'aggregate_spend_ceiling_seconds':7200,'stop_new_runs_at_seconds':5760,
       'prior_aggregate_seconds':__PRIOR_AGG_SECONDS__,'conservative_push_start_unix':PUSH_START,'deadline_unix':deadline,
       'boot_time_not_observed_by_entry':True})
    raw=base64.b64decode('__BUNDLE__',validate=True)
    if hashlib.sha256(raw).hexdigest()!='__BUNDLE_SHA__':raise ValueError('FROZEN: upload bundle hash mismatch')
    source=OUT/'source';source.mkdir()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if z.testzip() is not None:raise ValueError('FROZEN: source ZIP CRC')
        for name in z.namelist():
            if name.startswith('/') or '..' in Path(name).parts:raise ValueError('Unsafe uploaded path')
        z.extractall(source)
    members=json.loads((source/'execution_inventory.json').read_text())
    for row in members:
        if sha(source/row['path'])!=row['sha256']:raise ValueError('FROZEN: uploaded member mismatch '+row['path'])
    save('uploaded_member_verification.json',{'status':'PASS','members':members})
    sys.path.insert(0,str(source))
    locked=json.loads((source/'audit/route_b_launch_20261026/package_lock.json').read_text())
    for row in locked['files']:
        if sha(source/row['path']).lower()!=row['sha256'].lower():raise ValueError('FROZEN: original70 source mismatch')
    installed=False
    if importlib.util.find_spec('SimpleITK') is None:
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--index-url','https://pypi.org/simple','SimpleITK==2.4.1'],check=True,timeout=120);installed=True
    # Fail fast on payload/geometry/checkpoint identity BEFORE any model load.
    from experiments.route_b_payload_v1.adapter import verify_payload_manifest
    from experiments.route_b_preflight_v1.inputs import prepared_cases
    folders=[p for p in Path('/kaggle/input').glob('**/sabs_CT_normalized') if (p/'image_8.nii').is_file()]
    if len(folders)!=1:raise RuntimeError('INFRASTRUCTURE: data mount missing or ambiguous')
    start=time.perf_counter()
    checked,rows,volumes=verify_payload_manifest(source/'protocol/pilot_manifest_v1.csv',folders[0],
        source/'audit/route_b_preflight_20261025/development_entitlement_repaired_v1.csv',source/'payload_identities.json')
    checked['seconds']=time.perf_counter()-start;checked['status']='PASS';save('gpu_payload_verification_before_models.json',checked)
    cv0=list(Path('/kaggle/input').glob('**/FSMIS_train_SABS_cv0/ckpt.pth'))
    if len(cv0)!=1:raise RuntimeError('INFRASTRUCTURE: cv0 mount missing or ambiguous')
    if sha(cv0[0])!='e3c9ccc47f74fa562e20c56ba11768aa8661b4d69fd170eb815943df3ff63913':raise ValueError('FROZEN: cv0 hash mismatch')
    sam_meta={'mounted_candidates':[],'download_ran':False}
    candidates=list(Path('/kaggle/input').glob('**/sam_vit_b*.pth'))
    for p in candidates:
        h=sha(p);sam_meta['mounted_candidates'].append({'path':str(p),'sha256':h})
        if h!='ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912':raise ValueError('FROZEN: mounted SAM-B hash mismatch')
    if candidates:sam=candidates[0]
    else:
        import urllib.request
        sam=Path('/tmp/route_b_sam_b_attempt_01.pth')
        if sam.exists():raise RuntimeError('INFRASTRUCTURE: unexpected existing SAM destination')
        created_sam=sam;start=time.perf_counter();sam_meta['download_ran']=True
        event('official_sam_b_download_started')
        with urllib.request.urlopen('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',timeout=60) as response,sam.open('xb') as f:
            if __import__('urllib.parse',fromlist=['urlparse']).urlparse(response.geturl()).hostname!='dl.fbaipublicfiles.com':raise RuntimeError('INFRASTRUCTURE: unapproved download redirect')
            total=0
            while True:
                b=response.read(8*1024*1024)
                if not b:break
                f.write(b);total+=len(b)
                if total%(64*1024*1024)==0:event('sam_b_download_progress',bytes=total)
        sam_meta['download_seconds']=time.perf_counter()-start
    sam_meta['actual_sha256']=sha(sam)
    save('sam_b_setup.json',sam_meta)
    if sam_meta['actual_sha256']!='ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912':raise ValueError('FROZEN: SAM-B hash mismatch')
    import numpy as np,torch,torchvision,SimpleITK,scipy,cv2
    from analysis.gate_a.runtime import load_fob,official_sam_b,seed_all
    from experiments.route_b_preflight_v1 import launch_pilot_v1 as launch
    from experiments.route_b_preflight_v1 import runner as core
    from experiments.route_b_payload_v1.persistence import persistence_wrapper,array_hashes
    seed_all(42);torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('HARDWARE: device0 visibility check')
    torch.cuda.reset_peak_memory_stats()
    save('environment.json',{'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,
        'torch_cuda':torch.version.cuda,'numpy':np.__version__,'SimpleITK':SimpleITK.__version__,'scipy':scipy.__version__,
        'cv2':cv2.__version__,'SimpleITK_install_ran':installed,'device_inventory':hardware,'allocated_devices':allocated,
        'visible_device_count':torch.cuda.device_count(),'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,
        'TF32_cudnn':torch.backends.cudnn.allow_tf32,'precision':'float32; no autocast; no batching change'})
    clock=launch.Clock(torch.cuda.synchronize)
    event('model_loading_started',kernel_session_id=session_id)
    model,fob_load=clock.call('FoB_setup_and_strict_loading',load_fob,'local',10,cv0[0],canonical_unused=True)
    predictor,sam_load=clock.call('SAM_setup_and_strict_loading',official_sam_b,sam)
    save('strict_loads.json',{'FoB':fob_load,'SAM_B':sam_load})
    backend=launch.MeasuredBackend(core.PilotBackend(model,predictor),clock)
    writer=core.RunWriter(OUT/'predictions')
    wrapped=persistence_wrapper(core.pilot_predict,OUT/'pre_evaluation_predictions',clock)
    original_prepared=iter(prepared_cases(rows,volumes))
    def cases():
        for i in range(8):
            name,legitimate,gt=clock.call('reading_and_preprocessing_next_case',next,original_prepared)
            event('case_started',case=name,patient=rows[i]['patient'],organ=rows[i]['organ'])
            yield name,legitimate,gt
            event('case_completed',case=name,actual_decodes=backend.case_decodes)
    setup_before_prediction_seconds=time.time()-FIRST
    status['real_model_forward_started']=True
    with patch.object(core,'pilot_predict',wrapped):result=launch.run_cases(cases(),backend,clock,writer)
    # Independent saved-array/hash and call-event agreement before declaring complete.
    case_results=[]
    for i in range(8):
        name='case'+str(i);record=json.loads((OUT/'predictions'/(name+'.json')).read_text())
        seal=json.loads((OUT/'pre_evaluation_predictions'/(name+'.seal.json')).read_text())
        if record['logical_requests']!=20 or record['configurations_decoded']>20 or record['api_decoder_invocations']!=record['configurations_decoded']:raise ValueError('FROZEN: per-case accounting discrepancy')
        if len([e for e in record['components'] if e['component']=='decoder_configuration_including_transfers'])!=record['configurations_decoded']:raise ValueError('FROZEN: decoder timing/call discrepancy')
        if sha(OUT/'pre_evaluation_predictions'/(name+'.npz'))!=seal['npz_sha256']:raise ValueError('FROZEN: persisted prediction file changed')
        with np.load(OUT/'predictions'/(name+'.npz'),allow_pickle=False) as z:
            if array_hashes({k:z[k] for k in z.files})!=seal['array_hashes']:raise ValueError('FROZEN: pre/post evaluation predictions differ')
        case_results.append({'case':name,'patient':rows[i]['patient'],'organ':rows[i]['organ'],
                             'class_id':int(rows[i]['class_id']),'role':rows[i]['role'],'record':record})
    if result['actual_decoder_configurations']>160:raise ValueError('FROZEN: aggregate configuration cap')
    result.update(setup_before_prediction_seconds=setup_before_prediction_seconds,peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
       process_wall_seconds=time.time()-FIRST,conservative_allocation_wall_seconds=time.time()-PUSH_START,
       allocated_devices=allocated,conservative_aggregate_gpu_seconds=allocated*(time.time()-PUSH_START),
       budget_definition='push-start through observation is conservative; entry misses pre-entry boot; final server termination recorded externally')
    save('runtime.json',result);save('case_results.json',case_results)
    status['status']='PILOT-COMPLETED';status['cases']=8;status['decoder_configurations']=result['actual_decoder_configurations'];code=0
except Exception:
    status['status']='PILOT-BLOCKED';status['error']=traceback.format_exc();event('attempt_blocked',error=status['error']);code=1
finally:
    if timer:timer.cancel()
    status['process_wall_seconds']=time.time()-FIRST
    status['temporary_download_created']=str(created_sam) if created_sam else None
    status['cleanup']='Not performed before verified output download; mounted data never altered'
    save('final_status.json',status)
    hashes=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file():hashes.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    save('output_hashes.json',hashes)
    event('attempt_finished',status=status['status'])
sys.exit(code)