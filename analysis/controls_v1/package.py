"""Package frozen source plus mask fingerprints, no credentials or new images."""
import argparse,base64,hashlib,io,json,zipfile
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--kernel-id',required=True);a=p.parse_args()
 lock=a.run_dir/'protocol_lock';c=json.loads((lock/'configuration.json').read_text());bundle=io.BytesIO()
 with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for name,h in c['code_sha256'].items():
   data=(lock/'source_snapshot'/name).read_bytes();assert hashlib.sha256(data).hexdigest()==h;z.writestr(name,data)
  for name in ['configuration.json','paired_episode_manifest.csv','diagnostic_manifest.csv','worker_assignment.csv','controls_v1.md','source_hashes.md','step8_fingerprints.json']:
   z.writestr('protocol_lock/'+name,(lock/name).read_bytes())
  for path in (lock/'baseline_diagnostic').glob('*.npz'):z.write(path,'protocol_lock/baseline_diagnostic/'+path.name)
  z.write(a.run_dir/'protocol_lock_commit.json','protocol_lock/protocol_lock_commit.json')
 payload=base64.b64encode(bundle.getvalue()).decode();out=a.run_dir/'submission';out.mkdir()
 script=f'''import os,time
os.environ['ADAF_OB_ALLOC_START']=str(time.time())
import base64,io,json,pathlib,subprocess,sys,zipfile
root=pathlib.Path('/kaggle/working/step9_locked_source');root.mkdir(exist_ok=False)
payload=base64.b64decode('{payload}')
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
 for member in archive.namelist():assert (root/member).resolve().is_relative_to(root.resolve())
 archive.extractall(root)
os.chdir(root)
artifacts=pathlib.Path('/kaggle/working/step9_artifacts')
command=[sys.executable,'-u','-m','analysis.controls_v1.remote','--lock-dir',str(root/'protocol_lock'),'--output',str(artifacts)]
print('Frozen Step9 T4x2, aggregate cap3.0 GPU-hours; full run only after measured gate',flush=True)
completed=subprocess.run(command)
with zipfile.ZipFile('/kaggle/working/step9_outputs.zip','x',compression=zipfile.ZIP_DEFLATED) as output:
 if artifacts.exists():
  for path in artifacts.rglob('*'):
   if path.is_file():output.write(path,path.relative_to(artifacts))
elapsed=time.time()-float(os.environ['ADAF_OB_ALLOC_START'])
cost={{'entry_exit_code':completed.returncode,'observed_notebook_wall_seconds':elapsed,'two_T4_allocated_gpu_hours':2*elapsed/3600,'source_bundle_sha256':'{hashlib.sha256(bundle.getvalue()).hexdigest()}'}}
with pathlib.Path('/kaggle/working/step9_entry_status.json').open('x') as f:json.dump(cost,f,indent=2)
print(json.dumps(cost),flush=True)
if completed.returncode:raise RuntimeError('Remote stage failed; preserve partial outputs and inspect logs')
'''
 with (out/'step9_entry.py').open('x',encoding='utf8',newline='\n') as f:f.write(script)
 m={'id':a.kernel_id,'title':'AdaFoB Step9 Original Pool Controls T4 20261024','code_file':'step9_entry.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':['nhatphatnguyen/adafob-env'],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
 with (out/'kernel-metadata.json').open('x') as f:json.dump(m,f,indent=2)
 print(json.dumps({'source_zip_bytes':len(bundle.getvalue()),'entry_bytes':len(script.encode()),'kernel_id':a.kernel_id}),flush=True)
if __name__=='__main__':main()