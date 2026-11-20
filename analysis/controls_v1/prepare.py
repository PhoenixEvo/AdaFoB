"""Prepare baseline byte fingerprints and pre-inference committed provenance."""
from pathlib import Path
import argparse,json,shutil,subprocess
from datetime import datetime,timezone
import numpy as np,pandas as pd
from .core import baseline_digest,coordinate_digest,ARMS,BASE,ZERO
from analysis.gate_a.runtime import sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args()
 c=json.loads(Path('audit/step8_20261023T130640Z/protocol_lock/configuration.json').read_text())
 for name,h in c['code_sha256'].items():assert sha256(name)==h,name
 proof={};diag=pd.read_csv('audit/step8_20261023T130640Z/protocol_lock/diagnostic_manifest.csv')
 diagkeys=set(zip(diag.episode_id,diag.native_z));diagdir=a.run_dir/'baseline_diagnostic';diagdir.mkdir()
 for ep in sorted(a.baseline.glob('worker*/episode_*')):
  eid=int(ep.name.split('_')[1])
  for cp in sorted(ep.glob('slice_*.commit.json')):
   meta=json.loads(cp.read_text());file=ep/meta['file'];assert sha256(file)==meta['sha256']
   with np.load(file,allow_pickle=False) as d:
    rec=json.loads(str(d['record_json']));z=rec['native_z'];key=f'{eid}:{z}'
    arrays={BASE+'_'+s:d['R2_'+s] for s in ['raw','cc']};arrays.update({ZERO+'_'+s:d['R3_'+s] for s in ['raw','cc']})
    proof[key]={'masks':baseline_digest(arrays),'coordinates':coordinate_digest(d['negative_source_xy'],d['positive_source_xy'],d['negative_restored_xy'],d['positive_restored_xy'])}
    if (eid,z) in diagkeys:
     with (diagdir/(key.replace(':','_')+'.npz')).open('xb') as f:np.savez_compressed(f,**arrays)
   if len(proof)%500==0:print(f'baseline fingerprints {len(proof)}/4009',flush=True)
 assert len(proof)==4009 and len(list(diagdir.glob('*.npz')))==8
 with (a.run_dir/'step8_fingerprints.json').open('x') as f:json.dump(proof,f,separators=(',',':'))
 c.update(version='controls_v1',aggregate_gpu_budget_seconds=10800,notebook_session_timeout_seconds=5400,greedy_tolerance=1e-4,
          bootstrap_replicates=5000,pilot_repeats=3,projection_overhead_factor=1.25,projection_startup_seconds=180,
          gap_threshold_stop=0.5,gap_threshold_continue=0.25,residual_lower_bound_pp=1.0,arms=list(ARMS),random_seeds=list(range(42,47)),
          budgets=[0,2,4,6,10],locked_utc=datetime.now(timezone.utc).isoformat(),code_sha256={})
 paths=set(json.loads(Path('audit/step8_20261023T130640Z/protocol_lock/configuration.json').read_text())['code_sha256'])
 paths.update(p.as_posix() for p in Path('analysis/controls_v1').glob('*.py'));paths.update(['tests/test_step9_controls.py','experiments/run_step5_diagnostics.py','audit/oracle_definition.md'])
 c['code_sha256']={p:sha256(p) for p in sorted(paths)};c['step8_fingerprint_sha256']=sha256(a.run_dir/'step8_fingerprints.json')
 lock=a.run_dir/'protocol_lock';lock.mkdir()
 for name in ['paired_episode_manifest.csv','diagnostic_manifest.csv','worker_assignment.csv']:
  shutil.copyfile(Path('audit/step8_20261023T130640Z/protocol_lock')/name,lock/name)
 shutil.copyfile(a.run_dir/'step8_fingerprints.json',lock/'step8_fingerprints.json');shutil.copytree(diagdir,lock/'baseline_diagnostic')
 with (lock/'configuration.json').open('x') as f:json.dump(c,f,indent=2)
 shutil.copyfile('protocol/controls_v1.md',lock/'controls_v1.md')
 for path in paths:
  dest=lock/'source_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
 with (lock/'source_hashes.md').open('x') as f:
  f.write('Frozen before inference UTC '+c['locked_utc']+'\n\n| Source | SHA256 |\n|---|---|\n');f.write('\n'.join('| '+k+' | '+v+' |' for k,v in c['code_sha256'].items()))
 (lock/'.gitattributes').write_text('* -text\n')
 for cmd in [['git','init','--initial-branch=main'],['git','add','.'],['git','-c','user.name=AdaFoB Audit','-c','user.email=audit@localhost','commit','-m','Lock original-pool controls and corrected references before inference']]:subprocess.run(cmd,cwd=lock,check=True,capture_output=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=lock,text=True).strip()
 with (a.run_dir/'protocol_lock_commit.json').open('x') as f:json.dump({'commit':commit,'utc':datetime.now(timezone.utc).isoformat(),'inference_not_started':True,'protocol_sha256':sha256('protocol/controls_v1.md'),'source_count':len(paths)},f,indent=2)
 print(json.dumps({'protocol_commit':commit,'source_count':len(paths)}),flush=True)
if __name__=='__main__':main()