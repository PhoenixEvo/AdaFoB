"""Recheck historical hashes and preserve the exact Gate A source changes."""
from pathlib import Path
import json,subprocess,ast,zipfile
from datetime import datetime,timezone
from .runtime import sha256
run=Path("audit/step7_20261023_041838")
rows=[];changes=[]
for i,line in enumerate(Path("audit/step6_20261023/artifacts/hashes.jsonl").read_text().splitlines()):
 r=json.loads(line);p=Path(r["path"])
 actual=sha256(p)
 item={**r,"current_sha256":actual,"unchanged":actual==r["sha256"]}
 if not item["unchanged"]:
  assert p.as_posix() in ["experiments/run_step4_baselines.py","experiments/run_step5_diagnostics.py"],f"unexpected historical change {p}"
  original=run/"source_before"/p.name
  assert sha256(original)==r["sha256"]
  item["before_source_preserved_at"]=str(original)
  changes.append(item)
 rows.append(item)
 if (i+1)%50==0:print(f"historical hashes {i+1}",flush=True)
with (run/"historical_hash_recheck.json").open("x") as f:json.dump({"utc":datetime.now(timezone.utc).isoformat(),"files":len(rows),"unchanged":sum(r["unchanged"] for r in rows),"intentional_source_changes":changes,"rows":rows},f,indent=2)
for label in ["step4","step5"]:
 old=json.loads(Path(f"audit/step6_20261023/{label}/analysis.json").read_text())
 assert sha256(old["source"])==old["source_sha256"]
source_checks=[]
for name,fnname in [("run_step4_baselines.py","compute_metrics"),("run_step5_diagnostics.py","compute_volume_metrics")]:
 def dump(p):
  t=ast.parse(p.read_text(encoding="utf-8-sig"))
  return ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==fnname),include_attributes=False)
 assert dump(run/"source_before"/name)==dump(Path("experiments")/name)
 source_checks.append({"file":name,"production_metric_AST_unchanged":True})
(run/"metric_source_preservation.json").write_text(json.dumps(source_checks,indent=2))
files=list(Path("results/step5_diagnostics").rglob("*"))
inventory={"utc":datetime.now(timezone.utc).isoformat(),"saved_prediction_files":[str(p) for p in files if p.is_file() and p.suffix.lower() in [".npz",".npy",".nii",".gz",".pt",".pth"]],
           "archives":{str(p):zipfile.ZipFile(p).namelist() for p in files if p.is_file() and p.suffix==".zip"}}
(run/"step5_mask_inventory.json").write_text(json.dumps(inventory,indent=2))
for name,args in [("ending_status.txt",["git","status","--short"]),("ending_tracked.diff",["git","diff"]),("diff_check.txt",["git","diff","--check"])]:
 out=subprocess.run(args,text=True,capture_output=True)
 (run/name).write_text(out.stdout+out.stderr,encoding="utf-8")
 assert out.returncode==0
print(json.dumps({"historical_files":len(rows),"unchanged":sum(r["unchanged"] for r in rows),"intended_source_changes":len(changes),"old_result_CSV_hashes_match_step6":True,"metric_AST_unchanged":True}),flush=True)
