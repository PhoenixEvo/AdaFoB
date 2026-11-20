from pathlib import Path
import json,gc
from analysis.gate_a.runtime import load_fob,sha256,seed_all
import torch
run=Path("audit/step7_20261023_041838")
expected={r["path"].replace(chr(92),"/"):r["sha256"] for r in map(json.loads,Path("audit/step6_20261023/artifacts/hashes.jsonl").read_text().splitlines())}
rows=[]
seed_all(42)
for fold in range(5):
 p=Path(f"checkpoints/fob_checkpoints/exps_train_on_SABS_FSMIS_FoB/FSMIS_train_SABS_cv{fold}/ckpt.pth")
 actual=sha256(p)
 assert expected[p.as_posix()]==actual
 for impl,width in [("upstream",10),("local",10),("local",24)]:
  model,meta=load_fob(impl,width,p,canonical_unused=True)
  rows.append({"fold":fold,"hash_matches_recorded":True,"raw_strict_status":"FAIL" if meta["raw_strict_load_error"] else "PASS","schema_adapter_strict_status":"PASS",**meta})
  del model;gc.collect();torch.cuda.empty_cache()
 with (run/f"canonical_loads_through_cv{fold}.json").open("x") as f:json.dump(rows,f,indent=2)
 print(f"completed cv{fold}",flush=True)
with (run/"checkpoint_schema_verification.json").open("x") as f:json.dump(rows,f,indent=2)
