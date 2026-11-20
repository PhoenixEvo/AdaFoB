"""Create a batch-only ledger from JUnit and explicit integration results."""
from pathlib import Path
import csv,json,xml.etree.ElementTree as ET
from datetime import datetime,timezone
run=Path("audit/step7_20261023_041838")
rows=[]
env="CPython3.11.9; torch2.8.0+cu128 torchvision0.23.0+cu128 scipy1.15.2 pytest8.4.2 SITK2.4.1 OpenCV4.10.0.84 MedPy0.5.2; audit/environment_gpu.json"
def add(name,command,status,log,reason,category="Fast CPU",hardware="CPU",required="data/source dependencies as in command"):
 rows.append({"test_name":name,"command":command,"environment":env,"category":category,"hardware":hardware,"required_hardware_checkpoints":required,"status":status,"log_path":str(log),"reason_and_consequence":reason,"recorded_utc":datetime.now(timezone.utc).isoformat()})
for xml,job in [("cpu_tests.xml","cpu_existing_tests"),("active_completeness.xml","active_completeness_tests"),("logging_tests.xml","logging_integration_tests"),("remaining_cpu.xml","remaining_cpu_tests")]:
 meta=json.loads((run/(job+".exit.json")).read_text())
 command=meta.get("args",[])
 if not command:
  command="see "+str(run/(job+".log"))+"; initial full CPU suite"
 command=str(command)
 for t in ET.parse(run/xml).iter("testcase"):
  status="FAIL" if t.find("failure") is not None or t.find("error") is not None else "SKIPPED" if t.find("skipped") is not None else "PASS"
  add(t.attrib.get("classname","")+"::"+t.attrib["name"],command,status,run/(job+".log"),"Executed this batch; JUnit "+str(run/xml)+"; isolated tests do not certify pretrained inference")
for stem in ["parity_width10","parity_width24","parity_width24_validring","parity_width24_trimmed"]:
 meta=json.loads((run/(stem+".exit.json")).read_text());result=json.loads((run/(stem+".json")).read_text())
 add(stem,str(meta.get("args",meta.get("command","UNVERIFIED command schema; see result/log"))),result["numerical_parity"],run/(stem+".log"),
     "Actual upstream/local coordinates plus shared SAM R1 logits, quality, masks; declared tolerance; raw strict FoB load separately FAIL; explicit unused-layer schema adapter",
     "GPU integration","RTX4050 Laptop","cv0 FoB and SAM-B")
strict=json.loads((run/"checkpoint_schema_verification.json").read_text())
for r in strict:
 label=f"cv{r['fold']}_{r['implementation']}_width{r['width']}"
 add("raw_strict_"+label,"python -u -m analysis.gate_a.verify_checkpoint_schemas",r["raw_strict_status"],run/"checkpoint_schema_verification.log",
     r["raw_strict_load_error"] or "strict load executed","GPU integration","RTX4050 Laptop","specified FoB checkpoint SHA in checkpoint_schema_verification.json")
 add("explicit_schema_strict_"+label,"python -u -m analysis.gate_a.verify_checkpoint_schemas",r["schema_adapter_strict_status"],run/"checkpoint_schema_verification.log",
     "Only unused reduce2 removed; original raw strict failure retained; zero missing/unexpected keys after explicit adapter; width24 pads three listed tensors",
     "GPU integration","RTX4050 Laptop","specified FoB checkpoint SHA")
sam=json.loads((run/"sam_b_strict_load.json").read_text())
add("SAM_B_strict_and_recorded_hash","python -u "+str(run/"environment_strict_check.py")+" "+str(run),"PASS",run/"environment_strict_loads.log","Executed; zero missing/unexpected keys; SHA matches Step6","GPU integration","RTX4050 Laptop","SAM-B ec2df627... recorded full hash")
coord=json.loads((run/"coordinate_diagnostic/summary.json").read_text())
add("coordinate_sensitivity_execution","python -u -m analysis.gate_a.coordinate_sensitivity --manifest "+str(run/"protocol_lock/coordinate_diagnostic_manifest.csv")+" --out-dir "+str(run/"coordinate_diagnostic")+" --data-dir D:/sabs_CT_normalized --sam-checkpoint D:/results/checkpoints/sam_vit_b_01ec64.pth","PASS",run/"coordinate_sensitivity.log","Eight locked slices and all declared arms executed, 36 actual decoder evaluations incl timing","Bounded diagnostic","RTX4050 Laptop","cv0/cv1 FoB, SAM-B and four authorized CT volumes")
add("legacy_grid_negligible_under_locked_thresholds","same coordinate diagnostic; threshold fixed in committed protocol","FAIL",run/"coordinate_diagnostic/coordinate_sensitivity.csv","One CC slice crosses fixed Dice and HD95 sensitivity thresholds; cohort effects require matched rebaseline","Bounded diagnostic","RTX4050 Laptop")
full=json.loads((run/"metrics_full/summary.json").read_text())
add("saved_masks_production_vs_independent_reference","python -u -m analysis.gate_a.measurements --data-dir D:/sabs_CT_normalized --out-dir "+str(run/"metrics_full")+" --workers 2","PASS",run/"saved_mask_full.log","120 episodes x five arms x raw/CC; all numerical/partition assertions executed; "+str(full),"Saved-mask analysis","CPU; two processes","240 existing NPZ archives and physical CT labels")
for name,reason in [("legacy_GPU_shape_repeatability_test","Not run: random-weight single-implementation test is not upstream parity; constructor requests uncached bootstrap; real checkpoint parity executed separately"),
                    ("legacy_GPU_allocator_smoke","Not run: selection/allocator GPU smoke outside this Gate A numerical study; constructor bootstrap unavailable"),
                    ("legacy_GPU_query_label_invariance","Not run: original test compares two nonempty labels; constructor bootstrap unavailable; no claim of generic invariance"),
                    ("SAM_H_official_byte_identity","UNVERIFIED: actual H checkpoint and official release bytes unavailable in inspected paths; no download"),
                    ("Step5_subset_mask_independent_recompute","UNVERIFIED: subset/random/GT-reference masks absent even in artifacts zip; CSV-only summaries labelled"),
                    ("full_120_episode_new_inference","Not run: material sensitivity requires reviewed resource estimate; batch stops here"),
                    ("cached_embedding_or_cold_start_timing","Not measured: only explicitly described resident-model uncached-embedding timing executed")]:
 add(name,"NOT EXECUTED", "SKIPPED",run/"step5_mask_inventory.json" if "Step5" in name else "reports/step7_gate_a.md",reason,"Deferred/blocked","as stated")
add("initial_silent_torch_import_probe","UNVERIFIED exact command; initial torch import probe","TIMED OUT",run/"execution_incidents.md","25-second initial probe timed out; subsequent staged GPU probe executed successfully; no test pass inferred","Environment","CPU/GPU import")
add("exec_command_driver","functions.exec / exec_command process startup","FAIL",run/"execution_incidents.md","CreateProcessSecurityEnvironment HRESULT0x80070020 before shell execution; subsequent node-host child_process recovery documented","Environment","Windows")
add("initial_upstream_constructor_preflight","python -u "+str(run/"environment_strict_check.py")+" "+str(run),"FAIL",run/"environment_strict_loads.log","First upstream load attempts failed at missing training-only info_nce import; later guarded inference adapter and actual raw strict checks recorded separately","Environment","RTX4050 Laptop")
out=Path("audit/test_status_v2.csv")
with out.open("x",newline="",encoding="utf-8") as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({"rows":len(rows),"executed_pytest_testcases":sum(1 for r in rows if "JUnit" in r["reason_and_consequence"]),"status_counts":{s:sum(r["status"]==s for r in rows) for s in ["PASS","FAIL","SKIPPED","TIMED OUT"]}}),flush=True)
