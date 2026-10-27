"""Create and commit the protocol snapshot before any inference/test execution."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil,subprocess
from datetime import datetime,timezone
import numpy as np,pandas as pd,SimpleITK as sitk

def digest(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--data-dir",type=Path,required=True);a=p.parse_args()
 lock=a.run_dir/"protocol_lock";lock.mkdir(exist_ok=False)
 paths=set(Path("analysis/rebaseline_v1").glob("*.py"))
 paths.update(Path("third_party/FoB_SAM").glob("models/*.py"))
 paths.update(Path("third_party/FoB_SAM_upstream").glob("models/*.py"))
 paths.update(Path("third_party/FoB_SAM/dataloaders").glob("*.py"))
 paths.update(Path("third_party/segment-anything/segment_anything").rglob("*.py"))
 paths.update(Path(p) for p in ["analysis/gate_a/__init__.py","analysis/gate_a/runtime.py","analysis/gate_a/coordinates.py","analysis/gate_a/measurements.py","analysis/patient_balanced_summary.py","models/postprocessing.py","experiments/run_step4_baselines.py","tests/test_step8_rebaseline.py"])
 hashes={p.as_posix():digest(p) for p in sorted(paths)}
 oldhash={r["path"].replace(chr(92),"/"):r["sha256"] for r in map(json.loads,Path("audit/step6_20261023/artifacts/hashes.jsonl").read_text().splitlines())}
 c={"version":"rebaseline_v1","locked_utc":datetime.now(timezone.utc).isoformat(),"starting_main_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "hardware":"two NVIDIA Tesla T4 GPUs only; no local GPU","aggregate_gpu_budget_seconds":9000,"notebook_session_timeout_seconds":4500,
    "seed":42,"support_parts":3,"support_patients":{str(i):f"image_{n}" for i,n in enumerate([2,8,14,20,25])},
    "total_slices":4009,"bootstrap_replicates":5000,"equivalence_style_margins":{"dice":0.5,"hd95":2.0},
    "margin_units":{"dice":"percentage points","hd95":"mm"},"margin_status":"working proposals accepted for this diagnostic audit, not clinical/confirmation margins",
    "parity_atol":1e-5,"dice_atol":1e-12,"hd95_atol_mm":1e-6,"pilot_repeats":3,"projection_overhead_factor":1.25,"projection_startup_seconds":180,
    "arms":{"R0":"saved B3B local width24 padded SPR, legacy coordinates","R1":"local width10 valid-channel SPR, legacy coordinates",
            "R2":"same width10; restore positive and negative xy pixel centres","R3":"same positives as R2; zero selected negative IDs","B4B":"saved legacy positive-only"},
    "code_sha256":hashes,"sam_b_sha256":"ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    "sam_b_official_url":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "fob_relative_paths":{str(i):f"exps_train_on_SABS_FSMIS_FoB/FSMIS_train_SABS_cv{i}/ckpt.pth" for i in range(5)},
    "fob_sha256":{},"data_semantic_sha256":{},"legacy_prediction_sha256":{}}
 for i in range(5):
  path=Path("checkpoints/fob_checkpoints")/c["fob_relative_paths"][str(i)]
  actual=digest(path);assert actual==oldhash[path.as_posix()];c["fob_sha256"][str(i)]=actual
 assert digest(Path("D:/results/checkpoints/sam_vit_b_01ec64.pth"))==c["sam_b_sha256"]
 for path in sorted(a.data_dir.glob("*.nii.gz")):
  stem=path.name.split(".nii")[0];itk=sitk.ReadImage(str(path));arr=sitk.GetArrayFromImage(itk)
  expected={"spacing":list(itk.GetSpacing()),"origin":list(itk.GetOrigin()),"direction":list(itk.GetDirection()),"shape_zyx":list(arr.shape),"file_sha256":digest(path)}
  if stem.startswith("image_"):
   canonical=np.ascontiguousarray(arr.astype("<f8"));norm=np.ascontiguousarray(((arr-arr.mean())/arr.std()).astype("<f4"))
   expected["normalized_fp32_sha256"]=hashlib.sha256(norm.tobytes()).hexdigest()
  elif stem.startswith("label_"):
   for old,new in [(200,1),(500,2),(600,3)]:arr[arr==old]=new
   canonical=np.ascontiguousarray(arr.astype("<i2"))
  else:continue
  expected["canonical_values_sha256"]=hashlib.sha256(canonical.tobytes()).hexdigest();c["data_semantic_sha256"][stem]=expected
 assert len(c["data_semantic_sha256"])==60
 for path in Path("results/step4_baselines/predictions").glob("*.npz"):
  actual=digest(path);assert actual==oldhash[path.as_posix()]
  c["legacy_prediction_sha256"][path.as_posix()]=actual
 assert len(c["legacy_prediction_sha256"])==120
 manifest=pd.read_csv("protocol/paired_episode_manifest.csv")
 counts=pd.read_csv("audit/step6_20261023/artifacts/active_slice_mapping.csv")
 assignment=manifest.merge(counts[["episode_id","active_slices"]],on="episode_id",validate="one_to_one")
 totals=[0,0];out=[]
 for fold in range(5):
  for row in assignment[assignment.fold==fold].sort_values(["active_slices","episode_id"],ascending=[False,True]).to_dict("records"):
   w=min(range(2),key=lambda i:(totals[i],i));totals[w]+=int(row["active_slices"])
   out.append({"episode_id":row["episode_id"],"fold":fold,"worker":w,"slices":row["active_slices"]})
 assert sum(totals)==c["total_slices"]
 c["worker_slices"]=totals
 (lock/"worker_assignment.csv").write_text(pd.DataFrame(out).sort_values("episode_id").to_csv(index=False))
 (lock/"configuration.json").write_text(json.dumps(c,indent=2))
 shutil.copyfile("protocol/paired_episode_manifest.csv",lock/"paired_episode_manifest.csv")
 shutil.copyfile("audit/step7_20261023_041838/protocol_lock/coordinate_diagnostic_manifest.csv",lock/"diagnostic_manifest.csv")
 timestamp=c["locked_utc"];head=c["starting_main_commit"]
 variants="\n".join(f"| {p} | {h} |" for p,h in hashes.items())
 ckpts="\n".join(f"| cv{k} | {v} |" for k,v in c["fob_sha256"].items())
 protocol=f"""# Corrected paired rebaseline v1

Locked UTC: {timestamp}. Main repository starting commit: {head}.
The user changed the requested hardware from local RTX4050 to Kaggle T4x2 before inference. No local GPU is used. The original 2.5 GPU-hour cap is retained conservatively as aggregate allocated T4 device-hours: both devices count throughout notebook execution, including idle/bootstrap time. Maximum server session:4500 seconds.

## Scope and provenance

Corrected rebaseline only. Existing CT outcomes are already inspected exploratory evidence. Step4/5 are not retrospectively preregistered. This timestamped plan governs only this batch, including the eight-slice cost pilot. Production FoB and its padded default are unchanged. All outputs use new paths; resumed slices require committed hash-valid files, and partial attempts are retained.

This protocol, the configuration, manifests, CPU validation tests and every executable source listed below are committed in an isolated local protocol repository before inference. Its commit SHA is recorded outside it in protocol_lock_commit.json. The main repository is dirty; its exact starting diff is retained. No main commit is falsely claimed.

## Arms and exact variants

All use fixed original ten-candidate FoB generation, the same cv0..cv4 checkpoint, frozen SAM-B R1 underlying token0, identical support partitions/positive generation/normalization/interpolation/threshold/CC and the120-episode manifest.

- R0: existing saved B3B masks, local width24 padded SPR, legacy256-grid coordinates applied on257.
- R1: local FewShotSeg max_points=10, allocator=None, use_skeleton=False, budget_Np=10. SPR operates on exactly ten valid channels; Step7 found exact upstream parity. This is the width10 alternative explicitly authorized by the user, not an unlabelled production patch.
- R2: R1 candidates and positives, mapped by x_native=(x_256+0.5)*257/256-0.5 and likewise y, both prompt labels.
- R3: same restored positive coordinates as R2, no negative IDs applied; retain/log the complete generated ten-candidate pool.
- B4B: existing saved legacy positive-only masks.

R1-R0 is the intended padding contrast; R2-R1 isolates coordinate restoration in the same current runtime. R2-R0 is combined, and R3-B4B checks restored positives. Runtime/backend drift in archived R0/B4B is a possible alternative explanation, not assumed absent.

Raw FoB strict loading still fails missing unused encoder.reduce2.weight. The explicit Step7 unused-layer schema deletion is used, followed by strict loading of all remaining keys. The redundant encoder bootstrap is disabled during construction only, and missing training-only InfoNCE is guarded against invocation. Checkpoint bytes and all adapters are hashed. No weights are trained.

| FoB checkpoint | SHA256 |
|---|---|
{ckpts}

SAM-B SHA256: {c["sam_b_sha256"]}. If not mounted, fetch only the declared official release URL and require this full hash. No substitute weights.

| Executable source / variant | SHA256 |
|---|---|
{variants}

The complete configuration pins source, checkpoint and120 saved prediction-archive hashes. Dataset .nii/.nii.gz wrappers may differ, so remote CT values, normalized float32 images, geometry and spacing must match the locked canonical hashes for all60 files. Query GT is only used for the inherited target-positive slice filtering and retrospective metrics, never for prompt selection or tuning.

## Endpoints, estimands and uncertainty

Primary endpoints: patient-balanced CC Dice and symmetric native-benchmark-grid3D HD95 in mm. These are the resampled257x257 benchmark volumes/contiguous target-positive slabs with verified physical spacing, not original-acquisition whole scans. Nativez indices are saved and preserved. Surface=foreground minus6-neighbour erosion; HD95=95th percentile pooled surface-to-surface distances in both directions.

Episode weighting is reported alongside. Patient balancing averages repeated evaluations within patient-organ, macro-averages four organs within patient, then averages25 patients. No120-independent-patient interpretation. Include raw/CC metrics, per-organ results, median/p90 HD95, recall and physical FP/FN volumes.

Independent implementation: direct voxel-count Dice and physical KD-tree surface distances. Compare against actual production metric AST/MedPy, tolerances1e-12 Dice /1e-6mm HD95. No empty/failed/outlier omission. Production one-empty diagonal fallback and independent infinity are explicitly different policies: preserve both and withhold matching/gate success if unresolved. Disjoint FP partition and removed TP/FP are as in Step7; no clinical causal labels.

Paired query-patient percentile bootstrap:5000 replicates, seed42, same sampled patients across methods. Four contrasts R1-R0,R2-R1,R2-R0,R3-B4B, both raw/CC and both estimands, overall/per-organ. Report exploratory pointwise95% intervals conditional on fixed supports/checkpoints; no training/support-selection uncertainty or simultaneous/multiplicity-controlled interval claim.

## Equivalence-style working proposals

Use the user's default proposals: +/-0.5 percentage points Dice and +/-2mm HD95. They are pragmatic diagnostic working margins, not clinical non-inferiority margins or submission thresholds. No changes are made. Any future change requires a timestamped protocol amendment before comparison results are examined.

For the legacy-matching decision use the patient-balanced R2-R0 CC interval on each endpoint. Matching is established only when each entire95% interval lies strictly inside its margin. Point estimates inside a margin, nonsignificance or a boundary-crossing interval do not establish equivalence.

## Measurement, speed and compute gate

Kaggle T4x2 only, float32/autocast off/TF32 off. Same resolution and precision in every timing row. No RTX4050 timings enter the T4 table. Use the eight previously committed diagnostic slices (two exploratory patients/four organs), one warm-up per slice/strategy and three measured repeats, CUDA synchronization.

Measure FoB generation, SAM image-encoding API (including preprocessing/transfers), decoder API per call (including prompt encoder/postprocessing/transfers), CC, serialization/write, total shared and separate passes, peak allocated VRAM and actual bytes. Validate shared-versus-separate logits/quality within1e-5 absolute and final masks exactly, without GT-based choices. Do not infer point-count latency savings.

One FoB generation and one image embedding per slice are shared across three serial decoder calls. Two independent worker processes receive disjoint episodes, balanced by already-known active-slice counts only; worker0={totals[0]} slices, worker1={totals[1]}. Folds and episodes are sequential within each worker. This parallelization is an explicit hardware optimization requested by the user, replacing global fold serialization without changing episodes.

Projection fixed before timing: p90(shared slice compute + serialization/write)*4009/2*1.25 plus180 seconds startup allowance, then add observed elapsed notebook allocation. Count two GPUs for the full observed notebook wall time. If projected aggregate exceeds2.5 GPU-hours, save pilot evidence and stop without full inference. Runtime deadline reserves60 seconds for termination/packaging. No concurrent job on the timing GPU. No controls in this batch.

Save per-slice raw/CC for all R1/R2/R3; full candidate/positive coordinates in both grids; immutable generation/selected IDs; actual counts, continuous/rounded out-of-bounds counters, hook-counted actual decoder calls, failures, seed, support/nativez and runtime. Exclusive files and checksum commit records support resumption. Assert120x3x2 new episode-arm-post cells,4009x3 slice-arm records, and all legacy references. Failed slices are retained and prevent an unqualified gate pass.

## Fixed decision table

| Condition on primary patient-balanced R2-R0 CC intervals | Locked interpretation/action |
|---|---|
| A: both whole intervals inside their respective margins | Legacy conclusions retained only as exploratory; next separate batch may run simple controls on corrected R2 |
| B: either interval not wholly inside its margin | Step4/5 remain legacy-pipeline comparisons; all controls must be rerun on R2 |
| B with interval crossing margin | Matching not established, not proof of a material difference |
| Incomplete/failed inputs/inference/reference agreement | Corrected baseline gate FAIL or BLOCKED; report defect, do not run controls |

Gate PASS requires completed input/code/checkpoint checks, all episodes and mask cells, actual call/count records, no unresolved failure/nonfinite measurement and production/reference agreement. A changed corrected score alone is not a pipeline defect. A/B and padding/coordinate contributions must be reported separately.

Final report reports exactly one corrected-baseline gate PASS/FAIL/BLOCKED and one recommendation: proceed to separate simple controls if integrity passes; otherwise fix the concrete defect, pivot only if supported, or stop. No controls, selector/response prototype, relocation policy, adaptation, MRI or manuscript work in this batch.
"""
 with Path("protocol/rebaseline_v1.md").open("x",encoding="utf-8") as f:f.write(protocol)
 shutil.copyfile("protocol/rebaseline_v1.md",lock/"rebaseline_v1.md")
 snapshot=lock/"source_snapshot"
 for path in paths:
  target=snapshot/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
 subprocess.run(["git","init","--initial-branch=main"],cwd=lock,check=True,capture_output=True)
 subprocess.run(["git","add","."],cwd=lock,check=True,capture_output=True)
 subprocess.run(["git","-c","user.name=AdaFoB Research Audit","-c","user.email=audit@localhost","commit","-m","Lock corrected paired rebaseline before inference"],cwd=lock,check=True,capture_output=True)
 commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=lock,text=True).strip()
 proof={"commit":commit,"locked_utc":timestamp,"recorded_utc":datetime.now(timezone.utc).isoformat(),"main_head":c["starting_main_commit"],
        "protocol_sha256":digest("protocol/rebaseline_v1.md"),"configuration_sha256":digest(lock/"configuration.json"),"worker_slices":totals,"inference_has_not_started":True}
 with (a.run_dir/"protocol_lock_commit.json").open("x") as f:json.dump(proof,f,indent=2)
 print(json.dumps(proof),flush=True)

if __name__=="__main__":main()
