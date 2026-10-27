"""Kaggle T4x2 orchestrator, bounded by aggregate allocated-GPU time."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata as md,importlib.util,json,os,subprocess,sys,time,urllib.request,zipfile
from pathlib import Path
import numpy as np
from .common import read_config,read_manifest,write_json,sha256,nifti_path

def monitor(processes,output,deadline):
    last_print=0
    while True:
        states=[p.poll() for p,log in processes]
        if all(v is not None for v in states):return states
        if time.time()>deadline:
            for p,log in processes:
                if p.poll() is None:p.terminate()
            for p,log in processes:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            raise TimeoutError("allocated GPU budget stop; partial commits retained")
        if time.time()-last_print>=30:
            for p,log in processes:
                tail=Path(log).read_text(errors="replace").splitlines()[-1:] if Path(log).exists() else []
                print(json.dumps({"pid":p.pid,"exit":p.poll(),"latest_progress":tail,"remaining_wall_seconds":round(deadline-time.time(),1)}),flush=True)
            last_print=time.time()
        time.sleep(5)

def child(command,log,device):
    env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(device);env["OMP_NUM_THREADS"]="2";env["MKL_NUM_THREADS"]="2"
    env["PYTHONUNBUFFERED"]="1"
    f=Path(log).open("x")
    p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env)
    f.close()
    return p,str(log)

def verify_data(folder,c,out):
    import SimpleITK as sitk
    checked=[]
    for stem,expected in c["data_semantic_sha256"].items():
        path=nifti_path(folder,stem);itk=sitk.ReadImage(str(path));arr=sitk.GetArrayFromImage(itk)
        if stem.startswith("image_"):
            raw=np.ascontiguousarray(arr.astype("<f8"))
            normalized=np.ascontiguousarray(((arr-arr.mean())/arr.std()).astype("<f4"))
            assert hashlib.sha256(normalized.tobytes()).hexdigest()==expected["normalized_fp32_sha256"],f"normalized inputs differ: {stem}"
        else:
            for old,new in [(200,1),(500,2),(600,3)]:arr[arr==old]=new
            raw=np.ascontiguousarray(arr.astype("<i2"))
        assert hashlib.sha256(raw.tobytes()).hexdigest()==expected["canonical_values_sha256"],f"native values differ: {stem}"
        for value,key in [(itk.GetSpacing(),"spacing"),(itk.GetOrigin(),"origin"),(itk.GetDirection(),"direction")]:
            np.testing.assert_allclose(value,expected[key],atol=1e-6,rtol=0)
        checked.append({"stem":stem,"native_file_sha256":sha256(path),"semantic_match":True})
        if len(checked)%10==0:print(f"data hash checks {len(checked)}/60",flush=True)
    write_json(out/"remote_data_verification.json",checked)

def main():
    p=argparse.ArgumentParser();p.add_argument("--lock-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    start=float(os.environ.get("ADAF_OB_ALLOC_START",time.time()))
    c=read_config(a.lock_dir);deadline=start+c["aggregate_gpu_budget_seconds"]/2
    import torch
    names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    assert len(names)==2 and all("T4" in n for n in names),f"Wrong hardware {names}"
    stages={"allocation_observation_start_epoch":start,"deadline_epoch":deadline,"device_names":names,"planned_aggregate_gpu_hours":c["aggregate_gpu_budget_seconds"]/3600}
    write_json(a.output/"hardware_budget.json",stages)
    print(json.dumps(stages),flush=True)
    if importlib.util.find_spec("SimpleITK") is None:
        print("Installing declared missing CPU reader SimpleITK==2.4.1; no torch upgrade",flush=True)
        subprocess.run([sys.executable,"-m","pip","install","--no-deps","SimpleITK==2.4.1"],check=True,timeout=120)
    inventory={d.metadata["Name"]:d.version for d in md.distributions() if d.metadata["Name"]}
    write_json(a.output/"environment_kaggle.json",{"python":sys.version,"packages":inventory,"gpu_names":names,"torch_cuda":torch.version.cuda,"precision":"float32 TF32 off; no autocast"})
    roots=list(Path("/kaggle/input").glob("**/sabs_CT_normalized"))
    folder=next(p for p in roots if (p/"image_0.nii").exists() or (p/"image_0.nii.gz").exists())
    verify_data(folder,c,a.output)
    cv0=list(Path("/kaggle/input").glob("**/FSMIS_train_SABS_cv0/ckpt.pth"))
    assert cv0,"FoB checkpoint source unavailable; no automatic alternate weights"
    checkpoint_root=cv0[0].parents[2]
    for fold in range(5):assert sha256(checkpoint_root/c["fob_relative_paths"][str(fold)])==c["fob_sha256"][str(fold)]
    sam_candidates=list(Path("/kaggle/input").glob("**/sam_vit_b*.pth"))
    sam=next((p for p in sam_candidates if sha256(p)==c["sam_b_sha256"]),None)
    if sam is None:
        sam=Path("/tmp/adafob_step8_sam_b.pth")
        assert not sam.exists()
        print("Fetching declared official SAM-B release; verifying full SHA256",flush=True)
        with urllib.request.urlopen(c["sam_b_official_url"],timeout=60) as response,sam.open("xb") as f:
            count=0
            while True:
                block=response.read(8*1024*1024)
                if not block:break
                f.write(block);count+=len(block)
                if count%(64*1024*1024)==0:print(f"SAM-B bytes {count}",flush=True)
    assert sha256(sam)==c["sam_b_sha256"],"SAM-B byte identity differs"
    args=["--lock-dir",str(a.lock_dir),"--data-dir",str(folder),"--checkpoint-root",str(checkpoint_root),"--sam-checkpoint",str(sam)]
    try:
        cmd=[sys.executable,"-u","-m","analysis.rebaseline_v1.pilot",*args,"--output",str(a.output/"pilot")]
        pilot=child(cmd,a.output/"pilot.log",0)
        assert monitor([pilot],a.output,deadline-60)==[0],"pilot execution failed; see pilot.log"
        result=json.loads((a.output/"pilot/pilot_summary.json").read_text())
        projected_total_gpu_seconds=2*(time.time()-start+result["projection_wall_seconds"])
        projection={"utc_epoch_before_full_run":time.time(),"elapsed_allocated_gpu_seconds":2*(time.time()-start),
                    "full_projection_gpu_seconds":2*result["projection_wall_seconds"],"projected_total_gpu_hours":projected_total_gpu_seconds/3600,
                    "limit_gpu_hours":c["aggregate_gpu_budget_seconds"]/3600,"within_authorized_budget":projected_total_gpu_seconds<=c["aggregate_gpu_budget_seconds"],
                    "storage_bytes_estimate":result["estimated_slice_archive_bytes"],"measured_shared_execution_valid":result["all_shared_separate_masks_equal"]}
        write_json(a.output/"projection_before_full_run.json",projection);print(json.dumps(projection),flush=True)
        if not projection["within_authorized_budget"]:
            write_json(a.output/"remote_status.json",{"status":"BLOCKED","reason":"T4 projection exceeds authorized aggregate budget; full run not launched"})
            return
        children=[]
        for worker in range(2):
            cmd=[sys.executable,"-u","-m","analysis.rebaseline_v1.inference",*args,"--output",str(a.output/"inference"/f"worker{worker}"),
                 "--worker",str(worker),"--deadline-epoch",str(deadline-60)]
            children.append(child(cmd,a.output/f"worker{worker}.log",worker))
        states=monitor(children,a.output,deadline-60)
        assert states==[0,0],f"worker exits {states}; partial predictions retained"
        completed=list((a.output/"inference").glob("worker*/episode_*/episode_complete.json"))
        assert len(completed)==120
        total_slices=sum(json.loads(p.read_text())["slices"] for p in completed)
        assert total_slices==c["total_slices"]
        write_json(a.output/"remote_status.json",{"status":"INFERENCE_COMPLETED","episodes":120,"new_arms":3,"posts":2,"slices":total_slices,
                                                  "completeness_cells":120*3*2,"analysis_not_yet_executed":True})
    except Exception as exc:
        write_json(a.output/"remote_failure.json",{"status":"FAIL","type":type(exc).__name__,"reason":str(exc)})
        raise
    finally:
        elapsed=time.time()-start
        write_json(a.output/"observed_gpu_cost_before_packaging.json",{"observed_allocation_wall_seconds":elapsed,"conservative_two_T4_gpu_hours":2*elapsed/3600,"limit_gpu_hours":2.5})
        print(f"completed observation wall={elapsed:.2f}s two_T4_GPU_hours={2*elapsed/3600:.6f}",flush=True)
if __name__=="__main__":main()
