"""Fixed original-pool inference; no selection. All coordinates are xy pixels."""
from __future__ import annotations
import gc,io,json,time
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from analysis.gate_a.runtime import load_fob,official_sam_b,seed_all,sha256
from analysis.gate_a.coordinates import restore_xy

ARMS=("R1","R2","R3")
ORGANS={"LIVER":6,"RK":2,"LK":3,"SPLEEN":1}

def read_config(lock_dir):
    path=Path(lock_dir)/"configuration.json"
    c=json.loads(path.read_text())
    root=Path(__file__).resolve().parents[2]
    for name,expected in c["code_sha256"].items():
        assert sha256(root/name)==expected,f"Frozen code changed: {name}"
    return c

def write_json(path,value):
    with Path(path).open("x",encoding="utf-8") as f:json.dump(value,f,indent=2,allow_nan=False)

def read_manifest(path):
    df=pd.read_csv(path)
    if "patient_id" not in df:
        df["patient_id"]=df.vol_id.str.replace(chr(92),"/",regex=False).str.split("/").str[-1].str.split(".nii").str[0]
    assert len(df)==120 and df.episode_id.nunique()==120
    assert df.groupby("fold").size().to_dict()=={i:24 for i in range(5)}
    assert df.patient_id.nunique()==25
    return df.sort_values(["fold","episode_id"])

@lru_cache(maxsize=2)
def normalized_volume(path):
    import SimpleITK as sitk
    itk=sitk.ReadImage(str(path));v=sitk.GetArrayFromImage(itk)
    norm=((v-v.mean())/v.std()).astype(np.float32)
    assert norm.shape[-2:]==(257,257)
    return norm,np.asarray(itk.GetSpacing())

@lru_cache(maxsize=2)
def label_volume(path):
    import SimpleITK as sitk
    itk=sitk.ReadImage(str(path));v=sitk.GetArrayFromImage(itk)
    for old,new in [(200,1),(500,2),(600,3)]:v[v==old]=new
    return v,np.asarray(itk.GetSpacing())

def nifti_path(folder,stem):
    for suffix in [".nii.gz",".nii"]:
        p=Path(folder)/(stem+suffix)
        if p.is_file():return p
    raise FileNotFoundError(stem)

def tracked_sam(path):
    predictor,meta=official_sam_b(path)
    predictor.audit_decoder_calls=0
    predictor.audit_encoder_calls=0
    def decoder_count(module,inputs):predictor.audit_decoder_calls+=1
    def encoder_count(module,inputs):predictor.audit_encoder_calls+=1
    predictor.model.mask_decoder.register_forward_pre_hook(decoder_count)
    predictor.model.image_encoder.register_forward_pre_hook(encoder_count)
    return predictor,meta

def episode_inputs(row,data_dir,c):
    data_dir=Path(data_dir);label=ORGANS[row["organ"]]
    image,spacing=normalized_volume(str(nifti_path(data_dir,row["patient_id"])))
    qlabel,label_spacing=label_volume(str(nifti_path(data_dir,row["patient_id"].replace("image_","label_"))))
    np.testing.assert_allclose(spacing,label_spacing,atol=1e-6,rtol=0)
    np.testing.assert_allclose(spacing,[row["spacing_x"],row["spacing_y"],row["spacing_z"]],atol=1e-6,rtol=0)
    native_z=np.flatnonzero((qlabel==label).any(axis=(1,2)))
    assert len(native_z)>0 and np.all(np.diff(native_z)==1)
    support=c["support_patients"][str(int(row["fold"]))]
    simage,_=normalized_volume(str(nifti_path(data_dir,support)))
    slabel,_=label_volume(str(nifti_path(data_dir,support.replace("image_","label_"))))
    sz=np.flatnonzero((slabel==label).any(axis=(1,2)))
    n=c["support_parts"]
    pcts=np.array([1/(2*n)+(1-1/n)/(n-1)*i for i in range(n)])
    support_z=sz[(pcts*len(sz)).astype(int)]
    cuts=np.linspace(0,len(native_z),n+1).astype(int)
    for j,z in enumerate(native_z):
        part=next(k for k in range(n) if cuts[k]<=j<cuts[k+1])
        yield {"episode_id":int(row["episode_id"]),"fold":int(row["fold"]),"patient_id":row["patient_id"],"organ":row["organ"],
               "compact_z":j,"native_z":int(z),"active_slices":len(native_z),"support_patient":support,"support_part":part,
               "support_native_z":int(support_z[part]),"spacing_xyz":spacing,
               "query":np.repeat(image[z][None,None],3,axis=1),
               "supp":np.repeat(simage[support_z[part]][None,None],3,axis=1),
               "support_mask":(slabel[support_z[part]]==label)[None].astype(np.float32),
               "gt":qlabel[z]==label,"other":(qlabel[z]>0)&(qlabel[z]!=label)}

def generate(model,case):
    import torch
    if case["supp"].max()==0:
        return np.zeros((0,2),np.float32),np.zeros((0,2),np.float32)
    supp=torch.from_numpy(case["supp"]).cuda()
    mask=torch.from_numpy(case["support_mask"]).cuda()
    qry=torch.from_numpy(case["query"]).cuda()
    dummy=torch.ones((1,*qry.shape[-2:]),dtype=torch.long,device="cuda")
    with torch.inference_mode():
        out=model([[supp]],[[mask]],[qry],dummy,train=False,use_skeleton=False,budget_Np=10)
    return np.asarray(out[0]).reshape(-1,2)[:10],np.asarray(out[1]).reshape(-1,2)[:10]

def rgb_image(case):
    rgb=case["query"][0].transpose(1,2,0)
    return ((rgb-rgb.min())/(rgb.max()-rgb.min()+1e-8)*255).astype(np.uint8)

def timed(fn):
    import torch
    torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize()
    return result,time.perf_counter()-start

def configurations(neg,pos):
    nr,pr=restore_xy(neg,(256,256),(257,257)),restore_xy(pos,(256,256),(257,257))
    return {"R1":(neg,pos),"R2":(nr,pr),"R3":(nr[:0],pr)},nr,pr

def predict(predictor,neg,pos):
    import torch
    points=np.concatenate([pos,neg]);labels=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    if not len(points):
        return (np.zeros((257,257),bool),np.zeros((1,257,257),np.float32),np.zeros(1,np.float32),0)
    with torch.inference_mode():
        logits,quality,_=predictor.predict(point_coords=points,point_labels=labels,multimask_output=False,return_logits=True)
    return logits[0]>predictor.model.mask_threshold,logits,quality,1

def cc_mask(raw,pos):
    from models.postprocessing import filter_connected_components
    return filter_connected_components(raw.astype(np.uint8),pos).astype(bool)

def oob(points,hw=(257,257)):
    if not len(points):return {"continuous":0,"rounded":0,"total":0}
    rounded=np.rint(points)
    upper=np.array([hw[1]-1,hw[0]-1])
    return {"continuous":int(np.any((points<0)|(points>upper),axis=1).sum()),
            "rounded":int(np.any((rounded<0)|(rounded>upper),axis=1).sum()),"total":len(points)}

def run_shared(model,predictor,case,capture_logits=False):
    started=time.perf_counter();failures=[]
    stage={};arrays={"gt":case["gt"]};fob_start=time.perf_counter();encoder_start=predictor.audit_encoder_calls
    try:
        (neg,pos),stage["fob_seconds"]=timed(lambda:generate(model,case))
        if len(neg)!=10:failures.append("emitted_pool_not_ten")
    except Exception as exc:
        failures.append("FoB:"+type(exc).__name__+":"+str(exc))
        neg,pos=np.empty((0,2),np.float32),np.empty((0,2),np.float32);stage["fob_seconds"]=time.perf_counter()-fob_start
    configs,nr,pr=configurations(neg,pos)
    rgb,stage["image_conversion_seconds"]=timed(lambda:rgb_image(case))
    _,stage["encoder_api_seconds"]=timed(lambda:predictor.set_image(rgb))
    arm_records={};logits_by_arm={}
    for arm,(n,p) in configs.items():
        decoder_start=predictor.audit_decoder_calls
        arm_start=time.perf_counter()
        try:
            (raw,logits,quality,calls),dt=timed(lambda:predict(predictor,n,p))
            cc,ct=timed(lambda:cc_mask(raw,p))
            event={"decoder_seconds":dt,"cc_seconds":ct,"decoder_calls":predictor.audit_decoder_calls-decoder_start,"failure":None}
            if capture_logits:logits_by_arm[arm]=(logits.copy(),quality.copy())
        except Exception as exc:
            failures.append(arm+":"+type(exc).__name__+":"+str(exc))
            raw=cc=np.zeros((257,257),bool)
            event={"decoder_seconds":time.perf_counter()-arm_start,"cc_seconds":None,"decoder_calls":predictor.audit_decoder_calls-decoder_start,"failure":failures[-1]}
        arrays[arm+"_raw"]=raw;arrays[arm+"_cc"]=cc
        event.update(positive_count=len(p),negative_count=len(n),total_count=len(n)+len(p),
                     selected_ids=list(range(len(n))),positive_oob=oob(p),negative_oob=oob(n))
        arm_records[arm]=event
    arrays.update(negative_source_xy=neg,negative_restored_xy=nr,positive_source_xy=pos,positive_restored_xy=pr)
    record={k:v for k,v in case.items() if k not in ["query","supp","support_mask","gt","other","spacing_xyz"]}
    record.update(spacing_xyz=case["spacing_xyz"].tolist(),seed=42,precision="float32; TF32 off",
                  negative_dtype=str(neg.dtype),positive_dtype=str(pos.dtype),generation_order_ids=list(range(len(neg))),
                  source_grid_hw=[256,256],native_grid_hw=[257,257],arms=arm_records,failures=failures,
                  decoder_calls=sum(x["decoder_calls"] for x in arm_records.values()),image_encoder_calls=predictor.audit_encoder_calls-encoder_start,
                  runtime={**stage,"slice_compute_seconds":time.perf_counter()-started})
    return arrays,record,logits_by_arm

def serialized_slice(arrays,record):
    stream=io.BytesIO()
    np.savez_compressed(stream,**arrays,record_json=np.asarray(json.dumps(record,separators=(",",":"),allow_nan=False)))
    return stream.getvalue()

def load_models(c,checkpoint_root,sam_checkpoint,fold):
    import torch
    path=Path(checkpoint_root)/c["fob_relative_paths"][str(fold)]
    assert sha256(path)==c["fob_sha256"][str(fold)]
    model,meta=load_fob("local",10,path,canonical_unused=True)
    assert not meta["missing_keys"] and not meta["unexpected_keys"] and not meta["shape_adaptations"]
    return model,meta
