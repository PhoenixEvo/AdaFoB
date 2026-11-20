"""Locked eight-slice diagnostic only. All original ten negatives, no selection."""
from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from .runtime import load_fob,official_sam_b,seed_all,sha256
from .coordinates import restore_xy
from .candidate_records import candidate_record,write_records
from .measurements import independent_surfaces,surface

def forward(model,case,implementation):
    import torch
    supp=torch.from_numpy(case["supp"]).cuda()
    mask=torch.from_numpy(case["support_mask"]).cuda()
    qry=torch.from_numpy(case["query"]).cuda()
    dummy=torch.ones((1,257,257),dtype=torch.long,device="cuda")
    kwargs={} if implementation=="upstream" else {"use_skeleton":False,"budget_Np":10}
    with torch.inference_mode():
        out=model([[supp]],[[mask]],[qry],dummy,train=False,**kwargs)
    return np.asarray(out[0]).reshape(-1,2),np.asarray(out[1]).reshape(-1,2)

def decode(predictor,neg,pos):
    import torch
    from models.postprocessing import filter_connected_components
    pts=np.concatenate([pos,neg]);labels=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    with torch.inference_mode():
        logits,quality,_=predictor.predict(point_coords=pts,point_labels=labels,multimask_output=False,return_logits=True)
    raw=logits[0]>predictor.model.mask_threshold
    cc=filter_connected_components(raw.astype(np.uint8),pos)>0
    return raw,cc,float(quality[0])

def metrics(mask,gt,spacing):
    denominator=int(mask.sum()+gt.sum())
    return {"dice":2*int((mask&gt).sum())/denominator if denominator else 1.,
            "slice_surface_hd95_mm":independent_surfaces(mask,gt,spacing)[0]}

def boundary_change(mask,before,after,spacing):
    s=surface(mask)
    if not s.any():return {"boundary_status":"empty; undefined"}
    distance=ndimage.distance_transform_edt(~s,sampling=spacing)
    signed=distance*np.where(mask,-1,1)
    def sample(a):
        return ndimage.map_coordinates(signed,[a[:,1],a[:,0]],order=1,mode="nearest")
    b,a=sample(before),sample(after)
    return {"fraction_negative_boundary_distance_changed":float(np.mean(np.abs(a-b)>1e-8)),
            "fraction_negative_boundary_side_changed":float(np.mean((a<0)!=(b<0))),
            "fraction_negative_within_1mm_changed":float(np.mean((np.abs(a)<=1)!=(np.abs(b)<=1))),
            "mean_absolute_boundary_distance_change_mm":float(np.mean(np.abs(a-b))),
            "negative_out_of_bounds_before":int(np.any((before<0)|(before>256),axis=1).sum()),
            "negative_out_of_bounds_after":int(np.any((after<0)|(after>256),axis=1).sum())}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--data-dir",type=Path,required=True);p.add_argument("--sam-checkpoint",type=Path,required=True)
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=False)
    import torch
    seed_all(42)
    manifest=pd.read_csv(a.manifest)
    assert len(manifest)<=8
    cases=[]
    # Match historical whole-volume z-score before slicing and RGB replication.
    for row in manifest.to_dict("records"):
        imgs={}
        for role,patient,z in [("query",row["patient_id"],row["native_z"]),("supp",row["support_patient"],row["support_native_z"])]:
            itk=sitk.ReadImage(str(a.data_dir/(patient+".nii.gz")))
            volume=sitk.GetArrayFromImage(itk)
            im=((volume-volume.mean())/volume.std())[int(z)].astype(np.float32)
            imgs[role]=np.repeat(im[None,None],3,axis=1)
            if role=="query":imgs["spacing"]=np.asarray(itk.GetSpacing())[1::-1]
        qlabel=sitk.GetArrayFromImage(sitk.ReadImage(str(a.data_dir/(row["patient_id"].replace("image_","label_")+".nii.gz"))))
        slabel=sitk.GetArrayFromImage(sitk.ReadImage(str(a.data_dir/(row["support_patient"].replace("image_","label_")+".nii.gz"))))
        imgs.update(row)
        imgs["gt"]=qlabel[int(row["native_z"])]==int(row["label_id"])
        imgs["support_mask"]=(slabel[int(row["support_native_z"])]==int(row["label_id"]))[None].astype(np.float32)
        rgb=imgs["query"][0].transpose(1,2,0)
        imgs["rgb"]=((rgb-rgb.min())/(rgb.max()-rgb.min()+1e-8)*255).astype(np.uint8)
        cases.append(imgs)
    print("inputs prepared; no query GT passed to FoB or SAM",flush=True)
    generated={};loads=[]
    for implementation,width in [("local",24),("upstream",10),("local",10)]:
        key=implementation+str(width);generated[key]={}
        for fold in sorted(manifest.fold.unique()):
            ckpt=Path(f"checkpoints/fob_checkpoints/exps_train_on_SABS_FSMIS_FoB/FSMIS_train_SABS_cv{fold}/ckpt.pth")
            seed_all(42)
            model,meta=load_fob(implementation,width,ckpt,canonical_unused=True);loads.append(meta)
            for c in cases:
                if c["fold"]==fold:
                    generated[key][c["episode_id"]]=forward(model,c,implementation)
                    print(f"generated {key} episode {c['episode_id']}",flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
    predictor,sam_load=official_sam_b(a.sam_checkpoint)
    rows=[];records=[];deltas=[];decoder_calls=0
    for c in cases:
        eid=c["episode_id"];predictor.set_image(c["rgb"]);masks={}
        arrays={"gt":c["gt"]}
        for variant in ["local24_legacy","local24_restored","upstream10_legacy","local10_legacy"]:
            key=variant.split("_")[0]
            neg,pos=generated[key][eid]
            n,p=(restore_xy(neg,(256,256),(257,257)),restore_xy(pos,(256,256),(257,257))) if variant.endswith("restored") else (neg,pos)
            raw,cc,quality=decode(predictor,n,p);decoder_calls+=1;masks[variant]=(raw,cc)
            records.append(candidate_record(eid,c["native_z"],neg,pos,(256,256),(257,257),list(range(len(neg))),1,variant))
            arrays[variant+"_raw"]=raw;arrays[variant+"_cc"]=cc
            for post,m in [("raw",raw),("cc",cc)]:
                rows.append({"episode_id":eid,"patient_id":c["patient_id"],"organ":c["organ"],"variant":variant,"post":post,"quality":quality,**metrics(m,c["gt"],c["spacing"])})
        neg,pos=generated["local24"][eid]
        for kind,points in [("negative",neg),("positive",pos)]:
            shift=restore_xy(points,(256,256),(257,257))-points
            shift_norm=np.linalg.norm(shift,axis=1);physical=np.linalg.norm(shift*c["spacing"][::-1],axis=1)
            arrays[kind+"_source_xy"]=points
            arrays[kind+"_restored_xy"]=points+shift
            deltas.append({"episode_id":eid,"kind":kind,"count":len(points),"mean_shift_px":float(shift_norm.mean()),"max_shift_px":float(shift_norm.max()),"mean_shift_mm":float(physical.mean()),"max_shift_mm":float(physical.max())})
        for post,i in [("raw",0),("cc",1)]:
            legacy,restored=masks["local24_legacy"][i],masks["local24_restored"][i]
            base=metrics(legacy,c["gt"],c["spacing"]);new=metrics(restored,c["gt"],c["spacing"])
            same10=np.array_equal(masks["upstream10_legacy"][i],masks["local10_legacy"][i])
            row={"episode_id":eid,"post":post,"changed_pixels":int((legacy!=restored).sum()),"changed_fraction":float(np.mean(legacy!=restored)),
                 "dice_change":new["dice"]-base["dice"],"surface_hd95_change_mm":new["slice_surface_hd95_mm"]-base["slice_surface_hd95_mm"],
                 "upstream10_local10_masks_equal":same10,
                 "width24_width10_changed_pixels":int((legacy!=masks["local10_legacy"][i]).sum()),
                 **boundary_change(legacy,neg,restore_xy(neg,(256,256),(257,257)),c["spacing"])}
            row["materiality_trigger"]=bool(abs(row["dice_change"])>=.005 or abs(row["surface_hd95_change_mm"])>=1 or row["changed_fraction"]>=.005)
            deltas.append(row)
        np.savez_compressed(a.out_dir/f"episode_{eid:03d}.npz",**arrays)
        print(f"decoded diagnostic episode {eid}; calls={decoder_calls}",flush=True)
    # Same hardware/precision, models resident, volume inputs cached, embedding uncached.
    # Includes transfers, FoB, image conversion, SAM encoder, final R1 decode and CC.
    c=cases[0]
    model,meta=load_fob("local",10,Path("checkpoints/fob_checkpoints/exps_train_on_SABS_FSMIS_FoB/FSMIS_train_SABS_cv0/ckpt.pth"),canonical_unused=True)
    times=[]
    torch.cuda.reset_peak_memory_stats()
    for repeat in range(4):
        torch.cuda.synchronize();start=time.perf_counter()
        neg,pos=forward(model,c,"local")
        neg,pos=restore_xy(neg,(256,256),(257,257)),restore_xy(pos,(256,256),(257,257))
        rgb=c["query"][0].transpose(1,2,0)
        rgb=((rgb-rgb.min())/(rgb.max()-rgb.min()+1e-8)*255).astype(np.uint8)
        predictor.set_image(rgb)
        decode(predictor,neg,pos);decoder_calls+=1
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if repeat:times.append(elapsed)
        print(f"timing {'warmup' if repeat==0 else repeat}: {elapsed:.6f}s",flush=True)
    timing={"hardware":torch.cuda.get_device_name(0),"precision":"float32 TF32 disabled","scenario":"models/volume slices cached; embedding uncached; transfers, FoB, coordinate restoration, image conversion, SAM encoder+R1+CC included; disk I/O and initial model loads excluded",
            "warmup_runs":1,"repeats":3,"seconds":times,"mean_seconds":float(np.mean(times)),"median_seconds":float(np.median(times)),
            "std_seconds":float(np.std(times,ddof=1)),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "estimated_4009_slice_seconds_at_mean":float(np.mean(times)*4009),"estimate_not_full_run":True}
    pd.DataFrame(rows).to_csv(a.out_dir/"slice_metrics.csv",index=False)
    pd.DataFrame(deltas).to_csv(a.out_dir/"coordinate_sensitivity.csv",index=False)
    write_records(a.out_dir/"candidate_records.jsonl",records)
    result={"manifest_sha256":sha256(a.manifest),"unique_slices":len(cases),"decoder_evaluations_total":decoder_calls,"diagnostic_decoder_evaluations":32,
            "timing_decoder_evaluations":4,"sam_encoder_evaluations":12,"timing":timing,"loads":loads,"sam_load":sam_load}
    (a.out_dir/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()
