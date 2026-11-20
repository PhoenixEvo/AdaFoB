"""Complete numerical parity with explicit schema adapter and immutable outputs."""
from __future__ import annotations
import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
from .runtime import load_fob, official_sam_b, seed_all, sha256


def difference(a, b, atol, rtol, floor):
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape:
        return {"shape_reference":list(x.shape),"shape_local":list(y.shape),"pass":False,"reason":"shape mismatch"}
    delta=np.abs(x.astype(float)-y.astype(float))
    relative=delta/np.maximum(np.abs(x.astype(float)),floor)
    return {"shape":list(x.shape),"max_absolute":float(delta.max(initial=0)),
            "max_relative":float(relative.max(initial=0)),
            "pass":bool(np.all(delta<=atol+rtol*np.abs(x.astype(float)))),
            "different_elements":int(np.count_nonzero(delta)),
            "atol":atol,"rtol":rtol,"relative_denominator_floor":floor}


def forward(model, supp, mask, qry, dummy, implementation):
    import torch
    stages={}
    handles=[]
    for name in ["encoder","masked_attention","head","refine"]:
        def hook(module, inputs, output, key=name):
            tensor=output[0] if isinstance(output,(list,tuple)) else output
            if key in ["head","refine"]:
                tensor=tensor[:,:10]
            stages[key]=tensor.detach().cpu().numpy().copy()
        handles.append(getattr(model,name).register_forward_hook(hook))
    try:
        kwargs={} if implementation=="upstream" else {"use_skeleton":False,"budget_Np":10}
        with torch.inference_mode():
            out=model([[supp.clone()]],[[mask.clone()]],[qry.clone()],dummy.clone(),train=False,**kwargs)
        return [np.asarray(out[0]).copy(),np.asarray(out[1]).copy()],stages
    finally:
        for handle in handles:handle.remove()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fob-checkpoint",type=Path,required=True)
    p.add_argument("--sam-checkpoint",type=Path,required=True)
    p.add_argument("--max-points",type=int,choices=[10,24],required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--trim-refine-valid",action="store_true",help="separate fixed arm: run SPR only on valid channels")
    p.add_argument("--valid-ring-only",action="store_true",help="single-factor audit arm; production unchanged")
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    config=json.loads(a.protocol.read_text())
    tolerance=config["parity_tolerance"]
    import torch
    seed_all(a.seed)
    generator=torch.Generator().manual_seed(a.seed)
    supp=torch.randn(1,3,256,256,generator=generator).cuda()
    qry=torch.randn(1,3,256,256,generator=generator).cuda()
    mask=torch.zeros(1,256,256,device="cuda");mask[:,60:120,60:120]=1
    dummy=torch.ones(1,256,256,dtype=torch.long,device="cuda")
    outputs=[]; stages=[]; loads=[]
    for implementation,width in [("upstream",10),("local",a.max_points)]:
        seed_all(a.seed)
        print("BEGIN load "+implementation,flush=True)
        model,meta=load_fob(implementation,width,a.fob_checkpoint,canonical_unused=True)
        if implementation == "local" and a.valid_ring_only:
            from .ring_diagnostic import install_valid_ring
            install_valid_ring(model)
        if implementation == "local" and a.trim_refine_valid:
            from .ring_diagnostic import install_trimmed_refine
            install_trimmed_refine(model)
        print("BEGIN forward "+implementation,flush=True)
        out,stage=forward(model,supp,mask,qry,dummy,implementation)
        outputs.append(out);stages.append(stage);loads.append(meta)
        del model
        gc.collect();torch.cuda.empty_cache()
        print("END forward "+implementation,flush=True)
    comparisons={}
    for i,name in enumerate(["negative_xy","positive_xy"]):
        comparisons[name]=difference(outputs[0][i],outputs[1][i],**tolerance)
    stage_differences={name:difference(stages[0][name],stages[1][name],**tolerance) for name in stages[0]}
    del stages
    predictor,sam_meta=official_sam_b(a.sam_checkpoint)
    rgb=qry[0].permute(1,2,0).cpu().numpy()
    rgb=((rgb-rgb.min())/(rgb.max()-rgb.min())*255).astype(np.uint8)
    print("BEGIN cached SAM embedding",flush=True)
    predictor.set_image(rgb)
    decoded=[]; masks=[]; cc=[]
    from models.postprocessing import filter_connected_components
    for out in outputs:
        neg,pos=out[0].reshape(-1,2),out[1].reshape(-1,2)
        points=np.concatenate([pos,neg]);labels=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
        with torch.inference_mode():
            d=predictor.predict(point_coords=points,point_labels=labels,multimask_output=False,return_logits=True)
        decoded.append(d)
        m=d[0][0]>predictor.model.mask_threshold
        masks.append(m)
        cc.append(filter_connected_components(m.astype(np.uint8),pos))
    for i,name in enumerate(["sam_R1_postprocessed_logits","sam_R1_quality","sam_R1_low_resolution_logits"]):
        comparisons[name]=difference(decoded[0][i],decoded[1][i],**tolerance)
    comparisons["final_binary_mask"]=difference(masks[0],masks[1],0,0,tolerance["floor"])
    comparisons["final_CC_mask"]=difference(cc[0],cc[1],0,0,tolerance["floor"])
    passed=all(v["pass"] for v in comparisons.values())
    result={"single_factor_arm":"trim_valid_refine" if a.trim_refine_valid else "valid_ring_only" if a.valid_ring_only else "legacy", "numerical_parity":"PASS" if passed else "FAIL","max_points":a.max_points,"seed":a.seed,
            "tolerance_protocol_sha256":sha256(a.protocol),"comparisons":comparisons,
            "stage_differences":stage_differences,"loads":loads,"sam_load":sam_meta,
            "raw_checkpoint_strict_load":"FAIL: missing unused encoder.reduce2.weight",
            "inference_schema_variant":"unused_reduce2_removed; original forward code unchanged",
            "sam_decoder_evaluations":2,"final_mask_rule":"R1/token0; logits>0; common CC",
            "precision":"float32; TF32 disabled; one cached image embedding"}
    with a.out.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if passed else 2)


if __name__=="__main__":main()
