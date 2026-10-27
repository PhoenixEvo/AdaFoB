"""Deterministic standard controls; no fitted scoring or new coordinates."""
from __future__ import annotations
import hashlib,time
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree,distance
from analysis.rebaseline_v1.common import generate,rgb_image,timed,predict,cc_mask,oob,restore_xy
BUDGETS=(2,4,6)
SEEDS=tuple(range(42,47))
BASE="full10"
ZERO="positive0"
PREFIX_REF="GT_prefix_reference"
GREEDY_REF="GT_greedy_reference"
ARMS=(BASE,ZERO,*[f"prefix{k}" for k in BUDGETS],*[f"random{k}_s{s}" for k in BUDGETS for s in SEEDS],
      *[f"fps{k}" for k in BUDGETS],*[f"boundary_{v}{k}" for v in ("near","far") for k in BUDGETS],PREFIX_REF,GREEDY_REF)
assert len(ARMS)==31

def fps(points,ids,k):
    """Max-min Euclidean xy pixels; farthest from centroid first, stable ID ties."""
    points=np.asarray(points,dtype=np.float64);ids=np.asarray(ids,dtype=int)
    order=np.argsort(ids);points,ids=points[order],ids[order]
    selected=[];d=((points-points.mean(0))**2).sum(1)
    for _ in range(k):
        available=[i for i in range(len(ids)) if i not in selected]
        i=min(available,key=lambda j:(-float(d[j]),int(ids[j])))
        selected.append(i)
        d=np.minimum(d,((points-points[i])**2).sum(1)) if len(selected)>1 else ((points-points[i])**2).sum(1)
    return [int(ids[i]) for i in selected]

def boundary_distances(points,mask):
    """Continuous restored xy to 4-neighbour raw-mask boundary, native pixels."""
    boundary=mask&~ndimage.binary_erosion(mask,structure=ndimage.generate_binary_structure(2,1),border_value=0)
    xy=np.argwhere(boundary)[:,::-1]
    if not len(xy):return np.full(len(points),np.inf),xy
    return cKDTree(xy).query(np.asarray(points,dtype=np.float64))[0],xy

def proximity(points,ids,k,mask,far=False):
    ds,_=boundary_distances(points,mask)
    return [int(ids[i]) for i in sorted(range(len(ids)),key=lambda i:((-ds[i] if far else ds[i]),int(ids[i])))[:k]]

def random_ids(ids,k,seed,episode_id,native_z):
    """Nested per-seed subsets keyed by stable episode and native slice identities."""
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(episode_id),int(native_z)]))
    return rng.permutation(sorted(map(int,ids)))[:k].tolist()

def dispersion(points,boundary_xy):
    points=np.asarray(points,dtype=np.float64)
    pairs=distance.pdist(points) if len(points)>1 else np.array([])
    result={"pair_mean_px":float(pairs.mean()) if len(pairs) else None,"pair_min_px":float(pairs.min()) if len(pairs) else None,
            "near_duplicate_pairs_le1px":int((pairs<=1).sum()),"exact_duplicate_pairs":int((pairs==0).sum())}
    if len(points) and len(boundary_xy):
        ds=cKDTree(points).query(boundary_xy)[0]
        result.update(boundary_coverage_le5px=float((ds<=5).mean()),boundary_mean_nearest_px=float(ds.mean()))
    else:result.update(boundary_coverage_le5px=0.,boundary_mean_nearest_px=None)
    return result

def dice(pred,gt):
    p,g=pred.sum(),gt.sum()
    if p==0 and g==0:return 1.
    if p==0 or g==0:return 0.
    return float(2.*np.logical_and(pred,gt).sum()/(p+g))

def references(decode,gt,tolerance=1e-4):
    """Exact legacy per-slice raw-Dice prefix and path-ordered greedy semantics."""
    zero=decode([]);best=dice(zero,gt);prefix=[];pt=[{"ids":[],"dice":best,"accepted":True}]
    for k in range(1,11):
        ids=list(range(k));score=dice(decode(ids),gt);accept=score>best
        pt.append({"ids":ids,"dice":score,"accepted":bool(accept)})
        if accept:best=score;prefix=ids
    selected=[];remaining=list(range(10));current=dice(zero,gt);trajectory=[]
    while remaining:
        step_best=current;winner=None;trials=[]
        for candidate in remaining:
            trial=selected+[candidate];score=dice(decode(trial),gt);accepted=score>step_best+tolerance
            trials.append({"ids":trial,"dice":score,"running_update":bool(accepted)})
            if accepted:step_best=score;winner=candidate
        trajectory.append({"current_ids":selected.copy(),"current_dice":current,"trials":trials,"winner":winner})
        if winner is None:break
        selected.append(winner);remaining.remove(winner);current=step_best
    assert sum(len(x["trials"]) for x in trajectory)<=55
    return prefix,selected,{"prefix":pt,"greedy":trajectory}

def baseline_digest(arrays):
    return hashlib.sha256(b"".join(np.ascontiguousarray(arrays[a+"_"+p],dtype=np.bool_).tobytes()
                                 for a in (BASE,ZERO) for p in ("raw","cc"))).hexdigest()

def coordinate_digest(neg,pos,nr,pr):
    return hashlib.sha256(b"".join(np.ascontiguousarray(x).tobytes() for x in (neg,pos,nr,pr))).hexdigest()

def run_slice(model,predictor,case,c,expected):
    """One embedding; order-preserving decoder memoization, explicit real calls."""
    begin=time.perf_counter();encoder0=predictor.audit_encoder_calls;decoder0=predictor.audit_decoder_calls
    (neg,pos),ft=timed(lambda:generate(model,case));assert neg.shape==(10,2)
    nr,pr=restore_xy(neg,(256,256),(257,257)),restore_xy(pos,(256,256),(257,257))
    assert coordinate_digest(neg,pos,nr,pr)==expected["coordinates"],"Step8 FoB coordinates changed"
    image,it=timed(lambda:rgb_image(case));_,et=timed(lambda:predictor.set_image(image))
    cache={};requests=[];decoder_seconds=[]
    def decode(ids):
        key=tuple(map(int,ids));hit=key in cache;request={"ids":list(key),"cache_hit":hit}
        if not hit:
            before=predictor.audit_decoder_calls
            (mask,_,_,_),dt=timed(lambda:predict(predictor,nr[list(key)],pr))
            assert predictor.audit_decoder_calls-before==1
            cache[key]=mask;decoder_seconds.append(dt);request["decoder_seconds"]=dt
        requests.append(request);return cache[key]
    arrays={"gt":case["gt"],"negative_source_xy":neg,"positive_source_xy":pos,"negative_restored_xy":nr,"positive_restored_xy":pr}
    events={};cc_seconds=[]
    def save(arm,ids,seed=None):
        before=predictor.audit_decoder_calls;raw=decode(ids);cc,ct=timed(lambda:cc_mask(raw,pr));cc_seconds.append(ct)
        arrays[arm+"_raw"]=raw;arrays[arm+"_cc"]=cc
        events[arm]={"selected_ids":list(ids),"selection_seed":seed,"positive_count":len(pr),"negative_count":len(ids),"total_count":len(pr)+len(ids),
                     "actual_decoder_calls":predictor.audit_decoder_calls-before,"positive_oob":oob(pr),"negative_oob":oob(nr[list(ids)]),"cc_seconds":ct}
    save(BASE,list(range(10)));save(ZERO,[])
    _,boundary_xy=boundary_distances(nr,arrays[BASE+"_raw"])
    selection_start=time.perf_counter();choices={f"prefix{k}":(list(range(k)),None) for k in BUDGETS}
    for k in BUDGETS:
        for s in SEEDS:choices[f"random{k}_s{s}"]=(random_ids(range(10),k,s,case["episode_id"],case["native_z"]),s)
        choices[f"fps{k}"]=(fps(nr,np.arange(10),k),None)
        for v in ("near","far"):choices[f"boundary_{v}{k}"]=(proximity(nr,np.arange(10),k,arrays[BASE+"_raw"],far=v=="far"),None)
    st=time.perf_counter()-selection_start
    for arm,(ids,seed) in choices.items():save(arm,ids,seed)
    calls_before=predictor.audit_decoder_calls;req_before=len(requests)
    prefix,greedy,trajectories=references(decode,case["gt"],c["greedy_tolerance"])
    reference_actual=predictor.audit_decoder_calls-calls_before;reference_requests=len(requests)-req_before
    save(PREFIX_REF,prefix);save(GREEDY_REF,greedy)
    for event in events.values():event["dispersion"]=dispersion(nr[event["selected_ids"]],boundary_xy)
    assert baseline_digest(arrays)==expected["masks"],"C0 raw/CC differs from Step8 R2/R3"
    record={k:v for k,v in case.items() if k not in ("query","supp","support_mask","gt","other","spacing_xyz")}
    record.update(seed=c["seed"],source_grid_hw=[256,256],native_grid_hw=[257,257],spacing_xyz=case["spacing_xyz"].tolist(),generation_order_ids=list(range(10)),
                  arms=events,trajectories=trajectories,decode_requests=requests,failures=[],image_encoder_calls=predictor.audit_encoder_calls-encoder0,
                  decoder_calls=predictor.audit_decoder_calls-decoder0,reference_actual_decoder_calls=reference_actual,reference_requests_including_cache_hits=reference_requests,
                  pool_dispersion=dispersion(nr,boundary_xy),runtime={"fob_seconds":ft,"image_conversion_seconds":it,"encoder_api_seconds":et,
                  "decoder_seconds":sum(decoder_seconds),"decoder_seconds_per_call":float(np.mean(decoder_seconds)),"cc_seconds":sum(cc_seconds),
                  "selection_seconds":st,"slice_compute_seconds":time.perf_counter()-begin},c0_step8_exact=True)
    assert record["image_encoder_calls"]==1 and set(events)==set(ARMS)
    return arrays,record