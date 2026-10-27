"""Synthetic-testable scoring primitives. No query label or metric input.
Not a production selector or a checkpoint-integrated pilot runner.
"""
import numpy as np
from scipy import ndimage

EPS = np.finfo(np.float32).eps

def evidence(support_features, support_mask_on_feature_grid, query_features, native_hw):
    """CxHxW frozen SAM image features; cosine contrast, not probability."""
    s=np.asarray(support_features,dtype=np.float64)
    q=np.asarray(query_features,dtype=np.float64)
    m=np.asarray(support_mask_on_feature_grid,dtype=bool)
    if s.ndim!=3 or q.ndim!=3 or s.shape[0]!=q.shape[0] or m.shape!=s.shape[1:]:
        raise ValueError('Feature/mask shapes')
    if not np.isfinite(s).all() or not np.isfinite(q).all():return None
    sn=np.linalg.norm(s,axis=0);qn=np.linalg.norm(q,axis=0)
    su=s/np.maximum(sn,EPS);qu=q/np.maximum(qn,EPS)
    core=ndimage.binary_erosion(m,structure=np.ones((3,3)),border_value=0)&(sn>EPS)
    bg=~ndimage.binary_dilation(m,structure=np.ones((3,3)),border_value=0)&(sn>EPS)
    if not core.any() or not bg.any() or (qn<=EPS).any():return None
    fg=su[:,core].mean(axis=1);back=su[:,bg].mean(axis=1)
    fn=np.linalg.norm(fg);bn=np.linalg.norm(back)
    if min(fn,bn)<=EPS:return None
    fg/=fn;back/=bn
    if np.linalg.norm(fg-back)<=EPS:return None
    coarse=np.clip(np.einsum('c,chw->hw',fg-back,qu)/2,-1,1)
    h,w=native_hw
    yy=(np.arange(h)+.5)*q.shape[1]/h-.5
    xx=(np.arange(w)+.5)*q.shape[2]/w-.5
    y,x=np.meshgrid(yy,xx,indexing='ij')
    return ndimage.map_coordinates(coarse,[y,x],order=1,mode='nearest',prefilter=False)

def signed_regions(full,deleted):
    full=np.asarray(full,dtype=bool);deleted=np.asarray(deleted,dtype=bool)
    if full.shape!=deleted.shape:raise ValueError('Mask shapes')
    return deleted&~full,full&~deleted

def top2(scores,eligible=None):
    values=np.asarray(scores,dtype=float)
    if values.shape!=(10,) or not np.isfinite(values).all():return tuple(range(10))
    ids=list(range(10)) if eligible is None else list(eligible)
    if len(ids)<2:return tuple(range(10))
    return tuple(sorted(sorted(ids,key=lambda i:(-values[i],i))[:2]))

def score_subsets(full,deleted_masks,negative_xy,support_evidence):
    """Fixed ten-candidate, k2 research scores; evidence failure -> original pool."""
    full=np.asarray(full,dtype=bool);deletions=np.asarray(deleted_masks,dtype=bool)
    xy=np.asarray(negative_xy,dtype=float)
    if deletions.shape!=(10,)+full.shape or xy.shape!=(10,2):raise ValueError('Pool shapes')
    added=deletions&~full;lost=full&~deletions
    mag=(added|lost).mean(axis=(1,2))
    out={'magnitude':top2(mag),'magnitude_scores':mag,'added':added,'lost':lost}
    if support_evidence is None:
        out.update(coordinate=tuple(range(10)),signed=tuple(range(10)),protected=tuple(range(10)),fallback='invalid_support_evidence')
        return out
    e=np.asarray(support_evidence,dtype=float)
    h,w=full.shape
    # Retain out-of-bounds coordinates in records; never clamp them into the image.
    valid=np.isfinite(xy).all() and np.all((xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1))
    if e.shape!=full.shape or not np.isfinite(e).all() or not valid:
        out.update(coordinate=tuple(range(10)),signed=tuple(range(10)),protected=tuple(range(10)),fallback='invalid_evidence_or_coordinate')
        return out
    coord=-ndimage.map_coordinates(e,[xy[:,1],xy[:,0]],order=1,mode='nearest',prefilter=False)
    n=full.size
    a=(added*e).sum(axis=(1,2))/n;l=(lost*e).sum(axis=(1,2))/n
    score=l-a
    # Hard protection, no tuned coefficient: veto retained points whose deletion
    # adds more support-compatible foreground evidence than it loses.
    af=(added*np.maximum(e,0)).sum(axis=(1,2))/n
    lf=(lost*np.maximum(e,0)).sum(axis=(1,2))/n
    eligible=np.flatnonzero(af<=lf).tolist()
    out.update(coordinate=top2(coord),signed=top2(score),protected=top2(score,eligible),
               coordinate_scores=coord,signed_scores=score,added_evidence=a,lost_evidence=l,
               added_foreground=af,lost_foreground=lf,eligible=eligible,
               fallback='protection_fewer_than_two' if len(eligible)<2 else None)
    return out

def random2(seed):return tuple(sorted(np.random.default_rng(seed).choice(10,2,replace=False).tolist()))

def request_cache_keys(selected):
    """Logical 20 requests, canonical ordered IDs; coordinates retain generation IDs."""
    if len(selected)!=5:raise ValueError('near2 plus four scoring subsets required')
    req=[tuple(range(10))]+[tuple(j for j in range(10) if j!=i) for i in range(10)]
    req += [tuple(sorted(ids)) for ids in selected]+[random2(s) for s in range(42,46)]
    if len(req)!=20:raise AssertionError('Configuration cap')
    return req,list(dict.fromkeys(req))