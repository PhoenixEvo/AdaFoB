"""Research-only slice pilot integration. No query labels accepted by prediction.
No real model forward is run by this module's synthetic tests or CPU preflight.
"""
from pathlib import Path
from dataclasses import replace
import json,time,os
import numpy as np
from scipy import ndimage
from experiments.utility_feasibility_v1.slice_api import predict_slice,FrozenConfig,FrozenFoBSAMBackend
from experiments.utility_feasibility_v1.scoring_v1 import evidence,score_subsets,random2
from analysis.controls_v1.core import proximity

class PilotBackend(FrozenFoBSAMBackend):
    """Already strictly loaded models only; this adapter does not allocate them."""
    def __init__(self,model,predictor):
        super().__init__(model,predictor);self.support_cache={};self.support_encoder_calls=0
    def prepare_support(self,support,mask):
        import hashlib,torch
        from analysis.rebaseline_v1.common import rgb_image
        key=hashlib.sha256(support.tobytes()+mask.tobytes()).hexdigest()
        if key not in self.support_cache:
            self.predictor.set_image(rgb_image({'query':support}))
            f=self.predictor.features.detach().cpu().numpy()[0].copy()
            self.support_cache[key]=f;self.support_encoder_calls+=1
        self.support_features=self.support_cache[key]
    def encode(self,query):
        super().encode(query)
        self.query_features=self.predictor.features.detach().cpu().numpy()[0].copy()

def mask_on_feature_grid(mask,shape):
    m=np.asarray(mask,dtype=bool).squeeze(axis=0)
    h,w=m.shape;fh,fw=shape
    iy=np.floor((np.arange(fh)+.5)*h/fh).astype(int)
    ix=np.floor((np.arange(fw)+.5)*w/fw).astype(int)
    return m[np.ix_(iy,ix)]

def pilot_predict(*,query_image,support_image,support_mask,geometry,configuration,backend):
    """Same actual slice API + frozen rules, no evaluation labels/metadata."""
    if geometry.native_hw[0]!=geometry.native_hw[1]:raise ValueError('Square native geometry required')
    if configuration!=FrozenConfig():raise ValueError('Pilot must start with frozen full10 config')
    backend.prepare_support(support_image,support_mask)
    full=predict_slice(query_image=query_image,support_image=support_image,support_mask=support_mask,
                       geometry=geometry,configuration=configuration,backend=backend)
    neg,pos=full['negative_restored_xy'],full['positive_restored_xy']
    cache={tuple(range(10)):full['raw']};requests=[{'ids':list(range(10)),'cache_hit':False}]
    decoder_times=[]
    def decode(ids):
        key=tuple(sorted(map(int,ids)));hit=key in cache
        if len(set(key))!=len(key) or any(i<0 or i>=10 for i in key):raise ValueError('Invalid candidate IDs')
        if not hit:
            if len(cache)>=20:raise RuntimeError('Per-slice configuration cap')
            start=time.perf_counter();cache[key]=np.asarray(backend.decode(neg[list(key)],pos),dtype=bool)
            decoder_times.append(time.perf_counter()-start)
            if cache[key].shape!=geometry.native_hw:raise ValueError('Mask grid')
        requests.append({'ids':list(key),'cache_hit':hit});return cache[key]
    deleted=np.stack([decode([j for j in range(10) if j!=i]) for i in range(10)])
    fm=mask_on_feature_grid(support_mask,backend.support_features.shape[1:])
    e=evidence(backend.support_features,fm,backend.query_features,geometry.native_hw)
    scores=score_subsets(full['raw'],deleted,neg,e)
    choices={'full10':tuple(range(10)),
             'boundary_near2':tuple(sorted(proximity(neg,list(range(10)),2,full['raw']))),
             'magnitude':scores['magnitude'],'coordinate':scores['coordinate'],
             'signed':scores['signed'],'protected':scores['protected']}
    choices.update({f'random_s{s}':random2(s) for s in range(42,46)})
    masks={'full10_raw':full['raw'],'full10_cc':full['cc']}
    # full10 was already the first request, not a second logical request.
    for arm,ids in list(choices.items())[1:]:
        raw=decode(ids);masks[arm+'_raw']=raw.copy();masks[arm+'_cc']=backend.cc(raw,pos).copy()
    if len(requests)!=20 or len(cache)>20:raise AssertionError('Configuration accounting')
    arrays={**masks,'negative_source_xy':full['negative_source_xy'],
            'negative_restored_xy':neg,'positive_source_xy':full['positive_source_xy'],
            'positive_restored_xy':pos,'single_deletion_raw':deleted,
            'added':scores['added'],'lost':scores['lost']}
    if e is not None:arrays['evidence']=e
    record={'logical_requests':len(requests),'configurations_decoded':len(cache),
            'api_decoder_invocations':len(cache),'query_encoder_calls':1,
            'support_encoder_calls_cumulative':backend.support_encoder_calls,
            'generation_order':list(range(10)),'selected_ids':{a:list(v) for a,v in choices.items()},
            'actual_negative_counts':{a:len(v) for a,v in choices.items()},'requests':requests,
            'fallback':scores.get('fallback'),'timing_note':'adapter uses sequential calls; GPU timing requires synchronized harness',
            'negative_oob_count':int(np.any((neg<0)|(neg>np.array(geometry.native_hw[::-1])-1),axis=1).sum()),
            'positive_oob_count':int(np.any((pos<0)|(pos>np.array(geometry.native_hw[::-1])-1),axis=1).sum())}
    return arrays,record

def evaluate_saved(arrays,query_mask):
    """Evaluation boundary: called ONLY after label-free prediction returns."""
    g=np.asarray(query_mask,dtype=bool);out={}
    for name,p in arrays.items():
        if not (name.endswith('_raw') or name.endswith('_cc')) or p.ndim!=2:continue
        tp=int((p&g).sum());fp=int((p&~g).sum());fn=int((~p&g).sum())
        out[name]={'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                   'recall':tp/(tp+fn) if tp+fn else None,'fp_pixels':fp,'fn_pixels':fn}
    return out

class RunWriter:
    def __init__(self,path):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=False)
        self.state={'status':'partial','completed_cases':[],'configurations_decoded':0}
        self._state()
    def _state(self):
        tmp=self.path/'status.tmp'
        with tmp.open('x',encoding='utf-8') as f:json.dump(self.state,f,indent=2)
        os.replace(tmp,self.path/'status.json') # Only own new run's status is replaceable.
    def save(self,case_id,arrays,record):
        if case_id in self.state['completed_cases']:raise FileExistsError('Case already committed')
        if not case_id.isalnum():raise ValueError('Safe case identifier required')
        total=self.state['configurations_decoded']+record['configurations_decoded']
        if total>160 or record['configurations_decoded']>20:raise RuntimeError('Aggregate cap')
        with (self.path/(case_id+'.partial.npz')).open('xb') as f:np.savez_compressed(f,**arrays)
        with (self.path/(case_id+'.json')).open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
        os.rename(self.path/(case_id+'.partial.npz'),self.path/(case_id+'.npz'))
        self.state['completed_cases'].append(case_id);self.state['configurations_decoded']=total;self._state()
    def complete(self):
        if len(self.state['completed_cases'])!=8:raise RuntimeError('Incomplete eight-slice run')
        self.state['status']='completed';self._state()