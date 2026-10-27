"""Slice-only prediction boundary. No benchmark assembly or query labels.
Injected backends are trusted implementations; isolation tests use a CPU test double.
"""
from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class Geometry:
    native_hw: tuple = (257,257)
    source_hw: tuple = (256,256)
    spacing_xy: tuple = (1.,1.)
    origin_xyz: tuple = (0.,0.,0.)
    direction_xyz: tuple = (1.,0.,0.,0.,1.,0.,0.,0.,1.)

@dataclass(frozen=True)
class FrozenConfig:
    arm: str = 'full10'
    token: int = 0
    candidate_count: int = 10

class Backend(Protocol):
    def generate(self, query, support, support_mask): ...
    def encode(self, query): ...
    def decode(self, negative_xy, positive_xy): ...
    def cc(self, mask, positive_xy): ...

def restore(points, geometry):
    scale=np.array(geometry.native_hw[::-1])/np.array(geometry.source_hw[::-1])
    return ((np.asarray(points,dtype=np.float64)+.5)*scale-.5).astype(np.float32)

def predict_slice(*, query_image, support_image, support_mask, geometry,
                  configuration, backend):
    if type(geometry) is not Geometry or type(configuration) is not FrozenConfig:
        raise TypeError('Only explicit geometry and frozen configuration accepted')
    if configuration.token != 0 or configuration.candidate_count != 10:
        raise ValueError('Only corrected R2 token0 / ten-candidate pool')
    if configuration.arm not in ('full10','boundary_near2'):
        raise ValueError('Only existing controls; no utility selector here')
    h,w=geometry.native_hw
    if min(*geometry.native_hw,*geometry.source_hw,*geometry.spacing_xy)<=0:
        raise ValueError('Invalid geometry')
    query=np.array(query_image,dtype=np.float32,copy=True)
    support=np.array(support_image,dtype=np.float32,copy=True)
    mask=np.array(support_mask,dtype=np.float32,copy=True)
    if query.shape!=(1,3,h,w) or support.shape!=query.shape or mask.shape!=(1,h,w):
        raise ValueError('Expected normalized FoB tensors: 1x3xHxW, mask1xHxW')
    if not all(np.isfinite(x).all() for x in (query,support,mask)):
        raise ValueError('Nonfinite input')
    if not np.isin(mask,[0,1]).all() or not mask.any():
        raise ValueError('Nonempty binary support mask required')
    neg,pos=backend.generate(query,support,mask)
    neg=np.asarray(neg);pos=np.asarray(pos)
    if neg.shape!=(10,2) or pos.ndim!=2 or pos.shape[1]!=2:
        raise ValueError('Invalid original candidate pool')
    if not np.isfinite(neg).all() or not np.isfinite(pos).all():
        raise ValueError('Nonfinite coordinates')
    nr,pr=restore(neg,geometry),restore(pos,geometry)
    backend.encode(query)
    full=np.asarray(backend.decode(nr,pr),dtype=bool)
    if full.shape!=(h,w):raise ValueError('Mask grid mismatch')
    ids=list(range(10));calls=1
    raw=full
    if configuration.arm=='boundary_near2':
        from analysis.controls_v1.core import proximity
        ids=proximity(nr,list(range(10)),2,full)
        raw=np.asarray(backend.decode(nr[ids],pr),dtype=bool);calls+=1
    cc=np.asarray(backend.cc(raw,pr),dtype=bool)
    if raw.shape!=(h,w) or cc.shape!=raw.shape or (cc&~raw).any():
        raise ValueError('Invalid raw/CC output')
    return dict(raw=raw.copy(),cc=cc.copy(),selected_ids=ids,
                negative_source_xy=neg.copy(),negative_restored_xy=nr,
                positive_source_xy=pos.copy(),positive_restored_xy=pr,
                image_encoder_calls=1,decoder_configurations=calls,
                api_decoder_invocations=calls)

class FrozenFoBSAMBackend:
    """Adapter for already loaded, matched width10 FoB and official SAM-B.
    No checkpoint construction/download here. Caller must verify checkpoint hashes.
    This adapter is NOT real-checkpoint integration-tested by CPU double tests.
    """
    def __init__(self, model, predictor):
        self.model=model;self.predictor=predictor
    def generate(self,query,support,support_mask):
        from analysis.rebaseline_v1.common import generate
        return generate(self.model,dict(query=query,supp=support,support_mask=support_mask))
    def encode(self,query):
        from analysis.rebaseline_v1.common import rgb_image
        self.predictor.set_image(rgb_image(dict(query=query)))
    def decode(self,negative_xy,positive_xy):
        from analysis.rebaseline_v1.common import predict
        return predict(self.predictor,negative_xy,positive_xy)[0]
    def cc(self,mask,positive_xy):
        from analysis.rebaseline_v1.common import cc_mask
        return cc_mask(mask,positive_xy)