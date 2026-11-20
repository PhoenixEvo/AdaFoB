"""Audit-only single-factor replacement. Does not modify production source."""
import types

def install_valid_ring(model):
    def valid_ring(self,K,B,device,valid_mask=None):
        import torch
        ids=torch.arange(K,device=device) if valid_mask is None else torch.nonzero(valid_mask).flatten()
        A=torch.zeros(K,K,device=device)
        for j,i in enumerate(ids):
            A[i,ids[(j-1)%len(ids)]]=1
            A[i,ids[(j+1)%len(ids)]]=1
        return A.unsqueeze(0).expand(B,-1,-1)
    model.refine.gcn.build_ring_adj=types.MethodType(valid_ring,model.refine.gcn)

def install_trimmed_refine(model):
    original=model.refine.forward
    def trimmed(self,query_feats,query_points,support_feats,feat_map,A_prior=None,valid_mask=None):
        n=query_feats.shape[1] if valid_mask is None else int(valid_mask.sum())
        if valid_mask is not None:
            assert bool(valid_mask[:n].all()) and not bool(valid_mask[n:].any()), 'diagnostic requires prefix-valid channels'
        prior=None if A_prior is None else A_prior[:,:n,:n]
        return original(query_feats[:,:n],query_points[:,:n],support_feats[:,:n],feat_map,A_prior=prior,valid_mask=None)
    model.refine.forward=types.MethodType(trimmed,model.refine)
