"""Parity test: ensure variable N_p modifications do not alter baseline behavior."""

import os
import sys
import torch
import numpy as np
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..")))
sys.path.append(os.path.abspath(os.path.join(_HERE, "..", "third_party", "FoB_SAM")))

from third_party.FoB_SAM.models.FoB import FewShotSeg as NewFoB
from third_party.FoB_SAM.models.FoB_orig import FewShotSeg as OrigFoB

def test_parity():
    # Fix seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Initialize models
    dummy_args = type("A", (), {})()
    model_orig = OrigFoB(dummy_args).cuda().eval()
    
    torch.manual_seed(42)
    model_new = NewFoB(dummy_args).cuda().eval()
    
    # Sync weights exactly
    model_new.load_state_dict(model_orig.state_dict())
    
    # Create dummy inputs
    B, C, H, W = 1, 3, 256, 256
    supp_imgs = [[torch.randn(B, C, H, W).cuda()]]  # 1-way, 1-shot (list of lists of tensors)
    supp_mask = [[torch.randint(0, 2, (B, H, W)).cuda().float()]]
    qry_imgs = [torch.randn(B, C, H, W).cuda()]
    qry_labels = torch.randint(0, 2, (B, H, W)).cuda().long()
    
    print("Testing Inference (train=False)...")
    with torch.no_grad():
        neg_orig, pos_orig = model_orig(
            [[s.clone() for s in way] for way in supp_imgs], 
            [[s.clone() for s in way] for way in supp_mask], 
            [q.clone() for q in qry_imgs], 
            qry_labels.clone(), 
            train=False, use_skeleton=False
        )
        neg_new, pos_new = model_new(
            [[s.clone() for s in way] for way in supp_imgs], 
            [[s.clone() for s in way] for way in supp_mask], 
            [q.clone() for q in qry_imgs], 
            qry_labels.clone(), 
            train=False, use_skeleton=False
        )
        
    # We use atol=1e-1 because floating-point non-associativity in PyTorch's SDPA 
    # and 3 iterations of SPR deformable attention accumulates ~0.02 pixel differences
    np.testing.assert_allclose(neg_orig, neg_new, atol=1e-1, err_msg="neg_point mismatch")
    np.testing.assert_allclose(pos_orig, pos_new, atol=1e-1, err_msg="pos_point mismatch")
    print("Inference parity: PASSED")
    
    print("Testing Training Losses (train=True)...")
    model_orig.train()
    model_new.train()
    
    with torch.no_grad():
        loss_orig = model_orig(
            [[s.clone() for s in way] for way in supp_imgs], 
            [[s.clone() for s in way] for way in supp_mask], 
            [q.clone() for q in qry_imgs], 
            qry_labels.clone(), 
            train=True, use_skeleton=False
        )
        loss_new = model_new(
            [[s.clone() for s in way] for way in supp_imgs], 
            [[s.clone() for s in way] for way in supp_mask], 
            [q.clone() for q in qry_imgs], 
            qry_labels.clone(), 
            train=True, use_skeleton=False
        )
    
    for i, (l_o, l_n) in enumerate(zip(loss_orig, loss_new)):
        torch.testing.assert_close(l_o, l_n, atol=1e-5, rtol=1e-5, msg=f"loss {i} mismatch")
        
    print("Training parity: PASSED")

if __name__ == "__main__":
    test_parity()
