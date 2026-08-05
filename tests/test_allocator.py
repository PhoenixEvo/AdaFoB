import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from third_party.FoB_SAM.models.FoB import FewShotSeg
from types import SimpleNamespace

def test_allocator():
    dummy_args = SimpleNamespace(
        max_points=24,
        budget_Np=10,
        n_ways=1,
        n_shots=1
    )
    model = FewShotSeg(dummy_args).cuda().eval()
    
    # Dummy inputs
    supp_imgs = [[torch.randn(1, 3, 256, 256).cuda()]]
    supp_mask = [[torch.zeros(1, 1, 256, 256).cuda()]]
    # Make a dummy foreground mask
    supp_mask[0][0][0, 0, 100:150, 100:150] = 1.0
    
    qry_imgs = [torch.randn(1, 3, 256, 256).cuda()]
    qry_labels = torch.zeros(1, 1, 256, 256).cuda()
    qry_labels[0, 0, 100:150, 100:150] = 1.0
    
    try:
        with torch.no_grad():
            out = model(supp_imgs, supp_mask, qry_imgs, qry_labels, train=False)
            print("Allocator ran successfully!")
            print("neg_point shape:", out[1].shape)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_allocator()
