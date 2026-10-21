"""
Test Query Ground Truth Invariance
===================================
Verifies that query-time inference does NOT depend on query ground-truth masks.
Prediction, prompt generation, and budget allocation must remain strictly invariant
regardless of what is passed in query label arguments during evaluation.
"""

import os
import sys
import numpy as np
import torch
import pytest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "third_party", "FoB_SAM"))

from models.FoB import FewShotSeg
from models.allocator import PromptBudgetAllocator

def test_allocator_signature_has_no_query_gt():
    """Verify that PromptBudgetAllocator.allocate does not take query GT."""
    import inspect
    sig = inspect.signature(PromptBudgetAllocator.allocate)
    param_names = list(sig.parameters.keys())
    assert "query_gt" not in param_names
    assert "qry_gt" not in param_names
    assert "query_label" not in param_names
    assert "qry_labels" not in param_names

def test_inference_query_label_invariance():
    """
    Test that FewShotSeg forward pass in evaluation mode (train=False) 
    produces identical predictions when given different query labels.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for FewShotSeg execution")

    class DummyArgs:
        dataset = "SABS"
        max_points = 24
        n_ways = 1
        n_shots = 1
        
    model = FewShotSeg(DummyArgs()).cuda().eval()
    
    # Generate synthetic inputs on CUDA
    torch.manual_seed(42)
    B, C, H, W = 1, 3, 256, 256
    supp_img = [[torch.randn(1, C, H, W).cuda()]]
    supp_mask = [[torch.zeros(1, 1, H, W).cuda()]]
    supp_mask[0][0][:, :, 50:100, 50:100] = 1.0  # valid fg mask
    
    qry_img = [torch.randn(1, C, H, W).cuda()]
    
    # Query label A: dummy foreground
    qry_label_A = torch.zeros(1, H, W).long().cuda()
    qry_label_A[:, 60:90, 60:90] = 1
    
    # Query label B: completely different foreground mask
    qry_label_B = torch.zeros(1, H, W).long().cuda()
    qry_label_B[:, 120:180, 120:180] = 1

    with torch.no_grad():
        torch.manual_seed(100)
        np.random.seed(100)
        out_A = model(supp_img, supp_mask, qry_img, qry_label_A, train=False, budget_Np=10)

        torch.manual_seed(100)
        np.random.seed(100)
        out_B = model(supp_img, supp_mask, qry_img, qry_label_B, train=False, budget_Np=10)

    neg_A, pos_A = out_A[0], out_A[1]
    neg_B, pos_B = out_B[0], out_B[1]

    # Prompt points must be completely identical
    assert np.allclose(neg_A, neg_B, atol=1e-5), "Negative prompts changed when query GT changed!"
    assert np.allclose(pos_A, pos_B, atol=1e-5), "Positive prompts changed when query GT changed!"
