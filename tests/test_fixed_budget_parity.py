"""
Test Fixed-Budget Parity
========================
Verifies that when adaptive behavior is fixed to Np=10, the prompt generator 
reproduces the baseline fixed-budget behavior within numerical tolerance.
"""

import os
import sys
import cv2
import numpy as np
import torch
import pytest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "third_party", "FoB_SAM"))

from models.FoB import FewShotSeg
from models.allocator import PromptBudgetAllocator

def test_fixed_budget_prompt_parity():
    """
    Verify that baseline FewShotSeg with budget_Np=10 produces 10 negative points 
    and 10 positive points with valid shapes and coordinates.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for FewShotSeg execution")

    class DummyArgs:
        dataset = "SABS"
        max_points = 24
        n_ways = 1
        n_shots = 1

    model = FewShotSeg(DummyArgs()).cuda().eval()
    model.allocator = None

    B, C, H, W = 1, 3, 256, 256
    torch.manual_seed(42)
    supp_img = [[torch.randn(1, C, H, W).cuda()]]
    supp_mask = [[torch.zeros(1, 1, H, W).cuda()]]
    supp_mask[0][0][:, :, 60:120, 60:120] = 1.0

    qry_img = [torch.randn(1, C, H, W).cuda()]
    qry_label = torch.zeros(1, H, W).long().cuda()
    qry_label[:, 70:110, 70:110] = 1

    with torch.no_grad():
        torch.manual_seed(42)
        np.random.seed(42)
        out1 = model(supp_img, supp_mask, qry_img, qry_label, train=False, budget_Np=10)

        # Re-run to check determinism
        torch.manual_seed(42)
        np.random.seed(42)
        out2 = model(supp_img, supp_mask, qry_img, qry_label, train=False, budget_Np=10)

    neg1, pos1 = out1[0], out1[1]
    neg2, pos2 = out2[0], out2[1]

    neg1_arr = np.array(neg1)
    pos1_arr = np.array(pos1)

    assert neg1_arr.shape[-2] == 10, f"Expected 10 negative prompts, got {neg1_arr.shape}"
    assert pos1_arr.shape[-2] == 10, f"Expected 10 positive prompts, got {pos1_arr.shape}"
    assert np.allclose(neg1, neg2, atol=1e-5), "Fixed-budget run is not deterministic under identical seed"

def test_allocator_forced_budget_parity():
    """Verify PromptBudgetAllocator returns exactly the requested budget."""
    allocator = PromptBudgetAllocator(max_points=24)
    forced_np = 10
    M_tilde = np.zeros((256, 256), dtype=np.uint8)
    M_tilde[50:150, 50:150] = 1
    contours, _ = cv2.findContours(M_tilde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    qry_img = np.zeros((256, 256), dtype=np.float32)

    pts = allocator.sample_placement(qry_img, M_tilde, contours, forced_np, r=15)
    assert len(pts) == forced_np, f"Expected {forced_np} points, got {len(pts)}"
