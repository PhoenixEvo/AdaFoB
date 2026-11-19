"""Selectors and arm definitions for AdaFoB v2 Stage 3 CHAOST2 cross-modality evaluation."""
from __future__ import annotations

from experiments.adafob_v2_stage2.selectors import select_best_candidate
from experiments.option_a_reassessment_v1.selectors import (
    compute_support_evidence,
    mask_on_feature_grid,
    place_pba_fixed_k2,
    select_response_arms,
)

RANDOM_SEEDS = (42, 43, 44, 45, 46)

RAW_ARMS = (
    "full10",
    "positive0",
    "random2_s42",
    "random2_s43",
    "random2_s44",
    "random2_s45",
    "random2_s46",
    "boundary_near2",
    "resp_signed_localring_k2",
    "pba_f10_int_k2",
    "cbv_f10_fixed_supp",
    "cbv_f10_no0_supp",
)

SUMMARY_ARMS = (
    "full10",
    "positive0",
    "random2_mean5",
    "boundary_near2",
    "resp_signed_localring_k2",
    "pba_f10_int_k2",
    "cbv_f10_fixed_supp",
    "cbv_f10_no0_supp",
)

METHOD_ARMS = (
    "resp_signed_localring_k2",
    "pba_f10_int_k2",
    "cbv_f10_fixed_supp",
    "cbv_f10_no0_supp",
)

ALL_REPORTED_ARMS = (
    "full10",
    "positive0",
    "random2_mean5",
    "random2_s42",
    "random2_s43",
    "random2_s44",
    "random2_s45",
    "random2_s46",
    "boundary_near2",
    "resp_signed_localring_k2",
    "pba_f10_int_k2",
    "cbv_f10_fixed_supp",
    "cbv_f10_no0_supp",
)

__all__ = [
    "ALL_REPORTED_ARMS",
    "METHOD_ARMS",
    "RANDOM_SEEDS",
    "RAW_ARMS",
    "SUMMARY_ARMS",
    "compute_support_evidence",
    "mask_on_feature_grid",
    "place_pba_fixed_k2",
    "select_best_candidate",
    "select_response_arms",
]
