"""Unit tests for experiments/reconcile_stats.py against hand-computed values.

Run:  python -m pytest -q tests/test_reconcile_stats.py   (or python tests/test_reconcile_stats.py)
"""
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from experiments.reconcile_stats import (  # noqa: E402
    describe, holm_adjust, cluster_bootstrap_means, boot_p_two_sided,
    tost_from_boot, tost_paired_t, wilcoxon_p, load_results,
)


def test_describe_hand_computed():
    d = describe(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert d["mean"] == 3.0 and d["median"] == 3.0
    assert d["q1"] == 2.0 and d["q3"] == 4.0 and d["iqr"] == 2.0


def test_holm_hand_computed():
    # sorted p: 0.01, 0.03, 0.04 -> 3*0.01=0.03, 2*0.03=0.06, max(0.06, 1*0.04)=0.06
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    # sorted p: 0.5, 0.9 -> 2*0.5=1.0, then monotone max(1.0, 1*0.9)=1.0 -> both 1.0
    assert np.allclose(holm_adjust([0.5, 0.9]), [1.0, 1.0])
    # single test is unchanged
    assert np.allclose(holm_adjust([0.0383]), [0.0383])


def test_cluster_bootstrap_constant_and_weighting():
    rng = np.random.default_rng(0)
    vals = np.array([2.0, 2.0, 2.0, 2.0])
    boot = cluster_bootstrap_means(vals, np.array([1, 1, 2, 3]), 200, rng)
    assert np.allclose(boot, 2.0)
    # Two clusters: cluster A has values {0,0,0}, cluster B has {4}. Any replicate mean is
    # a row-weighted mean of the drawn clusters, so it must lie in {0, 1, 4}:
    # (A,A)=0, (B,B)=4, (A,B)/(B,A)=4/4=1.
    boot = cluster_bootstrap_means(np.array([0.0, 0.0, 0.0, 4.0]), np.array([0, 0, 0, 1]), 500,
                                   np.random.default_rng(1))
    assert set(np.round(np.unique(boot), 9)).issubset({0.0, 1.0, 4.0})


def test_boot_p_and_tost():
    boot_pos = np.full(1000, 0.5)
    assert boot_p_two_sided(boot_pos) < 0.01
    boot_zero = np.zeros(1000)
    assert boot_p_two_sided(boot_zero) == 1.0
    t = tost_from_boot(np.full(1000, 0.002), margin=0.01)
    assert t["p_tost"] < 0.05
    t = tost_from_boot(np.full(1000, 0.02), margin=0.01)
    assert t["p_tost"] > 0.5
    assert tost_paired_t(np.full(10, 0.001), 0.01) == 0.0
    assert tost_paired_t(np.full(10, 0.02), 0.01) == 1.0


def test_wilcoxon_all_equal_returns_one():
    x = np.array([0.1, 0.2, 0.3])
    assert wilcoxon_p(x, x.copy()) == 1.0


def test_load_results_validation(tmp_path):
    rows = []
    for fold in (0, 1):
        for organ in ("A", "B"):
            for v in range(2):
                rows.append({"fold": fold, "organ": organ, "vol_id": f"/x/image_{fold * 10 + v}.nii",
                             "dice_baseline": 0.5, "hd95_baseline": 1.0, "dice_lora_fob": 0.6,
                             "hd95_lora_fob": 1.0, "dice_lora_ada": 0.6, "hd95_lora_ada": 1.0})
    p = tmp_path / "ok.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    df = load_results(str(p), expected_rows=8)
    assert set(df["vid"]) == {0, 1, 10, 11}
    try:
        load_results(str(p), expected_rows=9)
        raise AssertionError("expected ValueError for wrong row count")
    except ValueError:
        pass


if __name__ == "__main__":
    import tempfile
    import pathlib
    test_describe_hand_computed()
    test_holm_hand_computed()
    test_cluster_bootstrap_constant_and_weighting()
    test_boot_p_and_tost()
    test_wilcoxon_all_equal_returns_one()
    with tempfile.TemporaryDirectory() as d:
        test_load_results_validation(pathlib.Path(d))
    print("All tests passed")
