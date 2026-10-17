"""The permanent saved-data/saved-schedule panel runs with normal pytest."""

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from benchmarks.classic import BETA, covariance, selection_metrics, summarize
from subsampwinner import subsamp
from subsampwinner._core import analyze_schedule

DIRECTORY = Path(__file__).parent / "reference/classic"
MANIFEST = json.loads((DIRECTORY / "manifest.json").read_text())


@pytest.mark.parametrize("case", MANIFEST["cases"], ids=lambda case: case["id"])
def test_classic_saved_data_and_schedules(case):
    path = DIRECTORY / case["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
    with np.load(path, allow_pickle=False) as frozen:
        X, y = frozen["X"], frozen["y"]
        statistics = analyze_schedule(X, y, frozen["schedule"])
        assert_allclose(statistics.t_values, frozen["t_values"], rtol=1e-10, atol=1e-12)
        assert_allclose(statistics.rss, frozen["rss"], rtol=1e-11, atol=1e-12)
        assert_allclose(statistics.weighted_rss, frozen["rss"], rtol=1e-11, atol=1e-12)
        assert_array_equal(statistics.df_resid, 80 - case["s"] - 1)
        for mode, top in (("all", False), ("top_s", True)):
            swa = subsamp(case["s"], case["m"], case["q"])
            state = swa._finish(X, y, None, statistics, top, .05, "bonferroni")
            assert_allclose(state["feature_weights"], frozen[f"{mode}_scores"], rtol=1e-10, atol=1e-12)
            for key in ("semifinalists", "finalists"):
                assert_array_equal(state[key], frozen[f"{mode}_{key}"])
            assert_allclose(state["p_adjusted"], frozen[f"{mode}_p_adjusted"], rtol=1e-9, atol=1e-14)
            for key in ("params", "bse", "pvalues", "tvalues"):
                assert_allclose(getattr(state["final_model"], key), frozen[f"{mode}_{key}"], rtol=1e-9, atol=1e-13)
            assert selection_metrics(state["semifinalists"], state["finalists"]) == case["metrics"][mode]


def test_classic_covariances_population_snr_and_panel_coverage():
    expected = dict(independent=76.01, block11=179.188515625, full90=417.657110837)
    assert len(MANIFEST["cases"]) == 25
    assert sum(case["m"] == 5000 for case in MANIFEST["cases"]) == 1
    assert np.count_nonzero(BETA) == 10
    for name, signal_variance in expected.items():
        sigma = covariance(name)
        assert np.linalg.eigvalsh(sigma).min() > 0
        assert_allclose(BETA @ sigma @ BETA, signal_variance, rtol=1e-11)
    block = covariance("block11")
    assert_array_equal(block[11:, 11:], np.eye(89))
    assert_array_equal(block[:11, 11:], 0)
    for case in MANIFEST["cases"]:
        if "snr" in case:
            assert_allclose(BETA @ covariance(case["covariance"]) @ BETA / case["noise_sd"]**2, case["snr"])


def test_metric_definitions_and_failures_are_not_empty_selections():
    records = [selection_metrics(range(10), [0, 10, 11]), selection_metrics(range(10), [0, 1])]
    result = summarize(records)["selection_estimates"]
    assert_allclose(result["fdr"], 1 / 3)
    assert result["fwer"] == .5
    assert result["per_signal_power"] == [1., .5] + [0.] * 8
    failed = summarize(records + [dict(error="fit failed")])
    assert failed["failures"] == 1
    assert failed["selection_estimates"] is None
