"""Valid numerical references and explicit characterization of adaptations.

Known defects are recorded in reference/legacy_probes.json, not asserted here
as behavior that a repaired implementation must preserve.
"""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.stats import t as student_t

from subsampwinner import SubsampDoubleAssurance, subsamp


REFERENCE = json.loads((Path(__file__).parent / "reference/fixed_schedule.json").read_text())["observations"]


def data():
    return tuple(np.array(REFERENCE[key]) for key in ("X", "y", "weights", "schedule"))


def least_squares_reference(X, y, weights=None):
    """Independent full-rank OLS/WLS calculation for the tiny fixture only."""
    design = np.column_stack([np.ones(len(y)), X])
    root_w = np.ones(len(y)) if weights is None else np.sqrt(weights)
    weighted_design = design * root_w[:, None]
    coefficients = np.linalg.lstsq(weighted_design, y * root_w, rcond=None)[0]
    residuals = y - design @ coefficients
    variance = np.sum((residuals * root_w) ** 2) / (len(y) - design.shape[1])
    covariance = variance * np.linalg.inv(weighted_design.T @ weighted_design)
    tvalues = coefficients / np.sqrt(np.diag(covariance))
    pvalues = 2 * student_t.sf(np.abs(tvalues), len(y) - design.shape[1])
    return coefficients, tvalues, pvalues, np.sum(residuals ** 2)


@pytest.mark.parametrize("label", ["ols", "wls"])
def test_submodels_match_frozen_values_and_independent_least_squares(label):
    X, y, weights, schedule = data()
    weights = weights if label == "wls" else None
    swa = subsamp(s=3, m=len(schedule), qnum=3)
    for row, frozen in zip(schedule, REFERENCE["cases"][label]["submodels"]):
        result = swa._analyze_subsample(X, y, row, weights)
        _, tvalues, _, rss = least_squares_reference(X[:, row], y, weights)
        assert_allclose(result["rss"], rss, rtol=1e-11)
        assert_allclose(result["t_vals"][row], tvalues[1:], rtol=1e-11, atol=1e-12)
        assert_allclose(result["t_vals"], frozen["t_vals"], rtol=1e-11, atol=1e-12)
        assert_allclose(result["rss"], frozen["rss"], rtol=1e-11)
        assert_array_equal(result["t_vals"][np.setdiff1d(np.arange(6), row)], 0)


@pytest.mark.parametrize("label", ["ols", "wls"])
@pytest.mark.parametrize("top", [False, True])
def test_fixed_schedule_scores_selection_and_semifinalist_model(label, top):
    X, y, weights, schedule = data()
    weights = weights if label == "wls" else None
    swa = subsamp(s=3, m=len(schedule), qnum=3)
    results = [swa._analyze_subsample(X, y, row, weights) for row in schedule]
    semi, scores = swa.process_subsample_results(results, 3, use_top_s=top)
    finalists, adjusted, model = swa.select_finalists(X, y, semi, weights=weights)
    frozen = REFERENCE["cases"][label]["selections"]["top_s" if top else "all"]
    assert_allclose(scores, frozen["scores"], rtol=1e-11, atol=1e-12)
    assert_array_equal(semi, frozen["semifinalists"])
    assert_array_equal(finalists, frozen["finalists"])
    assert_allclose(adjusted, frozen["p_adjusted"], rtol=1e-10, atol=1e-14)
    params, _, pvalues, _ = least_squares_reference(X[:, semi], y, weights)
    assert_allclose(model.params, params, rtol=1e-11, atol=1e-12)
    assert_allclose(model.params, frozen["params"], rtol=1e-11, atol=1e-12)
    assert_allclose(model.pvalues, pvalues, rtol=1e-10, atol=1e-14)
    assert model.df_resid == frozen["df_resid"] == len(y) - 4
    assert len(model.params) == len(semi) + 1
    assert len(finalists) < len(semi)  # No refit on the smaller finalist set.
    assert model.model.exog_names == ["const"] + [f"v{i + 1}" for i in semi]


def test_score_formula_nonzero_counts_and_retained_models():
    results = [
        {"indices": np.array([0, 1]), "t_vals": np.array([0., 2., 0., 0.]), "rss": 1.},
        {"indices": np.array([0, 2]), "t_vals": np.array([4., 0., 3., 0.]), "rss": 4.},
        {"indices": np.array([0, 1]), "t_vals": np.array([2., 6., 0., 0.]), "rss": 9.},
    ]
    swa = subsamp(s=2, m=3, qnum=2)
    # Feature 0 appears three times but has two nonzero t statistics.
    assert_allclose(swa._compute_feature_weights(results), [4/3, 2., 1.5, 0.])
    assert_allclose(swa._compute_feature_weights(results, option_s=2), [2., 2., 1.5, 0.])
    assert_allclose(swa._compute_feature_weights(results, option_s=4), [4/3, 2., 1.5, 0.])


def test_rss_exponents_can_reverse_a_ranking():
    # An algebraic example, not an empirical comparison or new algorithm.
    tvalues, rss = np.array([1., 3.]), np.array([1., 4.])
    assert np.argmax(tvalues / np.sqrt(rss)) == 1
    assert np.argmax(tvalues / rss) == 0


@pytest.mark.parametrize("method, expected, finalists", [
    ("bonferroni", [.03, .12, .6], [2]),
    ("fdr_bh", [.03, .06, .2], [2]),
    ("none", [.01, .04, .2], [2, 4]),
])
def test_final_filter_excludes_intercept_and_maps_feature_indices(method, expected, finalists):
    model = SimpleNamespace(pvalues=np.array([.0001, .01, .04, .2]))
    selected, adjusted = subsamp(s=3, m=4)._select_finalists(model, [2, 4, 6], method=method)
    assert selected == finalists
    assert_allclose(adjusted, expected)


def test_seeded_serial_fit_repeats_and_uses_distinct_columns_within_submodels():
    X, y, _, _ = data()
    swa = subsamp(s=3, m=12, qnum=3, seed=7)
    swa.fit(X, y)
    first_scores, first_finalists = swa.feature_weights.copy(), swa.finalists.copy()
    swa.fit(X, y)
    assert_array_equal(swa.feature_weights, first_scores)
    assert swa.finalists == first_finalists
    # Test the structural sampling rule, not an RNG-specific sequence.
    for result in swa.subsample_anlysis(X, y):
        assert len(np.unique(result["indices"])) == 3
        assert np.all((result["indices"] >= 0) & (result["indices"] < X.shape[1]))


def test_assurance_cumulative_union_and_largest_set_update(monkeypatch):
    """Characterize this adaptation; the union is not a stable consensus."""
    calls = []
    selections = {3: {1, 5}, 4: {0, 1}, 5: {1, 2, 3}, 6: {1, 4}}

    class FixedSelection:
        def __init__(self, s, **kwargs):
            self.s = s

        def fit(self, *args, **kwargs):
            calls.append(self.s)
            self.finalists = selections[self.s]

    module = importlib.import_module("subsampwinner.SubsampDoubleAssurance")
    monkeypatch.setattr(module, "subsamp", FixedSelection)
    X, y, _, _ = data()
    result = SubsampDoubleAssurance(m=12, qnum=3, seed=7).double_assurance(
        X, y, s0=4, T=.9, I_max=2, init_range=.25, r=.75, return_objects=True)
    assert calls == [4, 3, 5, 5, 4, 6]
    assert result == {"best_s0": 4, "final_range": [4, 5, 6],
                      "stabilized_union_set": {0, 1, 2, 3, 4, 5}}


def test_assurance_threshold_stops_after_first_triplet(monkeypatch):
    calls = []

    class FixedSelection:
        def __init__(self, s, **kwargs):
            self.s = s

        def fit(self, *args, **kwargs):
            calls.append(self.s)
            self.finalists = [0, 1]

    module = importlib.import_module("subsampwinner.SubsampDoubleAssurance")
    monkeypatch.setattr(module, "subsamp", FixedSelection)
    X, y, _, _ = data()
    result = SubsampDoubleAssurance(m=12, qnum=3, seed=7).double_assurance(
        X, y, s0=4, T=.8, I_max=3, init_range=.25, return_objects=True)
    assert calls == [4, 3, 5]
    assert result["best_s0"] == 4
    assert result["stabilized_union_set"] == {0, 1}
