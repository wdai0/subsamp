"""Independent arithmetic, statistical contracts and explicit repair checks."""

import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.stats import t as student_t

from subsampwinner import (SubsampDoubleAssurance, SubsampleStatistics, default_scorer,
                          fit_least_squares, subsamp)
from subsampwinner._core import analyze_schedule, filter_pvalues, sample_schedule
from subsampwinner.GenerateData import generate_heteroskedastic_data
from subsampwinner.scoring import rank_features, retain_submodels


def data():
    rng = np.random.RandomState(31)
    X = rng.normal(size=(40, 8))
    return X, 2 + 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(size=40)


def test_analytic_orthogonal_ols_with_nonzero_intercept():
    x1 = np.repeat([-1., 1.], 4)
    x2 = np.tile(np.repeat([-1., 1.], 2), 2)
    residuals = .5 * np.tile([1., -1.], 4)
    fit = fit_least_squares(np.column_stack([x1, x2]), 2 + 3*x1 - 1.5*x2 + residuals)
    expected_beta = np.array([2., 3., -1.5])
    assert_allclose(fit.coefficients, expected_beta, atol=1e-14)
    assert_allclose(fit.residuals, residuals, atol=1e-14)
    assert_allclose(fit.rss, 2.)
    assert_allclose(fit.standard_errors, np.sqrt(.05))
    assert_allclose(fit.t_values, expected_beta / np.sqrt(.05))
    assert_allclose(fit.p_values, 2 * student_t.sf(np.abs(expected_beta) / np.sqrt(.05), 5))
    assert fit.df_resid == 5


def test_analytic_wls_raw_and_weighted_residual_scales():
    # A'WA=[[6,10],[10,24]], A'Wy=[15,32]; determinant 44.
    fit = fit_least_squares(np.arange(4.)[:, None], np.array([1., 2., 2., 4.]), np.array([1., 2., 1., 2.]))
    assert_allclose(fit.coefficients, [10/11, 21/22])
    assert_allclose(fit.residuals, [1/11, 3/22, -9/11, 5/22])
    assert_allclose(fit.rss, 181/242)
    assert_allclose(fit.weighted_rss, 9/11)
    assert fit.df_resid == 2
    assert_allclose(fit.standard_errors, np.sqrt(27) / np.array([11., 22.]))
    assert_allclose(fit.t_values, np.array([10., 21.]) / np.sqrt(27))
    assert_allclose(fit.p_values, 2*student_t.sf(np.array([10., 21.]) / np.sqrt(27), 2))


def test_review_hand_scores_and_raw_rss_retention():
    stats = SubsampleStatistics(np.array([[0, 1], [0, 2], [1, 3]]),
                                np.array([[4., -2.], [2., 6.], [8., 2.]]),
                                np.array([4., 9., 16.]), np.array([7, 1, 9]), 4,
                                np.full(3, 10), np.array([100., 50., 1.]))
    top = retain_submodels(stats, 2)
    assert_array_equal(top.repetition_ids, [7, 1])  # weighted RSS would choose [9,1].
    assert_allclose(default_scorer(top), [4/3, 1, 2, 0])
    assert_array_equal(rank_features(default_scorer(top), 2), [2, 0])
    assert_allclose(default_scorer(stats), [4/3, 1.5, 2, .5])
    assert_array_equal(rank_features(default_scorer(stats), 2), [2, 1])


def test_default_callback_parity_and_custom_negative_scores_own_ranking():
    X, y = data()
    ordinary, explicit = subsamp(3, 20, seed=7), subsamp(3, 20, seed=7, scorer=default_scorer)
    ordinary.fit(X, y)
    explicit.fit(X, y)
    assert_array_equal(ordinary.feature_weights, explicit.feature_weights)
    seen = []
    scores = np.arange(-8., 0.)

    def callback(statistics):
        seen.append(statistics)
        return scores

    custom = subsamp(3, 20, seed=7, scorer=callback)
    assert custom.fit(X, y, use_top_s=True) is None
    assert_array_equal(custom.semifinalists, [7, 6, 5])
    assert_array_equal(custom.feature_weights, scores)
    scores[:] = 500  # coordinator owns output; it did not normalize negative scores.
    assert_array_equal(custom.feature_weights, np.arange(-8., 0.))
    assert len(seen) == 1 and len(seen[0].rss) == 3
    assert seen[0].n_features == 8
    assert seen[0].feature_indices.shape == seen[0].t_values.shape == (3, 3)
    for value in (seen[0].rss, seen[0].t_values, seen[0].feature_indices, seen[0].repetition_ids):
        assert not value.flags.writeable


@pytest.mark.parametrize("bad", [np.zeros(7), np.zeros((8, 1)), np.full(8, np.nan),
                                  np.full(8, np.inf), np.ones(8, dtype=complex), None])
def test_invalid_callback_output_preserves_successful_state(bad):
    X, y = data()
    swa = subsamp(3, 12, seed=7)
    swa.fit(X, y)
    previous = {key: getattr(swa, key) for key in ("X", "y", "final_model", "semifinalists", "finalists",
                                                "feature_weights", "p_adjusted", "subsample_statistics_")}
    swa.scorer = lambda statistics: bad
    with pytest.raises(ValueError):
        swa.fit(X + .1, y)
    assert all(getattr(swa, key) is value for key, value in previous.items())


def test_callback_exception_and_final_rank_failure_are_atomic():
    X, y = data()
    swa = subsamp(3, 12, seed=7)
    swa.fit(X, y)
    model = swa.final_model

    def broken(statistics):
        raise RuntimeError("callback test error")

    swa.scorer = broken
    with pytest.raises(RuntimeError, match="callback test error"):
        swa.fit(X, y)
    assert swa.final_model is model
    # Individually estimable subsets; callback requests collinear finalists.
    singular = X.copy()
    singular[:, 1] = singular[:, 0]
    swa.s, swa.qnum = 1, 2
    swa.scorer = lambda statistics: np.r_[10., 9., np.zeros(6)]
    with pytest.raises(ValueError, match="rank deficient"):
        swa.fit(singular, y)
    assert swa.final_model is model


@pytest.mark.parametrize("kwargs", [dict(s=0), dict(m=0), dict(qnum=0), dict(s=1.5), dict(m=True),
                                     dict(seed=-1), dict(seed=2**32), dict(scorer=3)])
def test_constructor_rejects_invalid_parameters(kwargs):
    values = dict(s=3, m=12)
    values.update(kwargs)
    with pytest.raises(ValueError):
        subsamp(**values)


@pytest.mark.parametrize("mutation", ["nan_X", "nan_y", "wrong_y", "constant", "complex", "saturated",
                                       "zero_weight", "negative_weight", "matrix_weight", "nan_weight", "method", "alpha", "q_large"])
def test_invalid_fit_inputs_are_rejected_without_publishing_state(mutation):
    X, y = data()
    options, s, q = {}, 3, 3
    if mutation == "nan_X": X[0, 0] = np.nan
    elif mutation == "nan_y": y[0] = np.inf
    elif mutation == "wrong_y": y = y[:, None]
    elif mutation == "constant": X[:, 0] = 1
    elif mutation == "complex": X = X.astype(complex)
    elif mutation == "saturated": X, y, s = X[:4], y[:4], 3
    elif mutation == "zero_weight": options["weights"] = np.r_[0., np.ones(39)]
    elif mutation == "negative_weight": options["weights"] = -np.ones(40)
    elif mutation == "matrix_weight": options["weights"] = np.eye(40)
    elif mutation == "nan_weight": options["weights"] = np.full(40, np.nan)
    elif mutation == "method": options["method"] = "bonferoni"
    elif mutation == "alpha": options["alpha"] = 1.1
    elif mutation == "q_large": q = 9
    swa = subsamp(s, 12, q)
    with pytest.raises(ValueError):
        swa.fit(X, y, **options)
    assert swa.X is swa.y is swa.final_model is None


def test_rank_checked_per_fitted_subset_and_perfect_fit_rejected():
    X, y = data()
    X[:, 1] = X[:, 0]
    with pytest.raises(ValueError, match="repetition 0.*rank deficient"):
        analyze_schedule(X, y, np.array([[0, 1]]))
    with pytest.raises(ValueError, match="perfect"):
        fit_least_squares(X[:, [0]], 3 + 2*X[:, 0])
    # Intended p>n remains valid, even though full X is necessarily rank deficient.
    wide = np.random.RandomState(2).normal(size=(20, 100))
    swa = subsamp(2, 15, seed=3)
    swa.fit(wide, y[:20])
    assert swa.final_model.df_resid == 17


def test_global_rng_is_untouched_and_schedule_matches_legacy_seed():
    X, y = data()
    global_before = np.random.get_state()
    for seed in (7, None):
        subsamp(3, 12, seed=seed).fit(X, y)
    after = np.random.get_state()
    assert global_before[0] == after[0]
    assert_array_equal(global_before[1], after[1])
    assert global_before[2:] == after[2:]
    rng = np.random.RandomState(7)
    expected = np.array([rng.choice(8, 3, replace=False) for _ in range(12)])
    assert_array_equal(sample_schedule(8, 3, 12, 7), expected)
    # Frozen original seeded end-to-end observation.
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(60, 12))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(size=60)
    swa = subsamp(4, 80, 4, 7)
    assert swa.fit(X, y) is None
    assert_array_equal(swa.semifinalists, [0, 1, 10, 7])
    assert swa.finalists == [0, 1]


def test_shift_and_predictor_rescaling_invariance_on_fixed_schedule():
    X, y = data()
    schedule = sample_schedule(8, 3, 25, 6)
    original = analyze_schedule(X, y, schedule)
    scaled = analyze_schedule(X * np.array([3., -2., 4., 1., .5, 2., 1., 1.]), y + 17, schedule)
    assert_allclose(original.rss, scaled.rss, rtol=1e-12)
    assert_allclose(np.abs(original.t_values), np.abs(scaled.t_values), rtol=1e-11, atol=1e-12)
    assert_allclose(default_scorer(original), default_scorer(scaled), rtol=1e-11, atol=1e-12)
    assert_array_equal(rank_features(default_scorer(original), 3), rank_features(default_scorer(scaled), 3))


def test_exact_ties_near_ties_and_pvalue_boundaries():
    # Legacy descending quicksort for these tied arrays; no randomized tie breaks.
    assert_array_equal(rank_features(np.ones(4), 2), [3, 2])
    scores = np.array([1., np.nextafter(1., 2.), 0.])
    assert_array_equal(rank_features(scores, 1), [1])
    alpha = .05
    for method in ("bonferroni", "fdr_bh", "none"):
        at, _ = filter_pvalues([alpha], [3], alpha, method)
        below, _ = filter_pvalues([np.nextafter(alpha, 0.)], [3], alpha, method)
        above, _ = filter_pvalues([np.nextafter(alpha, 1.)], [3], alpha, method)
        assert at == ([] if method == "none" else [3])
        assert below == [3] and above == []
    tied = SubsampleStatistics(np.array([[0], [1], [2]]), np.ones((3, 1)), np.ones(3), np.array([9, 5, 3]), 3)
    assert_array_equal(retain_submodels(tied, 2).repetition_ids, [9, 5])


def test_serial_import_and_fit_with_mpi_import_explicitly_blocked():
    script = '''
import importlib.abc, sys
class NoMPI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('mpi4py'):
            raise ImportError('MPI intentionally unavailable')
sys.meta_path.insert(0, NoMPI())
from subsampwinner import subsamp
import numpy as np
X = np.random.RandomState(1).normal(size=(20, 5))
swa = subsamp(2, 3, seed=1)
swa.fit(X, np.arange(20.))
assert 'mpi4py' not in sys.modules
try:
    swa.fit(X, np.arange(20.), parallel=True)
except ImportError as error:
    assert 'subsampwinner[mpi]' in str(error)
else:
    raise AssertionError('expected optional dependency error')
'''
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    subprocess.run([sys.executable, "-c", script], env=environment, check=True, timeout=30, capture_output=True)


def test_assurance_later_empty_triplet_preserves_union_and_history(monkeypatch):
    calls = []
    responses = iter([{0}, {1, 2}, {2}, set(), set(), set()])

    class FixedSelection:
        def __init__(self, s, **kwargs): self.s = s
        def fit(self, *args, **kwargs):
            calls.append(self.s)
            self.finalists = next(responses)

    module = importlib.import_module("subsampwinner.SubsampDoubleAssurance")
    monkeypatch.setattr(module, "subsamp", FixedSelection)
    X, y = data()
    assurance = SubsampDoubleAssurance(10, qnum=2)
    result = assurance.double_assurance(X, y, 4, T=.9, I_max=3, init_range=.25, return_objects=True)
    assert calls == [4, 3, 5, 3, 2, 4]
    assert result["final_range"] == [2, 3, 4]
    assert result["stabilized_union_set"] == {0, 1, 2}
    assert assurance.termination_reason_ == "empty_selection"
    result["stabilized_union_set"].clear()
    assurance.history_[0]["cumulative_union"].clear()
    assert assurance.history_[1]["cumulative_union"] == {0, 1, 2}
    assert assurance.history_[0]["union"] == {0, 1, 2}


def test_assurance_last_range_and_invalid_bounds(monkeypatch):
    class FixedSelection:
        def __init__(self, s, **kwargs): self.s = s
        def fit(self, *args, **kwargs): self.finalists = list(range(self.s))
    module = importlib.import_module("subsampwinner.SubsampDoubleAssurance")
    monkeypatch.setattr(module, "subsamp", FixedSelection)
    X, y = data()
    assurance = SubsampDoubleAssurance(10, qnum=2)
    result = assurance.double_assurance(X, y, 4, T=1, I_max=1, init_range=.25, return_objects=True)
    assert result["final_range"] == [3, 4, 5]  # not [3,5,5] after the update.
    history = assurance.history_
    for kwargs in (dict(I_max=0), dict(s0=8), dict(s0=1)):
        with pytest.raises(ValueError):
            assurance.double_assurance(X, y, **dict(dict(s0=4), **kwargs))
        assert assurance.history_ is history
    result = assurance.double_assurance(X, y, 4, init_range=0, return_objects=True)
    assert result["final_range"] == [4, 4, 4]
    assert len(assurance.history_) == 1


def test_generator_preserves_diagonal_recipe_and_global_rng():
    before = np.random.get_state()
    X, y, variance, _ = generate_heteroskedastic_data(10, 3, lambda x: x + 1, seed=7)
    rng = np.random.RandomState(7)
    expected_X = rng.normal(size=(10, 3))
    signal = expected_X @ np.ones(3)
    expected_var = np.maximum(signal + abs(signal.min()) + 1, 1)
    expected_y = signal + rng.normal(np.zeros(10), np.sqrt(expected_var))
    assert_array_equal(X, expected_X)
    assert_array_equal(y, expected_y)
    assert_array_equal(variance, expected_var)
    after = np.random.get_state()
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    _, _, scalar_var, _ = generate_heteroskedastic_data(10, 3, lambda x: 2., seed=7)
    assert_array_equal(scalar_var, np.full(10, 2.))
    with pytest.raises(ValueError, match="positive semidefinite"):
        generate_heteroskedastic_data(30, 3, lambda x: np.ones(30), seed=7, type="unstructured")
    with pytest.raises(ValueError, match="nonnegative"):
        generate_heteroskedastic_data(10, 3, lambda x: -np.ones(10), seed=7)


def test_wls_common_weight_scale_and_rank_policy():
    X, y = data()
    weights = np.linspace(.2, 3., len(y))
    first = fit_least_squares(X[:, :3], y, weights)
    second = fit_least_squares(X[:, :3], y, weights * 100.)
    for field in ("coefficients", "standard_errors", "t_values", "p_values", "rss"):
        assert_allclose(getattr(first, field), getattr(second, field), rtol=1e-11, atol=1e-13)
    assert_allclose(second.weighted_rss, 100 * first.weighted_rss, rtol=1e-12)
    near_singular = np.column_stack([X[:, 0], X[:, 0] + 1e-16 * X[:, 1]])
    with pytest.raises(ValueError, match="rank deficient"):
        fit_least_squares(near_singular, y, weights)


def test_compatibility_helper_stream_advances_without_global_seed():
    swa = subsamp(3, 12, seed=7)
    rng = np.random.RandomState(7)
    for _ in range(3):
        assert_array_equal(swa._subsample_s_features(8, 3), rng.choice(8, 3, replace=False))


def test_malformed_legacy_summary_cannot_silently_drop_invalid_entries():
    swa = subsamp(1, 1)
    for t in ([1., np.nan], [1., 2.]):
        with pytest.raises(ValueError):
            swa._compute_feature_weights([dict(indices=[0], t_vals=t, rss=1.)])
