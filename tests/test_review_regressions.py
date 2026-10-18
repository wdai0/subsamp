"""Reproductions from the external c881315 review, independent of golden data."""

import builtins
import inspect

import numpy as np
from numpy.testing import assert_allclose
import pytest
from statsmodels.regression.linear_model import OLS, WLS, RegressionResults
from statsmodels.tools.tools import pinv_extended

from subsampwinner import fit_least_squares, subsamp, SubsampDoubleAssurance
from subsampwinner._core import analyze_schedule
from subsampwinner.least_squares import _PINV_RCOND
from subsampwinner.scoring import score_features, SubsampleStatistics


def data():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(40, 2))
    return X, 1e5 + X @ np.array([2., 3.]) + rng.normal(size=40)


def test_rank_policy_covers_backend_truncation():
    assert _PINV_RCOND >= inspect.signature(pinv_extended).parameters['rcond'].default
    X = (1 + 2e-15 * np.array([-1., 0., 1.]))[:, None]
    y = np.array([99.5, 99., 101.5])
    for operation in (lambda: fit_least_squares(X, y), lambda: analyze_schedule(X, y, [[0]])):
        with pytest.raises(ValueError, match='rank deficient'):
            operation()


@pytest.mark.parametrize('weighted', [False, True])
def test_extreme_scaling_retains_finite_backend_inference(weighted):
    X, y = data()
    weights = np.full(len(y), 1e300) if weighted else None
    y = y if weighted else y * 1e150
    design = np.column_stack((np.ones(len(y)), X))
    backend = (OLS(y, design) if weights is None else WLS(y, design, weights=weights)).fit()
    result = fit_least_squares(X, y, weights)
    compact = analyze_schedule(X, y, [[0, 1]], weights)
    for actual, expected in ((result.coefficients, backend.params), (result.p_values, backend.pvalues),
                             (result.t_values, backend.tvalues), (result.weighted_rss, backend.ssr),
                             (compact.t_values[0], backend.tvalues[1:]), (compact.rss[0], result.rss)):
        assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
    assert np.isfinite(result.rss) and np.isfinite(result.weighted_rss)


def test_unrepresentable_residual_scale_is_not_reported_as_perfect():
    X, y = data()
    with pytest.raises(ValueError, match='numerical range'):
        fit_least_squares(X, (y - y.mean()) * 1e160)


def test_conservative_large_offset_diagnosis():
    X, y = data()
    with pytest.raises(ValueError, match='large offsets'):
        fit_least_squares(X, y + 1e15)


@pytest.mark.parametrize('stage', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_public_fit_owns_inputs_before_uncached_diagnostic(stage, weighted):
    X, y = data()
    y -= 1e5
    weights = np.linspace(1., 2., len(y)) if weighted else None
    reference = fit_least_squares(X, y, weights).model.rsquared
    if stage:
        model = subsamp(1, 2).select_finalists(X, y, [0, 1], weights)[2]
    else:
        model = fit_least_squares(X, y, weights).model
    assert 'rsquared' not in model._results._cache
    X[:] = 0
    y *= 10
    if weights is not None:
        weights[:] = 100
    assert_allclose(model.rsquared, reference)


@pytest.mark.parametrize('output', [np.array([True, False]), np.array(['1', '2']),
                                    np.array([1, 2], dtype=object),
                                    np.array(['2026-01-01', '2026-02-01'], dtype='datetime64[D]'),
                                    np.array([1j, 2j])])
def test_scores_reject_non_real_numeric_dtypes(output):
    statistics = SubsampleStatistics([[0, 1]], [[1., 2.]], [3.], [0], 2)
    with pytest.raises(ValueError, match='dtype'):
        score_features(statistics, lambda _: output)


@pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
def test_scores_accept_owned_real_numeric_outputs(dtype):
    statistics = SubsampleStatistics([[0, 1]], [[1., 2.]], [3.], [0], 2)
    source = np.array([1, 2], dtype=dtype)
    scores = score_features(statistics, lambda _: source)
    source[:] = 100
    assert_allclose(scores, [1, 2])
    assert_allclose(score_features(statistics, lambda _: np.array([-2., -1.])), [-2., -1.])


@pytest.mark.parametrize('options', [dict(parallel='False'), dict(use_top_s='False'),
                                     dict(parallel=True, use_top_s='False'), dict(parallel=1)])
def test_flags_rejected_before_fitting_or_mpi_import(options, monkeypatch):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name == '_parallel' or name.startswith('mpi4py'):
            raise AssertionError('MPI must not be imported for invalid flags')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    with pytest.raises(ValueError, match='Boolean'):
        subsamp(1, 2).fit(None, None, **options)


def test_stage_flags_and_repetition_ids_fail_at_boundary():
    swa = subsamp(1, 2)
    with pytest.raises(ValueError, match='Boolean'):
        swa.subsample_anlysis(None, None, parallel='False')
    with pytest.raises(ValueError, match='Boolean'):
        swa.process_subsample_results([], 1, use_top_s='False')
    with pytest.raises(ValueError, match='Boolean'):
        SubsampDoubleAssurance(2).double_assurance(None, None, 1, return_objects='False')
    X, y = data()
    with pytest.raises(ValueError, match='repetition_ids'):
        analyze_schedule(X, y, [[0], [1]], repetition_ids=[1])
    swa.fit(X, y, parallel=np.bool_(False), use_top_s=np.bool_(True))


def test_submodel_path_does_not_compute_pvalues(monkeypatch):
    X, y = data()
    def unused(self):
        raise AssertionError('intermediate p-values are unused')
    monkeypatch.setattr(RegressionResults, 'pvalues', property(unused))
    statistics = analyze_schedule(X, y, [[0], [1]])
    assert np.isfinite(statistics.t_values).all()


def test_collapsed_assurance_range_label(monkeypatch):
    import importlib
    module = importlib.import_module('subsampwinner.SubsampDoubleAssurance')
    class FixedSelection:
        def __init__(self, **kwargs):
            self.finalists = [0]
        def fit(self, *args, **kwargs):
            pass
    monkeypatch.setattr(module, 'subsamp', FixedSelection)
    X, y = data()
    assurance = SubsampDoubleAssurance(2)
    assurance.double_assurance(X, y, 1, init_range=0, return_objects=True)
    assert assurance.termination_reason_ == 'collapsed_range'
