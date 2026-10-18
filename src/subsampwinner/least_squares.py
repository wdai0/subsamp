"""Conventional full-rank OLS/WLS through one statsmodels fitting engine.

Public results own backend-retained inputs. Internal subset fits reuse validated
coordinator data and extract only the statistics needed for SWA scoring.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from scipy.linalg import norm
from statsmodels.regression.linear_model import OLS, WLS

from ._validation import validate_data

# statsmodels.tools.tools.pinv_extended's default cutoff. Acceptance must be at
# least this strict even for tiny designs whose dimension-based cutoff is lower.
_PINV_RCOND = 1e-15


@dataclass(frozen=True)
class LeastSquaresResult:
    """Statistics include the intercept; RSS is raw even for WLS.

``weighted_rss / df_resid`` is the conventional WLS residual-scale estimate.
``model`` preserves statsmodels compatibility, including genuine p-values.
    """

    coefficients: np.ndarray
    standard_errors: np.ndarray
    t_values: np.ndarray
    p_values: np.ndarray
    residuals: np.ndarray
    rss: float
    weighted_rss: float
    df_resid: int
    model: Any


@dataclass(frozen=True)
class _SubmodelResult:
    """Only signed slope t-values and scale metadata survive a subset fit."""

    t_values: np.ndarray
    rss: float
    weighted_rss: float
    df_resid: int


def _fit_validated(X, y, weights=None, feature_names=None):
    """One engine and acceptance policy for already validated, owned data."""
    design = np.column_stack((np.ones(len(y)), X))
    n, k = design.shape
    if n <= k:
        raise ValueError("least-squares fit requires positive residual degrees of freedom")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        backend = OLS(y, design) if weights is None else WLS(y, design, weights=weights)
        if not np.all(np.isfinite(backend.wexog)) or not np.all(np.isfinite(backend.wendog)):
            raise ValueError("numerical range exceeded in weighted design or response")
        if feature_names is not None:
            if len(feature_names) != X.shape[1]:
                raise ValueError("feature_names must have one name per fitted predictor")
            backend.exog_names[1:] = list(feature_names)
        model = backend.fit(method="pinv", cov_type="nonrobust", use_t=True)
        singular_values = backend.wexog_singular_values
        relative_tolerance = max(max(n, k) * np.finfo(float).eps, _PINV_RCOND)
        if singular_values[-1] <= relative_tolerance * singular_values[0]:
            raise ValueError("fitted design is numerically rank deficient")
        if model.df_resid != n - k:
            raise ValueError("backend rank disagrees with the full-rank contract")
        residuals = model.resid
        weighted_residuals = residuals if weights is None else residuals * np.sqrt(weights)
        rss = float(np.dot(residuals, residuals))
        weighted_rss = rss if weights is None else float(np.dot(weighted_residuals, weighted_residuals))
        # BLAS nrm2 avoids intermediate squaring overflow. Normalize the bound
        # before multiplying: neither ||y|| + s_max*||beta|| nor its product
        # needs to be represented in floating point.
        y_norm, beta_norm, residual_norm = (
            norm(a, check_finite=False) for a in (backend.wendog, model.params, weighted_residuals))
        if not np.all(np.isfinite([rss, weighted_rss, y_norm, beta_norm, residual_norm])):
            raise ValueError("numerical range exceeded in regression statistics or residual scale")
        scale = max(y_norm, beta_norm)
        eps_dimension = max(n, k) * np.finfo(float).eps
        bound = 0. if scale == 0 else (
            eps_dimension * (y_norm / scale) +
            (eps_dimension * singular_values[0]) * (beta_norm / scale))
        if scale == 0 or residual_norm / scale <= bound:
            raise ValueError("perfect or numerically unresolved fit at the response/design scale; "
                             "residual-scale inference is unreliable (large offsets can trigger this bound)")
        if (rss <= 0 or weighted_rss <= 0 or np.any(model.bse <= 0) or
                not all(np.all(np.isfinite(a)) for a in (model.params, model.bse, model.tvalues, residuals))):
            raise ValueError("least-squares fit produced invalid statistics or residual scale")
    return model, rss, weighted_rss, n - k


def _fit_submodel(X, y, weights=None):
    """Private path: inputs must already be validated and owned by the caller."""
    model, rss, weighted_rss, df = _fit_validated(X, y, weights)
    return _SubmodelResult(model.tvalues[1:].copy(), rss, weighted_rss, df)


def fit_least_squares(X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None,
                      feature_names: Optional[Sequence[str]] = None) -> LeastSquaresResult:
    """Own inputs and fit an intercept with positive residual df and full rank.

Rank cutoff is max(max(n,k)*eps, 1e-15) relative to the largest singular value
of the actual weighted design, consistent with the backend pseudoinverse.
The conservative scale-relative residual bound can reject large offsets.
    """
    X, y, weights = validate_data(X, y, weights)
    # column_stack in the engine owns X; the backend otherwise aliases y/weights.
    model, rss, weighted_rss, df = _fit_validated(
        X, y.copy(), None if weights is None else weights.copy(), feature_names)
    arrays = [np.array(a, dtype=float, copy=True) for a in
              (model.params, model.bse, model.tvalues, model.pvalues, model.resid)]
    if not np.all(np.isfinite(arrays[3])):
        raise ValueError("least-squares fit produced invalid p-values")
    return LeastSquaresResult(*arrays, rss, weighted_rss, df, model)
