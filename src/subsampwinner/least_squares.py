"""Owned boundary for conventional full-rank OLS and inverse-variance WLS.

Statsmodels is the sole initial backend. Robust covariance, estimated-weight
inference, frequency weights, and singular-design pseudoinverse fits are outside
this contract. Multiple-testing correction belongs to selection, not fitting.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import statsmodels.api as sm

from ._validation import validate_data


@dataclass(frozen=True)
class LeastSquaresResult:
    """Regression statistics include the intercept at position zero.

``rss`` is unweighted; ``weighted_rss`` estimates the conventional WLS scale
when divided by ``df_resid``. ``model`` preserves statsmodels compatibility.
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


def fit_least_squares(X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None,
                      feature_names: Optional[Sequence[str]] = None) -> LeastSquaresResult:
    """Fit with an explicit intercept, positive residual df and full numerical rank.

Rank tolerance is ``max(n, k) * eps * largest_singular_value`` on the weighted
actual fitted design, not the full candidate matrix. Numerically perfect fits
are rejected using a scale-relative residual roundoff bound, never an RSS floor.
    """
    X, y, weights = validate_data(X, y, weights)
    design = np.column_stack((np.ones(len(y)), X))
    n, k = design.shape
    if n <= k:
        raise ValueError("least-squares fit requires positive residual degrees of freedom")
    root_weights = np.ones(n) if weights is None else np.sqrt(weights)
    weighted_design = design * root_weights[:, None]
    weighted_y = y * root_weights
    if not np.all(np.isfinite(weighted_design)) or not np.all(np.isfinite(weighted_y)):
        raise ValueError("weighted design or response overflowed")
    backend = sm.OLS(y, design) if weights is None else sm.WLS(y, design, weights=weights)
    if feature_names is not None:
        if len(feature_names) != X.shape[1]:
            raise ValueError("feature_names must have one name per fitted predictor")
        backend.exog_names[1:] = list(feature_names)
    model = backend.fit(method="pinv", cov_type="nonrobust", use_t=True)
    # The backend already factors the weighted design. Reuse its singular values
    # while applying OUR rank policy; no second SVD or silent pseudoinverse fit.
    singular_values = backend.wexog_singular_values
    tolerance = max(n, k) * np.finfo(float).eps * singular_values[0]
    if singular_values[-1] <= tolerance:
        raise ValueError("fitted design is numerically rank deficient")
    if model.df_resid != n - k:
        raise ValueError("backend rank disagrees with the full-rank contract")
    residuals = np.asarray(model.resid)
    weighted_residuals = residuals * root_weights
    roundoff = max(n, k) * np.finfo(float).eps * (
        np.linalg.norm(weighted_y) + singular_values[0] * np.linalg.norm(model.params))
    if np.linalg.norm(weighted_residuals) <= roundoff:
        raise ValueError("perfect or numerically perfect fit has undefined residual-scale inference")
    rss = float(np.dot(residuals, residuals))
    weighted_rss = float(np.dot(weighted_residuals, weighted_residuals))
    arrays = [np.array(a, dtype=float, copy=True) for a in
              (model.params, model.bse, model.tvalues, model.pvalues, residuals)]
    if (not all(np.all(np.isfinite(a)) for a in arrays) or not np.isfinite(rss) or
            not np.isfinite(weighted_rss) or rss <= 0 or weighted_rss <= 0 or np.any(arrays[1] <= 0)):
        raise ValueError("least-squares fit produced invalid statistics or residual scale")
    return LeastSquaresResult(*arrays, rss, weighted_rss, n - k, model)
