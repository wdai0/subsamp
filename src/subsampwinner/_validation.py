"""Shared input contracts, separate from the statistical algorithm."""

import numbers
import numpy as np


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_seed(seed):
    if seed is not None and (isinstance(seed, (bool, np.bool_)) or
                             not isinstance(seed, numbers.Integral) or not 0 <= seed < 2**32):
        raise ValueError("seed must be None or an integer in [0, 2**32)")


def finite_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_data(X, y, weights=None):
    X, y = finite_array(X, "X"), finite_array(y, "y")
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must be a nonempty two-dimensional matrix with at least two rows")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a one-dimensional vector with one entry per row of X")
    if np.any(np.all(X == X[0], axis=0)):
        raise ValueError("X must not contain constant columns; an intercept is added explicitly")
    if weights is not None:
        weights = finite_array(weights, "weights")
        if weights.shape != y.shape or np.any(weights <= 0):
            raise ValueError("weights must be a positive vector with one entry per observation")
    return X, y, weights


def validate_sizes(s, m, q, n, p):
    for value, name in ((s, "s"), (m, "m"), (q, "qnum")):
        positive_integer(value, name)
    if s > p or q > p:
        raise ValueError("s and qnum must not exceed the number of predictors")
    if s + 1 >= n or q + 1 >= n:
        raise ValueError("s and qnum must leave positive residual degrees of freedom including the intercept")


def validate_inference(alpha, method):
    if not isinstance(alpha, numbers.Real) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one")
    if method not in ("bonferroni", "fdr_bh", "none"):
        raise ValueError("method must be 'bonferroni', 'fdr_bh', or 'none'")


def feature_indices(indices, p):
    array = np.asarray(indices)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
        raise ValueError("feature indices must be a nonempty integer vector")
    if np.any(array < 0) or np.any(array >= p) or np.unique(array).size != array.size:
        raise ValueError("feature indices must be unique and within the original feature range")
    return array.astype(np.intp, copy=True)
