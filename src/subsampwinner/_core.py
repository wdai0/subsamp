"""Statistical stages shared by serial and MPI orchestration."""

import numpy as np
from statsmodels.stats.multitest import multipletests

from ._validation import finite_array, validate_inference
from .least_squares import fit_least_squares
from .scoring import SubsampleStatistics


def sample_schedule(p, s, m, seed):
    """Local legacy RandomState preserves historical seeded subset order."""
    rng = np.random.RandomState(seed)
    return np.array([rng.choice(p, s, replace=False) for _ in range(m)], dtype=np.intp)


def analyze_schedule(X, y, schedule, weights=None, repetition_ids=None):
    """Fit ordered subsets once and keep only the statistics used downstream."""
    schedule = np.asarray(schedule)
    if (schedule.ndim != 2 or min(schedule.shape) < 1 or schedule.dtype.kind not in "iu" or
            np.any(schedule < 0) or np.any(schedule >= X.shape[1]) or
            np.any(np.diff(np.sort(schedule, axis=1), axis=1) == 0)):
        raise ValueError("schedule must contain distinct valid integer feature IDs in each row")
    t_values = np.empty(schedule.shape)
    rss, weighted_rss, df = (np.empty(len(schedule)) for _ in range(3))
    repetitions = np.arange(len(schedule)) if repetition_ids is None else np.asarray(repetition_ids)
    for row, indices in enumerate(schedule):
        try:
            fit = fit_least_squares(X[:, indices], y, weights)
        except (ValueError, np.linalg.LinAlgError) as error:
            raise ValueError(f"submodel repetition {repetitions[row]}: {error}") from error
        t_values[row] = fit.t_values[1:]
        rss[row], weighted_rss[row], df[row] = fit.rss, fit.weighted_rss, fit.df_resid
    return SubsampleStatistics(schedule, t_values, rss, repetitions, X.shape[1], df, weighted_rss)


def filter_pvalues(pvalues, semifinalists, alpha=.05, method="bonferroni"):
    """Filter the q semifinalist tests, excluding the intercept.

For compatibility, adjusted tests use statsmodels' inclusive threshold;
``method='none'`` preserves the historical strict p < alpha convention.
    """
    validate_inference(alpha, method)
    pvalues = finite_array(pvalues, "slope p-values")
    if pvalues.shape != (len(semifinalists),) or np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("invalid slope p-values or feature alignment")
    if method == "none":
        adjusted, reject = pvalues.copy(), pvalues < alpha
    else:
        reject, adjusted, _, _ = multipletests(pvalues, alpha=alpha, method=method)
    return [int(semifinalists[i]) for i in np.flatnonzero(reject)], adjusted
