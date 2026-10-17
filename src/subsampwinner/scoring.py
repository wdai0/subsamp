"""Compact submodel statistics, retention and replaceable feature scoring."""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ._validation import finite_array, positive_integer


def _readonly(array):
    array = np.array(array, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class SubsampleStatistics:
    """Scorer input: retained rows, with original feature and repetition IDs.

``feature_indices`` and signed ``t_values`` have shape (retained_models, s).
``rss``, ``repetition_ids``, and optional scale/df arrays have one entry per row.
``n_features`` is the ORIGINAL number of predictors, including unobserved ones.
Arrays are owned read-only copies. RSS is raw even for WLS.
    """

    feature_indices: np.ndarray
    t_values: np.ndarray
    rss: np.ndarray
    repetition_ids: np.ndarray
    n_features: int
    df_resid: Optional[np.ndarray] = None
    weighted_rss: Optional[np.ndarray] = None

    def __post_init__(self):
        p = positive_integer(self.n_features, "n_features")
        ids = np.asarray(self.feature_indices)
        t = finite_array(self.t_values, "t_values")
        rss = finite_array(self.rss, "rss")
        repetitions = np.asarray(self.repetition_ids)
        if ids.ndim != 2 or min(ids.shape) < 1 or ids.dtype.kind not in "iu" or t.shape != ids.shape:
            raise ValueError("feature_indices and t_values must be aligned nonempty matrices")
        if np.any(ids < 0) or np.any(ids >= p) or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
            raise ValueError("each submodel must contain unique valid original feature IDs")
        if rss.shape != (len(ids),) or np.any(rss <= 0):
            raise ValueError("rss must contain one positive finite value per submodel")
        if (repetitions.shape != rss.shape or repetitions.dtype.kind not in "iu" or
                np.any(repetitions < 0) or len(np.unique(repetitions)) != len(repetitions)):
            raise ValueError("repetition_ids must be unique nonnegative integers aligned with rows")
        for name, value in (("feature_indices", ids.astype(np.intp)), ("t_values", t),
                            ("rss", rss), ("repetition_ids", repetitions.astype(np.intp))):
            object.__setattr__(self, name, _readonly(value))
        for name in ("df_resid", "weighted_rss"):
            value = getattr(self, name)
            if value is not None:
                value = finite_array(value, name)
                if value.shape != rss.shape or np.any(value <= 0):
                    raise ValueError(f"{name} must contain one positive value per submodel")
                object.__setattr__(self, name, _readonly(value))

    def __reduce__(self):
        # Re-run ownership/validation after MPI pickle transport.
        return (type(self), (self.feature_indices, self.t_values, self.rss,
                             self.repetition_ids, self.n_features, self.df_resid, self.weighted_rss))


Scorer = Callable[[SubsampleStatistics], np.ndarray]


def retain_submodels(statistics: SubsampleStatistics, count: Optional[int] = None) -> SubsampleStatistics:
    """Retain all rows or the smallest raw RSS values (legacy NumPy quicksort ties)."""
    if count is None:
        return statistics
    positive_integer(count, "retention count")
    if count >= len(statistics.rss):
        return statistics
    rows = np.argsort(statistics.rss, kind="quicksort")[:count]
    return SubsampleStatistics(statistics.feature_indices[rows], statistics.t_values[rows],
                              statistics.rss[rows], statistics.repetition_ids[rows], statistics.n_features,
                              None if statistics.df_resid is None else statistics.df_resid[rows],
                              None if statistics.weighted_rss is None else statistics.weighted_rss[rows])


def default_scorer(statistics: SubsampleStatistics) -> np.ndarray:
    """Average abs(t)/sqrt(raw RSS) over NONZERO t values for each original feature.

Features with no nonzero contributions receive zero. Storage is O(ms + p),
without padding each submodel to the full candidate dimension.
    """
    ids = statistics.feature_indices.ravel()
    contributions = np.abs(statistics.t_values) / np.sqrt(statistics.rss[:, None])
    totals = np.bincount(ids, weights=contributions.ravel(), minlength=statistics.n_features)
    counts = np.bincount(ids, weights=(statistics.t_values != 0).ravel(), minlength=statistics.n_features)
    return np.divide(totals, counts, out=np.zeros_like(totals), where=counts != 0)


def score_features(statistics: SubsampleStatistics, scorer: Optional[Scorer] = None) -> np.ndarray:
    """Validate and own callback output without altering its scores."""
    scores = finite_array((default_scorer if scorer is None else scorer)(statistics), "scorer output")
    if scores.shape != (statistics.n_features,):
        raise ValueError("scorer must return one finite score per original feature")
    return scores.copy()


def rank_features(scores: np.ndarray, q: int) -> np.ndarray:
    """Preserve legacy descending NumPy quicksort ordering, including exact ties."""
    positive_integer(q, "qnum")
    if q > len(scores):
        raise ValueError("qnum must not exceed the number of scores")
    return np.argsort(scores, kind="quicksort")[-q:][::-1]
