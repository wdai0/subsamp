"""Legacy adaptive assurance, with explicit and auditable iteration state."""

import math
import numpy as np

from .subsamp import subsamp
from ._validation import (positive_integer, validate_boolean, validate_data, validate_inference,
                          validate_seed, validate_sizes)


class SubsampDoubleAssurance:
    """Adaptive largest-set update and cumulative union; no union-model refit."""

    def __init__(self, m, qnum=None, seed=None, *, scorer=None):
        self.m = positive_integer(m, "m")
        self.qnum = None if qnum is None else positive_integer(qnum, "qnum")
        validate_seed(seed)
        if scorer is not None and not callable(scorer):
            raise ValueError("scorer must be callable or None")
        self.seed, self.scorer = seed, scorer
        self.history_ = []
        self.best_stability_ = self.termination_reason_ = None

    @staticmethod
    def calculate_individual_stability(individual_set, union_set):
        return len(individual_set) / len(union_set) if union_set else 0.

    def double_assurance(self, X, y, s0, T=.8, I_max=10, init_range=.3, r=.75,
                         weights=None, alpha=.05, method="bonferroni", return_objects=False):
        """Run center/lower/upper fits and retain every selected feature so far.

The intersection/union score controls stopping; the largest individual set
chooses the next center (ties: center, lower, upper). An empty triplet has score
zero and stops without erasing earlier selections. Invalid ranges raise rather
than clip. ``final_range`` is the last evaluated [lower, center, upper].
``history_`` contains independent snapshots after a successful call. As before,
return_objects=True returns the three-key result dictionary; otherwise print it.
        """
        validate_boolean(return_objects, "return_objects")
        X, y, weights = validate_data(X, y, weights)
        positive_integer(I_max, "I_max")
        validate_sizes(s0, self.m, s0 if self.qnum is None else self.qnum, *X.shape)
        validate_inference(alpha, method)
        for value, name, valid in ((T, "T", lambda x: 0 <= x <= 1),
                                   (init_range, "init_range", lambda x: 0 <= x < 1),
                                   (r, "r", lambda x: 0 < x <= 1)):
            if not np.isscalar(value) or not np.isfinite(value) or not valid(value):
                raise ValueError(f"{name} is outside its supported range")

        def run_swa(s):
            swa = subsamp(s=s, m=self.m, qnum=self.qnum, seed=self.seed, scorer=self.scorer)
            swa.fit(X, y, weights=weights, alpha=alpha, method=method)
            return set(swa.finalists)

        best_s0, best_stability, cumulative, history = s0, 0., set(), []
        reason = "iteration_limit"
        for iteration in range(I_max):
            radius = init_range * r**iteration
            lower, upper = math.floor((1 - radius) * s0), math.ceil((1 + radius) * s0)
            for size in (lower, s0, upper):
                validate_sizes(size, self.m, size if self.qnum is None else self.qnum, *X.shape)
            evaluated_range = [lower, s0, upper]
            center_set = run_swa(s0)
            selections = [center_set, run_swa(lower), run_swa(upper)] if lower != upper else [center_set] * 3
            union = set.union(*selections)
            intersection = set.intersection(*selections)
            cumulative.update(union)
            score = len(intersection) / len(union) if union else 0.
            if score > best_stability:
                best_s0, best_stability = s0, score
            history.append(dict(iteration=iteration, evaluated_range=evaluated_range.copy(),
                                selections=[selected.copy() for selected in selections],
                                union=union.copy(), intersection=intersection.copy(),
                                cumulative_union=cumulative.copy(), stability=score))
            if not union:
                reason = "empty_selection"
                break
            if lower == upper:
                reason = "collapsed_range"
                break
            if score >= T:
                reason = "stability_threshold"
                break
            contributions = [self.calculate_individual_stability(selected, union) for selected in selections]
            s0 = [s0, lower, upper][int(np.argmax(contributions))]
        result = dict(best_s0=best_s0, final_range=evaluated_range.copy(), stabilized_union_set=cumulative.copy())
        self.history_, self.best_stability_, self.termination_reason_ = history, best_stability, reason
        if return_objects:
            return result
        print(f"Best subsample size: {best_s0}; best stability score: {best_stability:.4f}")
        print(f"Last evaluated range: {evaluated_range}; termination: {reason}")
        print(f"Cumulative selected features: {sorted(cumulative)}")
        return None
