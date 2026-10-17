"""Subsampling Winner Algorithm with replaceable scoring and explicit stages."""

import logging
import numpy as np

from ._core import analyze_schedule, filter_pvalues, sample_schedule
from ._validation import (feature_indices, finite_array, positive_integer, validate_data,
                          validate_inference, validate_seed, validate_sizes)
from .least_squares import fit_least_squares
from .scoring import (SubsampleStatistics, default_scorer, rank_features,
                      retain_submodels, score_features)


class subsamp:
    """SWA on original feature IDs; ``fit`` preserves its historical None return.

``s`` is predictors per submodel, ``m`` the repetitions and ``qnum`` the
semifinalist count (defaults to s). ``scorer`` receives retained compact
SubsampleStatistics and must return p finite scores; larger scores rank first.
WLS weights are positive inverse-variance weights with conventional t inference.
    """

    def __init__(self, s, m, qnum=None, seed=None, *, scorer=None):
        self.s = positive_integer(s, "s")
        self.m = positive_integer(m, "m")
        self.qnum = self.s if qnum is None else positive_integer(qnum, "qnum")
        validate_seed(seed)
        if scorer is not None and not callable(scorer):
            raise ValueError("scorer must be callable or None")
        self.seed, self.scorer = seed, scorer
        self.X = self.y = None
        self.feature_weights = self.semifinalists = self.finalists = None
        self.p_adjusted = self.final_model = self.subsample_statistics_ = None

    def __repr__(self):
        return f"SWA(s={self.s}, m={self.m}, qnum={self.qnum}, seed={self.seed})"

    def _validate_fit(self, X, y, weights, alpha, method):
        X, y, weights = validate_data(X, y, weights)
        validate_sizes(self.s, self.m, self.qnum, *X.shape)
        validate_seed(self.seed)
        validate_inference(alpha, method)
        if self.scorer is not None and not callable(self.scorer):
            raise ValueError("scorer must be callable or None")
        return X.copy(), y.copy(), None if weights is None else weights.copy()

    def _finish(self, X, y, weights, statistics, use_top_s, alpha, method):
        retained = retain_submodels(statistics, self.s if use_top_s else None)
        scores = score_features(retained, self.scorer)
        semi = rank_features(scores, self.qnum)
        fit = fit_least_squares(X[:, semi], y, weights, [f"v{i + 1}" for i in semi])
        finalists, adjusted = filter_pvalues(fit.p_values[1:], semi, alpha, method)
        return dict(X=X, y=y, feature_weights=scores, semifinalists=semi,
                    finalists=finalists, p_adjusted=adjusted, final_model=fit.model,
                    subsample_statistics_=retained)

    def fit(self, X, y, parallel=False, weights=None, use_top_s=False, alpha=.05, method="bonferroni"):
        """Fit and publish state only after all stages succeed.

All models contribute by default; top-s retention uses raw RSS even in WLS.
``final_model`` fits semifinalists; filtering does not refit on finalists.
In MPI, call on every rank with the same data/options. Root owns the schedule
and calls the scorer once; all ranks receive complete fitted state or an error.
        """
        if parallel:
            from ._parallel import analyze_parallel, collective_call, communicator, root_call
            comm = communicator()
            X, y, weights = collective_call(comm, "validation", lambda: self._validate_fit(X, y, weights, alpha, method))
            statistics = analyze_parallel(comm, X, y, self.s, self.m, self.seed, weights)
            state = root_call(comm, "scoring and final fitting", lambda: self._finish(
                X, y, weights, statistics, use_top_s, alpha, method))
        else:
            X, y, weights = self._validate_fit(X, y, weights, alpha, method)
            schedule = sample_schedule(X.shape[1], self.s, self.m, self.seed)
            statistics = analyze_schedule(X, y, schedule, weights)
            state = self._finish(X, y, weights, statistics, use_top_s, alpha, method)
        self.__dict__.update(state)
        logging.info("SWA selects the following feature indices: %s.", self.finalists)
        return None

    # Compatibility adapters: explicit requests for dense legacy result lists
    # may use O(mp) space; fit() never constructs them.
    def _subsample_s_features(self, p, s):
        # Standalone helper calls advance a private stream. Full fits reset their
        # separate schedule stream to self.seed for historical repeatability.
        if not hasattr(self, "_helper_rng") or self._helper_seed != self.seed:
            validate_seed(self.seed)
            self._helper_rng = np.random.RandomState(self.seed)
            self._helper_seed = self.seed
        return self._helper_rng.choice(p, s, replace=False)

    def _fit_base_model(self, X, y, weights=None):
        return fit_least_squares(X, y, weights).model

    def _compute_model_statistics(self, model, p, subsample_indices):
        indices = feature_indices(subsample_indices, p)
        t_values = np.zeros(p)
        t_values[indices] = model.tvalues[1:]
        return t_values, float(np.sum(model.resid ** 2)), model.resid

    def _analyze_subsample(self, X, y, subsample_indices, weights=None):
        X, y, weights = validate_data(X, y, weights)
        indices = feature_indices(subsample_indices, X.shape[1])
        statistics = analyze_schedule(X, y, indices[None, :], weights)
        result = self._dense_results(statistics)[0]
        # A standalone call has no schedule-level repetition identity.
        result.pop("repetition_id")
        return result

    @staticmethod
    def _dense_results(statistics):
        results = []
        for i, indices in enumerate(statistics.feature_indices):
            dense = np.zeros(statistics.n_features)
            dense[indices] = statistics.t_values[i]
            results.append(dict(indices=indices.copy(), t_vals=dense, rss=float(statistics.rss[i]),
                                repetition_id=int(statistics.repetition_ids[i]),
                                df_resid=None if statistics.df_resid is None else float(statistics.df_resid[i]),
                                weighted_rss=None if statistics.weighted_rss is None else float(statistics.weighted_rss[i])))
        return results

    @staticmethod
    def _compact_results(results):
        if isinstance(results, SubsampleStatistics):
            return results
        if not results:
            raise ValueError("submodel results must not be empty")
        p = len(results[0]["t_vals"])
        indices = [feature_indices(row["indices"], p) for row in results]
        dense_t = [finite_array(row["t_vals"], "legacy t_vals") for row in results]
        if any(t.shape != (p,) for t in dense_t):
            raise ValueError("legacy t_vals must have one entry per original feature")
        for t, idx in zip(dense_t, indices):
            if np.count_nonzero(t) != np.count_nonzero(t[idx]):
                raise ValueError("legacy t_vals must be zero outside their listed feature indices")
        optional = {key: None if any(row.get(key) is None for row in results)
                    else np.array([row[key] for row in results]) for key in ("df_resid", "weighted_rss")}
        return SubsampleStatistics(np.array(indices),
                                   np.array([t[idx] for t, idx in zip(dense_t, indices)]),
                                   np.array([row["rss"] for row in results]),
                                   np.array([row.get("repetition_id", i) for i, row in enumerate(results)]), p, **optional)

    def subsample_anlysis(self, X, y, parallel=False, weights=None):
        """Legacy spelling and dense-list return retained; use fit for compact storage."""
        if parallel:
            from ._parallel import analyze_parallel, collective_call, communicator
            comm = communicator()
            X, y, weights = collective_call(comm, "validation", lambda: self._validate_fit(X, y, weights, .05, "bonferroni"))
            statistics = analyze_parallel(comm, X, y, self.s, self.m, self.seed, weights)
        else:
            X, y, weights = self._validate_fit(X, y, weights, .05, "bonferroni")
            statistics = analyze_schedule(X, y, sample_schedule(X.shape[1], self.s, self.m, self.seed), weights)
        return self._dense_results(statistics)

    def _compute_feature_weights(self, results, option_s=None):
        """Legacy default-score helper: average abs(t)/sqrt(RSS) over nonzero t."""
        return default_scorer(retain_submodels(self._compact_results(results), option_s))

    def _select_semifinalist(self, results, qnum, use_top_s=False):
        retained = retain_submodels(self._compact_results(results), self.s if use_top_s else None)
        scores = score_features(retained, self.scorer)
        return rank_features(scores, qnum), scores

    def process_subsample_results(self, results, qnum, use_top_s=False):
        """Compatibility stage call; publish its scores and semifinalists on success."""
        semi, scores = self._select_semifinalist(results, qnum, use_top_s)
        self.semifinalists, self.feature_weights = semi, scores
        return semi, scores

    def _fit_final_model(self, X, y, semifinalists, weights=None):
        X, y, weights = validate_data(X, y, weights)
        semi = feature_indices(semifinalists, X.shape[1])
        return fit_least_squares(X[:, semi], y, weights, [f"v{i + 1}" for i in semi]).model

    def _select_finalists(self, final_model, semifinalists, alpha=.05, method="bonferroni"):
        return filter_pvalues(final_model.pvalues[1:], semifinalists, alpha, method)

    def select_finalists(self, X, y, semifinalists, weights=None, alpha=.05, method="bonferroni"):
        """Fit the joint semifinalist model and filter its slope tests; no finalist refit."""
        validate_inference(alpha, method)
        model = self._fit_final_model(X, y, semifinalists, weights)
        finalists, adjusted = self._select_finalists(model, semifinalists, alpha, method)
        self.final_model, self.finalists, self.p_adjusted = model, finalists, adjusted
        return finalists, adjusted, model
