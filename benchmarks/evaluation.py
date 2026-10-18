"""Paired method evaluation on shared datasets, separate from legacy equality."""

import numpy as np

from subsampwinner._core import analyze_schedule, filter_pvalues
from subsampwinner._validation import validate_inference
from subsampwinner.least_squares import fit_least_squares
from subsampwinner.scoring import default_scorer, rank_features, retain_submodels, score_features
from .classic import selection_metrics


def validate_method(config):
    """Explicit JSON configuration; s/q/m and seeds come from the dataset profile."""
    if set(config) != {'retention', 'alpha', 'correction', 'family', 'scorer'}:
        raise ValueError('method requires retention, alpha, correction, family and scorer')
    if config['retention'] not in ('all', 'top_s') or config['family'] not in ('q', 'p'):
        raise ValueError('retention must be all/top_s and family must be q/p')
    validate_inference(config['alpha'], config['correction'])
    scorer = config['scorer']
    if scorer == {'name': 'default'}:
        return config
    if set(scorer) != {'name', 't_power', 'rss_power', 'scale'} or scorer['name'] != 'power':
        raise ValueError('scorer must name default or explicitly parameterized power')
    for key in ('t_power', 'rss_power', 'scale'):
        value = scorer[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f'scorer {key} must be a finite real number')
    if scorer['t_power'] <= 0 or scorer['rss_power'] < 0 or scorer['scale'] <= 0:
        raise ValueError('power scorer requires t_power>0, rss_power>=0 and scale>0')
    return config


def method_scorer(config):
    if config == {'name': 'default'}:
        return default_scorer
    def power(statistics):
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            values = np.abs(statistics.t_values)**config['t_power'] / statistics.rss[:, None]**config['rss_power']
            total = np.bincount(statistics.feature_indices.ravel(), weights=values.ravel(),
                                minlength=statistics.n_features)
            count = np.bincount(statistics.feature_indices.ravel(), weights=(statistics.t_values != 0).ravel(),
                                minlength=statistics.n_features)
            return config['scale'] * np.divide(total, count, out=np.zeros_like(total), where=count != 0)
    return power


def configured_method(case, config):
    return dict(n=80, p=100, s=case['s'], q=case['q'], m=case['m'], **config)


def outcome(state):
    result = {key: np.asarray(value).tolist() for key, value in state.items()}
    result['metrics'] = selection_metrics(state['semifinalists'], state['finalists'])
    return result


def evaluate_method(X, y, schedule, s, q, config):
    statistics = retain_submodels(analyze_schedule(X, y, schedule), s if config['retention'] == 'top_s' else None)
    scores = score_features(statistics, method_scorer(config['scorer']))
    semi = rank_features(scores, q)
    fit = fit_least_squares(X[:, semi], y)
    pvalues = fit.p_values[1:]
    # Explicit benchmark policy: p-family includes untested predictors with
    # p=1. This preserves Bonferroni/BH family size without fitting p predictors.
    if config['family'] == 'p':
        pvalues = np.r_[pvalues, np.ones(X.shape[1] - q)]
    selected, adjusted = filter_pvalues(pvalues, np.arange(len(pvalues)), config['alpha'], config['correction'])
    final = [int(semi[i]) for i in selected if i < q]
    return outcome(dict(semifinalists=semi, finalists=final, scores=scores, p_adjusted=adjusted[:q],
                        params=fit.coefficients, bse=fit.standard_errors, tvalues=fit.t_values, pvalues=fit.p_values))


def evaluate_pair(X, y, schedule, case, baseline, candidate):
    methods = {}
    for label, config in (('baseline', baseline), ('candidate', candidate)):
        try:
            result = evaluate_method(X, y, schedule, case['s'], case['q'], config)
            methods[label] = dict(configuration=configured_method(case, config), error=None, outcome=result)
        except Exception as error:
            # Preserve the other method's outcome even if one method fails.
            methods[label] = dict(configuration=configured_method(case, config),
                                  error=f'{type(error).__name__}: {error}', outcome=None)
    valid = all(result['error'] is None for result in methods.values())
    differences = None
    if valid:
        first, second = (methods[label]['outcome'] for label in ('baseline', 'candidate'))
        differences = {key: first[key] != second[key] for key in ('scores', 'semifinalists', 'finalists')}
    return dict(methods=methods, valid_pair=valid, valid_differences=differences)
