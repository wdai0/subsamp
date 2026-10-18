"""Scientific reporting contracts: paired datasets, family size, and failures."""

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from benchmarks.classic import dataset, scenarios, selection_metrics
from benchmarks.evaluation import evaluate_method, evaluate_pair, validate_method
from benchmarks.reporting import compatibility_summary, evaluation_summary, regenerate
from benchmarks.run_classic import work_items
from subsampwinner._core import sample_schedule

ROOT = Path(__file__).resolve().parents[1]


def config(name='default-q'):
    return json.loads((ROOT / f'benchmarks/methods/{name}.json').read_text())


def test_scaled_scores_are_valid_method_differences():
    case = dict(scenarios()[0], s=10, q=10, m=120)
    X, y = dataset(case, 91000)
    schedule = sample_schedule(100, 10, 120, 92000)
    result = evaluate_pair(X, y, schedule, case, config(), config('scaled-q'))
    assert result['valid_pair']
    assert result['valid_differences'] == dict(scores=True, semifinalists=False, finalists=False)
    a, b = (result['methods'][label]['outcome'] for label in ('baseline', 'candidate'))
    assert_allclose(np.array(a['scores']) * 2, b['scores'])


def test_q_and_p_multiplicity_families_are_explicit():
    case = dict(scenarios()[0], s=10, q=10, m=120)
    X, y = dataset(case, 91000)
    schedule = sample_schedule(100, 10, 120, 92000)
    q = evaluate_method(X, y, schedule, 10, 10, config())
    p = evaluate_method(X, y, schedule, 10, 10, config('default-p'))
    assert q['semifinalists'] == p['semifinalists']
    assert_allclose(q['p_adjusted'], np.minimum(np.array(q['pvalues'][1:]) * 10, 1))
    assert_allclose(p['p_adjusted'], np.minimum(np.array(p['pvalues'][1:]) * 100, 1))


def test_method_failure_keeps_other_outcome_and_blocks_paired_estimates(monkeypatch):
    from benchmarks import evaluation
    case = dict(scenarios()[0], s=10, q=10, m=120)
    original = evaluation.evaluate_method
    def fail_candidate(*args):
        if args[-1]['scorer']['name'] == 'power':
            raise ValueError('injected failure')
        return original(*args)
    monkeypatch.setattr(evaluation, 'evaluate_method', fail_candidate)
    X, y = dataset(case, 91000)
    result = evaluate_pair(X, y, sample_schedule(100, 10, 120, 92000), case, config(), config('scaled-q'))
    assert result['methods']['baseline']['outcome'] is not None
    assert result['methods']['candidate']['error'] is not None
    record = dict(case=case, error=None, **result)
    summary = evaluation_summary([record], dict(baseline=config(), candidate=config('scaled-q')))
    group = summary['groups'][0]
    assert summary['failures'] == 1 and group['valid_pairs'] == 0
    assert group['paired_candidate_minus_baseline'] is None
    assert group['method_summaries']['candidate']['selection_estimates'] is None


def test_paired_fdp_mcse_uses_independent_dataset_differences():
    # Two datasets, FDP deltas +1/2 and -1/2: mean 0, MCSE 1/2.
    records = []
    for baseline, candidate in (([0], [0, 10]), ([0, 10], [0])):
        records.append(dict(case=dict(scenarios()[0], s=10, q=10, m=5000), error=None,
                            valid_pair=True, valid_differences=dict(scores=True, semifinalists=False, finalists=True),
                            methods={label: dict(error=None, outcome={'metrics': selection_metrics([0], final)})
                                     for label, final in (('baseline', baseline), ('candidate', candidate))}))
    summary = evaluation_summary(records, dict(baseline=config(), candidate=config()))['groups'][0]
    assert summary['datasets'] == 2  # Never the 10,000 submodels.
    assert summary['paired_candidate_minus_baseline']['fdr'] == dict(mean=0., mcse=.5)
    assert summary['method_summaries']['baseline']['selection_estimates']['fdr'] == .25


def test_historical_summary_separates_budgets_without_rewriting_records(tmp_path):
    # The frozen panel already records these historical outcomes. Reconstruct
    # only the three input records needed to test reporting, not a whole audit.
    source = tmp_path / 'records'
    source.mkdir()
    manifest = json.loads((ROOT / 'tests/reference/classic/manifest.json').read_text())
    for case in manifest['cases']:
        if case['name'] == 'paper-independent':
            record = dict(case=case, error=None, modes=case['metrics'])
            (source / (case['id'] + '.json')).write_text(json.dumps(record))
    before = {path.name: path.read_bytes() for path in source.glob('*.json')}
    output = tmp_path / 'corrected.json'
    regenerate(source, output)
    summary = json.loads(output.read_text())
    groups = [g for g in summary['groups'] if g['scenario']['name'] == 'paper-independent'
              and g['configuration']['retention'] == 'top_s']
    assert len(groups) == 2
    by_m = {g['configuration']['m']: g['summary'] for g in groups}
    assert_allclose(by_m[120]['selection_estimates']['average_signal_recovery'], .4)
    assert by_m[5000]['selection_estimates']['average_signal_recovery'] == .7
    assert by_m[120]['selection_estimates']['fdr'] == 0
    assert by_m[5000]['selection_estimates']['fdr'] == .125
    assert by_m[5000]['monte_carlo']['fdr']['mcse'] is None
    assert before == {path.name: path.read_bytes() for path in source.glob('*.json')}
    with pytest.raises(ValueError, match='new file'):
        regenerate(source, output)


def test_paper_profile_is_separate_from_frozen_panel():
    from types import SimpleNamespace
    args = SimpleNamespace(profile='paper', scenario='paper-independent', start=0, stop=1)
    case = work_items(args)[0]
    assert (case['s'], case['q'], case['m']) == (30, 30, 5000)
    frozen = json.loads((ROOT / 'tests/reference/classic/manifest.json').read_text())
    assert all((c['s'], c['q']) == (10, 10) for c in frozen['cases'])


@pytest.mark.parametrize('change', [{'family': 'unknown'}, {'retention': 'False'},
                                    {'scorer': {'name': 'power', 't_power': 0, 'rss_power': .5, 'scale': 2}}])
def test_invalid_method_specifications_fail_early(change):
    with pytest.raises(ValueError):
        validate_method(dict(config(), **change))
