"""Configuration-aware summaries and dataset-level Monte Carlo uncertainty."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .classic import summarize
from .evaluation import configured_method

METRICS = dict(per_signal_power='signal_selected', average_signal_recovery='signal_recovery',
               semifinalist_recall='semifinalist_recall', mean_false_positives='false_positives',
               fdr='false_discovery_proportion', fwer='any_false_positive',
               mean_selected_size='selected_size', exact_support_frequency='exact_support')


def mean_mcse(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=np.mean(values, axis=0).tolist(),
                mcse=(np.std(values, axis=0, ddof=1) / np.sqrt(len(values))).tolist() if len(values) > 1 else None)


def uncertainty(metrics):
    return {label: mean_mcse([row[key] for row in metrics]) for label, key in METRICS.items()}


def scenario_config(case):
    return {key: case[key] for key in ('name', 'covariance', 'noise_sd', 'snr') if key in case}


def compatibility_summary(records):
    groups = defaultdict(list)
    for record in records:
        case = record['case']
        for mode in ('all', 'top_s'):
            config = configured_method(case, dict(retention=mode, alpha=.05, correction='bonferroni',
                                                  family='q', scorer={'name': 'default'}))
            key = json.dumps(dict(scenario=scenario_config(case), configuration=config), sort_keys=True)
            groups[key].append(record)
    result = []
    for key, rows in sorted(groups.items()):
        metadata = json.loads(key)
        mode = metadata['configuration']['retention']
        metrics = [dict(error=row['error']) if row['error'] else row['modes'][mode] for row in rows]
        summary = summarize(metrics)
        summary['monte_carlo'] = uncertainty(metrics) if not summary['failures'] and metrics else None
        result.append(dict(metadata, summary=summary))
    return dict(completed_cases=len(records), failures=sum(r['error'] is not None for r in records), groups=result)


def evaluation_summary(records, configs):
    groups = defaultdict(list)
    for record in records:
        pair_config = {label: configured_method(record['case'], cfg) for label, cfg in configs.items()}
        key = json.dumps(dict(scenario=scenario_config(record['case']), methods=pair_config), sort_keys=True)
        groups[key].append(record)
    output = []
    for key, rows in sorted(groups.items()):
        group = dict(json.loads(key), datasets=len(rows), method_summaries={})
        for label in configs:
            metrics = [(dict(error=row['error']) if row['error'] else
                        dict(error=row['methods'][label]['error']) if row['methods'][label]['error'] else
                        row['methods'][label]['outcome']['metrics']) for row in rows]
            summary = summarize(metrics)
            summary['monte_carlo'] = uncertainty(metrics) if not summary['failures'] else None
            group['method_summaries'][label] = summary
        valid = [row for row in rows if row['error'] is None and row['valid_pair']]
        group['valid_pairs'] = len(valid)
        group['failed_pairs'] = len(rows) - len(valid)
        group['valid_difference_counts'] = {key: sum(row['valid_differences'][key] for row in valid)
                                            for key in ('scores', 'semifinalists', 'finalists')}
        # Withhold estimates on incomplete pairs; never silently drop failures.
        group['paired_candidate_minus_baseline'] = None
        if len(valid) == len(rows):
            group['paired_candidate_minus_baseline'] = {
                label: mean_mcse([np.asarray(row['methods']['candidate']['outcome']['metrics'][metric]) -
                                  np.asarray(row['methods']['baseline']['outcome']['metrics'][metric])
                                  for row in valid]) for label, metric in METRICS.items()}
        output.append(group)
    return dict(completed_cases=len(records),
                failures=sum(row['error'] is not None or not row['valid_pair'] for row in records), groups=output)


def regenerate(directory, output):
    """Write a new summary from historical records without altering those records."""
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise ValueError('regenerated output must be a new file')
    paths = [path for path in sorted(directory.glob('*.json')) if path.name not in ('run.json', 'summary.json')]
    records = [json.loads(path.read_text()) for path in paths]
    summary = compatibility_summary(records)
    summary['source_records_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('output')
    args = parser.parse_args()
    regenerate(args.directory, args.output)
