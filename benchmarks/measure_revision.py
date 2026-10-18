"""Repeat the frozen sentinel with legacy, c881315, and current production code.

One warm-up per implementation, rotating execution order, five timed repeats;
tracemalloc and cProfile each run separately. No diagnostic code substitution.
"""

import argparse
import cProfile
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import pstats
import subprocess
import sys
import tempfile
import time
import tracemalloc
from importlib.metadata import version

import numpy as np

from .reference import BASE, load_legacy
from .run_classic import ROOT, comparison_operations

PREVIOUS = 'c8813150ea84c32d564e032f5718ecb36408e2a7'


def previous_package(directory):
    root = Path(directory)
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', PREVIOUS, 'src/subsampwinner'], cwd=ROOT, text=True).splitlines()
    for filename in files:
        if filename.endswith('.py'):
            (root / Path(filename).name).write_bytes(subprocess.check_output(['git', 'show', f'{PREVIOUS}:{filename}'], cwd=ROOT))
    spec = importlib.util.spec_from_file_location('_previous_subsamp', root / '__init__.py', submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_outputs(candidate, reference):
    for got, expected in zip(candidate[1:], reference[1:]):
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
    for got, expected in zip(candidate[0], reference[0]):
        for key in ('semifinalists', 'finalists'):
            np.testing.assert_array_equal(got[key], expected[key])
        for key in ('scores', 'params', 'bse', 'tvalues', 'pvalues', 'p_adjusted'):
            np.testing.assert_allclose(got[key], expected[key], rtol=1e-9, atol=1e-13)


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('use a new measurement output file')
    manifest = json.loads((ROOT / 'tests/reference/classic/manifest.json').read_text())
    case = next(case for case in manifest['cases'] if case['m'] == 5000)
    path = ROOT / 'tests/reference/classic' / case['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case['sha256']
    with np.load(path, allow_pickle=False) as data:
        X, y, schedule = (data[key] for key in ('X', 'y', 'schedule'))
    legacy = load_legacy()
    with tempfile.TemporaryDirectory(prefix='subsamp-previous-') as directory:
        previous = previous_package(directory)
        legacy_op, current_op = comparison_operations(X, y, schedule, case['s'], case['q'], legacy)
        _, previous_op = comparison_operations(X, y, schedule, case['s'], case['q'], legacy,
                                               previous.subsamp, sys.modules['_previous_subsamp._core'].analyze_schedule)
        operations = dict(legacy=legacy_op, previous=previous_op, revised=current_op)
        reference = legacy_op()
        check_outputs(previous_op(), reference)
        check_outputs(current_op(), reference)
        timings = {name: [] for name in operations}
        names = list(operations)
        for iteration in range(5):
            for name in names[iteration % 3:] + names[:iteration % 3]:
                started = time.perf_counter()
                actual = operations[name]()
                timings[name].append(time.perf_counter() - started)
                check_outputs(actual, reference)  # Outside timer.
        measurements = {}
        for name, operation in operations.items():
            tracemalloc.start()
            try:
                operation()
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            profiler = cProfile.Profile()
            profiler.runcall(operation)
            stats = pstats.Stats(profiler)
            top = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:15]
            measurements[name] = dict(seconds=timings[name], median_seconds=float(np.median(timings[name])),
                                      min_seconds=min(timings[name]), max_seconds=max(timings[name]),
                                      python_traced_peak_bytes=peak, profile_total_calls=stats.total_calls,
                                      profile_top_cumulative=[dict(function=f'{Path(key[0]).name}:{key[1]}:{key[2]}',
                                                                   calls=val[1], seconds=val[3]) for key, val in top])
    result = dict(legacy_commit=BASE, previous_commit=PREVIOUS, case=case, measurements=measurements,
                  environment=dict(python=platform.python_version(), platform=platform.platform(),
                                   packages={name: version(name) for name in ('numpy', 'scipy', 'statsmodels', 'pandas', 'patsy')}),
                  source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for directory in ('src/subsampwinner', 'benchmarks') for p in sorted((ROOT / directory).glob('*.py'))},
                  validation='All timed outputs matched legacy; feature IDs exact, statistics within frozen-panel tolerances.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(measurements, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
