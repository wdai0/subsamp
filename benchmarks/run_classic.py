"""Restartable paired benchmark; the large profile requires an explicit chunk.

Run as `PYTHONPATH=src:. python -m benchmarks.run_classic --help`.
Failures are recorded and make the process fail. They are never empty selections.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import tracemalloc

import numpy as np
import scipy
import statsmodels

from .classic import dataset, scenarios, selection_metrics
from .evaluation import evaluate_pair, outcome, validate_method
from .reporting import compatibility_summary, evaluation_summary
from .reference import BASE, load_legacy
from subsampwinner import subsamp
from subsampwinner._core import analyze_schedule, sample_schedule

ROOT = Path(__file__).resolve().parents[1]


def measured(operation):
    # Timing without instrumentation; memory is a separate same-input replay.
    started = time.perf_counter()
    value = operation()
    seconds = time.perf_counter() - started
    tracemalloc.start()
    try:
        operation()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, dict(seconds=seconds, python_traced_peak_bytes=peak)


def comparison_operations(X, y, schedule, s, q, legacy_class, candidate_class=subsamp, analyze=analyze_schedule):
    """Identical stage scope for compatibility checks and repeated measurements."""
    def legacy_run():
        swa = legacy_class(s, len(schedule), q)
        results = [swa._analyze_subsample(X, y, indices) for indices in schedule]
        states = []
        for top in (False, True):
            semi, scores = swa.process_subsample_results(results, q, top)
            final, adjusted, model = swa.select_finalists(X, y, semi)
            states.append(dict(semifinalists=semi, finalists=final, scores=scores, p_adjusted=adjusted,
                               params=model.params, bse=model.bse, tvalues=model.tvalues, pvalues=model.pvalues))
        return states, np.array([r["rss"] for r in results]), np.array([r["t_vals"][ids] for r, ids in zip(results, schedule)])

    def candidate_run():
        statistics = analyze(X, y, schedule)
        swa = candidate_class(s, len(schedule), q)
        states = []
        for top in (False, True):
            state = swa._finish(X, y, None, statistics, top, .05, "bonferroni")
            model = state["final_model"]
            states.append(dict(semifinalists=state["semifinalists"], finalists=state["finalists"],
                               scores=state["feature_weights"], p_adjusted=state["p_adjusted"],
                               params=model.params, bse=model.bse, tvalues=model.tvalues, pvalues=model.pvalues))
        return states, statistics.rss, statistics.t_values

    return legacy_run, candidate_run


def paired(X, y, schedule, s, q, legacy_class):
    legacy_run, candidate_run = comparison_operations(X, y, schedule, s, q, legacy_class)
    reference, baseline_measurement = measured(legacy_run)
    candidate, candidate_measurement = measured(candidate_run)
    for got, expected in zip(candidate[1:], reference[1:]):
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
    output = dict(baseline=baseline_measurement, candidate=candidate_measurement, modes={}, outcomes={})
    for mode, actual, expected in zip(("all", "top_s"), candidate[0], reference[0]):
        for key in ("semifinalists", "finalists"):
            np.testing.assert_array_equal(actual[key], expected[key])
        for key in ("scores", "p_adjusted", "params", "bse", "tvalues", "pvalues"):
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-9, atol=1e-13)
        output["outcomes"][mode] = dict(baseline=outcome(expected), candidate=outcome(actual))
        output["modes"][mode] = dict(selection_metrics(actual["semifinalists"], actual["finalists"]),
                                      semifinalists=actual["semifinalists"].tolist(),
                                      finalists=[int(j) for j in actual["finalists"]])
    return output


def work_items(args):
    if args.profile == "fixed":
        manifest = json.loads((ROOT / "tests/reference/classic/manifest.json").read_text())
        cases = manifest["cases"]
        if args.scenario:
            cases = [case for case in cases if case["id"] == args.scenario]
        if not cases:
            raise ValueError("no matching fixed case ID")
        return cases
    config = json.loads((ROOT / f"benchmarks/{args.profile}-profile.json").read_text())
    if (config["n"], config["p"], config["alpha"], config["method"], config["family"]) != (
            80, 100, .05, "bonferroni", "p candidates" if args.profile == "paper" else "q semifinalists"):
        raise ValueError("profile requests a method outside this classic benchmark contract")
    if config["scenarios"] != len(scenarios()):
        raise ValueError("profile scenario count disagrees with the scenario definitions")
    if args.start is None or args.stop is None or not 0 <= args.start < args.stop <= config["datasets_per_scenario"]:
        raise ValueError("large profile requires explicit --start/--stop within [0,1000]")
    available = scenarios()
    matches = [(i, scenario) for i, scenario in enumerate(available) if scenario["name"] == args.scenario]
    if len(matches) != 1:
        raise ValueError("large profile requires an explicit valid --scenario")
    i, scenario = matches[0]
    return [dict(scenario, id=f"{scenario['name']}-{replicate:04d}",
                 data_seed=config["data_seed_base"] + 1000*i + replicate,
                 schedule_seed=config["schedule_seed_base"] + 1000*i + replicate,
                 s=config["s"], q=config["q"], m=config["m"]) for replicate in range(args.start, args.stop)]


def run(args):
    cases = work_items(args)
    mode = getattr(args, 'mode', 'compatibility')
    configs = None
    if mode == 'evaluation':
        if not getattr(args, 'baseline_config', None) or not getattr(args, 'candidate_config', None):
            raise ValueError('evaluation requires explicit --baseline-config and --candidate-config')
        configs = {label: validate_method(json.loads(Path(getattr(args, label + '_config')).read_text()))
                   for label in ('baseline', 'candidate')}
    elif args.profile == 'paper':
        raise ValueError('paper profile requires evaluation with explicit q/p correction-family configurations')
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for directory in ("src/subsampwinner", "benchmarks") for path in sorted((ROOT / directory).glob("*.py"))}
    signature = dict(profile=args.profile, mode=mode, methods=configs, reference_base=BASE, source_hashes=source_hashes,
                     environment=dict(python=platform.python_version(), numpy=np.__version__,
                                      scipy=scipy.__version__, statsmodels=statsmodels.__version__,
                                      pandas=importlib.metadata.version("pandas"), patsy=importlib.metadata.version("patsy")),
                     profile_sha256=hashlib.sha256((ROOT / (f"benchmarks/{args.profile}-profile.json" if args.profile != "fixed"
                                                           else "tests/reference/classic/manifest.json")).read_bytes()).hexdigest())
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "run.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != signature:
            raise ValueError("output belongs to different code/environment/profile; use a new directory")
    elif any(destination.iterdir()):
        raise ValueError("output directory has unrecognized contents")
    else:
        manifest_path.write_text(json.dumps(signature, indent=2) + "\n")
    legacy_class = load_legacy() if mode == "compatibility" else None
    for case in cases:
        result_path = destination / (case["id"] + ".json")
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            if saved["case"] != case:
                raise ValueError("saved case specification differs")
            continue  # Includes failures: never silently retry/replace a failed dataset.
        try:
            if args.profile == "fixed":
                path = ROOT / "tests/reference/classic" / case["file"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
                    raise ValueError("frozen dataset checksum mismatch")
                with np.load(path, allow_pickle=False) as frozen:
                    X, y, schedule = (frozen[key] for key in ("X", "y", "schedule"))
            else:
                X, y = dataset(case, case["data_seed"])
                schedule = sample_schedule(100, case["s"], case["m"], case["schedule_seed"])
                np.savez_compressed(destination / (case["id"] + ".npz"), X=X, y=y, schedule=schedule)
            result = (paired(X, y, schedule, case['s'], case['q'], legacy_class) if mode == 'compatibility' else
                      evaluate_pair(X, y, schedule, case, configs['baseline'], configs['candidate']))
            record = dict(case=case, error=None, **result)
        except Exception as error:
            record = dict(case=case, error=f"{type(error).__name__}: {error}")
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(result_path)
    records = [json.loads(path.read_text()) for path in sorted(destination.glob("*.json"))
               if path.name not in ("run.json", "summary.json")]
    summary = compatibility_summary(records) if mode == 'compatibility' else evaluation_summary(records, configs)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(completed_cases=summary["completed_cases"], failures=summary["failures"])))
    return bool(summary["failures"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("fixed", "large", "paper"), default="fixed")
    parser.add_argument("--scenario", help="Fixed case ID or large scenario name")
    parser.add_argument("--start", type=int, help="First large-profile replicate, inclusive")
    parser.add_argument("--stop", type=int, help="Last large-profile replicate, exclusive")
    parser.add_argument('--mode', choices=('compatibility', 'evaluation'), default='compatibility')
    parser.add_argument('--baseline-config', help='Explicit method JSON for evaluation')
    parser.add_argument('--candidate-config', help='Explicit method JSON for evaluation')
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args()))
