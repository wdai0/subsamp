"""Capture a small valid numerical baseline and report defects separately.

Run from the repository root with PYTHONPATH=src.
Output must be a new directory; this script never overwrites approved fixtures.
The source guard prevents silently recapturing changed implementation behavior.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace
import warnings

import mpi4py
import numpy as np

mpi4py.rc.initialize = False

from subsampwinner import SubsampDoubleAssurance, subsamp
from subsampwinner.GenerateData import generate_heteroskedastic_data


BASE = "e0366d3bcf917b875b982214a2ef342f8d463e42"
ROOT = Path(__file__).resolve().parents[1]


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def fixed_schedule_reference():
    """Save data and feature subsets explicitly, independently of future RNGs."""
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(30, 6))
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(size=30)
    weights = np.linspace(0.5, 2.0, len(y))
    schedule = np.array([
        [0, 1, 2], [0, 2, 3], [1, 3, 4], [2, 4, 5],
        [0, 1, 5], [1, 2, 4], [0, 3, 5], [1, 4, 5],
        [0, 2, 4], [1, 2, 3], [0, 3, 4], [2, 3, 5],
    ])
    cases = {}
    for label, sample_weights in [("ols", None), ("wls", weights)]:
        swa = subsamp(s=3, m=len(schedule), qnum=3, seed=7)
        results = [swa._analyze_subsample(X, y, row, sample_weights)
                   for row in schedule]
        cases[label] = {"submodels": results, "selections": {}}
        for use_top_s in (False, True):
            semi, scores = swa.process_subsample_results(results, 3, use_top_s)
            final, adjusted, model = swa.select_finalists(
                X, y, semi, weights=sample_weights)
            cases[label]["selections"]["top_s" if use_top_s else "all"] = {
                "scores": scores, "semifinalists": semi, "finalists": final,
                "p_adjusted": adjusted, "params": model.params,
                "pvalues": model.pvalues, "df_resid": model.df_resid,
            }
    return {"X": X, "y": y, "weights": weights,
            "schedule": schedule, "s": 3, "qnum": 3, "cases": cases}


def observe(call):
    """Record an observation, never assert that a defect is desired behavior."""
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()) as printed:
        warnings.simplefilter("always")
        try:
            result = {"value": call()}
        except Exception as error:
            result = {"exception": type(error).__name__, "message": str(error)}
    result["warnings"] = [str(item.message) for item in caught]
    if printed.getvalue():
        result["stdout"] = printed.getvalue()
    return result


def legacy_probes():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(60, 12))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(size=60)

    def fit_summary(X=X, y=y, **kwargs):
        swa = subsamp(**kwargs)
        returned = swa.fit(X, y)
        return {"returned": returned, "semifinalists": swa.semifinalists,
                "finalists": swa.finalists,
                "final_model_feature_count": len(swa.final_model.params) - 1,
                "df_resid": swa.final_model.df_resid}

    def covariance():
        _, _, cov, _ = generate_heteroskedastic_data(
            30, 5, lambda v: np.ones_like(v) * 2, seed=7,
            type="unstructured")
        return {"minimum_eigenvalue": np.linalg.eigvalsh(cov).min()}

    def rng_side_effect():
        np.random.seed(1234)
        before = np.random.get_state()
        subsamp(s=4, m=5, qnum=3, seed=7).fit(X, y)
        after = np.random.get_state()
        return {"global_state_changed": not all(
            np.array_equal(a, b) for a, b in zip(before, after))}

    calls = {
        "tiny_valid_example": lambda: fit_summary(s=4, m=80, qnum=4, seed=7),
        "constant_predictor": lambda: fit_summary(
            X=np.column_stack([np.ones(60), X[:, :3]]),
            s=4, m=5, qnum=3, seed=7),
        "zero_qnum": lambda: fit_summary(s=4, m=5, qnum=0, seed=7),
        "empty_assurance_union": lambda: SubsampDoubleAssurance(
            m=12, qnum=2, seed=7).double_assurance(
                X, y, s0=3, I_max=1, alpha=1e-300, return_objects=True),
        "oversized_assurance_candidate": lambda: SubsampDoubleAssurance(
            m=5, seed=7).double_assurance(X, y, s0=12, I_max=1, return_objects=True),
        "misspelled_method": lambda: subsamp(s=2, m=2)._select_finalists(
            SimpleNamespace(pvalues=np.array([.001, .03, .2])),
            [0, 1], method="bonferoni"),
        "unstructured_covariance": covariance,
        "global_rng": rng_side_effect,
        "zero_iterations": lambda: SubsampDoubleAssurance(
            m=5, seed=7).double_assurance(X, y, s0=3, I_max=0, return_objects=True),
        "singular_predictors": lambda: fit_summary(
            X=np.column_stack([X[:, 0], X[:, 0], X[:, 1]]),
            s=3, m=5, qnum=3, seed=7),
        "zero_residual_df": lambda: fit_summary(
            X=X[:5, :4], y=y[:5], s=4, m=5, qnum=4, seed=7),
    }
    return {name: observe(call) for name, call in calls.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output directory")
    args = parser.parse_args()
    subprocess.run(["git", "diff", "--exit-code", BASE, "--", "src"],
                   cwd=ROOT, check=True, capture_output=True)
    environment = {"python": platform.python_version(),
                   "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ["numpy", "scipy", "statsmodels", "mpi4py", "pytest"]}}
    provenance = {"base_commit": BASE, "environment": environment,
                  "serial_mpi_initialization": False,
                  "source_sha256": {str(path.relative_to(ROOT)):
                                    hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted((ROOT / "src").rglob("*.py"))}}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, payload in [("fixed_schedule.json", fixed_schedule_reference()),
                          ("legacy_probes.json", legacy_probes())]:
        (args.output / name).write_text(
            json.dumps({"provenance": provenance, "observations": payload},
                       default=serializable, indent=2, allow_nan=False) + "\n")
    print(f"Captured reference and observations in {args.output}")


if __name__ == "__main__":
    main()
