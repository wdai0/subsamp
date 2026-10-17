"""Capture NEW fixtures from pinned legacy source, never the working implementation.

Usage: PYTHONPATH=. .venv/bin/python tests/capture_classic.py NEW_DIRECTORY
The output directory must not exist. Nothing is overwritten.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import mpi4py
import numpy as np
import scipy
import statsmodels

from benchmarks.classic import BETA, covariance, dataset, panel, selection_metrics

BASE = "e0366d3bcf917b875b982214a2ef342f8d463e42"
ROOT = Path(__file__).resolve().parents[1]


def capture(destination):
    destination.mkdir(parents=True, exist_ok=False)
    mpi4py.rc.initialize = False
    source = subprocess.check_output(["git", "show", f"{BASE}:src/subsampwinner/subsamp.py"], cwd=ROOT)
    legacy = types.ModuleType("pinned_legacy_subsamp")
    exec(compile(source, f"{BASE}/subsamp.py", "exec"), legacy.__dict__)
    manifest = dict(base=BASE, source_sha256=hashlib.sha256(source).hexdigest(),
                    environment=dict(numpy=np.__version__, scipy=scipy.__version__, statsmodels=statsmodels.__version__),
                    policy="2 datasets per 12 scenarios plus one m=5000; s=q=10; alpha=.05; Bonferroni over q",
                    signal_variances={name: float(BETA @ covariance(name) @ BETA)
                                      for name in ("independent", "block11", "full90")}, cases=[])
    for case in panel():
        started = time.perf_counter()
        X, y = dataset(case, case["data_seed"])
        rng = np.random.RandomState(case["schedule_seed"])
        schedule = np.array([rng.choice(100, case["s"], replace=False) for _ in range(case["m"])])
        swa = legacy.subsamp(case["s"], case["m"], case["q"])
        results = [swa._analyze_subsample(X, y, row) for row in schedule]
        arrays = dict(X=X, y=y, schedule=schedule,
                      t_values=np.array([r["t_vals"][row] for r, row in zip(results, schedule)]),
                      rss=np.array([r["rss"] for r in results]))
        metrics = {}
        for mode, top in (("all", False), ("top_s", True)):
            semi, scores = swa.process_subsample_results(results, case["q"], top)
            final, adjusted, model = swa.select_finalists(X, y, semi)
            for name, value in dict(scores=scores, semifinalists=semi, finalists=np.asarray(final, dtype=int),
                                    p_adjusted=adjusted, params=model.params, bse=model.bse,
                                    pvalues=model.pvalues, tvalues=model.tvalues).items():
                arrays[f"{mode}_{name}"] = value
            metrics[mode] = selection_metrics(semi, final)
        path = destination / (case["id"] + ".npz")
        np.savez_compressed(path, **arrays)
        manifest["cases"].append(dict(case, file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                      baseline_seconds=time.perf_counter() - started, metrics=metrics))
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Captured {len(manifest['cases'])} cases from {BASE}; no failures dropped.")


if __name__ == "__main__":
    capture(Path(sys.argv[1]))
