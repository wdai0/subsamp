"""Actual MPI integration assertions; launched by test_mpi.py."""

import json
import sys

from mpi4py import MPI
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from subsampwinner import default_scorer, subsamp
from subsampwinner._core import sample_schedule

comm = MPI.COMM_WORLD
m = int(sys.argv[1])
rng = np.random.RandomState(51)
X = rng.normal(size=(40, 8))
y = X[:, 0]*3 - X[:, 2]*2 + rng.normal(size=40)
checks = []

for weighted in (False, True):
    weights = np.linspace(.3, 2., 40) if weighted else None
    for top in (False, True):
        calls = []

        def scorer(statistics):
            calls.append(1)
            return default_scorer(statistics)

        parallel = subsamp(3, m, seed=7, scorer=scorer)
        assert parallel.fit(X, y, weights=weights, use_top_s=top, parallel=True) is None
        counts = comm.allgather(len(calls))
        assert counts == [1] + [0] * (comm.size - 1)
        serial = subsamp(3, m, seed=7)
        serial.fit(X, y, weights=weights, use_top_s=top)
        for key in ("feature_weights", "semifinalists", "p_adjusted"):
            assert_array_equal(getattr(parallel, key), getattr(serial, key))
        assert parallel.finalists == serial.finalists
        assert_allclose(parallel.final_model.params, serial.final_model.params, rtol=0, atol=0)
        assert_array_equal(parallel.subsample_statistics_.repetition_ids, serial.subsample_statistics_.repetition_ids)
        assert not parallel.subsample_statistics_.rss.flags.writeable
        assert parallel.final_model.model.exog_names == serial.final_model.model.exog_names
        assert "OLS" in str(parallel.final_model.summary()) or "WLS" in str(parallel.final_model.summary())
        checks.append(f"{'WLS' if weighted else 'OLS'} {'top' if top else 'all'}")

# Legacy adapter must also return complete, canonically ordered rows to all ranks.
results = parallel.subsample_anlysis(X, y, parallel=True)
assert_array_equal([r['indices'] for r in results], sample_schedule(8, 3, m, 7))
assert_array_equal([r['repetition_id'] for r in results], np.arange(m))
checks.append("complete legacy adapter")

for failure in ("validation", "worker_fit", "root_callback", "bad_callback_output"):
    previous = parallel.final_model
    invalid_X, invalid_y = X.copy(), y.copy()
    if failure == "validation" and comm.rank == 1:
        invalid_y[0] = np.nan
    if failure == "worker_fit" and comm.rank == 1:
        invalid_X[:] = invalid_X[:, :1]

    def broken(statistics):
        if failure == "root_callback":
            raise ValueError("intentional root callback error")
        if failure == "bad_callback_output":
            return np.full(statistics.n_features, np.nan)
        return default_scorer(statistics)

    parallel.scorer = broken
    caught = None
    try:
        parallel.fit(invalid_X, invalid_y, parallel=True)
    except RuntimeError as error:
        caught = str(error)
    messages = comm.allgather(caught)
    assert caught is not None and len(set(messages)) == 1, messages
    assert parallel.final_model is previous
    checks.append(f"coordinated {failure}")

if comm.rank == 0:
    print(json.dumps(dict(ranks=comm.size, repetitions=m, checks=checks, vendor=MPI.get_vendor())))
