"""Optional MPI transport; statistical operations remain in the serial core."""

import pickle
import numpy as np

from ._core import _analyze_schedule_validated, sample_schedule
from .scoring import SubsampleStatistics


def communicator():
    try:
        from mpi4py import MPI
    except ImportError as error:
        raise ImportError("parallel=True requires subsampwinner[mpi] and an MPI runtime") from error
    return MPI.COMM_WORLD


def collective_call(comm, label, operation):
    """All ranks reach error agreement before any dependent collective."""
    result, error = None, None
    try:
        result = operation()
    except Exception as caught:
        error = f"rank {comm.rank}: {type(caught).__name__}: {caught}"
    errors = comm.allgather(error)
    if any(errors):
        raise RuntimeError(f"MPI {label} failed: " + "; ".join(e for e in errors if e))
    return result


def root_call(comm, label, operation):
    """Execute once and broadcast a result or error, including serialization errors."""
    message = None
    if comm.rank == 0:
        try:
            message = (True, pickle.dumps(operation(), protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as error:
            message = (False, f"{type(error).__name__}: {error}")
    success, payload = comm.bcast(message, root=0)
    if not success:
        raise RuntimeError(f"MPI {label} failed on root: {payload}")
    return pickle.loads(payload)


def analyze_parallel(comm, X, y, s, m, seed, weights=None):
    schedule = root_call(comm, "schedule", lambda: sample_schedule(X.shape[1], s, m, seed))
    repetitions = np.array_split(np.arange(len(schedule)), comm.size)[comm.rank]
    local = collective_call(comm, "submodel fitting", lambda: (
        _analyze_schedule_validated(X, y, schedule[repetitions], weights, repetitions) if len(repetitions) else None))
    batches = comm.gather(local, root=0)

    def combine():
        active = [batch for batch in batches if batch is not None]
        ids = np.concatenate([batch.repetition_ids for batch in active])
        order = np.argsort(ids)
        fields = [np.concatenate([getattr(batch, key) for batch in active])[order]
                  for key in ("feature_indices", "t_values", "rss", "repetition_ids", "df_resid", "weighted_rss")]
        return SubsampleStatistics(*fields[:4], X.shape[1], *fields[4:])

    return root_call(comm, "statistics gathering", combine)
