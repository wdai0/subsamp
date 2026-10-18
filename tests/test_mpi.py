"""Opt-in real multi-process tests; subprocess timeout catches deadlocks."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SUBSAMP_TEST_MPI") != "1", reason="set SUBSAMP_TEST_MPI=1 for real MPI integration")


@pytest.mark.parametrize("ranks,repetitions", [(3, 5), (4, 2)])
def test_actual_mpi_serial_equivalence_and_coordinated_errors(ranks, repetitions):
    # Opting in is a requirement to run, not permission to silently skip missing MPI.
    assert shutil.which("mpiexec") and importlib.util.find_spec("mpi4py"), "MPI runtime and mpi4py required"
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if environment.get("SUBSAMP_TEST_INSTALLED") == "1":
        environment.pop("PYTHONPATH", None)
    else:
        environment["PYTHONPATH"] = str(root / "src")
    command = [shutil.which("mpiexec"), "-n", str(ranks), sys.executable,
               str(root / "tests/mpi_driver.py"), str(repetitions)]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=50)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip())
    assert report["ranks"] == ranks and len(report["checks"]) == 9
