"""Small checks for restart, failure accounting and explicit large-run chunks."""

import json
from types import SimpleNamespace
import tracemalloc

import pytest

from benchmarks import run_classic


def arguments(directory, **kwargs):
    return SimpleNamespace(**dict(dict(profile="fixed", scenario="paper-independent-0", start=None,
                                       stop=None, output=str(directory)), **kwargs))


def test_explicit_large_chunks_and_nonoverlapping_seeds(tmp_path):
    with pytest.raises(ValueError, match="explicit --start/--stop"):
        run_classic.work_items(arguments(tmp_path, profile="large", scenario="paper-independent"))
    with pytest.raises(ValueError, match="explicit valid --scenario"):
        run_classic.work_items(arguments(tmp_path, profile="large", scenario=None, start=0, stop=1))
    first = run_classic.work_items(arguments(tmp_path, profile="large", scenario="paper-independent", start=0, stop=2))
    second = run_classic.work_items(arguments(tmp_path, profile="large", scenario="paper-block11-low", start=0, stop=2))
    assert len({case["data_seed"] for case in first + second}) == 4
    assert all(case["m"] == 5000 for case in first + second)


def test_resume_retains_failure_and_rejects_mismatched_provenance(tmp_path, monkeypatch):
    calls = []

    def failed_fit(*args):
        calls.append(1)
        raise ValueError("injected scientific-fit failure")

    monkeypatch.setattr(run_classic, "paired", failed_fit)
    monkeypatch.setattr(run_classic, "load_legacy", lambda: None)
    args = arguments(tmp_path / "audit")
    assert run_classic.run(args)
    assert run_classic.run(args)
    assert len(calls) == 1  # Resume does not replace or retry a failed dataset.
    summary = json.loads((tmp_path / "audit/summary.json").read_text())
    assert summary["failures"] == 1
    assert summary["groups"][0]["summary"]["selection_estimates"] is None
    manifest = tmp_path / "audit/run.json"
    contents = json.loads(manifest.read_text())
    contents["reference_base"] = "different revision"
    manifest.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="different code/environment/profile"):
        run_classic.run(args)


def test_memory_instrumentation_stops_on_a_failed_replay():
    calls = []

    def fails_second_time():
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("memory replay failure")

    with pytest.raises(ValueError, match="memory replay"):
        run_classic.measured(fails_second_time)
    assert not tracemalloc.is_tracing()
