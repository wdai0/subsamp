"""Optional live cross-language reference check; downloads nothing during tests."""

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest


SOURCE_SHA256 = "7d5fcff9d3b5c2d7255646c7e70dc8afa36e5b31c1e0d67df6324f31977537e9"


def test_archived_cran_scoring_on_identical_data_and_subsets(tmp_path):
    source_name = os.environ.get("SUBSAMP_CRAN_SOURCE")
    if not source_name:
        pytest.skip("Set SUBSAMP_CRAN_SOURCE to the verified subsamp 0.1.0 R source")
    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("Rscript is unavailable")
    source = Path(source_name).resolve()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    test_dir = Path(__file__).parent
    fixture = json.loads((test_dir / "reference/fixed_schedule.json").read_text())["observations"]
    for key in ("X", "y", "schedule"):
        np.savetxt(tmp_path / f"{key}.csv", fixture[key], delimiter=",",
                   fmt="%d" if key == "schedule" else "%.17g")
    subprocess.run([
        rscript, "--vanilla", str(test_dir / "run_cran_reference.R"), str(source),
        str(tmp_path / "X.csv"), str(tmp_path / "y.csv"),
        str(tmp_path / "schedule.csv"), str(tmp_path),
    ], check=True, capture_output=True, text=True, timeout=30)

    r_submodels = np.loadtxt(tmp_path / "submodels.csv", delimiter=",")
    py_submodels = fixture["cases"]["ols"]["submodels"]
    py_rss = np.array([result["rss"] for result in py_submodels])
    py_t = np.array([np.array(result["t_vals"])[row]
                     for result, row in zip(py_submodels, fixture["schedule"])])
    df = len(fixture["y"]) - fixture["s"] - 1
    assert_allclose(r_submodels[:, 0], py_rss, rtol=1e-11)
    assert_allclose(r_submodels[:, 1], np.sqrt(py_rss / df), rtol=1e-11)
    assert_array_equal(r_submodels[:, 2], df)
    assert_allclose(r_submodels[:, 3:], py_t, rtol=1e-11, atol=1e-12)

    r_ranked = np.loadtxt(tmp_path / "scores.csv", delimiter=",", skiprows=1)
    r_indices = r_ranked[:, 0].astype(int) - 1
    py_selection = fixture["cases"]["ols"]["selections"]["top_s"]
    py_scores = np.array(py_selection["scores"])
    assert_allclose(r_ranked[:, 1], np.sqrt(df) * py_scores[r_indices],
                    rtol=1e-11, atol=1e-12)
    assert_array_equal(r_indices[:3], py_selection["semifinalists"])

    # Final selection uses stepwise AIC in R, so do not assert final-model parity.
    with (tmp_path / "final_coefficients.csv").open() as stream:
        r_features = [row[0] for row in list(csv.reader(stream))[1:]]
    max_error = np.max(np.abs(r_ranked[:, 1] / np.sqrt(df) - py_scores[r_indices]))
    print(f"\n{(tmp_path / 'r_version.txt').read_text().strip()}; "
          f"max score difference after scaling: {max_error:.3g}; "
          f"R stepwise coefficient names: {r_features}")
