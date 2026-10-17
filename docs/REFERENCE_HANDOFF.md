# Stage 1 handoff

Historical stage 1 record: for the subsequent implementation and current validation status, see [the refactor audit](REFACTOR_AUDIT.md).

## Scope and workspace

Completed reference verification and characterization only, following the user's narrowed scope.
No changes were made to package source, imports, public APIs, dependency metadata, the notebook, or publishing workflows.
No implementation repairs, remote writes, releases, or heavy simulations were performed.

- Worktree: `/Users/wei/.codex/worktrees/e10e/subsamp`.
- Base: `e0366d3bcf917b875b982214a2ef342f8d463e42`.
- Branch: `codex/reference-core-repair` (created before the scope was narrowed; the diff contains stage 1 only).
- The original checkout at `/Users/wei/Documents/GitHub/subsamp` remains on `main`; its files were not edited.
- New `.venv` is local to this worktree and ignored by Git.

## Main findings

The supported score is `1/sqrt(RSS)`: the accessible preprint's equation (3) and the current Python computation agree.
The R reference divides by residual standard error, which gives the same ranking at fixed `n,s` for full-rank OLS but multiplies scores by `sqrt(n-s-1)`.
The final publisher version's equation remains unverified because full-text access failed; the evidence must not be described as a completed final-journal audit.

The CRAN regression package was found in the archive as `subsamp` 0.1.0, rather than the classification package `swa`.
Running its actual scorer on the saved Python subsets confirmed the scale relationship to numerical precision.
Its final stepwise AIC selection differs from Python's significance filtering, and the Python adaptive assurance loop is a further adaptation.
See [the behavior specification](REFERENCE_BEHAVIOR.md) for precise sources, hashes, formulas, assumptions, and unresolved choices.

## Reviewable artifacts

| Path | Purpose |
| --- | --- |
| `docs/REFERENCE_BEHAVIOR.md` | Source comparison and behavior specification, with defects separated from preservation targets. |
| `tests/reference/fixed_schedule.json` | Explicit 30-by-6 data, response, positive WLS weights, 12 three-feature subsets, OLS/WLS submodel values, and all-model/top-`s` results. |
| `tests/reference/legacy_probes.json` | Historical observations, including failures; not asserted as desired behavior. |
| `tests/test_reference_behavior.py` | Independent numerical checks and characterization of the existing assurance adaptation. |
| `tests/test_cran_reference.py`, `tests/run_cran_reference.R` | Optional local execution of the downloaded, hash-verified R reference; no downloads during tests. |
| `tests/capture_reference.py` | Captures the baseline into a new directory; refuses changed `src` relative to the base commit. |
| `tests/requirements.txt`, `docs/reference-environment.txt` | Direct test requirements and full installed-package snapshot. |

## Validation environment

Validation used Python 3.10.7 on macOS arm64, with NumPy 1.26.4, SciPy 1.13.0, statsmodels 0.14.2, mpi4py 4.1.1, and pytest 8.4.2.
The numerical-library versions meet the declared package minimums.
R 4.3.2 executed the archived R source using base R packages, without installing the archived package or changing the user's R library.
Open MPI 5.0.7 is installed on this machine, but MPI initialization was explicitly disabled for all serial probes and tests.
The original checkout's environment was not used or repaired.

## Commands and results

All commands below run from the worktree root.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r tests/requirements.txt
PYTHONPATH=src .venv/bin/python tests/capture_reference.py tests/reference
PYTHONPATH=src .venv/bin/python -m pytest -q
SUBSAMP_CRAN_SOURCE=/tmp/subsamp-reference/subsamp.R PYTHONPATH=src .venv/bin/python -m pytest -q -s
PYTHONPATH=src .venv/bin/python -m compileall -q src tests
git diff --check
git diff --exit-code e0366d3bcf917b875b982214a2ef342f8d463e42 -- src setup.py requirements.txt .github demo_subsamp.ipynb
```

The capture command was run before `tests/reference` existed; it deliberately refuses to overwrite that directory.
For a repeat capture, choose a new scratch directory and compare results without replacing the approved fixture.

The initial Python suite passed **14 tests**.
With the CRAN cross-language check included, **15 tests passed** in 1.47 seconds.
After adding the optional R test, a default run without `SUBSAMP_CRAN_SOURCE` reports 14 passed and one explicit skip; that skip is not R validation.
The independent calculation checks coefficients, t statistics, p values, and raw RSS using least squares and the classical covariance formula.
The fixed-schedule cases cover OLS and WLS, each with all-model and top-`s` aggregation.

Observed fixed-schedule outputs, using zero-based indices:

| Fit | Retained models | Semifinalists | Finalists |
| --- | --- | --- | --- |
| OLS | All 12 | `[1,0,5]` | `[1,0]` |
| OLS | Best 3 | `[0,1,5]` | `[0,1]` |
| WLS | All 12 | `[1,0,5]` | `[1,0]` |
| WLS | Best 3 | `[0,1,5]` | `[0,1]` |

The R scorer's maximum score difference from Python after dividing by `sqrt(26)` was `4.44e-16`.
The actual R stepwise result retained `(Intercept), X1, X2, X6`; whole-pipeline parity is therefore explicitly not claimed.

The separate seeded `n=60,p=12,s=4,m=80,qnum=4,seed=7` probe reproduced semifinalists `[0,1,10,7]`, finalists `[0,1]`, a four-feature `final_model`, and a `None` return from `fit`.
All eight previously reported serial probes were rerun, including the covariance eigenvalue and global RNG side effect.
Additional probes covered singular predictors, zero residual degrees of freedom, and zero assurance iterations.

## Reproducing the optional R check

The R source is not vendored into this repository.
Download the archive into scratch space and verify it before extracting the single source file:

```sh
mkdir -p /tmp/subsamp-reference
curl -L --fail -o /tmp/subsamp-reference/subsamp_0.1.0.tar.gz https://cran.r-project.org/src/contrib/Archive/subsamp/subsamp_0.1.0.tar.gz
shasum -a 256 /tmp/subsamp-reference/subsamp_0.1.0.tar.gz
tar -xOf /tmp/subsamp-reference/subsamp_0.1.0.tar.gz subsamp/R/subsamp.R > /tmp/subsamp-reference/subsamp.R
SUBSAMP_CRAN_SOURCE=/tmp/subsamp-reference/subsamp.R PYTHONPATH=src .venv/bin/python -m pytest -q -s
```

Expected archive SHA256: `cd02097e9f4d2e936139005208857b9d6ffebc562163d4e2c88a078fc517dfad`.
The Python test also checks the exact source SHA256 before executing it.
The test harness supplies the saved feature subsets through an isolated R function environment; it does not compare unrelated R and NumPy seed sequences.
It evaluates the original nested scorer and then calls the full R function with plotting disabled.

## Remaining limits

The final journal full text still needs an accessible copy for a definitive version-to-version equation audit.
MPI behavior has been inspected in source only; no multi-process test was run.
No large-dimensional performance, memory, cross-platform, general FDR, selective-inference, or heteroskedastic-robustness claim is supported by these examples.
Known defects remain present because implementation repairs are outside this narrowed stage.
The assurance tests characterize the current cumulative union and update rule; they do not endorse those choices as the published procedure or as a validated stability method.
