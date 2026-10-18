# Classic benchmark and paired audit

The permanent regression panel has 25 fixed datasets: two for each of the 12 scenarios below, plus one independent case with 5,000 submodels.
All cases have `n=80`, `p=100`, `s=q=10`, and coefficients `(0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5)` followed by 90 zeros.
The first 24 cases use `m=120`.
This exact allocation, seeds, and `s=q=10` are engineering choices because those details were absent from the pasted review; they are not claimed as the paper's experiment settings.
The actual data, ordered subsets, expected intermediate statistics, and both all/top-s outputs are saved in `tests/reference/classic/`, protected by SHA-256 checksums.

## Scenarios

The three paper-derived scenarios, as specified in the supplied review, use independent predictors with noise SD 1, or an AR(1) block on the first 11 predictors with rho 0.5 and noise SD 1 or 3.
The remaining 89 predictors in the block cases are independent, with zero cross-block covariance.
The nine engineering scenarios cross independent, first-11-block, and full-100 AR(1) rho 0.9 covariances with population SNR 1, 5, and 20.
Noise SD is `sqrt(beta.T @ Sigma @ beta / SNR)`.
The corresponding signal variances are 76.01, 179.188515625, and 417.657110837.

The supplied review cites the published paper's equation (2) and simulation settings.
The published PDF and linked extended handoff were not themselves attached here; this refactor uses the pasted decisions, the previously inspected preprint, and the pinned Python/CRAN source comparisons.
Python still adjusts over `q` semifinalists, rather than the paper comparison's 100 candidates.

## Normal regression tests

```bash
python -m pytest -q
```

This runs the complete fixed panel and analytic tests.
It checks raw RSS, signed slope t-statistics, scores, feature IDs, coefficients, standard errors, t-tests, and adjusted p-values.
Selected IDs must match exactly; tolerances only cover floating-point statistics.
No all-ten-signals recovery requirement is imposed.
The weakest signals are deliberately difficult.
The tests include separate exact ties, adjacent floating-point values, and significance-threshold boundaries.

`tests/capture_classic.py NEW_DIRECTORY` can recreate evidence from the pinned unmodified Git source only, with a new output directory.
It must never be used to bless outputs from the changed implementation.
It requires mpi4py solely because the historical source imported MPI unconditionally.

## Paired measurement and restart

```bash
PYTHONPATH=src:. python -m benchmarks.run_classic --profile fixed --output output/paired-classic
```

The optional paired runner executes the pinned original code and current code on exactly the same arrays and ordered subsets, then checks numerical and selection agreement.
Install mpi4py for its historical reference adapter; normal tests and the package's serial path do not need it.
Timing excludes memory instrumentation.
A separate deterministic replay measures Python-traced peak allocation; this is not whole-process RSS or total native BLAS/LAPACK memory.
The measurement scope includes schedule analysis and both retention/final-filter modes; data generation and schedule creation are outside the timer.

Each completed case has an atomic JSON result.
Failures are retained as failures, cause a nonzero exit status, and are not silently rerun, replaced, omitted, or scored as empty selections.
Resume requires matching source hashes, reference revision, environment, and profile.
Use one writer per output directory; parallel chunks need separate directories.

Per-case outputs include each signal's selection indicator, signal recovery, semifinalist recall, false positives, false discovery proportion, any-false-positive indicator, selection size, and exact-support indicator.
The summary reports per-signal power, mean recovery, mean false-positive count, FDR as mean `FP/max(R,1)`, and FWER as the proportion with any false positive.
Summaries are grouped by scenario and the complete method configuration: `n,p,s,q,m`, retention, alpha, correction method, correction family, and scorer parameters.
The `m=5000` sentinel is a separate group from `m=120` cases.
Dataset-level Monte Carlo standard errors use the sample standard deviation divided by the square root of the number of independent datasets; a single dataset has no estimated MCSE.
With any failed dataset, that group's selection estimates are withheld rather than changing the denominator.
The fixed panel is a regression check; its one or two datasets per group do not support precise statistical performance estimates.

## Larger profile: prepared, not run at scale

`large-profile.json` specifies 1,000 independent datasets per each of 12 scenarios, `m=5000`, and both retention modes.
Its coefficient vector, population SNR, correction family, and implementation pairing match the explicit benchmark contract.
A chunk must be requested explicitly:

```bash
PYTHONPATH=src:. python -m benchmarks.run_classic \
  --profile large --scenario paper-independent --start 0 --stop 1 \
  --output output/large-smoke
```

The one-dataset command is a local pipeline smoke check, not a 1,000-dataset evaluation.
Each generated dataset and ordered schedule is saved before fitting.
For later scale-up, choose disjoint replicate intervals within `[0,1000)` and keep separate output directories.
The full profile contains 60 million submodels per implementation per replay, or 240 million submodel fits including both implementations and the separate memory replay.
Final regressions add two per implementation per replay per dataset.
No cluster wrapper, submission, or broad run is started by these changes.

## Method evaluation

Compatibility is the default `--mode compatibility`; it requires exact feature IDs and tolerance-based numerical agreement with the pinned legacy implementation.
Both methods' outcomes are saved alongside timing and memory measurements.
Use `--mode evaluation` for intentional method changes:

```bash
PYTHONPATH=src:. python -m benchmarks.run_classic \
  --mode evaluation --profile fixed \
  --baseline-config benchmarks/methods/default-q.json \
  --candidate-config benchmarks/methods/scaled-q.json \
  --output output/scaled-score-evaluation
```

Both configurations explicitly specify retention, alpha, correction method, correction family (`q` semifinalists or `p` original candidates), and scorer.
The selected profile supplies the shared `s,q,m` and ordered subsets; those values are also included in each saved method configuration.
The evaluation mode does not import the legacy source or require MPI.
It compares statistical methods using the current engine, while compatibility mode compares implementations of the original method.
The example power scorer computes the average of `abs(t)**t_power / RSS**rss_power` over nonzero t-values, then multiplies by positive `scale`.
The scale-two example is a reporting check: it changes score values while preserving ranks; it is not a proposed algorithmic improvement.
Other scorer functions can be added explicitly to this small benchmark module without changing the package's callback interface.

Evaluation records both outcomes independently, including a successful baseline if the candidate fails.
Valid differences in scores, semifinalists, or finalists are recorded separately from failures.
The paired summary reports candidate-minus-baseline differences and MCSEs across shared independent datasets.
In particular, paired FDR differences use per-dataset FDP differences, not the number of repeated submodels.
A failed pair withholds paired estimates; each method's estimates are separately withheld if that method failed on any dataset in its group.
One writer per directory and separate directories for disjoint chunks remain required.
These estimates do not establish post-selection error control; small smoke runs do not establish scientific improvement.

## Paper-derived configuration

The separately versioned `paper-profile.json` uses `s=q=30`, `m=5000`, `n=80`, `p=100`, and the same ordered ten-signal vector.
The supplied external review is the source of the `s=q=30` specification; its linked published PDF was unavailable here and has not been independently rechecked for this revision.
The profile prepares 1,000 datasets per scenario but requires an explicit chunk.
To compare the two correction families with identical data and scoring:

```bash
PYTHONPATH=src:. python -m benchmarks.run_classic \
  --mode evaluation --profile paper --scenario paper-independent --start 0 --stop 1 \
  --baseline-config benchmarks/methods/default-q.json \
  --candidate-config benchmarks/methods/default-p.json \
  --output output/paper-family-smoke
```

The p-family benchmark pads untested features with p-values of one before Bonferroni/BH adjustment, preserving the declared family size without fitting an unidentifiable 100-predictor regression.
The production package continues to adjust over semifinalists.
Explicit method JSON overrides the profile's descriptive correction-family default so this sensitivity comparison is fully recorded.
The existing 25 fixtures and checksums are unchanged.

## Repeat production measurements or regenerate historical summaries

```bash
PYTHONPATH=src:. python -m benchmarks.measure_revision --output output/repeated-performance.json
PYTHONPATH=src:. python -m benchmarks.reporting \
  /tmp/subsampwinner-audit-18f982b/output/refactor-audit/final-classic output/corrected-historical-summary.json
```

The measurement command compares unmodified legacy, the committed first refactor `c881315`, and current production code on the saved 5,000-submodel sentinel.
After one warm-up, it rotates execution order over five timing repetitions and checks each result outside the timer.
Traced allocation and cProfile are separate replays.
It requires the local Git history and MPI dependency for the historical adapter.
The summary command writes a new file with source-record checksums; historical records and frozen expectations are never replaced.

Generated results are ignored under `output/`; the package's tests depend only on the frozen fixtures in `tests/reference/`.
The historical records used by the final command are in the [audit archive](../docs/AUDIT_ARCHIVE.md); extract that snapshot to `/tmp` before regenerating its summary.
Keep new configurations and scripts in Git and archive large run records separately with their commit and checksums.
