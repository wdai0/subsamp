# SubsampWinner 0.1.0

This release refactors the existing SWA implementation into separate sampling, least-squares, retention, scoring, and inference stages.
The default algorithm remains unchanged on the frozen reference panel: average `abs(t)/sqrt(raw RSS)` over nonzero contributions, optional top-s retention by raw RSS, and significance filtering on the joint semifinalist model.
It adds an explicit scorer callback and a small least-squares interface for future extensions without replacing the statsmodels fitting engine.

## Compatibility

- Original class imports, the legacy `subsample_anlysis` spelling, and the `None` return from `fit` remain supported.
- Python 3.9 or newer is required; MPI is optional through `subsampwinner[mpi]` and requires an MPI runtime.
- Fits now reject nonfinite inputs, constant supplied columns, singular fitted designs, nonpositive residual degrees of freedom, invalid weights, numerically unresolved residual scales, and invalid options.
- Randomness uses local RandomState streams; fits and default data generation no longer reset NumPy's global random state.
- Failed fits preserve the previous successful state, and retained model inputs are owned copies.
- Adaptive assurance exposes iteration history and termination reasons, stops safely on empty selections, and continues to return the historical cumulative union without a union-model refit.
- Invalid covariance draws from the legacy unstructured generator raise instead of silently producing data from an indefinite matrix.

## Validation and measured trade-offs

The local combined suite passed 123 tests, including the pinned R comparison and actual MPI processes.
Three local dependency environments each passed the 120 serial tests against an installed wheel, with the three optional R/MPI checks skipped.
The frozen panel contains 25 datasets and schedules, with exact selected-feature agreement in all 50 dataset/retention comparisons against legacy.
These checks establish the tested behavior, not general selection-adjusted inference or large-scale method performance.

On the recorded macOS arm64 workload (`n=80, p=100, s=q=10, m=5000`), median execution time was 0.552 seconds versus 0.442 seconds for legacy: 24.8% slower, with 88.1% less Python-traced peak allocation.
The revised implementation was 30.2% faster than the first refactor.
Traced allocation is not whole-process resident memory, and this workload is not a cross-platform speed guarantee.
See [the revision audit](REVISION_AUDIT.md) for methodology and limits.

## Distribution contents

The wheel contains runtime code and package metadata.
The source archive retains correctness tests, essential fixtures, benchmark tools, and concise documentation.
Generated results and audit logs are excluded from both distributions and the maintained source tree; their historical checkpoint is described in [the audit archive guide](AUDIT_ARCHIVE.md).
