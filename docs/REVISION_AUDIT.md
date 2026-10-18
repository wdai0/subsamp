# External review revisions and validation

Implemented locally on October 18, 2026, in `codex/reference-core-repair`, starting from `c8813150ea84c32d564e032f5718ecb36408e2a7`.
This revision addresses the supplied external review without changing the default SWA algorithm, its scoring equation, the statsmodels fitting engine, or the frozen reference data.
Generated evidence was subsequently moved into the [preserved audit archive](AUDIT_ARCHIVE.md#revision-audit); the results below describe the verified revision checkpoint.
The earlier [implementation audit](REFACTOR_AUDIT.md) and [performance diagnosis](PERFORMANCE_MEASUREMENTS.md) remain historical reports for the first refactor.

## Evaluation and changes

The review's architecture recommendation is supported by the local evidence.
The rank-boundary and overflow examples were reproduced before editing, and the existing reference panel continued to pass after the repairs.
The changes focus on reliable public boundaries, fewer repeated calculations, and interpretable experiment records.

| Review finding | Implemented change | Validation |
| --- | --- | --- |
| Reference environment omitted transitive dependencies | Freeze the complete serial Python 3.10 stack, including pandas/patsy; import OLS/WLS directly | Three fresh environments pass `pip check`, isolated wheel-import/minimal-fit checks, and the serial suite |
| Wrapper could accept a truncated pseudoinverse | Relative rank cutoff is `max(max(n,k)*eps, 1e-15)` on the actual fitted weighted design | Three-observation reproducer is rejected on both full and compact paths; test verifies the policy covers the installed backend's default cutoff |
| Residual guard could overflow | Use SciPy/BLAS vector norms and normalized scalar bound arithmetic | Response scaling by `1e150` and common WLS weights `1e300` retain the finite backend inference |
| Public model retained caller response/weights | Copy retained response and weights; design construction owns predictor data | Mutate caller X/y/weights before reading uncached R-squared through both public fitting APIs |
| Scorer output accepted inappropriate dtypes | Check original dtype before conversion; require signed/unsigned integer or floating values, finite shape `(p,)`, then copy | Reject Boolean, string, object, date, and complex outputs; retain negative scores and ownership |
| Flags used Python truthiness | Validate Boolean `parallel`, `use_top_s`, and `return_objects`, including NumPy booleans | Invalid fit flags fail before MPI import; stage helpers reject the same invalid flags |
| Every subset extracted unused p-values and repeated shared-data validation | One fitting engine and numerical policy, compact subset extraction, full public/final extraction | Regression test forbids intermediate p-value access; all reference statistics/selections remain unchanged |
| Benchmark treated intentional method differences as errors | Separate strict compatibility and explicit method-evaluation modes | A scale-two scorer produces valid changed scores and identical selected IDs |
| Summary pooled different repetition budgets | Group by scenario and complete statistical configuration | Historical `m=120` and `m=5000` records produce separate summaries without rewriting source records |
| Classic profile and CI safeguards were incomplete | Assert ordered coefficient vector; add separate `s=q=30` profile and PR/push workflow | Permanent 25-case panel stays unchanged; separate paper-profile smoke and three local runtime checks pass |
| Smaller helper/state issues | Validate repetition IDs before fitting, validate sampling-helper sizes, report collapsed assurance ranges | Targeted malformed-ID and collapsed-range tests pass |

The production score is still the average of `abs(t)/sqrt(raw RSS)` over nonzero t-statistics.
Retention still uses raw RSS for OLS and WLS; final inference still uses conventional Student-t tests on the joint semifinalist model and adjusts over `q` tests.
No replacement solver, weighted-design cache, batching framework, or new production statistical method was introduced.

## Numerical contract

Every subset and final fit uses the same engine and checks the selected design's rank, residual degrees of freedom, weighted-data finiteness, residual-scale acceptance, and finite required statistics.
The compact result contains signed slope t-values, raw RSS, weighted RSS, and residual df.
Only public/final extraction evaluates p-values and preserves the complete statsmodels result.
The serial/MPI coordinators validate and own shared data before the repeated-fit loop.

For `k` fitted columns including the intercept, the rank policy is at least as strict as statsmodels' `1e-15` pseudoinverse truncation cutoff.
It rejects the reviewed near-singular design rather than treating a centered closed-form result as a substitute solver.
The installed-backend cutoff check is also part of CI, so an incompatible future backend default fails a test.

The residual bound remains mathematically `max(n,k)*eps*(||weighted_y|| + sigma_max*||beta||)`.
Norms use overflow-safe BLAS primitives; the comparison is normalized before multiplying or adding large magnitudes.
An unrepresentable residual scale is reported as a numerical-range failure.
A finite but very large response offset can still trigger the conservative roundoff bound, which the error message now explains.
The moderate shift/rescaling tests and new extreme-scaling examples are checked cases, not an unlimited invariance guarantee over floating-point inputs.

## Repeated production performance

The saved sentinel has `n=80`, `p=100`, `s=q=10`, and `m=5000`.
The measured operation includes subset analysis and both retention/final-filter modes; data generation and schedule construction are excluded.
All three implementations use the same arrays and ordered subsets.
After one warm-up each, five uninstrumented runs rotate execution order; numerical and exact-ID comparisons run outside the timer.
Memory and profiling use separate replays.
The previous implementation is loaded unmodified from `c881315`, and legacy from `e0366d3bcf917b875b982214a2ef342f8d463e42`.

| Metric | Legacy | First refactor | Revised production |
| --- | ---: | ---: | ---: |
| Median time, seconds | 0.442094 | 0.790256 | 0.551720 |
| Observed range, seconds | 0.435288–0.448215 | 0.778084–0.798764 | 0.539101–0.555067 |
| Python-traced peak allocation, bytes | 18,587,548 | 2,231,532 | 2,217,214 |
| Profiled Python call count | 1,496,098 | 2,861,884 | 2,152,071 |

The revised implementation is **30.2% faster than the first refactor**, remains **24.8% slower than legacy**, and uses **88.1% less traced peak allocation than legacy** on this workload.
The earlier refactor's extra runtime relative to legacy is reduced by about 68.5% in these repeated measurements.
All timed results preserve exact feature IDs and numerical agreement within the fixed-panel tolerances.

The production change removes unused intermediate p-value computation, repeated shared-data checks, and unnecessary full-result copies.
The remaining checks and result handling still have a cost; the residual runtime difference has not been fully decomposed into individual causes.
The separate profiler records 5,002 calls to the shared fitting engine, including the two final regressions, and shows statsmodels fitting/factorization among the main remaining costs.
Profiled times are not the timing estimates above, and Python call counts are not floating-point-operation counts.
No additional caching or custom numerical kernel was added after profiling.

These are local measurements on macOS arm64 with Python 3.10.7, NumPy 1.26.4, SciPy 1.13.0, and statsmodels 0.14.2.
Traced allocation is not whole-process resident memory and does not include every native-library allocation.
Five repeats on one workload do not establish a cross-platform speed guarantee.
Raw measurements, all repetitions, environment metadata, and source hashes are in [production-performance.json](AUDIT_ARCHIVE.md#revision-audit).
The reproducible command is in [benchmark instructions](../benchmarks/README.md).

## Verification results

| Check | Result |
| --- | --- |
| Combined suite, including pinned R source and actual MPICH processes | **123 passed**, zero skips |
| Fresh serial reference environment | **120 passed**, three explicit optional R/MPI skips |
| Fresh current dependency environment | **120 passed**, three optional skips; three backend warnings on deliberately singular test inputs |
| Fresh minimum-backend/pandas-3 environment | **120 passed**, three optional skips |
| Installed wheel in each fresh environment | Import, finite OLS inference, minimal SWA fit, and `pip check` pass |
| Strict fixed-panel runner | **25 completed, zero failures**, 50 exact dataset/retention selection comparisons |
| Scale-two method smoke | One valid pair, changed scores, unchanged semifinalists/finalists, no failures |
| Paper-profile correction-family smoke | One `s=q=30,m=5000` dataset, two valid methods, no failures |
| Frozen fixtures and original raw audit records | No changes |
| Working diff | Whitespace/error check passes |

The fresh environments were:

| Stack | Python | NumPy | SciPy | statsmodels | pandas | patsy |
| --- | --- | --- | --- | --- | --- | --- |
| Reference | 3.10.7 | 1.26.4 | 1.13.0 | 0.14.2 | 2.3.3 | 1.0.3 |
| Current resolved | 3.12 | 2.5.4 | 1.18.1 | 0.15.0 | 3.0.6 | 1.0.3 |
| Minimum backend with pandas 3 | 3.12 | 1.26.4 | 1.13.0 | 0.14.2 | 3.0.6 | 1.0.3 |

The installed-package smoke runs use Python isolated mode and assert a `site-packages` import path, avoiding pytest's source-checkout path injection.
The complete pytest suites test the source checkout under each environment.
The runtime metadata retains its existing lower bounds; no permanent pandas upper bound was added.
These three combinations are verified locally; every possible dependency combination and Python/platform version is not covered.

Logs and full environment freezes are in [environment-reference.json](AUDIT_ARCHIVE.md#revision-audit), [environment-current.json](AUDIT_ARCHIVE.md#revision-audit), and [environment-pandas3.json](AUDIT_ARCHIVE.md#revision-audit).
The combined result is in [combined-tests.json](AUDIT_ARCHIVE.md#revision-audit).
The workflow builds an installed wheel and runs the serial reference suite on pull requests and pushes, using these three dependency strategies.
It has not yet run on GitHub, and making its status checks mandatory still requires branch-protection settings.

## Experiment reporting

Compatibility mode continues to require agreement with the legacy implementation and now saves both outcomes.
Method-evaluation mode uses two explicit method configurations with shared dataset/schedule settings and the current fitting engine.
It records both outcomes, each method's failures, and valid changes in scores and selected IDs independently.
The provided configurable power scorer is experimental infrastructure; no improved method is claimed.

Summaries include the full scenario and `n,p,s,q,m`, retention, alpha, correction method, correction family, and scorer identity/parameters.
FDR is the mean per-dataset `FP/max(R,1)`; paired FDR comparisons average candidate-minus-baseline FDP differences across shared datasets.
Monte Carlo SE is the sample SD of those dataset-level differences divided by `sqrt(number of datasets)`.
Repeated submodels are never treated as independent statistical replications.
A single dataset has `mcse=null`; any failed pair withholds the paired estimates, and method-specific failures withhold that method's estimates.

The historical independent/top-s records now give:

| Configuration | Datasets | Mean signal recovery | Mean FDP |
| --- | ---: | ---: | ---: |
| `s=q=10,m=120` | 2 | 0.40 | 0 |
| `s=q=10,m=5000` | 1 | 0.70 | 0.125 |

The original pooled recovery of 0.50 is superseded as a summary of a configured procedure.
The regenerated [historical summary](AUDIT_ARCHIVE.md#revision-audit) records source-record hashes; the old records and fixtures remain intact.
Fresh compatibility output is in [fixed-compatibility/summary.json](AUDIT_ARCHIVE.md#revision-audit).

The separate [paper profile](../benchmarks/paper-profile.json) has `s=q=30` and requires explicit evaluation configs to expose the `q` versus `p=100` correction-family choice.
Its source is the supplied review; the review's linked published PDF and detailed memo were not attached and have not been independently checked here.
For p-family correction, untested predictors are assigned p-values of one before Bonferroni/BH adjustment.
This is a benchmark sensitivity setting; production filtering remains over `q` semifinalists.
The single smoke dataset selected eight true signals and no noise variables under both correction families; that observation only validates the execution path.
It does not estimate power/FDR precisely or establish equivalence between the families.
The 1,000-dataset-per-scenario profiles are prepared but were not run at scale.

## Scope and remaining limits

All implementation changes are confined to the existing branch.
This report accompanies the user-approved revision commit.
The original checkout, main branch, releases, and repository settings were not modified.
This revision fixes implementation and reporting issues while preserving the core method on the frozen supported cases.
It does not establish new statistical error-control guarantees, validate a new scoring method, or claim exhaustive numerical stability.
GitHub execution/required-check enforcement, independent verification of the published-paper configuration, and a properly sized method simulation remain separate work.
