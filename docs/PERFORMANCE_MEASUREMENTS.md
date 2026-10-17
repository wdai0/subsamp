# Performance measurements and slowdown diagnosis

This report accompanies the current refactor and [implementation audit](REFACTOR_AUDIT.md).
The legacy reference is Git revision `e0366d3bcf917b875b982214a2ef342f8d463e42`.
No runtime optimizations from the diagnostic experiments below have been applied to the implementation in this snapshot.

## Recorded paired audit

Both implementations received identical saved datasets and ordered feature subsets.
The main comparison uses 80 observations, 100 candidate features, 10 features per submodel, 10 semifinalists, and 5,000 submodels.
The measured operation includes submodel analysis and both all-model and top-s retention/final-filter modes.
Data generation and schedule construction are excluded.
Timing and Python-traced peak allocation were measured in separate executions.

| Metric | Legacy | Refactored |
| --- | ---: | ---: |
| 5,000-submodel execution time | 0.446950 s | 0.797369 s |
| 5,000-submodel traced peak allocation | 18,587,484 bytes | 2,215,297 bytes |
| Summed execution time across the 25-case panel | 0.717410 s | 1.270410 s |
| Selected feature IDs, 25 datasets and two retention modes | Reference | 50/50 exact matches |

The large case uses 88.1% less traced peak memory, an 8.4-fold reduction, and takes 78.4% longer in this measurement.
The 25-case panel includes 7,880 submodel fits per implementation per replay.
Its total measured execution time increases by 77.1%.
These timings are local observations, not cross-platform estimates.
Python-traced allocation is not whole-process resident memory and excludes some native-library allocations.
Raw measurements, per-case selections, and scenario summaries are in `output/refactor-audit/final-classic/`.

## Follow-up diagnosis

The saved 5,000-submodel workload was rerun with a warm-up and five uninstrumented timing repetitions per variant, alternating execution order.
A separate cProfile run identified expensive call paths.
The following table reports uninstrumented medians and ranges, not profiler-inflated times.

| Variant | Median | Observed range |
| --- | ---: | ---: |
| Pinned legacy | 0.445149 s | 0.442330–0.447427 s |
| Current refactor | 0.799703 s | 0.797058–0.807159 s |
| Diagnostic: skip unused submodel p-values | 0.610294 s | 0.597051–0.614820 s |
| Diagnostic: also skip repeated input validation | 0.557711 s | 0.554716–0.564864 s |

The diagnostic variants were temporary in-memory clones of the regression function, used only on the known valid saved dataset.
They did not modify repository files.
The first variant replaced the unused intermediate submodel p-value arrays with placeholders, while retaining the original final-regression implementation and its real p-values.
The second additionally bypassed repeated input validation inside those submodel calls.
Both variants preserved feature scores within the existing tolerance and selected feature IDs exactly on this workload.
Neither variant is a production-ready change or a validation of skipping checks on arbitrary inputs.

The current implementation eagerly computes p-values for all 5,000 submodels through the owned least-squares result.
SWA scoring uses their t-statistics and RSS, so these submodel p-values are unused.
The final semifinalist regressions still need p-values.
Avoiding the unused intermediate calculation saved about 0.189 s, approximately 53% of the observed extra runtime.

Repeated checks of dimensions, finiteness, and constant columns add another approximately 0.053 s, or 15% of the extra runtime in this sequential experiment.
The response vector is unchanged across submodels but is checked repeatedly.
Together, the two diagnostic changes removed approximately 68% of the additional runtime on this workload.
Remaining overhead includes numerical acceptance checks, result construction, and array copying; it was not further isolated quantitatively.

The profiled workload recorded approximately 1.50 million function calls for legacy and 2.86 million for the current implementation.
Both used the same number of statsmodels fits and SVD calls.
The profiler identified statsmodels p-value calculation and SciPy's Student-t survival-function path as prominent added costs.
Profiler call counts do not measure floating-point operations or hardware efficiency.

## Implementation and reliability metrics

| Measure | Legacy to refactored |
| --- | --- |
| Submodel-statistics storage | O(mp) to O(ms + p), excluding input/design storage |
| Main t-statistic entries in this large case | 500,000 to 50,000 |
| Separate regression-fitting implementations | Two duplicated paths to one shared owned boundary |
| Mandatory runtime dependencies | Four to three; MPI is optional |
| Repository automated tests | No test files at the pinned base to 90 passing checks in the combined audit |
| Reference selection preservation | Exact IDs across all 50 dataset/retention comparisons |
| Failed refits | Partial fitted state possible to previous successful state preserved |
| Custom scoring | Code modification/subclassing to a documented callback |

The compact representation also stores feature IDs and scale/df metadata; the measured memory figures include that overhead.
The dense legacy compatibility adapter still uses O(mp) space when explicitly requested.
The 90-check audit includes the external pinned R comparison and actual MPI tests with uneven work and idle ranks.
Normal serial testing in a clean environment without mpi4py yielded 87 passes and three explicit optional-check skips.

## Next optimization, not implemented here

The evidence supports moving shared-data validation outside the repeated-fit loop, computing only the statistics needed by submodel scoring, and keeping necessary per-fit rank, residual-df, residual-scale, and finite-output checks.
The public least-squares boundary and statsmodels engine can remain.
A future implementation must rerun the same frozen reference panel and failure-contract tests, then repeat timing and memory measurements.
No speedup or safety preservation for that future production change is claimed by these diagnostic experiments.
