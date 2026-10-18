# subsampwinner

NumPy-based feature selection with the Subsampling Winner Algorithm (SWA), conventional OLS/WLS fitting, and an adaptive assurance procedure.
The implementation separates subset sampling, regression, retention, scoring, and significance filtering.

## Install and run

Python 3.9 or newer is required.
For this development branch, install from the checkout:

```bash
python -m pip install -e .
```

```python
import numpy as np
from subsampwinner import subsamp

rng = np.random.RandomState(7)
X = rng.normal(size=(80, 100))
beta = np.r_[.1, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, np.zeros(90)]
y = X @ beta + rng.normal(size=80)

swa = subsamp(s=10, m=120, qnum=10, seed=7)
swa.fit(X, y)
print(swa.finalists)             # zero-based original feature IDs
print(swa.final_model.summary()) # model on ALL semifinalists
```

`fit` returns `None`, as before.
The stored `final_model` is a statsmodels regression on the semifinalists; `finalists` filters those slope tests and does not trigger another regression.
Bonferroni and Benjamini–Hochberg adjust over the `qnum` semifinalists.
Use `method="none"` for unadjusted filtering; misspelled methods raise an error.
The post-selection p-values do not acquire a general selection-adjusted inferential guarantee from this refactor.

The default score averages `abs(t) / sqrt(raw RSS)` over nonzero t-statistics for each feature.
All submodels contribute by default.
`fit(..., use_top_s=True)` retains the best `min(s, m)` submodels by raw RSS before scoring, including when fitting WLS.
Positive `weights` mean inverse observation variances with conventional covariance and Student-t inference.

## Custom scoring

A scorer receives a public `SubsampleStatistics` record after retention.
It contains aligned `(retained_models, s)` arrays `feature_indices` and signed `t_values`, along with `rss`, `repetition_ids`, `n_features`, `df_resid`, and `weighted_rss`.
Arrays are owned read-only copies; IDs refer to original features and repetitions.
Return a finite vector of shape `(p,)` with a real integer or floating dtype.
Boolean, string, object, date, and complex outputs are rejected before conversion.
Negative scores are allowed, larger scores rank first, and the coordinator does not rescale or overwrite the returned values.

```python
from subsampwinner import SubsampleStatistics, default_scorer

def powered_score(stats: SubsampleStatistics) -> np.ndarray:
    # Illustrative method extension, not a validated recommended score.
    contribution = np.abs(stats.t_values)**1.5 / stats.rss[:, None]**.5
    ids = stats.feature_indices.ravel()
    totals = np.bincount(ids, weights=contribution.ravel(), minlength=stats.n_features)
    counts = np.bincount(ids, weights=(stats.t_values != 0).ravel(), minlength=stats.n_features)
    return np.divide(totals, counts, out=np.zeros_like(totals), where=counts != 0)

custom = subsamp(10, 120, qnum=10, seed=7, scorer=powered_score)
custom.fit(X, y, use_top_s=True)
# scorer=None and scorer=default_scorer give the same default method.
```

The package owns `fit_least_squares` and `LeastSquaresResult` as its regression boundary.
Statsmodels is the initial fitting engine.
Robust covariance, alternative weight interpretations, and a replacement solver require explicit extensions to that contract.
The public boundary owns inputs retained by the returned model; reusing caller arrays cannot change later diagnostics.
Repeated subset fits use the same numerical acceptance checks but do not calculate unused p-values.
See the [review revisions and validation](docs/REVISION_AUDIT.md) and [original implementation audit](docs/REFACTOR_AUDIT.md).

## Adaptive assurance and compatibility

```python
from subsampwinner import SubsampDoubleAssurance

assurance = SubsampDoubleAssurance(m=120, qnum=10, seed=7)
result = assurance.double_assurance(X, y, s0=10, I_max=3, return_objects=True)
print(result["stabilized_union_set"])
print(assurance.history_)
```

This preserves the existing adaptive largest-set update and cumulative union across iterations.
It is not the paper's union-model refit procedure, and the returned union is not necessarily a stable consensus.
An empty triplet stops with score zero and preserves earlier selections.
Invalid candidate ranges raise rather than being clipped.

Original class imports and the misspelled `subsample_anlysis` method remain available.
That compatibility method returns dense legacy dictionaries; `fit` uses compact storage of order `m*s + p` instead of `m*p`.
Both fitting and the default data generator own local RandomState streams and leave NumPy's global stream untouched.
Failed fits preserve previously successful fitted state.

Nonfinite inputs, supplied constant columns, singular fitted designs, nonpositive residual degrees of freedom, invalid weights, and numerically perfect fits raise informative errors.
Rank is checked on each actual fitted design, so `p > n` remains supported.
The relative rank cutoff is `max(max(n,k)*eps, 1e-15)`, including the intercept in `k`, to respect the backend pseudoinverse cutoff.
Overflow-safe norms and normalized bound arithmetic protect the residual check; a very large response offset can still be rejected by its conservative roundoff policy.
The `parallel`, `use_top_s`, and assurance `return_objects` flags require Boolean values, including NumPy booleans.
The old unstructured generator recipe can produce an invalid covariance; these draws now raise instead of generating data from an indefinite matrix.

## Optional MPI

```bash
python -m pip install -e '.[mpi]'
mpiexec -n 3 python your_script.py
```

Every rank calls `fit(..., parallel=True)` with identical inputs/options.
Root generates one schedule and invokes the scorer once; all ranks receive the complete fitted result.
The normal serial installation does not import or require mpi4py.
An MPI runtime is additionally required for parallel execution.

## Verification

```bash
python -m pip install -r tests/requirements.txt
python -m pytest -q
```

`tests/requirements.txt` freezes the complete serial reference stack for CPython 3.10, including pandas and patsy.
The PR/push workflow also tests fresh current dependencies and the minimum statsmodels backend with pandas 3 on Python 3.12.
It builds a source distribution and a wheel from that source distribution, checks their contents, and runs the full serial suite against the installed wheel using the source archive's tests.
Branch protection must separately require the workflow checks in repository settings.

The normal command includes the 25 saved classic datasets and schedules, including a 5,000-submodel case.
Expected outputs were captured from the unmodified pinned legacy revision before refactoring.
Optional R and actual multi-process checks are explicit:

```bash
SUBSAMP_CRAN_SOURCE=/path/to/pinned/subsamp.R python -m pytest tests/test_cran_reference.py -q
SUBSAMP_TEST_MPI=1 python -m pytest tests/test_mpi.py -q
```

See [benchmark instructions](benchmarks/README.md) for the paired audit and the prepared larger evaluation.
The [reference behavior specification](docs/REFERENCE_BEHAVIOR.md) and [stage 1 handoff](docs/REFERENCE_HANDOFF.md) preserve the pre-refactor evidence.
The [refactor audit](docs/REFACTOR_AUDIT.md) records implemented repairs, measured trade-offs, verification, and remaining limits.

The [revision audit](docs/REVISION_AUDIT.md) records the external-review fixes and repeated production timing against both legacy and the first refactor.

## Repository and release contents

Code, tests, the essential frozen reference fixtures, benchmark tools/configurations, and concise documentation stay together in this repository.
Generated reports and run outputs belong in the ignored `output/` directory and are archived separately when needed.
The full pre-cleanup evidence is retained in the [versioned audit archive](docs/AUDIT_ARCHIVE.md).
The wheel contains only the runtime package and metadata; the source distribution also includes the complete tests, fixtures, and benchmark helpers needed for verification.
See [packaging and release checks](docs/PACKAGING.md) for local build commands and the checks required before publication.
