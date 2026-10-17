# Refactor implementation and audit

Implemented and checked locally on October 17, 2026, in branch `codex/reference-core-repair` and worktree `/Users/wei/.codex/worktrees/e10e/subsamp`.
The reference base is `e0366d3bcf917b875b982214a2ef342f8d463e42`.
The two supplied review texts govern this implementation, including custom scoring now and an owned least-squares boundary backed by statsmodels.
This supersedes the earlier proposal's deferral of public scoring extensibility.

## Objective and implemented structure

Preserve the existing SWA method on supported inputs while making the implementation readable, testable, compact in memory, and ready for explicit future method extensions.
The production package remains NumPy/SciPy/statsmodels based.
No Torch or second regression solver was introduced.

| Component | Responsibility |
| --- | --- |
| `src/subsampwinner/subsamp.py` | Existing class/imports, staged coordination, atomic fitted-state publication, compatibility adapters |
| `src/subsampwinner/least_squares.py` | Owned typed OLS/WLS result and input/numerical contract; sole statsmodels regression backend boundary |
| `src/subsampwinner/scoring.py` | Public compact statistics, retention, named default scorer, callback validation, ranking |
| `src/subsampwinner/_core.py` | Ordered subset scheduling, repeated fitting, separate p-value filtering |
| `src/subsampwinner/_validation.py` | Input, dimension, count, seed, weight, and method validation |
| `src/subsampwinner/_parallel.py` | Lazy MPI import, root schedule, canonical gather, root-only scoring and coordinated errors |
| `src/subsampwinner/SubsampDoubleAssurance.py` | Existing adaptive procedure with corrected empty/range/history bookkeeping |
| `src/subsampwinner/GenerateData.py` | Local randomness and input/covariance validation for the historical generator |

No plugin registry, estimator hierarchy, build-system migration, package rename, release, or notebook rewrite is included.
The original checkout and its `main` branch were not edited.

## Preserved statistical behavior

- Default scores average `abs(t)/sqrt(raw RSS)` over **nonzero** t-statistics; an appearance with exactly zero t does not increment the denominator.
- The Python default retains all submodels; `use_top_s=True` retains `min(s,m)` by **raw** RSS, also for WLS.
- Retention precedes scoring; the public callback receives only retained rows and original feature/repetition IDs.
- Callback scores may be negative but must be a finite vector of length `p`; they are copied without normalization or post-processing.
- Ranking uses the legacy NumPy quicksort and reversed descending slice, including exact ties; no new random or lexicographic tie rule was substituted.
- The joint semifinalist model and original-feature mapping are preserved.
- `final_model` remains a statsmodels fit on semifinalists, supporting `summary()`; there is no finalist-model refit.
- Bonferroni/BH use the `q` semifinalist tests; unadjusted filtering is explicitly `method="none"`.
- Existing adjusted-test inclusive thresholds and unadjusted strict `p < alpha` are preserved and tested at adjacent floating-point values.
- `fit` returns `None`; original class imports, attribute names, and practical stage methods remain available.
- The adaptive assurance procedure still chooses the next center by largest individual selected-set size and returns the cumulative union without refitting it.

The paper/preprint top-s method, Python's all-model default, and R's final backward-AIC step remain distinct.
For full-rank same-size OLS submodels, the R score differs from Python's top-s score by `sqrt(n-s-1)`.
The actual pinned R source was rerun to verify matching stages; its final AIC selection is not used as a Python finalist equality target.

## Explicit repairs and boundaries

**Least squares.** An intercept is always added explicitly.
Constant input columns are rejected.
Each actual fitted weighted design must have full numerical rank and positive residual degrees of freedom; the entire candidate matrix is not rank-tested, so high-dimensional `p > n` inputs remain supported.
The rank threshold is `max(n,k) * eps * largest singular value`, including the intercept among the `k` columns.
The boundary reuses the statsmodels factorization's singular values to apply this policy without a redundant SVD.
A backend residual-df disagreement is also rejected.

WLS weights are positive diagonal inverse-variance weights.
The covariance is conventional/nonrobust, with Student-t reference inference and `n-k` residual df.
Raw and weighted RSS are separately named and retained.
Numerically perfect fits are rejected when the weighted residual norm is at or below `max(n,k)*eps*(norm(weighted_y) + largest_singular_value*norm(beta_hat))`.
This rejects inference at the numerical roundoff scale; no RSS floor, model dropping, or resampling occurs.
Robust covariance, frequency weights, estimated-weight uncertainty, and alternate engines remain explicit future statistical extensions.

**State and RNG.** A complete fit is calculated before fitted attributes are assigned.
Validation errors, callback errors, singular final fits, and coordinated MPI errors leave the previous successful fit intact.
Inputs and returned scores are owned copies.
Local `RandomState` preserves the verified legacy seeded subset sequence without resetting the caller's NumPy stream.
Repeated full fits reset their local schedule stream to the configured seed; standalone sampling-helper calls advance a separate private stream.
A custom generator `dist` callable remains responsible for its own randomness.

**Assurance.** Empty triplets stop with stability zero and keep the cumulative union from earlier iterations.
`final_range` is the last triplet actually evaluated, not a mix of old bounds and a newly chosen center.
`history_` contains independent set/list snapshots, with `best_stability_` and `termination_reason_` for inspection.
The existing three-key result dictionary is preserved.
A collapsed range evaluates its center once.
Zero iterations and invalid proposed candidate sizes raise; candidate bounds are not silently clipped.

**Generator.** The supported diagonal recipe and seeded values are preserved, including the historical variance lower bound of one.
Scalar variances are expanded to one value per observation.
The legacy unstructured recipe can produce an indefinite covariance: these draws now raise a clear error before error-vector generation.
No alternative PSD covariance distribution has been substituted; choosing one remains a separate data-generating-method decision.

**MPI.** Root generates one schedule; jobs carry original repetition IDs and are recombined in canonical order.
Root runs scoring and final filtering once, then broadcasts complete fitted state to all ranks.
Input and fitting errors are agreed collectively; root-stage and result-serialization errors are broadcast before continuing.
Every rank must enter the collective call with the same data/options.
This handles ordinary Python exceptions, not killed processes, transport failure, or callbacks that themselves perform mismatched collectives.

**Compatibility limits.** Unknown testing-method strings now raise rather than silently selecting unadjusted tests.
Invalid or nonidentifiable inputs that previously failed obscurely or returned unreliable values now fail explicitly.
Direct calls to the dense `subsample_anlysis` adapter still allocate `O(mp)` storage; normal `fit` uses `O(ms+p)` submodel statistics/scores, apart from input/design storage.
Python metadata now requires 3.9+, consistent with the existing numerical dependency floors.
MPI is an optional extra instead of a mandatory serial dependency.
The package version is unchanged; no release was made.

## Verification and audit evidence

The original 15 reference tests were recovered and passed before production edits.
The 25 classic datasets and ordered schedules were then captured from the unmodified pinned source; the capture process checks the source revision and records hashes.
No goldens were regenerated from candidate outputs.
The saved panel contains two datasets for each of the 12 specified scenarios and one additional `m=5000` case; `s=q=10`, the first 24 cases' `m=120`, and all seeds are explicitly frozen engineering choices.
These exact settings were not present in the pasted review and are not represented as paper settings.

| Check | Result |
| --- | --- |
| Final combined suite | **90 passed**, including external R and actual MPI checks |
| Classic panel | All 25 datasets and both retention modes match selected IDs exactly; intermediate statistics and final regressions match documented tolerances |
| Independent arithmetic | Hand scores, exact-zero t denominator, nonzero-intercept orthogonal OLS, analytic WLS coefficients/SE/t/p and raw-vs-weighted RSS pass |
| Invariance and boundaries | Predictor rescaling, response shift, common WLS weight scaling, rank threshold, exact/near ties, and p-value threshold tests pass |
| Public scoring/state | Default-callback parity, negative custom rankings, output ownership, retained input, malformed outputs, exception rollback, and singular-final-fit rollback pass |
| Repairs | Invalid counts/shapes/methods/weights, singular/perfect fits, global RNG isolation, high-dimensional fitting, assurance snapshots/ranges/empty selections, and covariance rejection pass |
| MPI | Three ranks/five repetitions and four ranks/two repetitions pass for OLS/WLS and both retention modes, including idle ranks, canonical legacy adapter output, one root callback, and coordinated failures |
| Paired measurement runner | All 25 fixed cases pass against freshly executed pinned legacy code; zero failures |
| Larger-profile smoke | One independent dataset with 5,000 submodels passes; restart reuses the completed result |
| Packaging | Wheel built; installed and executed in a temporary clean environment with no mpi4py; dependency check passes |
| Static checks | Python compilation and `git diff --check` pass |

Normal tests require neither MPI nor the external R source; the optional checks report skips unless explicitly enabled.
The serial suite was also run in the clean environment without mpi4py.
The tested numerical stack is Python 3.10.7, NumPy 1.26.4, SciPy 1.13.0, and statsmodels 0.14.2.
See `refactor-environment.txt` for the local environment.
Other Python/library combinations were not exhaustively tested.

The MPI integration tests passed with mpi4py 4.1.1 and MPICH 5.0.2 installed only in the worktree's virtual environment.
The system Open MPI 5.0.7 failed during MPI initialization before package code, both with its default transport and attempted TCP loopback settings.
An [upstream macOS issue](https://github.com/open-mpi/ompi/issues/13129) records the same symptom, but the precise cause on this host was not established.
The virtual-environment runtime follows the [mpi4py MPI-wheel installation option](https://mpi4py.github.io/mpi4py/stable/html/install.html).
No system MPI or firewall settings were changed.
Multi-host MPI/HPC operation remains unverified.

## Measured trade-off

The final paired audit measures schedule analysis and both retention/final-filter modes on the same inputs.
Timing uses an uninstrumented execution; Python-traced peak allocation uses a separate replay.

| 5,000-submodel independent case | Pinned legacy | Refactored |
| --- | ---: | ---: |
| Wall time | 0.447 s | 0.797 s |
| Python-traced peak allocation | 18,587,484 bytes | 2,215,297 bytes |

This is an approximately **8.4-fold reduction in traced peak memory**, with approximately **1.8-fold higher runtime** in this local measurement.
Summed times over the 25 cases were 0.717 s and 1.270 s, respectively.
Explicit validation and collection of the owned regression statistics add work; this refactor does not establish a runtime speedup.
These single local timing observations are not a cross-platform benchmark, and traced allocations are not whole-process resident memory.
A future specialized backend should be justified by profiling and required to match the same statistical contract.

The 5,000-submodel example recovers six of ten signals with all-model retention and seven with top-s retention; both agree exactly with the baseline.
Top-s also selects one noise variable in this dataset.
This is intentionally retained evidence of method behavior, not a reason to change the fixtures or to claim all signals should be recovered.
The full per-scenario metrics and individual selections are in `output/refactor-audit/final-classic/`.

## Reproduce and review

```bash
python -m pip install -r tests/requirements.txt
python -m pytest -q

# Optional actual MPI and pinned R checks:
python -m pip install -r tests/requirements-mpi.txt
PATH="$PWD/.venv/bin:$PATH" SUBSAMP_TEST_MPI=1 \
  SUBSAMP_CRAN_SOURCE=/path/to/pinned/subsamp.R \
  .venv/bin/python -m pytest -q

# Paired audit; use a new destination for a changed revision/environment:
PYTHONPATH=src:. python -m benchmarks.run_classic \
  --profile fixed --output output/new-paired-audit
```

See [benchmark instructions](../benchmarks/README.md) for per-scenario definitions, failure accounting, restart constraints, and the explicit larger-profile command.
The larger 1,000-dataset-per-scenario evaluation is prepared but has not been run at scale.
No new scoring rule has been statistically validated or recommended.
The direct published-PDF/extended-handoff contents were unavailable here; the supplied review's equation numbering and classic settings are attributed to that review, while locally rerun source comparisons remain separately identified.
The earlier [reference specification](REFERENCE_BEHAVIOR.md), [stage 1 handoff](REFERENCE_HANDOFF.md), and proposal PDF remain historical evidence rather than current implementation status.
