# Reference behavior: stage 1

Historical stage 1 record: for the subsequent implementation and current validation status, see [the refactor audit](REFACTOR_AUDIT.md).

This is a characterization of SubsampWinner at `e0366d3bcf917b875b982214a2ef342f8d463e42`, verified on October 17, 2026.
It separates source evidence, valid numerical behavior, implementation adaptations, and defects.
It does not approve a redesign or establish statistical guarantees.
The implementation and packaging remain unchanged in this stage.

## Sources and verification boundary

| Source | What was checked | Status |
| --- | --- | --- |
| Fan and Sun (2021), *Subsampling from features in large regression to find “winning features”*, Statistical Analysis and Data Mining 14(2), 168–184, [DOI](https://doi.org/10.1002/sam.11499) | Publisher bibliographic record and abstract | Verified; the final article's full text and equation could not be retrieved from Wiley. |
| Fan and Sun, [arXiv:2002.02903v1](https://arxiv.org/abs/2002.02903v1), [PDF](https://arxiv.org/pdf/2002.02903), February 2020 | Algorithm 1, equation (3), Algorithm 3; equation visually inspected on PDF page 4 | Primary methodological evidence available in full; not assumed identical to the final journal article. |
| CRAN [`subsamp` 0.1.0](https://cran.r-project.org/src/contrib/Archive/subsamp/subsamp_0.1.0.tar.gz), November 2017 | `DESCRIPTION`, `R/subsamp.R`, and `man/subsamp.Rd`; executed the scoring function and full selection function on saved subsets | Primary executable reference, predating the journal publication. |
| Python package in this repository | All four source modules, README, setup metadata, serial probes, deterministic tests | Direct local characterization of version 0.0.8's source. |

The preprint retains the `s` submodels with smallest RSS and uses `|t|/sqrt(RSS)`, averaged over nonzero t statistics.
It fits the semifinalists jointly, applies a multiplicity adjustment, and allows optional stepwise selection.
Its double-assurance procedure combines significant features from selected runs, then performs another base analysis and selection on that combined set.
These statements refer to the preprint's Algorithms 1 and 3, not to an unverified final-journal equation.

The relevant CRAN package is **`subsamp` for regression**, rather than the separately named `swa` classification package.
The [CRAN package page](https://cran.r-project.org/package=subsamp) reports archival on April 20, 2024.
The downloaded archive has SHA256 `cd02097e9f4d2e936139005208857b9d6ffebc562163d4e2c88a078fc517dfad`.
Its `R/subsamp.R` has SHA256 `7d5fcff9d3b5c2d7255646c7e70dc8afa36e5b31c1e0d67df6324f31977537e9`.
An inspectable [CRAN Git mirror at commit e56d150](https://github.com/cran/subsamp/blob/e56d150a9cd566067dec47197546f70548d906ac/R/subsamp.R) provides line references below.
The GPL-licensed R source was downloaded into temporary storage for reference testing; it was not incorporated into this MIT package.

## Score: square root, counts, and retained models

For retained submodel indices `K`, the Python calculation is

\[
S_j=\sum_{i\in K}1\{t_{ij}\ne0\},\qquad
w_j=\begin{cases}
S_j^{-1}\sum_{i\in K}|t_{ij}|/\sqrt{RSS_i},&S_j>0,\\
0,&S_j=0.
\end{cases}
\]

Unselected features receive zero t values.
`S_j` counts **nonzero t statistics**, not literally every inclusion in a submodel.
An included feature with an exactly zero statistic does not increment this denominator.
The preprint and R implementation use the same nonzero-statistic convention; replacing it by a literal inclusion count would be a methodological change in that edge case.

The Python score docstring says `1/RSS`, but the executable calculation at `subsamp.py:198` uses `1/sqrt(RSS)`.
The latter agrees with the visually inspected equation (3).
The documentation error is recorded here and left untouched in the source baseline.

The distinction can matter: with one contributing statistic per feature, `|t|=(1,3)` and `RSS=(1,4)` give scores `(1,1.5)` using the square root, versus `(1,0.75)` using RSS directly.
This algebraic example reverses the ranking; it is not an empirical estimate of how often rankings change.

| Choice | Python behavior | CRAN R behavior |
| --- | --- | --- |
| Retention | Default `use_top_s=False` retains all `m` models; `True` retains the smallest-RSS `min(s,m)` models. | Always retains the best `s` models; no all-model option. |
| Denominator within each contribution | Square root of raw residual sum of squares. | `summary(lm)$sigma`, despite the variable name `rss.new` (line 96). |
| Feature count in average | Nonzero t statistics in retained models. | Same convention (lines 107–108). |
| Semifinalists | Top `qnum` scores; `qnum=None` means `s`. | Top `qnum` scores; `qnum` is required. |
| Final selection | Bonferroni by default; BH or unadjusted filtering available. | `summary(step(lm.final, trace=FALSE))` (line 160), using R's default AIC stepwise search. |

For full-rank OLS submodels with an intercept and a common number of observations, residual degrees of freedom are `d=n-s-1`.
R's `sigma=sqrt(RSS/d)` implies `w_R=sqrt(d)*w_Python` when Python uses `use_top_s=True` and the same subsets.
Thus submodel ordering and feature ranking agree under those conditions, while score magnitudes differ.
That multiplier changes with `s`; absolute scores in diagnostic plots across different `s` values are not identical.
Rank deficiency or different missing-data handling would invalidate the common-df argument.

On the saved `n=30, p=6, s=3, m=12` example, the actual R scorer and Python top-`s` scores agree to a maximum absolute difference of `4.44e-16` after division by `sqrt(26)`.
Both choose semifinalists `[0,1,5]` in zero-based Python indexing.
R's stepwise model retains all three (`X1`, `X2`, `X6`), whereas Python's Bonferroni filter returns `[0,1]`.
This is evidence of agreement in scoring under stated assumptions, not whole-algorithm equivalence.

## Python model and sampling behavior

- Each repetition samples `s` distinct feature columns without replacement and uses every observation.
  Repetitions are separate draws and may repeat an entire subset.
- Ordinary serial usage with this baseline still imports `mpi4py` and MPI at package import time.
  The reference harness explicitly disables MPI initialization before importing the package; this is a harness setting, not a production fix.
- Without `weights`, the base model is statsmodels OLS with its conventional covariance estimator.
  With `weights`, it is statsmodels WLS; weights are passed through as supplied and are not estimated.
- For classical t-test interpretation in a fixed, full-rank model, OLS requires the usual mean specification and independent homoscedastic normal errors; WLS additionally requires weights proportional to inverse error variances for the corresponding conventional inference.
  This package does not provide heteroskedasticity-robust covariance or a GLS fit for correlated errors.
- In both OLS and WLS scoring, `RSS` is `sum(model.resid**2)`, the **unweighted** residual sum of squares.
  WLS t statistics still use statsmodels' weighted fit and variance estimate.
  This hybrid WLS scoring rule is an implementation adaptation, not a source-verified generalization of the paper.
- `sm.add_constant` normally adds an intercept, which is excluded from scores and multiplicity adjustment.
  The current use of its default constant detection is unsafe for supplied constant predictors.
- Semifinalists are stored in descending score order using NumPy's default `argsort` behavior.
  Ties have no documented scientific tie-breaking rule; zero-score features can enter if `qnum` is large enough.
  Reference selections avoid ties at the selection boundary.

The reference examples use finite numeric arrays, `X` of shape `(n,p)`, one-dimensional `y` of length `n`, and positive finite one-dimensional WLS weights.
Every fitted design, including its intercept, is full rank, has positive RSS, and has positive residual degrees of freedom.
With this intercept convention, `s<=n-2` and `qnum<=n-2` are sufficient dimension bounds for positive degrees of freedom when designs are full rank; each is also bounded by `p`.
These describe the tested domain, not validation already enforced by the package.

## Final filtering and result meanings

`select_finalists` fits **all semifinalists together**, then tests their non-intercept coefficients.
`bonferroni` and `fdr_bh` call statsmodels `multipletests` on the `qnum` feature p values.
For an explicit `method="none"`, the current fallback uses raw `p < alpha`.
The adjusted paths use the library's rejection rule, including its equality boundary.
Any other string currently takes the same unadjusted fallback; this is a defect, not an accepted alias policy.

`feature_weights` has length `p`; `semifinalists` is an index array; `finalists` is a list of original zero-based indices; `p_adjusted` aligns with semifinalist order.
`final_model` remains the joint **semifinalist** fit even when the finalist list is smaller or empty.
Its feature names are `v{index+1}`, while returned feature indices remain zero-based.
`fit()` mutates these attributes and currently returns `None` through `logging.info`; its claimed finalist return value is incorrect documentation.
`select_finalists()` returns `(finalists, p_adjusted, final_model)`.
No refit on the finalist set is implicit.

These are ordinary model p values computed after selection on the same data.
Small numerical checks do not establish post-selection validity, general FDR control, or robustness to misspecified weights.

## Random state and parallelism

`fit(seed=k)` resets NumPy's global legacy random state before sampling; repeat fits with that seed reproduce scores in the checked environment and alter the caller's global state.
With `seed=None`, the current global NumPy state is consumed.
Calling `subsample_anlysis` directly does not itself apply `self.seed`.
The data generator also seeds and uses the global state, including its default `np.random.normal` distribution.
The valid numerical fixture stores both the data and the full subset schedule so a later local-generator change can be evaluated independently of historical seed sequences.
No random generator has been changed here.

Source inspection of the MPI path shows that ranks divide `m`, call the same sampling function, and gather results only to root.
If every rank calls `fit` with the same fixed seed, their sampling prefixes repeat; non-root ranks return their local results and continue through selection without receiving the combined result.
These are source-based findings only: no MPI communicator was initialized and no multi-process execution was validated in this stage.

## Adaptive double assurance: current adaptation

For iteration `i=0,...,I_max-1`, set `a_i=init_range*r**i`, `s1=floor((1-a_i)*s0)`, and `s2=ceil((1+a_i)*s0)`.
If `s1==s2`, the loop stops before evaluating that triplet.
Otherwise it runs SWA at the center, lower, and upper values, in that order, reusing the same supplied seed for each fit and using the default all-model aggregation.
There is no bound clipping and no deduplication of candidate sizes.

For finalist sets `F0,F1,F2`, define `U=F0∪F1∪F2` and `C=F0∩F1∩F2`.
The iteration score is `|C|/|U|` when `U` is nonempty.
The best center updates only on a strict improvement over the best score, initialized at zero.
The loop stops when the score reaches `T`, when the range collapses, or after `I_max` iterations.
Otherwise the next center maximizes `|Fk|/|U|`, which selects the **largest finalist set**, with ties favoring center, then lower, then upper.
This rule has not been replaced by a consensus-based stability criterion.

With `return_objects=True`, the result contains `best_s0`, `final_range`, and `stabilized_union_set`.
The last field is the **cumulative union across every evaluated iteration**, not the intersection, not the set at the best center, and not a stable consensus.
There is no final union refit or inferential filter.
`final_range` can mix the last evaluated bounds with the newly updated center when the iteration limit ends; that reporting issue remains unresolved.
With `return_objects=False`, the procedure prints its summary and returns `None`; even `True` can print a stopping message.
The archived R documentation describes combining selected sets and applying another base analysis, rather than this adaptive loop.

## Defects and unresolved choices

Observed failures are saved in [`legacy_probes.json`](../tests/reference/legacy_probes.json) with source hashes and environment versions.
They are not golden expected outputs for future repairs.

| Probe | Observation at the baseline | Status |
| --- | --- | --- |
| Constant predictor, `s=4` | t-value assignment shape mismatch after intercept detection | Reproduced defect |
| `qnum=0` | Selects every feature via the `[-0:]` slice | Reproduced defect |
| Misspelled `bonferoni` | Silently uses unadjusted p values | Reproduced defect |
| All three assurance finalist sets empty | `ZeroDivisionError` | Reproduced defect; empty-set convention still needs documentation in a repair |
| Initial assurance size `s0=p` | Upper candidate exceeds `p` and sampling fails | Reproduced defect |
| `I_max=0` | Unbound `s1` in returned range | Reproduced defect |
| Duplicate predictor columns | Fits with reduced rank and can select both duplicate columns | Reproduced; no explicit identifiability policy |
| Saturated final model | Zero residual degrees of freedom without a useful validation error | Reproduced; unsuitable for the intended tests |
| Unstructured generator, `n=30,p=5,seed=7` | Covariance minimum eigenvalue `-10.375342260280775` and a non-PSD warning | Reproduced invalid covariance construction |
| Seeded fit | Mutates global NumPy random state | Reproduced side effect |

Other source-level gaps include missing shape/range/finite-value checks, no explicit zero-RSS handling, and contradictory Python support metadata (`>=3.7` versus dependency minimums requiring Python 3.9 or later).
This stage does not set a new constant/singular-predictor policy, empty-selection policy, WLS scoring rule, assurance criterion, or final-model inferential target.
It establishes evidence needed to review such changes separately.
