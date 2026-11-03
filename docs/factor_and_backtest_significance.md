---
myst:
  html_meta:
    description: >-
      Source-only qis significance tests for supplied return means, regression alphas and
      paired Sharpe differences, with Holm/BHY corrections, stationary-bootstrap p-values
      and explicit finite-sample validation limits.
---

# Significance tests for factors and backtests

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-11-03](https://github.com/ArturSepp/QuantInvestStrats/commit/e4c3383d91a24c49d1b78461ee7d60e40127b6e8)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

A significance test measures evidence against a specified null hypothesis, conditional on its
sample, model and assumptions. A family correction addresses testing several supplied hypotheses;
it does not reconstruct an unrecorded search for successful strategies.

> **Pitfall.** The family interfaces in this chapter are **source-only development interfaces**,
> not top-level exports in `qis.__all__` or a verified released API. The finite-sample qualification
> stage remains incomplete. Correctness/reference checks do not establish nominal size or
> coverage. A separate prewhitened V3-A prototype remains private and experimental and is not
> selected by the functions demonstrated here. No legacy default or validation gate is changed.

## Overview

Three questions have different estimands:

| Question | Null hypothesis | Calculation |
|---|---|---|
| Does each supplied return series have a nonzero mean? | Mean equals zero | Equal-weight mean with Bartlett HAC inference |
| Does each supplied strategy have alpha against supplied factors? | Regression intercept equals zero | Common-sample OLS with an intercept and Bartlett HAC inference |
| Does each candidate differ from one fixed benchmark? | Arithmetic Sharpe difference equals zero | Existing paired Sharpe-difference test, followed by family correction |

Heteroskedasticity- and autocorrelation-consistent (HAC) standard errors address dependence
under their asymptotic assumptions. Holm controls family-wise error rate (FWER); the
Benjamini–Yekutieli procedure, called `BHY` in qis, controls false discovery rate (FDR) under
arbitrary dependence when the marginal p-values are valid. These are different error targets,
not interchangeable names for the same test.

The optional stationary bootstrap changes the marginal p-values of the mean and alpha tests.
It does not implement bootstrap Sharpe inference, simultaneous intervals, a maximum-statistic
test, or a rerun of strategy selection. Factor-return means and time-series alphas are not
cross-sectional prices of factor risk.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Supplied fractional simple returns; the caller has already applied reference-currency, cash/funding and cost treatment consistently |
| Sampling grid | One complete common `DatetimeIndex`; declare `sampling_frequency` or an exact `expected_index`; no implicit resampling or pair-specific deletion |
| Annualisation | `af=None` retains per-period units; explicit `af` scales mean/alpha estimates, SEs and bounds linearly, and paired Sharpe quantities by its square root |
| Mean adjustment | Equal-weight estimation with an intercept; mean bootstrap subtracts sample means, alpha bootstrap subtracts fitted intercepts only |
| Timing | Full-sample descriptive inference on supplied realised returns; decision timing and model selection must already be free of look-ahead |
| Output units | Fractional return per period for mean/alpha, dimensionless Sharpe-difference units for the paired test; p-values and statistics are dimensionless |
| qis default | Family method `HOLM`, significance level .05, pointwise confidence level .95, HAC lags 3, no annual display; generated bootstrap B=9,999, expected block length 6, seed 20261101 |

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $T$, $M$, $k$ | Common observations, supplied hypotheses, design columns including the intercept | Counts; $T\geq24$ and $T>k$ |
| $y_{j,t}$, $f_t$ | Response $j$ and supplied factor vector | Fractional simple returns on identical dates |
| $\theta_j$ | Mean, alpha or candidate-minus-benchmark arithmetic Sharpe difference | The chosen test's estimand |
| $L$, $B$ | Bartlett lag count and bootstrap draws | Observation counts, not calendar durations |
| $q$, $c$ | Requested error-rate target and pointwise confidence level | `significance_level` and `confidence_level`, separately chosen |
| $N_{\mathrm{false}}$, $N_{\mathrm{rej}}$ | False rejections and all rejections in the declared family | Counts in one experiment |

Supply real numeric DataFrames with unique, nonblank hypothesis labels and finite values.
Responses, factors and the fixed benchmark must have exactly the same ordered dates. An attached
`index.freq` alone does not declare the sampling contract. With a regular frequency, missing
interior observations fail validation. A custom `expected_index` explicitly defines its sessions;
it is not inferred from the observed gaps.

Factors must be nonempty and full rank, without constants, duplicate columns, self-regressors
or a column named `Intercept`. Invalid family members reject the entire call. Zero or
scale-relative degenerate residuals/SEs are errors, not hypotheses silently dropped from $M$.
The paired Sharpe test retains its existing identical/positively proportional-series convention.

The descriptions `return_basis` and `cost_basis` are required provenance, not transformations.
For excess-return inference, prepare the returns before calling the tests; see
[reference-currency excess returns](reference_currency_excess_returns.md). The code does not
subtract cash, convert currencies, deduct costs or choose factors from these strings.

## Methodology

### Means and regression alphas

**Definition.** Each two-sided null is $H_{0,j}:\theta_j=0$. For alpha, fit the supplied model

$$
y_{j,t}=\alpha_j+\beta_j^\top f_t+\varepsilon_{j,t}.
$$

The mean test is the intercept-only case. Both delegate to the shared equal-weight regression
core. For design rows $x_t$, fitted residuals $\hat\varepsilon_{j,t}$ and scores
$g_{j,t}=x_t\hat\varepsilon_{j,t}$, the Bartlett meat is

$$
S_j=\sum_{t=1}^T g_{j,t}g_{j,t}^\top
+\sum_{\ell=1}^{L}\left(1-\frac{\ell}{L+1}\right)
\sum_{t=\ell+1}^{T}
\left(g_{j,t}g_{j,t-\ell}^\top+g_{j,t-\ell}g_{j,t}^\top\right).
$$

The covariance estimate is

$$
\widehat{\operatorname{Cov}}(\hat b_j)
=\frac{T}{T-k}(X^\top X)^{-1}S_j(X^\top X)^{-1}.
$$

The intercept's diagonal element gives its squared HAC standard error. The correction is
$T/(T-k)$, not an effective-sample-size replacement. See
[regression and HAC inference](regression_and_hac.md) for derivation and assumptions, and
[Newey–West (1987)](https://www.nber.org/papers/t0055) for the covariance estimator.

The statistic $z_j=\hat\theta_j/\widehat{\operatorname{se}}_j$ uses the standard normal
reference: $p_j=2\Phi(-|z_j|)$. The unadjusted pointwise interval is

$$
\hat\theta_j\;\pm\;z_{(1+c)/2}\widehat{\operatorname{se}}_j.
$$

This is not an exact finite-sample Student distribution. `hac_lags=None` selects the existing
`newey_west_lag_rule(T)` once; an explicit lag must satisfy $0\leq L<T$ and is not silently capped.

### Paired Sharpe differences

For each candidate, `compute_sharpe_significance` delegates to the existing
`qis.sharpe_difference_test` against the same fixed benchmark on the same complete sample.
The estimand is candidate minus benchmark arithmetic Sharpe, not the Sharpe ratio of their
return difference. The paired delta method retains population moments (`ddof=0`), the existing
uncorrected Bartlett covariance convention and its neutral result for identical or positively
proportional returns. It is not the mean/alpha covariance above with a new label.

Read the [paired Sharpe methodology](performance_analytics_and_sharpe.md#paired-hac-test-of-a-sharpe-difference)
for the shared-return covariance and sufficient-moment assumptions. A fixed-benchmark family
is not an automatic all-pairs search.

### Family error control

**Definition.** The two error targets are

$$
\mathrm{FWER}=\Pr(N_{\mathrm{false}}\geq1),
\qquad
\mathrm{FDR}=\mathbb{E}\!\left[
\frac{N_{\mathrm{false}}}{\max(N_{\mathrm{rej}},1)}\right].
$$

Declare the family before inspecting its results, including unsuccessful variants. For sorted
p-values $p_{(1)}\leq\cdots\leq p_{(M)}$, the adjusted values are

$$
\widetilde p^{\mathrm{Holm}}_{(i)}
=\min\!\left(1,\max_{j\leq i}(M-j+1)p_{(j)}\right),
$$

$$
\widetilde p^{\mathrm{BHY}}_{(i)}
=\min\!\left(1,\min_{j\geq i}\frac{M H_M}{j}p_{(j)}\right),
\qquad H_M=\sum_{j=1}^{M}\frac1j.
$$

Results return to the supplied label order, and `reject` means adjusted p-value $\leq q$.
`MultipleTestingMethod.HOLM` implements [Holm (1979)](https://www.jstor.org/stable/4615733);
`MultipleTestingMethod.BHY` implements the harmonic dependence correction of
[Benjamini–Yekutieli (2001)](https://doi.org/10.1214/aos/1013699998), not ordinary
Benjamini–Hochberg. Neither procedure repairs invalid marginal p-values or adjusts the displayed
pointwise intervals. Under an all-null family, FDR and FWER coincide; under a mixture they differ.

### Null-centred joint stationary bootstrap

For a mean test, construct $y^0_{j,t}=y_{j,t}-\bar y_j$. For alpha, construct
$y^0_{j,t}=y_{j,t}-\hat\alpha_j$ and leave factor means unchanged. Draw entire paired rows of
the responses and factors with one circular stationary-bootstrap index matrix. Refit the
intercept, all factor slopes and the HAC SE in every draw, keeping the originally selected lag.
Do not sort the sampled sequence by its repeated original dates.

For $K_j$ inclusive absolute-tail exceedances among $B$ studentised draws,

$$
K_j=\sum_{b=1}^{B}\mathbf{1}\{|z^*_{j,b}|\geq|z_j|\},
\qquad p^{\mathrm{boot}}_j=\frac{1+K_j}{B+1}.
$$

The finite-resample adjustment prevents a zero p-value. Holm/BHY correct these approximate
bootstrap p-values. The retained effect intervals remain **normal HAC pointwise intervals**;
no bootstrap effect CI is constructed. The separate 95% Clopper–Pearson bounds describe the
conditional tail probability underlying $K_j/B$, before the add-one adjustment, assuming
independent draws. They are not confidence intervals for a mean or alpha.

Generated draws use the existing [stationary bootstrap](reproducibility.md) of
[Politis–Romano (1994)](https://doi.org/10.1080/01621459.1994.10476870), with minimum block length
one and expected block length `block_size`. One means IID row sampling. Supplied integer indices
of shape `(T, B)` are authoritative, must lie in `[0, T-1]`, and carry their caller's sampling
law; the code does not infer a stationary-bootstrap origin from them.

The minimum p-value is $1/(B+1)$. The first-ranked threshold is $q/M$ for Holm and $q/(M H_M)$
for BHY. An unreachable threshold emits a warning; BHY can still reject at higher ranks.
An invalid draw aborts with its zero-based replicate/hypothesis context: no deletion, redraw,
clipping, fallback or replacement changes the denominator.

### Annual display is not a different test

**Proposition.** A common positive display scale applied to an estimate and its SE leaves its
statistic, p-values and family decisions unchanged.

**Proof.** The scale cancels in their ratio; the same p-value vector enters the correction.
For mean/alpha use $\mathrm{af}$, and for arithmetic Sharpe differences use
$\sqrt{\mathrm{af}}$. This is a display convention, not a new temporal aggregation or
independence assumption. $\square$

## Worked example

### A correction with hand-checkable values

These fixed p-values isolate the family arithmetic; they are not empirical return results.

```python
import numpy as np
import pandas as pd
from qis.perfstats.multiple_testing import MultipleTestingMethod, adjust_pvalues

p = pd.Series([0.01, 0.04, 0.03], index=["A", "B", "C"])
holm = adjust_pvalues(p, family_id="fixed_three_tests")
bhy = adjust_pvalues(p, family_id="fixed_three_tests", method=MultipleTestingMethod.BHY)
np.testing.assert_allclose(holm["pvalue_adjusted"], [0.03, 0.06, 0.06])
np.testing.assert_allclose(bhy["pvalue_adjusted"], [0.055, 11 / 150, 11 / 150])
assert holm["reject"].tolist() == [True, False, False]
assert not bhy["reject"].any()
```

| Hypothesis | Raw p-value | Holm adjusted | BHY adjusted |
|---|---:|---:|---:|
| A |.010|.030|.055|
| B |.040|.060|.073333…|
| C |.030|.060|.073333…|

At .05 Holm rejects A; BHY rejects none. This particular outcome does not make either error
target a substitute for the other.

### Mean, alpha and paired Sharpe families on the frozen synthetic panel

Use the first 72 monthly returns of the fixed synthetic market panel, after discarding only the
initial undefined price return. The two responses and one factor/benchmark are fixed in advance.
All inputs are gross fractional simple total returns, with cash assumed zero for this teaching
example; no costs or cash adjustment is performed by the tests. The three families answer
different questions and are not merged into one family after seeing their results.

```python
import qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.perfstats.significance import (
    compute_alpha_significance, compute_mean_significance, compute_sharpe_significance,
)

prices = generate_synthetic_prices()
panel = qis.to_returns(
    prices[["SEQ_US", "SBD_IG", "SBD_TSY"]], freq="ME", is_log_returns=False
).iloc[1:73]
responses = panel[["SEQ_US", "SBD_IG"]]
factors = panel[["SBD_TSY"]]
settings = dict(
    return_basis="simple total returns; zero cash assumed",
    cost_basis="gross; no fees or transaction costs",
    sampling_frequency="ME", hac_lags=3, af=12,
)
means = compute_mean_significance(responses, family_id="synthetic_means", **settings)
alphas = compute_alpha_significance(
    responses, factors, family_id="synthetic_alphas", **settings
)
pairs = compute_sharpe_significance(
    responses, panel["SBD_TSY"], family_id="synthetic_sharpe_pairs", **settings
)
assert means.metadata["nobs"] == alphas.metadata["nobs"] == pairs.metadata["nobs"] == 72
np.testing.assert_allclose(means.table["estimate"], 12 * responses.mean())
design = np.column_stack([np.ones(len(panel)), factors.to_numpy()])
coef = np.linalg.lstsq(design, responses.to_numpy(), rcond=None)[0]
np.testing.assert_allclose(alphas.table["estimate"], 12 * coef[0], rtol=1e-10)

# Independent dense Bartlett reference for the first mean's annual-display SE.
y = responses.iloc[:, 0].to_numpy()
e = y - y.mean()
distance = np.abs(np.subtract.outer(np.arange(len(y)), np.arange(len(y))))
kernel = np.maximum(1 - distance / 4, 0)
reference_se = np.sqrt((e @ kernel @ e) / (len(y) * (len(y) - 1)))
np.testing.assert_allclose(means.table["hac_se"].iloc[0], 12 * reference_se, rtol=1e-10)
assert means.table.attrs["confidence_interval"] == "unadjusted/pointwise"
```

Inspect each result's `table` together with `metadata`, not just `reject`. These are synthetic
teaching calculations, not evidence about real factors, investability or finite-sample calibration.

### A small bootstrap interface demonstration

The following uses only 39 draws to exercise the interfaces, **not** to support an empirical
significance claim or a calibration. It uses the existing default seed and block length.
The alpha call reuses the exact indices generated by the mean call, without a new sampler call.

```python
from qis.perfstats.bootstrap_significance import (
    compute_bootstrap_alpha_significance, compute_bootstrap_mean_significance,
)

boot_mean = compute_bootstrap_mean_significance(
    responses, family_id="synthetic_bootstrap_means", num_samples=39,
    block_size=6, seed=20261101, include_diagnostics=True, **settings
)
boot_alpha = compute_bootstrap_alpha_significance(
    responses, factors, family_id="synthetic_bootstrap_alphas",
    bootstrapped_indices=boot_mean.metadata["bootstrap_indices"], **settings
)
assert boot_mean.metadata["bootstrap_index_shape"] == (72, 39)
assert boot_mean.metadata["minimum_pvalue"] == 1 / 40
assert boot_mean.metadata["index_source"] == "generated"
assert boot_alpha.metadata["index_source"] == "supplied"
assert boot_alpha.metadata["seed"] is None
assert boot_alpha.metadata["indices_sha256"] == boot_mean.metadata["indices_sha256"]
np.testing.assert_allclose(
    boot_mean.table["pvalue_raw"], (1 + boot_mean.table["bootstrap_exceedances"]) / 40
)
np.testing.assert_allclose(boot_mean.table["pvalue_hac"], means.table["pvalue_raw"])
np.testing.assert_allclose(boot_mean.table["ci_lower_pointwise"], means.table["ci_lower_pointwise"])
np.testing.assert_allclose(boot_alpha.table["ci_upper_pointwise"], alphas.table["ci_upper_pointwise"])
```

The bootstrap p-value columns change their source; the normal HAC bounds do not become bootstrap
intervals. Supplied-index metadata does not assert the original generation law: retain the mean
call's generation settings as the provenance of this particular matrix.

## Implementation in qis

The names below exist in the development submodules, but are not new `qis.*` exports. An installed
release may not contain these modules; no released support floor is asserted here. Use a checkout
containing the linked source when reproducing the examples. Documenting the interfaces does not
promote them or complete their outstanding qualification.
The source-only modules and this new chapter must be committed together before publication;
their GitHub source links do not establish that an installed release already includes them.

| Submodule | Entry points | Contract |
|---|---|---|
| [`qis.perfstats.multiple_testing`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/multiple_testing.py) | `MultipleTestingMethod`, `adjust_pvalues` | One complete labelled p-value Series; no return estimation |
| [`qis.perfstats.significance`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/significance.py) | `SignificanceResult`, `compute_mean_significance`, `compute_alpha_significance`, `compute_sharpe_significance` | One complete common sample and an explicit family |
| [`qis.perfstats.bootstrap_significance`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/bootstrap_significance.py) | `compute_bootstrap_mean_significance`, `compute_bootstrap_alpha_significance` | Null-centred joint resampling; no bootstrap Sharpe or effect CI |

`SignificanceResult.table` has `estimate`, `hac_se`, `statistic`, `pvalue_raw`,
`pvalue_adjusted`, `reject`, `ci_lower_pointwise` and `ci_upper_pointwise`. Sharpe families add
`sharpe_candidate` and `sharpe_benchmark`. Bootstrap tables add `pvalue_hac`,
`bootstrap_exceedances` and `bootstrap_tail_mc_lower`/`bootstrap_tail_mc_upper`.

Metadata records the family ID/members/method, null/alternative, sample boundaries/count,
declared calendar, factor columns or fixed benchmark, lag choice, funding/cost descriptions and
display units. Bootstrap metadata also records effective $B$, the index source/hash, generated
block/seed settings where applicable, minimum p-value and first-rank reachability. With
`include_diagnostics=True` it retains copied `(T, B)` indices and `(B, M)` statistics, not $B$
full market panels. Without it the draw statistics are streamed into counts.

The canonical numerical owners are the shared
[`regression.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/utils/regression.py),
[`sharpe_inference.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/sharpe_inference.py)
and existing sampler. The examples above are executed offline by
[`documentation_examples_test.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/tests/documentation_examples_test.py):

```console
python -m pytest src/qis/tests/documentation_examples_test.py -k factor_and_backtest_significance
```

The frozen market generator and its seed are unchanged. No market-data download, Fama–French
study, calibration runner or private V3-A adapter is invoked by these examples.

## Interpretation and limitations

- Valid marginal inference is a prerequisite for family control. HAC needs stationary, weakly
  dependent scores, sufficient moments and an appropriate bandwidth; alpha needs an identified,
  exogenous full-rank factor design. Paired Sharpe inference needs sufficient fourth moments.
- The 24-observation check is an input-validity threshold, not a claim that 24 observations give
  reliable asymptotic inference. An automatic lag rule is not a finite-sample guarantee.
- The finite-sample validation programme did not meet all its predeclared marginal-size and
  pointwise-coverage gates. The implementation/reference checks and passing family-error checks
  do not override those failures. Qualification remains incomplete for the new family workflow.
- A private prewhitened prototype was investigated but not promoted. It remains experimental;
  the examples use the unchanged legacy Bartlett calculations. No calibration is resumed here.
- Corrected p-values are not posterior probabilities, and nonrejection is not proof of zero
  return or zero alpha. Statistical evidence does not establish economic materiality or net
  investability. State funding, fees, leverage and the tested effect units.
- A pointwise CI excluding zero need not agree with a Holm/BHY family decision. The intervals
  are not simultaneous or selection-adjusted, and tail MC bounds are not effect intervals.
- Freeze the tested family, sample, factors, benchmark, lags and resampling plan before looking
  at results. Testing only winners, changing families, repeated peeking, selecting block length
  by the smallest p-value or using revised data as point-in-time data invalidates the intended
  interpretation. These functions do not rerun or reconstruct the research search.
- No Fama–French empirical validation or production promotion is claimed by the synthetic
  examples. Published Sharpe/return conventions elsewhere in qis are not newly certified or
  changed by this development chapter.

## See also

- [Regression and HAC inference](regression_and_hac.md)
- [Sharpe ratios: conventions and paired inference](performance_analytics_and_sharpe.md)
- [Resampling and the bootstrap](reproducibility.md)
- [Returns, NAVs and excess returns](returns_and_navs.md)
- [Reference-currency conversion and excess returns](reference_currency_excess_returns.md)
- [Benchmark-relative performance](benchmark_relative_performance.md)
- [Bibliography](bibliography.md)

## References

1. Newey, W. K., and West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708. [Working paper and published-version record](https://www.nber.org/papers/t0055).
2. Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. [Publisher archive](https://www.jstor.org/stable/4615733).
3. Benjamini, Y., and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *The Annals of Statistics*, 29(4), 1165–1188. [DOI: 10.1214/aos/1013699998](https://doi.org/10.1214/aos/1013699998).
4. Politis, D. N., and Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313. [DOI: 10.1080/01621459.1994.10476870](https://doi.org/10.1080/01621459.1994.10476870).
5. Opdyke, J. D. (2007). Comparing Sharpe Ratios: So Where Are the p-Values? *Journal of Asset Management*, 8(5), 308–336. [DOI: 10.1057/palgrave.jam.2250084](https://doi.org/10.1057/palgrave.jam.2250084).
6. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
