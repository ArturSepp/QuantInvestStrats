---
myst:
  html_meta:
    description: >-
      Supplied-covariance Gaussian Monte Carlo VaR in qis, with benchmark VaR ratios,
      ex-ante beta, tracking error, horizon scaling and offline monitoring examples.
---

# Gaussian Monte Carlo VaR and benchmark risk

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Value at risk (VaR) is a quantile of a portfolio's loss over a specified horizon.
This chapter describes the supplied-covariance Gaussian calculation in
[qis](https://github.com/ArturSepp/QuantInvestStrats), including comparison with a
benchmark and a closed-form check. Software citation:
[CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

## Overview

Use `qis.compute_gaussian_portfolio_risk` when an annualised asset covariance is
already available, for example from a factor model or an EWMA estimator. A single
call reports portfolio and benchmark Monte Carlo VaR, their ratio, ex-ante beta,
tracking error, volatilities and a closed-form portfolio VaR check.

Typical uses are comparing strategic and tactical allocation proposals under the
same risk model, monitoring each portfolio against its benchmark, and building
historical risk snapshots from dated weights and covariances. The function consumes
those inputs; it does not estimate covariance or optimise allocations.

The existing price-based functions, such as
`qis.compute_portfolio_correlated_var_by_groups` and
`PortfolioData.compute_portfolio_vars`, retain their own EWM estimation and
one-period 99% normal VaR conventions. They are described in
[portfolio risk contributions](risk_contributions.md#parametric-value-at-risk).

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Zero-drift linear fractional P&L. The example covariance uses log returns; no exponential conversion or compounding is applied to the resulting loss. |
| Sampling grid | Monthly log returns and monthly covariance snapshots in the example; supplied covariance can come from another stated grid. |
| Annualisation | Annualised covariance, with monthly factor $\mathrm{af}=12$ in the example. Horizon $\tau$ is in years. The risk function performs no covariance annualisation. |
| Mean adjustment | Simulation drift is zero. The example separately uses point-in-time EWM demeaning when estimating covariance. |
| Timing | Weights and covariance known at date $t$ describe prospective risk from $t$. The caller selects the date and applies any implementation lag. |
| Output units | VaR is a positive loss fraction of current NAV; volatility and tracking error are annualised fractions; beta, VaR ratio and relative MC error are dimensionless. |
| qis default | `GaussianVarSpec(confidence=0.99, horizon_years=1.0, simulations=500_000, random_seed=0)`. |

Let $p_t$ and $b_t$ denote the portfolio and benchmark exposure vectors, and let
$\Sigma_t$ be their common annualised asset covariance matrix. Exposures are
fractions of current NAV: 0.10 means 10%. They are not normalised internally;
signed exposures and leverage remain as supplied. All assets must use the same
valuation currency and consistent FX-hedging and return conventions.

Locally, $c$ is the loss confidence level, $z_c$ the standard-normal $c$-quantile,
$v_p$ and $v_b$ the portfolio and benchmark variances, $k$ their cross-covariance,
and $M$ the number of Monte Carlo scenarios. The horizon is $\tau>0$ years.

The covariance must have matching, unique row and column labels and be symmetric,
finite on the held universe and positive semidefinite within the implementation's
numerical tolerances. Weight labels must be unique and values finite. Omitted
weights become zero. A nonzero exposure outside the covariance universe raises
`ValueError`. Missing risk for an asset held in either book also raises; unavailable
rows for assets unheld in both books can be zeroed without changing either risk.

## Methodology

### Gaussian loss quantiles

**Definition.** The model freezes the supplied fractional exposures and takes the
horizon asset shock to be $X_{t,\tau}\sim\mathcal{N}(0,\tau\Sigma_t)$.
Portfolio loss is $L_p=-p_t^\top X_{t,\tau}$, with the same construction for
the benchmark. This is a linear loss approximation, including when the covariance
was estimated from log returns. Analytical and Monte Carlo VaR are standard
approaches to this model; see Jorion in the [references](#references).

**Proposition.** Define the annualised joint moments by

$$
v_p=p_t^\top\Sigma_t p_t,\qquad
v_b=b_t^\top\Sigma_t b_t,\qquad
k=p_t^\top\Sigma_t b_t.
$$

Then the closed-form portfolio VaR is

$$
\operatorname{VaR}_{c,\tau}(p_t)=z_c\sqrt{\tau v_p}.
$$

**Proof.** A linear projection of a zero-mean Gaussian is Gaussian, with
variance $\tau v_p$. Its loss quantile is its standard deviation times $z_c$.
The same argument applies to the benchmark. $\square$

`GaussianMonteCarlo` draws two standard normals for each half-sample and appends
their negatives, giving $M$ antithetic scenarios. When $v_b>0$, it constructs
jointly Gaussian benchmark and portfolio shocks as

$$
R_b=\sqrt{v_b}\,Z_1,\qquad
R_p=\frac{k}{\sqrt{v_b}}Z_1+
\sqrt{v_p-\frac{k^2}{v_b}}\,Z_2.
$$

The reported VaRs are empirical $c$-quantiles of the negative shocks multiplied
by $\sqrt{\tau}$. This two-dimensional projection has the same joint distribution
as simulating every asset and applying both sets of linear weights. It avoids
allocating a scenario matrix with one column per asset. When benchmark variance
is zero, portfolio shocks use the second draw directly and benchmark VaR is zero.

### Ratio, beta and tracking error

**Definition.** With nonzero benchmark risk, the reported ratio and beta are

$$
\mathrm{VaRRatio}=\frac{\widehat{\operatorname{VaR}}_{c,\tau}(p_t)}
{\widehat{\operatorname{VaR}}_{c,\tau}(b_t)},\qquad
\beta=\frac{k}{v_b}.
$$

The hat denotes an empirical Monte Carlo quantile. Under the closed-form Gaussian
model, the VaR ratio is $\sqrt{v_p/v_b}$; finite simulation introduces sampling
error. Beta instead measures covariance with the benchmark relative to benchmark
variance, so it need not equal the VaR ratio. Both are `NaN` if benchmark risk is zero.

**Identity.** Annualised tracking error satisfies

$$
\mathrm{TE}=\sqrt{(p_t-b_t)^\top\Sigma_t(p_t-b_t)}
=\sqrt{v_p+v_b-2k}.
$$

**Proof.** Expanding the quadratic form and using covariance symmetry gives the
second expression. $\square$

QIS delegates beta, volatility and tracking error to `qis.RiskModel`. Tracking
error is a standard deviation of active returns; it is not a VaR ratio or a
quantile of active losses. Changing `horizon_years` scales VaR by the square root
of the horizon, while beta, tracking error and the reported volatilities retain
the supplied covariance units.

> **Insight.** Reuse one `GaussianMonteCarlo` instance for comparisons across
> allocations and dates. It preserves common random numbers and gives exactly
> unit VaR ratio and zero tracking error for an identical portfolio and benchmark.
> It does not simulate a joint path through those reporting dates.

## Worked example

The canonical offline example uses the frozen synthetic multi-asset universe,
monthly log returns and annualised 36-month EWMA covariances. Estimation starts
in January 2014; reporting starts on 31 December 2022 after roughly three spans
of burn-in. `warmup_period=36` additionally masks an asset until it has more than
36 returns. The sample ends on 31 December 2025. No market data are downloaded.

### Compare a strategic allocation and an active proposal

```python
from examples.portfolios.gaussian_var_monitoring import compute_monitoring_tables

latest, history = compute_monitoring_tables()
print(latest[['portfolio_var', 'benchmark_var', 'var_ratio',
              'ex_ante_beta', 'tracking_error']].round(4))

assert set(latest.index) == {'Strategic', 'Active'}
assert len(history) == 74  # 37 month ends, two allocations
assert history.index.get_level_values('date').max().strftime('%Y-%m-%d') == '2025-12-31'
assert latest['mc_relative_error'].abs().max() < 0.015
assert round(100.0 * latest.loc['Strategic', 'portfolio_var'], 2) == 17.81
assert round(100.0 * latest.loc['Active', 'portfolio_var'], 2) == 19.95
assert round(latest.loc['Strategic', 'var_ratio'], 3) == 0.906
assert round(latest.loc['Active', 'var_ratio'], 3) == 1.015
assert latest['benchmark_var'].mul(100.0).round(2).eq(19.66).all()
assert latest['ex_ante_beta'].round(3).to_dict() == {'Active': 0.997, 'Strategic': 0.897}
assert latest['tracking_error'].mul(100.0).round(3).to_dict() == {
    'Active': 1.572, 'Strategic': 1.405
}
```

The strategic weights are 45% US equity, 10% European equity, 30% Treasury bonds,
10% investment-grade bonds and 5% gold. The active proposal has 50%, 10%, 20%,
10% and 10% respectively. Their benchmark has 40%, 20%, 25%, 10% and 5%.
The output compares both allocations using the same dated covariance and draws.
These are illustrative fixed allocations, not optimised strategy backtests.

At 31 December 2025, the computed comparison is:

| Allocation | One-year 99% VaR | Benchmark VaR | VaR ratio | Ex-ante beta | Annualised TE |
|---|---|---|---|---|---|
| Strategic | 17.81% | 19.66% | 0.906 | 0.897 | 1.405% |
| Active | 19.95% | 19.66% | 1.015 | 0.997 | 1.572% |

The strategic allocation has lower modelled total risk than the benchmark. The
active proposal has almost unit beta but slightly higher total risk, illustrating
why beta and the VaR ratio answer different questions. These synthetic results
come from the same call as the displayed table; they are not investment forecasts.

### Read a historical risk series

```python
active_history = history.xs('Active', level='allocation')
print(active_history[['portfolio_var', 'var_ratio', 'tracking_error']].tail(3))
assert active_history.index.is_monotonic_increasing
assert len(active_history) == 37
```

Each row estimates prospective risk at that month end using information through
that date. The allocations in this teaching example remain fixed, so the history
isolates changes in estimated covariance. For a production history, supply the
actual dated allocations and benchmark weights known at each date; do not apply
today's optimised weights to the entire past and label it a strategy risk history.

### Change the horizon, keeping risk inputs fixed

```python
from dataclasses import replace
import numpy as np
import qis

spec = qis.GaussianVarSpec()
annual_simulation = qis.GaussianMonteCarlo(spec)
monthly_simulation = qis.GaussianMonteCarlo(replace(spec, horizon_years=1.0 / 12.0))
annual_var = annual_simulation.var(0.04, 0.01, 0.005)
monthly_var = monthly_simulation.var(0.04, 0.01, 0.005)
np.testing.assert_allclose(monthly_var, np.asarray(annual_var) / np.sqrt(12.0))
assert abs(annual_var[0] / (2.3263478740408408 * 0.20) - 1.0) < 0.015
```

Here the annualised portfolio variance is 0.04, benchmark variance 0.01 and
cross-covariance 0.005. The closed-form one-year 99% portfolio VaR is 46.53% of
NAV. A one-month horizon gives 13.43%, under the same frozen Gaussian covariance.
This changes the risk horizon; it does not refit covariance on monthly rather
than daily observations.

## Implementation in qis

The public entry points are:

| Entry point | Contract |
|---|---|
| `qis.GaussianVarSpec` | Immutable settings. Confidence must be strictly between 0.5 and 1, horizon finite and positive, and simulations an even integer of at least 1,000. A local seeded generator leaves the global NumPy RNG untouched. |
| `qis.GaussianMonteCarlo(spec)` | Reusable antithetic draws. Its `var(portfolio_variance, benchmark_variance, cross_covariance)` method returns the two loss-positive VaRs and rejects invalid joint moments. |
| `qis.compute_gaussian_portfolio_risk(covariance, portfolio_weights, benchmark_weights, date, simulation)` | Validates and aligns labelled inputs, then returns a dictionary of nine scalar metrics. `date` labels the supplied matrix; the function does no as-of selection or freshness check. |

The result keys are:

| Key | Meaning |
|---|---|
| `portfolio_var`, `benchmark_var` | Monte Carlo loss quantiles over the configured horizon, as fractions of NAV. |
| `var_ratio` | Portfolio VaR divided by benchmark VaR; `NaN` for zero benchmark VaR. |
| `ex_ante_beta` | Covariance-implied portfolio beta to the benchmark; `NaN` for zero benchmark variance. |
| `tracking_error` | Volatility of the active exposures under the supplied covariance. |
| `portfolio_volatility`, `benchmark_volatility` | Standard deviations in the supplied covariance units. |
| `analytical_var` | Closed-form portfolio Gaussian VaR at the same confidence and horizon. |
| `mc_relative_error` | `portfolio_var / analytical_var - 1`; zero by convention when analytical VaR is zero. This is a simulation diagnostic, not a model-validation error. |

For an external covariance, replace the estimator in the example and call the
same function with a labelled `pandas.DataFrame` and exposure `pandas.Series`.
Supply fractional units: a 20% annual volatility has variance 0.04, not 400.
If covariance is still monthly, annualise it consistently before using a horizon
in years. Multiply fractional VaR by current NAV to report a monetary loss.

Run the complete example from the repository root:

```console
python -m examples.portfolios.gaussian_var_monitoring
```

Canonical sources:
[offline example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/gaussian_var_monitoring.py),
[VaR implementation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/gaussian_var.py),
[independent numerical checks](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/tests/gaussian_var_test.py),
and the generated [API reference](api/index.rst). Use a QIS source or installed
build containing these three exports; the addition is recorded under
[Unreleased in the changelog](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CHANGELOG.md).
The example requires the core installation only.

## Interpretation and limitations

- **Gaussian linear risk.** Simulation does not add fat tails, nonlinear option
  repricing, liquidity losses, default jumps or uncertainty in estimated covariance.
  The analytical answer is available, so compare `mc_relative_error` before
  interpreting small changes between allocations.
- **Horizon scaling.** Square-root scaling assumes covariance accumulates linearly
  with time and exposures remain fixed. One-year VaR is a prospective frozen-book
  approximation, not a simulated year of rebalancing, fees or compounded wealth.
- **Return convention.** A covariance estimated from log returns still feeds the
  linear P&L approximation. A VaR above 100% is possible under Gaussian or levered
  assumptions and is not clipped to a funded portfolio's NAV.
- **Reported or stale prices.** The function does not unsmooth private-asset
  returns, nowcast missing prices or detect stale marks. Covariance quality and
  data coverage remain the caller's responsibility.
- **Forecast versus realised.** Ex-ante beta and TE use current exposures and
  covariance. To monitor realised risk alongside them, use
  `qis.compute_ewm_beta_alpha_forecast` on aligned realised returns and
  `qis.compute_ewma_realised_tracking_error` on portfolio and benchmark NAVs.
  An EWMA span of 12 monthly observations is a decay setting, not a hard trailing
  12-month cutoff. See [benchmark-relative risk](tracking_error_and_risk.md).

> **Pitfall.** A VaR ratio near one does not imply a small tracking error. Two
> portfolios can have similar total volatility but different benchmark exposures.
> Report ratio, beta and tracking error together, with confidence and horizon.

## See also

- [Portfolio risk monitoring guide](risk_monitoring.md)
- [Portfolio risk and Euler contributions](risk_contributions.md)
- [Tracking error and benchmark-relative risk](tracking_error_and_risk.md)
- [Covariance, correlation and principal components](covariance_correlation_pca.md)
- [Factor risk models](factor_risk_models.md)
- [Factor stress testing](stress_testing.md)

## References

1. Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*, 3rd edition. McGraw-Hill. Chapters 7 and 12 cover analytical portfolio risk and Monte Carlo methods; see the [publisher's edition record](https://www.mheducation.com/highered/mhp/product/value-risk-3rd-ed.html).
2. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
