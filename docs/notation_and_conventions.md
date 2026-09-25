---
myst:
  html_meta:
    description: >-
      The notation and calculation conventions used throughout the qis documentation: reserved
      symbols, simple and log returns, per-annum returns, annualisation, excess returns and timing.
---

# Notation and conventions

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

This chapter fixes the symbols and calculation conventions that every other chapter uses. A
reported statistic is defined only once its return basis, sampling grid, annualisation, mean
adjustment and timing are known. Each chapter therefore opens its inputs section with a
convention card of the same seven rows, and this chapter defines what those rows mean.

## Overview

The chapter answers three questions a reader needs settled before any formula:

1. **What does a symbol mean?** A reserved set of symbols has one meaning in the whole book.
   A chapter may introduce further symbols of its own, and declares them in its notation table.
2. **Which return convention applies?** Simple returns compound through time and aggregate
   across assets; log returns add through time but not across assets.
3. **How are periodic quantities annualised, and when is information available?**

Prose uses British spelling (annualisation, normalised). Python names keep the American
spelling in which they were published, for example `qis.get_annualization_factor` and the
argument `annualize_less_1y`; the two refer to the same concept.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Simple and log returns, both defined here |
| Sampling grid | Any regular pandas frequency; examples use month-ends (`ME`) |
| Annualisation | $\mathrm{AN}$ periods per year from `qis.get_annualization_factor` |
| Mean adjustment | Sample moments are demeaned, with `ddof=1`, unless a chapter states otherwise |
| Timing | A return at $t$ covers $(t-1,t]$; a decision at $t$ applies over $(t,t+1]$ |
| Output units | Decimal fractions: `0.10` means 10% |
| qis default | `qis.to_returns` returns simple returns and forward-fills missing prices |

The table defines each convention-card row:

| Row | What it states |
|---|---|
| Return basis | Simple or log returns, and whether they are total or in excess of cash |
| Sampling grid | The pandas frequency on which returns are formed before estimation |
| Annualisation | The factor $\mathrm{AN}$ and how it is applied: $\mathrm{AN}$ for means, $\sqrt{\mathrm{AN}}$ for volatilities |
| Mean adjustment | Which mean, if any, is removed before second moments are formed |
| Timing | Which information a quantity dated $t$ may use, and when it is applied |
| Output units | The units of the result |
| qis default | The default arguments of the principal entry point |

### Reserved symbols

These symbols have one meaning in every chapter. Hats denote estimators, and a bar denotes a
sample mean.

| Symbol | Meaning |
|---|---|
| $t$, $T$ | Observation date; number of observations in a sample |
| $i$, $j$ | Asset or instrument indices |
| $P_{i,t}$ | Positive price, NAV or wealth level of asset or strategy $i$ at $t$ |
| $V_t$ | Portfolio NAV or portfolio value in its reporting currency |
| $r_{i,t}$ | Simple return over $(t-1,t]$ |
| $\ell_{i,t}$ | Log return over $(t-1,t]$ |
| $r^{f}_t$ | Cash return accrued over $(t-1,t]$ |
| $\tilde r_{i,t}$ | Return in excess of cash |
| $\mathrm{TR}$, $R_{\mathrm{pa}}$ | Total return and per-annum (compound) return |
| $Y$ | Elapsed calendar years, days divided by 365.25 |
| $\tau$ | A horizon or maturity measured in years |
| $\mathrm{AN}$ | Annualisation factor: periods per year of the sampling grid |
| $w_{i,t}$, $w^{*}_{i,t}$ | Realised (drifted) weight; target weight |
| $u_{i,t}$ | Units held |
| $\mu$, $\sigma$, $\Sigma$, $\rho$ | Mean, volatility, covariance matrix, correlation |
| $s(x)$ | Sample standard deviation of $x$ with `ddof=1` |
| $\lambda$, $N$, $H$ | EWM decay, span and half-life, with $\lambda=1-2/(N+1)$ |
| $\alpha$, $\beta$, $\varepsilon_t$ | Regression intercept, slope and residual |
| $D_t$ | Drawdown from the running peak |
| $\mathrm{SR}$, $\mathrm{TE}$, $\mathrm{IR}$, $\mathrm{IC}$ | Sharpe ratio, tracking error, information ratio, information coefficient |
| $^{\top}$, $\operatorname{Var}$, $\operatorname{Cov}$, $\mathbb{E}$ | Transpose, variance, covariance, expectation |

$\mathrm{AN}$ is set upright and read as one symbol, like $\mathrm{TE}$; it is never the product
of $A$ and $N$. Counts with decorations, such as an effective number $N_{\mathrm{eff}}$, keep the
count meaning of $N$. A chapter that needs a symbol outside this table declares it locally and
does not reuse a reserved one.

## Methodology

### Simple and log returns

**Definition.** For a positive level $P_t$, the simple and log returns over $(t-1,t]$ are

$$
r_t=\frac{P_t}{P_{t-1}}-1,
\qquad
\ell_t=\log\frac{P_t}{P_{t-1}}=\log(1+r_t).
$$

**Identity (aggregation through time).** Over $m$ consecutive periods,

$$
\frac{P_m}{P_0}=\prod_{t=1}^{m}(1+r_t),
\qquad
\log\frac{P_m}{P_0}=\sum_{t=1}^{m}\ell_t .
$$

**Proof.** The product of consecutive ratios $P_t/P_{t-1}$ telescopes to $P_m/P_0$; taking
logarithms turns the product into a sum. $\square$

**Proposition (aggregation across assets).** A portfolio holding $u_{i,t-1}$ units over
$(t-1,t]$, with no trades, flows or costs in the period, has simple return

$$
r_{p,t}=\sum_i w_{i,t-1}\,r_{i,t},
\qquad
w_{i,t-1}=\frac{u_{i,t-1}P_{i,t-1}}{V_{t-1}} .
$$

**Proof.** $V_t-V_{t-1}=\sum_i u_{i,t-1}(P_{i,t}-P_{i,t-1})=\sum_i u_{i,t-1}P_{i,t-1}\,r_{i,t}$.
Divide by $V_{t-1}$. $\square$

The weights are the realised weights at the start of the period, not target weights. The
identity has no log-return analogue: $\ell_{p,t}=\log\big(\sum_i w_{i,t-1}e^{\ell_{i,t}}\big)$, which is not
$\sum_i w_{i,t-1}\ell_{i,t}$.

> **Pitfall.** Averaging asset log returns with portfolio weights does not give the portfolio
> log return, and averaging asset returns with *fixed target* weights describes a portfolio
> rebalanced every period, not a portfolio that holds units.

### Total and per-annum returns

**Definition.** For a history from $t_0$ to $t_1$ with $Y$ elapsed years, the total return and
the per-annum return are

$$
\mathrm{TR}=\frac{P_{t_1}}{P_{t_0}}-1,
\qquad
R_{\mathrm{pa}}=
\begin{cases}
(1+\mathrm{TR})^{1/Y}-1, & Y>1,\\
\mathrm{TR}, & Y\le 1.
\end{cases}
$$

$Y$ is the number of calendar days divided by 365.25. A history of one year or less keeps its
total return by default; it is not extrapolated to a one-year figure.

**Identity (per-annum return and log returns).** For $Y>1$ and $T$ returns on a grid with
$\mathrm{AN}$ periods per year,

$$
\log(1+R_{\mathrm{pa}})=\frac{1}{Y}\sum_{t=1}^{T}\ell_t=\frac{T}{Y}\,\bar\ell\approx\mathrm{AN}\,\bar\ell .
$$

**Proof.** By the aggregation identity, $\log(1+\mathrm{TR})=\sum_t\ell_t$, and
$\log(1+R_{\mathrm{pa}})=\log(1+\mathrm{TR})/Y$. On a regular grid $T/Y$ is close to
$\mathrm{AN}$. $\square$

> **Insight.** The per-annum return is the exponential map of the annualised mean log return.
> This is why a Sharpe ratio built on $R_{\mathrm{pa}}$ is numerically close to one built on
> $\mathrm{AN}\,\bar\ell$. Both numerators sit below the arithmetic one, $\mathrm{AN}\,\bar r$,
> by roughly half the annualised variance, because $\ell\approx r-r^2/2$.

### Annualisation

**Definition.** $\mathrm{AN}$ is the number of periods per year of the sampling grid, returned
by `qis.get_annualization_factor`:

| Grid | Pandas frequency | $\mathrm{AN}$ |
|---|---|---:|
| Business day | `B` | 252 |
| Calendar day | `D` | 365 |
| Week | `W-WED` | 52 |
| Month-end | `ME` | 12 |
| Quarter-end | `QE` | 4 |
| Year-end | `YE` | 1 |

A periodic mean is annualised by $\mathrm{AN}$ and a periodic standard deviation by
$\sqrt{\mathrm{AN}}$:

$$
\hat\mu_{\mathrm{ann}}=\mathrm{AN}\,\bar x,
\qquad
\hat\sigma_{\mathrm{ann}}=\sqrt{\mathrm{AN}}\,s(x).
$$

**Proposition (square-root-of-time).** If $\ell_1,\ldots,\ell_k$ are uncorrelated with common
variance $\sigma^2$, then $\operatorname{Var}\big(\sum_{t=1}^{k}\ell_t\big)=k\sigma^2$.

**Proof.** The variance of a sum is the sum of all covariances; with zero off-diagonal
covariances only the $k$ variances remain. $\square$

The scaling is exact for uncorrelated log returns and approximate for simple returns. Serial
correlation changes it; the [reporting-frequency chapter](frequency_convention_note.md) gives
the variance-ratio correction.

> **Pitfall.** Daily report windows and EWM spans are sized with 260 observations per year,
> but daily volatilities are annualised with 252. The two numbers answer different questions:
> how long a window is, and how many periods a year contains.

### Excess returns and cash

**Definition.** With an annual cash rate $y_t$ quoted on its own calendar, the period cash
return and the excess return are

$$
r^{f}_t=y_{t-1}\,\frac{d_t-d_{t-1}}{365},
\qquad
\tilde r_t=r_t-r^{f}_t,
$$

where $d_t$ is the calendar date of observation $t$. The rate is the quote known at the start of
the period, one observation earlier, and accrues on an ACT/365 day count. Per-annum returns use
365.25-day years; the two day counts are separate conventions.

### Timing and information

A quantity dated $t$ may use observations up to and including $t$. A portfolio decision dated
$t$ earns the return over $(t,t+1]$. Estimators used inside a backtest must be point in time:
an exponentially weighted mean or an expanding mean qualifies; a full-sample mean does not.

> **Pitfall.** `qis.MeanAdjType.INSAMPLE` subtracts the full-sample mean. It is correct for a
> descriptive exhibit and wrong inside a backtest, where it leaks later observations into
> earlier estimates.

### Sample moments and exponential weighting

The sample mean and standard deviation are

$$
\bar x=\frac{1}{T}\sum_{t=1}^{T}x_t,
\qquad
s(x)=\sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(x_t-\bar x)^2}.
$$

Exponentially weighted estimators are parameterised by the span $N$, with decay
$\lambda=1-2/(N+1)$ and recursion $m_t=\lambda m_{t-1}+(1-\lambda)x_t$. A span of $N$
observations gives the same mean age of information and the same effective number of
observations as an $N$-observation equal-weight window.

## Worked example

Four month-end prices 100, 105, 102.9 and 108.045 have simple returns 5%, −2% and 5%. Their
log returns sum to $\log 1.08045$. A two-year history from 100 to 121 has
$Y=730/365.25$ and a per-annum return of about 10.007%. These fixed inputs illustrate the
identities; they are not market data.

```python
from math import isclose, log

import numpy as np
import pandas as pd
import qis

prices = pd.Series([100.0, 105.0, 102.9, 108.045],
                   index=pd.date_range('2024-01-31', periods=4, freq='ME'))
simple = qis.to_returns(prices=prices, is_log_returns=False, drop_first=True)
log_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True)
np.testing.assert_allclose(simple, [0.05, -0.02, 0.05], atol=1e-12)
assert isclose(log_returns.sum(), log(prices.iloc[-1] / prices.iloc[0]), abs_tol=1e-12)
assert isclose(np.prod(1.0 + simple) - 1.0, 0.08045, abs_tol=1e-12)

two_years = pd.Series([100.0, 121.0], index=pd.to_datetime(['2020-12-31', '2022-12-31']))
years = qis.compute_num_years(prices=two_years)
assert isclose(years, 730 / 365.25, abs_tol=1e-12)
assert isclose(qis.compute_pa_return(prices=two_years), 1.21 ** (1.0 / years) - 1.0,
               abs_tol=1e-12)

assert [qis.get_annualization_factor(freq) for freq in ['B', 'W-WED', 'ME', 'QE']] == [
    252.0, 52.0, 12.0, 4.0]

cash = pd.Series([0.0365, 0.0365, 0.073], index=pd.to_datetime(
    ['2024-01-31', '2024-02-29', '2024-03-31']))
returns = pd.Series([0.0, 0.01, 0.02], index=cash.index)
excess = qis.compute_excess_returns(returns=returns, rates_data=cash)
np.testing.assert_allclose(excess.iloc[1:], [0.01 - 0.0365 * 29 / 365,
                                             0.02 - 0.0365 * 31 / 365], atol=1e-15)
```

The last check shows the one-observation lag: the March excess return uses the February
rate of 3.65%, not the 7.3% quoted at the end of March.

## Implementation in qis

| Quantity | Formula | qis entry point |
|---|---|---|
| Simple and log returns | $r_t$, $\ell_t$ | `qis.to_returns(prices, is_log_returns=..., freq=...)` |
| Levels from returns | $P_0\prod(1+r_t)$ | `qis.returns_to_nav` |
| Elapsed years | days / 365.25 | `qis.compute_num_years` |
| Total and per-annum return | $\mathrm{TR}$, $R_{\mathrm{pa}}$ | `qis.compute_total_return`, `qis.compute_pa_return` |
| Annualisation factor | $\mathrm{AN}$ | `qis.get_annualization_factor`, `qis.infer_annualisation_factor_from_df` |
| Excess returns | $\tilde r_t$ | `qis.compute_excess_returns` |
| Statistic conventions | the convention card | `qis.PerfParams` |

The implementations are in
[returns.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/returns.py)
and [annualisation.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/utils/annualisation.py).
`qis.infer_annualisation_factor_from_df` reads the index frequency; on an irregular index it
warns and falls back to 252, so resample to an explicit grid before estimating.

## Interpretation and limitations

- `qis.to_returns` forward-fills missing prices by default. A filled price creates a zero
  return, not information; use `ffill_nans=False` when a gap must stay missing.
- The square-root-of-time rule assumes uncorrelated increments. Smoothed or illiquid returns
  violate it; see [private-asset unsmoothing](private_asset_unsmoothing.md).
- Three day counts coexist by design: 365.25-day years for per-annum returns, ACT/365 for
  cash accrual, and $\mathrm{AN}$ periods per year for sampled statistics.
- A convention card states defaults. An explicit argument overrides them, and a report that
  changes a default should say so.

## See also

- [Reporting frequency and annualisation](frequency_convention_note.md)
- [Performance analytics and Sharpe conventions](performance_analytics_and_sharpe.md)
- [Incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md)
- [Portfolio backtesting](portfolio_backtesting.md)
- [Bibliography](bibliography.md)

## References

1. Campbell, J. Y., Lo, A. W., and MacKinlay, A. C. (1997). *The Econometrics of Financial Markets*. Princeton University Press.
2. Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and Attribution*, 2nd edition. Wiley.
3. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
