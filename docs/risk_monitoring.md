---
myst:
  html_meta:
    description: >-
      Portfolio risk monitoring with qis: volatility, risk contributions, value at risk, tracking
      error, beta, drawdowns, correlations and stress tests, mapped to qis functions and chapters.
---

# Portfolio risk monitoring with qis

*Author: [Artur Sepp](https://github.com/ArturSepp)*

This guide is part of [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Risk monitoring asks the same questions of a portfolio every day: how much risk it runs, where
that risk comes from, how far it can drift from its benchmark, how deep its losses are and what a
market shock would do to it. This page maps each question to the qis functions that answer it and
to the handbook chapter that defines the calculation, then computes a point-in-time risk snapshot
of one portfolio offline.

## Monitoring questions and where qis answers them

| Question | qis entry points | Chapter |
|---|---|---|
| How much risk is the portfolio running? | `PortfolioData.compute_portfolio_vol`, `qis.compute_portfolio_vol`, `qis.compute_ewm_vol` | [Exponentially weighted estimators](ewm_estimators.md) |
| Where does the risk come from? | `qis.compute_portfolio_risk_contribution_ratios`, `PortfolioData.compute_risk_contributions_implied_by_covar`, `qis.RiskModel` | [Portfolio risk and Euler contributions](risk_contributions.md), [factor risk models](factor_risk_models.md) |
| How large is a bad day? | `PortfolioData.compute_portfolio_vars`, `qis.compute_portfolio_correlated_var_by_groups` | [Parametric value at risk](risk_contributions.md#parametric-value-at-risk) |
| How far can it drift from the benchmark? | `qis.RiskModel` for ex-ante tracking error, `qis.compute_ewma_realised_tracking_error`, `PortfolioData.compute_portfolio_benchmark_betas` | [Tracking error](tracking_error_and_risk.md), [alpha and beta](benchmark_relative_performance.md) |
| How deep and how long are the losses? | `qis.compute_max_current_drawdown`, `qis.compute_rolling_drawdown_time_under_water`, `qis.compute_drawdowns_stats_table` | [Drawdowns and time under water](drawdowns.md) |
| Is diversification holding up? | `qis.compute_ewm_corr_df`, `qis.compute_pca_r2`, `qis.compute_portfolio_breadth` | [Covariance, correlation and principal components](covariance_correlation_pca.md), [portfolio breadth](portfolio_breadth.md) |
| What would a market shock do? | `qis.compute_factor_sensitivity`, `qis.generate_portfolio_stress_report` | [Factor stress testing](stress_testing.md), [instrument portfolios and stress reports](portfolio_stress.md), [stress testing with options](stress_testing_with_options.md) |
| How does it behave in bear markets? | `qis.compute_bnb_regimes_pa_perf_table` | [Regime-conditional performance](regime_conditional_performance.md) |
| Do the reported returns understate risk? | `qis.compute_ewm_vector_autocorr_df`, `qis.estimate_dimson_beta` | [Serial dependence](serial_dependence.md), [private-asset unsmoothing](private_asset_unsmoothing.md) |

## A point-in-time risk snapshot

This example backtests a monthly-rebalanced 60/30/10 portfolio on the frozen synthetic universe
and computes one monitoring measure per question. Every estimate dated $t$ uses returns up to $t$
and the weights held at $t$.

~~~python
import qis
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe(
    start='2018-01-02', end='2025-12-31', seed=20260725, apply_quirks=False
)
prices = universe.prices[['SEQ_US', 'SBD_TSY', 'SCM_GLD']]
benchmark_prices = universe.benchmark_prices
weights = qis.generate_static_weights_schedule(
    prices=prices, weights={'SEQ_US': 0.6, 'SBD_TSY': 0.3, 'SCM_GLD': 0.1}, rebalancing_freq='ME'
)
portfolio = qis.backtest_model_portfolio(
    prices=prices, weights=weights, rebalancing_freq=None, ticker='Portfolio'
)
nav = portfolio.get_portfolio_nav()

# How much risk: annualised EWM volatility of weekly returns, span 13 weeks.
portfolio_vol = portfolio.compute_portfolio_vol(freq='W-WED', span=13)

# Where it comes from: Euler shares of volatility under monthly EWM covariances.
covar_dict = qis.estimate_rolling_ewma_covar(
    prices=prices, returns_freq='W-WED', rebalancing_freq='ME', span=52
)
risk_shares = portfolio.compute_risk_contributions_implied_by_covar(
    covar_dict=covar_dict, normalise=True
)
ex_ante_vol = portfolio.compute_ex_anti_portfolio_vol_implied_by_covar(covar_dict=covar_dict)

# A bad day: one-day 99% normal value at risk of the current positions.
value_at_risk, _ = portfolio.compute_portfolio_vars(is_correlated=True, freq='B', vol_span=33)

# Distance from the benchmark: EWM beta and realised tracking error of monthly returns.
betas = portfolio.compute_portfolio_benchmark_betas(
    benchmark_prices=benchmark_prices, factor_beta_span=63
)
tracking_error = qis.compute_ewma_realised_tracking_error(
    portfolio_nav=nav, benchmark_nav=benchmark_prices.iloc[:, 0], ewma_span=36, freq='ME'
)

# Losses: maximum and current drawdown, and calendar days below the last peak.
max_drawdown, current_drawdown = qis.compute_max_current_drawdown(prices=nav)
drawdowns, time_under_water = qis.compute_rolling_drawdown_time_under_water(prices=nav)

# Diversification: EWM correlations of weekly returns, one column per pair.
correlations = qis.compute_ewm_corr_df(
    df=qis.to_returns(prices=prices, freq='W-WED', drop_first=True), span=52
)
~~~

On the last date, 31 December 2025, the snapshot reads:

| Measure | Value | Units and convention |
|---|---|---|
| EWM volatility, `instrument weighted vol` | 8.8% | Annualised; weekly simple returns, held weights lagged one week |
| EWM volatility, `strategy returns vol` | 8.9% | Annualised; the portfolio's own weekly returns |
| Covariance-implied volatility | 10.7% | Annualised; the 52-week EWM covariance at the month end |
| Risk shares | SEQ_US 100.8%, SBD_TSY −2.9%, SCM_GLD 2.1% | Euler shares that sum to one |
| Value at risk, `Total` | 1.37% | One-day 99% normal VaR, fraction of NAV, log returns |
| Beta to SBM_6040 | 1.02 | EWM beta of daily returns, span 63 days |
| Realised tracking error | 1.6% | Annualised; EWM of monthly return differences, span 36 months |
| Maximum and current drawdown | −18.6% and −6.1% | Fractions of the running peak |
| Time under water | 198 | Calendar days since the last peak |

The two volatility columns differ by weight drift and rebalancing within each week, by costs and
because the first applies the latest weights to the whole covariance history. With 60% of the
capital, the equity sleeve carries about 101% of the portfolio's volatility; the negative share of
the bond sleeve means that adding to it would reduce portfolio risk at the margin. The
[risk contributions chapter](risk_contributions.md) proves the decomposition and the
[EWM chapter](ewm_estimators.md) explains spans, seeds and warm-up.

## Conventions to fix before monitoring

- **Timing.** Pair the weights of date $t$ with a covariance estimated from returns up to $t$.
  `compute_portfolio_vol` lags the weights by one row by default (`weight_lag=1`); the VaR
  functions do not lag them. Inside a backtest, any full-sample estimate is a look-ahead.
- **Return basis and grid.** The VaR functions use log returns on `freq`; the volatility columns
  above use simple returns. Daily, weekly and monthly estimates of the same portfolio differ, and
  autocorrelation changes how they scale; see the
  [reporting-frequency chapter](frequency_convention_note.md).
- **Span.** An EWM span of $N$ periods has a mean lag of $(N-1)/2$ periods, an effective sample
  size of $N$ and a half-life of about $0.35N$
  ([EWM spans](ewm_estimators.md#span-mean-lag-effective-sample-size-and-half-life)). A short
  span reacts quickly and is noisy; a long span is stable and lags.
- **Ex-ante versus realised.** Covariance-implied volatility and `RiskModel` tracking error are
  forecasts from the current weights. Realised tracking error and drawdowns measure what happened.
  Monitor both: a persistent gap between forecast and realised risk is itself a signal.
- **Normality.** The normal VaR is a fixed multiple of volatility. Fat tails make it too small;
  compare it with the realised worst days and with stress scenarios.

## Reporting

The single-strategy factsheet collects most of these measures on one or more pages: NAV and
drawdowns, rolling volatility and Sharpe ratio, weights, turnover and benchmark betas. Its
`add_current_position_var_risk_sheet=True` option adds a page of current position value at risk.

~~~python
risk_report = qis.factsheet(
    portfolio, benchmark_prices=benchmark_prices, add_current_position_var_risk_sheet=True
)
~~~

`qis.save_figs_to_pdf` writes the figures to a PDF. The [factsheet reference](factsheets.md)
lists the report options and the [gallery](gallery.md) shows each report type.

## See also

- [Portfolio risk and Euler contributions](risk_contributions.md)
- [Tracking error and benchmark-relative risk](tracking_error_and_risk.md)
- [Factor stress testing](stress_testing.md)
- [Drawdowns and time under water](drawdowns.md)
- [Portfolio backtesting](portfolio_backtesting.md)

## References

- Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet
  reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
