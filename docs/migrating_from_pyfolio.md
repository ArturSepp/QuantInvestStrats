---
myst:
  html_meta:
    description: >-
      Move a pyfolio-reloaded workflow to qis: tear sheets, perf_stats rows and plots mapped to
      qis functions, with the conventions that make each statistic agree or differ.
---

# Migrating from pyfolio-reloaded to qis

*Author: [Artur Sepp](https://github.com/ArturSepp)*

This guide is part of [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

This guide maps a [pyfolio-reloaded](https://github.com/stefan-jansen/pyfolio-reloaded) workflow
onto qis: the tear sheets, the rows of `perf_stats` and the individual plots. For each statistic it
states whether the two packages agree on the same input and, where they do not, which convention
differs. Some pyfolio analyses, such as round trips and capacity, have no qis counterpart; the
[last section](#what-has-no-counterpart-in-qis) lists them.

The mappings were checked on 6 October 2026 with qis 5.33.3, pyfolio-reloaded 0.9.9 and
empyrical-reloaded 0.5.12, on the frozen synthetic series used below. Check the release you install
before relying on an exact signature.

## From returns to a NAV

pyfolio takes noncumulative daily simple returns. The qis performance tables and reports take
price or NAV levels instead, so compound the returns first. Give the NAV a base observation one
business day before the first return: `qis.returns_to_nav` starts the NAV at the first return's
date, so without a base the first return would not enter any statistic.

~~~python
import pandas as pd

import qis
from qis.datasets import generate_synthetic_universe

# Stand-ins for the series passed to pyfolio: daily noncumulative simple returns.
universe = generate_synthetic_universe(
    start='2018-01-02', end='2025-12-31', seed=20260725, apply_quirks=False
)
returns = universe.prices['SEQ_US'].pct_change().dropna().rename('strategy')
benchmark_rets = universe.benchmark_prices.iloc[:, 0].pct_change().dropna().rename('benchmark')


def nav_with_base(simple_returns: pd.Series) -> pd.Series:
    """Compound simple returns into a NAV that starts at 1.0 one business day earlier."""
    base = pd.Series(0.0, index=[simple_returns.index[0] - pd.offsets.BDay(1)])
    return qis.returns_to_nav(pd.concat([base, simple_returns])).rename(simple_returns.name)


navs = pd.concat([nav_with_base(returns), nav_with_base(benchmark_rets)], axis=1)
~~~

## Tear sheets

| pyfolio-reloaded | qis | Notes |
|---|---|---|
| `create_returns_tear_sheet(returns, benchmark_rets=...)` | `qis.factsheet(navs, benchmark='benchmark')` | A multi-asset report on the strategy and benchmark NAVs. |
| `create_simple_tear_sheet` | `qis.factsheet` with `reporting_frequency='monthly'` | Choose the reporting grid explicitly; see [Factsheets and reporting](factsheets_and_reporting.md). |
| `create_full_tear_sheet` with positions | `qis.backtest_model_portfolio`, then `qis.factsheet(portfolio_data, benchmark_prices=...)` | Needs instrument prices and weights; see [below](#positions-and-the-strategy-report). |
| `create_position_tear_sheet` | The strategy report from a `PortfolioData` | Weights, exposures and attribution panels. |
| `create_txn_tear_sheet` | `PortfolioData.get_turnover` | Turnover uses the qis [turnover conventions](turnover_conventions.md), not pyfolio's `turnover_denom`. |
| `create_interesting_times_tear_sheet` | No direct counterpart | [Regime-conditional performance](regime_conditional_performance.md) answers a related question. |
| `create_perf_attrib_tear_sheet` | No direct counterpart | See [factor risk models](factor_risk_models.md) and [Brinson attribution](brinson_attribution.md). |

The returns tear sheet becomes one call. The facade returns a list of matplotlib figures; pass
them to `qis.save_figs_to_pdf` to write a PDF.

~~~python
figures = qis.factsheet(
    navs,
    benchmark='benchmark',
    reporting_frequency='monthly',
    factsheet_name='Strategy versus benchmark',
)
~~~

`qis.factsheet` also accepts returns with `data_is_returns=True`. That path compounds through
`qis.returns_to_nav` without a base observation, so pass the NAVs above when the first return
matters.

## perf_stats rows and qis columns

`pyfolio.timeseries.perf_stats` reports daily simple-return statistics annualised with 252 periods.
qis states its conventions in `PerfParams`. These parameters select the same daily simple-return
grid:

~~~python
daily = qis.PerfParams(freq='B', freq_skewness='B', return_type=qis.ReturnTypes.RELATIVE)
performance = qis.compute_ra_perf_table_with_benchmark(
    prices=navs, benchmark='benchmark', perf_params=daily
)
moments = qis.compute_risk_table(prices=navs, perf_params=daily)
~~~

`freq='B'` sets the volatility, regression and excess-return grids to business days, and
`ReturnTypes.RELATIVE` computes volatility from simple rather than log returns. `freq` does not
set the skewness grid, which keeps its monthly default unless `freq_skewness` is given.

On the synthetic strategy above, with the NAV base observation:

| pyfolio row | pyfolio | qis column | qis | Agree? |
|---|---|---|---|---|
| Cumulative returns | 0.5594 | `TOTAL_RETURN` | 0.5594 | Yes |
| Annual volatility | 0.1636 | `VOL` | 0.1636 | Yes, with `ReturnTypes.RELATIVE` |
| Sharpe ratio | 0.4098 | `SHARPE_ARITH` | 0.4098 | Yes |
| Max drawdown | −0.2869 | `MAX_DD` | −0.2869 | Yes, with the base observation |
| Beta | 1.6636 | `BETA` | 1.6636 | Yes, on the daily regression grid |
| Annual return | 0.0551 | `PA_RETURN` | 0.0571 | No: year count |
| Calmar ratio | 0.1922 | `CALMAR_RATIO` | 0.1992 | No: annual return in the numerator |
| Sortino ratio | 0.6021 | `SORTINO_RATIO` | 0.6154 | No: different definition |
| Alpha | 0.005940 | `ALPHA_AN` | 0.005923 | No: compounded versus arithmetic annualisation |
| Skew | 0.1468 | `SKEWNESS` | 0.1469 | Nearly: bias correction |
| Kurtosis | 0.0646 | `KURTOSIS` | 0.0676 | Nearly: bias correction |

The differences come from a few conventions:

- **Annual return.** pyfolio counts years as the number of daily returns divided by 252. qis counts
  calendar days divided by 365.25 between the first and last NAV dates. The
  [statistic catalogue](performance_statistics.md) gives every qis formula.
- **Two Sharpe ratios.** pyfolio's Sharpe ratio is the arithmetic one, $\sqrt{252}\,\bar r/s(r)$,
  which is `SHARPE_ARITH`. The qis column labelled `Sharpe (rf=0)`, `SHARPE_RF0`, divides the
  per-annum compounded return by volatility; on this series it is 0.349. See
  [the three table conventions](performance_analytics_and_sharpe.md#the-three-table-conventions).
- **Sortino ratio.** pyfolio divides the annualised arithmetic mean by the root mean square of the
  negative returns over all observations. qis divides the per-annum excess return by the standard
  deviation of the negative returns only.
- **Alpha.** pyfolio compounds the daily regression alpha as $(1+\alpha)^{252}-1$; `ALPHA_AN` is
  $252\,\alpha$. The betas agree.
- **Skew and kurtosis.** pyfolio uses the biased `scipy.stats` moments; qis reports the
  bias-corrected $G_1$ and excess $G_2$.

pyfolio's Stability, Omega ratio, Tail ratio and Daily value at risk (the mean minus two standard
deviations) have no column in the qis tables.

## Plots

| pyfolio-reloaded | qis | Notes |
|---|---|---|
| `plot_rolling_returns` | `qis.plot_prices_with_dd` or `qis.plot_prices` | Cumulative NAV, optionally with the drawdown panel. |
| `plot_drawdown_underwater` | `qis.plot_rolling_drawdowns` | |
| `plot_drawdown_periods`, `show_worst_drawdown_periods` | `qis.plot_top_drawdowns_paths`, `qis.compute_drawdowns_stats_table` | The table lists start, trough, end, depth and recovery days. |
| `plot_rolling_sharpe` | `qis.plot_rolling_perf_stat` with `RollingPerfStat.SHARPE` | qis rolls over log returns; values differ from pyfolio's simple-return version. |
| `plot_rolling_volatility` | `qis.plot_rolling_perf_stat` with `RollingPerfStat.VOL` | |
| `plot_rolling_beta` | `qis.compute_one_factor_ewm_betas` | An exponentially weighted beta rather than fixed 6- and 12-month windows. |
| `plot_monthly_returns_heatmap` | `qis.plot_returns_heatmap` | Years by months, with an annual column. |
| `plot_annual_returns` | `qis.plot_periodic_returns_table` with `freq='YE'` | `qis.compute_periodic_returns` returns the numbers. |
| `plot_monthly_returns_dist` | `qis.plot_histogram` of `qis.to_returns(..., freq='ME')` | |

pyfolio's default rolling window is 126 business days (six months); qis defaults to 260, so pass
`roll_periods` to compare like with like:

~~~python
strategy_nav = navs['strategy']
underwater = qis.plot_rolling_drawdowns(prices=strategy_nav)
worst_drawdowns = qis.compute_drawdowns_stats_table(price=strategy_nav, max_num=5)
rolling_sharpe = qis.plot_rolling_perf_stat(
    prices=strategy_nav, rolling_perf_stat=qis.RollingPerfStat.SHARPE,
    roll_periods=126, roll_freq='B',
)
rolling_beta = qis.compute_one_factor_ewm_betas(
    x=benchmark_rets, y=returns.to_frame(), span=126
)
heatmap = qis.plot_returns_heatmap(prices=strategy_nav)
annual_returns = qis.compute_periodic_returns(prices=navs, freq='YE')
~~~

## Positions and the strategy report

pyfolio's positions frame holds the dollar value of each holding plus a `cash` column, so its rows
sum to the portfolio value. Dividing by that sum gives weights:

~~~python
def positions_to_weights(positions: pd.DataFrame) -> pd.DataFrame:
    """Turn pyfolio positions, dollar values with a cash column, into weights of portfolio value."""
    return positions.drop(columns='cash').div(positions.sum(axis=1), axis=0)
~~~

qis builds its strategy report from a `PortfolioData` object. `qis.backtest_model_portfolio`
creates one from instrument prices and target weights: it converts the targets into held units,
applies costs and records the weights, turnover and attribution the report shows. This re-runs
the allocation under qis accounting. It does not replay pyfolio's recorded transactions, so the
NAV can differ from the original record through execution timing and costs. The
[backtesting guide](portfolio_backtesting.md) defines the accounting.

~~~python
asset_prices = universe.prices[['SEQ_US', 'SBD_TSY', 'SCM_GLD']]
weights = qis.generate_static_weights_schedule(
    prices=asset_prices,
    weights={'SEQ_US': 0.6, 'SBD_TSY': 0.3, 'SCM_GLD': 0.1},
    rebalancing_freq='ME',
)
portfolio_data = qis.backtest_model_portfolio(
    prices=asset_prices, weights=weights, rebalancing_freq=None, ticker='Strategy'
)
strategy_report = qis.factsheet(portfolio_data, benchmark_prices=navs['benchmark'])
~~~

Replace the static schedule with the weights from `positions_to_weights` on the dates your
strategy rebalanced.

## What has no counterpart in qis

These pyfolio analyses have no qis equivalent:

- Round-trip analysis and its statistics (`create_round_trip_tear_sheet`).
- Capacity and liquidity analysis (`create_capacity_tear_sheet`).
- Slippage sweeps and transaction-level diagnostics beyond turnover and costs.
- Event-window performance (`create_interesting_times_tear_sheet`).
- In-sample and out-of-sample splits by `live_start_date`. In qis, compute the tables on each
  `TimePeriod` separately.

qis in turn adds explicit return, grid and annualisation conventions, excess returns over a
supplied rate series, [Sharpe inference](performance_analytics_and_sharpe.md#sampling-uncertainty),
[regime-conditional performance](regime_conditional_performance.md) and covariance-based
[risk and tracking error](tracking_error_and_risk.md).

## See also

- [Migrating from QuantStats to qis](migrating_from_quantstats.md)
- [Choosing between qis, bt, QuantStats, pyfolio-reloaded, and vectorbt](package_comparison.md)
- [The performance-statistic catalogue](performance_statistics.md)
- [Factsheet reference](factsheets.md)

## References

- pyfolio-reloaded contributors. [Repository](https://github.com/stefan-jansen/pyfolio-reloaded),
  including [tears.py](https://github.com/stefan-jansen/pyfolio-reloaded/blob/main/src/pyfolio/tears.py)
  and [timeseries.py](https://github.com/stefan-jansen/pyfolio-reloaded/blob/main/src/pyfolio/timeseries.py).
- empyrical-reloaded contributors. [stats.py](https://github.com/stefan-jansen/empyrical-reloaded/blob/main/src/empyrical/stats.py).
- Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet
  reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
