---
myst:
  html_meta:
    description: >-
      Move a QuantStats workflow to qis: reports, qs.stats metrics and qs.plots charts mapped to
      qis functions, with the conventions that make each statistic agree or differ.
---

# Migrating from QuantStats to qis

*Author: [Artur Sepp](https://github.com/ArturSepp)*

This guide is part of [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

This guide maps a [QuantStats](https://github.com/ranaroussi/quantstats) workflow onto qis: the
reports, the `qs.stats` metrics and the `qs.plots` charts. For each metric it states whether the
two packages agree on the same input and, where they do not, which convention differs.
The `qis.factsheet` facade returns Matplotlib figures or saves a PDF. A separate optional
[PyBloqs backend and examples](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/factsheets/pybloqs_factsheets.py)
support HTML/PDF reporting.

The mappings were checked on 6 October 2026 with qis 5.33.3 and QuantStats 0.0.86, on the frozen
synthetic series used below. QuantStats has changed its statistic conventions between releases,
including how `cagr` counts years, so check the release you install before comparing numbers.

## Inputs: returns, prices and NAVs

QuantStats accepts returns or prices and guesses which it was given: a series whose minimum is at
least zero and whose maximum exceeds one is treated as prices. qis does not guess. Its performance
tables take price or NAV levels, and its report facade takes prices unless `data_is_returns=True`.

To start from the returns you passed to QuantStats, compound them into a NAV with a base
observation one business day before the first return. `qis.returns_to_nav` starts the NAV at the
first return's date, so without a base the first return would not enter any statistic.

~~~python
import pandas as pd

import qis
from qis.datasets import generate_synthetic_universe

# Stand-ins for the series passed to QuantStats: daily simple returns.
universe = generate_synthetic_universe(
    start='2018-01-02', end='2025-12-31', seed=20260725, apply_quirks=False
)
returns = universe.prices['SEQ_US'].pct_change().dropna().rename('strategy')
benchmark = universe.benchmark_prices.iloc[:, 0].pct_change().dropna().rename('benchmark')


def nav_with_base(simple_returns: pd.Series) -> pd.Series:
    """Compound simple returns into a NAV that starts at 1.0 one business day earlier."""
    base = pd.Series(0.0, index=[simple_returns.index[0] - pd.offsets.BDay(1)])
    return qis.returns_to_nav(pd.concat([base, simple_returns])).rename(simple_returns.name)


navs = pd.concat([nav_with_base(returns), nav_with_base(benchmark)], axis=1)
~~~

A QuantStats benchmark given as a ticker string is downloaded through yfinance. In qis, pass the
benchmark series yourself, as above.

## Reports

| QuantStats | qis | Notes |
|---|---|---|
| `qs.reports.html(returns, benchmark, output=...)` | `qis.factsheet(navs, benchmark='benchmark')`, then `qis.save_figs_to_pdf` | A multi-page matplotlib report saved as PDF rather than HTML. |
| `qs.reports.full`, `qs.reports.basic` | `qis.factsheet` | Choose the reporting grid with `reporting_frequency`. |
| `qs.reports.metrics(mode='full')` | `qis.compute_ra_perf_table_with_benchmark` and `qis.compute_risk_table` | Tables as pandas DataFrames; see [below](#quantstats-metrics-and-qis-columns). |
| `qs.reports.plots` | `qis.factsheet` or the individual plots [below](#plots) | |

~~~python
figures = qis.factsheet(
    navs,
    benchmark='benchmark',
    reporting_frequency='monthly',
    factsheet_name='Strategy versus benchmark',
)
~~~

`qis.save_figs_to_pdf(figures, file_name='tearsheet', local_path=...)` writes the figures to one PDF.
The [factsheet reference](factsheets.md) covers the report types, including the single-strategy
report built from a backtested portfolio.

`qis.factsheet` also accepts the returns themselves with `data_is_returns=True`. In qis releases
after 5.33.3 that path adds the same base observation, one period of the inferred native
frequency before a non-zero first return, so it renders the same report as the NAVs above. In
qis 5.33.3 and earlier it compounds through `qis.returns_to_nav` without a base and drops the
first return, so pass the NAVs above instead.

## QuantStats metrics and qis columns

QuantStats computes its metrics on daily simple returns annualised with `periods=252`. qis states
its conventions in `PerfParams`. These parameters select the same daily simple-return grid:

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

| QuantStats metric | QuantStats | qis column | qis | Agree? |
|---|---|---|---|---|
| `comp` | 0.5594 | `TOTAL_RETURN` | 0.5594 | Yes |
| `volatility` | 0.1636 | `VOL` | 0.1636 | Yes, with `ReturnTypes.RELATIVE` |
| `sharpe` | 0.4098 | `SHARPE_ARITH` | 0.4098 | Yes |
| `max_drawdown` | −0.2869 | `MAX_DD` | −0.2869 | Yes, with the base observation |
| `skew` | 0.1469 | `SKEWNESS` | 0.1469 | Yes, with `freq_skewness='B'` |
| `kurtosis` | 0.0676 | `KURTOSIS` | 0.0676 | Yes, with `freq_skewness='B'` |
| `greeks` beta | 1.6636 | `BETA` | 1.6636 | Yes, on the daily regression grid |
| `greeks` alpha | 0.005923 | `ALPHA_AN` | 0.005923 | Yes, on the daily regression grid |
| `cagr` | 0.0551 | `PA_RETURN` | 0.0571 | No: year count |
| `calmar` | 0.1922 | `CALMAR_RATIO` | 0.1992 | No: annual return in the numerator |
| `sortino` | 0.6021 | `SORTINO_RATIO` | 0.6154 | No: different definition |
| `information_ratio` | 0.0256 | `qis.compute_te_ir_errors` | 0.4064 | No: qis annualises by $\sqrt{252}$ |

The differences come from a few conventions:

- **Annual return.** In 0.0.86, `cagr` counts years as the number of returns divided by `periods`.
  qis counts calendar days divided by 365.25 between the first and last NAV dates. The
  [statistic catalogue](performance_statistics.md) gives every qis formula.
- **Two Sharpe ratios.** `qs.stats.sharpe` is the arithmetic Sharpe ratio, $\sqrt{252}\,\bar r/s(r)$,
  which is `SHARPE_ARITH`. The qis column labelled `Sharpe (rf=0)`, `SHARPE_RF0`, divides the
  per-annum compounded return by volatility; on this series it is 0.349. See
  [the three table conventions](performance_analytics_and_sharpe.md#the-three-table-conventions).
- **Sortino ratio.** QuantStats divides the mean return by the root mean square of the negative
  returns over all observations. qis divides the per-annum excess return by the standard deviation
  of the negative returns only.
- **Information ratio.** `qs.stats.information_ratio` is the per-period mean of the active return
  over its standard deviation, not annualised. `qis.compute_te_ir_errors` returns the annualised
  tracking error and information ratio of the return differences you pass it.

The information ratio in the table comes from daily simple-return differences:

~~~python
active = (returns - benchmark).to_frame('strategy vs benchmark')
tracking_error, information_ratio = qis.compute_te_ir_errors(active)
~~~

### Risk-free rate

QuantStats takes `rf` as an annual rate, or a series, and converts a constant rate to a per-period
rate as $(1+r_f)^{1/252}-1$. qis takes a series of annualised cash rates in
`PerfParams.rates_data` and accrues it by calendar days; the excess columns, such as
`SHARPE_EXCESS` and `SHARPE_ARITH_EXCESS`, use it. See
[excess returns and funding](performance_analytics_and_sharpe.md#excess-returns-and-funding).

### Metrics without a column

`qs.stats.sharpe(smart=True)` applies an autocorrelation penalty. qis does not adjust the table
Sharpe ratio; the [serial dependence](serial_dependence.md#the-ar1-benchmark) chapter shows how
autocorrelation changes annualised volatility and the Sharpe ratio. `value_at_risk`, `conditional_value_at_risk` and
the remaining QuantStats metrics have no column in the qis performance tables. `win_rate` excludes
zero returns from its denominator, while the `POSITIVE` column of `qis.compute_desc_table` counts
every observation.

## Plots

| QuantStats | qis | Notes |
|---|---|---|
| `qs.plots.snapshot` | `qis.plot_prices_with_dd` | Cumulative NAV with the drawdown panel. |
| `qs.plots.returns`, `qs.plots.log_returns` | `qis.plot_prices`, with `is_log=True` for a log scale | |
| `qs.plots.drawdown` | `qis.plot_rolling_drawdowns` | |
| `qs.plots.drawdowns_periods` | `qis.plot_top_drawdowns_paths`, `qis.compute_drawdowns_stats_table` | |
| `qs.plots.rolling_sharpe` | `qis.plot_rolling_perf_stat` with `RollingPerfStat.SHARPE` | qis rolls over log returns; values differ from the simple-return version. |
| `qs.plots.rolling_volatility` | `qis.plot_rolling_perf_stat` with `RollingPerfStat.VOL` | |
| `qs.plots.rolling_beta` | `qis.compute_one_factor_ewm_betas` | An exponentially weighted beta rather than fixed windows. |
| `qs.plots.monthly_heatmap` | `qis.plot_returns_heatmap` | Years by months, with an annual column. |
| `qs.plots.yearly_returns` | `qis.plot_periodic_returns_table` with `freq='YE'` | `qis.compute_periodic_returns` returns the numbers. |
| `qs.plots.histogram`, `qs.plots.distribution` | `qis.plot_histogram` of `qis.to_returns(..., freq='ME')` | |
| `qs.plots.montecarlo` | No direct counterpart | qis [resampling and the bootstrap](reproducibility.md) covers block-bootstrap paths. |

QuantStats' default rolling window is 126 periods; qis defaults to 260, so pass `roll_periods` to
compare like with like:

~~~python
strategy_nav = navs['strategy']
snapshot = qis.plot_prices_with_dd(prices=strategy_nav)
rolling_sharpe = qis.plot_rolling_perf_stat(
    prices=strategy_nav, rolling_perf_stat=qis.RollingPerfStat.SHARPE,
    roll_periods=126, roll_freq='B',
)
heatmap = qis.plot_returns_heatmap(prices=strategy_nav)
yearly = qis.plot_periodic_returns_table(prices=navs, freq='YE')
monthly_returns = qis.to_returns(prices=strategy_nav, freq='ME', drop_first=True)
distribution = qis.plot_histogram(df=monthly_returns)
~~~

## Pandas extensions

`qs.extend_pandas()` attaches QuantStats functions as methods on pandas objects, as in
`returns.sharpe()`. qis has no pandas extension; call the qis functions with the series as an
argument.

## See also

- [Migrating from pyfolio-reloaded to qis](migrating_from_pyfolio.md)
- [Choosing between qis, bt, QuantStats, pyfolio-reloaded, and vectorbt](package_comparison.md)
- [The performance-statistic catalogue](performance_statistics.md)
- [Factsheet reference](factsheets.md)

## References

- QuantStats contributors. [Repository](https://github.com/ranaroussi/quantstats), including
  [stats.py](https://github.com/ranaroussi/quantstats/blob/main/quantstats/stats.py) and
  [reports.py](https://github.com/ranaroussi/quantstats/blob/main/quantstats/reports.py).
- Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet
  reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
