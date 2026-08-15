---
myst:
  html_meta:
    description: >-
      Compute risk-adjusted performance tables, rolling statistics, drawdowns, and explicitly
      labelled Sharpe-ratio conventions with qis.
---

# Performance analytics and Sharpe conventions

Use this layer when the input is an observed price or NAV history and the question is how return,
risk, drawdown, and risk-adjusted performance behaved. It is descriptive analytics: it does not
construct a portfolio, infer missing economic observations, or make a full-sample statistic
point-in-time safe for a backtest.

## Data and calculation contract

- **Input:** a `pandas.Series` or `pandas.DataFrame` of positive price or NAV levels with a
  `DatetimeIndex`; columns are assets or strategies.
- **Units:** prices may use any consistent positive scale. Returns and risk outputs are decimal
  fractions, so `0.10` means 10%. Sharpe and drawdown-to-volatility ratios are dimensionless.
- **Return convention:** `PerfParams.return_type` controls the return series used for volatility
  and higher moments. `ReturnTypes.LOG` is the default; `ReturnTypes.RELATIVE` requests simple
  returns. Compound p.a., arithmetic, and log return columns remain separately labelled in the
  full table.
- **Frequency and annualisation:** `freq_vol`, `freq_skewness`, `freq_drawdown`, and `freq_reg`
  are independent pandas frequencies. Annualised volatility multiplies the sampled standard
  deviation by the square root of the periods-per-year factor. The common monthly factor is 12.
- **NaNs:** table calculations respect heterogeneous start and end dates by evaluating each
  asset over its observed history. Price resampling and `qis.to_returns` forward-fill by default;
  direct callers can pass `ffill_nans=False` to `qis.to_returns` when a gap must remain missing.
  Rolling statistics remain NaN until a complete window is available. Running drawdown starts at
  the first valid price and forward-fills the drawdown state across later missing prices.

Do not use silent forward-filling to turn stale or genuinely low-frequency observations into
new information. Classify the data problem before choosing a sampling grid.

## Minimal offline example

This example uses the deterministic synthetic universe and needs no network or data extra.

```python
import qis
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe(
    start='2014-01-02', end='2025-12-31', apply_quirks=False
)
prices = universe.prices[['SEQ_US', 'SBD_TSY']]
params = qis.PerfParams(freq='ME', return_type=qis.ReturnTypes.LOG)

performance = qis.compute_ra_perf_table(prices=prices, perf_params=params)
sharpe_columns = [
    qis.PerfStat.SHARPE_RF0.to_str(),
    qis.PerfStat.SHARPE_ARITH.to_str(),
    qis.PerfStat.SHARPE_LOG_AN.to_str(),
]
sharpe_table = performance[sharpe_columns]

rolling_vol, rolling_label = qis.compute_rolling_perf_stat(
    prices=prices,
    rolling_perf_stat=qis.RollingPerfStat.VOL,
    roll_freq='ME',
    roll_periods=36,
)
drawdowns = qis.compute_rolling_drawdowns(prices=prices)
```

`performance` and `sharpe_table` are DataFrames indexed by asset. `rolling_vol` and `drawdowns`
are DataFrames aligned to time; `rolling_label` is a description of the selected window. A
drawdown of `-0.20` means the price is 20% below its running peak. A 36-month rolling volatility
observation describes only its trailing window, while a row in `performance` describes the
asset's available sample.

## The three Sharpe objects

Let `a` be periods per year, `r` simple periodic returns, `l = log(1 + r)`, and `sigma` the
corresponding annualised volatility. The three conventions answer different questions:

| Convention | Numerator and denominator | Full-table columns | Appropriate use |
|---|---|---|---|
| P.a. (`SharpeConvention.PA`) | compound annual return divided by annualised volatility | `SHARPE_RF0`, `SHARPE_EXCESS` | investor and factsheet reporting where the numerator should reconcile to CAGR |
| Arithmetic (`SharpeConvention.ARITHMETIC`) | `sqrt(a) * mean(r) / std(r)` | `SHARPE_ARITH`, `SHARPE_ARITH_EXCESS` | inference and additive return decompositions |
| Log (`SharpeConvention.LOG`) | `sqrt(a) * mean(l) / std(l)` | `SHARPE_LOG_AN`, `SHARPE_LOG_EXCESS` | time-additive analysis in log-return space |

The `_RF0` column assumes a zero risk-free rate. Excess columns need annualised risk-free-rate
data in `PerfParams.rates_data`; rates are applied with a one-period lag. With no rates series,
the excess and zero-rate objects collapse to the same input economics.

`PerfParams.sharpe_convention` selects which object a **regime-conditional** Sharpe display uses.
`compute_ra_perf_table` keeps the convention-specific columns side by side, so select the named
`PerfStat` rather than treating an unlabelled number as a universal Sharpe ratio. Rolling
`RollingPerfStat.SHARPE` has its own fixed log-return implementation and does not read a
`PerfParams` object.

## Rolling statistics and drawdowns

`compute_rolling_perf_stat` returns `(data, label)` and supports total return, p.a. return,
volatility, Sharpe, skewness, and EWMA volatility. `roll_periods` is a count on `roll_freq`, not a
number of calendar days: 36 with `ME` is a three-year monthly window, while 260 with `B` is a
one-year business-day window.

`compute_rolling_drawdowns` returns the path `price / running_peak - 1`. Use
`compute_drawdowns_stats_table` for episode dates, depths, and recovery durations. Drawdowns must
be computed from price or NAV levels, not from a return column mistaken for a level.

## Constraints and common failure modes

- Do not compare statistics computed at different sampling frequencies without labelling the
  difference; monthly volatility is not daily volatility merely rescaled.
- Specify simple versus log returns. They aggregate differently across assets and through time.
- Do not call an excess Sharpe a funded result unless `rates_data` represents the intended cash
  series and units.
- Short histories, constant series, or too few negative observations can produce NaN or
  degenerate risk ratios. Inspect observation counts and the underlying return series.
- Full-sample means and tables are descriptive. Recomputing them inside a backtest without a
  rolling point-in-time window introduces look-ahead.

## See also

- {doc}`Generated PerfParams API <api/generated/qis.PerfParams>`
- {doc}`Generated risk-adjusted table API <api/generated/qis.compute_ra_perf_table>`
- [Sharpe convention decision note](_included/sharpe_conventions.md)
- [Reporting-frequency convention](_included/reporting_frequencies.md)
- [Canonical performance examples](https://github.com/ArturSepp/QuantInvestStrats/tree/main/examples/perfstats)
