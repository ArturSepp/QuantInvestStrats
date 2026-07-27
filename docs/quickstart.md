# Quickstart

Everything on this page runs offline. `qis.datasets.synthetic` draws a seeded ten-instrument
panel, so no network, data file or vendor licence is needed to reproduce any of it.

## A panel to work with

```python
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe()
prices = universe.prices                      # business-day panel, 10 instruments
benchmark_prices = universe.benchmark_prices  # 60/40 blend, no nans
group_data = universe.group_data              # asset class per ticker
```

The panel is deliberately imperfect: one instrument lists late, one is delisted, one reports only
at month ends, one carries appraisal smoothing, one has fat tails and one repeats stale prices.
Analytics that only work on clean data will show it here.

## Performance statistics

```python
import qis

perf_params = qis.PerfParams(freq='ME')
ra_perf = qis.compute_ra_perf_table(prices=prices, perf_params=perf_params)
print(ra_perf)
```

The return convention is stated, never implied — `qis.to_returns(..., is_log_returns=...)` takes
it explicitly, and every Sharpe object is labelled with the convention it uses. See
[Sharpe conventions](_included/sharpe_conventions.md).

## A backtest

```python
weights = {ticker: 0.1 for ticker in prices.columns}
portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                              weights=weights,
                                              rebalancing_freq='QE',
                                              rebalancing_costs=10)  # annualised, in bp
print(portfolio_data.get_navs().tail())
```

Rebalancing weights are drift-adjusted, so reported turnover is the turnover actually traded.

## A factsheet

```python
qis.factsheet(prices,
              benchmark_prices=benchmark_prices,
              reporting_frequency='monthly',
              file_name='universe')
```

One call picks the report archetype from the input type and calibrates every window, regression,
regime and annualisation to the requested reporting frequency. See
[reporting frequencies](_included/reporting_frequencies.md) for what that recalibrates, and the
[gallery](_included/gallery.md) for what each report looks like.

## Where to go next

- [Gallery](_included/gallery.md) — the four factsheet archetypes, each from one call
- [API reference](api/index.rst) — every exported symbol
