---
myst:
  html_meta:
    description: >-
      Run a deterministic qis portfolio backtest and performance analysis in under five minutes,
      using only the core package and offline synthetic data.
---

# Offline quickstart

This workflow needs only `pip install qis`: no network access, credentials, optional extras,
data files, or repository-local imports. It uses a seeded business-day panel, creates quarterly
targets over the instruments live at each decision date, and reports performance plus
benchmark-relative risk. A clean-wheel verification run took **6.5 seconds on Windows with
Python 3.12**, including imports and first-use compilation.

From a repository checkout, run:

```console
python examples/getting_started/offline_quickstart.py
```

With only the installed package, copy the complete script below into
`offline_quickstart.py` and run `python offline_quickstart.py`. This displayed code comes from
the runnable file; it is not maintained as a second copy.

```{literalinclude} ../examples/getting_started/offline_quickstart.py
:language: python
:linenos:
```

## What the result establishes

The output records the 2,087-row business-day input, the 33-by-3 quarterly target schedule,
final target and realised weights, a compact performance table, terminal NAV, and monthly
tracking error/information ratio against the synthetic 60/40 benchmark. The current deterministic
terminal NAV is `120.1104`; the repository test checks it and the benchmark-relative output.

The weight schedule is point-in-time and live-universe-aware: `SEQ_EM` receives no allocation
before it has a price. A target decided at *t* sets the units held over *[t, t+1]*. Between
quarterly rebalances, `qis` holds units rather than fixed weights, so realised weights drift with
prices. Transaction cost `0.0010` means 10 basis points of traded notional.

The performance table samples **simple monthly returns** and selects the arithmetic, zero-rate
Sharpe convention explicitly. Tracking error and information ratio use the same monthly simple
strategy-minus-benchmark returns. This alignment matters when comparing the numbers with another
tool or report.

## What to change first

- **Statistic set:** edit `performance_columns`, choosing members of `qis.PerfStat`.
- **Return and Sharpe convention:** edit `return_type` and `sharpe_convention` in `perf_params`;
  excess-return statistics also need `rates_data`.
- **Rebalance cadence:** change `REBALANCING_FREQ`; the generated weight DataFrame owns those
  dates when it enters the backtest.
- **Transaction cost:** change `TRANSACTION_COST` in fractional units of traded notional.
- **Benchmark:** replace `benchmark_nav`, then form the return difference on the same frequency
  and convention as the strategy.
- **Reporting frequency:** use the matching `reporting_frequency` when moving to a factsheet; it
  recalibrates report windows together rather than changing only a label.

## Next reporting step

The quickstart deliberately writes no plot, PDF, or factsheet. Continue with
[factsheets and reporting](factsheets_and_reporting.md) to inspect figures in memory or save a
report explicitly. The [portfolio backtesting guide](portfolio_backtesting.md) explains the
held-unit and transaction-cost contracts in more detail; the [API reference](api/index.rst)
lists the complete exported surface.
