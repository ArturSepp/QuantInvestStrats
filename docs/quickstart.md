---
myst:
  html_meta:
    description: >-
      Run a reproducible qis chart, portfolio backtest, performance table and
      benchmark-relative analysis using core-only offline synthetic data.
---

# Offline quickstart

*[author / affiliation / date — placeholder]*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Start with an [installed core package](install.md). The examples below use the frozen synthetic
fixture with default seed 20260725 and a fixed 2 January 2018–31 December 2025 sample.
Their outputs demonstrate software behaviour under those inputs, not historical market returns.

## First chart

The first example draws a quarterly rebalanced 60/40 portfolio of synthetic US equities and
government bonds. It charges 10 basis points per unit of traded notional and suppresses
performance labels so the chart presents the NAV path without implying a Sharpe convention.
It displays a Matplotlib figure and writes no report file. In a headless session, use the
figure object for saving or embedding instead of expecting a desktop window.

The first backtest in a fresh environment may pause while Numba compiles the portfolio kernel.
The script prints a notice before that work begins; runtime depends on the environment.

[Open the complete first-chart script](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/getting_started/first_chart.py).
The rendered site includes it below. In a Markdown viewer that shows the include directive as
text, follow the source link.

```{literalinclude} ../examples/getting_started/first_chart.py
:language: python
:linenos:
```

## Complete checked workflow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArturSepp/QuantInvestStrats/blob/main/notebooks/offline_quickstart_colab.ipynb)

After installation, the calculation needs no network, credentials, optional extras or local
data files. It builds a target schedule over the instruments priced at each decision date,
runs the portfolio, then prints performance and benchmark-relative risk.

The optional Colab notebook first installs qis from PyPI, which requires network access.
It prints the installed version/import path and runs a cell checked against the authoritative
script. The committed notebook has no execution outputs. The
[stable documentation](https://quantinveststrats.readthedocs.io/en/stable/quickstart.html)
provides a release-oriented entry point; the latest/source page may contain unreleased changes.

From a repository checkout, run:

~~~console
python examples/getting_started/offline_quickstart.py
~~~

With only the installed package, save the
[complete workflow source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/getting_started/offline_quickstart.py)
as `offline_quickstart.py` and run `python offline_quickstart.py`. The site displays that
same file below, avoiding a second code copy.

```{literalinclude} ../examples/getting_started/offline_quickstart.py
:language: python
:linenos:
```

## What the result establishes

The checked workflow uses synthetic US equities, government bonds and emerging-market equities.
The last instrument, `SEQ_EM`, starts late. The schedule redistributes the target allocation
across priced instruments until it becomes available.

| Output | Fixed-fixture result or convention |
|---|---|
| Price panel | 2,087 business-day rows, three instruments. |
| Target schedule | 33 quarterly decision dates, three weights per date. |
| Terminal NAV | 119.9866, rounded to four decimals; initial capital is 100. |
| Tracking error | 0.0299 annualised, about 2.99%, rounded to four decimals. |
| Information ratio | −0.2899, annualised and rounded to four decimals. |
| Benchmark | The fixture's synthetic 60/40 reference, `SBM_6040`. |

The [quickstart regression test](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/tests/test_quickstart.py)
checks the schedule, output and terminal NAV against the current source. These values are a
reproducibility reference for this fixture and cost setup, not a forecast or an assertion that
every published release produces identical results.

A target decided at time $t$ sets the units held over $[t,t+1]$, with zero implementation lag.
Between quarterly rebalances, units remain held while weights drift. Cost `0.0010` is 10 basis
points of traded notional; it is not a flat deduction of 10 basis points from every day's NAV.

The performance table uses **simple monthly returns** and the arithmetic, zero-rate Sharpe
convention explicitly. Its headline annual return is compounded. Tracking error and information
ratio use monthly simple strategy-minus-benchmark returns, sample standard deviation and
12 periods per year. For exact definitions, see [performance and Sharpe conventions](
performance_analytics_and_sharpe.md) and [tracking error](tracking_error_and_risk.md).

## What to change first

- **Statistic set:** edit `performance_columns`, using `qis.PerfStat` labels.
- **Return and Sharpe convention:** edit `return_type` and `sharpe_convention` in `perf_params`;
  excess-return statistics also require a rate series.
- **Rebalance cadence:** change `REBALANCING_FREQ`. The weight DataFrame then owns the decision
  dates passed to the backtest.
- **Transaction costs:** change `TRANSACTION_COST` in fractional units of traded notional.
- **Benchmark:** replace `benchmark_nav` and align its return convention, observation grid and
  common sample with the strategy.
- **Reporting cadence:** use the corresponding `reporting_frequency` when creating a factsheet,
  and check its [reporting-grid conventions](factsheets_and_reporting.md#methodology).

## Next reporting step

The complete checked workflow prints its results and writes no chart, PDF or factsheet.
Continue with [factsheets and reporting](factsheets_and_reporting.md) to create report figures.
The [portfolio backtesting guide](portfolio_backtesting.md) explains held units and costs,
while the [API reference](api/index.rst) and
[public API record](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/api.py)
locate the exported functions.

## References

- qis contributors. [First chart](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/getting_started/first_chart.py),
  [checked workflow](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/getting_started/offline_quickstart.py),
  and [frozen synthetic fixture](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/datasets/synthetic.py).
- Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet
  reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
