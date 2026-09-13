---
myst:
  html_meta:
    description: >-
      Backtest target weights with held-unit drift, implementation lags, traded-notional costs,
      and explicit cash balances using qis.
---

# Portfolio backtesting: targets, held units, and implementation

Use `qis.backtest_model_portfolio` when you have price histories and a target allocation and need
the realised NAV, positions, weights, and trading costs. It is a holdings backtest, not a weighted
average of contemporaneous returns: a rebalance converts target weights into units, then those
units remain fixed until the next rebalance while realised weights drift with prices.

## Data and calculation contract

- **Prices:** a `pandas.DataFrame` of levels, unique non-missing dates in a `DatetimeIndex`, assets
  in columns. The backtester orders price rows chronologically on a local copy before constructing
  state; duplicate timestamps are rejected because their state-transition order is ambiguous.
  Price units can differ by asset because the engine holds units. Columns must be unique.
- **Target weights:** a dictionary, Series, list, array, or date-by-asset DataFrame. Named inputs
  align by ticker; list and array inputs are positional. A fixed vector is reapplied at
  `rebalancing_freq`; a DataFrame supplies its own decision dates and ignores that frequency.
- **Time convention:** a decision at *t* must fund the return over *[t, t+1]*. For a point-in-time
  DataFrame of signals, `weight_implementation_lag=1` trades on the next observation of the price
  index. The lag counts observations, not calendar days. The default zero lag may trade on the
  first price date at or after the weight timestamp and is appropriate only when that execution
  assumption is intentional.
- **Returns and annualisation:** prices produce simple holding-period P&L. `funding_rate`,
  `management_fee`, and `instruments_carry` are annualised decimal rates converted to the price
  grid. Dated funding and carry inputs are ordered chronologically before that alignment. A cash
  residual earns `funding_rate`, which defaults to zero.
- **Trading costs:** `rebalancing_costs` is a decimal fraction of absolute traded notional;
  `0.0010` is 10 bp. A scalar applies everywhere, a ticker-indexed Series varies by asset, and a
  date-by-ticker DataFrame is forward-filled through time. Costs are read on the trade date.
- **Missing data:** a target for an unpriced asset is not traded and remains cash. An internal NaN
  is not repaired: units remain held, the leg drops out of NAV on that date, and a warning is
  raised. Forward-fill prices before the call only when carrying the last mark is the stated data
  policy.

## Minimal offline example

The target rows below are decisions on observed business dates. A one-observation lag prevents
the same close from being used both to decide and execute.

```python
import pandas as pd
import qis
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe(
    start='2020-01-02', end='2023-12-29', apply_quirks=False
)
prices = universe.prices[['SEQ_US', 'SBD_TSY']]
decision_dates = prices.index[[0, 260, 520, 780]]
targets = pd.DataFrame(
    [[0.60, 0.40], [0.50, 0.50], [0.70, 0.30], [0.60, 0.40]],
    index=decision_dates,
    columns=prices.columns,
)

portfolio = qis.backtest_model_portfolio(
    prices=prices,
    weights=targets,
    weight_implementation_lag=1,
    rebalancing_costs=0.0010,
    initial_nav=100.0,
    ticker='Lagged 60/40 policy',
)

nav = portfolio.get_portfolio_nav()
realised_weights = portfolio.weights
held_units = portfolio.units
costs_by_asset = portfolio.realized_costs
```

`portfolio` is a `qis.PortfolioData`. `nav` is a Series; the other three outputs are DataFrames
on the price grid. `realised_weights` equals a target immediately after an executable rebalance
apart from costs and unavailable legs, then drifts. `held_units` changes only when the strategy
trades. `costs_by_asset` is in NAV currency, not basis points.

## Turnover: executed contracts and the reporting denominator

Turnover has two independent choices: which trades form the numerator and what capital base forms
the denominator. Cash portfolios value changes in executed units at current prices. Futures and
other derivatives must instead value changes in executed contracts at their full contract
notionals, including multipliers and currency conversion; a normalized return index is not a
contract value.

For factsheets, transaction-cost comparisons, and investor-level reporting, divide that executed
notional by NAV with `TurnoverComputationType.EXECUTED_NOTIONAL_NAV`. This retains leverage. Within
a portfolio running at 2x gross exposure, `EXECUTED_NOTIONAL_GROSS` divides the same traded
notional by approximately `2 × NAV`, so its result is half the NAV-normalized result. That
convention is useful as a secondary book-replacement or capacity diagnostic, not as the primary
investor turnover statistic.

See [Two-sided turnover conventions](turnover_conventions.md) for the formulas and the Yahoo
60/40 comparison at 1x and 2x leverage.

## Constraints and failure modes

- A static vector does not dynamically redistribute an unavailable asset's weight. The residual
  is cash. Build an availability-aware schedule explicitly when that is the investment rule.
- Two decision rows cannot resolve to the same traded date. A weight schedule denser than the
  price grid raises instead of silently applying later rows to the wrong dates.
- A target row is an allocation instruction, not proof that every leg traded. Inspect realised
  weights, units, cash residual (`1 - weights.sum(axis=1)` for an unlevered long-only book), and
  warnings.
- Costs apply to changes in units times current prices: traded notional, not target gross exposure
  and not end-of-period NAV. A time-varying cost panel has no effect before its first dated value.
- Fixed units between rebalances are the source of drift. Replacing this with a weighted average
  of returns implicitly rebalances every period and answers a different question.
- Lagging only the output or shifting returns after the backtest does not repair look-ahead. Lag
  the target's execution through `weight_implementation_lag`.

## See also

- {doc}`Generated backtester API <api/generated/qis.backtest_model_portfolio>`
- {doc}`Generated PortfolioData API <api/generated/qis.PortfolioData>`
- [Reporting-frequency convention](_included/reporting_frequencies.md)
- [Two-sided turnover conventions](turnover_conventions.md)
- [Canonical portfolio examples](https://github.com/ArturSepp/QuantInvestStrats/tree/main/examples/portfolios)
