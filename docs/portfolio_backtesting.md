---
myst:
  html_meta:
    description: >-
      Explain portfolio backtesting through target weights, held-unit drift, execution timing,
      cash balances, and traded-notional costs, with reproducible qis examples.
---

# Portfolio backtesting: targets, held units, and implementation

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-15](https://github.com/ArturSepp/QuantInvestStrats/commit/8a10dc6b72ed8db593e42d5eafe6d3e4b23419e6)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

A portfolio backtest applies an allocation rule to historical prices and accounts for the resulting
holdings, cash, costs, and net asset value (NAV). In qis, a rebalance converts target weights into
units. Units remain fixed between trades, while their values and realised capital weights change
with prices.

## Overview

Use `qis.backtest_model_portfolio` when price histories and target allocations are available and
the required outputs are NAV, executed positions, realised weights, and transaction costs. The
function returns a `PortfolioData` object for subsequent attribution and reporting.

Target weights describe the intended allocation at an execution time. Realised weights describe
the holdings after price movement and accounting adjustments. Keeping that distinction makes
drift, implementation lags, unavailable assets, and costs visible.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Simple price returns; NAV accumulates held units and cash |
| Sampling grid | The native price index |
| Annualisation | Annual funding, fee and carry rates accrue ACT/365 over elapsed days |
| Mean adjustment | Not applicable |
| Timing | A dated target executes at the first price at or after its date, plus `weight_implementation_lag` observations; units are held over $(t,t+1]$ |
| Output units | NAV and costs in currency; weights as fractions of NAV |
| qis default | `rebalancing_freq='QE'`, no lag, no costs, `initial_nav=100` |

| Symbol or input | Meaning | Units and timing |
|---|---|---|
| $P_{i,t}$ / `prices` | Price of asset $i$ | Monetary value per asset unit; unique non-missing dates in a `DatetimeIndex`, unique asset columns |
| $w^*_{i,t}$ / `weights` | Target capital fraction | Signed decimal weights; named inputs align by ticker, arrays/lists are positional |
| $u_{i,t}$ | Units held after execution on date $t$ | Shares or model units |
| $C_t$ | Cash balance after execution and costs | NAV currency; negative cash can represent funding |
| $V_t$ | Portfolio NAV | Same currency as marked holdings and cash |
| $\kappa_{i,t}$ / `rebalancing_costs` | Cost per absolute unit of traded notional | Decimal rate; `0.0010` means 10 bp |
| `weight_implementation_lag` | Delay for a dated target schedule | Non-negative count of price-index observations; `None` means zero |

### Data and calculation contract

Prices are a `pandas.DataFrame`. The backtester rejects missing or duplicate timestamps, then
orders price rows chronologically on a local copy before constructing state; it does not mutate
the caller's frame. A fixed vector, dictionary, or Series is reapplied at `rebalancing_freq`. A
date-by-asset DataFrame supplies its own decision dates and ignores that frequency; its
implementation lag maps decisions onto the price grid. A lag on a fixed vector does not create a
delayed signal schedule: use dated targets when timing matters.

`funding_rate`, `management_fee`, and `instruments_carry` are annualised decimal inputs converted
to the price grid. Dated funding and carry inputs are ordered chronologically before alignment.
The residual cash balance earns the funding rate, which defaults to zero. Returns arise from
simple holding-period P&L; do not substitute asset log returns into the cash accounting.

Cost inputs may be a scalar, ticker-indexed Series, or date-by-ticker DataFrame. A dated cost
panel is forward-filled and read on the trade date, with zero cost before its first dated value.

The equations below describe priced cash-security holdings with the default NAV-based trade
sizing. The optional `constant_trade_level` mode sizes targets against that supplied amount
instead of current NAV. Contract multipliers, economic notionals, and real execution quantities
must be supplied consistently when constructing derivative reporting objects.

## Methodology

### Held-unit accounting and drift

With no external flows, costs, fees, funding, or carry over an interval, wealth evolves as:

$$
V_t=\sum_i u_{i,t}P_{i,t}+C_t,
\qquad
V_{t+1}-V_t=\sum_i u_{i,t}(P_{i,t+1}-P_{i,t}).
$$

The units in the second expression are the positions established at $t$, held over $[t,t+1]$.
Their realised weights at the next mark are:

$$
w_{i,t+1}=\frac{u_{i,t}P_{i,t+1}}{V_{t+1}}.
$$

Those weights drift even when no new target has been supplied. Averaging each period's asset
returns with the original fixed target weights instead describes repeated rebalancing.

### Rebalancing and costs

Let $V_t^-$ be wealth marked at the current prices, after applicable funding, fees, and carry
but before the trade. For an executable target in the default sizing mode:

$$
u_{i,t}=\frac{V_t^-w^*_{i,t}}{P_{i,t}},
\qquad
K_t=\sum_i \kappa_{i,t}P_{i,t}
\left|u_{i,t}-u_{i,t^-}\right|.
$$

Here $u_{i,t^-}$ is the pre-trade holding and $K_t$ is the transaction cost. The cash update
pays for position changes and deducts $K_t$. Positions are sized from pre-cost wealth, so
post-cost realised weights need not equal target weights exactly.

Opening positions are trades too: the first-date cost uses the change from zero units.
Thus `initial_nav=100` with a fully invested opening target and 10 bp costs produces an
initial recorded NAV of 99.9, when prices are available and other charges are absent.
`realized_costs` records currency amounts, not basis points or NAV-normalised turnover.

### Decision and execution dates

A decision at $t$ must use information available before its assumed execution and fund subsequent
returns. For a dated schedule, qis locates the first price observation at or after the decision
timestamp, then moves forward by `weight_implementation_lag` observations.

Zero lag permits execution at that first mark. One lag trades at the following observation;
it does not create a next-open price or a model of intraday fills. Two decision rows may not map
to the same traded date. Lagging the final NAV series cannot repair look-ahead embedded in the
target construction or execution assumption.

### Unpriced assets and cash

A target for an unpriced asset cannot be executed and leaves that allocation in cash. A static
vector does not redistribute it across available assets. Use an explicit availability-aware
schedule if redistribution is part of the investment rule.

An internal price NaN is not repaired. Units remain held, but the unmarked leg drops out of the
NAV sum on that date and the backtester warns. This is not an estimate of its economic value.
Leading or trailing missing data also require an explicit entry/exit and valuation policy.
Forward-fill only when carrying the last mark is the intended policy.

<a id="minimal-offline-example"></a>

## Worked example

Start with NAV 100, invest equally in two assets priced at 100, and hold 0.5 units of each.
When the first price rises to 110, NAV is 105 and its realised weight is $55/105$, about 52.38%.
When both prices are 110, NAV is 110. These are fixed arithmetic inputs, not sampled market data.
An explicit opening target schedules exactly one trade and no later rebalancing.

```python
import numpy as np
import pandas as pd
import qis

marks = pd.DataFrame(
    {'Asset A': [100.0, 110.0, 110.0], 'Asset B': [100.0, 100.0, 110.0]},
    index=pd.bdate_range('2024-01-02', periods=3),
)
opening_target = pd.DataFrame(
    [[0.5, 0.5]], index=marks.index[:1], columns=marks.columns
)
held = qis.backtest_model_portfolio(
    prices=marks, weights=opening_target, initial_nav=100.0, rebalancing_costs=0.0,
)
np.testing.assert_allclose(held.get_portfolio_nav(), [100.0, 105.0, 110.0])
np.testing.assert_allclose(held.units, 0.5)
assert abs(held.weights.iloc[1, 0] - 55.0 / 105.0) < 1e-12

with_costs = qis.backtest_model_portfolio(
    prices=marks, weights=opening_target, initial_nav=100.0, rebalancing_costs=0.001,
)
assert abs(with_costs.realized_costs.iloc[0].sum() - 0.1) < 1e-12
assert abs(with_costs.get_portfolio_nav().iloc[0] - 99.9) < 1e-12
```

The second run isolates the opening charge. No price forecast or future observation is used to
decide the initial allocation.

## Implementation in qis

For a longer example, the frozen synthetic universe supplies complete prices. The dated targets
below are allocation decisions on observed business dates; a one-observation implementation lag
separates each decision date from its execution date.

```python
import pandas as pd
import qis
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe(
    start='2020-01-02', end='2023-12-29', seed=20260725, apply_quirks=False
)
prices = universe.prices[['SEQ_US', 'SBD_TSY']]
decision_dates = prices.index[[0, 260, 520, 780]]
targets = pd.DataFrame(
    [[0.60, 0.40], [0.50, 0.50], [0.70, 0.30], [0.60, 0.40]],
    index=decision_dates,
    columns=prices.columns,
)
portfolio = qis.backtest_model_portfolio(
    prices=prices, weights=targets, weight_implementation_lag=1,
    rebalancing_costs=0.0010, initial_nav=100.0, ticker='Lagged allocation policy',
)
nav = portfolio.get_portfolio_nav()
realised_weights = portfolio.weights
held_units = portfolio.units
costs_by_asset = portfolio.realized_costs
```

`nav` is a Series; the other outputs are DataFrames on the price grid. Inspect realised weights,
held units, and costs to understand whether each target executed. With consistent marked holdings,
the residual capital share is $1-\sum_i w_{i,t}$; costs and leverage can make it negative.

The [backtester source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/backtester.py)
owns execution and accounting. [PortfolioData](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/portfolio_data.py)
owns the resulting reporting container. Full examples are in the
[portfolio example directory](https://github.com/ArturSepp/QuantInvestStrats/tree/main/examples/portfolios).

### Turnover: executed contracts and the reporting denominator

Turnover has two independent choices: traded amount and capital denominator. Cash securities
use executed unit changes valued at current prices. Futures use actual contract changes valued
at full contract notionals, including multipliers and FX conversion. A normalised return index
is not a contract value, and model units from such an index are not automatically actual contracts.

For investor reporting and comparison with costs expressed per NAV, use
`TurnoverComputationType.EXECUTED_NOTIONAL_NAV`. This retains leverage.
`EXECUTED_NOTIONAL_GROSS` divides the same traded notional by current gross exposure; at
2x gross exposure it is approximately half the NAV-normalised number. It is a useful book
replacement diagnostic when that denominator is explicitly intended.

`VOLATILITY_NORMALIZED_WEIGHTS` instead multiplies absolute target-weight changes by annualised
volatility. It excludes realised drift and execution effects. Its theoretical basis is
[Sepp and Lucic (2026), Definition 4.5](https://arxiv.org/html/2607.19497v1#S4.SS4).
The supplied volatility panel must align exactly with the target schedule. The
[turnover article](turnover_conventions.md) provides all four formulas, an offline arithmetic
example, and the separate Yahoo SPY/TLT illustration.

<a id="constraints-and-failure-modes"></a>

## Interpretation and limitations

- A target row records an allocation instruction, not proof that every asset traded.
  Inspect units, cash, realised weights, warnings, and missing-price policy.
- Negative or non-integral implementation lags on dated schedules are rejected. A lag
  counts observations, not elapsed days, and cannot correct a forward-looking signal.
- Opening costs and later costs follow the same traded-notional rule. A first-row turnover
  NaN does not imply that the opening trade was free.
- Portfolio prices, holdings, funding, and costs must use compatible units and currencies.
  A total-return index does not itself encode contract multipliers, fills, or margin accounting.
- The engine does not model partial fills, lot rounding, market impact, or an exchange order
  book merely because target weights are supplied.
- Backtest results describe the chosen sample and assumptions. They do not establish that the
  allocation could have been selected without hindsight.

## See also

- [Two-sided turnover conventions](turnover_conventions.md)
- [Portfolio breadth](portfolio_breadth.md)
- [Performance analytics](performance_analytics_and_sharpe.md)
- [Brinson attribution](brinson_attribution.md)
- {doc}`Backtester API <api/generated/qis.backtest_model_portfolio>`
- {doc}`PortfolioData API <api/generated/qis.PortfolioData>`
- [Reporting-frequency note](_included/reporting_frequencies.md) and
  [packaged source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/docs/reporting_frequencies.md)

## References

1. Sepp, A., and Lucic, V. (2026). The Science and Practice of Trend-Following Systems. Working paper. [arXiv:2607.19497](https://arxiv.org/abs/2607.19497). Definition 4.5 and equation 4.15 concern volatility-normalised turnover.
2. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
