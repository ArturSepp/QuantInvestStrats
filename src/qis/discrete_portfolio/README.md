# Discrete portfolio replay

`qis.discrete_portfolio` is the event-driven trading pipeline in QIS. It is intended for
strategies that decide whether to submit an order after each observed market event and need an
explicit order/fill history. The same deterministic replay logic can sit behind research and a
live adapter; QIS portfolio analytics are attached afterward through `PortfolioData`.

Use this module when order timing, units, cash, transaction costs, or unfilled orders matter.
For strategies defined primarily by periodic target weights, the scheduled portfolio backtester
under `qis.portfolio` remains the simpler abstraction.

## Event contract

Every timestamp is processed in this order:

1. Execute orders decided at the preceding observation using the current observed price.
2. Apply fill notionals and transaction costs to cash and signed units.
3. Mark all holdings and create an immutable `DiscretePortfolioState`.
4. Pass only the current observed price row and post-fill state to the strategy.
5. Queue the strategy's new orders for the next observation.

An order decided at time *t* therefore cannot fill before *t+1*. An order created on the final
observation is retained as `UNFILLED_END_OF_DATA`. If its next price is missing, non-finite, or
non-positive, it is retained as `UNFILLED_MISSING_PRICE`; the engine does not wait for a later,
potentially more favourable bar.

The engine holds units between events, not constant weights. Weights in a state are realised
weights computed from the current marks.

## Package structure

| Module | Responsibility |
|---|---|
| `types.py` | Immutable orders, trades and states; strategy and execution protocols; result container |
| `execution.py` | Deterministic next-observation full fills with optional fees and slippage |
| `backtester.py` | Deployment-oriented event loop and raw order, trade, cash and state output |
| `portfolio_adapter.py` | Conversion to `PortfolioData` with independent accounting reconciliation |

The separation is deliberate. `replay_discrete_portfolio` does not construct a reporting object,
while `backtest_discrete_portfolio` runs the same replay and attaches one. A live system can reuse
the event types and execution boundary without making reporting part of its trading loop.

## Minimal strategy

A strategy implements `on_bar(timestamp, prices, state)` and returns a sequence of `Order`
objects. It can retain its own signal history, but it should derive each decision only from the
values delivered up to the current callback.

```python
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qis.discrete_portfolio import DiscretePortfolioState, Order


@dataclass
class MovingAverageCross:
    ticker: str
    fast: int = 12
    slow: int = 36
    history: list[float] = field(default_factory=list)
    previous_signal: bool | None = None
    order_number: int = 0

    def on_bar(
            self,
            timestamp: pd.Timestamp,
            prices: pd.Series,
            state: DiscretePortfolioState,
    ) -> list[Order]:
        price = float(prices[self.ticker])
        if not np.isfinite(price) or price <= 0.0:
            return []
        self.history.append(price)
        if len(self.history) < self.slow:
            return []

        signal = np.mean(self.history[-self.fast:]) > np.mean(self.history[-self.slow:])
        if signal == self.previous_signal:
            return []
        self.previous_signal = signal

        target_units = float(np.floor(state.nav / price)) if signal else 0.0
        quantity = target_units - float(state.units[self.ticker])
        if quantity == 0.0:
            return []
        self.order_number += 1
        return [
            Order(
                order_id=f"signal-{self.order_number:04d}",
                decision_time=timestamp,
                ticker=self.ticker,
                quantity=quantity,
                reason="long" if signal else "flat",
            )
        ]
```

The `prices` row and every Series in `state` are read-only snapshots. `quantity` is signed:
positive means buy and negative means sell. Order identifiers must be unique within a replay.

## Running a replay

Prices are a timestamp-by-ticker `DataFrame` with a unique, increasing `DatetimeIndex`.

```python
from qis.discrete_portfolio import FullFillExecution, backtest_discrete_portfolio

result = backtest_discrete_portfolio(
    prices=prices,
    strategy=MovingAverageCross(ticker="SPY"),
    initial_cash=100_000.0,
    execution_model=FullFillExecution(
        transaction_cost_rate=0.0001,
        slippage_rate=0.0002,
    ),
    ticker="SPY intraday momentum",
)
```

Rates are fractions of notional: `0.0001` is 1 basis point. For signed quantity *q*, reference
price *p*, slippage rate *s*, and transaction-cost rate *c*, the default model computes:

```text
executed_price  = p * (1 + sign(q) * s)
notional        = q * executed_price
transaction_cost = abs(notional) * c
```

Slippage is adverse on buys and sells and is embedded in the execution price. Transaction cost
is a separate cash charge. Version one requires a complete fill at the next observation; partial
fills, cancellations, market impact, and multi-bar latency require a deliberate extension of the
execution contract.

For a raw result without reporting conversion, call `replay_discrete_portfolio` with the same
first four arguments. Later, call `to_portfolio_data(raw_result, ticker=...)` if reporting is
needed.

## Result and analytics

`DiscreteBacktestResult` exposes both audit-level trading output and reporting output:

| Field | Contents |
|---|---|
| `order_ledger` | Every submitted order, its reason, final status, fill time and status reason |
| `trade_ledger` | Every full fill with reference/executed price, signed notional, fee and slippage |
| `cash` | Post-fill cash at every market observation |
| `states` | Immutable post-fill units, marks, position values, NAV and realised weights |
| `portfolio_data` | QIS-aligned NAV, holdings and contribution panels, or `None` after raw replay |

Useful ledger analyses include turnover from absolute trade notionals, cost attribution from
`transaction_cost` and `slippage`, fill latency from `fill_time - decision_time`, order counts by
`reason`, and unfilled rates by `status`.

The reporting adapter constructs these aligned `PortfolioData` fields:

- `nav`: marked portfolio value at every event timestamp;
- `units`: signed units held after fills;
- `weights`: realised weights after fills and marking;
- `prices`: current valuation marks, carrying the last valid mark only for an existing holding;
- `realized_costs`: transaction costs by timestamp and instrument;
- `instrument_pnl`: contribution returns by timestamp and instrument;
- `is_rebalancing`: whether at least one fill occurred at the timestamp.

For instrument *i* over observation interval *t-1* to *t*, the adapter independently computes
currency P&L as:

```text
units[i, t-1] * (mark[i, t] - mark[i, t-1])
+ fill_quantity[i, t] * (mark[i, t] - executed_price[i, t])
- transaction_cost[i, t]
```

Dividing by prior NAV produces `instrument_pnl`. The adapter verifies that summed instrument
contributions equal the marked NAV return and that ledger fills equal changes in units. It raises
on a mismatch instead of passing inconsistent data into a factsheet.

## Timestamped NAV and factsheets

`portfolio_data.nav` is already on the event timestamp grid. The same path can be reconstructed
from the contribution returns, which is useful when combining multiple strategy outputs:

```python
portfolio_data = result.portfolio_data
strategy_returns = portfolio_data.instrument_pnl.sum(axis=1)
timestamped_nav = portfolio_data.nav.iloc[0] * (1.0 + strategy_returns).cumprod()
timestamped_nav.name = portfolio_data.nav.name
```

Keep execution and signals on the native event grid, but use business-day closing marks for the
visual report. A second aligned panel makes the signed end-of-day position explicit:

```python
import matplotlib.pyplot as plt
import qis

reporting_navs = timestamped_navs.resample("B").last().dropna(how="any")
reporting_position = portfolio_data.units["SPY"].resample("B").last()
reporting_position = reporting_position.reindex(reporting_navs.index)

figure, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
qis.plot_time_series(
    df=reporting_navs,
    x_date_freq="B",
    date_format="%d-%b",
    legend_stats=qis.LegendStats.NONE,
    ylabel="NAV",
    title="Momentum-event strategy: B reporting (5m execution)",
    ax=axes[0],
)
qis.plot_time_series(
    df=reporting_position.rename("SPY position"),
    x_date_freq="B",
    date_format="%d-%b",
    legend_stats=qis.LegendStats.NONE,
    ylabel="Signed units",
    title="End-of-day position size",
    ax=axes[1],
)
```

Use the same aligned business-day NAVs for the performance factsheet and state the reporting
frequency explicitly:

```python
import qis
from qis.portfolio.reports.multi_assets_factsheet import generate_multi_asset_factsheet

figure = generate_multi_asset_factsheet(
    prices=reporting_navs,
    benchmark="SPY buy-and-hold",
    perf_params=qis.PerfParams(freq="B"),
    regime_classifier=qis.BenchmarkReturnsQuantilesRegime(freq="B"),
    heatmap_freq="W-FRI",
    factsheet_name="SPY intraday momentum-event strategy",
)
```

This daily conversion is a reporting choice only. It does not move decisions or fills, and it
avoids silently applying a daily annualisation convention directly to 5-minute returns.

## Complete yfinance example

The repository example downloads adjusted SPY 5-minute closes, runs a moving-average momentum
strategy, prints the trade ledger and reconciliation error, plots business-day strategy and
buy-and-hold NAV with end-of-day position size, and opens a QIS multi-asset factsheet:

```bash
pip install -e ".[data]"
python -m examples.discrete_portfolio.discrete_trend_backtest
```

The default uses one month of 5-minute bars. To use 1-minute bars, set `INTERVAL = "1m"` and a
short yfinance-supported period such as `PERIOD = "5d"`.

## Operational boundaries

- The replay is deterministic for a fixed price panel, strategy, and execution model.
- There is no synthetic terminal liquidation; final positions remain marked and final orders
  remain unfilled.
- Missing current observations are visible to the strategy. Existing holdings may use their last
  finite positive mark for valuation, but a pending order cannot execute on a missing price.
- The built-in execution model is a research baseline, not an exchange simulator.
- A live adapter must provide durable order identifiers, venue reconciliation, recovery after
  restart, market-calendar handling, and a deliberate policy for partial fills and cancellations.
- Do not reuse a full-sample statistic or any value revised after time *t* inside `on_bar`; the
  next-observation fill rule cannot compensate for a forward-looking signal.

Run the focused tests with:

```bash
pytest src/qis/discrete_portfolio/tests/discrete_backtester_test.py -q
```
