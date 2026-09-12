# Two-sided turnover conventions

Turnover answers two separate questions:

1. What changed: requested target weights or actually executed units?
2. What scales the traded amount: portfolio NAV or current gross exposure?

QIS keeps these choices explicit with `TurnoverComputationType`. All supported modes are
**two-sided**: purchases and sales are added in absolute value without multiplying their sum by
one half.

The calculation engine is `qis.compute_turnover`. Factsheets and `PortfolioData.get_turnover`
delegate to it before applying frequency resampling, grouping, or rolling sums. The complete
Yahoo example compares a 100% funded 60/40 portfolio with the same portfolio at 2x leverage:
[`examples/perfstats/turnover_conventions.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/perfstats/turnover_conventions.py).

## Inputs

The executed modes distinguish return prices from unit notionals:

- `prices` is the series used to calculate instrument returns.
- `units` is the number of shares, fund units, or derivative contracts held.
- `turnover_unit_notional` is the current value of one unit for turnover.

For an ordinary cash security, `turnover_unit_notional` normally equals `prices`, which is the
`PortfolioData` default. For a derivative, it must include the contract multiplier and currency
conversion. A futures backtest should therefore pass its USD contract-value panel rather than a
normalized total-return index.

## `TARGET_WEIGHTS`

$$
T_{i,t} = \left|w^{target}_{i,t} - w^{target}_{i,t-1}\right|
$$

The portfolio total is the sum across instruments. This mode requires only `input_weights` and
is useful when a model supplies allocation targets but no executed holdings.

It is a target-turnover proxy, not executed turnover. It does not observe rounding, trading
thresholds, partial fills, or trades required to maintain an unchanged exposure as an
instrument's unit notional changes.

Use it for:

- signal and allocation research where only target weights exist;
- comparing how quickly competing allocation rules change;
- compatibility with historical weight-only backtests.

Do not interpret it as actual traded volume when reliable units are available.

## `EXECUTED_NOTIONAL_NAV`

$$
N_{i,t} = \left|u_{i,t} - u_{i,t-1}\right|q_{i,t}
$$

$$
T_{i,t}^{NAV} = \frac{N_{i,t}}{NAV_t}
$$

Here, `u` is the executed number of units and `q` is the unit notional. This mode measures traded
notional as a percentage of investor capital. It is the default for `qis.compute_turnover` and
for QIS-created `PortfolioData` objects.

Use it for:

- long-only and long-short portfolios compared on the same capital base;
- relating turnover to transaction costs and performance expressed as a percentage of NAV;
- reporting how much market notional a strategy trades per dollar of capital.

For an unlevered long-only portfolio, NAV and gross exposure are usually close, so this mode and
the gross-exposure mode will also be close. They diverge for levered or market-neutral books.

Managed futures still require this NAV denominator for investor reporting. What makes the
calculation appropriate for futures is the **numerator**: changes in executed contracts valued at
their full contract notionals. The choice of denominator is separate. NAV keeps leverage visible
and puts turnover on the same capital base as returns, volatility, transaction costs, and fees.
If a strategy runs at gross leverage `L`, gross normalization divides by approximately `L × NAV`;
at 2x leverage it therefore reports about half the NAV-normalized turnover for the same trades.

## `EXECUTED_NOTIONAL_GROSS`

$$
G_t = \sum_i \left|u_{i,t}q_{i,t}\right|
$$

$$
T_{i,t}^{gross} = \frac{N_{i,t}}{G_t}
$$

This mode measures trading relative to the size of the current gross book. It answers an
implementation question—how quickly the deployed book is replaced—not the investor-capital
question answered by NAV normalization. Because the denominator grows with leverage, it removes
the leverage effect that investor reporting normally needs to retain.

Use it for:

- implementation and capacity diagnostics for managed-futures or derivatives books;
- comparing book replacement rates after intentionally normalizing away different leverage
  targets;
- answering what fraction of the currently deployed gross book was traded.

Do not use it as the primary factsheet turnover measure when comparing transaction-cost drag or
trading activity per dollar of investor capital. Use `EXECUTED_NOTIONAL_NAV` for those purposes.

Gross exposure can be zero while a strategy is flat. QIS emits a `RuntimeWarning` and returns
`NaN` for those dates rather than dividing by zero.

## Example: unchanged exposure can still require trading

Suppose a futures strategy maintains USD 100,000 of exposure. One contract is initially worth
USD 100,000, so the strategy holds one contract. If its contract value rises to USD 110,000, the
strategy must reduce the holding to approximately 0.909 contracts to keep the same exposure.

- `TARGET_WEIGHTS` reports zero because the requested exposure did not change.
- Both executed modes recognize the sale of approximately 0.091 contracts.
- The NAV and gross modes differ only in which portfolio-level denominator scales that traded
  notional.

## Portfolio defaults and overrides

QIS defaults to NAV-normalized executed turnover:

```python
portfolio = qis.PortfolioData(
    nav=nav,
    prices=prices,
    units=units,
    turnover_unit_notional=prices,
)
```

A managed-futures producer should pass full contract values while retaining NAV normalization
for factsheets and investor-level analytics:

```python
portfolio = qis.PortfolioData(
    nav=nav,
    prices=futures_return_indices,
    units=contract_sizes,
    turnover_unit_notional=contract_value_usd,
    turnover_computation_type=(
        qis.TurnoverComputationType.EXECUTED_NOTIONAL_NAV
    ),
)
```

Gross-normalized book churn remains available as an explicit diagnostic:

```python
gross_book_churn = portfolio.get_turnover(
    turnover_computation_type=qis.TurnoverComputationType.EXECUTED_NOTIONAL_GROSS,
)
```

Callers can also select the target-weight proxy explicitly:

```python
turnover = portfolio.get_turnover(
    turnover_computation_type=qis.TurnoverComputationType.TARGET_WEIGHTS,
)
```

The former `is_unit_based_traded_volume` turnover selector remains temporarily available for
compatibility. `True` maps to `EXECUTED_NOTIONAL_GROSS`; `False` maps to `TARGET_WEIGHTS`. New code
should use the enum because the boolean cannot express NAV-normalized executed turnover.

## Resampling and rolling reports

`compute_turnover` returns per-instrument, per-period turnover. `PortfolioData.get_turnover` then
performs the requested transformations in this order:

1. aggregate instruments or groups when requested;
2. sum observations to `freq`, when supplied;
3. sum the resulting series over `roll_period`, when supplied;
4. restrict the result to `time_period`.

These transformations do not change the underlying turnover convention. Factsheet titles use
“Two-sided Turnover” to make the purchase-plus-sale convention visible.
