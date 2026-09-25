---
myst:
  html_meta:
    description: >-
      Compare four two-sided turnover conventions: target weights, volatility-normalised
      targets, executed notional over NAV, and executed notional over gross exposure.
---

# Two-sided turnover conventions

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-12](https://github.com/ArturSepp/QuantInvestStrats/commit/334d942db051de44b5abecedc1f5a2f1fa167c51)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Portfolio turnover measures changes in an allocation or its executed holdings. Its interpretation
depends on what changed and which denominator scales that change. qis provides four explicit
conventions, all **two-sided**: purchases and sales contribute their absolute amounts, without
multiplying their sum by one half.

## Overview

`qis.compute_turnover` is the calculation engine. `PortfolioData.get_turnover` and factsheets
use its results before grouping, resampling, or rolling aggregation.

| Convention | Numerator | Denominator / scaling | Principal use |
|---|---|---|---|
| `TARGET_WEIGHTS` | Absolute target-weight changes | Capital fractions already encoded in weights | Allocation-change proxy when executed units are unavailable |
| `VOLATILITY_NORMALIZED_WEIGHTS` | Absolute target-weight changes | Multiplied by annualised instrument volatility | Theoretical signal and sizing analysis |
| `EXECUTED_NOTIONAL_NAV` | Unit changes at current unit notionals | Current portfolio NAV | Investor reporting and comparison with costs per NAV; qis default |
| `EXECUTED_NOTIONAL_GROSS` | Same executed traded notional | Current gross exposure | Book replacement and implementation diagnostics |

A target proxy describes instructions. Executed turnover describes changes in held units.
Changing the denominator does not turn one numerator into the other.

<a id="inputs"></a>

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Not applicable: turnover is formed from weight or unit changes |
| Sampling grid | Dates of the supplied holdings or targets; reports resample and roll-sum |
| Annualisation | None per observation; `vols` are already annualised |
| Mean adjustment | Not applicable |
| Timing | Each row is the change from the previous dated row |
| Output units | Two-sided fraction of NAV or of gross exposure |
| qis default | `EXECUTED_NOTIONAL_NAV`; `PortfolioData.get_turnover(roll_period=260)` |

| Symbol or input | Meaning | Units and contract |
|---|---|---|
| $w^*_{i,t}$ / `input_weights` | Target weight | Signed decimal fraction; dated rows and asset columns |
| $u_{i,t}$ / `units` | Held units after trading | Shares, fund units, or actual contracts |
| $n_{i,t}$ / `unit_notional` | Current exposure represented by one unit | Non-negative notional in the NAV currency |
| $V_t$ / `nav` | Current portfolio NAV | Same monetary currency as unit notionals |
| $G_t$ | Current gross book exposure | Sum of absolute position notionals |
| $\sigma^{\mathrm{ann}}_{i,t}$ / `vols` | Annualised instrument volatility | Decimal fraction; aligned exactly with target weights |

In `PortfolioData` the unit-notional field is named `turnover_unit_notional`; in
`compute_turnover` the argument is `unit_notional`. Return prices and unit notionals serve
different purposes:

- `prices` drives instrument returns and P&L.
- `turnover_unit_notional` converts unit changes into monetary traded exposure.
- For a cash security, one unit is normally one share or fund unit. The notional is its price
  converted to the portfolio currency; `PortfolioData` defaults this field to `prices`.
- For a futures contract, use the full contract notional, including multiplier and FX conversion.
  A normalised return index or a contract's near-zero initial accounting value is not that notional.
- If units already measure exposure in portfolio currency, the per-unit notional may be 1.

For dated inputs, `compute_turnover` orders each input chronologically on a local object before
computing changes or aligning companions. Physical row order therefore does not change turnover,
and caller-owned objects are not reordered. Duplicate or `NaT` dates are rejected because they do
not define one unambiguous temporal transition. Executed modes align the notional panel to the
units' dates and columns; they do not infer missing market values. All unit columns must be present.
The caller is responsible for compatible currencies, actual contract quantities, and economically
meaningful notionals.

The first output row is normally missing because no preceding holding or target exists.
Include an explicit prior flat row if an opening trade should appear in a turnover series.
Do not silently interpret the first row as zero trading.

## Methodology

### Executed traded notional

Current notionals value changes in executed units:

$$
N^{\mathrm{trade}}_{i,t}
=\left|u_{i,t}-u_{i,t-1}\right|n_{i,t}.
$$

For example, a change of two contracts at CHF 150,000 notional per contract trades CHF 300,000.
On a CHF 1,000,000 NAV, the contribution to NAV-normalised turnover is 30%.

### `TARGET_WEIGHTS`

$$
T^{\mathrm{target}}_{i,t}
=\left|w^*_{i,t}-w^*_{i,t-1}\right|.
$$

Only `input_weights` is required. Sum across instruments for total two-sided target turnover.
This proxy is useful for comparing allocation rules or historical weight-only backtests.
It does not observe trades caused by drift, contract rounding, thresholds, partial fills, or
maintaining unchanged exposure when a unit's notional changes.

### `VOLATILITY_NORMALIZED_WEIGHTS`

[Sepp and Lucic (2026), Definition 4.5 and equation 4.15](https://arxiv.org/html/2607.19497v1#S4.SS4)
define volatility-normalised turnover using periodic volatility $\sigma_{i,t}$ and
annualisation factor $\mathrm{AN}$:

$$
U_{i,t}=\sqrt{\mathrm{AN}}\,\sigma_{i,t}
\left|w^*_{i,t}-w^*_{i,t-1}\right|.
$$

qis takes `vols` already annualised, so the implemented equivalent is:

$$
U_{i,t}=\sigma^{\mathrm{ann}}_{i,t}
\left|w^*_{i,t}-w^*_{i,t-1}\right|,
\qquad
\sigma^{\mathrm{ann}}_{i,t}=\sqrt{\mathrm{AN}}\,\sigma_{i,t}.
$$

This weights target changes by instrument risk. It includes changes caused by the target rule's
signal and sizing estimates, but excludes realised holding drift and execution effects.
It is a theoretical comparison measure, not executed market volume.

After dated rows are ordered chronologically, `vols` and `input_weights` must have identical
indexes, columns, and column order. Negative volatilities are rejected; warm-up NaNs propagate.
qis neither lags nor annualises `vols` here. Supply a point-in-time panel appropriate for the
target decision.

Each output still corresponds to one target-change interval. Using annualised volatility does
not by itself sum or annualise a turnover history. Group and time aggregation remain separate.

### `EXECUTED_NOTIONAL_NAV`

$$
T^{\mathrm{NAV}}_{i,t}
=\frac{N^{\mathrm{trade}}_{i,t}}{V_t}.
$$

This is the default for `compute_turnover` and qis-created `PortfolioData` objects.
It expresses traded exposure per unit of investor capital and retains leverage. It is the
primary convention when comparing trading activity, transaction costs, fees, and performance
on a common NAV basis.

The numerator makes it appropriate for derivatives: executed contract changes must be valued
at full contract notionals. The NAV denominator remains appropriate for investor reporting.
For the same traded notional, a book with gross exposure $L V_t$ has a gross-normalised
turnover equal to its NAV-normalised turnover divided by $L$.

### `EXECUTED_NOTIONAL_GROSS`

$$
G_t=\sum_i\left|u_{i,t}n_{i,t}\right|,
\qquad
T^{\mathrm{gross}}_{i,t}
=\frac{N^{\mathrm{trade}}_{i,t}}{G_t}.
$$

This measures turnover relative to the **current post-trade gross book**, rather than investor
capital. It is useful for book replacement or capacity diagnostics when leverage normalisation
is intentional. At 2x gross exposure, the same trades produce half the NAV-normalised turnover.

In an unlevered, fully invested long-only portfolio, gross exposure and NAV are usually close.
They can differ materially for leveraged or market-neutral portfolios, or when the book holds
cash. A zero gross denominator produces a `RuntimeWarning` and NaNs; a zero NAV denominator
has the same treatment in the NAV mode. Liquidating the final position can therefore produce
undefined gross-normalised turnover even though the trade has a meaningful NAV denominator.

<a id="example-unchanged-exposure-can-still-require-trading"></a>

## Worked example

### A 2x book: the denominator is visible

Consider two dated holdings rows with constant NAV 100 and constant unit notionals 100.
Positions move from $(1,1)$ units to $(0.8,1.2)$, preserving gross exposure 200. Purchases plus
sales trade 40 of notional. The targets change by $(−0.2,+0.2)$; annualised volatilities are
20% and 10%.

| Convention | Total on the second row |
|---|---|
| Target weights | $0.2+0.2=0.40$ (40%) |
| Volatility-normalised targets | $0.20(0.2)+0.10(0.2)=0.06$ |
| Executed notional / NAV | $40/100=0.40$ (40%) |
| Executed notional / gross | $40/200=0.20$ (20%) |

The following fixed accounting illustration is fully offline:

```python
from math import isclose

import pandas as pd
import qis

dates = pd.to_datetime(['2024-01-02', '2024-01-03'])
units = pd.DataFrame({'Asset A': [1.0, 0.8], 'Asset B': [1.0, 1.2]}, index=dates)
unit_notionals = pd.DataFrame(100.0, index=dates, columns=units.columns)
nav = pd.Series(100.0, index=dates, name='Illustrative 2x book')
targets = units.copy()  # Here one unit equals one NAV of exposure.
annualized_vols = pd.DataFrame(
    {'Asset A': [0.20, 0.20], 'Asset B': [0.10, 0.10]}, index=dates
)
expected = {
    qis.TurnoverComputationType.TARGET_WEIGHTS: 0.40,
    qis.TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS: 0.06,
    qis.TurnoverComputationType.EXECUTED_NOTIONAL_NAV: 0.40,
    qis.TurnoverComputationType.EXECUTED_NOTIONAL_GROSS: 0.20,
}
for convention, total in expected.items():
    result = qis.compute_turnover(
        computation_type=convention, units=units, unit_notional=unit_notionals,
        nav=nav, input_weights=targets, vols=annualized_vols,
    )
    assert result.iloc[0].isna().all()
    assert isclose(result.iloc[1].sum(), total, abs_tol=1e-12)
```

### Unchanged exposure can still require trading

Hold NAV fixed for this illustration. A strategy with USD 100,000 of exposure holds one contract with
USD 100,000 unit notional. At USD 110,000 per contract, a fractional holding of about 0.9091
keeps that exposure unchanged. Selling about 0.0909 contracts trades USD 10,000 of notional.

An unchanged capital target reports zero target turnover, while executed modes recognise the
sale. Fractional contracts are used only for this arithmetic illustration; actual contract
rounding changes the realised quantities and must be reflected in `units`.

## Implementation in qis

### Portfolio defaults and overrides

The [turnover engine](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/turnover.py)
owns the four calculations. [PortfolioData](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/portfolio_data.py)
adds reporting transformations and stores the preferred convention.

Continuing the offline example above:

```python
portfolio = qis.PortfolioData(
    nav=nav, prices=unit_notionals, units=units, input_weights=targets,
    turnover_unit_notional=unit_notionals,
    turnover_computation_type=qis.TurnoverComputationType.EXECUTED_NOTIONAL_NAV,
)
executed_turnover = portfolio.get_turnover(roll_period=None)
gross_book_churn = portfolio.get_turnover(
    turnover_computation_type=qis.TurnoverComputationType.EXECUTED_NOTIONAL_GROSS,
    roll_period=None,
)
target_turnover = portfolio.get_turnover(
    turnover_computation_type=qis.TurnoverComputationType.TARGET_WEIGHTS,
    roll_period=None,
)
theoretical_turnover = portfolio.get_turnover(
    turnover_computation_type=qis.TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS,
    vols=annualized_vols,
    roll_period=None,
)
```

For derivative reporting, pass the instrument return indices as `prices`, actual held contracts
as `units`, and full contract values in portfolio currency as `turnover_unit_notional`. Do not
reinterpret units generated from a normalised return index as actual executed contracts.

The deprecated `is_unit_based_traded_volume` selector remains available for compatibility:
`True` maps to `EXECUTED_NOTIONAL_GROSS`; `False` maps to `TARGET_WEIGHTS`. It cannot select the
NAV-normalised or volatility-normalised modes. Use the enum for new code, and do not pass both
the enum and the deprecated selector.

### Resampling and rolling reports

`compute_turnover` returns per-instrument, per-observation results. `PortfolioData.get_turnover`
then applies, in order:

1. instrument or group aggregation, when requested;
2. sums to `freq`, when supplied;
3. a sum over `roll_period` observations on the resulting grid;
4. restriction to `time_period`.

The default `roll_period` is 260. Use `roll_period=None` to inspect the underlying observations,
as in the example. Resampling and rolling sums do not change the turnover convention; a rolling
sum of per-date NAV ratios is not a single traded amount divided by one common NAV.
The returned default table includes a total alongside instrument columns; do not sum that total
again with its constituents. Factsheet titles label the convention as “Two-sided Turnover”.

The separate [Yahoo tactical SPY/TLT example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/perfstats/turnover_conventions.py)
compares 1x and 2x leverage on downloaded data. It requires the data extra and network access;
its market sample is separate from the fixed arithmetic example above.

## Interpretation and limitations

- Two-sided turnover counts purchases plus sales. Do not compare it directly with a measure
  that halves their sum without reconciling definitions.
- Target changes can miss drift, rounding, or execution-driven trades. Use recorded unit
  changes when the question concerns actual trading.
- NAV normalisation keeps leverage visible; gross normalisation intentionally divides by
  current book size. Always identify the denominator.
- A missing first row means no previous holding was supplied. Backtesting costs can still
  include opening trades; turnover and costs require consistent opening boundaries to reconcile.
- The cost engine uses trade-date cost rates and monetary traded amounts. A changing cost
  panel or reporting aggregation means a turnover total alone cannot reconstruct all costs.
- Missing prices/notionals, zero denominators, incompatible currencies, or theoretical units
  mistaken for contracts can make the result uninterpretable even when inputs have matching shapes.
- Volatility-normalised target turnover is a theoretical risk-scaled measure, not a replacement
  for traded-notional turnover in investor factsheets.

## See also

- [Portfolio backtesting and execution timing](portfolio_backtesting.md)
- [Portfolio breadth](portfolio_breadth.md)
- [Factsheets and reporting](factsheets_and_reporting.md)
- [Reporting-frequency convention](frequency_convention_note.md)
- [Turnover implementation and enum](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/turnover.py)
- [Turnover regression examples](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/tests/turnover_test.py)

## References

1. Sepp, A., and Lucic, V. (2026). The Science and Practice of Trend-Following Systems. Working paper. [arXiv:2607.19497](https://arxiv.org/abs/2607.19497). Definition 4.5 and equation 4.15; qis accepts annualised volatility in the equivalent formula.
2. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
