---
myst:
  html_meta:
    description: >-
      Distinguish ragged starts, missing observations, stale marks, delisted tails and
      reporting frequency; construct explicit available-universe allocations with qis.
---

# Incomplete and mixed-frequency data

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-15](https://github.com/ArturSepp/QuantInvestStrats/commit/8a10dc6b72ed8db593e42d5eafe6d3e4b23419e6)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Incomplete data have missing observations; mixed-frequency data combine series reported on
different calendars. Neither condition has a single economic meaning. A price absent before
inception, a holiday gap, an unchanged appraisal and a missing terminal price require different
treatment before performance analysis or backtesting.

## Overview

Choose a policy for each series before aligning it with others. The policy determines which
observations can inform an estimate, which assets are eligible for allocation, and how existing
holdings are valued. qis implements explicit operations for these tasks; it cannot infer an
asset's reporting or trading status from a price panel alone.

### Classify the defect first

| Pattern | Possible meaning | Treatment to establish |
|---|---|---|
| Ragged start | Asset did not exist or was outside the universe | Retain leading NaNs; establish its first eligible price |
| Internal missing observation | Data failed to arrive within a reported history | Repair upstream when justified, or retain missing values and investigate |
| Stale price | Repeated level without new valuation information | Identify actual update dates and choose a suitable estimation grid |
| Delisted tail | Trading or reporting ceased | Retain the distinction from an ordinary gap; specify liquidation, recovery or default treatment |
| Low-frequency report | NAV is genuinely monthly or quarterly | Use that information grid rather than treating carried daily marks as new observations |

The frozen synthetic universe illustrates all five: `SEQ_EM` starts late, `SEQ_EU` has internal
gaps, `SCM_GLD` includes stale marks, `SCM_BCOM` has a delisted tail, and `SAL_PE` reports monthly.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Simple and log returns, stated per function |
| Sampling grid | Declared per asset or sleeve; storage dates are not information dates |
| Annualisation | Per sleeve, from its own grid |
| Mean adjustment | Not applicable |
| Timing | An asset that becomes available is admitted at the next scheduled rebalance |
| Output units | Weights as fractions of NAV; decimal returns |
| qis default | `to_returns(ffill_nans=True)`; `generate_static_weights_schedule(is_rescale_to_live_universe=True)` |

| Symbol or input | Meaning | Units or convention |
|---|---|---|
| $P_{i,t}$ | Observed price or NAV for asset $i$ | Positive finite level; NaN denotes unavailable |
| $b_i$ | Fixed requested allocation | Signed fraction of portfolio NAV |
| $E=\sum_i b_i$ | Requested total **net** exposure | Decimal fraction; not gross exposure |
| $I_{i,t}$ | Price-availability indicator at a rebalance | One for nonmissing, zero for missing |
| $w^*_{i,t}$ | Scheduled target weight | Fraction of NAV; distinct from drifted holdings |
| $r_{i,t}$, $\ell_{i,t}$ | Simple and log returns | Decimal simple return and log price ratio |
| `freq` | Estimation or reporting calendar | pandas offset such as `ME` or `QE` |

Use a sorted `DatetimeIndex` and asset columns with consistent identifiers. Validate finite,
positive prices upstream: the static-schedule helper's availability test is `notna()`, not a
validation of price quality or executability. A per-asset frequency Series must map the intended
asset names to their information grids.

The examples use synthetic levels and no risk-free-rate series. Backtesting starts from the
supplied observation grid; same-date valuation does not establish that a closing mark was known
in time to trade. Apply an execution lag where the strategy requires one.

## Methodology

### Allocate over the available universe

At each scheduled rebalance, the default `generate_static_weights_schedule` rule is

$$
w^*_{i,t}
= E\,\frac{I_{i,t}b_i}{\sum_j I_{j,t}b_j}.
$$

This preserves the sum of the original weights. It preserves gross exposure as well for a
long-only book, but not generally for a long/short book. Rescaling a zero-net specification,
or a live sleeve whose signed weights cancel, is undefined and raises an error. A date on which
no instrument is priced receives all-zero weights.

With `is_rescale_to_live_universe=False`, the schedule instead uses
$w^*_{i,t}=I_{i,t}b_i$. Missing target exposure remains unallocated. With rescaling enabled and
`is_preserve_total_exposure=False`, the requested row sum becomes one. That is a different
allocation policy, particularly when a deliberate cash sleeve is present.

The helper includes the first observation by default (`include_start_date=True`), followed by
the requested rebalance dates that exist on the price grid. An asset becoming available between
rebalances is admitted at a subsequent rebalance. Supplying this dated schedule makes opening
allocation explicit.

### Mixed-frequency analysis

Separate storage dates from information dates. For $m$ consecutive, fully observed simple
returns and their corresponding log returns:

$$
R_{1:m}=\prod_{t=1}^{m}(1+r_t)-1,
\qquad
L_{1:m}=\sum_{t=1}^{m}\ell_t.
$$

An endpoint price ratio determines the period's cumulative return. It does not reveal how that
return occurred between appraisals. Forward-filled intermediate levels therefore do not identify
a daily economic return process.

Resample liquid and illiquid sleeves onto declared grids, or use a method explicitly supporting
per-asset frequencies. State observations and annualisation per sleeve. Square-root annualisation
does not by itself correct serial dependence; see [Lo (2002)](https://alo.mit.edu/publications/page/18/)
and the [frequency convention](frequency_convention_note.md).

### Missing-value policies are method-specific

`qis.to_returns` forward-fills by default; use `ffill_nans=False` when gaps must remain missing.
Input already on the requested `freq` grid, such as month-end NAVs passed with `freq='ME'`, keeps
its gaps whatever `ffill_nans` says: on its own grid a missing value is a missing report.
Specify `is_log_returns` explicitly. Filling should stop at the economically justified boundary,
especially after a terminal event.

The backtester warns about internal NaNs. Units can remain held while `np.nansum` omits the
unpriced leg from that day's valuation, creating artificial NAV movements. A warning is not a
repair. An unpriced target cannot be acquired; supplying a fixed 25% target in that leg leaves
the opening allocation in cash rather than redistributing it automatically.

The mixed-frequency FX helper `FxRatesData.compute_fx_adjusted_returns` replaces exact zero
returns with NaN for estimation. This removes structural zeros and also any genuine exact zero.
It does not detect stale marks or delistings.

## Worked example

For requested weights 50%, 25%, 25%, suppose only the first two assets have a price. The default
live schedule divides by 0.75 and preserves total exposure one:

| Asset | Requested | Rescaled live target | Unallocated-exposure policy |
|---|---:|---:|---:|
| `SEQ_US` | 50% | 66.6667% | 50% |
| `SBD_TSY` | 25% | 33.3333% | 25% |
| `SEQ_EM`, unavailable | 25% | 0% | 0% |
| Residual cash | 0% | 0% | 25% |

This is an allocation illustration. It says nothing about subsequent performance or whether a
particular observed price could have been executed.

### Minimal offline example: a late-starting asset

`SEQ_EM` has a late start in the fixed synthetic sample below. The assertions check the two
policies against the fractions in the table.

```python
import numpy as np
import qis
from qis.datasets.synthetic import generate_synthetic_prices

tickers = ['SEQ_US', 'SBD_TSY', 'SEQ_EM']
prices = generate_synthetic_prices(
    start='2005-01-03', end='2025-12-31', seed=20260725, apply_quirks=True
)[tickers]
static_targets = {'SEQ_US': 0.50, 'SBD_TSY': 0.25, 'SEQ_EM': 0.25}

live_schedule = qis.generate_static_weights_schedule(
    prices=prices, weights=static_targets, rebalancing_freq='QE',
)
cash_schedule = qis.generate_static_weights_schedule(
    prices=prices, weights=static_targets, rebalancing_freq='QE',
    is_rescale_to_live_universe=False,
)
assert live_schedule.index[0] < prices['SEQ_EM'].first_valid_index()
np.testing.assert_allclose(live_schedule.iloc[0], [2.0 / 3.0, 1.0 / 3.0, 0.0])
np.testing.assert_allclose(cash_schedule.iloc[0], [0.50, 0.25, 0.0])

portfolio = qis.backtest_model_portfolio(
    prices=prices, weights=live_schedule, rebalancing_costs=0.0010,
    ticker='Available-universe allocation',
)
```

`live_schedule` and `cash_schedule` are date-by-asset DataFrames; `portfolio` is
`qis.PortfolioData`. The backtest charges 10 bp per unit traded. Its cash funding rate defaults
to zero. Costs and holding drift can make realised weights differ from scheduled targets.

## Implementation in qis

| Operation | Public entry point | Output |
|---|---|---|
| Schedule a fixed allocation over priced assets | `generate_static_weights_schedule` | Rebalance-date-by-asset target weights |
| Convert price levels on a stated grid | `to_returns` | Return Series or DataFrame |
| Apply dated targets with held-unit accounting | `backtest_model_portfolio` | `PortfolioData` |
| Translate and group assets by their frequencies | `FxRatesData.compute_fx_adjusted_returns` | Return DataFrames keyed by frequency |

Implementation owners are the [weight-schedule source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/utils/df_to_weights.py),
[return conversion](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/returns.py),
[backtester](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/backtester.py),
and [FX container](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/market_data/fx_rates_data.py).
The [canonical late-start example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/static_weight_with_missing_prices.py)
demonstrates the larger workflow. Repository tests in
[test_static_weights_schedule.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/utils/tests/test_static_weights_schedule.py)
check exposure preservation and failure cases.

<a id="constraints-and-failure-modes"></a>

## Interpretation and limitations

For multi-asset periodic tables, `qis.compute_periodic_returns` preserves a leading missing region
until an asset supplies two observed price boundaries. A column with fewer than two observations
has neither periodic nor total returns. The table calculation continues to forward-fill internal
and trailing gaps; that display-oriented convention is not a liquidation or delisting policy.

- Backfilling a ragged start invents pre-inception history and can create look-ahead.
- Carrying a terminated investment indefinitely can hide liquidation, recovery or default losses.
- Repeated marks change observed return dependence; they are not evidence of low economic risk.
- Rescaling changes allocation. State whether the intended constraint is net exposure, gross
  exposure, or cash; the schedule's default preserves the signed sum.
- Price availability, universe eligibility and execution are separate contracts. A static panel
  assembled later may also omit failed assets or contain revised historical observations.
- Repair decisions need source information. Levels alone cannot reliably distinguish holidays,
  operational outages, stale appraisals and terminal events.

## See also

- [Portfolio backtesting](portfolio_backtesting.md)
- [Private-asset unsmoothing](private_asset_unsmoothing.md)
- [FX hedging](fx_hedging_and_market_data.md)
- [Frequency convention](frequency_convention_note.md)
- {doc}`Static-schedule API <api/generated/qis.generate_static_weights_schedule>` and
  [source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/utils/df_to_weights.py)
- {doc}`Return-conversion API <api/generated/qis.to_returns>` and
  [source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/returns.py)

## References

1. Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*, 58(4), 36–52. [DOI: 10.2469/faj.v58.n4.2453](https://doi.org/10.2469/faj.v58.n4.2453). Estimation and time aggregation with serial dependence.
2. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
