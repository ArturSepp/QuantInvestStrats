---
myst:
  html_meta:
    description: >-
      Handle ragged starts, internal gaps, stale prices, delisted tails, and mixed-frequency
      investment data explicitly with qis.
---

# Incomplete and mixed-frequency data

Use this guide before performance analysis or backtesting when a panel combines different
observation histories or reporting cadences. “Missing” is not one economic state. A pre-inception
NaN, a holiday gap, an unchanged stale mark, a post-delisting tail, and an asset that reports only
quarterly require different policies; qis does not promise to infer the right one silently.

## Classify the defect first

| Pattern | Meaning | Defensible treatment |
|---|---|---|
| Ragged start | Asset did not exist or was outside the universe | Keep leading NaNs; admit it only after a tradable price |
| Internal missing observation | Data failed to arrive inside an otherwise reported history | Repair upstream if the last price is economically valid; otherwise keep missing and investigate |
| Stale price | A reported level repeats but carries little new information | Keep the level, but sample/estimate at the actual information frequency |
| Delisted tail | Asset stopped trading or reporting | Keep trailing NaNs and define liquidation/default handling outside the analytics call |
| Low-frequency report | NAV is genuinely monthly or quarterly | Estimate on that grid; do not manufacture daily observations by forward-filling |

The seeded synthetic universe contains all five patterns: `SEQ_EM` starts late, `SEQ_EU` has
internal gaps, `SCM_GLD` is stale, `SCM_BCOM` has a delisted tail, and `SAL_PE` reports monthly.
Prices are levels in arbitrary positive units; returns are decimal fractions. Frequencies are
pandas offsets such as `B`, `ME`, and `QE`. Annualisation must follow the selected information
grid, not the density of a forward-filled frame.

## Minimal offline example: a late-starting asset

The intended policy here is always 100% invested across assets that have a price on the quarterly
rebalance date. `generate_static_weights_schedule` makes that policy explicit before the
backtest.

```python
import qis
from qis.datasets.synthetic import generate_synthetic_prices

tickers = ['SEQ_US', 'SBD_TSY', 'SEQ_EM']
prices = generate_synthetic_prices()[tickers]
static_targets = {'SEQ_US': 0.50, 'SBD_TSY': 0.25, 'SEQ_EM': 0.25}

live_schedule = qis.generate_static_weights_schedule(
    prices=prices,
    weights=static_targets,
    rebalancing_freq='QE',
)
portfolio = qis.backtest_model_portfolio(
    prices=prices,
    weights=live_schedule,
    rebalancing_costs=0.0010,
    ticker='Available-universe allocation',
)

first_em_price = prices['SEQ_EM'].first_valid_index()
before_inception = live_schedule.loc[:first_em_price].iloc[0]
```

`live_schedule` is a date-by-asset DataFrame. Before `SEQ_EM` exists, its zero target is explicit
and the available 0.50/0.25 allocation is rescaled to 2/3 and 1/3 while preserving the original
total exposure. `portfolio` is `qis.PortfolioData`. The schedule changes eligibility only on its
quarterly dates; it does not claim that an asset was tradable between those dates.

If the intended policy is instead to retain unavailable exposure as cash, either pass the static
target directly or use `is_rescale_to_live_universe=False`. With direct targets, an unpriced 25%
leg stays as a 25% cash residual, accrues `funding_rate` (zero by default), and triggers a warning.
For a deliberately 90%-invested target, the default schedule preserves 90% total exposure; set
`is_preserve_total_exposure=False` only when forcing the live sleeve to 100% is intended.

## Mixed-frequency analysis

Separate economic frequency from storage frequency. A monthly appraisal series stored on a daily
index with repeated values is still monthly data. Resample liquid and illiquid sleeves onto their
declared grids before computing return statistics, or use APIs whose `freq` argument accepts a
per-asset Series. Report annualisation and observation counts per sleeve.

NaN behavior is API-specific. `qis.to_returns` forward-fills by default; pass
`ffill_nans=False` when gaps must survive. The backtester warns on an internal NaN and does not
heal it. FX mixed-frequency returns replace exact structural zeros with NaN for rolling estimation;
that intentionally also removes the rare organic zero. None of these policies identifies a stale
mark or delisting event for you.

## Constraints and failure modes

- Forward-filling a low-frequency NAV then estimating daily volatility treats “no new report” as
  a zero economic return and biases risk downward.
- Backfilling a ragged start invents history and creates an investable asset before inception.
- Forward-filling beyond a delisting can turn a loss or liquidation into a risk-free flat asset.
- Rescaling the available universe changes exposures. Preserve the stated total exposure when a
  cash allocation or gross target below one is intentional.
- A price present at quarter-end does not prove it was executable at that mark. Eligibility,
  valuation, and execution are separate data contracts.
- Inspect warnings and diagnostics. No generic cleaning function can distinguish holiday gaps,
  operational outages, stale appraisals, and terminal events from levels alone.

## See also

- {doc}`Generated static-schedule API <api/generated/qis.generate_static_weights_schedule>`
- {doc}`Generated return-conversion API <api/generated/qis.to_returns>`
- [Frequency convention note](_included/frequency_convention_note.md)
- [Canonical late-start example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/portfolios/static_weight_with_missing_prices.py)
