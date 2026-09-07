---
myst:
  html_meta:
    description: >-
      Translate local assets into a reference currency, apply CIP-consistent FX hedges, and keep
      optional market-data acquisition separate with qis.
---

# FX hedging and market-data boundaries

Use `qis.FxRatesData` when asset prices are denominated in local currencies and performance must
be translated into one reference currency, with an explicit hedge ratio and covered-interest-
parity carry. The container is analytics infrastructure: it accepts spot and rate panels you
already have. Downloading those panels is a separate optional or vendor-specific concern, and
Bloomberg is not a prerequisite.

## Data and calculation contract

- **FX spots:** a DataFrame quoted as USD per one unit of each currency. `USD` is 1.0. The cross
  for a local asset in a reference frame is `spot[local] / spot[reference]`, with units of
  reference currency per one unit of local currency.
- **Domestic rates:** a DataFrame of annualised short rates as decimals on currency columns.
  Construction reindexes rates onto the spot calendar and forward-fills both panels. It cannot
  fill a missing leading rate or validate whether a carried observation remains economically
  current.
- **Asset prices:** levels in each asset's local currency. A hedge ratio of 0 is unhedged and 1
  hedges opening principal. Ratios can be constant by asset or time-varying.
- **Return convention:** construct asset and forward payoffs in simple returns, retaining the
  exact local/FX cross-product. Log output is `log1p` of the completed payoff. For an unhedged
  position this equals local log return plus FX log return. Supplied forward premiums must use
  the convention selected by `is_log_returns`; the hedge calculation converts them internally.
- **Frequency and annualisation:** `freq` is both the return/hedge cadence and the CIP period.
  With annualised local and reference rates and `dt = 1 / annualisation_factor(freq)`, the simple
  local-over-reference cash-growth premium is `f = (1 + dt*r_local)/(1 + dt*r_reference) - 1`.
  This getter convention is preserved. With spot `S` quoted as reference per local, the actual
  forward is `F = S*(1 + dt*r_reference)/(1 + dt*r_local)`, hence `F/S = 1/(1+f)`.
- **Timing and NaNs:** the hedge ratio and forward premium are lagged one period, so the return
  over *[t-1, t]* uses information set at *t-1*. Price and cross-rate inputs are aligned and
  forward-filled inside the pair calculation. Hedge and forward quotes are as-of aligned to
  the actual return grid before lagging; treat this as a declared valuation policy, not
  proof that a market was open.

## Minimal offline example

This deterministic example constructs its own EUR/USD spot and short-rate panels. It needs no
network, API key, or data extra.

```python
import pandas as pd
import qis

dates = pd.bdate_range('2023-01-02', '2024-12-31')
step = pd.Series(range(len(dates)), index=dates, dtype=float)
spots = pd.DataFrame(
    {'USD': 1.0, 'EUR': 1.05 * (1.0 + 0.0001 * step)}, index=dates
)
annualised_rates = pd.DataFrame({'USD': 0.04, 'EUR': 0.03}, index=dates)
fx = qis.FxRatesData(fx_spots=spots, domestic_rates=annualised_rates)

eur_asset = pd.Series(100.0 * (1.0 + 0.0002) ** step, name='EUR asset')
eur_in_usd = fx.get_local_to_reference_fx_rate('EUR', 'USD')
monthly_forward = fx.get_forward_rate_for_local_ccy(
    local_ccy='EUR', reference_ccy='USD', freq='ME', is_log_returns=False
)
unhedged_nav, unhedged_returns = (
    fx.compute_performance_of_local_ccy_asset_in_reference_ccy(
        asset_price_local_ccy=eur_asset,
        hedge_ratio=0.0,
        local_ccy='EUR',
        reference_ccy='USD',
        freq='ME',
        is_log_returns=False,
    )
)
hedged_nav, hedged_returns = (
    fx.compute_performance_of_local_ccy_asset_in_reference_ccy(
        asset_price_local_ccy=eur_asset,
        hedge_ratio=1.0,
        local_ccy='EUR',
        reference_ccy='USD',
        freq='ME',
        is_log_returns=False,
    )
)
```

`eur_in_usd`, `monthly_forward`, and each NAV/return output are Series. The unhedged path contains
the full EUR/USD spot return. The fully hedged path removes that direct spot leg but pays or earns
the exact lagged forward cost `f/(1+f)`; it is not generally equal to the local-currency asset return. In this
example EUR rates are below USD rates, so the EUR forward premium is negative under the stated
sign convention.

For a panel, `compute_returns_in_reference_ccy` returns `(navs, returns)` DataFrames at one
frequency. `compute_fx_adjusted_returns` accepts per-asset frequencies and returns a dictionary of
return DataFrames keyed by frequency. When `is_excess_returns=True`, simple excess is `R-c`;
log excess is `log1p(R)-log1p(c)`, the log return relative to cash. Here `c` is the reference-
currency simple short-rate accrual, using the rate known at the start of the asset return
period. The raw rate quotes are as-of aligned to the actual asset grid, divided by the frequency's
annualisation factor, and shifted one period. This also applies to native-currency excess returns
from `compute_returns_adjusted_by_local_rate`. Its third output remains the annual rate quote
panel for reporting. Simple and log **total** returns produce identical NAVs; the two **excess**
conventions are distinct and should not be tested with an `expm1` identity.

## Hedged, unhedged, cash, and futures exposures

An unhedged cash asset translates its full notional: simple reference return is
`r_local*(1+r_fx) + r_fx`. A hedge ratio `h` changes this to
`r_local*(1+r_fx) + (1-h)*r_fx - h*f/(1+f)`, with `h` and `f` lagged. Futures are
different because only P&L, not funded notional, is translated; use the dedicated futures helper
rather than applying the cash identity.

To verify the hedge independently, start with local asset value `V_local` and sell
`h*V_local` local currency forward at `F`. Terminal reference-currency wealth is

```text
V_terminal = V_local*(P_terminal/P_start)*S_terminal
             + h*V_local*(F-S_terminal)
R_reference = V_terminal/(V_local*S_start) - 1
```

A hedge ratio of one covers opening principal, so local investment gains and cash interest remain
exposed to terminal FX. To test exact covered-interest parity for a known local cash payoff,
hedge the terminal cash amount: `h = 1 + c_local`. This produces exactly `1 + c_reference`
terminal gross wealth, independently of terminal FX. The equal-excess-return identity does not
generally hold for a risky asset with only opening principal hedged.

`compute_fx_optimal_hedge` estimates carry-tilted and beta-aware hedge ratios.
Its carry term uses the same `f/(1+f)` forward cost, annualised by the existing period length;
the EWMA mean-variance approximation and clipping bounds are unchanged. Its EWMA state must
be given enough history, and any full-sample normalization is descriptive. Optimized hedge ratios
are model outputs, not guaranteed savings, executable forward quotes, or a substitute for
transaction-cost and liquidity modelling.

## Data acquisition boundary

Core FX construction and hedging use pandas inputs and require no vendor. Free-data examples use
Yahoo through the optional `data` extra (`pip install "qis[data]"`) and require network access.
CSV/parquet I/O uses the `io` extra. Bloomberg construction belongs to `bbg-fetch` and needs the
user's Bloomberg environment; it is one possible source, not an installation or runtime
requirement for qis analytics.

## Constraints and failure modes

- Reversing the spot quote reverses FX-return and carry signs. Confirm “reference per local” before
  interpreting a hedge.
- Nonpositive local/reference cash gross factors or simple forward gross factors raise an error.
  Log performance also raises if the combined asset/hedge payoff leaves nonpositive wealth;
  losses are not clipped or replaced by missing observations. NaNs remaining after the declared
  alignment and forward-fill policy remain missing.
- CIP derives a theoretical premium from short rates; it is not a live executable forward quote
  and omits basis, spreads, collateral, and trading costs.
- A full hedge removes the direct spot leg under the model, not local asset risk or the
  local-return/FX cross-product.
- Forward-filled rates and spots may be stale across holidays or outages. The container aligns
  data but does not certify freshness.
- Exact zero returns in mixed-frequency FX panels may be converted to NaN for estimation. This
  removes structural zeros but can also remove a genuine zero.
- Reference-currency translation changes the unit of account; it does not change the underlying
  asset's legal denomination, liquidity, or cash flows.

## See also

- {doc}`Generated FxRatesData API <api/generated/qis.FxRatesData>`
- {doc}`Generated FX hedge API <api/generated/qis.compute_performance_of_local_ccy_asset_in_reference_ccy>`
- [Reporting-frequency convention](_included/reporting_frequencies.md)
- [Canonical free-data FX examples](https://github.com/ArturSepp/QuantInvestStrats/tree/main/examples/market_data)
