# `qis.fx`

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-06-19](https://github.com/ArturSepp/QuantInvestStrats/commit/a39da9f97a11a25848e1d6ff8bc644f7501d53a8)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Currency data, cash accrual, typed return analytics, hedge estimation and reporting.
Provider fetching and ticker definitions remain in the consuming data pipeline. Factor prices
are owned by [`qis.portfolio.risk.factors_data`](../portfolio/risk/factors_data.py), and
self-financed positions and settlement by [`qis.portfolio.fx_accounting`](../portfolio/fx_accounting.py).

```text
src/qis/fx/
  __init__.py                 currency API exports
  fx_rates_data.py            FxRatesData and provider-neutral CSV loading
  fx_returns.py               typed asset/currency requests, payoffs and optional indices
  cash_index.py               common monthly-reset currency cash index
  fx_hedging.py               quote, principal-hedge, volatility/beta and hedge-selection kernels
  reports/fx_hedging_report.py
  run_local/fx_rates_data_run.py
  tests/                     offline FX regression tests
examples/fx/                 runnable workflows; existing file basenames retained
```

Start with the offline [returns and accounting example](../../../examples/fx/fx_returns_and_accounting_example.py).
The [migration guide](../../../docs/return_conventions_and_fx_adjustments.md#immediate-deprecation-and-direct-migration)
explains the QIS 6.0.0 namespace move and removal of seven deprecated container methods.
Existing root object imports such as `from qis import FxRatesData` remain available.

---

## `FxRatesData` — data contract

A dataclass holding two aligned panels:

```python
@dataclass
class FxRatesData:
    fx_spots: pd.DataFrame        # FX spot levels
    domestic_rates: pd.DataFrame  # domestic short rates
```

* **`fx_spots`** — every column is **USD per 1 unit of that currency**. `USD` is pinned to `1.0`.
  Columns are 3-letter currency codes (`USD`, `EUR`, `GBP`, `CHF`, `JPY`, `AUD`, `CAD`, `NZD`, …)
  plus two special columns: `GBp` (pence) = `0.01 * GBP`, and `XAU` (gold) = `1.0`. A cross rate is
  formed by dividing two columns, so the quoting convention cancels and you get reference-per-local.
* **`domestic_rates`** — same columns as `fx_spots`; values are **annualised short rates as decimals**
  (e.g. `0.045` for 4.5%). `XAU` carries the USD rate as a financing-cost proxy; `GBp` shares the
  `GBP` sovereign curve.

`__post_init__` sorts and forward-fills `fx_spots`, then forward-fills `domestic_rates` on the union
of the rate and spot calendars before selecting the spot dates. Leading unknown observations
remain missing; callers must verify coverage for their requested currencies and dates. Alignment
never carries future observations backward or discards off-grid rate updates.

### Construction

```python
# 1. Directly from two DataFrames you already have (the convention above must hold):
fx = FxRatesData(fx_spots=spots_df, domestic_rates=rates_df)

# 2. From the persisted CSVs (file_name 'fx_hedging_data'):
fx = FxRatesData.load(local_path=qis.local_path.get_resource_path(), time_period=qis.TimePeriod('31Dec2005', None))

# 3. Production: built from Bloomberg by `create_fx_rates_data` — which lives in rosaa, not qis.
```

`load_fx_rates_data(local_path, file_name='fx_hedging_data')` is the standalone CSV reader; it returns
`(fx_spots, domestic_rates)` and heals a historical CSV bug where `GBp` was stored as a 1:1 copy of
`GBP` (the correct relation is `GBp = 0.01 * GBP`).

---

## FX conversions and CIP

| Method | Returns |
|---|---|
| `get_local_to_reference_fx_rate(local_ccy, reference_ccy)` | Cross rate = reference per 1 local (`fx_spots[local] / fx_spots[reference]`). |
| `get_local_rate(freq, local_ccy, annualise=False)` | Opening annual quote times actual calendar days/365 (`annualise=False`); unscaled annual quotes otherwise. |
| `get_forward_rate_for_local_ccy(local_ccy, reference_ccy, freq, is_log_returns=False)` | Per-period CIP forward premium (see below). |
| `compute_fx_returns(data, FxCurrencyReturnsRequest(...))` | Typed spot returns or exact forward P&L; no automatic NAV. |

**CIP and exact settlement.** Opening annual quotes accrue over actual calendar days / 365.
For local and reference cash increments `c_local` and `c_reference`, the inverse-quote premium
is `f = (1+c_local)/(1+c_reference)-1`. The reference-per-local forward is
`F/S = (1+c_reference)/(1+c_local)`. Long-local forward settlement per opening reference
notional is `spot_return + f/(1+f)`, not `spot_return + f`.

`get_daily_carry_local_return`, `get_fx_total_return_nav` and `get_carry_fx_return_nav` are
removed in 6.0.0 after deprecation in the 5.47.x public releases. Their old premium-based histories were not
exact forward payoffs. Use `FX_FORWARD` and an explicit `FxIndexNormalization` for a synthetic
capital index. Current report panels use a daily-reset, unit-notional-to-capital forward index.

---

## Cash NAVs

| Method | Purpose |
|---|---|
| `build_local_cash_nav(local_ccy, freq='B')` | Money-market NAV compounding the local short rate (rate lagged one period — no look-ahead). |
| `build_cross_fx_cash_nav(local_ccy, reference_ccy, freq='B', output_freq='ME')` | Foreign cash held unhedged by a reference-ccy investor; the synthetic series used to isolate Fx β ≈ +1 for a foreign cash leg. Returns an empty series when `local_ccy == reference_ccy`. |

---

## Common monthly cash measurement

`compute_monthly_cash_index(data, currency, dates)` samples one ACT/365 Fixed account:
opening quotes reset at calendar month-ends, interest is simple within each month and
reinvested at month-end. Missing opening history is not backfilled.
`FxCashPolicy(accrual=FxCashAccrual.MONTHLY_INDEX)` uses endpoint cash ratios for local or
reference excess returns. It does not reprice forwards or change account financing/reset
schedules. `PERIOD_OPENING` remains the default. The
[offline workflow](../../../docs/multi_currency_portfolio_pnl.md#sample-one-monthly-cash-index-at-different-frequencies)
checks daily/monthly/quarterly LOG excess against independent monthly deposits.

---

## Multi-asset FX-translation pipeline

The current entry point consumed by covariance estimation and alpha aggregation is typed:

```python
results = qis.compute_fx_returns(fx, qis.FxAssetReturnsRequest(
    prices=prices, local_ccys=local_ccys,
    return_type=qis.FxReturnType.REFERENCE_EXCESS,
    coordinate=qis.FxReturnCoordinate.LOG,
    reference_ccy='USD', hedge_ratios=hedge_ratios, frequency='ME',
    zero_return_to_nan=True))
returns_by_freq = {freq: result.values for freq, result in results.items()}
```

The request states the economic kind and coordinate. Zero masking is explicit. Log excess is
log asset growth minus log cash growth, distinct from arithmetic excess. `LOCAL_EXCESS_TRANSLATED`
uses arithmetic local excess times spot growth; `FX_FORWARD` is a currency request. Both notional
payoffs require `SIMPLE`, even if an optional normalized synthetic index is requested. Funded
reference excess subtracts reference cash once; local financing is a different economic object.

See the [executed convention and migration guide](../../../docs/return_conventions_and_fx_adjustments.md)
for all supported kinds, exact identities, optional normalization and removed methods.

---

## FX hedging analytics (`fx_hedging`)

Pure, stateless functions on plain price/return Series:

| Function | Returns |
|---|---|
| `compute_local_and_fx_return(asset_price_local_ccy, local_to_reference_fx_rate, freq, is_log_returns)` | `(local_return, fx_return)` legs. |
| `compute_performance_of_local_ccy_asset_in_reference_ccy(..., hedge_ratio, freq, is_log_returns)` | `(hedged_nav, hedged_return)` at a given hedge ratio. |
| `compute_fx_vol_beta(..., freq, span)` | `(fx_vol, fx_beta)` — EWMA annualised FX vol and the local-on-FX beta. |
| `compute_fx_optimal_hedge(..., freq, span, risk_aversion_lambda, min_max_hedge)` | `(optimal, max_carry, beta_hedged)` hedge-ratio series. |

**Hedged return.** `local_return*(1 + fx_return) + (1 - h)*fx_return - h*kappa`,
where `kappa=f/(1+f)` uses opening cash quotes and `h` is the opening principal hedge.
The local gain remains exposed to terminal spot. Typed results retain a missing first return;
synthetic indices require explicit normalization.

**Hedge strategies** (from EWMA vol/beta and annualised short-forward cost):
`annualised_cost = kappa / tau`, where `tau` is the ACT/365F year fraction from the
opening decision to the next reset and `kappa = f/(1+f)`. Statistical volatility
uses periods per year and an 8% annual variance prior at the requested frequency;
`carry_ratio = (annualised_cost / fx_var) / (2*risk_aversion_lambda)`; `Max Carry = 1 - carry_ratio`;
`Beta Hedge = 1 + fx_beta`; `Optimal = 1 - carry_ratio + fx_beta`. All optionally clipped to `min_max_hedge`.

---

## Reports (`reports.fx_hedging_report`)

| Function | Output |
|---|---|
| `run_asset_fx_hedging_report(asset_price_local_ccy, fx_rates_data, local_ccy, reference_ccy, time_period, freq, span, risk_aversion_lambda, min_max_hedge)` | Single-asset tearsheet `Figure`. |
| `compute_multi_asset_fx_hedging(asset_prices, fx_rates_data, time_period, local_ccys, reference_ccy, freq, span, risk_aversion_lambda, min_max_hedge)` | Dict of `pas` / `sharpes` / `betas` / `vols` (asset × strategy) and `last_hedges`. |
| `plot_multi_asset_fx_hedging_report(asset_prices, fx_rates_data, ..., span, risk_aversion_lambda, min_max_hedge)` | Heatmap-table `Figure`. |

`local_ccys` may be a single string (all assets share a currency — the cross rate/forward are
computed once) or a per-asset `pd.Series` (resolved inside the loop). The optimizer knobs
(`span`, `risk_aversion_lambda`, `min_max_hedge`) are passed through explicitly rather than relying
on `compute_fx_optimal_hedge`'s defaults.

---

## `FactorsData`

`FactorsData` now lives in `qis.portfolio.risk.factors_data` and remains available as
`qis.FactorsData`. See the [factor risk-model guide](../../../docs/factor_risk_models.md)
for its generic price-panel and optional factor-enum contract. Its CSV basename is unchanged.

---

## Examples and the data backends

The offline `fx_returns_and_accounting_example.py` covers typed SIMPLE/LOG returns,
cash-relative returns, exact notional payoffs, explicit synthetic index normalisation,
monthly cash measurement, zero/nonzero-net local financing, hedge resets and settlement.
Each runnable file uses a `Locals` dispatcher.

* **`fx_rates_data_bloomberg_example.py`** uses a provider-specific basket of term rates
  and overnight CAD/SGD benchmarks (`CAONREPO Index` and `SIBCSORA Index`). It requires a
  Bloomberg terminal and `bbg-fetch`. Imports are deferred; unknown opening rates remain
  missing. The example does not persist or update any production dataset.
* **`fx_rates_data_yahoo_example.py`** uses `yfinance` for spots and a USD Treasury yield.
  Its non-USD rates are explicitly stylised USD-rate differentials, for illustration only.
  They are not validated funding curves or market forward prices.

Provider availability, benchmark choices, quote units and historical coverage belong to the
consuming data pipeline. Generic QIS analytics take the supplied frames and currency IDs;
they do not infer a currency universe, download a source or certify a vendor rate's basis.
Factor-universe definitions and factor construction also remain with their consumer; the
shared `FactorsData` price container belongs to `qis.portfolio.risk`.

---

## Fixed constants (do not rename)

Local/Ramen CSVs depend on these file names:

* `FactorsData` CSV → `futures_risk_factors` (`factors_data.FILE_NAME`).
* FX CSVs → `fx_hedging_data` (`fx_rates_data` `load_fx_rates_data` default).

---

## Quick start

```python
import qis as qis
from qis.fx import FxRatesData

# Load the FX universe (or build from Bloomberg in rosaa).
fx = FxRatesData.load(local_path=qis.local_path.get_resource_path())

# Convert a USD-asset price panel into CHF, fully hedged, as excess returns.
results = qis.compute_fx_returns(fx, qis.FxAssetReturnsRequest(
    prices=prices, hedge_ratios=hedge_ratios, local_ccys=local_ccys,
    reference_ccy='CHF', frequency='ME',
    return_type=qis.FxReturnType.REFERENCE_EXCESS,
    coordinate=qis.FxReturnCoordinate.LOG))
returns = {frequency: result.values for frequency, result in results.items()}


# Hedge-overlay study for a panel:
from qis.fx.reports.fx_hedging_report import compute_multi_asset_fx_hedging
out = compute_multi_asset_fx_hedging(asset_prices=prices, fx_rates_data=fx,
                                     local_ccys='USD', reference_ccy='CHF', freq='ME')
out['sharpes']      # Sharpe table, asset × strategy
out['last_hedges']  # current-snapshot hedge ratios per strategy
```
