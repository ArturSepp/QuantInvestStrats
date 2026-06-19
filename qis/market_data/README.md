# `qis.market_data`

Generic market-data containers and FX-hedging analytics, lifted out of `rosaa.market_data` into
the open `qis` package. The module holds two dataclass containers (`FxRatesData`, `FactorsData`),
a leaf of pure FX math (`fx_hedging`), and a small FX-hedging reporting pipeline (`reports`). It is
deliberately free of any production wiring: the Bloomberg builders, the ticker universes, and the
MATF factor model stay in `rosaa` (see [What stays in rosaa](#what-stays-in-rosaa)).

```
qis/market_data/
  __init__.py                 exports FxRatesData, load_fx_rates_data, FactorsData
  fx_rates_data.py            FxRatesData container + load_fx_rates_data (CSV reader)
  fx_hedging.py               pure FX math (leaf): local/FX decomposition, vol/beta, optimal hedge
  factors_data.py             FactorsData — generic tradable-factor price container
  reports/
    fx_hedging_report.py      single- and multi-asset FX-hedging tearsheets (on qis.plots)
  tests/                      factors_data_test, fx_cip_identity_test, fx_rates_data_test

qis/examples/market_data/     runnable demos (under qis.examples, per the package convention)
  fx_rates_data_yahoo_example.py     build FxRatesData from free Yahoo data + exercise it
  fx_rates_data_bloomberg_example.py build FxRatesData from Bloomberg via bbg-fetch + exercise it
  fx_cip_identity_yahoo_example.py   covered-interest-parity identity check on Yahoo data
  fx_hedging_yahoo_example.py        single- and multi-asset hedging demo on Yahoo data
  fx_hedging_example.py              hedging demo on the CSV-backed production universe
```

Import graph (no cycles): `fx_hedging` (leaf) ← `fx_rates_data` ← `reports`; `factors_data` is
independent. `compute_multi_asset_fx_hedging` lives in `reports` (not `fx_hedging`) because it takes
a `FxRatesData`; keeping it in the pure layer would create a cycle.

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

`__post_init__` forward-fills `fx_spots` and reindexes `domestic_rates` onto the spot calendar, so a
container is ready to use the moment it is constructed.

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
| `get_local_rate(freq, local_ccy, annualise=False)` | Domestic short rate at `freq`; per-period (`annualise=False`) or annualised. |
| `get_forward_rate_for_local_ccy(local_ccy, reference_ccy, freq, is_log_returns=False)` | Per-period CIP forward premium (see below). |
| `get_daily_carry_local_return(local_ccy, reference_ccy, is_log_returns=False)` | `(spot_return, carry_return)` daily legs. |
| `get_fx_total_return_nav(local_ccy, reference_ccy, time_period, freq)` | NAV of spot + carry. |
| `get_carry_fx_return_nav(local_ccy, reference_ccy, is_normalise_by_spot_vol, time_period, freq, is_causal)` | Carry-only NAV (vol-matched to spot). |

**CIP forward premium.** With annualised short rates `r_loc`, `r_ref` and `dt = 1 / annualisation_factor(freq)`,
the per-period forward premium earned on the local currency is `(1 + dt*r_loc)/(1 + dt*r_ref) - 1`
(simple) or `log((1 + dt*r_loc)/(1 + dt*r_ref))` (log). This is the financing carry that a fully
hedged position pays/earns relative to the reference currency.

**Look-ahead note.** `get_carry_fx_return_nav(is_normalise_by_spot_vol=True)` matches the carry NAV's
realised vol to the spot series. With `is_causal=False` (default) the mean/variance are computed on
the full sample — fine for ex-post reporting, **not** for backtests. Pass `is_causal=True` for
expanding-window statistics.

---

## Cash NAVs

| Method | Purpose |
|---|---|
| `build_local_cash_nav(local_ccy, freq='B')` | Money-market NAV compounding the local short rate (rate lagged one period — no look-ahead). |
| `build_cross_fx_cash_nav(local_ccy, reference_ccy, freq='B', output_freq='ME')` | Foreign cash held unhedged by a reference-ccy investor; the synthetic series used to isolate Fx β ≈ +1 for a foreign cash leg. Returns an empty series when `local_ccy == reference_ccy`. |

---

## Multi-asset FX-translation pipeline

The primary entry point consumed by the covariance estimator and the alpha aggregator:

```python
returns_by_freq = fx.compute_fx_adjusted_returns(
    prices=prices,            # native-ccy price panel (columns = assets)
    hedge_ratios=hedge_ratios,# per-asset hedge ratio in [0, 1] (Series)
    local_ccys=local_ccys,    # per-asset currency (Series)
    reference_ccy='USD',
    freq='ME',                # str (one freq) OR pd.Series (per-asset → grouped)
    is_log_returns=True,
    is_excess_returns=False)
# -> {freq_string: returns_dataframe}
```

* `compute_returns_in_reference_ccy(...)` is the single-frequency core (returns `(navs, returns)`);
  `compute_fx_adjusted_returns(...)` wraps it in a per-frequency groupby and replaces structural
  zero returns with `NaN` (so step-function PE/PD series don't bias rolling β toward zero).
* `compute_returns_adjusted_by_local_rate(asset_prices, local_ccys, freq, is_log_returns)` returns
  `(local_returns, excess_returns, local_rates)` in each asset's **own** currency (no FX conversion).
* `fetch_local_rates(local_ccys, freq, annualise=False)` builds the per-asset short-rate panel.

**Excess-return convention.** When `is_excess_returns=True` the excess is taken over the **reference**
currency risk-free rate, not the asset's local rate. Under CIP the hedged forward premium already
converts the asset's native rf into the reference rf, so subtracting the reference rf is the correct
and non-double-counting choice.

---

## FX hedging analytics (`fx_hedging`)

Pure, stateless functions on plain price/return Series:

| Function | Returns |
|---|---|
| `compute_local_and_fx_return(asset_price_local_ccy, local_to_reference_fx_rate, freq, is_log_returns)` | `(local_return, fx_return)` legs. |
| `compute_performance_of_local_ccy_asset_in_reference_ccy(..., hedge_ratio, freq, is_log_returns)` | `(hedged_nav, hedged_return)` at a given hedge ratio. |
| `compute_fx_vol_beta(..., freq, span)` | `(fx_vol, fx_beta)` — EWMA annualised FX vol and the local-on-FX beta. |
| `compute_fx_optimal_hedge(..., freq, span, risk_aversion_lambda, min_max_hedge)` | `(optimal, max_carry, beta_hedged)` hedge-ratio series. |

**Hedged return.** `hedged_return = local_return*(1 + fx_return) + (1 - h)*fx_return - h*forward_premium`,
with the hedge ratio `h` and the forward premium **lagged one period** so the return over `[t-1, t]`
uses the hedge decided at `t-1` (no look-ahead). The first period is forced to 0 so the NAV starts at 1.0.

**Hedge strategies** (from EWMA vol/beta and the annualised forward premium):
`carry_ratio = (annualised_forward / fx_var) / (2*risk_aversion_lambda)`; `Max Carry = 1 - carry_ratio`;
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

A generic container for tradable factor prices:

```python
@dataclass
class FactorsData:
    factors_prices: pd.DataFrame          # columns = factor names, index = datetime
    factors: Optional[Type[Enum]] = None  # optional str-valued Enum for typed access/validation
```

It is factor-set agnostic. Pass `factors=` a str-valued `Enum` (e.g. a MATF `RiskFactors`) for typed
access and column validation; leave it `None` to stay fully generic. `load(local_path, file_name='futures_risk_factors', factors=None, time_period=None)` reads the CSV; `from_sql(...)` is a forward hook
for database-backed loading. Accessors: `factor_names`, `get_factor_prices(factor, time_period)`,
`get_prices(time_period)`.

---

## Examples and the data backends

`qis/examples/market_data/` provides two ways to build an `FxRatesData` and exercise it (cross rates, CIP forward
premia, FX total-return NAVs, money-market cash NAVs, reference-currency translation, and the hedging
reports). Each file has a `LocalTests` dispatcher; run a case from `__main__`.

* **`fx_rates_data_bloomberg_example.py`** — the production data path: real 3M domestic rates
  (overnight for SGD) and the full currency set (EUR/GBP/CHF/JPY/AUD/CAD/NZD/HKD/SGD/NOK/SEK), via
  `bbg-fetch`. The ticker universe and the spot/rate construction are extracted from rosaa's
  `create_fx_rates_data`; the example builds the container in-memory (the persisted-CSV builder stays
  in rosaa). Requires a Bloomberg terminal + `bbg-fetch`; the import is deferred so the module loads
  without one.
* **`fx_rates_data_yahoo_example.py`** — a free, no-terminal path (`yfinance`).

**On the Yahoo path, FX spots are real but rates are not.** Yahoo's only reliable interest-rate series
are US Treasuries — `^IRX` (13-week T-bill), `^FVX` (5y), `^TNX` (10y), `^TYX` (30y). It has **no
usable non-USD 3M rate**:

* **CHF** — there is a `SARON.SW` ticker, but SARON is an **overnight** rate (not 3M) and its Yahoo
  feed is **stale** (it reports a negative-rate-era value frozen at an old close, while the real rate
  is now positive). It is therefore not used.
* **EUR / GBP / JPY / AUD / CAD / NZD** — no 3M rate on Yahoo at all.

So the Yahoo example takes the USD 3M from `^IRX` (real) and approximates the others as the USD rate
plus a small, clearly-labelled stylised differential (`_ILLUSTRATIVE_3M_SPREAD_VS_USD`) — **for
illustration only**. For real rates and the full currency set, use the Bloomberg example.

---

## What stays in rosaa

This module is the **generic** layer. The following are construction/production code and remain in
`rosaa` (see `MIGRATION_NOTES.md`):

* `create_fx_rates_data` (the Bloomberg build) and the ticker constants `USD_ASSETS`, `FX_SPOTS`,
  `DOMESTIC_RATES`.
* the MATF factor model `matf_risk_model.py`: the `RiskFactors` enum + `FACTOR_VOLS`, the factor
  construction functions, `create_bbg_price_data`, and `update_matf_factors`. rosaa constructs
  `FactorsData(prices, factors=RiskFactors)`.

## Fixed constants (do not rename)

Local/Ramen CSVs depend on these file names:

* `FactorsData` CSV → `futures_risk_factors` (`factors_data.FILE_NAME`).
* FX CSVs → `fx_hedging_data` (`fx_rates_data` `load_fx_rates_data` default).

---

## Quick start

```python
import qis as qis
from qis.market_data import FxRatesData

# Load the FX universe (or build from Bloomberg in rosaa).
fx = FxRatesData.load(local_path=qis.local_path.get_resource_path())

# Convert a USD-asset price panel into CHF, fully hedged, as excess returns.
returns = fx.compute_fx_adjusted_returns(
    prices=prices,
    hedge_ratios=hedge_ratios,   # pd.Series, per asset, in [0, 1]
    local_ccys=local_ccys,       # pd.Series, per asset, e.g. 'USD'
    reference_ccy='CHF',
    freq='ME',
    is_log_returns=True,
    is_excess_returns=True)

# Hedge-overlay study for a panel:
from qis.market_data.reports.fx_hedging_report import compute_multi_asset_fx_hedging
out = compute_multi_asset_fx_hedging(asset_prices=prices, fx_rates_data=fx,
                                     local_ccys='USD', reference_ccy='CHF', freq='ME')
out['sharpes']      # Sharpe table, asset × strategy
out['last_hedges']  # current-snapshot hedge ratios per strategy
```
