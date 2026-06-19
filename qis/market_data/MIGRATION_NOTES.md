# qis/market_data — migration notes

New qis submodule holding the two generic market-data containers lifted out of
`rosaa.market_data`. Verified against qis 4.2.7: all qis API calls resolve, all
files parse, intra-package imports resolve, no seaborn, LF endings, `verbouse`
typo fixed. Drop `qis/market_data/` into the qis package and add
`from qis.market_data.__init__ import *` to `qis/__init__.py`.

## What is here (qis layer)

```
qis/market_data/
  factors_data.py            FactorsData — generic factor-price container.
                             Accepts an injected `factors` enum (e.g. MATF
                             RiskFactors) for typed access/validation; stays
                             generic when factors=None. load() + from_sql() hook.
  fx_rates_data.py           FxRatesData container + load_fx_rates_data (CSV).
                             load_fx_rates_data returns (fx_spots, domestic_rates)
                             only; usd_assets is examples-only (see examples/).
                             Pure: cross rates, CIP forwards, carry, reference-
                             ccy conversion. Imports compute_performance_* from
                             fx_hedging.
  fx_hedging.py              Pure FX math (leaf): compute_local_and_fx_return,
                             compute_performance_of_local_ccy_asset_in_reference_ccy,
                             compute_fx_vol_beta, compute_fx_optimal_hedge.
  reports/fx_hedging_report.py
                             Research pipeline on qis.plots (seaborn removed):
                             run_asset_fx_hedging_report,
                             compute_multi_asset_fx_hedging,
                             plot_multi_asset_fx_hedging_report.
  examples/fx_hedging_example.py
                             load_usd_assets (examples-only USD benchmark loader)
                             + run_local_test hedging demo. Uses qis.local_path,
                             not rosaa; data-build (CREATE_DATA) dropped — that
                             stays in rosaa.
  tests/                     factors_data_test (pytest), fx_cip_identity_test
                             (assertions), fx_rates_data_test (integration).
```

Import graph (no cycles): `fx_hedging` (leaf) ← `fx_rates_data` ← `reports`;
`factors_data` independent. `compute_multi_asset_fx_hedging` lives in `reports`,
not `fx_hedging`, because it takes a `FxRatesData` (keeping it in the pure layer
would cycle).

## What STAYS in rosaa (construction / production — do not move to qis)

From `matf_risk_model.py`:
- `RiskFactors` enum (renamed from `MatfRiskFactors`) + `FACTOR_VOLS` — the
  MATF factor-set identity. rosaa constructs `FactorsData(prices, factors=RiskFactors)`.
- factor construction: `compute_risk_factors`, the nine `compute_*_factor`,
  `create_bbg_price_data`, `load_base_futures_prices`, `load_rates`,
  `compute_beta_loading_signs_for_matf`, the portfolio/vol-target helpers.
- new free function `update_matf_factors(local_path)` ← the demoted
  `TradableMultiFactorsData.update()` body (build factors, factsheet, save CSV).

From `fx_rates_data.py`:
- `create_fx_rates_data` (Bloomberg build) + the ticker constants
  `USD_ASSETS`, `FX_SPOTS`, `DOMESTIC_RATES`.

## Fixed constants (do not rename — local/Ramen data depends on them)

- `FactorsData` CSV: `futures_risk_factors` (in factors_data.FILE_NAME).
- FX CSVs: `fx_hedging_data` (in fx_rates_data.FILE_NAME). rosaa's
  `create_fx_rates_data` must write the same name.

## Consumer follow-through (separate steps)

- rosaa: re-point `from rosaa.market_data...` to `from qis import FactorsData,
  FxRatesData`; pass `factors=RiskFactors`; replace `.update()` callers with
  `update_matf_factors`; keep `matf_risk_model_test` + the CREATE_DATA case.
- privateassets: load factors via `qis.FactorsData.load(...)` (factors=None);
  this removes the last rosaa dependency.
