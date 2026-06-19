# qis examples

Worked examples organised by `qis` sub-package. Each script is runnable
standalone and either prints output or shows a matplotlib figure.

Most examples pull data from `yfinance`. A few use Bloomberg via `bbg_fetch`
and require an open Bloomberg terminal — those are noted below.

## Layout

```
examples/
├── _helpers/                     shared helpers, imported by examples
├── perfstats/                    qis.perfstats — performance metrics on price series
├── models/                       qis.models — EWM, regression, vol estimation, bootstrap
├── regimes/                      qis.perfstats.regime_classifier — regime-conditional analytics
├── portfolios/                   qis.backtest_model_portfolio — backtests
├── factsheets/                   qis.generate_*_factsheet — full factsheets
├── plots/                        qis.plots — plotting primitives showcase
├── utils/                        qis.utils — date schedules
├── case_studies/                 cross-cutting domain studies (VIX, credit)
└── market_data/                  qis.market_data — FX rates, CIP/carry, FX hedging
```

## perfstats — performance metrics

| File | What it shows |
|---|---|
| `quickstart.py` | Minimal: `plot_prices`, `plot_ra_perf_table`, `plot_ra_perf_table_benchmark`. Same code shown in the package README. |
| `full_performance_report.py` | Five-figure summary on a yfinance universe (ETFs, crypto, vol ETFs…). Uses `_helpers.reporting_helpers`. |
| `sharpe_vs_sortino.py` | Sharpe vs Sortino across return frequencies. |
| `risk_return_frontier.py` | Bond-ETF risk/return scatter using `compute_ra_perf_table`. |
| `rolling_performance.py` | Rolling per-annum returns via `compute_rolling_perf_stat`. **Bloomberg.** |
| `cboe_vol_strats_perf.py` | CBOE SVRPO vol strat vs SPY — uses a CSV ship in `qis.get_resource_path()`. |
| `miss_best_worst_days_impact.py` | Performance with the best / worst N days per month removed. |
| `infrequent_returns_interpolation.py` | `interpolate_infrequent_returns` for monthly/quarterly hedge-fund-like series. |
| `unsmoothing_and_delevering.py` | End-to-end walkthrough of `delever_returns`, `implied_leverage`, `unsmooth_returns_ar1_ewma` and `unsmooth_returns_glm` on a bundled OCSL/GCF dataset. |

## models — EWM, regression, vol estimation

| File | What it shows |
|---|---|
| `ewm_kernels.py` | Numba-vs-pandas timing benchmark of `ewm_recursion`, `compute_ewm`, and a covariance-tensor cross-check. |
| `ewm_linear_model.py` | Time-varying multivariate factor loadings via `EwmLinearModel`. |
| `ewm_correlation_table.py` | EWMA correlation heatmap-table via `plot_returns_ewm_corr_table`. |
| `multivariate_ols.py` | `fit_multivariate_ols` with intercept / no-intercept. |
| `rolling_correlations.py` | Rolling 3m/6m/12m correlations between BTC and QQQ. |
| `ohlc_volatility_estimators.py` | OHLC realised-vol estimators via `estimate_hf_ohlc_vol`. |
| `crypto_intraday_vol.py` | BTC hourly EWMA vol — handles 24/7 markets without weekend gaps. |
| `overnight_intraday_returns.py` | Decomposes close-to-close returns into overnight + intraday components. |
| `bootstrap_analysis.py` | Block bootstrap of price paths via `bootstrap_price_data`. |

## regimes — regime-conditional analytics

| File | What it shows |
|---|---|
| `bull_bear_normal_sharpe.py` | Bull / bear / normal regime Sharpe via `BenchmarkReturnsQuantilesRegime`. |
| `boxplot_conditional.py` | Conditional return boxplots by VIX regime via `df_boxplot_by_classification_var`. |
| `seasonality.py` | Returns conditional on calendar month. |
| `us_election_regimes.py` | Returns conditional on divided / unified US government. **Bloomberg.** |

## portfolios — backtests

| File | What it shows |
|---|---|
| `balanced_60_40.py` | 60/40 SPY/IEF with management fee — `backtest_model_portfolio`. |
| `balanced_60_40_with_btc.py` | Impact of adding a 2% BTC sleeve to a 60/40 portfolio. |
| `constant_notional_short.py` | Constant-notional vs constant-weight short SPY simulation. |
| `leveraged_etf_strategies.py` | SSO/IEF leveraged-ETF backtest with rebalancing costs. |
| `long_short.py` | Long IEF / short LQD pair (Treasury duration vs IG credit). |
| `vol_target_and_trend.py` | Vol-target + trend-following sweep via `qis.portfolio.strats.quant_strats_delta1`. |

## factsheets — full multi-page reports

| File | What it shows |
|---|---|
| `strategy.py` | `generate_strategy_factsheet` on a volparity portfolio. |
| `strategy_benchmark.py` | `generate_strategy_benchmark_factsheet_plt` — strategy vs benchmark. |
| `multi_assets.py` | `generate_multi_asset_factsheet` on an asset-class universe. |
| `multi_strategy.py` | `generate_multi_portfolio_factsheet` over a span sweep. |
| `strategy_reporting_frequencies.py` | `generate_strategy_factsheet` reproduced across the DAILY/WEEKLY/MONTHLY/QUARTERLY × {long, short} reporting-frequency grid via `fetch_default_report_kwargs`, on one volparity portfolio. |
| `strategy_benchmark_reporting_frequencies.py` | `generate_strategy_benchmark_factsheet_plt` across the same reporting-frequency grid — volparity vs equal-weight. |
| `multi_strategy_reporting_frequencies.py` | `generate_multi_portfolio_factsheet` across the same grid, on a vol-parity span sweep. |
| `multi_assets_reporting_frequencies.py` | `generate_multi_asset_factsheet` across the same grid, on the asset-class universe (no backtest). |
| `momentum_indices.py` | Multi-asset factsheet on momentum index family. **Bloomberg.** |
| `europe_futures.py` | Strategy factsheet on volume-weighted European futures. **Bloomberg.** |
| `hedge_funds.py` | Multi-asset factsheet on HFRX/HFRI/CTA index family. **Bloomberg.** |
| `bbg_universe.py` | Multi-asset factsheet template for any Bloomberg ticker dict. **Bloomberg.** |
| `pybloqs_factsheets.py` | Optional: pybloqs-rendered factsheets (RA-perf / multi-portfolio / strategy-benchmark). Requires `pybloqs` and a small jinja patch — see file docstring. |

## plots — plotting primitives

| File | What it shows |
|---|---|
| `dual_axis_figure.py` | Building a 2-axis time-series plot via `plot_time_series_2ax`. |
| `scatter_with_regression.py` | Scatter + regression diagnostics with synthetic data. |

## utils — date schedules

| File | What it shows |
|---|---|
| `option_rolls_schedule.py` | `generate_fixed_maturity_rolls` for option/futures roll calendars. |

## case_studies — cross-cutting domain studies

| File | What it shows |
|---|---|
| `credit_spreads.py` | Credit spread vs equity / rates beta, regime regression. **Bloomberg.** |
| `vix_beta_to_equities_bonds.py` | Rolling beta of VIX ETF to SPY/TLT. |
| `vix_conditional_returns.py` | Conditional returns on short-front-month VIX strategy. **Bloomberg.** |
| `vix_spy_scatter_by_year.py` | VIX changes vs SPY returns scattered by year. |
| `vix_term_structure.py` | VIX term-structure correlation with SPX returns. **Bloomberg.** |

## market_data — FX rates & hedging

| File | What it shows |
|---|---|
| `fx_rates_data_yahoo_example.py` | Build `FxRatesData` from free `yfinance` FX spots; cross rates, CIP forward premia, FX total-return NAVs, cash NAVs, reference-ccy translation. USD rate from `^IRX`, others stylised. |
| `fx_rates_data_bloomberg_example.py` | The same, built from Bloomberg via `bbg_fetch` — real 3M rates, full currency set. **Bloomberg.** |
| `fx_cip_identity_yahoo_example.py` | Covered-interest-parity check: USD excess vs CHF-hedged excess agree to within bp. |
| `fx_hedging_yahoo_example.py` | Single/multi-asset FX hedging: optimal/carry/beta ratios, hedged NAVs, EWM FX vol/beta, hedge reports. |
| `fx_hedging_example.py` | Hedging demo on the CSV-backed production universe (`load_fx_rates_data`, `load_usd_assets`). |

---

### Output files

Examples write generated PDFs / PNGs to `qis.local_path.get_output_path()`.
Output figures committed to `examples/figures/` are gitignored — the path
exists for README assets only and is regenerated on demand.
