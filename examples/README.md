# qis examples

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-05-05](https://github.com/ArturSepp/QuantInvestStrats/commit/8f36d19bc6580e06cf9def5772d21837fd6d427a)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Examples for [qis](https://github.com/ArturSepp/QuantInvestStrats).
See the [software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Worked examples organised by `qis` sub-package. They are repository documentation and are not
included in the installed wheel. Run a script from the repository root as a module, for example
`python -m examples.perfstats.quickstart`; each either prints output or shows a matplotlib figure.

Start with the offline examples below; they use the frozen synthetic universe and need only
the core installation. Yahoo examples need `pip install "qis[data]"` and network access.
Bloomberg examples require `bbg_fetch` and an open Bloomberg terminal; they are marked explicitly.

## Layout

```
examples/
├── _helpers/                     shared helpers, imported by examples
├── getting_started/              core-only, offline first results
├── data_fetching/                optional vendor universe definitions
├── perfstats/                    qis.perfstats — performance metrics on price series
├── models/                       qis.models — EWM, regression, vol estimation, bootstrap
├── regimes/                      qis.perfstats.regime_classifier — regime-conditional analytics
├── portfolios/                   qis.backtest_model_portfolio — scheduled backtests
├── discrete_portfolio/           bar-by-bar orders, fills, and trade ledgers
├── factsheets/                   qis.generate_*_factsheet — full factsheets
├── plots/                        qis.plots — plotting primitives showcase
├── utils/                        qis.utils — date schedules
├── case_studies/                 cross-cutting domain studies (VIX, credit)
└── market_data/                  qis.market_data — FX rates, CIP/carry, FX hedging
```

## getting_started — start here, offline

| File | What it shows |
|---|---|
| [first_chart.py](getting_started/first_chart.py) | A quarterly 60/40 synthetic backtest and first chart. |
| [offline_quickstart.py](getting_started/offline_quickstart.py) | A live-universe-aware weight schedule, backtest, performance table and accounting checks. Included in the [quickstart guide](../docs/quickstart.md). |

## perfstats — performance metrics

| File | What it shows |
|---|---|
| `quickstart.py` | Yahoo-data plotting walkthrough: prices, drawdowns and risk-adjusted tables. Uses YAML `OUTPUT_PATH`; the package README starts with offline data. |
| `full_performance_report.py` | Five-figure summary on a yfinance universe (ETFs, crypto, vol ETFs…). Uses `_helpers.reporting_helpers`. |
| `sharpe_vs_sortino.py` | Sharpe vs Sortino across return frequencies. |
| `risk_return_frontier.py` | Bond-ETF risk/return scatter using `compute_ra_perf_table`. |
| `rolling_performance.py` | Rolling per-annum returns via `compute_rolling_perf_stat`. **Bloomberg.** |
| `cboe_vol_strats_perf.py` | [Cboe SVRPO](https://www.cboe.com/us/indices/dashboard/svrpo/) market-neutral volatility risk-premia index vs SPY — downloads Cboe's public CSV and Yahoo benchmark/rate data. No local resource file required. |
| `miss_best_worst_days_impact.py` | Performance with the best / worst N days per month removed. |
| `infrequent_returns_interpolation.py` | Yahoo QQQ quarterly returns interpolated to monthly using SPY as the liquid reference. |
| `timeseries_backfill.py` | Extend a newer provider history backwards with `bfill_timeseries`, preserving its recent price path. |
| `turnover_conventions.py` | Yahoo SPY/TLT comparison of all four turnover conventions for a 100% funded tactical portfolio and the same portfolio at 2x leverage. |
| `unsmoothing_and_delevering.py` | End-to-end walkthrough of `delever_returns`, `implied_leverage`, `unsmooth_returns_ar1_ewma` and `unsmooth_returns_glm` on a bundled OCSL/GCF dataset. |

## models — EWM, regression, vol estimation

| File | What it shows |
|---|---|
| `ewm_kernels.py` | Numba-vs-pandas timing benchmark of `ewm_recursion`, `compute_ewm`, and a covariance-tensor cross-check. |
| `ewm_linear_model.py` | Time-varying multivariate factor loadings via `EwmLinearModel`. |
| `ewm_correlation_table.py` | EWMA correlation heatmap-table via `plot_returns_ewm_corr_table`. |
| `multivariate_ols.py` | `fit_multivariate_ols` with intercept / no-intercept. |
| `rolling_correlations.py` | Rolling 3m/6m/12m correlations between BTC and QQQ. |
| `crypto_intraday_vol.py` | BTC hourly EWMA vol — handles 24/7 markets without weekend gaps. |
| `overnight_intraday_returns.py` | Decomposes close-to-close returns into overnight + intraday components. |
| `bootstrap_analysis.py` | Fifty illustrative block-bootstrap price paths by default; the larger autocorrelation sweep is a separate `Locals` branch. Uses YAML `OUTPUT_PATH`. |
| [bootstrap_convention.py](models/bootstrap_convention.py) | Offline comparison of circular versus truncated stationary-bootstrap blocks. |
| [ar_bootstrap_gaps.py](models/ar_bootstrap_gaps.py) | Offline illustration of preserving lag spacing when fitting an AR process across missing observations. |
| `pca_variance_explained.py` | Rolling EWMA-covariance PCA variance shares on the seeded synthetic universe. |

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
| `brinson_attribution.py` | Offline BHB sector attribution with Frongello linking, prior holdings, native-date trading costs, monthly/quarterly reconciliation and optional QIS PDF/PNG/CSV export. See [methodology](../docs/brinson_attribution.md). |
| `balanced_60_40.py` | 60/40 SPY/IEF with management fee — `backtest_model_portfolio`. |
| `balanced_60_40_with_btc.py` | Impact of adding a 2% BTC sleeve to a 60/40 portfolio. |
| `constant_notional_short.py` | Constant-notional vs constant-weight short SPY simulation. |
| `leveraged_etf_strategies.py` | SSO/IEF and funded SPY/IEF backtests with rebalancing costs; uses YAML `OUTPUT_PATH`. |
| `long_short.py` | Long IEF / short LQD pair (Treasury duration vs IG credit). |
| `ex_anti_tracking_error_and_risk.py` | Offline ex-ante TE, benchmark beta, and Euler marginal TE through `RiskModel`. |
| `ex_post_tracking_error_and_risk.py` | Offline realised EWMA TE, whole-sample TE/IR, and EWMA beta/annualised alpha. |
| [gaussian_var_monitoring.py](portfolios/gaussian_var_monitoring.py) | Offline supplied-covariance Gaussian VaR monitoring for two allocations, with a benchmark and closed-form control. See the [Gaussian VaR guide](../docs/gaussian_var.md). |
| `model_layer_attribution_simulated.py` | Offline, seeded layer and two-feature simulation for `compute_model_layer_alpha_beta_attribution` and `compute_model_feature_alpha_beta_attribution`: HAC(3) intervals, exact identities, return bridge, additive cumulative alpha and grouped Shapley sensitivity. Source of the figures in `docs/model_layer_attribution.md`. |
| `vol_target_and_trend.py` | Vol-target + trend-following sweep via `examples.portfolios.strats.qis_delta1`. |
| `seasonality_backtest.py` | Point-in-time calendar-month seasonality with annual trailing-window refits. |
| `instrument_portfolio_stress.py` | Offline funded and mixed books through the public portfolio/report interface: call/put decompositions, futures, local FX, Credit/Carry family splits, historical replay and optional PDF/Excel/CSV output. See the [instrument guide](../docs/portfolio_stress.md). |
| [observed_index_stress.py](portfolios/observed_index_stress.py) | Offline USD/EUR observed-index stress comparison using VIX point and log changes, with portfolio prediction R-squared and optional report output. |
| `composite_payoff_stress.py` | Offline consumer-owned terminal-knockout payoff through `HoldingPayoff`, preserving vanilla valuation and shared-response sensitivities; optional standard report. |
| `factor_stress_testing.py` | Offline assigned-model stress workflow: direct versus correlated targets, joint credit anchors, nonlinear P&L attribution and one-month conditional prediction bands; optional CSV/PNG/PDF export. See [analytics and formulas](../docs/stress_testing.md). |
| [account_equity_stress.py](portfolios/account_equity_stress.py) | Offline account equity, signed borrowing and equity after stress; optional report export. |
| [stress_testing_with_options.py](portfolios/stress_testing_with_options.py) | Offline option repricing across stress scenarios. Requires the separate `vanilla-option-pricers` package; see the [options guide](../docs/stress_testing_with_options.md). |
| [lagged_weight_implementation.py](portfolios/lagged_weight_implementation.py) | Offline demonstration of weight implementation lag measured in observations. |
| [static_weight_with_missing_prices.py](portfolios/static_weight_with_missing_prices.py) | Offline static allocations with staggered instrument starts and an explicit target schedule. |
| [optimal_leverage.py](portfolios/optimal_leverage.py) | Illustrative utility-weight and leverage sensitivity calculations. |

## discrete_portfolio — event-based backtests

| File | What it shows |
|---|---|
| `discrete_trend_backtest.py` | Long/flat momentum on free SPY 5-minute bars, with next-observation fills, a trade ledger and `PortfolioData`. Uses a compact factsheet below 63 daily marks and a full daily report for longer histories. Requires `qis[data]`; for 1-minute bars set `INTERVAL="1m"` and `PERIOD="5d"`. |

## factsheets — full multi-page reports

| File | What it shows |
|---|---|
| `strategy.py` | `generate_strategy_factsheet` on a volparity portfolio. |
| `strategy_benchmark.py` | `generate_strategy_benchmark_factsheet_plt` — strategy vs benchmark. |
| `multi_assets.py` | `generate_multi_asset_factsheet` on an asset-class universe. |
| `multi_strategy.py` | `generate_multi_portfolio_factsheet` over a span sweep. |
| [sharpe_difference_table.py](factsheets/sharpe_difference_table.py) | Offline paired HAC Sharpe comparisons from `MultiPortfolioData`, with a selected base and NAV window. Uses month-end returns with inferred annualisation, synthetic backtests with transaction costs or existing portfolio NAVs, and display or PNG/CSV export via `Locals.EXPORT`. |
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
| [cluster_dendrograms.py](plots/cluster_dendrograms.py) | Offline dendrogram and cluster-membership plots, including supplied axes and optional export. |

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
| `fx_hedging_example.py` | Yahoo FX/ETF demo by default, with illustrative non-USD rate spreads. `use_local_data=True` selects YAML `RESOURCE_PATH`; `input_dir` overrides it for reader-supplied `fx_hedging_data_fx_spots.csv`, `fx_hedging_data_domestic_rates.csv` and `fx_hedging_data_usd_assets.csv`; these files are not bundled. |

## data_fetching — optional vendor input

| File | What it shows |
|---|---|
| [long_vol_qis_universe.py](data_fetching/long_vol_qis_universe.py) | Long-volatility bank-index universe definitions, download and CSV persistence. **Bloomberg.** |

## Shared helpers

These support the workflows above; they are not standalone demonstrations.

| File | Purpose |
|---|---|
| [_helpers/reporting_helpers.py](_helpers/reporting_helpers.py) | Reusable five-figure performance report layout. |
| [portfolios/strats/qis_delta1.py](portfolios/strats/qis_delta1.py) | Vol-target and trend strategy helpers. |
| [portfolios/strats/seasonality_strat.py](portfolios/strats/seasonality_strat.py) | Point-in-time seasonality strategy helpers. |

---

### Input and output directories

Examples use [`qis.local_path`](../src/qis/local_path.py) as the single directory resolver.
No configuration is required: packaged defaults are `~/.qis/resources` for resource files
and caches, and `~/.qis/outputs` for reports. Keep `src/qis/settings.yaml` unchanged: it
contains release-checked portable defaults, not private paths or credentials. See the
[configuration guide](../src/qis/README.md) for permanent Windows setup and all settings.

Optionally create private YAML outside the checkout/package, for example
`C:/Data/qis/private-settings.yaml`:

```yaml
RESOURCE_PATH: 'C:/Data/qis/resources'
OUTPUT_PATH: 'C:/Data/qis/outputs'
```

Set its location before importing qis or starting an example:

```powershell
$env:QIS_SETTINGS_PATH = 'C:/Data/qis/private-settings.yaml'
& 'C:/Python/QuantInvestStrats312/Scripts/python.exe' -m examples.factsheets.multi_assets
```

On Linux/macOS, use `export QIS_SETTINGS_PATH=/absolute/path/private-settings.yaml`,
with paths appropriate to that machine. No private configuration file is generated or
modified by an example.

The precedence is **explicit directory argument > explicit YAML file >
`QIS_SETTINGS_PATH` YAML > packaged defaults**. Private YAML can contain just the keys
needed by a workflow; other keys retain the packaged defaults. A missing selected YAML,
malformed YAML, or invalid directory value produces an actionable error, not a
silent temporary-directory fallback.

```python
from qis import local_path

# Read defaults or the private configuration selected by QIS_SETTINGS_PATH.
resources = local_path.get_resource_path(create=True)  # Deliberate download/cache directory.
outputs = local_path.get_output_path()

# Explicit one-run overrides, without changing configuration.
outputs = local_path.get_output_path('C:/Data/qis/another-run')
resources = local_path.get_resource_path('C:/Data/qis/other-resources')  # Must exist.

# Alternatively, select a configuration file for this call only.
outputs = local_path.get_output_path(settings_path='C:/Data/qis/private-settings.yaml')
```

Output directories are created, including missing parents. Resource directories must already
exist; use `get_resource_path(..., create=True)` only for deliberate download/cache
locations. Paths expand `~` and environment variables. Relative YAML paths are relative
to the YAML file; explicit relative directory arguments are relative to the working
directory. Returned paths are absolute strings, without requiring a trailing separator.
Use `Path(directory) / filename`, not string concatenation.

Settings are cached by YAML file. After editing a file in a running session, call
`local_path.get_paths.cache_clear()`. Selecting a different file is recognised on the next
getter call. Legacy `file_utils` module constants are import-time snapshots, so examples
pass the resolved directory explicitly to file readers/writers.

### Local runners and explicit overrides

Use `class Locals(Enum)`, `run_local(local=...)`, and a main guard calling only the
selected local branch. Directory CLI flags and the duplicate example output helpers have
been removed. For an override, call the runner from Python or edit the local main call:

```python
from examples.factsheets.multi_assets import Locals, run_local

run_local(local=Locals.CORE_ETFS)  # YAML OUTPUT_PATH
run_local(local=Locals.CORE_ETFS, output_path='C:/Data/qis/reports/one-run')
```

Display-only and offline check branches remain write-free and do not require path
configuration. Select `Locals.REPORT` or `Locals.EXPORT` to save their output. Their
lower-level `run_example(output_dir=None)`/plot functions retain the no-export convention;
the local dispatcher selects YAML defaults when export is requested. A supplied
`output_path` also requests export. Stress reports require a fresh destination and
reject existing targets; YAML defaults use a named subfolder such as
`OUTPUT_PATH/account_equity_stress`. Pass a new `output_path` for each later report.

The FX hedging example still downloads public Yahoo data by default. Select
`run_local(..., use_local_data=True)` for CSV inputs under YAML `RESOURCE_PATH`, or
supply `input_dir` explicitly. Missing local files raise an error; they do not trigger
an undisclosed download. The options-stress example uses
`RESOURCE_PATH/options_stress` for its reusable Yahoo cache, with `input_path` as an
override. Its CHECK branch uses the default or configured cache and may download/write resource
data, but does not export a report. Frozen synthetic and bundled dataset loaders do not
require private directories.

Tests and disposable runs should pass their temporary directory explicitly. On the
maintainer's Windows host, use the repository's external Python and C-local execution
helpers described in [AGENTS.md](../AGENTS.md). Never put environments or generated
run output under the OneDrive checkout. The tracked previews in `examples/figures/`
are documentation assets, not default output destinations.

### Cluster dendrograms (offline)

`python -m examples.plots.cluster_dendrograms` displays the frozen synthetic example.
To export its two PNGs and membership CSV through YAML:

```python
from examples.plots.cluster_dendrograms import Locals, run_local

run_local(local=Locals.EXPORT)
# Or: run_local(local=Locals.EXPORT, output_path='C:/Data/qis/reports/clusters')
```

The example demonstrates supplied axes and titles; no FactorLasso, OptimalPortfolios
or vendor data is required.
