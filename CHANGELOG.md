# Changelog

All notable changes to qis are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [5.0.0] - 2026-07-12

Breaking release. The public API is reduced from 568 to 373 symbols and the
`qis.utils.df_agg` aggregators are renamed. There are no deprecation shims:
code that used the removed names must be updated at the same time as the
upgrade. Pin `qis <5` to stay on the previous API.

Every removed symbol is still importable by its defining module. Nothing is
deleted; only the top-level `qis` namespace is reduced. Where a name is
listed as removed below, the migration is:

```python
qis.set_spines(ax)                                # 4.x
from qis.plots.utils import set_spines            # 5.0
set_spines(ax)
```

### Breaking: renamed `qis.utils.df_agg` aggregators

`qis.nanmean`, `qis.nanmedian` and `qis.nansum` shadowed the numpy names of
the same spelling while carrying different semantics: they consume a
`pd.DataFrame`, return a `pd.Series`, exclude non-finite entries (`+-inf` is
mapped to nan and skipped), and default to `axis=1`, which is the opposite of
the pandas default. In a module importing both `numpy as np` and `qis`, the
name collision was a trap rather than a convenience. The whole module is
renamed for consistency.

| 4.x | 5.0 |
| --- | --- |
| `qis.nanmean` | `qis.df_nanmean` |
| `qis.nanmedian` | `qis.df_nanmedian` |
| `qis.nansum` | `qis.df_nansum` |
| `qis.nanmean_positive` | `qis.df_nanmean_positive` |
| `qis.nansum_positive` | `qis.df_nansum_positive` |
| `qis.nansum_negative` | `qis.df_nansum_negative` |
| `qis.nanmean_clip` | `qis.df_nanmean_clip` |
| `qis.nansum_clip` | `qis.df_nansum_clip` |
| `qis.nanmean_weighted` | `qis.df_nanmean_weighted` |
| `qis.abssum` | `qis.df_abssum` |
| `qis.abssum_positive` | `qis.df_abssum_positive` |
| `qis.abssum_negative` | `qis.df_abssum_negative` |
| `qis.last_row` | `qis.df_last_row` |
| `qis.sum_weighted` | `qis.series_nansum_weighted` |
| `qis.get_signed_np_data` | `qis.utils.df_agg._get_signed_np_data` (now private) |

`sum_weighted` is renamed rather than prefixed with `df_` because it takes two
`pd.Series` and returns a `float`; its first parameter was also named `df`,
and is now `data`.

### Breaking: reduced public namespace

| subpackage | 4.3.x | 5.0 |
| --- | --- | --- |
| `qis.utils` | 189 | 59 |
| `qis.plots` | 131 | 73 |
| `qis.perfstats` | 75 | 65 |
| `qis.models` | 92 | 92 |
| `qis.portfolio` | 49 | 49 |
| `qis.file_utils` | 27 | 27 |
| **total** | **568** | **373** |

The removed symbols are internal machinery that was published by accident: the
top-level namespace was assembled by `import *` over the subpackages, so
anything a module happened to define became part of the API. Analytics
(`qis.models`, `qis.portfolio`, `qis.perfstats`) is unchanged apart from four
enums, because those are the functions a user of the library calls.

`qis.utils` (130 removed) — numpy helpers (`np_nansum`, `np_shift`,
`repeat_by_rows`, `running_mean`, `to_finite_np`), DataFrame plumbing
(`df_zero_like`, `df_ones_like`, `align_df1_to_df2`, `dfs_to_upper_lower_diag`),
string formatting (`float_to_str`, `str_to_float`, `df_to_numeric`,
`series_to_str`, `date_to_str`), list and dict helpers (`flatten`, `list_diff`,
`list_intersection`, `split_dict`), and date helpers (`is_leap_year`,
`get_weekday`, `months_between`, `min_timestamp`). What survives is the API
proper: `TimePeriod`, `generate_dates_schedule`,
`generate_rebalancing_indicators`, the `df_agg` aggregators, `df_asfreq`,
`get_group_dict`, `split_df_by_groups`, `ColVar`, `ColumnData`, `EnumMap`,
`ValueType`, `update_kwargs`, `covar_to_corr`, `fit_multivariate_ols` and the
annualisation factors.

`qis.plots` (58 removed) — matplotlib axis and legend plumbing (`set_spines`,
`remove_spines`, `set_ax_tick_params`, `set_legend`, `set_title`,
`align_y_limits_axs`, `autolabel`, `rand_cmap`, `subplot_border`), the colour
palette accessors (`get_n_colors`, `get_n_sns_colors`, `get_cmap_colors`), the
table-styling setters in `qis.plots.table` (`set_cells_facecolor`,
`set_row_edge_color`, `set_data_colors`), and five table-computation helpers.
All 63 `plot_*` functions remain public, as do `TrendLine`, `LastLabel`,
`LegendStats` and `PdfType`, which appear in their signatures. `set_suptitle`
remains public.

`qis.perfstats` (10 removed) — the 14 `*_TABLE_COLUMNS` constants, the
`cond_regression` entry points, and the DataFrame operations listed under
*Moved* below.

### Added

- `qis.factsheet` — one-call facade over the four factsheet generators
  (`qis.portfolio.reports.factsheet_facade`). It picks the report archetype
  from the input type, calibrates windows / regressions / regimes /
  annualisation for the requested reporting frequency via
  `fetch_default_report_kwargs`, renders, and optionally writes a PDF. The
  four generators remain available and unchanged for full control. All qis
  imports are deferred into the function bodies, so the module never depends
  on `qis` being fully initialised.
- `qis.df_nanmean_negative` in `qis.utils.df_agg`, completing the
  sum / mean by positive / negative grid. `df_nansum_negative`,
  `df_nansum_positive` and `df_nanmean_positive` already existed;
  the mean of negative entries did not.
- `axis: Literal[0, 1] = 1` argument on `df_nansum_clip`, `df_nanmean_clip`,
  `df_abssum`, `df_abssum_positive`, `df_abssum_negative` and
  `agg_median_mad`. All six hardcoded `axis=1` and could not aggregate along
  the other axis.
- `__all__` in `qis/plots/utils.py`, declaring `TrendLine`, `LastLabel`,
  `LegendStats` and `set_suptitle` as the public surface of that module.

### Changed

- Library modules no longer reach through the `qis` namespace. `qis` imported
  itself — `qis/portfolio/reports/strategy_factsheet.py` called
  `qis.set_spines(...)`, `qis/plots/scatter.py` called `qp.get_n_sns_colors(...)`
  through `import qis.plots as qp`, and `qis/portfolio/backtester.py` called
  `qu.repeat_by_rows(...)` through `import qis.utils as qu`. The top-level
  namespace was therefore not an API decision but an internal calling
  convention that `import *` published. All 68 such call sites across 12 files
  now import from the defining module.
- `qis.utils.df_agg` aggregators share one `_to_agg_series()` helper that
  selects the index from the aggregated axis, replacing the repeated
  `if axis == 0 / else` blocks. `_validate_axis()` rejects values outside
  `{0, 1}`, which numpy would otherwise accept silently (`axis=-1`).
- `compute_df_desc_data` no longer takes a mutable default argument
  (`funcs: Dict = {...}` is now `Optional[Dict] = None`).
- `qis.plots.reports.econ_data_single` is deprecated and emits a
  `DeprecationWarning` on import. `econ_data_report` and `ReportType` are no
  longer exported. Scheduled for removal in 6.0.
- `qis.plots.derived.gantt_data_history` is not imported by
  `qis/plots/__init__.py`. It requires plotly, which is not a qis dependency;
  import it by full path if plotly is installed.

### Moved

Public names are unchanged unless listed under *Breaking* above. Only code
importing these by file path must update.

- `qis/plots/reports/price_history.py` -> `qis/plots/derived/price_history.py`.
- `qis/plots/reports/gantt_data_history.py` -> `qis/plots/derived/gantt_data_history.py`.
- `df_price_ffill_between_nans`, `df_ffill_negatives`,
  `df_fill_first_nan_by_cross_median`, `df_price_fill_first_nan_by_cross_median`
  and `replace_nan_by_median` from `qis.perfstats` to `qis.utils.df_ops`. These
  are pure DataFrame operations and compute nothing about performance.
- `compute_futures_fx_adjusted_returns` and `get_aligned_fx_spots` from
  `qis.perfstats.fx_ops` to `qis.market_data.fx_hedging`, consolidating FX
  handling with `FxRatesData`.
- `get_output_path`, `get_paths` and `get_resource_path` from `qis.file_utils`
  to `qis.local_path`.

### Fixed

- `nansum_negative(df, axis=0)` raised
  `ValueError: Length of values (3) does not match length of index (4)`. The
  function passed `axis` to `np.nansum` but hardcoded `index=df.index`, so the
  `axis=0` result (one entry per column) was given the row index. On a square
  frame it returned the correct numbers under the wrong labels, silently. Now
  `df_nansum_negative`, and correct on both axes.
- `agg_data_by_axis(df, axis=1)` mislabelled its result. It always used
  `index=df.columns`, contradicting its own docstring, so an `axis=1`
  aggregation (one entry per row) carried column labels.
- `qis.compute_desc_table` and `qis.DescTableType` resolved to different
  modules. `compute_desc_table` is defined in both `qis.perfstats.desc_table`
  and `qis.plots.derived.desc_table`, and `DescTableType` in both as well.
  `qis/__init__.py` imported perfstats before plots, so the top-level namespace
  bound `compute_desc_table` from plots and `DescTableType` from perfstats —
  two different Enum classes, for which `==` returns `False`. The plots export
  is removed and the pair now resolves consistently to `qis.perfstats`.
- Three names were exported from two modules each and silently shadowed by
  whichever import ran last: `compute_desc_table` (above),
  `add_bnb_regime_shadows` (`plots.derived.prices` and
  `plots.derived.regime_data`) and `separate_number_from_string`
  (`utils.dates` and `utils.struct_ops`). All deduplicated.
- `nanmean_positive` and `nanmean_negative` leaked
  `RuntimeWarning: Mean of empty slice` when a line contained no entries of the
  requested sign. `nan` is the intended result; the warning is now suppressed
  at the call to `np.nanmean` / `np.nanmedian` rather than propagated to the
  caller.
- Continuation-line alignment in
  `qis/portfolio/reports/overlays_smart_diversification.py`, where wrapped
  keyword arguments were indented 16 columns past the opening parenthesis.

### Removed

- All 15 deprecated `df_agg` aliases. This release is a hard break; there is no
  4.x compatibility layer.
- `qis.examples` exports (`load_usd_assets`, `generate_performance_report`,
  `DEFAULT_RA_TABLE_COLUMNS`) from the public namespace. Examples are
  documentation, not API.

### Migration

For downstream code, the mechanical steps are:

1. Rename the `df_agg` calls per the table above. `qis.nanmean_weighted` is the
   most commonly used and becomes `qis.df_nanmean_weighted`.
2. For any `AttributeError: module 'qis' has no attribute X`, import `X` from
   its defining module. `python -c "import qis.plots.utils as m; print(m.X)"`
   locates it; the module list is in `docs/REMOVED_5_0.md`.
3. Do not import from `qis/examples/` — it is documentation and is
   restructured without notice.

## [4.3.2] - 2026-06-28

### Added
- `qis.estimate_dimson_beta` in `qis.models.unsmoothing.dimson_beta` —
  Dimson (1979) aggregated-coefficient beta to detect return smoothing.
  Regresses each asset on the contemporaneous and lagged market return and
  reports `beta_dimson = sum_k b_k`; the `beta_dimson / b_0` ratio measures
  the contemporaneous understatement and the t-stat on the summed lagged
  slopes tests whether the lag effect is real. Pure numpy/pandas, importable
  standalone.
- `qis.adjust_returns_with_factor_lag` in `qis.models.unsmoothing.factor_lag`
  — factor-lag (Dimson) unsmoothing for illiquid / appraisal-based series.
  Companion to the own-lag AR(q) engine; removes smoothing that manifests as
  a lagged response to a liquid factor, which the own-lag AR cannot see (a
  fund-of-funds with near-zero own autocorrelation but a real lagged-equity
  beta). The correction is mean-preserving and lifts the contemporaneous
  loading to `beta_D`, so a plain contemporaneous regression recovers the
  true loading and the existing HCGL / factor-covariance estimator picks it
  up with no change.
- `qis.adjust_returns_with_joint_unsmoothing` in
  `qis.models.unsmoothing.joint_lag` — single-regression joint own-lag +
  factor-lag unsmoothing, fitting the own-lag coefficient and the
  lagged-factor beta jointly via the rolling EWMA cross-moment estimator.
  Removes the omitted-variable bias and stage-order dependence of running the
  AR engine and the factor-lag engine sequentially.
- Week-of-month / last-week-of-month anchored frequencies (`WOM-*`, `LWOM-*`)
  now resolve to a monthly (12.0) annualisation factor in
  `qis.utils.annualisation`, handled explicitly because the generic frequency
  regex cannot parse the week number in the anchor.

### Changed
- Reorganised unsmoothing into a `qis.models.unsmoothing` subpackage. The
  former `qis/models/unsmoothing.py` (own-lag AR(q) engine,
  `adjust_returns_with_ar`) is now `qis/models/unsmoothing/ar_lag.py`,
  alongside `dimson_beta.py`, `factor_lag.py`, `joint_lag.py` and a `tests/`
  directory. Package-level imports (`qis.adjust_returns_with_ar`, etc.) are
  preserved; only code importing the old module by file path must update.
- `multi_assets_factsheet` regime-Sharpe plotting accepts an optional
  `regime_classifier` argument, falling back to the instance default for a
  per-plot override.

### Fixed
- `RegimeClassifier` degenerate-benchmark guard. A constant / zero-return
  block (e.g. an overlay nav with longer history than the principal,
  back-padded over the union index) collapses interior quantiles, which
  previously surfaced as a bare pandas `Bin edges must be unique`. The new
  check mirrors `pd.qcut` exactly (unique edges <= number of labels), so it
  fires iff qcut would have failed and never on healthy data, and raises a
  descriptive error naming the benchmark, the number of non-empty bands, and
  the remedy (clip inputs to their common live window).
- `qis.plots.lineplot` marker indexing is now cyclic and None-safe
  (`markers[idx % len(markers)] if markers else None`), fixing an IndexError
  when the number of lines exceeds the number of supplied markers.

### Removed
- Internal `qis/market_data/MIGRATION_NOTES.md` scratch file; trimmed the
  `fx_hedging_example.py` example.

## [4.3.0] - 2026-06-19

### Added
- Python 3.14 support.
- `qis.delever_returns`, `qis.lever_returns`, `qis.implied_leverage` in
  `qis.perfstats.returns` for working with levered / unlevered return
  series given leverage and financing rate.
- `qis.unsmooth_returns_ar1_ewma`, `qis.unsmooth_returns_glm`, and
  `qis.compute_ar1_unsmoothed_prices` in `qis.perfstats.unsmoothing` for
  AR(1) EWMA and AR(q) Getmansky-Lo-Makarov unsmoothing of appraisal-based
  NAV series, with severity diagnostics.
- `qis.to_quarterly_returns` in `qis.perfstats.returns` for compounding
  daily / weekly / monthly returns to quarter-end with partial-quarter
  masking.
- Vectorised `qis.compute_risk_table`.
- Reorganised `qis/examples/` into themed sub-packages (`perfstats/`,
  `models/`, `regimes/`, `portfolios/`, `factsheets/`, `plots/`, `utils/`,
  `case_studies/`, `_helpers/`) with a per-folder `README.md` and a
  module-level docstring on every example file.
- New example `qis/examples/perfstats/unsmoothing_and_delevering.py` —
  end-to-end walkthrough of the leverage / unsmoothing functions on a
  bundled OCSL / Oaktree GCF / SPX / US HY / US Agg weekly NAV dataset.
- New example `qis/examples/models/multivariate_ols.py` demonstrating
  `qis.fit_multivariate_ols` directly (separated from the EWM linear-model
  example).
- `bbg-fetch >=2.0.0` listed as optional dependency for examples that
  pull data from a Bloomberg terminal.

### Changed
- Bumped minimum Python from 3.9 to 3.10. (numba 0.61 dropped Python 3.9
  support, and the bump to numba ≥0.63 for Python 3.14 forces the same
  floor here.)
- Bumped minimum numba from 0.60.0 to 0.63.0 (required for Python 3.14
  support; see numba 0.63.0 release notes, Dec 2025).
- Renamed several example files for clarity:
  - `models/ewm_filters.py` → `models/ewm_kernels.py`
  - `models/correlation_matrix.py` → `models/ewm_correlation_table.py`
  - `models/ewma_factor_betas.py` → `models/ewm_linear_model.py`
  - `portfolios/btc_marginal_contribution.py` → `portfolios/balanced_60_40_with_btc.py`
  - `perfstats/perf_excluding_best_worst_days.py` → `perfstats/miss_best_worst_days_impact.py`
- Moved `infrequent_returns_interpolation.py` from `examples/utils/` to
  `examples/perfstats/` (matches the API location:
  `qis.perfstats.timeseries_bfill`).
- `qis.adjust_navs_to_portfolio_pa` renamed to
  `qis.adjust_component_navs_to_portfolio`; the `asset_prices` parameter
  renamed to `component_navs`. The function decomposes a portfolio's
  PA return into its *additive components* (carry types, fundamental
  return sources, gross vs net vs costs), not into asset-level NAVs.
  The original names were misleading. The formula is unchanged: the
  function rescales component NAVs by a time-weighted factor so their
  PA returns sum to the portfolio PA return — useful for stacked-area
  visualisation of return decomposition. Docstring rewritten to
  document the actual invariant.
- `qis.to_portfolio_returns` and `qis.portfolio_returns_to_nav` docstrings
  now explicitly document the NaN convention: a NaN return contributes
  `0` to that period's portfolio PnL (interpreted as "asset held its
  notional but earned 0%"), rather than renormalising the remaining
  weights. Correct convention if NaN means "asset wasn't tradable, held
  cash"; wrong if NaN means "data missing, treat position as continuous".
  No code change; convention was previously undocumented.
- `qis.compute_net_return_ex_perf_man_fees` HWM crystallization block has
  an explanatory comment clarifying the GAV-after-CPF subtraction and
  the resulting audit-trail discontinuity. The numerical output is
  unchanged.
- `qis.utils.df_freq.df_asfreq` explicit-NaN-on-target-date bug. When the
  input DataFrame contained a NaN value on a date that coincided with a
  resample target timestamp (e.g. yfinance returning NaN on the US
  Independence Day Friday that was also the `W-FRI` bucket end),
  `df.reindex(index=freq_index, method='ffill')` returned NaN for that
  bucket — pandas' `reindex(method='ffill')` looks back through input
  *index labels*, not values, so it found the holiday Friday label
  directly and copied its NaN value. The post-reindex ffill could not
  recover the value because nothing earlier in the resampled output
  existed to fill from. Fix is a single pre-reindex `_apply_fill(df, ...)`
  call so the daily series has its NaNs filled before the reindex picks
  bucket anchors, matching `df.resample(freq).last()` on a ffilled series.
  Reported by Ben Richards.
- `qis.to_quarterly_returns` calendar-QE boundary bug. The previous
  implementation used `returns.reindex(q_returns.index).notna()` to detect
  partial trailing quarters, which silently masked the entire output for
  any input whose timestamps did not land on calendar quarter-end dates
  (W-FRI weekly, business-month-end series). The new implementation uses
  a calendar-month coverage check per column: a quarter ending at QE is
  complete iff the input's last non-NaN observation falls in the same
  calendar month as QE.

### Fixed
- `qis.compute_total_return` trailing-NaN handling. Previously the function
  used `prices.iloc[-1]` for the end value, so any series with NaN at the
  end (terminated fund, delisted ETF) silently returned NaN total return.
  Now mirrors the existing leading-NaN treatment via
  `get_last_nonnan_values`, with a matching warning. Fix propagates
  through to `compute_pa_return`, `compute_returns_dict`, and Sharpe /
  alpha computations downstream.
- `qis.compute_excess_returns` look-ahead bias. The function used
  `lag=None` for `multiply_df_by_dt`, applying today's risk-free rate to
  today's funding cost — a small contemporaneous-rate look-ahead.
  `get_excess_returns_nav` already used `lag=1`. Now both functions agree
  on lag=1 (funding cost at t uses the rate set at t-1).
- `qis.prices_at_freq` `ffill_nans=False` ignored when `freq is None`.
  Previously the no-freq branch gated only on `fill_na_method` (default
  `'ffill'`), so callers passing `ffill_nans=False` without also
  overriding `fill_na_method` got ffilled prices anyway — opposite of
  what the parameter name promised. Now `ffill_nans=False` disables fill
  in both branches consistently.
- `qis.df_price_ffill_between_nans` ignored its `method` parameter. The
  body hardcoded `.ffill()` regardless of input, so callers passing
  `method='bfill'` got silent ffill behaviour. Now `method` dispatches
  correctly to ffill / bfill / None.
- `qis.compute_pa_return` returned a 0-d scalar `array(0)` instead of a
  vector of zeros for DataFrame input when `num_years <= 0` (degenerate
  input). `np.zeros_like(n)` where `n` is an `int` returns a 0-d array;
  replaced with `np.zeros(n)`.
- `qis.to_zero_first_nonnan_returns` removed always-true defensive check
  in the `init_period=1` branch. Since `first_date = returns.index[0]`
  and any non-NaN index is by definition >= the first index, the guard
  was dead code. Behaviour is unchanged; code is simpler.

## [2.0.1] - 2023-07-08

### Removed
- `qis.portfolio.optimisation` layer, with core functionality moved to a
  stand-alone Python package
  [bop (Backtesting Optimal Portfolio)](https://pypi.org/project/bop/).
  Removes the cvxpy and sklearn dependencies.

### Added
- Factsheet reporting via [pybloqs](https://github.com/man-group/PyBloqs).
- Four factsheet types with examples in `qis.examples.factsheets`:
  - `multi_asset` — cross-sectional comparison
  - `strategy` — performance / risk / trading stats from `PortfolioData`
  - `strategy_benchmark` — strategy vs benchmark
  - `multi_strategy` — parameter sensitivity sweeps

## [1.0.1] - 2022-12-30

Initial public release.

---

Versions between 1.0.1 ↔ 2.0.0 and 2.0.2 onwards (prior to the next
release) have not been backfilled. Run `git log --tags --oneline` for
release-by-release commit history.
