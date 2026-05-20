# Changelog

All notable changes to qis are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed
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
