# Changelog

All notable changes to qis are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.31.0] - 2026-09-26

**This release changes computed values.** It fixes the defects found while the analytics
handbook was written. Most fixes remove crashes, silent zeros or infinities, or wrong docstrings,
but several change numbers that earlier versions reported:

- Excess returns no longer drop their first period, and every period accrues cash at the rate
  known at its start on the return grid, in `compute_ra_perf_table`, in the backtester's
  funding leg and in the financing leg of `lever_returns` and `delever_returns`.
- EWM estimators are point in time by default: `EwmLinearModel.fit`,
  `compute_ewm_beta_alpha_forecast`, `compute_portfolio_vol`, `ewm_xy_convolution`,
  `compute_ewm_cross_xy` (and with it `compute_fx_vol_beta`) and the EWM autocorrelation
  estimators no longer seed with full-sample statistics, and every EWM uses the first
  observation of each column. Early values move; late values barely do.
- `estimate_rolling_ewma_covar(demean=True)` no longer understates variances by about 6% at span
  52, gives NaN rather than zero before an asset's first return, and EWM covariance tensors
  default to a positive-semidefinite gap policy. The qis risk functions ignore such an asset when
  it is not held.
- Tracking-error contributions sum to the tracking error, the factsheet's P&L risk attribution
  shows Euler shares, and the risk-table drawdowns on a coarse grid include the last observation.
- Signal diagnostics pair signals with future returns only, use the library's annualisation
  factors and the right degrees of freedom.

Each entry below states the size of its effect and, where possible, how to restore the old value.

### Fixed

- Make `PortfolioData.get_instruments_pnl(is_net=True)` deduct each period's realised trading
  cost divided by the preceding NAV. It previously subtracted a trailing 260-period sum of costs
  divided by current NAV, which left the first 259 rows missing and overstated later cost drag.
  Net contributions of a portfolio without fees, funding or carry now sum to its NAV return.
- Make `compute_autocorr_df` return the requested `num_lags`; any value other than 20 previously
  raised a shape error because the lag count was not passed to the estimator.
- Seed the variance recursion of `compute_ewm_newey_west_vol` with the squared first
  observation, as `compute_ewm_vol` does; it was seeded with the unsquared value. The lag terms now
  use an explicitly supplied `ewm_lambda` instead of the 0.94 default, and a Series input with
  lags no longer fails. With `num_lags=0` the estimator equals the EWM variance exactly.
- Report `CALMAR_RATIO` as missing for a history that is never under water. `MAX_DD` is zero
  there and the ratio was minus infinity for a positive return, the wrong sign as well as
  undefined.
- Report `DOWNSIDE_VOL` and `SORTINO_RATIO` as missing when fewer than two `freq_vol` returns
  are negative. The downside volatility was 0.0 and the Sortino ratio plus or minus infinity.
  A zero downside volatility (equal losses) also gives a missing Sortino ratio.
- Report `MAX_DD_VOL` as missing when `VOL` is missing (a single sampled return) or zero; it was
  0.0.
- **Behaviour change.** Keep each asset's final observation on the `freq_drawdown` grid of
  `compute_risk_table` and `compute_ra_perf_table`. On a coarse grid such as
  `freq_drawdown='ME'` the trailing incomplete period was dropped, so a fall in the current month
  was invisible and `CURRENT_DD` referred to the last month-end. On the frozen synthetic universe
  ending 15 December 2025, month-end `CURRENT_DD` moves by up to 2.5 percentage points
  (`SAL_HF` −22.0% to −19.6%); with the full universe cut at 15 December 2025, `SEQ_US`
  `CURRENT_DD` is −45.40% instead of −44.89% and its `MAX_DD` deepens from −45.18% to −45.40%.
  `WORST` and `BEST` include the partial-period return.
  The default `freq_drawdown='D'` is unaffected; factsheets with monthly or quarterly presets,
  which set `freq_drawdown` to the reporting grid, now show the current drawdown at the last
  observation.
- Make `RegimeClassifier.compute_regimes_pa_perf_table` forward `additive_pa_returns_to_pa_total`
  and its other keywords (`is_report_pa_returns`) to
  `compute_regimes_pa_perf_table_from_sampled_returns`; they were documented but ignored, so the
  per-annum patch was always applied. The three classifiers' `compute_regimes_pa_perf_table`
  gain the optional keyword `additive_pa_returns_to_pa_total=True` and pass it on.
- Compute the `SharpeConvention.PA` regime Sharpe ratios before `is_use_benchmark_means=True`
  replaces the benchmark's per-annum regime values by its periodic means for display. The
  benchmark's regime Sharpe ratios were those periodic means divided by an annualised
  volatility.
- Stop the `SharpeConvention.LOG` regime branch from applying `log1p` to returns that a
  `ReturnTypes.LOG` classifier already produced as log returns; the `ARITHMETIC` branch now
  converts such returns to simple returns. `compute_regimes_pa_perf_table_from_sampled_returns`
  gains the optional keyword `sampled_return_type=ReturnTypes.RELATIVE`, which the classifiers
  set from their `return_type`. Classifiers with the default `ReturnTypes.RELATIVE` are
  unchanged.
- Report an empty regime as missing in `compute_regime_sharpe_decomposition`, as the regime
  table does; it was 0.0. The total column is unchanged and the regime columns still add up to
  it when missing values are skipped.
- Label `PerfStat.BEAR_AVG`, `NORMAL_AVG` and `BULL_AVG` 'Bear Average', 'Normal Average' and
  'Bull Average', the column names of the regime table. They were 'Bear Avg' and so on, and
  `plot_ra_perf_scatter(x_var=PerfStat.BEAR_AVG)` raised `KeyError`.
- Make `PerfParams.copy` keep `freq_skewness` and `freq`; both returned to 'ME' in the copy. The
  method gains the optional keyword `freq_skewness`.
- Accept a Series in `compute_performance_table`, as its signature states; it raised
  `TypeError`.
- Highlight the unrecovered episode in `plot_top_drawdowns_paths(highlight_ongoing=True)`. The
  plot compared episode ends with the penultimate date, so the ongoing episode, which ends on the
  last date, was never highlighted and an episode recovering on the penultimate date was.
- Compute the episodes of `plot_top_drawdowns_paths` on the plotted `freq` grid; they were always
  computed on calendar days. The x-axis reads 'Days in drawdown' on the default 'D' grid and
  'Observations in drawdown' otherwise, and with `freq=None` the legend's `days_dd` now counts
  observations, as the axis does.
- **Behaviour change.** Compound the first return period into every per-annum excess return.
  With `rates_data` starting on the first price date, the one-period lag left the first excess
  return missing, so the excess NAV started one period late while the elapsed years still
  covered the whole history. The first return date now accrues exactly zero cash (no time has
  elapsed), so `PA_EXCESS_RETURN`, `AN_LOG_EXCESS_RETURN`, `SHARPE_EXCESS`, `SHARPE_LOG_EXCESS`,
  `SORTINO_RATIO` and the Calmar numerator no longer depend on whether the rate series starts on
  or before the first price date. On the frozen synthetic universe (2 January 2014 to
  31 December 2025) with 2% cash from the first price date, `SEQ_US` `PA_EXCESS_RETURN` is
  −0.704% instead of −0.651% and `SCM_GLD` 6.323% instead of 6.265%; with monthly 2% cash from
  31 January 2014, `SCM_GLD` `SHARPE_EXCESS` is 0.3782 instead of 0.3925.
- **Behaviour change.** Charge the cash return of the period (t-1, t] at the rate known on the
  return date t-1 in `compute_excess_returns` and every helper built on it. The lag was one
  observation of the rate series on its own calendar, so with daily rates and monthly returns
  each month used the second-to-last daily quote of that month. The rate series is now aligned to
  the return grid as of each date and then lagged by one period of that grid. With a realistic
  time-varying daily cash path on the synthetic universe, monthly `SHARPE_EXCESS` of `SEQ_US`
  moves from −0.0130 to −0.0114 and the annualised regression alpha of `SBD_TSY` from 2.246% to
  2.274%.
- **Behaviour change.** Accrue backtest cash at the funding rate known at the start of each
  period. `backtest_model_portfolio` credited the cash held over (t-1, t] with the quote dated t,
  a one-period look-ahead that also disagreed with `compute_excess_returns`; both now use the same
  convention, and a cash-only portfolio earns exactly the cash return the excess helpers
  subtract. A 50/30 portfolio with 20% cash over 2014 to 2025 on a realistic rate path ends at a
  NAV of 142.878 instead of 142.884. A funding series with no quote known at the start of a
  period now warns that the NAV is missing from that date.
- **Behaviour change.** Charge the financing leg of `lever_returns` and `delever_returns` at the
  rate known at the start of each period. A time-varying `financing_rate` was forward-filled to
  each return date without a lag, so the period (t-1, t] paid the quote dated t, a one-period
  look-ahead that disagreed with `compute_excess_returns` and the backtester. The rate is now
  aligned to the return grid as of each date and lagged by one return date; the first date takes
  the latest quote strictly before it and is missing when there is none. A scalar rate is
  unchanged. Levering monthly `SEQ_US` returns of the synthetic universe once (`leverage=1`) over
  2014 to 2025 with a Fed-funds-like path (0.1%, hikes to 5.3% in 2022 and 2023, cuts from
  September 2024) ends at a NAV of 0.8593 instead of 0.8566; the largest monthly difference is
  2.6bp. To reproduce the old numbers, date each quote one period earlier.
- Lag a rate series with a single quote in the internal `multiply_df_by_dt`; the lag was skipped
  when the series had no more observations than the lag.
- Make `compute_pa_excess_compounded_returns` compound and annualise over the same window when
  `rates_data` starts after the first return date: the NAV starts at the date that opens the
  first period with a known rate and the elapsed years count from that date, with a warning.
  Missing periods were previously counted as flat but included in the years. DataFrame columns
  are now each annualised over their own window.
- **Behaviour change.** Count each return in one window of `compute_sampled_vols`. Windows
  included both boundaries, so a return dated on a boundary entered two adjacent windows; they are
  now right-closed. 102 of the 143 monthly volatilities of the synthetic `SEQ_US` change, by a
  median of 2.1% and at most 17%. `BenchmarkVolsQuantilesRegime` inherits the change.
  `split_df_by_freq` gains the optional keyword `inclusive='both'`; `'right'` gives the
  right-closed windows.
- Return a missing aggregate from `portfolio_returns_to_nav` on a date where every contribution is
  missing, as `to_portfolio_returns` does, instead of a zero return. The NAV is still one on the
  first date and flat through interior gaps, and now ends at the last date with an observed
  contribution.
- Warn when `to_returns` receives a keyword argument it does not use, naming the closest documented
  argument. A misspelt `is_log_return=True` silently returned simple returns.
- Run the fee account of each column of `compute_net_navs_ex_perf_man_fees` from its first
  observed NAV. A column starting after the first row came back as 1.0 on the first row and
  missing afterwards, because the missing gross returns entered the fee recursion.
- Keep the first period's excess return in `get_excess_returns_nav`. It zeroed the first observed
  return, which was the first real return, so the NAV was missing on the first date and omitted
  the first period.
- **Behaviour change.** Drop the incomplete first block of `T mod h` rows when
  `df_resample_at_int_index` aggregates, as for the block sums of
  `compute_autocorrelation_at_int_periods`; a partial sum was treated as a complete block. With
  `func=None` the block's last level is still kept. The function gains the optional keyword
  `drop_incomplete_first=None`. On the synthetic `SEQ_US` daily returns the 5-day block
  autocorrelation moves from 0.0396 to 0.0373.
- **Behaviour change.** Rebuild `interpolate_infrequent_returns` as a point-in-time Brownian
  bridge on log NAVs. The interpolated returns are on the pivot index, compound exactly to each
  reported return in the default simple mode and sum to it in log mode, use only data up to each
  report date, and have, per pivot period, the variance of an EWM of the reported returns
  (`span` reports). Previously the standardised pivot return was used as a level deviation with
  full-sample moments: on 20 quarterly reports the daily increments had a volatility of 22.2%
  against 7.2% implied by the reports and a lag-one autocorrelation of −0.47 (now 11.6% at the
  default span, 9.0% at `span=1`, and 0.03); `annualization_factor` acted as calendar days (a
  monthly pivot with 12 inflated the bridge 4.7 times; the path no longer depends on it); returns
  summed rather than compounded; an exactly zero report or a first report date off the pivot grid
  lost increments; and the output index was the union of pivot and report dates. The default
  `vol_adjustment` is 1.0 instead of 1.15, which compensated for the old scaling; pass
  `vol_adjustment=1.15` to add the same variance. `is_to_log_returns=True` now means that inputs
  and outputs are log returns, as documented; the default mode takes and returns simple returns.
- Make the internal `estimate_ols_alpha_beta` return NaN with a `UserWarning` when alpha and
  beta are not identified. A constant non-zero regressor or a single observation made statsmodels
  drop the intercept and the slope lookup raised `IndexError` outside the fallback, so
  `qis.compute_ra_perf_table_with_benchmark` failed on a benchmark with constant returns; an
  all-zero regressor reported a pseudo-inverse slope of zero.
- Return NaN rather than zeros from `estimate_ols_alpha_beta` when the fit fails, and an alpha
  p-value of NaN rather than 0.0 without an intercept: a zero p-value read as a highly
  significant alpha. The alpha itself stays 0.0 without an intercept, as imposed by the model.
- Make `LinearModel.get_model_residuals_corrs` return each asset's mean off-diagonal residual
  correlation. It returned `(n - 1) / (2n)` times that mean, one third of it for three assets.
- Align the two moments of `LinearModel.get_model_ewm_r2`: both are now geometrically weighted
  sums over the same dates with the same weights. The residual moment used to start from zero
  after the warm-up while the return moment had run since the first return, so the R² read
  0.97 to 0.999 on the first dates after the warm-up instead of the single-observation ratio
  (0.15, 0.00 and 0.79 in the factor-risk-model chapter example); last-date values move by less
  than 0.002 there.
- Select weights as of each loading date in `LinearModel.compute_agg_factor_exposures`, so a
  weight row dated off the loading grid (a calendar month-end on a weekend, month-end weights
  against weekly loadings) is no longer dropped. Exposures are NaN before the first weight row and
  wherever a held asset has no loading, including the warm-up rows that showed zero; missing
  loadings of assets with zero weight are ignored.
- Report NaN in the `Total` column of `LinearModel.get_asset_factor_attribution` while a lagged
  loading is missing; it summed the missing terms as zero.
- Read residual variances as of each date in `LinearModel.compute_factor_risk_contribution`, as
  weights and loadings already were; an off-grid date raised `KeyError`. A held asset without a
  loading or residual variance now makes the date NaN instead of silently zeroing the whole factor
  exposure, and undefined contribution ratios are NaN rather than 0.
- Report the benchmark attribution of `compute_benchmarks_beta_attribution_from_prices` and
  `compute_benchmarks_beta_attribution_from_returns` as NaN while a lagged beta is missing. The
  residual `Alpha` used to absorb the whole portfolio return during the warm-up (21 periods by
  default): in the synthetic gallery portfolio the cumulative monthly `Alpha` fell from 7.9% to
  3.9% once those rows are excluded. The returns variant no longer overwrites its first row with
  zeros; its total column still holds the portfolio return.
- Keep a genuine zero portfolio beta in `compute_portfolio_ewm_benchmark_betas`; zeros were
  replaced by the previous beta as if they were holidays, so a portfolio fully in cash kept its
  last beta. With the as-of weights above no holiday filling is needed: on the monthly grid of the
  gallery portfolio, 21 of 75 month-end betas were stale values from the previous month-end (up
  to 0.10 off).
- Stop `EwmLinearModel.fit` from overwriting `x` and `y` with the mean-adjusted panels; the
  adjustment now serves the moments only, so `get_factor_alpha` and `get_model_ewm_r2` work on
  the returns as supplied and the residual keeps the intercept.
- Validate `RiskModel` inputs for positive semi-definiteness of `covar` and `factor_covar` and
  for non-negative `residual_vars`, with a tolerance of 1e-10 relative to the largest diagonal
  element (at least 1e-10). Such inputs were accepted and failed only later, in the stress
  module.
- **Behaviour change.** Seed the EWM covariance recursion of `compute_portfolio_var_np`, and so of
  `compute_portfolio_vol`, the VaR functions and `PortfolioData.compute_portfolio_vol`, with a
  zero matrix. It was seeded with the final state of an EWM covariance over the whole sample,
  which put later observations into every early estimate, and inside `compute_portfolio_vol` that
  seed always used decay 0.94 whatever `span` or `ewm_lambda` was requested. On three synthetic
  instruments with span 33 the annualised volatility on the second date falls from 8.21% to 1.93%;
  the two paths agree within 0.11% after 100 observations and are identical at the end of the
  sample. A new optional `covar0` argument of `compute_portfolio_var_np` takes an explicit seed;
  passing `compute_ewm_covar(a=returns, ewm_lambda=0.94)` reproduces the former path.
- **Behaviour change.** Make `compute_portfolio_correlated_var_by_groups` pair the weights of
  date t with the EWM covariance through t, and make `compute_portfolio_independent_var_by_ac` use
  the diagonal of that same zero-seeded covariance. The correlated VaR previously lagged the
  weights one period and used the full-sample seed, while the undiversified VaR used same-date
  weights and a separate `compute_ewm_vol` estimate on the full return grid, so the undiversified
  figure fell below the correlated one on warm-up days and after weight cuts (the first six to
  eight dates of two synthetic examples). Both now run one recursion on the dates shared by
  weights and returns, the correlated function's existing alignment, and the bound holds on every
  date. On a quarterly rebalanced ten-asset synthetic
  backtest the correlated VaR changes by a median 0.2% after warm-up, by more than 1% on 56 of
  2,508 days (rebalancing and large-move days, at most 8.5%); the undiversified VaR is unchanged
  on these inputs.
- **Behaviour change.** Make `compute_benchmark_portfolio_risk_contributions` return Euler
  contributions to tracking error, d_i (Σd)_i / TE, which sum to TE and equal `RiskModel`'s
  `mcte`. It divided by the benchmark volatility, so its contributions summed to TE²/σ_b (1.69%
  instead of 4.94% in the risk-contributions worked example), returned infinities for a zero
  benchmark, and aligned only the portfolio weights to the covariance, so a benchmark Series with
  a missing or extra label raised "matrices are not aligned" and a reordered one came back in
  sorted order. Both weight vectors are now aligned by label and zero tracking error returns
  zeros. Multiply the result by TE/σ_b to reproduce the former level.
- **Behaviour change.** Make `PortfolioData.get_instruments_pnl_risk_attribution`, the
  `AttributionMetric.PNL_RISK` panel of the strategy factsheet ("P&L Risk Attribution,
  sum=100%"), return ex-post Euler shares Cov(x_i, x_p) / Var(x_p) of the portfolio P&L. It
  returned standalone P&L volatility shares (`ddof=0`, zero-P&L days dropped) normalised to 100%,
  which are not a decomposition of portfolio risk and show hedges as positive. In the
  risk-contributions worked example the Treasury sleeve moves from 13.0% to −1.9%. The new
  optional `is_standalone=True` returns the former shares.
- Make `PortfolioData.compute_ex_anti_portfolio_vol_implied_by_covar` and
  `compute_risk_contributions_implied_by_covar` with `freq=None` select the input weights as of
  each covariance date, as `RiskModel` does. They matched weights to covariance dates by exact
  date only and reported zero volatility and zero contributions wherever the two calendars
  differed; this affects the ex-ante volatility panel of the strategy-vs-benchmark TRE factsheet.
- Compute the `strategy returns vol` column of `PortfolioData.compute_portfolio_vol` from simple
  NAV returns, the basis of the instrument-weighted column beside it; it used log returns. On a
  ten-asset synthetic backtest the column moves by a median 0.03 and at most 0.18 volatility
  points.
- Make `PortfolioData.plot_ra_perf_table(benchmark_price=..., perf_params=None)` build the
  default `PerfParams` the benchmark table itself uses instead of raising `AttributeError` on
  `perf_params.freq_vol` when no title is passed.
- Make the marginal-active-risk development runner (`contributions_run.py`) divide d_i m_i by
  2 TE: the gradient of active variance sums to 2 TE² against the active weights, so its
  verification printed False and its percentage contributions summed to 200%.
- Make the EWM seed the state before a column's first finite observation, and let that
  observation update it, whether the column starts on row 0 or later. `ewm_recursion` used to set
  row 0 to the seed, so a finite first row never entered, while a column that started after
  missing rows did enter. `InitType.X0` now seeds each column with its own first finite value,
  so a late-starting column equals pandas `ewm(adjust=False)`; before, it was seeded with 0 and
  started at `(1 - lambda) x`. With the leading missing row of `qis.to_returns`, the first
  `compute_ewm_vol` variance is now `r_1^2` instead of `0.06 r_1^2` at `lambda = 0.94`, so the
  first inverse-volatility weight of `compute_ra_returns` is no longer about 4.1 times too large.
  A `ZERO` seed, an explicit `init_value`, `compute_ewm_sharpe` and
  `compute_ewm_long_short_filter` now use the first row. The change decays as `lambda^t`, below 5%
  after about 1.5 spans; on the synthetic daily panel at span 31 it vanishes (below 1e-12 of the
  level) within about 300 rows of a column's start, and an `X0` column that starts on row 0 is
  unchanged.
- Make `NanBackfill` behave as documented. Before a column's first observation every policy now
  returns NaN (`ZERO_FILL` and `NAN_FILL` returned 0 from row 1). `NAN_FILL` resets the state like
  `ZERO_FILL` and reports NaN at the gaps in every function; it returned zeros outside the
  covariance tensors, and inside them it reported every exactly-zero entry as missing, genuine
  zero covariances included.
- Seed `InitType.VAR` variance recursions with the variance of the observations: `compute_ewm_vol`
  and `compute_ewm_newey_west_vol` used `Var(x^2)`, about 2e-8 instead of 9e-5 for 1% daily
  returns. `compute_ewm` and `compute_rolling_mean_adj` now raise `ValueError` for `VAR`, which
  seeded a mean with a variance, and an EWMA mean adjustment requested with `VAR` takes `MEAN`.
- Make `compute_roll_mean` and `compute_rolling_mean_adj` with `MeanAdjType.INSAMPLE` use each
  column's `nanmean` and return the input's container: one NaN made a column NaN, and pandas
  input raised `ValueError`, which also broke `compute_ewm_cross_xy` and
  `EwmLinearModel.fit(mean_adj_type=INSAMPLE)`.
- Make the `compute_ewm_vol` volatility floor work for a Series and a one-dimensional ndarray: a
  Series raised `ValueError` (also through `compute_ra_returns(series, vol_floor_quantile=...)`)
  and a 1-d array returned a T x T array.
- Make `compute_ewm_cross_xy` accept the documented Series x DataFrame, Series x Series and 1-d
  ndarray inputs, which raised `TypeError`, a numba `TypingError` or `IndexError`; a DataFrame
  factor with a Series asset now raises a clear `TypeError` instead of `AttributeError`.
- Make `compute_ewm_covar` honour `is_corr` for a single cross-section.
- Make `compute_ewm_covar_newey_west` pass `ewm_lambda` and `nan_backfill` to its lag terms; with
  only `ewm_lambda` given they used 0.94.
- Make `compute_ewm_long_short_filter` accept a one-dimensional ndarray (numba `TypingError`).
- Make `compute_ewm_xy_beta_tensor` scale free and never report a cross moment as a beta. A factor
  second moment below 1e-8 replaced the inverse of the whole factor matrix by the identity, so a
  0.5bp-volatility factor got "betas" of 1e-9 instead of 2.0, and a singular matrix fell back to
  univariate betas. A factor with no variance now gets NaN betas while the others come from the
  reduced system, and a numerically singular (unit-diagonal eigenvalue ratio below 1e-12) system
  gives NaN. `compute_ewm_cross_xy` and `compute_ewm_beta_alpha_forecast` now mask only
  non-positive denominators instead of any below 1e-8.
- Fix the `compute_one_factor_ewm_betas` index-mismatch message, which printed `{x.index}`.
- Make `compute_ewm_newey_west_vol` non-negative by construction: the lag-k term now carries the
  factor `lambda^(k/2)`, the geometric mean of the EWM weights of the two dates it pairs, which
  makes the estimator a Bartlett quadratic form. The unweighted lags turned negative from row 531
  on `x_t = (-1)^t 0.94^(t/2)` and qis returned a NaN volatility and a ratio of 1; the corrected
  variance there is `0.94^t`. The lag terms now honour `nan_backfill` (they always held their state
  from a zero seed), and the ratio is NaN, not 1, where the EWM variance is not positive.
  `compute_ewm_covar_newey_west` uses the same factor. On monthly synthetic returns at span 36 the
  Newey-West variance ratio at the last date moves from 1.22 to 1.21.
- Make `compute_ewm_std1_norm` return unit standard deviation for IID input, as its name and
  docstring state: with the default same-span EWMA demeaning it returned `1 / sqrt(1 + lambda)`,
  0.71 at span 260; it now multiplies by `sqrt(1 + lambda)` and seeds its final EWM at zero instead
  of at its first value, which gave a transient of up to `sqrt(N)`.
- Make the `compute_ewm_covar_tensor_vol_norm_returns` volatility point in time: it was seeded
  with the full-sample mean of `x^2`; it now takes each column's first squared return (`X0`).
- Make `filter_outliers` silence invalid-value warnings locally; it called `np.seterr`, changing
  numpy's error state for the whole process.
- Make `ewm_insample_winsorising` use NaN-aware quantiles, so a column with a missing value is
  winsorised; make `compute_ewm_score` floor each column's volatility at that column's own
  `clip_quantile` quantile instead of one quantile pooled over all columns.
- Fix the two-dimensional branch of `ewm_winsdor_markovian_score`, which updated the state only
  at outliers; a missing observation now holds the state in both branches, as documented.
- Make the soft presets of `OutlierPolicyTypes` cut the EWM score at 3.57, the score of a
  10-standard-deviation move at `lambda = 0.94` (new helper `score_of_move`): the contemporaneous
  score is bounded by `sqrt(lambda / (1 - lambda)) = 3.96`, so the previous cut at 10 could never
  fire.
- **Behaviour change.** Make `estimate_rolling_ewma_covar(demean=True)` unbiased for iid returns.
  It centred each return on an EWM mean that already included it, so the residual was
  `lambda (x_t - m_{t-1})` and the covariance was scaled by `2 lambda^2 / (1 + lambda)`: 0.944 at
  the default span of 52, variances 5.6% and volatilities 2.9% low. The residual is now the
  one-step forecast error `x_t - m_{t-1}`, still point in time, and the matrix is multiplied by
  `N / (N + 1)`. Every entry of every matrix rises by the factor `(1 + lambda) / (2 lambda^2)`,
  1.0596 at span 52 (on the synthetic universe the 2020-09-30 US equity volatility moves from
  17.6% to 18.1%); correlations and `demean=False` output do not change. Multiply by
  `2 lambda^2 / (1 + lambda)` to reproduce earlier numbers. Ex-ante volatility, tracking error and
  absolute risk contributions built on these matrices rise by 2.9%; percentage contributions do
  not move.
- **Behaviour change.** Make `estimate_rolling_ewma_covar` report NaN before an asset's first
  return in both estimators, and seed its demeaning mean at zero. The default path returned a
  zero row and column before inception, which read as a riskless asset, while the vol-normalised
  path (`is_apply_vol_normalised_returns=True`) returned NaN also on the first return date,
  because the mean seeded at the first return made the first residual exactly zero. The mean now
  starts from zero, so an asset's first residual is its first return: on the covariance chapter's
  example the volatilities after 12 weekly returns move from 9.3%, 3.4% and 4.7% to 9.5%, 3.4%
  and 4.6%, and at the last date by less than 0.01%. On the synthetic universe with quirks,
  `SEQ_EM` (first price 2 April 2010) had zero variance on 21 of 84 quarterly matrices of the
  default path and now has NaN there; after inception both paths are finite and PSD. Consumers
  that need zeros can call `fillna(0.0)` on each matrix.
- **Behaviour change.** Default `compute_ewm_cross_xy(var_init_type=InitType.X0)` instead of
  `InitType.MEAN`. The full-sample mean square seeded the denominators of `BETA` and `CORR`, so
  the ratios over the first `1.5 N` rows depended on later data. `compute_fx_vol_beta`, which
  feeds `compute_fx_optimal_hedge` and the FX hedging report, now passes `X0` explicitly: on a
  synthetic `SEQ_EU` against an 8%-volatility EUR/USD, monthly from 2010 with span 36, its beta
  moves by up to 0.067 in the first 54 months, by up to 0.029 afterwards and by 1e-5 at the end
  of 2025. Pass `var_init_type=InitType.MEAN` to `compute_ewm_cross_xy` to restore the old
  values; `ewm_xy_convolution` is unaffected, as it passes `ZERO`.
- **Behaviour change.** Make `estimate_rolling_ewma_covar` apply the end of `time_period`; only its
  start was applied, so matrices after the end were returned. A `time_period` with a missing
  start or end no longer raises.
- **Behaviour change.** Compute the covariance of `compute_masked_covar_corr` on a panel with NaN
  pairwise-complete, as documented: each pair is centred on the means of its overlap, as pandas
  `DataFrame.cov` does, instead of on each series' full-history mean, and a pair with no common
  date is NaN instead of -0.0. On the two-series example of the covariance chapter the covariance
  moves from 0.25 to 1.0; on daily returns of the ragged synthetic universe entries move by less
  than 4e-5 in correlation units. `bias=True` divides the overlap sum by the overlap count.
- Make `apply_pca(eigen_signs=...)` flip whole eigenvectors. It flipped rows of the eigenvector
  matrix (asset coordinates), so the result was no longer an eigen-decomposition (residual 0.88
  on a 3x3 correlation matrix). Eigenvector `j` is now flipped when its first loading has the
  opposite sign to `eigen_signs[j]`, and a sign vector of the wrong length raises `ValueError`.
- Make `ewm_xy_convolution` run for every frequency. The horizon from `get_annualization_factor`
  is a float for `'ME'`, `'QE'`, `'W-WED'`, `'D'` and `'YE'`, and pandas `rolling` and `shift`
  rejected it, so only `'B'` worked; it is now an integer number of rows, and a frequency whose
  factor is not a whole number is rejected with `ValueError`.
- **Behaviour change.** Seed the second moments of `ewm_xy_convolution` at zero, like its cross
  moment. They were seeded with full-sample means, a look-ahead: a run on a prefix of the data did
  not reproduce the prefix of the full run (gap 0.09 at `freq='ME'`). At `freq='ME'` on synthetic
  daily returns the estimates move by up to 0.72 at the first date, 0.14 one horizon later and
  under 0.002 after three. New optional keyword `var_init_type`; `InitType.MEAN` restores the former seed.
- **Behaviour change.** Seed the second moment of `compute_ewm_vector_autocorr` and
  `compute_ewm_vector_autocorr_df` at zero. It was seeded with the full-sample `np.nanvar`, a
  look-ahead that also pulled the early estimates towards zero (0.009 to 0.028 at rows 2 to 4 of
  an AR(1) with coefficient 0.5 at span 60, against 0.37 to 0.45 now); the difference falls to
  0.03 after one span and 0.001 after three. The vector estimator now equals the diagonal of
  `compute_ewm_matrix_autocorr`. New optional keyword `var_init_type`; `InitType.VAR` restores the
  former seed.
- Report NaN instead of zero for the first `lag` rows of `compute_ewm_vector_autocorr` and
  `compute_ewm_matrix_autocorr`, and NaN instead of an infinite ratio where the vector
  estimator's second moment is zero.
- Return NaN for the off-diagonal mean of `compute_ewm_matrix_autocorr(aggregation_type='mean')`
  with a single column; it raised `ZeroDivisionError`.
- Make `compute_path_lagged_corr` return the contemporaneous correlation of `a1` and `a2` at lag
  0; it returned 1.0 for any pair. The autocorrelation kernels keep lag 0 at one.
- Accept lag 0 in `compute_path_lagged_corr_given_lags` and `compute_path_autocorr_given_lags`;
  it raised `ValueError`.
- Name the dispersion Series of `estimate_acf_from_paths` `'std'`; it was named `'str'`.
- Make `compute_autocorrelation_at_int_periods` reject `ewma_smoothin_span` with a
  `NotImplementedError` that names the argument; it raised a bare `NotImplementedError`.
- Add the f-prefix to the `ValueError` message of `compute_ewm_corr_single`, which printed
  `{returns.columns}` literally.
- **Behaviour change.** Make `compute_sum_freq_ra_returns(is_norm=True)` divide each calendar
  period's sum of risk-adjusted returns by the square root of the number of observations in that
  period. It divided by the square root of `get_annualization_factor(freq)`, the number of periods
  per year, so the "normalised" sums of unit-variance daily terms had standard deviations of about
  0.31 weekly, 1.32 monthly and 3.97 quarterly instead of 1. A period without observations is now
  missing rather than zero. The same scale enters the non-overlapping mode of
  `get_paired_rareturns_signals`. The old numbers are the new ones times
  `sqrt(n_J / get_annualization_factor(freq))`; `is_norm=False` still returns the plain sums.
- **Behaviour change.** Make the overlapping mode of `get_paired_rareturns_signals`
  (`is_nonoverlapping=False`) pair the rolling sum over rows (t - span, t] with the signal at
  t - span (`signal.shift(span)`). It used the signal at t - 1, which had already seen span - 1 of
  the returns it was said to predict: on the ten clean synthetic instruments a 63-day momentum
  signal with no true predictive power showed a correlation of 0.88 with its 63-day forward sums;
  paired forward it is -0.07. There is no switch back; shift the returned indicator by
  `1 - span` rows to reproduce the old pairing.
- Make `get_paired_rareturns_signals` run under pandas 3: the default `freq` is now `'BQE'`
  (the business quarter-end alias valid in pandas 2.2 and 3; `'BQ'` raised `ValueError` under
  pandas 3 and meant the same period), and `is_mean_adjust_returns=True` no longer passes the
  removed `axis` argument to `expanding`, which raised `TypeError`.
- **Behaviour change.** Make `map_signal_to_weight(signal_map_type=SignalMapType.ExpCDF)` fade a
  tail when only its own decay is given; a single `tail_decay_right` or `tail_decay_left` was
  silently ignored. Passing both is unchanged.
- Make `map_signal_to_weight` warn (`UserWarning`) when `SignalMapType.NormalCDF` or `LaplaceCDF`
  receives `tail_level`, a slope or a tail decay away from its default; those maps read only
  `loc` and `scale` and ignored the arguments silently.
- **Behaviour change.** Make string horizons of `estimate_signal_diagnostics` (e.g. `'YE'`)
  compound each asset's native returns within the period and keep a period only when the asset's
  frame covers it and the asset has a finite return at every row inside it, the rule of integer
  horizons. The rebuilt NAV was forward-filled, which gave exact-zero returns for whole periods
  before an asset's first return and after its last (delisting), entered partial periods at the
  sample edges as full-period returns, and treated a missing return inside a period as zero; the
  first complete period of the sample was always lost. For month-end assets `'YE'` now has the
  same pair content as `h=12` on a January phase, ragged starts and ends included.
- **Behaviour change.** Annualise `IC_IR_an` in `estimate_ic_ir` with qis's annualisation factor
  of the IC grid, `get_annualization_factor` of the finest native key divided by the integer
  horizon, or `get_annualization_factor(label)` for a string horizon. It used 365.25 over the
  median calendar-day gap of the IC dates: 11.78 on month-ends instead of 12 (ratio x0.991),
  11.98 for samples shorter than a year, and 365.25 on business days instead of 252, overstating
  the business-day annualised ratio by 1.20. A user `periods_per_year` is now the factor of the
  native (h = 1) grid and is divided by h for each integer horizon; it was applied unscaled to
  every horizon.
- **Behaviour change.** Charge one residual degree of freedom per regression date in the pooled
  and per-group t-statistics of `estimate_signal_diagnostics`: the residual variance uses
  `n - T - 1` (T dates) instead of `n - 1` without intercept and `n - 2` with it, because the
  cross-sectional demeaning at each date is a date fixed effect that also absorbs the intercept.
  Under the null the old t-statistic was overstated by about `sqrt(n_t / (n_t - 1))`, 1.12 at the
  default minimum of 5 names and 1.03 at 20; on the handbook example (20 names, 60 months) the
  pooled t-statistic moves from 2.89 to 2.81. `compute_per_asset_betas` keeps `n - 1`.
- Align a returns frame whose dates fall inside the periods of its key but off the
  `resample(key)` labels (business month-ends under `'ME'`) to those labels in
  `estimate_signal_diagnostics`, pairing each return with the last signal value observed at or
  before the return date that ends the previous native period. Such dates were dropped silently:
  a five-year business-month-end panel of 8 names under `'ME'` gave 344 pairs instead of 480.
  Pairs are dated at the labels. A frame with several dates in one period of its key (finer than
  its key) now warns.
- Emit the `UserWarning` that `estimate_signal_diagnostics` documented for an asset listed in
  several frequency frames; the asset is assigned to the first frame and its other frames are now
  ignored by string horizons too, which previously added a duplicate column.
- Warn (`UserWarning`, naming the first five) when `estimate_signal_diagnostics` drops assets of
  `asset_returns_dict` that have no signal column; they were dropped silently.
- Make `compute_ic_timeseries` and `estimate_ic_ir` raise `ValueError` for an IC `method` other
  than `'spearman'` or `'pearson'`; any other value, such as `'kendall'`, silently computed a
  Pearson IC.
- Name the fitted model in the default suptitle of `plot_signal_diagnostics`: it always said
  "(no intercept)", also for a result fitted with `fit_intercept=True`.
- **Behaviour change (example).** Default `vol_af` of the delta-one example helpers in
  `examples/portfolios/strats/qis_delta1.py` (`simulate_vol_target_strats`,
  `simulate_trend_strats` and their `_range` variants) to 252, qis's business-day factor. With
  260 the positions were `sqrt(252/260) = 0.985` of the size needed, so a 15% target delivered
  about 14.8% realised volatility as qis reports it. Pass `vol_af=260` for the old sizing.
- **Behaviour change.** Make the positivity rule of `bootstrap_ar_process` per column. A
  non-positive step was replaced by the 25% quantile of that step's values across columns: for a
  single series that quantile is the value itself, so the clamp never bit (a positive
  dividend-yield-like series left 1.1% of 1,500-step path values at or below zero), and for a
  panel it coupled independent columns, could itself be negative and rewrote mean-zero columns
  (in a four-column test their average path level was 0.007-0.009 instead of about 0.0007). A
  column whose observed values
  are all positive is now floored at its own lower quartile; other columns are never constrained.
  Pass the new optional keyword `is_positive=False` to switch the rule off. Results for a single
  series that never reaches zero, and for mean-zero series, do not move.
- **Behaviour change.** Make the constant-series test of `compute_ar_residuals` relative to scale.
  An absolute tolerance of 1e-8 on the variance set the AR(1) slope to zero for any series with a
  standard deviation below about 1e-4 (a persistent series with variance 8.6e-9 returned 0
  instead of 0.903); a column is now constant only when its lagged range is at most 1e-12 of its
  largest absolute value, so the slope no longer depends on units.
- Make `bootstrap_price_data` with `SERIES_TO_DF` output honour supplied `bootstrapped_indices`
  whose column count differs from `num_samples`; it repeated the anchor `num_samples` times and
  raised a broadcast `ValueError` (for example with 2 supplied paths and the default 10).
- Make `bootstrap_data` resample a Series as one column under the default `DF_TO_LIST_ARRAYS`
  output, as its docstring promised; it raised a numba `TypingError`.
- Make `bootstrap_data`, and through it `bootstrap_price_data` and
  `bootstrap_price_fundamental_data`, reject supplied `bootstrapped_indices` outside the data
  rows with `ValueError`. The `@njit` kernel has no bounds checking and returned adjacent memory
  as observations (for example 9.5e-322). `bootstrap_ar_process` now also rejects negative
  indices.
- Forward `init_to_end` from `bootstrap_price_fundamental_data` to its price paths through a new
  optional keyword (default `True`, the previous behaviour), and `is_positive` to its fundamental
  paths. With `SERIES_TO_DF` output and `is_price_weighted_fundamentals=True` the function
  raised `TypeError`, because it zipped two DataFrames; it now multiplies the paths element by
  element.
- Record `"grid_panels": 6` in the stress report manifest's display limits; the sensitivity page
  draws up to six grid panels and `StressReportConfig.selected_grids` accepts six, but the
  manifest said four.
- Leave the benchmark row's `ALPHA_PVALUE` in `compute_ra_perf_table_with_benchmark` missing
  when the benchmark's regression on itself is undefined (returns that do not vary), instead of
  reporting a p-value of 1.0 next to a missing alpha, beta and R2.
- Raise the minimum observation count that `min_obs_for_ar_unsmoothing` reports for the
  unconstrained AR unsmoother to `max(q, w + 1) + w + 2` for a given warm-up `w`, and to
  `max(23, q + 2)` for `warmup_period=None`. A lag that has not been observed now has a missing
  beta, where the tensor returned the raw cross moment, so the first complete coefficient vector
  appears later; the old floors let `InsufficientData.RAISE` accept frames that produce no output.

### Changed

- Add the optional `warmup_period` to `estimate_rolling_ewma_covar`: an asset's row and column
  stay NaN for its first `warmup_period` returns, counted from its own first return, so a late
  starter enters with the same history as the others. The default `None` masks only the dates
  before the first return.
- Add the optional `warmup_period` to `compute_ewm_cross_xy`: an output stays NaN until its pair
  has more than `warmup_period` joint observations, counted from the pair's own first one. The
  default `None` masks nothing.
- Make the covariance consumers ignore an asset whose variance is NaN when it has zero weight:
  `compute_portfolio_risk_contributions`, `compute_portfolio_risk_contribution_ratios`,
  `compute_benchmark_portfolio_risk_contributions`, `PortfolioData`'s ex-ante volatility, and
  `RiskModel`'s tracking error, marginal contributions and benchmark betas. Their results equal
  those of the available block; a nonzero weight on such an asset gives NaN, and its own
  benchmark-beta loading is NaN. `RiskModel` accepts non-finite entries in the rows and columns
  of assets with a non-finite variance and checks symmetry and PSD on the other assets; any other
  non-finite entry is still rejected.
- Label the rolling Sharpe statistic `RollingPerfStat.SHARPE` as "Sharpe ratio" in plot titles
  and legends; it previously read "Sharp ratio".
- Format the normality-test p-value of `compute_desc_table` with the four decimals of
  `PerfStat.NORMTEST`'s `ValueType.FLOAT4` instead of two, so 0.004 no longer prints as 0.00.
- Return a table indexed by ticker with no columns from
  `compute_desc_table(desc_table_type=DescTableType.NONE)`; it raised `TypeError`.
- Give `PerfStat.ALPHA_AN` the wrapped label 'An\nAlpha'. It shared 'Alpha' with
  `PerfStat.ALPHA`, so a wide table containing both could not tell them apart.
- Change the default title of `plot_regime_data` from 'Conditional Excess Sharpe ratio' to
  'Conditional Sharpe ratio'; no regime convention deducts cash.
- State in `to_returns` and `prices_at_freq` that input already on the `freq` grid keeps its
  missing values whatever `ffill_nans` says, the documented `df_asfreq` convention for periodic
  data; `freq=None` fills them.
- BEHAVIOUR CHANGE: default `EwmLinearModel.fit(init_type=InitType.X0)` instead of
  `InitType.MEAN`. With a mean adjustment (`MeanAdjType.EWMA`), the old default seeded the EWMA
  mean with the full-sample mean, a look-ahead with weight lambda^21 = 0.26 at the first
  reported beta for span 31. On synthetic monthly returns with span 36 the betas at the first
  reported date move by up to 0.28, by up to 0.16 one span later and by less than 0.01 three spans
  later; on weekly returns with span 31 by up to 0.04 at the first date. The default
  `MeanAdjType.NONE` is unaffected. Pass `init_type=InitType.MEAN` to restore the old values.
- BEHAVIOUR CHANGE: chart legends of `reg_model_params_to_str` with `alpha_an_factor` print the
  linear annualised alpha `AN * alpha`, the convention of `PerfStat.ALPHA_AN`, instead of
  `expm1(AN * alpha)`: a monthly alpha of 1.3% now prints `+16%` rather than `+17%`. No qis report
  passes `alpha_an_factor`.
- **Behaviour change.** Default `limit_weights_to_max_var_limit(annualization_factor=252.0)`,
  the factor qis applies to business-day returns; it was 260, which understated the one-day VaR
  by a factor sqrt(252/260) ≈ 0.985 for volatilities annualised with 252, so capped weights are
  now 1.55% smaller. Pass `annualization_factor=260` for the former caps.
- **Behaviour change.** Default `PortfolioData.compute_portfolio_benchmark_betas(freq_beta='B',
  factor_beta_span=63)`, the defaults of `compute_portfolio_benchmark_attribution`, so the betas
  a report shows are the betas its attribution applies. They were `None` and 65; on a ten-asset
  synthetic backtest the betas move by at most 0.0025. The factsheets pass both arguments
  explicitly and are unaffected.
- Add the optional `weight_lag` argument to `compute_portfolio_vol` (default 1, the former
  behaviour); the correlated VaR uses `weight_lag=0`.
- BEHAVIOUR CHANGE: default `nan_backfill` of `compute_ewm_covar`, `compute_ewm_covar_tensor`,
  `compute_ewm_covar_tensor_vol_norm_returns` and `compute_ewm_covar_newey_west` is now
  `NanBackfill.DEFLATED_FFILL` (a missing return is a zero return), which keeps every matrix
  positive semidefinite; `FFILL` gave eigenvalues down to -0.10 of the largest and correlations
  of 1.29 on a panel with 20% asynchronous gaps. This is also the default path of
  `compute_ewm_corr_df` and `compute_data_pca_r2`. Complete data are unaffected; on the synthetic
  panel the holiday gaps of `SEQ_EU` move EWM correlations by at most 0.08, and the correlations
  of a delisted asset now decay towards zero instead of staying frozen. Pass
  `nan_backfill=NanBackfill.FFILL` for the old behaviour.
- BEHAVIOUR CHANGE: `compute_ewm_beta_alpha_forecast` defaults to `init_type=InitType.X0`; the
  `MEAN` default seeded every moment with full-sample means, so the first beta was the
  full-sample slope through the origin and early betas changed when later data were added (by up
  to 0.07 on 120 simulated months). Its prediction is now the one-step-ahead forecast `beta_{t-1} x_t + alpha_{t-1}`
  (NaN on the first row) instead of the same-date fit; the beta, alpha, residual-variance and R^2
  outputs keep their contemporaneous definitions. The beta and factor-variance recursions now
  honour `nan_backfill`, and the factor-variance frame carries the asset column labels. All
  internal callers already passed `InitType.X0` and use only the betas and alphas; pass
  `init_type=InitType.MEAN` for the old seed.
- Add the optional `warmup_period=20` keyword to `compute_one_factor_ewm_betas`.
- Compile `ewm_recursion`, `compute_ewm_long_short` and the internal matrix-update and Newey-West
  kernels with a numba on-disk cache (in-memory when no cache location is writable): the first
  `compute_ewm_sharpe` call in a new process takes about 2 s instead of 9 s.
- Report `se_beta_dimson` and `t_beta_dimson`, the classical standard error and t-statistic of
  the Dimson beta, as two new trailing columns of `estimate_dimson_beta`. Existing columns keep
  their names and order.
- Add the field `fit_intercept` (default `False`) to `SignalDiagnosticsResult`, set by
  `estimate_signal_diagnostics`.
- Add the optional keyword `is_positive` to `bootstrap_ar_process` and the optional keywords
  `init_to_end` and `is_positive` to `bootstrap_price_fundamental_data`. Defaults keep the
  documented behaviour; see Fixed for the corrected positivity rule.

### Documentation

- Correct statements that business-daily statistics are annualised with 260 periods per year.
  Volatility and Sharpe ratios use `get_annualization_factor('B')`, which is 252; 260 remains the
  number of observations per year used to size daily report windows and EWM spans.
- Correct the packaged Sharpe note: the p.a. Sharpe denominator is the volatility of the
  `PerfParams.return_type` returns (log by default), and regime p.a. residuals are allocated in
  proportion to regime frequencies.
- State in the `PerfStat` docstring that the Sharpe columns are fixed per convention and that
  `PerfParams.sharpe_convention` applies to regime-conditional Sharpe ratios only.
- Correct the `EwmLinearModel.fit` description of `is_x_correlated`: `True` inverts the full
  factor cross-moment matrix.
- Execute the Python worked examples of every methodology article in the test suite, offline,
  so a calculation change that invalidates a documented number fails.
- Organise the methodology articles as the qis analytics handbook in six parts, with a new
  Notation and conventions chapter that reserves one meaning per symbol and writes the
  annualisation factor as AN. Every methodology article now opens its inputs section with the same
  seven-row convention card, and `tools/check_docs.py` enforces the card, one transpose and
  operator style, and TeX rather than plain-text formulas.
- Add a single bibliography page. Every methodology reference is a numbered verbatim entry of it,
  enforced by `documentation_bibliography_test.py`, and the software is cited one way.
- Render `> **Insight.**` and `> **Pitfall.**` blockquotes as admonitions in the Sphinx site
  through the new `qis_callouts` extension.
- Add thirteen handbook chapters: returns, NAVs, fees and leverage; the performance-statistic
  catalogue, with one formula for every `PerfStat` column; drawdowns and time under water; alpha,
  beta and benchmark-relative performance; regime-conditional performance; exponentially
  weighted estimators; covariance, correlation and principal components; serial dependence;
  regression and HAC inference; risk-adjusted returns and volatility targeting; signal
  diagnostics; portfolio risk and Euler contributions; and factor risk models.
- Rewrite the Sharpe chapter as Sharpe ratios: conventions and inference, covering every
  Sharpe-type estimator in qis and the sampling error of the ratio. Extend the reproducibility
  article into Resampling and the bootstrap, keeping its case study. Convert the instrument
  portfolio stress page to the methodology template. Model-layer attribution now links to the
  estimation chapters instead of re-deriving them.
- Shorten the packaged `qis/docs/sharpe_conventions.md` to a convention summary that points to
  the handbook chapter; the decision-record text is retired. The regime chapter replaces its
  approximation of the per-annum regime residual with an exact derivation.
- Add eleven handbook figures produced by `tools/docs_analytics/handbook.py` on the frozen
  synthetic universe. Each figure has an independent numerical check and is registered, with its
  parameters and conventions, in the documentation analytics manifest.
- Add 46 works to the bibliography, grouped by topic and marked pending a publisher check.
- Link each core capability on the API reference page to the chapters that derive its formulas,
  and print the PDF as one book titled The qis analytics handbook.
- Test that every core analytics symbol is named in a methodology chapter, that every `PerfStat`
  member appears in the catalogue, and that the reporting-preset tables agree with
  `fetch_default_report_kwargs`.
- Correct the `PerfStat` module docstring: without `rates_data` the excess columns equal their
  zero-rate counterparts rather than being undefined.
- State in the `RegimeData` docstring that the p.a. and Sharpe panels are frequency-weighted
  regime contributions, not within-regime statistics, and that under `SharpeConvention.PA` the
  bars add up to `PA_RETURN / VOL`, which equals `SHARPE_RF0` only when the native endpoints lie
  on the `freq_vol` grid.
- Correct the `SharpeConvention` docstring: the regime branches use total returns, not excess
  returns, and PA patches to the table's `PA_RETURN`.
- Correct the `PerfParams` docstring: `freq` sets `freq_drawdown` only when that is passed as
  None, `freq_skewness` also governs kurtosis, `freq_drawdown` governs `WORST` and `BEST`,
  `freq_excess_return` is not read by any calculation, and without `rates_data` the excess columns
  equal the zero-rate columns.
- Document in the `PerfStat` docstring that `SHARPE_RF0` is the compound p.a. convention, that the
  `ColVar` field `name` shadows `Enum.name` (kept; use `_name_`), and the grids of `WORST`, `BEST`,
  `SKEWNESS` and `KURTOSIS`.
- Document the `perf_params=None` inference of `compute_ra_perf_table`,
  `compute_ra_perf_table_with_benchmark` (which sets `freq_reg` to the index frequency, not 'QE')
  and `get_ra_perf_columns`, and that `get_ra_perf_columns` skips preset columns the
  risk-adjusted table does not produce.
- Document the Calmar numerator: native-endpoint `PA_EXCESS_RETURN`, because `MAX_DD` runs to the
  final observation on `freq_drawdown`.
- Document precisely the episode-start convention of `compute_drawdowns_stats_table` (first day of
  the peak plateau, calendar-day durations for any non-None `freq`), the rebased output grid of
  `compute_rolling_drawdown_time_under_water`, and the `>= 0`/`<= 0` filter and `is_max=True`
  default of `compute_avg_max_dd`.
- Correct the `perf_stats` module docstring: the arithmetic Sharpe pair is computed inline in
  `compute_risk_table`, not by `compute_sharpe_arithmetic`, and the gap between the log and
  simple-return volatilities is first order in the periodic volatility, not third order.
- Update the performance-statistic catalogue, drawdowns, regime-conditional performance and
  Sharpe chapters and the packaged Sharpe note to the fixed behaviour, with worked-example checks
  of the missing Calmar and Sortino ratios, the month-end drawdown grid of a history ending
  mid-month, the forwarded regime patch switch and the regime-average labels.
- Correct the `adjust_component_navs_to_portfolio` docstring: the rescaled components'
  per-annum returns sum to the portfolio's; the rescaled NAVs do not sum to the portfolio NAV.
- Rewrite the cash-timing passages of the returns, notation, and backtesting chapters for the
  unified rate known at t-1, with a proposition that a rate series starting on the first return
  date is enough, and new worked checks through `compute_returns_dict` and a cash-only backtest.
- Rewrite the interpolation section of Returns, NAVs, excess returns, fees and leverage with the
  new definition, the exactness identity, the sum-of-squares proposition and its Brownian-motion
  consequence, and a new worked example; update the fee, portfolio-NAV, keyword and
  sampled-volatility passages.
- Correct the return order in the `fit_multivariate_ols` docstring: it returns the prediction,
  the parameters and the label, in that order.
- Document `reg_model_params_to_str` and the legend keywords `alpha_an_factor`, `alpha_format`,
  `beta_format` and `r2_only` in the shared plotting-arguments note.
- State the warm-up of `EwmLinearModel.fit` exactly: positions 0 to `warmup_period` are
  missing, `warmup_period + 1` rows (21 by default).
- State the loading orientation in every `LinearModel`, `EwmLinearModel` and `RiskModel`
  docstring: `RiskModel` holds assets by factors, `LinearModel.loadings` one dates-by-assets frame
  per factor, and `get_loadings_at_date` factors by assets. The orientations are kept to preserve
  the public API.
- Document that the benchmark attribution applies log-return EWM betas to simple returns, an
  exact identity in simple returns for the supplied betas and a second-order approximation of a
  simple-return beta.
- Document the arguments of `LinearModel.get_factor_alpha` and the centring and lag of
  `get_model_ewm_r2`.
- Describe the model-layer `beta_init_value` accurately: it is a one-observation prior that
  replaces the first informative observation and stays in every later EWMA estimate with weight
  lambda^k, not only a placeholder until the first lagged estimate. State that the full-sample
  interval level is `confidence_level` and that the EWMA-WLS endpoint fit includes the net
  full-model return when a net NAV is supplied.
- Update the regression and HAC, factor risk model and benchmark-relative performance chapters
  to the fixed behaviour, with worked examples that assert it.
- Correct the docstrings of `compute_portfolio_vol` (the `init_type` argument seeds only the
  optional mean adjustment; `nan_backfill` has no effect because missing returns are set to zero
  first), `compute_portfolio_independent_var_by_ac` (the sum of standalone VaRs is the
  undiversified, perfectly aligned bound, not a VaR of independent assets), and the internal
  `calculate_marginal_active_risk` (it returns the gradient of active variance, not marginal
  active risk).
- Correct the `PortfolioData.compute_portfolio_benchmark_attribution` docstring: the attribution
  is per-period simple returns and nothing is compounded.
- Rewrite the pitfalls of the Portfolio risk and Euler contributions chapter for the fixed
  behaviour: the point-in-time EWM seed and its warm-up bias, one covariance and one weight
  timing for both VaR figures, Euler TE contributions from the single-matrix function, as-of
  covariance-implied risk, the 252-day VaR cap and ex-post Euler P&L risk shares. The tracking
  error chapter now names the single-matrix TE decomposition.
- Document `InitType`, `CrossXyType`, `ReplacementType` and `OutlierPolicyTypes` with
  `Attributes:` sections, and add Google-style docstrings to `compute_ewm_sharpe`,
  `compute_ewm_alpha_r2_given_prediction`, `compute_one_factor_ewm_betas`, `compute_ewm_score`,
  `filter_outliers` and `ewm_insample_winsorising`. State that `MeanAdjType.INSAMPLE` is
  forward-looking in `compute_roll_mean`, that norm 1 of `compute_ewm_sharpe` divides by a second
  moment about zero, how the `beta_init_value` prior enters, and that the first output of
  `compute_ewm_covar_tensor_vol_norm_returns` is always the covariance (`is_corr` switches the
  second).
- Rewrite the exponentially weighted estimators chapter for the fixed behaviour: the seed as the
  state before the first observation, the four `NanBackfill` policies, point-in-time defaults,
  the PSD-safe covariance default, scale-free betas, a proof that the EWM Newey-West variance is a
  Bartlett quadratic form and so non-negative, and the variance `1/(1 + lambda)` of the demeaned
  `compute_ewm_std1_norm` signal. The worked examples assert the new numbers.
- Document the pair selection of `CorrMatrixOutput`: `FULL` returns pairs (i, j) with j < i, and
  `SUB_TOP` returns the same pairs as `FULL` (kept for compatibility; `compute_ewm_corr_single`
  uses it). The `compute_ewm_corr_df` docstring claimed j > i.
- State in `estimate_acf_from_paths` that the default `is_pacf=True` returns partial
  autocorrelations, that lag 0 is included and that the dispersion uses `ddof=0`; document that
  `estimate_acf_from_path` drops NaNs, compressing gaps, and accepts an ndarray.
- Document that `demean` has no effect in `compute_autocorrelation_at_int_periods` and that `span`
  is a block length, not an EWM span.
- Add docstrings to `compute_pca_r2` (whose annotation now says it returns one array),
  `compute_data_pca_r2`, `matrix_regularization`, `compute_ewm_corr_df`, `compute_ewm_corr_single`,
  the `compute_path_*` kernels and the EWM autocorrelation functions, and state in the `pca`
  module that the default sign convention cannot prevent a flip under a near tie of the two
  largest loadings.
- Replace the stale module docstring of `dimson_beta.py` and document the `num_lags=0` case.
- Correct `plot_corr_matrix_from_covar`, which named `covar_to_corr` as its conversion.
- Update the covariance and serial-dependence chapters and their worked examples for the fixed
  behaviour: the unbiased EWM covariance, overlap-mean pairwise covariance, both ends of
  `time_period`, `eigen_signs`, point-in-time EWM autocorrelations, the horizon convolution at
  every frequency, lag-0 cross-correlations and the Dimson standard error.
- State in `compute_ra_returns` and the `ra_returns` module docstring that the volatility and
  `vol_target` are per period of the return grid (an annual target enters as
  `sigma_annual / sqrt(AN)`), add the missing Google-style docstring, and remove the redundant
  branch that set `annualize=False` twice.
- Correct the `compute_ewm_long_short_filtered_ra_returns` docstring: `weight_lag` lags the
  volatility normaliser, not the filter output. The output is a signal dated at formation, from
  returns through t (single leg) or t - 1 (two legs), applied over (t, t+1].
- Document `map_signal_to_weight` and `SignalMapType` exactly: the constant 1.5625 = 1.25^2 puts
  the `ExpCDF` anchor at 1.25 sqrt(scale), where the weight equals `slope_right` or `slope_left`
  (weight levels, not derivatives); `scale` acts as a variance; `tail_level` is both the weight
  cap and the fading threshold; `ExpCDF` is a Gaussian-shaped map with zero slope at the centre,
  not the distribution function of an exponential law.
- Explain in `compute_returns_transform` why its defaults `momentum_span=31` and `vol_span=33`
  differ from those of `compute_ewm_ra_returns_momentum`: span 31 matches the mean age of the
  31-row rolling transform and span 33 gives the RiskMetrics decay 0.94.
- Correct the `fit_intercept` docstring of `estimate_signal_diagnostics`: demeaning the returns
  does not make the intercept zero; it is `-beta * mean(z)`, zero only when the pooled signal
  mean is zero. Document the `is_log_returns=True` default against the simple-return default of
  `qis.to_returns`, the column order of `SignalDiagnosticsResult.pairs` (with `r`), and describe
  `IC_IR` as the stability of the IC over time rather than "breadth-adjusted".
- Update the handbook chapters Risk-adjusted returns and volatility targeting and Signal
  diagnostics for the fixed behaviour: unit-variance calendar sums, forward pairing with a measured
  look-ahead example, per-side tail fading, complete-period string horizons, `n - T - 1` degrees
  of freedom with a new proposition that makes the pooled standard error unbiased under the null,
  qis annualisation of the IC ratio, and the corrected worked-example numbers.
- Correct the `qis.models.bootstrap` module docstring: `seed` seeds numba's generator only, and a
  draw neither reads nor changes numpy's global random state.
- State the anchor convention of `bootstrap_price_data`: row 0 of every path is the anchor and
  the return drawn at index row 0 is not used, so a path of `index_length` levels carries
  `index_length - 1` returns; pass `index_length=K + 1` for `K` returns after the anchor.
- Add Google-style docstrings to `bootstrap_ar_process` and `bootstrap_price_fundamental_data`,
  including the one-step offset between price and fundamental paths, the full-sample-mean start
  of the fundamentals and the element-by-element price weighting.
- Correct `examples/models/ar_bootstrap_gaps.py`: the gap rule changed in 5.2.1, not before
  5.1.1. Title the `examples/models/bootstrap_analysis.py` plots as partial autocorrelations,
  which is what `estimate_acf_from_paths(..., is_pacf=True)` computes.
- Correct the stress report page counts in the packaged notes `qis/docs/portfolio_stress.md` and
  `qis/docs/stress_testing.md`: twelve analysis pages, an optional parser appendix as page 13
  and a final notation guide, not nine pages and a tenth appendix. The loadings page is page
  nine, and the PDF shows up to six grid panels, not four.
- Document the cluster page exactly: its heatmap shows the first 12 scenarios of the
  conditional-comparison batch, and the contributor column names the three largest absolute
  holding P&Ls in each row's worst scenario of that batch, chosen over all its scenarios, so it
  can lie beyond the 12 displayed columns. The PDF footnote says so too.
- Document that `KinkPolicy` binds only when the quote equals the strike exactly, with no
  tolerance, in the `KinkPolicy` and `InstrumentLeg.get_quote_delta` docstrings, the packaged
  note and the portfolio stress chapter.
- Update Resampling and the bootstrap for the fixed behaviour: the per-column positivity floor
  with a proposition and proof, the relative constant test with a units identity, the anchor-row
  convention, the forwarded arguments and function contracts, and a new worked example that
  checks the floor and the units invariance against a numpy recursion.

## [5.30.3] - 2026-09-22

### Added

- Add `zero_return_to_nan` to `FxRatesData.compute_fx_adjusted_returns`, defaulting
  to the existing exact-zero-to-missing behavior. Set it to `False` to retain
  genuine flat-price returns in covariance and alpha inputs.

### Fixed

- Preserve separate strategy and benchmark weight and return columns in
  `compute_brinson_attribution_table` when their display names match, instead of silently dropping
  one role during summary construction.

- Keep `adjust_returns_with_factor_lag` point-in-time by seeding EWMA means from the first
  observation and leaving warm-up coefficients unavailable, so later observations cannot revise
  earlier corrected returns or diagnostics.

- Validate `plot_errorbar` error labels before creating a figure, so invalid inputs do not leave
  an open pyplot figure behind.

- Make all `compute_turnover` conventions process uniquely dated inputs chronologically and reject
  duplicate or `NaT` dates, so row permutations cannot change reported turnover.

- Make `EwmLinearModel.fit` reject non-matching factor and asset return index labels or order before
  positional EWM estimation, preventing date-misaligned panels from producing incorrect betas or
  model state.

- Make `truncate_prior_to_start` return the complete Series or DataFrame when `start` predates the
  history, instead of raising while trying to construct a nonexistent prior anchor.

- Align pandas error magnitudes by row and column labels in `plot_errorbar`, rejecting missing,
  extra, or duplicate error labels instead of attaching uncertainty positionally.

- Restore `truncate_prior_to_start` for Series inputs on supported pandas versions while preserving
  the prior anchor and Series metadata.

- Preserve `compute_net_navs_ex_perf_man_fees` forward-filled gross-NAV gaps across pandas
  versions, so an interior missing price produces a flat return followed by the full cumulative
  return instead of truncating the net-NAV path.

- Make `df_to_weight_allocation_sum1` reject signed Series and DataFrame rows whose
  numerically zero net sum makes finite sum-to-one normalization impossible, while
  preserving the established zero allocation for zero-gross rows.

- Normalize covariance matrices through one warning-free kernel when an asset has zero, missing,
  or round-off-negative variance. EWM estimators and correlation plots now use the same rules:
  undefined rows remain missing and materially negative variances are rejected. NumPy inputs now
  honor the documented array return type, PCA rejects undefined unit-variance portfolios, and the
  Markovian outlier score masks zero variance before division.

- Return direct arithmetic P&L from `compute_futures_fx_adjusted_returns` in simple mode, preserving
  valid futures losses at or below -100% instead of converting them through an undefined logarithm.

- Preserve benchmark conditional means in
  `compute_regimes_pa_perf_table_from_sampled_returns` and
  `RegimeClassifier.compute_regimes_pa_perf_table` when `is_use_benchmark_means=True`, instead of
  replacing their P.a. and default PA-Sharpe regime cells with missing values during label-aligned
  assignment.

- Continue geometric NumPy NAV paths after interior missing returns, matching the established
  pandas gap-filling policy while preserving leading and trailing missing regions.

- Align NAV-normalized portfolio costs by row label, and reject missing, extra, or duplicate NAV
  and cost-report labels instead of pairing values positionally.

- Unify periodic NAV-level sampling in `returns_to_nav`, `PortfolioData`, `MultiPortfolioData`,
  and both `FxRatesData` NAV methods through `df_asfreq`. Completed periods now use the latest
  available level at the boundary, exact-boundary missing values follow the requested fill policy,
  sub-period histories remain empty unless an endpoint is explicitly requested, and already-periodic
  inputs preserve their existing missing-value masks.

- Align `FxRatesData` spot and domestic-rate panels chronologically so unsorted rows cannot carry
  future quotes backward and rate updates between spot dates remain available to later FX carry and
  conversion calculations.

- Interpret valid uniquely dated price histories chronologically throughout total-return,
  elapsed-time, annualized-return, and performance-table endpoint calculations.

- Normalize nullable floating inputs in `compute_total_return` so Series and DataFrame histories
  use their first and last finite prices without ambiguous `pd.NA` failures.

### Removed

- Remove the unused `PortfolioData.compute_mcap_participation()` method. Market-cap participation
  had no callers in the public package stack and no maintained reporting path.

## [5.30.2] - 2026-09-14

### Added

- Add a shared conditional-shock report page using annual factor volatility magnitudes
  as +/- simple-return anchors, alongside the existing +/-10% page. Export both new
  diagnostic tables and mark unsupported downside anchors unavailable.

### Changed

- Label the original conditional page "at 10% shocks", explain anchored columns and
  responding rows on both pages, and update the notation guide and page numbering.
  Requested scenarios, factor-family splitting and portfolio valuations are unchanged.

## [5.30.1] - 2026-09-14

### Added

- Integrate the instrument-portfolio stress framework omitted from the published 5.30.0
  source. Public qis.portfolio.stress interfaces cover funded assets, intrinsic calls/puts,
  futures and consumer-defined composite payoffs, with original-mark-anchored valuation,
  shared underlying responses, FX conversion and exact nonlinear scenario P&L.
- Add optional RiskModel.factor_groups and factor-family exposure aggregation. Family
  scenarios split simple-return bumps before log conversion and joint conditioning.
- Add historical replay, conditional factor grids, scenario-local one/two-sigma risk bands,
  and through-zero quadratic sensitivity fits. Risk bands describe conditional covariance
  uncertainty; they are distinct from fitted-regression confidence intervals.
- Add the unified PDF, Excel and CSV stress report: factor/family Euler contributions,
  six ranked sensitivity panels, fitted loadings, risk tables, cluster diagnostics and
  contributors, conditional-shock matrices, optional parser coverage and notation guide.
- Add optional cluster descriptions and signed portfolio weights to plot_clusters,
  with synthetic examples, shipped methodology and installed-wheel interface checks.

### Fixed

- Keep Python renderer modules such as _figures.py in source distributions and wheels;
  the old filename exclusion for figures also removed executable reporting source.
  Wheel and publication checks now require the stress renderer explicitly.
- Preserve the causal joint-unsmoothing warmup and missing-row fixes and finite-date
  as-of lookup fixes merged after 5.30.0.

### Removed

- Breaking plotting API change: remove `LastLabel` and the `plot_time_series`
  options `last_label`, `sort_by_value_stretch_factor` and
  `indices_for_shaded_areas`. Remove these arguments from callers; they no longer
  draw annotations or shaded areas. Use `legend_stats` for last/average values
  in the legend and caller-supplied Matplotlib axes for custom annotations.

The unreleased stress-branch versions 5.28-5.36 are consolidated into this release;
those branch labels did not identify published stress-framework distributions.

## [5.30.0] - 2026-09-13

### Changed

- Corrected the existing `compute_brinson_attribution_table` to recover sector
  returns from weighted contributions before BHB allocation/selection. It now
  defaults to Frongello-linked increments and compounded Return Total columns.
  Use `is_linked=False` for corrected arithmetic effects and Return Sum columns.
  The existing five-table tuple and report-module import path are retained.
- Corrected `MultiPortfolioData.compute_brinson_attribution` to use stored native
  instrument P&L and matching prior weights, with a shared NAV baseline, before
  aggregating linked effects for display. Native, monthly and quarterly totals
  now agree even with intra-month trades. Gross remains the default;
  `is_net=True` includes realised trading costs, not management fees/funding.
- Corrected `PortfolioData.get_brinson_inputs` to compound contribution
  aggregation within reporting periods instead of adding native returns.

### Added

- Brinson methodology/migration documentation and an executable offline
  synthetic example, including independent component and NAV reconciliation tests.

### Removed

- Removed the duplicate report-layer formula body and the temporary local 5.29
  `qis.portfolio.attribution.brinson.compute_brinson_attribution` entry point.
  All callers use the one canonical `qis.compute_brinson_attribution_table`.

## [5.29.0] - 2026-09-12

### Added

- Added the opt-in `qis.portfolio.attribution.brinson.compute_brinson_attribution`
  function: BHB effects use unweighted sector returns, with optional Frongello
  linking to reconcile compounded portfolio-minus-benchmark return at every date.
  Existing attribution APIs and their numerical defaults remain unchanged.

## [5.28.0] - 2026-09-12

### Added

- Added optional net-model NAV input to lagged EWMA model-layer attribution, preserving
  gross-model betas and exposing realised trading-cost drag and net total alpha.
- Added exact report-date cumulative alpha with `warmup_periods=0`, including a validated
  initial NAV baseline. Existing twelve-period warm-up defaults are unchanged.
- Added net cumulative display paths that retain nonzero costs and omit zero-cost entries.
- Added `PortfolioData.get_brinson_inputs` for full-history arithmetic contributions and
  applied holdings, with optional realised trading costs on the preceding-NAV basis.
  Slicing retains the first requested return; existing attribution defaults are unchanged.

### Fixed

- Omit identically zero trading-cost bars and annotations from the current EWMA return
  bridge while retaining its net endpoint and contiguous model-component positions.

## [5.27.0] - 2026-09-12

### Added

- Added `TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS`, implementing the theoretical
  volatility-normalized turnover of Sepp and Lucic (2026), Definition 4.5. `compute_turnover`
  accepts an aligned annualized `vols` panel and validates its index, columns, and values.

### Changed

- Changed volatility-adjusted turnover attribution to use annualized volatility times absolute
  target-weight changes, excluding portfolio drift and execution effects as required by the
  theoretical convention.

## [5.26.0] - 2026-09-12

### Added

- Export plot_dendrogram for supplied linkage trees and plot_clusters for composite
  tree/membership views. Caller-owned axes, display aliases, labelled groups and
  arbitrary cadence counts are supported without importing any consumer package.
- Add the offline examples/plots/cluster_dendrograms.py workflow.

## [5.24.0] and [5.25.0] - not published separately

These two versions were recorded together and never published to PyPI on their own; their changes
first reached users in 5.30.0. Items that name a version were introduced in it.

**Seeded `BootstrapType.IID` results produced by qis 5.23.0 and earlier may not reproduce exactly.**
Every draw now samples its terminal row instead of leaving it mapped to source row zero. When
`(index_length - 1)` is divisible by `num_data_index`, completing that row consumes another random
batch and also shifts later sample columns.

### Added

- Added `TurnoverComputationType` and the standalone `compute_turnover` engine, with explicit
  target-weight, executed-notional-over-NAV and executed-notional-over-gross conventions,
  detailed documentation and a synthetic example (5.25.0).

- Added the provider-neutral factor stress-testing API in portfolio/risk/stress_testing.py:
  explicit log-shock conversions, joint conditional shocks/covariances, exact model P&L
  attribution and analytical conditional-factor-plus-residual prediction bands. The new
  API is additive; existing RiskModel signatures and behaviour are unchanged (5.24.0).

- Added explicit stack dependency and optional-import boundary checks, including
  isolated maintainer adapters, plus a fresh Python 3.10 lowest-direct dependency CI lane.

- Added tag-driven PyPI Trusted Publishing with release-identity and distribution
  validation; creating a GitHub Release remains optional.

### Changed

- Changed QIS portfolio turnover to default to two-sided executed notional divided by NAV.
  Derivative producers can now supply a separate `turnover_unit_notional` panel and set their
  portfolio-level convention. The former boolean selector remains as a deprecated bridge.

- Aligned package summaries, software citations, README navigation, and documentation
  landing pages with the canonical package identity and Read the Docs documentation.

### Fixed

- Apply the selected `NanBackfill` policy to both EWM moments in
  `compute_ewm_xy_beta_tensor`, preventing missing factor observations from changing betas through
  inconsistent numerator and denominator state.

- Made `find_upto_date_from_datetime_index` select the latest eligible finite timestamp from
  unsorted inputs instead of returning a future, stale, or `NaT` entry.

- Kept joint own-lag and factor-lag unsmoothing point-in-time by using the established causal
  EWMA mean seed, updating all regression moments only on jointly observed rows, and applying
  each asset's warm-up mask exactly once instead of twice.

- Normalized nullable numeric transaction-cost DataFrames before portfolio backtesting, preserving
  the existing missing-cost, alignment, and traded-notional accounting conventions.

- Ordered portfolio-backtester prices, funding rates, and instrument carry chronologically before
  stateful processing, and rejected ambiguous duplicate or missing price timestamps.

- Normalized real-valued pandas regression inputs before standard and HAC statsmodels design
  construction, so nullable benchmark returns no longer fail or fall through to all-zero alpha,
  beta, and R-squared statistics. Pandas row indexes remain an enforced alignment boundary.

- Kept risk-adjusted-table returns, ratios, and benchmark regressions within each asset's sampled
  observed history, so a longer neighboring column no longer adds post-termination flat returns.

- Preserved missing pre-inception cells in multi-asset periodic-return tables and left periodic
  and total returns undefined for columns with fewer than two observed price boundaries.

- Anchored continuation price bootstraps to each input series' own last positive finite level, so
  a trailing-ragged asset no longer produces an entirely missing path beside a longer history.

- Corrected explicit three-quarter frequency aliases to annualize at four-thirds observations per
  year, consistently with anchored and case-normalized multiplier forms.

- Matched modern business month- and quarter-end aliases and their positive multipliers to the
  equivalent calendar-period annualization factors.

- Added opt-in causal FX spot alignment that leaves leading gaps unavailable, while preserving
  historical leading backfill by default. Both modes normalize source chronology and retain the
  exact price-panel axes before cash or futures return conversion.

- Filled every requested IID bootstrap position with a random source index instead of leaving the
  terminal row deterministically mapped to source row zero.

- Supported Series price bootstrapping in both output modes and restored log-return reconstruction
  for list outputs, with consistent first- or last-price anchoring across one-asset containers.

- Made stack-plot mean and cumulative annotations work for area and bar renderers, use consistent
  observed-value means for ordinary and nullable missing data, and preserve caller-owned colors
  when adding a total line.

- Honored `drop_benchmark` in rendered regime plots while retaining the benchmark in classifier
  component tables used by other analytical callers.

- Normalized nullable floating drawdowns before maximum and current reductions so ragged
  `Float64` price histories match ordinary floating inputs without ambiguous `pd.NA` failures;
  risk-table best and worst returns now preserve missing price gaps consistently across pandas
  versions.

- Rejected negative and non-integral dated portfolio implementation lags before schedule mapping,
  preventing targets from trading before their observation date or wrapping to the history end.

- Ordered dated portfolio-weight schedules chronologically before validation and execution,
  preventing input row order from assigning targets to the wrong trade dates.

- Normalized minute-frequency annualization so explicit, inferred, case-varied, and
  multiplier-parsed aliases use one clock-hour basis per selected active day.

- Raised the statsmodels minimum to 0.14.2 after the fresh Python 3.10 CI environment
  reproduced a NumPy 2.0 binary incompatibility in the previously allowed 0.14.0 wheel.

- Aligned the p.a., log, excess, and Sortino ratio numerators in risk-adjusted performance tables
  with the configured volatility-frequency boundaries while preserving visible return columns on
  their native observed endpoints.

## [5.23.0] - 2026-09-07

### Fixed

- Corrected FX log performance by calculating asset and forward payoffs in simple returns and
  converting the combined result with `log1p`; ordinary missing observations remain missing and
  nonpositive terminal wealth is rejected for log output.
- Corrected the FX hedge and optimal-hedge carry cost to `f/(1+f)` for the existing
  local-over-reference cash-growth premium. The getter's public quote convention is unchanged;
  hedged simple performance and optimal hedge ratios now reflect the exact forward settlement.
- Corrected reference- and native-currency excess returns to accrue the starting-period cash
  quote on the actual asset return grid and subtract `log1p(cash)` for log-relative excess.
  Simple excess remains asset return minus simple cash, with the timing corrected in both modes.
- Added deterministic terminal-wealth, covered-interest-parity and return-convention regression
  checks for FX hedges and cash adjustments; clarified the distinction between hedging opening
  principal and a known terminal cash amount.

- Charged and reported transaction costs when the opening portfolio target is traded, so initial
  net NAV includes the cost of its executed instrument notionals.
- Kept rolling AR unsmoothing point-in-time by leaving unidentified warmup coefficients missing
  and using the causal `InitType.X0` EWMA mean seed instead of the full sample.

## [5.22.5] - 2026-09-07

### Added

- Added a core-only, mechanically executed first-chart example using the shipped synthetic
  universe and the held-unit portfolio backtester.

### Changed

- Expanded the evidence-linked package-choice guide with the distinct `bt` strategy-framework
  workflow and refreshed its reviewed QIS version.
- Aligned the README, package keywords, citation keywords, and software BibTeX title around
  performance analytics, portfolio backtesting, risk analysis, and factsheet reporting.

### Fixed

- Honored `drop_benchmark` in volatility-regime performance tables while retaining the benchmark
  in their component regime data.
- Reported factor-beta estimation starts from the earliest finite, non-zero sampled portfolio
  return, avoiding misleading pre-inception dates from flat NAV histories.

## [5.22.4] - 2026-09-06

### Fixed

- Normalized nullable floating data before stacked-area rendering, representing missing values as
  Matplotlib-compatible `NaN` while leaving stacked-bar behavior unchanged.
- Sized multiline headers in wide risk-adjusted performance tables relative to the rendered data
  rows, preventing oversized headers while preserving explicit caller overrides.

## [5.22.3] - 2026-09-06

### Added

- Added point-in-time portfolio breadth analytics for investable and invested counts, independent
  universe, capital and risk effective counts, allocation-efficiency reconciliation and audit
  panels, together with history, current-layer comparison and concentration plots.
- Added expanding-prefix EWMA-WLS model-layer alpha paths, additive current EWMA Sharpe
  contributions, and full-sample Sharpe contributions with full-sample benchmark and model
  volatility denominators.

### Changed

- Moved model-layer attribution, model-feature attribution, and portfolio breadth into the
  dedicated `qis.portfolio.attribution` subpackage. The attribution modules are now named
  `model_layer` and `model_feature`; the public `qis.*` API and numerical behaviour are unchanged.
- Changed the EWMA model-layer Sharpe bridge from order-dependent stage differences to additive
  common-denominator contributions, and made the net return endpoint show gross systematic
  return, realised costs and gross total alpha with its regression interval.
- Standardised model-layer bridge colours, added systematic R-squared labels, split Sharpe
  endpoints into systematic, cost and combined-alpha contributions, and routed the rolling-alpha
  chart through the QIS time-series plot with average and latest legend statistics. Cost drag uses
  a consistent muted-pink semantic colour across return and Sharpe bridges, integration uses
  DarkSlateBlue, and the rolling-alpha legend is opaque at the upper left.
- Added display-only `start_date` clipping to the rolling EWMA-WLS alpha plot; estimation continues
  to use the complete expanding history while legend statistics use the displayed range.

### Fixed

- Honored caller-supplied regime classifiers in the benchmark-regime performance-table wrapper
  instead of silently rebuilding the default quantile policy.
- Rejected invalid or duplicate resolved benchmark labels before benchmark-aware performance and
  regime calculations can expose incidental downstream errors.
- Restricted `estimate_vol()` to its documented pandas and NumPy containers with real numeric
  dtypes, rejected lossy or categorical coercions, and corrected its scalar return annotation.

## [5.22.2] - 2026-09-05

### Added

- Added current model-layer geometric EWMA-WLS attribution with joint weighted-score Bartlett
  HAC inference, effective-sample diagnostics, exact return and sequential Sharpe bridges, and an
  optional net-of-cost step.

### Fixed

- Made time-series descriptive legends use native scalar text when their public plotting format
  is `None`.
- Made all-zero `FIRST_LAST_NON_ZERO` legends display undefined endpoints instead of raising an
  incidental indexing error.
- Stabilized statistic-legend sample standard deviation and dependent t-statistics for finite
  samples whose spread is very small relative to their level.

## [5.22.1] - 2026-09-05

### Changed

- Linked strategy-benchmark factsheet alpha to the displayed lagged EWMA betas and used monthly
  exposure sampling for quarterly reports.

### Fixed

- Made benchmark-beta history warn and return `NaN` on nonpositive benchmark-variance dates,
  while preserving strict point-in-time validation and filtering report dates before calculation.
- Rejected zero-dimensional and higher-dimensional NumPy arrays in `estimate_vol()` with a
  descriptive error instead of incidental exceptions or flattened estimates.
- Stabilized descriptive-table sample standard deviation, annualized volatility, and sample-mean
  t-statistics for finite samples whose spread is very small relative to their level.

## [5.22.0] - 2026-09-05

### Added

- Added a configurable start date for the second risk-adjusted performance table in factsheets,
  defaulting to 31 December 2020; passing `None` retains the trailing-one-year window.

### Fixed

- Validated infinity before rendering plots that request descriptive tables, preventing warnings,
  partial artists, and leaked figures.
- Defaulted legend text to normal weight so fonts without a light face no longer emit Matplotlib
  fallback warnings.
- Normalized missing indexed legend endpoints so ordinary and nullable floating dtypes use the
  same configured numeric display.
- Stabilized exact-constant and finite near-degenerate statistic-legend moments without emitting
  precision-loss warnings.
- Made statistic legends use native scalar text when their public plotting format is `None`.
- Stabilized descriptive-table skewness, kurtosis, and normality results for finite samples whose
  spread is very small relative to their level.
- Made benchmark-aware performance and regime tables share explicit-name and existing-column
  precedence when resolving a standalone benchmark Series.
- Made realized-volatility estimator selection depend on each column's finite observation count
  so missing-row padding cannot change the estimator or suppress a sparse-window result.
- Rejected positive and negative infinity in `compute_desc_table()` with a descriptive error
  before numerical reducers can emit warnings or return inconsistent partial statistics.
- Corrected missing and near-zero percentages in legend diagnostics and defined all-missing
  histories without an index error.
- Made standard-deviation legend modes consistently use warning-free sample spreads after their
  documented observation selection.
- Made `append_time_series()` apply its stable keep-last duplicate-date rule before overlap
  diagnostics and empty-newer initialization.
- Added deterministic calendar-index validation before resampling, backfilling, and infrequent
  return interpolation can expose incidental pandas errors or discard invalidly labelled data.

## [5.21.2] - 2026-08-31

### Added

- Added point-in-time model-layer attribution with lagged EWMA betas, an explicit beta prior,
  expanding annualised realised alpha, and post-warm-up additive cumulative-alpha paths.

### Fixed

- Preserved defined descriptive statistics for finite constant columns while returning undefined
  skewness, kurtosis, and normality without precision-loss warnings.
- Preserved unavailable realized-volatility windows without reduction warnings and added a
  descriptive error for volatility-regime benchmarks with no finite observations.

## [5.21.1] - 2026-08-30

### Added

- Expanded the seeded model-attribution example and documentation with additive cumulative-alpha
  paths and a two-feature Shapley sensitivity exhibit with Bartlett HAC intervals.

### Fixed

- Applied return-mode-specific endpoint validity in `to_returns()`: ratio and log returns now
  require finite positive endpoints, while difference and level modes accept finite signed levels.
- Made `append_time_series()` preserve the newer provider's names, column order, axis metadata,
  and declared empty schema while aligning compatible older history and handling unavailable
  providers deterministically.
- Validated nonzero-leverage financing values and date-axis compatibility before alignment,
  preserving explicit missing observations and compatible timezone-aware funding.

## [5.21.0] - 2026-08-30

### Changed

- Breaking: renamed the model-layer input `alpha_layer_nav` to `signal_layer_nav` on
  `compute_model_layer_alpha_beta_attribution` and `ModelLayerNavs`, the regression row
  `Alpha Layer` to `Signal Layer`, the components `Alpha Layer Return` and `Alpha Layer Alpha` to
  `Signal Layer Return` and `Signal Layer Alpha`, and the feature-summary columns
  `Alpha Layer Alpha*` to `Signal Layer Alpha*`. "Alpha" now names only the regression intercept
  and its components. Docs page and example updated; numbers unchanged.

### Fixed

- Aligned descriptive-table and legend t-statistics with the signed sample mean divided by its
  standard error, independently of volatility annualization, and made undefined legend samples
  warning-free.
- Reordered the bottom row of the Brinson strategy-versus-benchmark factsheet so the
  exposure-difference panel appears on the left and the selection-effect panel on the right.

## [5.20.0] - 2026-08-29

### Added

- Added `ModelLayerNavs`, `ModelFeatureAlphaBetaAttribution` and
  `compute_model_feature_alpha_beta_attribution()` for complete factorial experiments over model
  features, including Harsanyi interactions, order-independent Shapley paths, HAC inference and
  pathwise reconstruction audits.
- Exposed `hac_lags` and `confidence_level` on
  `compute_model_layer_alpha_beta_attribution` and recorded `freq`, `hac_lags` and
  `confidence_level` on `ModelLayerAlphaBetaAttribution`; added the opt-in
  `newey_west_lag_rule()` helper. Defaults unchanged.
- Added the `Risk Layer Return` column to model-layer `component_returns`.
- Documented model-layer attribution in `docs/model_layer_attribution.md`: the exact log-return
  bridge, the linearity, bar-height and excess-basis identities, the Bartlett HAC inference and
  the NAV-ratio protocol for feature impact, with the offline example
  `examples/portfolios/model_layer_attribution_simulated.py` and its figure.

### Changed

- Replaced the by-construction reconstruction checks in model-layer attribution with a finiteness
  guard.
- Moved `frequency_convention_note.md`, `factsheets.md` and `REMOVED_5_0.md` from the wheel-shipped
  `src/qis/docs/` to the Sphinx site under `docs/`, since no docstring cites them;
  `plotting_kwargs.md`, `reporting_frequencies.md` and `sharpe_conventions.md` stay in the package
  because docstrings and tests cite them by package path. The wheel-contents check in CI now
  expects those three notes only.
- Documented and regression-tested the established calendar-boundary behavior of
  `split_to_samples()` for the `TrendFollowingSystems` consumer.

### Removed

- Removed the unused public `TrainLivePeriod` and `TrainLiveSamples` containers and the legacy
  module-level `split_to_train_live_samples()` and `get_data_samples_df()` helpers;
  `split_to_samples()` remains the supported calendar-period slicer.
- Removed the unused deep-import helpers `qis.utils.np_ops.select_non_nan_x_y` and
  `qis.utils.df_ops.norm_df_by_ax_mean`; regression data cleanup continues to use
  `qis.utils.regression.filter_x_y`.
- Removed 23 additional unused deep-import symbols across low-level DataFrame, NumPy, date,
  plotting and structure utilities, including the defective legacy long-short indicator path;
  `df_to_top_bottom_n_indicators()` remains the supported top/bottom selector.

### Fixed

- Added `estimate_hac_mean()`, a constant-only Bartlett HAC estimator for the mean of a return
  series, and used it for the total-return intervals of the model-feature attribution summary.
  The previous zero-regressor fit emitted a `SingularMatrixWarning` on every call and applied
  the two-parameter small-sample correction to a one-parameter mean; the intervals now carry
  the `T/(T-1)` correction, which narrows them by a factor `sqrt((T-2)/(T-1))`.
- Enabled the MyST `dollarmath` extension in `docs/conf.py`, so `$` and `$$` equations render in
  the Sphinx site; the frequency convention note previously rendered them as literal text.
- Trimmed model-layer NAVs to their common valid range before resampling, so a layer that ends
  early no longer contributes forward-filled zero returns to the common sample.
- Defined empty-provider behavior for `bfill_timeseries()`, processing the available provider
  under the newer schema and rejecting newer DataFrames that cannot define any output columns.
- Preserved finite newer and single-observation older price anchors when
  `bfill_timeseries()` reconstructs joined price histories.
- Supported pandas nullable floating price panels in `to_returns()` without changing return
  conventions or accepted NumPy-backed results.
- Preserved both ordered positive/negative benchmark-return regimes when one is unobserved and
  rejected custom regime mappings that do not contain exactly two entries.
- Unified the historical `qis.plots.derived.desc_table` path with the canonical descriptive-table
  enum and implementation while preserving deep-import compatibility.
- Rejected zero-row descriptive-table inputs consistently and returned warning-free missing
  statistics for columns below each statistic's sample minimum, including normality below 20
  observations.
- Made the local ETF price runner fall back to the operating-system user cache and create its
  destination directory when `settings.yaml` still contains the distributed path placeholder.

## [5.19.0] - 2026-08-28

### Added

- Extended model-layer attribution with the signal-layer return and an optional net
  full-model NAV whose exact log-return trading-cost drag reconciles gross and net performance.

### Fixed

- Made `append_time_series()` independent of input row order before overlap selection,
  comparison, and concatenation.
- Restored `unsmooth_returns_glm()` compatibility with statsmodels 0.15 after removal of the
  deprecated `AutoReg.old_names` argument.
- Rejected unsupported `bfill_timeseries()` fill policies instead of silently treating them as
  forward-fill.
- Made `bfill_timeseries()` return canonical frequency metadata for already-regular output grids.
- Processed unique performance-fee return dates chronologically, rejected duplicate dates, and
  crystallized fees on the latest available observation on or before calendar period ends.

## [5.18.0] - 2026-08-25

### Added

- Added full-sample model-layer alpha/beta attribution in log-return space, including exact
  systematic, standalone-layer, and integration contributions.

### Fixed

- Made `compute_desc_table()` score repeated DataFrame column labels independently while
  retaining their original order.
- Made optional `compute_desc_table()` t-statistics undefined for zero-volatility samples instead
  of returning infinity with a divide-by-zero warning.
- Rejected invalid leverage ratios and annualization factors in `lever_returns()` and
  `delever_returns()` with consistent descriptive errors before applying either transform.
- Made leverage funding alignment independent of financing-Series storage order and rejected
  ambiguous duplicate funding dates for nonzero leverage with a descriptive error.

## [5.17.0] - 2026-08-25

### Changed

- Assign excluded Brinson interaction effects entirely to instrument selection while preserving
  total active return, and revise the factsheet page ordering to show grouped active effects in
  the top-right panel and cumulative active attribution in the middle-left panel.

## [5.16.0] - 2026-08-25

### Changed

- Split excluded Brinson interaction effects equally between allocation and selection while
  preserving total active return, and revised the factsheet page to show grouped active effects
  with regime backgrounds.

### Fixed

- Made annualized `AVG_WITH_POSITIVE_PROB` and `SKEW_KURTOSIS` descriptive tables return their
  reduced schemas instead of raising while removing the volatility column.
- Made `compute_desc_table()` support nullable `Float64` / `pd.NA` inputs and return formatted
  missing statistics for all-missing dated columns without reduction warnings or score-mode errors.
- Rejected invalid volatility-quantile bucket counts and classifications whose requested regimes
  cannot all be populated.

## [5.15.0] - 2026-08-24

### Fixed

- Preserved missing benchmark observations in positive/negative regime classification instead of
  assigning them to the positive regime.
- Removed a stale optional PyBloqs example mode that referenced a report generator which does not
  exist.

## [5.14.0] - 2026-08-24

### Added

- Added `RiskModel.compute_tre_by_group_loadings_at_date()` as the canonical ex-ante
  tracking-error calculation for fractional, overlapping, or signed asset-by-group loadings.

### Fixed

- Aligned time-varying financing rates by date in `lever_returns()` and
  `delever_returns()`, preserving pandas shape and labels and exact zero-leverage identity.
- Made `bfill_timeseries()` honor return fill policies on expanded frequency grids and support
  one- and two-observation histories without requiring frequency inference.
- Made `bfill_timeseries()` independent of provider row order before boundary selection and
  missing-value filling.
- Preserved all-missing newer price columns without older counterparts in
  `bfill_timeseries()` instead of raising during terminal-value alignment.
- Kept return- and volatility-quantile regime IDs, colors, and report columns aligned with the
  number and order of successfully created buckets.
- Made `compute_portfolio_risk_contributions()` return typed zeros for a
  non-positive-variance portfolio instead of undefined `0/0` values.

## [5.13.0] - 2026-08-23

### Removed

- Removed `OhlcEstimatorType`, `estimate_ohlc_var`, `estimate_hf_ohlc_vol`, and the
  `qis.models.stats.ohlc_vol` module. These model-facing estimators now live in
  `stochvolmodels.estimation`.
- Retired the legacy `dev` extra; contributor tooling now lives exclusively in the PEP 735
  `test`, `lint`, and `audit` dependency groups.

### Fixed

- Fixed automatic annualization in `compute_sharpe_arithmetic()`, restoring inferred Sharpe
  calculations for Series and DataFrame inputs.
- Made maximum/current drawdown calculations warning-free for all-missing Series and DataFrame
  columns while preserving NaN results.
- Corrected drawdown episode tables to include peak and recovery boundaries, distinguish
  recovered from ongoing episodes, support unnamed Series, and preserve calendar-day duration
  units after rebasing.
- Corrected arithmetic-excess statistics to charge the observable lagged funding rate over the
  first realized price interval.
- Preserved every DataFrame column returned by compounded excess-return calculations instead of
  returning only the first column; Series output remains scalar.
- Preserved the final non-overlapping older observation when `bfill_timeseries()` combines
  adjacent provider histories.
- Made regime frequencies depend on benchmark-classified dates rather than asset missingness or
  DataFrame column order.
- Made descriptive-table positive probabilities use each asset's non-missing observations rather
  than the panel's total row count.

## [5.12.0] - 2026-08-22

### Changed

- Promoted the staged 5.11.x improvements to the next tagged release, covering canonical
  normalized risk contributions, the tracking-error report correction, JOSS manuscript updates,
  robust return-to-NAV handling across pandas and NumPy, and expanded cross-platform
  installed-wheel verification.

## [5.11.3] - 2026-08-22

### Added

- Added NumPy-array support for constant-trade-level NAV conversion, including per-column
  initial-value scaling and log-return conversion.
- Added governance and software-design documentation and expanded the supported-platform CI
  matrix with installed-wheel verification.

### Changed

- Separated source-adjacent development runners from the automated pytest suite and kept those
  runners out of installed distributions.

### Fixed

- Made an explicit `first_date` take precedence during return-to-NAV initialization without
  mutating caller-owned data or manufacturing NAV histories for missing observations.
- Preserved leading, intermittent, and entirely missing NumPy histories during additive NAV
  accumulation while allowing later observed returns to resume the path.
- Scoped development-runner layout checks to repository checkouts so the shipped test suite runs
  correctly against an installed wheel.

## [5.11.2] - 2026-08-21

### Changed

- Updated the JOSS manuscript and references for the current submission requirements.
- Cleaned actionable pytest warnings and skip handling, including categorical boxplot palette
  sizing and all-missing instrument risk calculations.

### Fixed

- Preserved fully missing Series and DataFrame return histories during return-to-NAV conversion,
  rather than manufacturing a unit NAV, while retaining warnings for unsupported initialisation
  periods.

## [5.11.1] - 2026-08-18

### Added

- Added normalized asset and grouped Euler risk-contribution ratios to
  `qis.portfolio.risk.contributions`, giving the OSS stack one canonical implementation for
  covariance-implied risk attribution while preserving the existing volatility-unit API.

### Fixed

- Prevented the strategy-versus-benchmark tracking-error report from drawing its risk-adjusted
  performance table twice while preserving the full-precision table returned to callers.

## [5.11.0] - 2026-08-16

### Added

- Added `qis.discrete_portfolio`, an event-driven portfolio replay engine with explicit orders,
  fills, signed-unit holdings, cash accounting, order/trade ledgers, pluggable execution models,
  and an adapter to the existing `PortfolioData` reporting surface. Its default contract makes
  decisions at *t* and executes them at *t+1*.

### Changed

- Aligned the PyPI summary, README opening, and documentation titles around `qis` performance
  analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python.
- Moved runnable examples from `src/qis/examples/` to the repository-root `examples/` tree,
  matching the project layout used by `optimalportfolios` and keeping examples out of wheels.
- Added one authoritative, core-only offline quickstart for synthetic data, a live-universe-aware
  quarterly backtest, explicit return/Sharpe conventions, and benchmark-relative risk. The
  documentation includes the runnable script directly and the README points to the same file.
- Added an optional clean Colab entry point that installs the latest PyPI release, reports its
  version and import path, and mirrors the authoritative offline quickstart under a mechanical
  drift check without committing notebook outputs.

## [5.10.0] - 2026-08-15

### Changed

- Adopted the standard `src/` package layout, moving the import package to `src/qis/` while
  preserving the installed `qis` import name and public API. Packaging, tests, linting,
  documentation and audit tooling now resolve the new source root, and CI verifies the built
  wheel independently of the editable installation.

## [5.9.4] - 2026-08-11

### Fixed
- Brinson attribution now preserves canonical ticker alignment when portfolios use display-name
  mappings, so group assignments and attribution values remain unchanged.

### Changed
- P&L Attribution and P&L Risk Attribution panels now use one-decimal percentage formatting
  (`0.0%`) consistently for axes, bar labels, and folded-tail totals.
- The strategy factsheet's full-history monthly-return appendix is now centered on a portrait
  page and adapts its plot dimensions to the number of calendar-year rows.
- When a strategy has more than 10 portfolio groups, the dense RA performance, YE-return, and
  regime-Sharpe summary panels now show the strategy and benchmark only and emit one warning;
  other grouped panels remain unchanged. The same limit applies to grouped RA tables in the
  strategy-versus-benchmark factsheet and grouped regime-Sharpe panels in the multi-strategy
  factsheet.

## [5.9.3] - 2026-08-09

**Behaviour change.** The separate `heatmap_fontsize` argument is removed from
`generate_strategy_factsheet` and `generate_strategy_benchmark_factsheet_plt`; use the shared
`fontsize` argument instead. Passing the removed keyword raises `TypeError`.

### Changed
- Split the offline tracking-error example into `ex_anti_tracking_error_and_risk.py` for
  `RiskModel` ex-ante TE/beta/marginal TE and `ex_post_tracking_error_and_risk.py` for realised
  EWMA TE, whole-sample TE/IR, and point-in-time EWMA beta/alpha.
- Strategy factsheet summary and full-history monthly-return heatmaps, and the periodic-return
  heatmap in the strategy-versus-benchmark factsheet, now inherit the report's `fontsize`.
- Standalone `plot_returns_heatmap` figures default to 5-point text instead of 8-point text.

## [5.9.2] - 2026-08-09

**Behaviour changes.** `PerfStat.TE`, `PerfStat.IR`, and `TRE_TABLE_COLUMNS` are removed.
The in-sample `compute_te_ir_errors` output labels remain the literal strings `TE` and `IR`;
the removed preset requested statistics no qis table builder supplied, so its headline columns
were silently dropped. The `add_tracking_error_table` factsheet panel is retitled
"Per-instrument attribution and IR"; the compatibility keyword keeps its historical name.

### Added
- `weights_tracking_error_report_by_ac_subac` adds realised-TRE, ex-ante versus ex-post
  benchmark-beta, and annualised ex-post-alpha panels. NAV-only input still produces realised
  TRE and alpha; covariance-dependent beta is omitted when no covariance path is available.
- `compute_ewma_realised_tracking_error` is part of the documented `CORE_API` portfolio group.

### Changed
- The internal implementations of `compute_ewma_realised_tracking_error`,
  `compute_te_ir_errors`, and `compute_info_ratio_table` moved from `qis.perfstats` modules to
  `qis.portfolio.risk.ex_post_tracking_error`; their top-level `qis` imports are unchanged.

## [5.9.1] - 2026-08-09

### Added
- Strategy factsheets limit the summary-page monthly-return heatmap to the latest 10 calendar
  years by default. Longer histories emit a warning and add a landscape full-history heatmap;
  `monthly_returns_heatmap_max_years=None` restores the single complete summary table.
- `qis/examples/portfolios/tracking_error_and_risk.py` demonstrates offline ex-ante and
  realised tracking error, benchmark beta, and marginal risk.

### Changed
- Strategy-factsheet monthly-return heatmaps scale monthly cells independently from the YTD
  column, so larger annual returns no longer wash out the monthly colour variation. Displayed
  return annotations are unchanged; the summary and full-history appendix use the 5-point
  heatmap font by default.
- Example-only strategy helpers moved from `qis.portfolio.strats` to
  `qis.examples.portfolios.strats`; direct imports of these non-public modules must use the new
  path.

## [5.9.0] - 2026-08-09

### Added
- `qis.compute_ewm_beta_alpha_forecast` accepts `beta_init_value`, a point-in-time,
  one-observation beta prior. The first finite beta equals the seed and subsequent estimates
  use the existing EWMA cross-moment recursion; omitting it preserves existing results.

## [5.8.0] - 2026-08-08

**One behaviour change.** `MultiPortfolioData.compute_tracking_error_table` labels its
unchanged per-instrument `mean(pnl_diff) / std(pnl_diff)` ratio column `IR` instead of `TRE`.
The values and method name are unchanged.

### Added
- `qis.compute_ewma_realised_tracking_error`, the canonical annualised EWMA ex-post tracking
  error from portfolio and benchmark NAVs.
- `weights_tracking_error_report_by_ac_subac` accepts `covar_risk_model`; its covariance can
  supply ex-ante tracking-error panels when `MultiPortfolioData.covar_dict` is absent, and a
  complete factor block adds tracking-error decomposition and strategy factor-exposure panels.

### Deprecated
- `LinearModel.compute_active_factor_risk`; use
  `RiskModel.compute_tre_decomposition_at_date` or `compute_marginal_tre_at_date`.

## [5.7.0] - 2026-08-08

**One behaviour change.** `MultiPortfolioData.compute_tracking_error_implied_by_covar` now
selects weights as-of each covariance date. Callers whose weight dates all lie on the
covariance date grid see identical numbers (characterisation-tested against 5.6.x output).
Callers with weight dates off that grid previously received zero tracking error on every
date — the exact-date reindex silently dropped every off-grid weight row — and now receive
the tracking error implied by the latest weights known at each covariance date.

### Added
- `qis.RiskModel`, a point-in-time covariance risk layer for ex-ante tracking error,
  standalone group tracking error, factor exposures, benchmark beta and loadings,
  systematic/residual TE decomposition, and Euler marginal TE contributions.
- `qis.WEIGHT_TOL`, the documented tolerance used by strict risk-universe alignment.

### Changed
- `MultiPortfolioData.compute_tracking_error_implied_by_covar` now delegates internally to
  `RiskModel` with legacy non-strict alignment. Numbers are unchanged for weight dates on
  the covariance grid; off-grid weight dates follow the behaviour note above.

## [5.6.2] - 2026-08-08

### Changed
- Performance attribution now uses configured instrument display names, matching P&L-risk
  attribution and the labels shown elsewhere in portfolio reports.
- Factsheet configuration now warns and falls back to zero-rate statistics when
  `add_rates_data=True` but the optional `yfinance` dependency is unavailable.

## [5.6.1] - 2026-08-03

**Two page-geometry limits that used to produce an unreadable panel in silence are now
enforced.** Both are the same arithmetic: a decoration sized in points, laid on a panel sized in
inches, degrades once the series count outgrows the panel. The legend case warns and changes
nothing drawn. The attribution case reduces what is drawn - `plot_performance_attribution` falls
back to the tails of the distribution once the instrument labels would crowd, which is the one
behaviour change in this release. It engages only past the point where the panel had already
stopped naming anything, so a panel that reads today is untouched, and `max_bars=0` restores the
old behaviour unconditionally.

**A factsheet carrying more series than its panel legends hold now warns.** A panel legend is a
fixed height of one row per series, and matplotlib's `constrained_layout` counts the part of it
that spills out of the axes as a layout margin: once the legend outgrows the panel cell the
solver drives the axes height to zero and disables itself for the whole figure, so every panel on
the page reverts to the raw gridspec. One overflowing legend collapses the page, not one panel.
On A4 portrait at `fontsize=5` the calibrated capacity is 13 series, so the guard warns at 14; a
22-asset panel produced an unreadable page with no diagnostic other
than matplotlib's `constrained_layout not applied`, which names neither the cause nor a remedy.
Nothing is dropped and no number changes - pages that render today render identically.

### Added
- `qis.portfolio.reports.config.estimate_legend_capacity(figsize, fontsize, panel_rows,
  gridspec_rows)` returns the number of legend entries a panel carries at full height, and
  `validate_legend_capacity(...)` warns above it, naming the `fontsize` and the `figsize` that
  would fit the requested series count. Both are internal; the calibration constant
  `LEGEND_ROW_HEIGHT_PER_FONTSIZE = 0.01778` in/row/pt is re-measured against matplotlib by
  `qis/portfolio/tests/legend_capacity_test.py`, so a matplotlib change that invalidates it fails
  the suite rather than a report.
- The guard is called by `generate_multi_asset_factsheet` (capacity counted over the asset
  columns plus any benchmark added to the navs), `generate_multi_portfolio_factsheet` and
  `generate_strategy_benchmark_factsheet_plt` (over the portfolios plus any benchmark). The three
  pages share one panel cell, `figsize[1] * 2 / 14`, so they share one capacity.

**An attribution panel wider than its tick labels can name now shows the tails.** A 90-degree
rotated tick label occupies the font's line height horizontally whatever the instrument is
called, so `shorten_instrument_names` never bought room: only a smaller font or a wider axis
does. A half-page panel is 4.09 in and holds 45 labels at `fontsize=5`; an 84-instrument
portfolio drew 84 bars over a solid smear of overlapping names, on the strategy factsheet summary
page and on all ten panels of its attribution page.

### Changed
- `PortfolioData.plot_performance_attribution` takes `max_bars` and `fontsize`. `max_bars=None`,
  the default, keeps every bar until the labels would crowd and then reduces to the tails sorted
  by the attributed value; an explicit count forces the reduction; `max_bars=0` disables it. The
  cut is two-sided for a signed metric - `PNL`, where the losers matter as much as the winners -
  and top-only for a metric that is non-negative by construction, which is `PNL_RISK`, `COSTS`,
  `TURNOVER` and `VOL_ADJUSTED_TURNOVER`: a share of a total has no bottom tail to show.
- The folded remainder is stated in the panel title (`top and bottom 45 of 84: 39 folded away,
  summing to 18.53%`) rather than drawn as one aggregate bar, which would carry a third of the
  total and flatten every instrument the panel exists to show. The `sum=` claim in the title is
  therefore still true and its exception is on the line below it.
- The instrument order is untouched below the capacity, so the asset-class blocks stay visible on
  every panel that reads today. Sorting only happens where the alternative is an unreadable page,
  and it does cost the cross-panel alignment of the attribution page: each panel then sorts by
  its own metric.
- Not applied to `MultiPortfolioData.plot_performance_attribution`, which is a separate
  implementation drawing one column per portfolio on a full-width panel. That geometry holds 92
  labels and is not at risk at the universe sizes this addresses.

### Added
- `qis.plots.utils.estimate_bar_label_capacity(axis_width, fontsize)` and `estimate_axis_width(ax)`,
  the geometry behind the reduction. The constants `BAR_LABEL_WIDTH_PER_FONTSIZE = 0.0138`
  in/label/pt and `AXIS_SHARE_OF_CELL = 0.96` are re-measured against matplotlib by
  `qis/portfolio/tests/attribution_reduction_test.py`.
- `qis.portfolio.portfolio_data.reduce_attribution_to_tails(data, max_bars)` returns the retained
  entries and the folded total, so `kept.sum() + folded` is the original sum by construction.

## [5.6.0] - 2026-08-02

**`weight_implementation_lag` is counted in observations of the price index, not in calendar
days.** It selects the entry price for the units and nothing else: a weight observed at *t* is
traded at the price `weight_implementation_lag` observations later, and prices and instrument
returns are untouched. The calendar-day shift it replaces was resolved onto the price grid by
taking the next available date, so at a lag of two every Thursday/Friday pair landed on the same
Monday - 106 of 523 weight rows on a two-year daily schedule - and because weights are consumed
one row per rebalancing a collapsed date did not skip a row, it shifted every later row. A lag of
one, which is what every `optimalportfolios` call site and every qis example passes, resolves to
the same observation under both readings and no result at lag=1 moves; results at lag>=2 move and
were wrong before. `optimalportfolios` already documented the argument as periods, so the two
packages now agree.

### Fixed
- `backtest_model_portfolio` raises when two weight dates resolve to the same traded date on the
  price index, which happens when the weights frame is denser than the price panel - a
  calendar-daily weights frame against a business-day panel is the common case. Weights are
  consumed one row per rebalancing flag, so the rows after the first collision were applied at
  the wrong dates for the rest of the backtest, and the staleness grew: on a 33-row daily
  schedule at a lag of three, 15 rows were never consumed and the weights traded on 2020-01-31
  were the ones computed on 2020-01-15. Covered by
  `qis/portfolio/tests/test_backtester_weights_consistency.py`.

- `weight_implementation_lag` counts observations of the price index rather than calendar days;
  see the note above. A weight whose traded date would fall past the end of the price history is
  dropped with a warning rather than silently, and every weight date is now traded exactly once.

### Added
- `qis.generate_static_weights_schedule(prices, weights, rebalancing_freq=...)` turns a fixed
  allocation into the rebalancing weight frame `backtest_model_portfolio` consumes, allocating
  over the instruments priced on each rebalancing date. A static vector against a panel whose
  instruments start and stop at different dates leaves the missing instrument's weight in the
  cash balance - the backtester takes weights as given and does not modify them - which is
  rarely the intended allocation. The universe is read at the rebalancing date only, so the
  construction is point in time. Rescaling preserves the total exposure of the specification
  rather than forcing the row to one, so a book that is 90% invested by design stays 90%
  invested; `is_preserve_total_exposure=False` gives the force-to-one form, and
  `is_rescale_to_live_universe=False` gives the cash residual with an explicit `0.0` rather than
  a nan in the reported weights. Core API, `Portfolio and backtesting`.

- `qis.align_weights_to_columns(weights, columns)` is the shared normaliser for the weight
  argument, so `generate_static_weights_schedule` and `backtest_model_portfolio` cannot disagree
  on what a Dict, a pd.Series, a List or an np.ndarray means. Behaviour is unchanged: a Dict or
  pd.Series aligns by name, a List or np.ndarray is positional, and the error contract the
  backtester documented is preserved.

- `backtest_model_portfolio` warns when a weighted instrument has no price on its traded date -
  the leg is not traded and its weight stays in cash - and names the instruments and the first
  date. Checked at the traded date rather than the weight date, so a lag that carries a weight
  past an instrument's last price is caught too. A zero weight against a missing price is
  silent, since holding a column at 0.0 before an instrument starts is deliberate.

- `backtest_model_portfolio` warns when prices carry missing values *inside* an instrument's own
  reported history, and names `prices.asfreq('B', method='ffill')` or `prices.ffill()` as the
  remedy. Units are held through a nan price and `np.nansum` drops the leg from the nav on those
  dates, so a hole removes that leg's whole value from the portfolio with no error. Leading nans
  (not trading yet) and trailing nans (no longer reporting) are legitimate and stay silent.

- `qis/utils/tests/test_static_weights_schedule.py` and
  `qis/portfolio/tests/test_backtester_weights_consistency.py`, including the regression pin that
  a lag of one reproduces the calendar-day schedule for `ME`, `QE`, `W-FRI` and `B` weight
  frequencies.

- `examples/portfolios/static_weight_with_missing_prices.py` and
  `examples/portfolios/lagged_weight_implementation.py`. Both run on the seeded synthetic panel
  with no network and no data file, so `test_examples.py` executes them top to bottom rather than
  only reading them - the first two examples under `examples/portfolios/` that it can. The first
  shows the cash residual, the reallocated schedule and what preserving the total exposure means
  for a book that is 90% invested by design; the second runs one monthly trend book at lags of 0,
  1, 5 and 20 observations, and reports that turnover and realised costs do not move with the lag
  while return and Sharpe do.

## [5.5.0] - 2026-08-01

**Every `axis=1` `pd.concat` in library code states `sort=` explicitly, and the three resampling
entry points sort a panel that reaches them out of order.** pandas 2.2 sorted the union of two
DatetimeIndexes whatever `sort=` said; pandas 3.0 honours `sort=False` and leaves the union in
appearance order, and warns that pandas 4 will stop sorting when the argument is absent. So a
panel joining a benchmark series and a strategy nav on different calendars now arrives unsorted,
and a call that says nothing means one thing today and another after the next major release. No
number moves under pandas 3.0 - the reporting goldens are unchanged - and none moves under
pandas 4 either, which was the point. `load_df_from_csv` also gains `float_precision`, so a CSV
round trip can be exact.

### Fixed
- `df_asfreq` sorts a panel whose index is not in chronological order before resampling it, and
  `compute_periodic_returns` sorts before its ffill/bfill. pandas 3.0 changed
  `pd.concat(axis=1, sort=False)` to leave the union of two non-identical DatetimeIndexes in
  appearance order, so a panel joining a benchmark series and a strategy nav on different
  calendars now arrives unsorted: `df_asfreq` raised `ValueError: index must be monotonic
  increasing or decreasing` from inside `pandas.reindex`, and the fill in
  `compute_periodic_returns` ran in row order and carried the terminal price backwards onto the
  dates a column does not carry, without raising. Covered by
  `qis/utils/tests/test_df_freq_sorting.py` and
  `qis/plots/derived/tests/test_returns_heatmap_sorting.py`.

- `prices_at_freq` sorts the same way on its `freq=None` branch, where the ffill runs in place
  rather than through `df_asfreq`, so `to_returns` without a resample is order-independent too.
  Covered by `qis/perfstats/tests/test_returns_sorting.py`.

- Every `axis=1` `pd.concat` in library code states `sort=` explicitly - 133 call sites in 39
  modules. `sort=True` where the joined index is dates, which is what pandas 2.2 did whatever
  the argument said; `sort=False` where it is instrument or statistic labels, which pandas has
  never sorted. Three sites were joining non-identical DatetimeIndexes and relying on the
  implicit sort that pandas 3.0 deprecates and pandas 4 removes: `compute_fx_optimal_hedge`,
  the FX hedging report, and the multi-frequency nav in `signal_diagnostics`. No number moves
  under pandas 3.0 - the reporting goldens are unchanged - and none moves under pandas 4 either,
  which was the point.

### Added
- `qis/tests/test_concat_sort_convention.py`: an `axis=1` `pd.concat` in library code without an
  explicit `sort=` fails the suite. What the union of two DatetimeIndexes does when the argument
  is absent has changed twice in two major pandas versions, and the difference is a scrambled
  time axis rather than an error.

- `load_df_from_csv(..., float_precision=None)` and
  `load_df_dict_from_csv(..., float_precision=None)`, forwarded to `pd.read_csv`. pandas' default
  C float converter is fast and not correctly rounded: a frame written with `save_df_to_csv` and
  read back differed from the original by up to ~4e-16 per cell on a realistic panel, which makes
  "the file holds what I wrote" impossible to assert. Passing `float_precision='round_trip'`
  returns the value bit for bit. The default is `None`, so nothing changes for an existing caller.
  Both docstrings state why the default converter is not exact and when to pass `'round_trip'`.

## [5.4.0] - 2026-07-29

**`backtest_model_portfolio` accepts `rebalancing_costs` as a panel of dates x tickers, and the
numba kernel beneath it, `backtest_rebalanced_portfolio`, takes a `(t, n)` array.** A float or a
per-instrument `pd.Series` behaves exactly as before, so a caller of the public wrapper sees
nothing change; a date-indexed `Series` now raises rather than being read as per-instrument, and a
direct caller of the kernel has to pass the broadcast array. Everything else in this release is
documentation, tests and packaging metadata, most of it hardening the JOSS submission after an
external review.

### Added
- `backtest_model_portfolio` accepts `rebalancing_costs` as a `pd.DataFrame` of dates x
  tickers: each price date takes the last schedule row at or before it, so a cost schedule
  stated on era boundaries (the `trendfollowing` volume-cost panel is the motivating case)
  applies from each boundary onward. A float or per-instrument `pd.Series` behaves exactly as
  before; a date-indexed Series now raises rather than being misread as per-instrument.
  Covered by `qis/portfolio/tests/test_backtester_costs.py`.

- `qis/tests/test_version_metadata.py`: `pyproject.toml`, `CITATION.cff` and the `@software`
  BibTeX entry in `README.md` must carry the same version, and `date-released` must be an ISO
  date. Nothing had held them together, and in the sibling `optimalportfolios` repository the
  same three read 6.3.0, 6.2.0 and versionless at one commit.
- `qis/tests/test_documentation.py` gains an in-page anchor check: every `#anchor` link resolves
  to a heading or an explicit `<a name>` in the same document. The README table of contents is
  thirteen such links, and a renamed section leaves the entry above it pointing nowhere with
  nothing failing.
- `qis/tests/test_documentation.py` also reads the README's `python` blocks as one script, in
  document order, and requires every bare name a block loads to be a builtin or a name bound in
  a block above it. Static rather than executed: the blocks call `yfinance`, and no test here
  may reach the network.
- `qis/models/bootstrap/tests/test_bootstrap_convention.py` requires `paper.md` to state the
  four values the convention example computes, at the two decimals the manuscript prints. The
  file's failure messages had always claimed the paper quoted them; nothing read it.

### Changed
- `backtest_rebalanced_portfolio` (the numba kernel under `backtest_model_portfolio`) takes
  `rebalancing_costs` as a `(t, n)` array; the wrapper broadcasts the scalar and
  per-instrument forms. Breaking only for direct callers of the kernel, of which the public
  consumers have none.
- Packaging metadata migrated to PEP 639: `license = "MIT"` with `license-files`, and the
  deprecated `License :: OSI Approved :: MIT License` classifier removed. `build-system`
  requires `setuptools>=77.0`, which is where that spelling is supported; this is a
  build-from-source floor and does not affect installing the wheel.
- Every `paper_phrase` in `docs/audit/paper_numbers.json` is a sentence-level fragment rather
  than a bare number. `94` was satisfied by the citation key `politis1994`, so that check would
  have passed on a manuscript that had stopped quoting the count. The two `privateassets`
  counts carry no phrase: the paper cites the package without a figure, so those two checks
  could not fail. `active_months` gains one.
- `paper.md` revised after an external review: the capability table states `vectorbt` and `bt`
  at documented-interface strength, `arch` is credited under it for the paired-resampling
  primitive and the no-counterpart claim is scoped to the integration, the API and example
  inventories give way to the audit record, and the body is 1,745 words against JOSS's 1,750.
  The statement of need also names the two comparisons a research pipeline runs and extends the
  shared-convention argument to agentic AI tooling.
- `README.md`: the table-of-contents entry `Notebooks` is now `Runnable examples`, matching the
  section it has pointed at since the notebooks were removed, and its anchor is renamed with it.
- `qis/docs/gallery.md` and its four screenshots moved to `docs/`. `MANIFEST.in` excludes `*.png`,
  so the gallery shipped inside the wheel with four image links that resolved to nothing for every
  installed user. It is a documentation page rather than a package note, and shipping the images
  instead would have added 1.3 MB to a 730 KB wheel for four screenshots. `AGENTS.md` now states
  the rule that decides which tree a document belongs in.
- `README.md`'s notebooks section points at `qis/examples/` and names the four factsheet scripts.

### Fixed
- The README's example blocks used `PerfStat` twelve times without importing it, so a reader
  pasting them in order got `NameError` at the performance table. The import is added, and the
  new namespace check above fails if it is removed again. Three fences carrying `pip` and `git`
  commands were labelled `python`; they are labelled `bash`.
- `tools/paper_audit.py --check` exits nonzero on any measurement warning and on any difference
  between the generated and the stored metric key sets. It compared only the metrics it had
  measured, so with `docs/audit/consumers.json` absent it compared 16 of 22 and returned 0.

### Removed
- `notebooks/`, six Jupyter notebooks last touched on 2025-07-19. They were the only documented
  surface with no test covering it: `qis/tests/test_examples.py` checks all 58 example scripts,
  and nothing checked a notebook, so drift in one was invisible - the stored outputs kept
  rendering last year's numbers as embedded images. Every notebook duplicated ground already
  covered by a tested script: `multi_assets.py`, `strategy.py`, `strategy_benchmark.py` and
  `multi_strategy.py` under `qis/examples/factsheets/`, `us_election_regimes.py` under
  `qis/examples/regimes/`, and `quickstart.py` under `qis/examples/perfstats/`. Nothing shipped in
  the wheel changes; the notebooks were never in it.

## [5.3.0] - 2026-07-27

**`qis.__all__` now exists and fixes the public surface at 403 names.** Nothing is added to or
removed from the namespace, but `from qis import *` is now defined by an explicit list rather
than by whatever `dir()` returned at the time, and `dir(qis)` no longer answers the question of
what is public: importing a submodule binds its name on the package, so `dir(qis)` grew by one
whenever a process imported `qis.api`. Anything counting the public surface should read
`qis.__all__`.

The seeded data generator moved from `qis.tests.synthetic_data` to `qis.datasets.synthetic`.
The old path still imports the same module, so nothing breaks, but the quickstart no longer tells
a reader to import from a `tests` namespace. Seeds, draw order and every golden pinned to the
generator are unchanged.

### Added
- `qis.datasets`, re-exporting `generate_synthetic_universe`, `generate_synthetic_prices`,
  `SyntheticUniverseData`, `SyntheticInstrument`, `DataQuirk`, `SYNTHETIC_UNIVERSE`,
  `GROUP_ORDER`, `BENCHMARK_TICKER` and `BENCHMARK_WEIGHTS`. The module ships in the wheel; the
  `qis.tests.synthetic_data` path is kept as a compatibility shim.
- `qis.api.PUBLIC_API`, the export list as a literal, so a change to the public surface appears
  in a diff. `tools/sync_public_api.py` regenerates it and `qis/tests/test_core_api.py` fails
  when it disagrees with the namespace.
- `tools/paper_audit.py` and `docs/audit/paper_numbers.json`: every number `paper.md` quotes,
  generated rather than hand-measured. `qis/tests/test_paper_audit.py` fails when the record, the
  repository and the manuscript disagree, and when the manuscript quotes a large number the
  record does not know about.
- `tools/audit_consumers.py` and `docs/audit/consumers.json`: qis usage in its public consumers
  at pinned commits, with the counting rule stated in the script's docstring.
- `qis/models/bootstrap/tests/test_bootstrap_convention.py`, pinning the published values of the
  bootstrap convention example (0.110, 0.526, +2.15%, -0.32% and the rest) rather than only
  asserting that the example exits zero, and checking that `docs/reproducibility.md` still states
  them.
- `qis/tests/test_documentation.py`, asserting that every repository-internal documentation link
  resolves to a file that exists, and that the README's core dependency list is the `dependencies`
  table of `pyproject.toml`.

### Fixed
- `qis/api.py`'s module docstring stated 386 exports, 98 core symbols, 288 non-core, 109
  private-use and 179 uncalled, and said `market_data` had no core symbol, against a `CORE_API`
  holding 116 symbols in 12 groups including 13 market-data and FX names. Every count is removed
  from the prose; the generated record carries them.
- `README.md` listed `yfinance` and `pandas-datareader` as core dependencies; both are in the
  `[data]` extra.
- `README.md`'s ecosystem table omitted `privateassets`, which `paper.md` names as one of the
  three public consumers.
- `README.md` linked to `qis/examples/performances.py` and `qis/examples/notebooks`, neither of
  which has existed since the examples were reorganised, and embedded `perf1`, `perf2` and
  `perf3`, which `.gitignore` excludes by name, so the front page rendered three broken images.
  The links now point at `qis/examples/perfstats/quickstart.py` and `notebooks/`; the three
  embeds are removed and the runner that produces them is named instead.
- `qis/portfolio/backtester.py` had no final newline, which is why it measured 273 lines by
  `wc -l` and 274 by every other count.

### Changed
- `paper.md`: the state-of-the-field section is a capability comparison at checked versions
  rather than a two-class taxonomy, and adds `skfolio`. All ten rows were read from each
  package's current documentation on 2026-07-27. The impact section quotes consumer counts at
  pinned commits and gives no figure for private repositories; every count is taken from the
  generated record; "cannot drift" is replaced by the invariant the tests enforce; the body is
  1,749 words.

### Removed
- `qis/perfstats/ra_returns.py`, an unreferenced duplicate of `qis/models/linear/ra_returns.py`.
  It was added on 2026-04-19 and never imported: not by `qis/perfstats/__init__.py`, not by any
  module or test, and `qis.perfstats.ra_returns` was absent from `sys.modules` after
  `import qis`. Every `qis.<symbol>` in the pair already resolved to the `models.linear` copy, so
  nothing exported changes and the export count stays at 403. The copies had begun to diverge:
  `compute_ewm_long_short_filtered_ra_returns` gained its docstring and its span validation in
  the live module and not in the duplicate, so a deep import of `qis.perfstats.ra_returns`
  returned a version that accepted `vol_span` below 1 and produced NaN through a negative
  variance. Only a direct import of that module path is affected.

## [5.2.1] - 2026-07-27

**`compute_ar_residuals` and `bootstrap_ar_process` change their results on data with gaps, and
raise where they previously returned a number.** Any AR bootstrap run on a series with a missing
observation moves in this release, and a panel with a row incomplete across columns now raises
instead of returning NaN. On gap-free data the fitted AR(1) does not move: it is still ordinary
least squares on the lag pairs, agreeing with the previous `statsmodels AutoReg` fit to 6.7e-16
across persistences from -0.6 to 0.95 and lengths 60 to 2000. Seeded bootstrap paths do move on
gap-free data too, because the residual indices are now drawn over the n-1 residual rows instead
of the n data rows.

**5.2.0 is yanked and this release replaces it.** It was published from an uncommitted working
tree, so no commit set that version and the release could not be reproduced from source. It also
shipped without the second bounds check below.

### Fixed
- `compute_ar_residuals` raised `KeyError: 0` under pandas 3.0. `AutoReg(...).fit().params` is
  indexed by name, so `params[0]` was a label lookup. `bootstrap_ar_process` went down with it.
- `compute_ar_residuals` returned NaN residuals for data with gaps. The fit used `dropna()` while
  the residuals were computed on the original array, so every missing observation left NaN in two
  residual rows, and those entered every draw that sampled them.
- `compute_ar_residuals` fitted steps that spanned a gap as if they were one period apart,
  because `dropna()` makes the observations either side of a gap adjacent. An AR(1) at spacing k
  has persistence theta^k, so the estimate was pulled towards zero.
- `bootstrap_ar_process` drew indices over `len(data.index)` while the residual array has one row
  fewer, so the largest index was one row past the end. `get_bootstrap_ar_data_list` is `@njit`
  with bounds checking off, so the read did not raise: on a 50-point series it returned 5.6e-321
  from adjacent memory and used it as a residual. The draw is now taken over `len(residuals)`.
- `bootstrap_ar_process` rejects a supplied `bootstrapped_indices` that reaches past the residual
  rows. `bootstrap_price_fundamental_data` draws one index set over `len(prices.index)-1` and
  passes it to both the price path and the AR path, which is what keeps the two resampled
  together. Gaps in the fundamental panel shorten the residuals below that length, so the same
  out-of-bounds read returned through that path. Quarterly fundamental panels are where this
  bites.

### Added
- `qis/models/bootstrap/tests/test_bootstrap_ar.py` — 14 tests over the AR residual path. Each of
  the five defects above has a test that fails on it alone.
- `qis/examples/models/ar_bootstrap_gaps.py` — measures what a gap costs an AR(1) fit. Over 20
  gap patterns blanking 30% of a 3000-point series at persistence 0.7, dropping the lag pairs
  that straddle a gap deviates from the gap-free estimate by 0.001 to 0.026, while collapsing the
  gaps first deviates by 0.064 to 0.095. The ranges do not overlap. No network, and the test
  suite executes it.
- `qis/examples/models/bootstrap_convention.py` — measures what the block-resampling convention
  costs. The superseded truncating sampler draws the first observation of a 250-period sample at
  0.11x its uniform weight and the first decile at 0.53x; applied to a series with rising drift
  it reports a mean return 2.15% per year above the source, against 0.32% for the circular
  sampler now used.
- `docs/reproducibility.md` — the same measurement as a documentation page, with what follows
  from it for the return convention, the Sharpe conventions and the reported frequency.

### Changed
- `compute_ar_residuals` requires each row to be complete across all columns, and raises
  `ValueError` below three usable lag pairs. Rows are resampled jointly to preserve the
  cross-section, so a row missing one column cannot be resampled coherently; previously such a
  row produced NaN silently. A constant series now returns a beta of exactly 0.0 and an intercept
  at the level, where `AutoReg` returned 9.999e-05.
- `qis/models/bootstrap/bootstrap_numba.py` no longer imports `statsmodels`. The AR(1) is
  estimated directly; `statsmodels` remains a dependency, used in eight other modules.

## [5.1.0] - 2026-07-26

**`BootstrapType.STATIONARY` produces different draws in this release.** Blocks now wrap around
the end of the sample, which is the correct Politis-Romano construction; see *Changed* below.
Any result produced with `bootstrap_data`, `bootstrap_price_data`, `bootstrap_ar_process` or
`bootstrap_price_fundamental_data` under `STATIONARY` will move. Nothing else changes an
existing number.

### Added
- `BootstrapType.FIXED_BLOCK` — circular block resampling with a block of exactly `block_size`,
  for a block length chosen to match a known cycle rather than drawn.
- `min_block_size` on `generate_bootstrapped_indices`, `bootstrap_data`, `bootstrap_ar_process`,
  `bootstrap_price_data` and `bootstrap_price_fundamental_data`. Floors the drawn block length
  under `BootstrapType.STATIONARY`; set it to the number of periods in the slowest-reporting
  series when the panel mixes frequencies. Default `1`, which is the previous behaviour.
- `theta` on `unsmooth_returns_glm` — supply the Getmansky-Lo-Makarov smoothing weights instead
  of estimating them, for a coefficient that comes from outside the series (a panel estimate
  pooled across vintages, or a value fixed for a production run). A scalar or an array of
  length q; the sample-length guard does not apply, since nothing is fitted. Default `None`,
  which estimates as before.
- `qis/models/bootstrap/tests/test_bootstrap_numba.py` and
  `qis/models/unsmoothing/tests/test_ar_lag_glm.py`.
- `qis/tests/test_examples.py` — every file in `qis/examples/` must parse, every `qis.<name>`
  it references must resolve, every qis module it imports from must exist with the symbol it
  names, and every keyword it passes to a qis callable must be in that callable's signature.
  The nine examples that reach no data vendor are executed in a temporary directory. An
  optional dependency is a skip, never a failure.
- `qis/api.py` — `CORE_API`, the documented core of the public API: 103 symbols grouped by
  capability. Nothing is un-exported; the module records which exports the documentation
  promises to describe. The boundary is measured — a symbol is core when a package that depends
  on qis, or qis's own examples, README or docs, calls it — plus five bootstrap symbols
  promoted by intent. `qis/tests/test_core_api.py` enforces it: a core symbol without an
  `Args:`/`Attributes:` block fails the suite, and the `PENDING_DOCSTRINGS` backlog is a ratchet
  that cannot silently hide finished work.
- `qis/docs/plotting_kwargs.md` — the keyword arguments every `plot_*` function shares (`ax`,
  `title`, `var_format`, `x_date_freq`, `fontsize`, `colors`, `legend_loc`, `y_limits`),
  documented once so individual plot docstrings cover only what is specific to them.
- Six FX analytics exported from `qis`: `compute_fx_optimal_hedge`, `compute_fx_vol_beta`,
  `compute_performance_of_local_ccy_asset_in_reference_ccy`, `compute_multi_asset_fx_hedging`,
  `run_asset_fx_hedging_report` and `plot_multi_asset_fx_hedging_report`, plus
  `compute_local_and_fx_return` and `compute_cash_fx_adjusted_returns`. `market_data` published
  5 of the 11 symbols its consumers use, and the rest were reached by deep import into
  `qis.market_data.fx_hedging` and `qis.market_data.reports.fx_hedging_report`. Market data is
  now a capability group in `CORE_API`. Roadmap item T5.

### Changed
- **`BootstrapType.STATIONARY` blocks now wrap around the end of the sample**, as in
  Politis-Romano (1994). Previously a block was cut short at the last observation, so the
  realised block length was not geometric there and the first observations were drawn far less
  often than the rest: on a 250-period sample with `block_size=20` the first observation
  appeared at 0.11x the uniform rate and the first decile at 0.53x. **This changes every
  `STATIONARY` draw**, including `rosaa/research/analysis/crypto_publication.py`. The effect on
  a long daily sample is confined to the first `block_size` observations; on quarterly or
  monthly panels it is material.
- `generate_bootstrapped_indices` raises with the offending value on an unhandled
  `bootstrap_type`, rather than a bare `not implemented`.
- The API reference is split into **Core API**, grouped by capability, and **Also exported**.
  It is still generated from `dir(qis)` at build time, so it cannot drift from the exports.
- Docstrings with `Args:`/`Attributes:` blocks on `MeanAdjType`, `NanBackfill`,
  `BootstrapOutput` and on every plot function in the documented core: `plot_time_series`,
  `plot_prices`, `plot_prices_with_dd`, `plot_bars`, `plot_scatter`,
  `plot_classification_scatter`, `plot_heatmap`, `plot_qq`, `plot_df_table`,
  `df_boxplot_by_classification_var`, `df_boxplot_by_hue_var`, `set_suptitle` and
  `plot_exposures_strategy_vs_benchmark_stack`. Each documents only its own arguments and
  refers to `qis/docs/plotting_kwargs.md` for the shared ones. Prose docstrings on `PerfStat`
  and `LegendStats`, whose members are compositional and where a per-member block would restate
  the names.
- **Every symbol in `CORE_API` now carries an `Args:`/`Attributes:` block.** The remaining 56
  in this release: the `df_*` aggregation family, the `file_utils` readers and writers, the
  date and annualisation helpers, `get_group_dict` / `split_df_by_groups`, the three
  `df_to_*_allocation` normalisers, `covar_to_corr`, `np_array_to_df_columns`,
  `fit_multivariate_ols`, `compute_masked_covar_corr`, `estimate_rolling_ewma_covar`,
  `compute_ewm_covar_tensor_vol_norm_returns`, `compute_ewm_long_short_filtered_ra_returns`,
  `estimate_hf_ohlc_vol`, `interpolate_infrequent_returns`, `get_ra_perf_columns`,
  `unsmooth_returns_ar1_ewma`, `bootstrap_data`, `bootstrap_price_data`, `EwmLinearModel`,
  `FxRatesData`, `FactorsData`, the FX conversion functions, `fetch_default_report_kwargs`
  and the two factsheet generators. `fetch_default_report_kwargs` moves from numpydoc to the
  house Google style.
- The six remaining numpydoc docstrings in `qis` are converted to Google style:
  `get_nonnan_index`, `FxRatesData.build_cross_fx_cash_nav`, `estimate_dimson_beta`,
  `fetch_factsheet_config_kwargs`, `fetch_default_perf_params` and the FX rates example.
  `qis/tests/test_docstring_convention.py` fails the suite on a numpydoc section heading
  anywhere in the package. `factorlasso` keeps numpydoc; the exception is per-package.

### Fixed
- `unsmooth_returns_ar1_ewma`, `unsmooth_returns_glm` and `compute_ar1_unsmoothed_prices` are
  exported from `qis`. They were documented and referenced as `qis.<name>` by
  `qis/examples/perfstats/unsmoothing_and_delevering.py`, which raised `AttributeError`.
- `bootstrap_price_data` ignored `min_block_size` instead of forwarding it to the sampler.
- `qis/examples/plots/dual_axis_figure.py` wrote its PDF to `qis.get_output_path()`, which
  reads `settings.yaml` and ships as the placeholder `C:\Users\...\`. The example raised
  `FileNotFoundError` on every machine. It now writes to the working directory, as the other
  examples do.

## [5.0.10] - 2026-07-25

### Added
- `qis/tests/synthetic_data.py` — seeded synthetic multi-asset panel for tests, CI and
  documented examples. `generate_synthetic_prices` and `generate_synthetic_universe` draw a
  10-instrument panel carrying ragged starts, missing observations, stale prices, a delisted
  tail, fat tails, appraisal smoothing and a monthly-reported sleeve, with no network and no
  data file. Internal: not exported from `qis/__init__.py`.
- `qis/plots/tests/plot_smoke_test.py` — every `plot_*` exported from `qis` runs on that panel
  and must draw a figure. The parametrisation is read from `dir(qis)` at collection time, so a
  newly exported `plot_*` without a fixture fails the suite rather than going uncovered.
- CI runs `pytest` on the 3.10–3.14 matrix against a core install, repeats it with the
  `[data,io]` extras, and lints the lines a push or pull request changes with a pinned `ruff`.

### Changed
- `[tool.ruff.lint] select` drops `"I"`. The isort rule contradicts the documented import
  convention, which groups stdlib imports under `# packages` after numpy/pandas, so it failed
  every file written to the house style.

### Fixed
- `plot_prices_2ax` passed `trend_line` into `plot_time_series_2ax`, which takes `trend_line1` /
  `trend_line2`, and the stray keyword reached `plot_time_series` alongside `trend_line1`
  (`TypeError`).
- `plot_regime_pdf` called `_asdict()` on `BenchmarkReturnsQuantilesRegime`, which is a class
  and not a NamedTuple (`AttributeError`), and overwrote the caller's `regime_classifier` on the
  preceding line, so that argument was ignored.
- `plot_vbars` indexed a per-row colour array by column (`IndexError` above four columns) and
  left the y locator to matplotlib, so the label count matched only on a frame with exactly
  eight rows (`ValueError`). It also drew every bar in the wrong place on a `DatetimeIndex`: the
  value labels and total markers address rows by integer position, which `barh` honours only for
  non-numeric labels, so the index is now coerced to strings.
- `plot_multivariate_scatter_with_prediction` dereferenced `ax.get_legend()` unguarded and raised
  whenever `hue` was left at its default, since seaborn draws no legend in that case.
- `pytest` at the repository root collected nothing useful: `testpaths` pointed at a top-level
  `tests/` that does not exist, and three modules failed to import. `testpaths` is now `qis`,
  collection uses `--import-mode=importlib`, the `yfinance` import in
  `qis/tests/price_data_test.py` moved inside the branch that needs it, and the parquet and
  feather tests skip rather than fail without the `[io]` extra.

## [5.0.9] - 2026-07-22

### Added
- `SharpeConvention` is exported from `qis`. `compute_regimes_pa_perf_table` and
  `plot_regime_data` accept the convention, so regime tables state which Sharpe object they
  report. `qis/docs/sharpe_conventions.md` extended with the regime decomposition.

## [5.0.8] - 2026-07-16

### Added
- `SharpeConvention` (`PA`, `ARITHMETIC`, `LOG`) on `PerfParams`, defaulting to `PA` so no
  existing statistic changes value, and `compute_regime_sharpe_decomposition`, which is
  exactly additive in the arithmetic convention.

## [5.0.7] - unreleased tag

[TODO: 5.0.7 is on PyPI but no commit in this repository sets that version. Reconstruct from
the uploaded sdist or yank it.]

## [5.0.6] - unreleased tag

[TODO: 5.0.6 is on PyPI but no commit in this repository sets that version. Reconstruct from
the uploaded sdist or yank it.]

## [5.0.5] - 2026-07-13

### Fixed
- Typo fixes and method visibility corrections across `market_data.fx_rates_data`,
  `models.bootstrap.bootstrap_numba`, `portfolio` and the pybloqs factsheet examples.

## [5.0.4] - 2026-07-12

### Fixed
- `qis/portfolio/reports/config.py` did not import `infer_data_frequency_label`, so
  `import qis` failed in 5.0.3. Hotfix release.

## [5.0.3] - 2026-07-12

### Changed
- `yfinance` and `pandas-datareader` moved out of the core dependencies into a
  new `[data]` extra (`pip install qis[data]`). The analytics core no longer
  installs a data-vendor client. The two function-local imports in
  `qis/portfolio/reports/config.py` (the `^IRX` download behind
  `add_rates_data=True`) raise an `ImportError` naming the extra when it is not
  installed. The `[all]` extra includes `data`, so `pip install qis[all]` is
  unchanged.

## [5.0.2] - 2026-07-12

### Added
- Explicit arithmetic Sharpe convention alongside the p.a. (compound) default.
  New `PerfStat` members `AN_ARITH_RETURN`, `AN_ARITH_EXCESS_RETURN`,
  `AVG_ARITH_RETURN`, `AVG_ARITH_EXCESS_RETURN`, `SHARPE_ARITH` and
  `SHARPE_ARITH_EXCESS` report `a * mean(r_m)` and `a * mean(r_m - rf_m)` on
  simple returns at `freq_vol`, with the compounded and log Sharpe ratios
  (`SHARPE_EXCESS`, `SHARPE_LOG_AN`, `SHARPE_LOG_EXCESS`) unchanged. Every
  Sharpe object is now labeled in the output, so the convention is stated rather
  than implied.
- `qis/docs/sharpe_conventions.md` — decision record deriving the three Sharpe
  objects (compound, log, arithmetic) and their reconciliation, and
  `qis/perfstats/tests/sharpe_conventions_test.py` pinning the identities.

### Changed
- The p.a. (compound) Sharpe remains the qis default. No existing statistic
  changes value; the arithmetic convention is additive and opt-in.

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

## [4.3.4] - 2026-07-11

Never tagged; ships to users as part of 5.0.3.

### Added
- `qis.perfstats.signal_diagnostics` — cross-sectional predictive regression
  diagnostics for trading signals with per-asset native-cadence handling. For an
  N-asset panel of signal scores and per-frequency return panels,
  `estimate_signal_diagnostics` quantifies cross-sectional predictive content at
  one or more forward horizons via `y_{i,t,t+h} = beta * z_{i,t-1} + eps_{i,t}`
  (through the origin by default), with the horizon expressed in each asset's
  native periods. `compute_per_asset_betas`, `compute_ic_timeseries` and
  `estimate_ic_ir` expose the per-asset betas, the per-date information
  coefficient series and its IC-IR.
- `qis.min_obs_for_ar_unsmoothing` — minimum observation count required for an
  AR(q) unsmoothing fit given `ar_order` and `warmup_period`, validating both
  (`ValueError` naming the offending value).

### Changed
- AR unsmoothing (`qis.models.unsmoothing.ar_lag`) gains opt-in guards for
  short / degenerate columns, selected by enum. Defaults reproduce the previous
  behaviour exactly; `RAISE` reports the offending columns and their observation
  counts instead of silently returning an all-NaN column.

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
