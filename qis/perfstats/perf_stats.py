"""
Compute risk-adjusted performance tables.

This module provides the core performance and risk analytics for the QIS framework,
including return statistics, drawdown analysis, factor regressions, and information
ratios. All public function signatures are preserved for backward compatibility with
downstream modules.
"""
# packages
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from typing import Callable, Union, Dict, Tuple, Any, Optional, Literal

# qis
import qis.utils.regression as ols
import qis.perfstats.returns as ret
from qis.perfstats.config import PerfStat, PerfParams, ReturnTypes
from qis.utils.annualisation import get_annualization_factor, infer_annualisation_factor_from_df

# =============================================================================
# COLUMN PRESET DEFINITIONS
# =============================================================================
# These tuples define the standard column orderings used by reporting functions
# downstream. Do not reorder or remove without checking dependents.

STANDARD_TABLE_COLUMNS = (PerfStat.START_DATE,
                          PerfStat.END_DATE,
                          PerfStat.PA_RETURN,
                          PerfStat.VOL,
                          PerfStat.SHARPE_RF0,
                          PerfStat.MAX_DD,
                          PerfStat.MAX_DD_VOL,
                          PerfStat.SKEWNESS,
                          PerfStat.KURTOSIS)

LN_TABLE_COLUMNS = (PerfStat.START_DATE,
                    PerfStat.END_DATE,
                    PerfStat.TOTAL_RETURN,
                    PerfStat.PA_RETURN,
                    PerfStat.AN_LOG_RETURN,
                    PerfStat.VOL,
                    PerfStat.SHARPE_RF0,
                    PerfStat.SHARPE_LOG_AN,
                    PerfStat.MAX_DD,
                    PerfStat.MAX_DD_VOL,
                    PerfStat.SKEWNESS,
                    PerfStat.KURTOSIS)

LN_BENCHMARK_TABLE_COLUMNS = (PerfStat.START_DATE,
                              PerfStat.END_DATE,
                              PerfStat.TOTAL_RETURN,
                              PerfStat.PA_RETURN,
                              PerfStat.AN_LOG_RETURN,
                              PerfStat.VOL,
                              PerfStat.SHARPE_RF0,
                              PerfStat.SHARPE_LOG_AN,
                              PerfStat.MAX_DD,
                              PerfStat.MAX_DD_VOL,
                              PerfStat.SKEWNESS,
                              PerfStat.KURTOSIS,
                              PerfStat.ALPHA,
                              PerfStat.BETA,
                              PerfStat.R2)

LN_BENCHMARK_TABLE_COLUMNS_SHORT = (PerfStat.TOTAL_RETURN,
                                    PerfStat.PA_RETURN,
                                    PerfStat.AN_LOG_RETURN,
                                    PerfStat.VOL,
                                    PerfStat.SHARPE_RF0,
                                    PerfStat.SHARPE_LOG_AN,
                                    PerfStat.MAX_DD,
                                    PerfStat.MAX_DD_VOL,
                                    PerfStat.SKEWNESS,
                                    PerfStat.ALPHA,
                                    PerfStat.BETA,
                                    PerfStat.R2)

EXTENDED_TABLE_COLUMNS = (PerfStat.START_DATE,
                          PerfStat.END_DATE,
                          PerfStat.START_PRICE,
                          PerfStat.END_PRICE,
                          PerfStat.TOTAL_RETURN,
                          PerfStat.PA_RETURN,
                          PerfStat.VOL,
                          PerfStat.SHARPE_RF0,
                          PerfStat.SHARPE_EXCESS,
                          PerfStat.MAX_DD,
                          PerfStat.MAX_DD_VOL,
                          PerfStat.SKEWNESS,
                          PerfStat.KURTOSIS)

COMPACT_TABLE_COLUMNS = (PerfStat.TOTAL_RETURN,
                         PerfStat.PA_RETURN,
                         PerfStat.VOL,
                         PerfStat.SHARPE_RF0,
                         PerfStat.MAX_DD,
                         PerfStat.MAX_DD_VOL,
                         PerfStat.SKEWNESS)

SMALL_TABLE_COLUMNS = (PerfStat.TOTAL_RETURN,
                       PerfStat.PA_RETURN,
                       PerfStat.VOL,
                       PerfStat.SHARPE_EXCESS,
                       PerfStat.MAX_DD)

BENCHMARK_TABLE_COLUMNS = (PerfStat.PA_RETURN,
                           PerfStat.VOL,
                           PerfStat.SHARPE_RF0,
                           PerfStat.MAX_DD,
                           PerfStat.SKEWNESS,
                           PerfStat.ALPHA_AN,
                           PerfStat.BETA,
                           PerfStat.R2,
                           PerfStat.ALPHA_PVALUE)

BENCHMARK_TABLE_COLUMNS2 = (PerfStat.TOTAL_RETURN,
                            PerfStat.PA_RETURN,
                            PerfStat.VOL,
                            PerfStat.SHARPE_EXCESS,
                            PerfStat.MAX_DD,
                            PerfStat.MAX_DD_VOL,
                            PerfStat.SKEWNESS,
                            PerfStat.ALPHA_AN,
                            PerfStat.BETA,
                            PerfStat.R2)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _safe_downside_vol(returns_array: np.ndarray, vol_dt: float) -> float:
    """Compute annualised downside volatility with safe handling of empty/sparse arrays.

    The standard downside vol is the annualised stdev of strictly negative returns.
    When there are 0 or 1 negative observations, np.std with ddof=1 produces a
    RuntimeWarning and NaN. This helper returns 0.0 in those degenerate cases.

    Args:
        returns_array: 1-D array of returns (any sign).
        vol_dt: Square-root annualisation factor (e.g. sqrt(252) for daily).

    Returns:
        Annualised downside volatility, or 0.0 if fewer than 2 negative observations.
    """
    neg_returns = returns_array[returns_array < 0.0]
    if len(neg_returns) < 2:
        # ddof=1 with N<2 is undefined; downside vol is meaningless
        return 0.0
    return vol_dt * float(np.std(neg_returns, ddof=1))


def _resolve_benchmark(prices: pd.DataFrame,
                       benchmark: Optional[str],
                       benchmark_price: Optional[pd.Series]
                       ) -> Tuple[pd.DataFrame, str]:
    """Normalise the three input modes for benchmark specification.

    The benchmark can be supplied in three ways:
      Case 1: ``benchmark`` is a column name already in ``prices``, ``benchmark_price`` is None.
      Case 2: ``benchmark_price`` is a Series, ``benchmark`` is None — the Series name is used
              and the Series is reindexed/concatenated into ``prices``.
      Case 3: Both supplied — ``benchmark_price`` is concatenated into ``prices`` under the
              name given by ``benchmark`` (Series name is ignored). If the column is already
              present in ``prices`` it is left as-is.

    Args:
        prices: Asset price DataFrame.
        benchmark: Optional benchmark column name.
        benchmark_price: Optional benchmark price Series.

    Returns:
        Tuple of (prices DataFrame possibly augmented with benchmark column, benchmark name).

    Raises:
        ValueError: If neither benchmark nor benchmark_price is supplied, or if benchmark
            is given as a name but is not in prices and benchmark_price is None.
        TypeError: If benchmark_price is supplied but is not a pd.Series.
    """
    # Case 0: nothing supplied
    if benchmark is None and benchmark_price is None:
        raise ValueError("provide either benchmark name in prices or benchmark_price")

    # Case 1: benchmark name only — must already be in prices
    if benchmark_price is None:
        if benchmark not in prices.columns:
            raise ValueError(f"{benchmark} is not in {prices.columns.to_list()}")
        return prices, benchmark

    # benchmark_price was supplied — type check
    if not isinstance(benchmark_price, pd.Series):
        raise ValueError(f"benchmark_price must be pd.Series not {type(benchmark_price)}")

    # Case 2: benchmark_price only — use its .name as the column name
    if benchmark is None:
        name = benchmark_price.name
        if name in prices.columns:
            # column already present — trust the existing data
            return prices, name
        # reindex to prices' calendar with forward-fill for missing observations
        aligned = benchmark_price.reindex(index=prices.index, method='ffill').ffill()
        prices_out = pd.concat([aligned.rename(name), prices], axis=1)
        return prices_out, name

    # Case 3: both supplied — explicit name overrides Series.name
    if benchmark in prices.columns:
        # column already present — trust the existing data, ignore benchmark_price
        return prices, benchmark
    aligned = benchmark_price.reindex(index=prices.index, method='ffill').ffill()
    prices_out = pd.concat([aligned.rename(benchmark), prices], axis=1)
    return prices_out, benchmark


# =============================================================================
# PERFORMANCE TABLE
# =============================================================================

def compute_performance_table(prices: Union[pd.DataFrame, pd.Series],
                              perf_params: PerfParams,
                              ) -> pd.DataFrame:
    """Compute return-based performance metrics per asset.

    Iterates per asset and delegates to ``ret.compute_returns_dict``, which produces
    total return, p.a. return, log returns and related statistics. Per-asset iteration
    is required because each column is dropped of NaNs independently to handle assets
    with different inception dates.

    Args:
        prices: DataFrame of asset price levels (dates × assets).
        perf_params: Performance parameter object specifying frequencies and conventions.

    Returns:
        DataFrame indexed by asset with columns from the returns dict.

    Raises:
        TypeError: If ``prices`` is not a DataFrame.

    Note:
        ``compute_ra_perf_table`` calls both this and ``compute_risk_table``, which
        independently resample the same price data. This duplication is acceptable for
        typical universes but could be optimised via a shared cache for large panels.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"must be pd.Dataframe")

    if perf_params is None:
        perf_params = PerfParams()

    # Build per-asset return dict — each asset is dropna'd separately to respect
    # heterogeneous inception dates across the panel.
    dict_data = {}
    for asset in prices:
        asset_data = prices[asset].dropna()  # force drop na
        return_dict = ret.compute_returns_dict(prices=asset_data, perf_params=perf_params)
        dict_data[asset] = return_dict
        # keys will be rows = asset, column = keys in return_dict
    data = pd.DataFrame.from_dict(data=dict_data, orient='index')
    return data


# =============================================================================
# RISK TABLE
# =============================================================================

def compute_risk_table(prices: pd.DataFrame,
                       perf_params: PerfParams = None
                       ) -> pd.DataFrame:
    """Compute risk metrics (vol, drawdown, skew/kurtosis) per asset using vectorised ops.

    Resamples prices at the configured frequencies for vol, drawdown and skewness, then
    computes all risk metrics column-wise on the full DataFrame rather than per-asset.
    Per-asset NaN handling is preserved by dropna inside each metric computation.

    Args:
        prices: DataFrame of asset price levels (dates × assets).
        perf_params: Performance parameter object. If None, a default PerfParams() is used.

    Returns:
        DataFrame indexed by asset with risk metric columns including:
        VOL, DOWNSIDE_VOL, AVG_LOG_RETURN, AVG_ARITH_RETURN, AN_ARITH_RETURN,
        AVG_ARITH_EXCESS_RETURN, AN_ARITH_EXCESS_RETURN, SHARPE_ARITH,
        SHARPE_ARITH_EXCESS, START_DATE, END_DATE, NUM_OBS,
        MAX_DD, CURRENT_DD, MAX_DD_VOL, WORST, BEST, SKEWNESS, KURTOSIS.

    Raises:
        TypeError: If ``prices`` is not a DataFrame.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"prices must be dataframe.")

    if perf_params is None:
        perf_params = PerfParams()

    # ── Resample to the three configured frequencies (vol, drawdown, skewness) ──
    sampled_prices_vol = ret.prices_at_freq(prices=prices, freq=perf_params.freq_vol)

    # drawdowns are computed using its own freq
    if perf_params.freq_vol == perf_params.freq_drawdown:
        dd_sampled_prices = sampled_prices_vol
    else:
        dd_sampled_prices = ret.prices_at_freq(prices=prices, freq=perf_params.freq_drawdown)

    # skeweness computed at own frequency
    if perf_params.freq_vol == perf_params.freq_skewness:
        sampled_prices_skew = sampled_prices_vol
    else:
        sampled_prices_skew = ret.prices_at_freq(prices=prices, freq=perf_params.freq_skewness)

    an_factor = infer_annualisation_factor_from_df(data=sampled_prices_vol)
    vol_dt = np.sqrt(an_factor)

    # ── Compute returns once at the vol and skewness frequencies ──
    # to_returns is bulk-vectorised; per-asset dropna is applied inside metric loops.
    returns_vol = ret.to_returns(prices=sampled_prices_vol,
                                 return_type=perf_params.return_type,
                                 drop_first=True)
    returns_skew = ret.to_returns(prices=sampled_prices_skew,
                                  return_type=perf_params.return_type,
                                  drop_first=True)
    pct_returns_dd = dd_sampled_prices.pct_change()

    # ── Bulk vectorised metrics across all assets ──
    # std uses ddof=1 to match the original per-asset numpy call.
    vol_series = returns_vol.std(ddof=1) * vol_dt
    avg_log_return = returns_vol.mean()  # nanmean equivalent for pandas

    # ── Arithmetic Sharpe family ──
    # SR_arith = sqrt(a) * mean(r_m) / std(r_m) on simple returns at freq_vol
    # (Sharpe 1994 plug-in estimator). Numerator and denominator are paired on the
    # same simple-return series and do not reuse the table vol, which follows
    # perf_params.return_type (LOG by default); the std(r) vs std(l) wedge is third-order.
    # Note SR_arith is an estimate at the sampling frequency freq_vol.
    if perf_params.return_type == ReturnTypes.RELATIVE:
        returns_arith = returns_vol
    else:
        returns_arith = ret.to_returns(prices=sampled_prices_vol, return_type=ReturnTypes.RELATIVE, drop_first=True)
    avg_arith_return = returns_arith.mean()  # periodic mean of simple returns
    sharpe_arith = vol_dt * avg_arith_return.divide(returns_arith.std(ddof=1))
    if perf_params.rates_data is not None:
        # periodic excess simple returns r_m - rf_{m-1}*dt_m, lag=1 rate convention
        # consistent with the compounded excess-nav path in returns.py
        excess_returns_arith = ret.compute_excess_returns(returns=returns_arith, rates_data=perf_params.rates_data)
        avg_arith_excess_return = excess_returns_arith.mean()
        sharpe_arith_excess = vol_dt * avg_arith_excess_return.divide(excess_returns_arith.std(ddof=1))
    else:  # rf = 0: excess objects collapse to plain, mirroring the pa excess convention
        avg_arith_excess_return = avg_arith_return
        sharpe_arith_excess = sharpe_arith
    worst_series = pct_returns_dd.min()
    best_series = pct_returns_dd.max()

    # skew/kurtosis via scipy on dropna'd numpy arrays per column (scipy's pandas
    # methods don't expose bias=False consistently across versions).
    skew_series = returns_skew.apply(
        lambda s: skew(s.dropna().to_numpy(), bias=False) if s.dropna().size > 2 else np.nan
    )
    kurt_series = returns_skew.apply(
        lambda s: kurtosis(s.dropna().to_numpy(), bias=False) if s.dropna().size > 3 else np.nan
    )

    # downside vol — guarded against empty negative-return arrays
    downside_vol_series = returns_vol.apply(
        lambda s: _safe_downside_vol(s.dropna().to_numpy(), vol_dt)
    )

    # max / current drawdown using the dedicated helper (operates on full DataFrame)
    max_dds_arr, current_dds_arr = compute_max_current_drawdown(prices=dd_sampled_prices)
    # Wrap in Series for O(1) by-name lookup in the asset loop below.
    max_dds = pd.Series(max_dds_arr, index=dd_sampled_prices.columns)
    current_dds = pd.Series(current_dds_arr, index=dd_sampled_prices.columns)

    # ── Assemble per-asset rows ──
    # Per-asset metadata (start/end dates, obs count) still needs the dropna trick.
    dict_data = {}
    for asset in sampled_prices_vol:
        sampled_price = sampled_prices_vol[asset].dropna()
        n_obs = len(sampled_price.index) - 1 if len(sampled_price.index) > 1 else 0

        if n_obs > 0:
            vol = float(vol_series[asset])
            max_dd_val = float(max_dds[asset])
            asset_dict = {
                PerfStat.VOL.to_str(): vol,
                PerfStat.DOWNSIDE_VOL.to_str(): float(downside_vol_series[asset]),
                PerfStat.AVG_LOG_RETURN.to_str(): float(avg_log_return[asset]),
                PerfStat.AVG_ARITH_RETURN.to_str(): float(avg_arith_return[asset]),
                PerfStat.AN_ARITH_RETURN.to_str(): float(an_factor * avg_arith_return[asset]),
                PerfStat.AVG_ARITH_EXCESS_RETURN.to_str(): float(avg_arith_excess_return[asset]),
                PerfStat.AN_ARITH_EXCESS_RETURN.to_str(): float(an_factor * avg_arith_excess_return[asset]),
                PerfStat.SHARPE_ARITH.to_str(): float(sharpe_arith[asset]),
                PerfStat.SHARPE_ARITH_EXCESS.to_str(): float(sharpe_arith_excess[asset]),
                PerfStat.START_DATE.to_str(): sampled_price.index[0],
                PerfStat.END_DATE.to_str(): sampled_price.index[-1],
                PerfStat.NUM_OBS.to_str(): n_obs,
                PerfStat.MAX_DD.to_str(): max_dd_val,
                PerfStat.CURRENT_DD.to_str(): float(current_dds[asset]),
                PerfStat.MAX_DD_VOL.to_str(): max_dd_val / vol if vol > 0.0 else 0.0,
                PerfStat.WORST.to_str(): float(worst_series[asset]),
                PerfStat.BEST.to_str(): float(best_series[asset]),
                PerfStat.SKEWNESS.to_str(): float(skew_series[asset]) if pd.notna(skew_series[asset]) else np.nan,
                PerfStat.KURTOSIS.to_str(): float(kurt_series[asset]) if pd.notna(kurt_series[asset]) else np.nan,
            }
        else:
            # not enough data for any metric
            asset_dict = {PerfStat.VOL.to_str(): np.nan,
                          PerfStat.AVG_LOG_RETURN.to_str(): np.nan,
                          PerfStat.START_DATE.to_str(): np.nan,
                          PerfStat.END_DATE.to_str(): np.nan,
                          PerfStat.NUM_OBS.to_str(): 0
                          }
        dict_data[asset] = asset_dict

    data = pd.DataFrame.from_dict(data=dict_data, orient='index')
    return data


# =============================================================================
# RISK-ADJUSTED PERFORMANCE TABLE
# =============================================================================

def compute_ra_perf_table(prices: Union[pd.DataFrame, pd.Series],
                          perf_params: PerfParams = None
                          ) -> pd.DataFrame:
    """Compute the full risk-adjusted performance table.

    Combines outputs from ``compute_performance_table`` (return metrics) and
    ``compute_risk_table`` (volatility, drawdowns, higher moments) and derives
    the standard ratio family: Sharpe (rf=0, excess, log-annualised, log-excess,
    arithmetic, arithmetic-excess), Sortino, Calmar.

    Args:
        prices: DataFrame or Series of asset price levels.
        perf_params: Performance parameter object. If None, frequency is inferred.

    Returns:
        DataFrame indexed by asset with all performance and risk columns merged.
        Overlapping columns (e.g. START_DATE, END_DATE) are present once, taken
        from the performance table.
    """
    if perf_params is None:
        perf_params = PerfParams(freq=pd.infer_freq(prices.index))

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    perf_table = compute_performance_table(prices=prices, perf_params=perf_params)

    # is we only need sharpe we only comptute vol without higher order risk
    risk_table = compute_risk_table(prices=prices, perf_params=perf_params)

    # ── Derive ratio metrics from vol ──
    vol = risk_table[PerfStat.VOL.to_str()]
    perf_table[PerfStat.SHARPE_RF0.to_str()] = perf_table[PerfStat.PA_RETURN.to_str()] / vol
    perf_table[PerfStat.SHARPE_EXCESS.to_str()] = perf_table[PerfStat.PA_EXCESS_RETURN.to_str()] / vol
    perf_table[PerfStat.SHARPE_LOG_AN.to_str()] = perf_table[PerfStat.AN_LOG_RETURN.to_str()] / vol
    perf_table[PerfStat.SHARPE_LOG_EXCESS.to_str()] = perf_table[PerfStat.AN_LOG_EXCESS_RETURN.to_str()] / vol
    # SHARPE_ARITH / SHARPE_ARITH_EXCESS arrive via risk_table: they are computed in
    # compute_risk_table where numerator and denominator share the simple-return series

    if PerfStat.DOWNSIDE_VOL.to_str() in risk_table.columns:
        perf_table[PerfStat.SORTINO_RATIO.to_str()] = perf_table[PerfStat.PA_EXCESS_RETURN.to_str()] / risk_table[PerfStat.DOWNSIDE_VOL.to_str()]
    if PerfStat.MAX_DD.to_str() in risk_table.columns:
        perf_table[PerfStat.CALMAR_RATIO.to_str()] = -1.0*perf_table[PerfStat.PA_EXCESS_RETURN.to_str()] / risk_table[PerfStat.MAX_DD.to_str()]

    # ── Merge perf and risk tables, dropping duplicates ──
    # Both tables produce START_DATE / END_DATE; we keep the perf_table version
    # (added first) and drop the duplicates from risk_table to avoid the historical
    # "_y" suffixed columns that polluted downstream output.
    overlap = risk_table.columns.intersection(perf_table.columns)
    risk_table_clean = risk_table.drop(columns=overlap)
    ra_perf_table = pd.concat([perf_table, risk_table_clean], axis=1)
    return ra_perf_table


def compute_ra_perf_table_with_benchmark(prices: pd.DataFrame,
                                         benchmark: str = None,
                                         benchmark_price: pd.Series = None,
                                         perf_params: PerfParams = None,
                                         is_log_returns: bool = False,
                                         drop_benchmark: bool = False,
                                         **kwargs
                                         ) -> pd.DataFrame:
    """Compute the risk-adjusted performance table augmented with alpha/beta to a benchmark.

    Extends ``compute_ra_perf_table`` by regressing each asset's returns on the benchmark
    and appending alpha (raw and annualised), beta, R² and the p-value of alpha.

    Args:
        prices: DataFrame of asset price levels.
        benchmark: Column name of the benchmark in ``prices``. Optional if
            ``benchmark_price`` is supplied.
        benchmark_price: Stand-alone benchmark price Series. Optional if ``benchmark``
            is in ``prices``. See ``_resolve_benchmark`` for the three-way branching.
        perf_params: Performance parameter object. If None, frequency is inferred.
        is_log_returns: If True, compute log returns instead of arithmetic returns
            for the regression.
        drop_benchmark: If True, exclude the benchmark row from the output table.
        **kwargs: Reserved for forward compatibility; currently unused.

    Returns:
        DataFrame indexed by asset with all base RA metrics plus ALPHA, ALPHA_AN,
        BETA, R2 and ALPHA_PVALUE columns.
    """
    # Resolve the three input modes (name only / price only / both) into a
    # consistent (prices_with_benchmark, benchmark_name) pair.
    prices, benchmark = _resolve_benchmark(prices=prices,
                                           benchmark=benchmark,
                                           benchmark_price=benchmark_price)

    if perf_params is None:
        perf_params = PerfParams(freq=pd.infer_freq(prices.index))

    ra_perf_table = compute_ra_perf_table(prices=prices, perf_params=perf_params)

    # ── Run benchmark regression at the configured regression frequency ──
    returns = ret.to_returns(prices=prices, freq=perf_params.freq_reg, is_log_returns=is_log_returns)

    # use excess returns if rates data is given
    if perf_params.rates_data is not None:
        returns = ret.compute_excess_returns(returns=returns, rates_data=perf_params.rates_data)

    # Estimate OLS alpha/beta per asset against the benchmark column
    alphas, betas, r2, alpha_pvalue = {}, {}, {}, {}
    for column in returns.columns:
        joint_data = returns[[benchmark, column]].dropna()
        if joint_data.empty or len(joint_data.index) < 2:
            alphas[column], betas[column], r2[column], alpha_pvalue[column] = np.nan, np.nan, np.nan, np.nan
        else:
            alphas[column], betas[column], r2[column], alpha_pvalue[column] = ols.estimate_ols_alpha_beta(x=joint_data.iloc[:, 0],
                                                                                                          y=joint_data.iloc[:, 1])

    alpha_an_factor = get_annualization_factor(freq=perf_params.freq_reg)
    ra_perf_table[PerfStat.ALPHA.to_str()] = pd.Series(alphas)
    ra_perf_table[PerfStat.ALPHA_AN.to_str()] = alpha_an_factor * pd.Series(alphas)
    ra_perf_table[PerfStat.BETA.to_str()] = pd.Series(betas)
    ra_perf_table[PerfStat.R2.to_str()] = pd.Series(r2)
    ra_perf_table[PerfStat.ALPHA_PVALUE.to_str()] = pd.Series(alpha_pvalue)

    if drop_benchmark:
        ra_perf_table = ra_perf_table.drop([benchmark], axis=0)
    else:  # set p-value of benchmark alpha to 1.0 (regression of benchmark on itself)
        ra_perf_table.loc[benchmark, PerfStat.ALPHA_PVALUE.to_str()] = 1.0
    return ra_perf_table


# =============================================================================
# DESCRIPTIVE FREQUENCY TABLE
# =============================================================================

def compute_desc_freq_table(df: pd.DataFrame,
                            freq: str = 'YE',
                            agg_func: Callable = np.sum
                            ) -> pd.DataFrame:
    """Aggregate a time series at a given frequency and compute descriptive statistics.

    Useful for producing tables of period-aggregate statistics — e.g. yearly returns
    summaries with mean, stdev, ±1σ quantiles and median.

    Args:
        df: Time series DataFrame to aggregate.
        freq: Pandas resample frequency (default 'YE' = year-end).
        agg_func: Aggregation function applied to each period (default sum).

    Returns:
        DataFrame indexed by original column with descriptive statistic columns:
        AVG, STD, QUANT_M_1STD (16th percentile), MEDIAN, QUANT_P1_STD (84th percentile).
    """
    freq_data = df.resample(freq).agg(agg_func)

    # drop na rows for all
    freq_data = freq_data.dropna(axis=0, how='any')

    data_values = freq_data.to_numpy()
    data_table = pd.DataFrame(index=freq_data.columns)
    data_table[PerfStat.AVG.to_str()] = np.nanmean(data_values, axis=0)
    data_table[PerfStat.STD.to_str()] = np.nanstd(data_values, ddof=1, axis=0)
    data_table[PerfStat.QUANT_M_1STD.to_str()] = np.nanquantile(data_values, q=0.16, axis=0)
    # Bug fix: previously used np.mean which gave the mean, not the median.
    data_table[PerfStat.MEDIAN.to_str()] = np.nanmedian(data_values, axis=0)
    data_table[PerfStat.QUANT_P1_STD.to_str()] = np.nanquantile(data_values, q=0.84, axis=0)

    return data_table


# =============================================================================
# TRACKING ERROR / INFORMATION RATIO
# =============================================================================

def compute_te_ir_errors(return_diffs: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Compute tracking error and information ratio from return differentials.

    Args:
        return_diffs: DataFrame of (strategy_return - benchmark_return) per period.

    Returns:
        Tuple of (tracking_error_series, information_ratio_series), both indexed by
        the columns of return_diffs and annualised.
    """
    vol_dt = np.sqrt(infer_annualisation_factor_from_df(return_diffs))
    avg = np.nanmean(return_diffs, axis=0)
    vol = np.nanstd(return_diffs, axis=0, ddof=1)
    # NumPy 2.x: explicit out= buffer so masked positions (vol==0) are deterministic nan.
    ir = vol_dt * np.divide(
        avg, vol,
        out=np.full_like(avg, np.nan, dtype=float),
        where=np.greater(vol, 0.0),
    )
    te = pd.Series(vol_dt * vol, index=return_diffs.columns, name=PerfStat.TE.to_str())
    ir = pd.Series(ir, index=return_diffs.columns, name=PerfStat.IR.to_str())
    return te, ir


def compute_info_ratio_table(return_diffs_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute TE and IR tables across multiple asset-class return-diff DataFrames.

    Args:
        return_diffs_dict: Mapping of asset class label → return-diffs DataFrame.

    Returns:
        Tuple of (te_table, ir_table) DataFrames, columns = asset classes,
        rows = strategies.
    """
    te_ac_datas = []
    ir_ac_datas = []
    for ac, data in return_diffs_dict.items():
        te, ir = compute_te_ir_errors(return_diffs=data)
        te_ac_datas.append(te.rename(ac))
        ir_ac_datas.append(ir.rename(ac))
    te_table = pd.concat(te_ac_datas, axis=1)
    ir_table = pd.concat(ir_ac_datas, axis=1)
    return te_table, ir_table


# =============================================================================
# DRAWDOWN COMPUTATIONS
# =============================================================================

def compute_rolling_drawdowns(prices: Union[pd.DataFrame, pd.Series],
                              min_periods: int = 1
                              ) -> Union[pd.DataFrame, pd.Series]:
    """Compute the running drawdown series from peak.

    Drawdown at time t is defined as price_t / max(price_0..t) - 1, always ≤ 0.

    Args:
        prices: Price level Series or DataFrame.
        min_periods: Minimum observations required by the expanding peak. The default
            of 1 means the peak starts from the first valid price; for daily series
            with leading NaNs after reindexing, this means the first valid value
            becomes the initial peak immediately. Increase this if you want the
            drawdown to remain NaN until a longer warm-up window has elapsed.

    Returns:
        Drawdown Series or DataFrame matching the input shape, with values in (-1, 0].

    Raises:
        ValueError: If ``prices`` is neither Series nor DataFrame.
    """
    if not isinstance(prices, pd.Series) and not isinstance(prices, pd.DataFrame):
        raise ValueError(f"unsuported type {type(prices)}")
    peak = prices.expanding(min_periods=min_periods).max()
    drawdown = (prices.divide(peak)-1.0).ffill()  # ffill nans
    return drawdown


def compute_rolling_drawdown_time_under_water(prices: Union[pd.DataFrame, pd.Series],
                                              sampling_freq: Literal['B', 'D'] = 'D'
                                              ) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
    """Compute joint drawdown and time-under-water series.

    Time under water counts consecutive periods spent below the prior peak,
    resetting to zero each time a new high is established.

    Args:
        prices: Price level Series or DataFrame.
        sampling_freq: 'D' for calendar days, 'B' for business days. Calendar days
            are recommended for most reporting use cases as they reflect investor
            wall-clock recovery time.

    Returns:
        Tuple of (drawdown, time_under_water), both matching the input shape.

    Raises:
        ValueError: If ``prices`` is neither Series nor DataFrame.
    """
    if not isinstance(prices, pd.Series) and not isinstance(prices, pd.DataFrame):
        raise ValueError(f"unsuported type {type(prices)}")
    prices = prices.asfreq(freq=sampling_freq, method='ffill').ffill()
    # find expanding peak
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices.divide(peak)-1.0).ffill()  # ffill nans
    if isinstance(prices, pd.DataFrame):
        is_in_dd = pd.DataFrame(np.where(prices < peak, 1.0, np.nan), index=prices.index, columns=prices.columns)
    else:
        is_in_dd = pd.Series(np.where(prices < peak, 1.0, np.nan), index=prices.index, name=prices.name)

    # cumsum until first nan — groups by NaN-separated runs.
    # The apply(axis=0) pattern only works on DataFrames; for Series we apply
    # the groupby directly.
    if isinstance(is_in_dd, pd.DataFrame):
        time_under_water = is_in_dd.apply(lambda x: x.groupby(x.isna().cumsum()).cumsum(), axis=0)
    else:
        time_under_water = is_in_dd.groupby(is_in_dd.isna().cumsum()).cumsum()
    time_under_water = time_under_water.fillna(0.0)
    return drawdown, time_under_water


def compute_max_current_drawdown(prices: Union[pd.DataFrame, pd.Series]
                                 ) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """Compute the realised maximum drawdown and the current drawdown.

    Args:
        prices: Price level Series or DataFrame.

    Returns:
        Tuple of (max_drawdowns, current_drawdowns). For a DataFrame input, both
        elements are 1-D ndarrays indexed by column. For a Series input, both
        elements are scalar floats.
    """
    max_dd_data = compute_rolling_drawdowns(prices=prices)
    if isinstance(prices, pd.DataFrame):
        max_dds = np.nanmin(max_dd_data.to_numpy(), axis=0)
        current_dds = max_dd_data.iloc[-1, :].to_numpy()
    else:
        # Series case: return scalars (not arrays) to match the actual return shape.
        max_dds = float(np.nanmin(max_dd_data.to_numpy()))
        current_dds = float(max_dd_data.iloc[-1])
    return max_dds, current_dds


def compute_avg_max_dd(ds: pd.Series,
                       is_max: bool = True,
                       q: float = 0.1
                       ) -> (float, float, float, float):
    """Compute summary statistics of a drawdown (or run-up) series.

    Args:
        ds: Drawdown or running-extreme Series.
        is_max: If True, consider only positive values (run-ups); if False, only
            negative values (drawdowns).
        q: Quantile for the tail statistic (default 0.1 = 10th/90th percentile).

    Returns:
        Tuple of (avg, quantile, extreme, last). For ``is_max=True`` quantile is the
        upper (1-q) quantile and extreme is the max; for ``is_max=False`` quantile
        is the lower q quantile and extreme is the min.
    """
    if is_max:
        nan_data = np.where(ds.to_numpy() >= 0, ds.to_numpy(), np.nan)
    else:
        nan_data = np.where(ds.to_numpy() <= 0, ds.to_numpy(), np.nan)

    avg = np.nanmean(nan_data)
    if is_max:
        quant = np.nanquantile(nan_data, 1.0 - q)
        nmax = np.nanmax(nan_data)
    else:
        quant = np.nanquantile(nan_data, q)
        nmax = np.nanmin(nan_data)

    last = ds.iloc[-1]

    return avg, quant, nmax, last


def compute_drawdowns_stats_table(price: pd.Series,
                                  max_num: Optional[int] = None,
                                  freq: Optional[str] = 'D'  # need to rebase to calendar days
                                  ) -> pd.DataFrame:
    """Compute a sorted table of drawdown episodes and their statistics.

    Splits the running drawdown series into NaN-separated blocks (each block is one
    drawdown episode from peak to recovery), computes summary statistics per block,
    sorts by max drawdown depth and optionally truncates to the worst N.

    Block detection uses a pandas-version-stable groupby on a cumulative-sum index
    rather than relying on SparseArray internals (which have shifted across pandas
    releases).

    Args:
        price: Price level Series.
        max_num: Maximum number of drawdown episodes to return (worst N).
        freq: Frequency to rebase to before block detection. Default 'D' (calendar
            days) gives recovery times in wall-clock days. Pass None to skip rebasing.

    Returns:
        DataFrame sorted by max_dd ascending, with columns:
        start, trough, end, max_dd, days_dd, days_to_trough, days_recovery,
        peak, bottom, recovery, is_recovered.
    """
    if freq is not None:
        price = price.asfreq(freq, method='ffill')
    max_dd, time_under_water = compute_rolling_drawdown_time_under_water(prices=price)
    # Replace zeros with NaN so that NaN-separated blocks delimit drawdown episodes.
    max_dd = max_dd.replace({0.0: np.nan})
    time_under_water = time_under_water.replace({0.0: np.nan})

    # Pack the three series side-by-side for per-block slicing.
    joint = pd.concat([max_dd.rename('max_dd'), time_under_water.rename('days'), price], axis=1)

    def process_bslice(bslice: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for a single drawdown episode slice."""
        max_idx = np.argmin(bslice['max_dd'].to_numpy())
        # An episode is recovered if the price at the end of the block has returned
        # to or above the peak at the start of the block.
        is_recovered = bool(bslice[price.name].iloc[-1] >= bslice[price.name].iloc[0])
        out = dict(start=bslice.index[0],
                   trough=bslice.index[max_idx],
                   end=bslice.index[-1],
                   max_dd=bslice['max_dd'].iloc[max_idx],
                   days_dd=bslice['days'].iloc[-1],
                   days_to_trough=bslice['days'].iloc[max_idx],
                   days_recovery=bslice['days'].iloc[-1]-bslice['days'].iloc[max_idx],
                   peak=bslice[price.name].iloc[0],
                   bottom=bslice[price.name].iloc[max_idx],
                   recovery=bslice[price.name].iloc[-1],
                   is_recovered=is_recovered)
        return out

    # ── Pandas-version-stable block detection ──
    # Mark observations that are "in drawdown" (max_dd is not NaN). Each NaN
    # increments a counter; consecutive non-NaN values share the same counter
    # value, giving us a natural group key per drawdown episode.
    is_in_dd = max_dd.notna()
    # cumsum on the negated mask: each True in (~is_in_dd) bumps the counter,
    # so all observations within a single drawdown share the same group id.
    block_id = (~is_in_dd).cumsum()
    # Group only the in-drawdown rows; each group is one episode.
    outputs = {}
    for bid, bslice in joint[is_in_dd].groupby(block_id[is_in_dd]):
        if not bslice.empty:
            outputs[bid] = process_bslice(bslice=bslice)

    df = pd.DataFrame.from_dict(outputs, orient='index')
    if df.empty:
        # No drawdown episodes — return empty table with expected columns
        return pd.DataFrame(columns=['start', 'trough', 'end', 'max_dd', 'days_dd',
                                     'days_to_trough', 'days_recovery', 'peak',
                                     'bottom', 'recovery', 'is_recovered'])

    df = df.sort_values(by='max_dd')
    if max_num is not None:
        df = df.iloc[:max_num, :]
    return df