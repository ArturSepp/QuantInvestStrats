"""
the return conventions of the stack: prices to returns, returns to nav, and back.

Everything downstream - performance statistics, factsheets, backtest attribution - reads its
numbers through this module, so a convention set here is a convention everywhere.

The conversion is stated, never implied. ``to_returns`` takes ``is_log_returns`` or the fuller
``return_type``, and the two are not a rescaling of each other:

    RELATIVE   r = S1 / S0 - 1     additive across assets within a period
    LOG        r = ln(S1 / S0)     additive across periods for one asset

``returns_to_nav`` and ``log_returns_to_nav`` invert them on the matching convention; passing
log returns to the arithmetic inverse compounds the wrong quantity and raises nothing. Prices
are resampled by ``prices_at_freq`` before differencing, so ``freq`` selects the return
frequency rather than resampling returns after the fact. Ratio and log returns require finite
positive levels at both endpoints; difference and level modes also accept finite signed values.

Annualisation is geometric and driven by elapsed calendar time, not by observation count:
``compute_pa_return`` raises the total return to 1/num_years, where num_years counts calendar
days over 365.25. Samples shorter than a year are returned untouched unless
``annualize_less_1y`` asks for linear scaling, which is the safer default for a young track
record. Volatility is annualised by the periods-per-year implied by the index frequency,
inferred in ``qis/utils/annualisation.py`` rather than assumed here.

Main entry points: ``to_returns``, ``returns_to_nav``, ``compute_total_return``,
``compute_pa_return``, ``compute_sampled_vols``, and ``compute_excess_returns``, which subtracts
an annualised ``rates_data`` series lagged one period, so the funding cost charged at t is the
rate observable at t-1. Leverage and fee arithmetic - ``lever_returns``, ``delever_returns``,
``compute_net_navs_ex_perf_man_fees`` - is here because it is a return transform; ratio
statistics such as Sharpe are assembled in ``qis/perfstats/perf_stats.py``.
"""
# packages
import warnings
from math import isfinite
from numbers import Integral, Real
import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, cast

# qis
import qis.utils.dates as da
import qis.utils.df_freq as dff
import qis.utils.np_ops as npo
import qis.utils.df_ops as dfo
from qis.utils.df_ops import df_price_ffill_between_nans
from qis.utils.df_groups import get_group_dict
from qis.perfstats.config import PerfStat, ReturnTypes, PerfParams
from qis.utils.annualisation import infer_annualisation_factor_from_df, CALENDAR_DAYS_PER_YEAR_SHARPE


def compute_num_days(prices: Union[pd.DataFrame, pd.Series]) -> int:
    """Compute number of calendar days in price series.

    Args:
        prices: Price time series

    Returns:
        Number of days between first and last price

    Raises:
        ValueError: If dates are in reverse chronological order
    """
    if prices.index[0] > prices.index[-1]:
        raise ValueError(f"inconsistent dates t0={prices.index[0]} t1={prices.index[-1]}")
    return np.maximum((prices.index[-1] - prices.index[0]).days, 1)


def compute_num_years(prices: Union[pd.DataFrame, pd.Series],
                      days_per_year: float = CALENDAR_DAYS_PER_YEAR_SHARPE
                      ) -> float:
    """Compute number of years in price series.

    Args:
        prices: Price time series
        days_per_year: Calendar days per year (default: 365.25)

    Returns:
        Number of years as float
    """
    return compute_num_days(prices=prices) / days_per_year


def to_returns(prices: Union[pd.Series, pd.DataFrame],
               is_log_returns: bool = False,
               return_type: ReturnTypes = ReturnTypes.RELATIVE,
               freq: Optional[str] = None,
               include_start_date: bool = False,
               include_end_date: bool = False,
               ffill_nans: bool = True,
               drop_first: bool = False,
               is_first_zero: bool = False,
               **kwargs
               ) -> Union[pd.Series, pd.DataFrame]:
    """Convert prices to returns with specified methodology.

    Args:
        prices: Numeric price time series. Pandas nullable floating values are supported
        is_log_returns: If True, compute log returns (overrides return_type)
        return_type: Type of return calculation. LOG and RELATIVE require finite positive
            endpoints; DIFFERENCE, LEVEL, and LEVEL0 accept finite signed levels
        freq: Resampling frequency (e.g., 'D', 'W', 'M')
        include_start_date: Include period start date when resampling
        include_end_date: Include period end date when resampling
        ffill_nans: Forward-fill NaN prices before computing returns
        drop_first: Drop first return observation
        is_first_zero: Set first non-NaN return to zero
        **kwargs: Additional arguments

    Returns:
        Return time series matching the input's pandas shape and labels. LOG and RELATIVE are
        missing unless both endpoints are finite and positive. DIFFERENCE is missing unless both
        signed endpoints are finite; LEVEL and LEVEL0 are missing unless their returned level is
        finite. Nullable floating inputs retain a nullable output dtype
    """
    # Resample prices to specified frequency
    prices = prices_at_freq(prices=prices, freq=freq,
                            include_start_date=include_start_date,
                            include_end_date=include_end_date,
                            ffill_nans=ffill_nans)

    # Validate the endpoints consumed by each convention, and neutralize invalid inputs before
    # arithmetic so masked observations cannot leak divide or log warnings.
    finite_prices: Union[pd.Series, pd.DataFrame] = prices.notna() & prices.abs().lt(np.inf)
    if return_type == ReturnTypes.LOG or is_log_returns:
        # Log returns require positive levels at both endpoints of the period.
        valid_prices: Union[pd.Series, pd.DataFrame] = finite_prices & prices.gt(0.0)
        calculation_prices: Union[pd.Series, pd.DataFrame] = prices.where(valid_prices, other=1.0)
        log_prices = cast(Union[pd.Series, pd.DataFrame], np.log(calculation_prices))
        returns = log_prices.diff(1)
        valid_returns = valid_prices & valid_prices.shift(1, fill_value=False)
    elif return_type == ReturnTypes.RELATIVE:
        # Ratio returns share the log convention's positive two-endpoint domain.
        valid_prices = finite_prices & prices.gt(0.0)
        calculation_prices = prices.where(valid_prices, other=1.0)
        returns = calculation_prices.div(calculation_prices.shift(1)).sub(1.0)
        valid_returns = valid_prices & valid_prices.shift(1, fill_value=False)
    elif return_type == ReturnTypes.DIFFERENCE:
        # Signed rates and spreads may cross zero, so differences require only finite endpoints.
        calculation_prices = prices.where(finite_prices, other=0.0)
        returns = calculation_prices - calculation_prices.shift(1)
        valid_returns = finite_prices & finite_prices.shift(1, fill_value=False)
    elif return_type == ReturnTypes.LEVEL:
        # LEVEL reports the current observation and therefore validates only that returned value.
        returns = prices
        valid_returns = finite_prices
    elif return_type == ReturnTypes.LEVEL0:
        # LEVEL0 reports the predecessor, so its validity mask shifts with the returned value.
        returns = prices.shift(1)
        valid_returns = finite_prices.shift(1, fill_value=False)
    else:
        raise NotImplementedError(f"{return_type}")

    # Remove neutral calculation values after applying the selected convention's endpoint rule.
    returns = returns.where(valid_returns, other=np.nan)

    # Handle first observation
    if is_first_zero:
        returns = to_zero_first_nonnan_returns(returns=returns, init_period=0)
    elif drop_first:
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[1:, :]
        else:
            returns = returns.iloc[1:]

    return returns


def compute_asset_returns_dict(prices: pd.DataFrame,
                               returns_freqs: Union[str, pd.Series],
                               drop_first: bool = False,
                               is_first_zero: bool = True,
                               is_log_returns: bool = False
                               ) -> Dict[str, pd.DataFrame]:
    """Compute returns for assets grouped by frequency.

    Args:
        prices: Price DataFrame with asset columns
        returns_freqs: Series mapping asset tickers to return frequencies
        drop_first: Drop first return observation
        is_first_zero: Set first non-NaN return to zero
        is_log_returns: Use log returns

    Returns:
        Dictionary with frequency keys and return DataFrames as values
    """
    if isinstance(returns_freqs, str):
        group_freqs = {returns_freqs: prices.columns.to_list()}
    elif isinstance(returns_freqs, pd.Series):
        group_freqs = get_group_dict(group_data=returns_freqs)
    else:
        raise NotImplementedError(f"returns_freqs={returns_freqs} with type {type(returns_freqs)}")

    asset_returns_dict = {}
    for freq, asset_tickers in group_freqs.items():
        asset_returns_dict[freq] = to_returns(prices=prices[asset_tickers],
                                              is_log_returns=is_log_returns,
                                              is_first_zero=is_first_zero,
                                              drop_first=drop_first,
                                              freq=freq)
    return asset_returns_dict


def to_total_returns(prices: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """Compute total returns over entire price history.

    Args:
        prices: Price time series

    Returns:
        Series of total returns indexed by asset
    """
    total_returns = compute_total_return(prices=prices)
    if isinstance(prices, pd.DataFrame):
        total_returns = pd.Series(total_returns, index=prices.columns)
    elif isinstance(prices, pd.Series):
        total_returns = pd.Series(total_returns, name=prices.name)
    else:
        raise NotImplementedError(f"{type(prices)}")

    return total_returns


def compute_total_return(prices: Union[pd.DataFrame, pd.Series]) -> Union[np.ndarray, float]:
    """Compute total return from first to last price.

    Args:
        prices: Price time series

    Returns:
        Array of total returns for DataFrame, float for Series
    """
    if len(prices.index) == 1:
        if isinstance(prices, pd.DataFrame):
            return np.full(len(prices.columns), fill_value=np.nan)
        elif isinstance(prices, pd.Series):
            return np.nan
        else:
            raise TypeError(f"unsuported type={type(prices)}")

    # Get initial prices, handling NaN at start
    if isinstance(prices, pd.DataFrame):
        price_0 = prices.iloc[0, :].to_numpy()
        if np.any(np.isnan(price_0)):
            price_0 = dfo.get_first_nonnan_values(df=prices)
            warnings.warn(f"detected nan price for prices = {prices.iloc[0, np.isnan(price_0)]},"
                          f" using first non nan price = {price_0} for {prices.columns}")
        # Symmetric: handle NaN at the END too. Common case: a fund that
        # terminated mid-dataset, or an ETF that delisted, leaving trailing
        # NaN. Without this fix, total return is NaN even though the data
        # to compute it is right there in the series.
        price_end = prices.iloc[-1, :].to_numpy()
        if np.any(np.isnan(price_end)):
            price_end = dfo.get_last_nonnan_values(df=prices)
            warnings.warn(f"detected nan price for prices = {prices.iloc[-1, np.isnan(price_end)]},"
                          f" using last non nan price = {price_end} for {prices.columns}")

    elif isinstance(prices, pd.Series):
        price_0 = prices.iloc[0]
        if np.isnan(price_0):
            price_0 = dfo.get_first_nonnan_values(df=prices)
            warnings.warn(f"detected nan price for prices at date = {prices.index[0]},"
                          f" using first non nan price = {price_0} for {prices.name}")
        # Same symmetric fix for the Series branch.
        price_end = prices.iloc[-1]
        if np.isnan(price_end):
            price_end = dfo.get_last_nonnan_values(df=prices)
            warnings.warn(f"detected nan price for prices at date = {prices.index[-1]},"
                          f" using last non nan price = {price_end} for {prices.name}")
    else:
        raise TypeError(f"unsuported type={type(prices)}")

    num_years = compute_num_years(prices=prices)
    if num_years > 0.0:
        total_return = price_end / price_0 - 1.0
    else:
        warnings.warn(f"total return has inconsistent dates t0={prices.index[0]} and t1={prices.index[-1]}")
        total_return = np.full(len(prices.columns), fill_value=np.nan) if isinstance(prices, pd.DataFrame) else np.nan
    return total_return


def compute_pa_return(prices: Union[pd.DataFrame, pd.Series],
                      annualize_less_1y: bool = False
                      ) -> Union[np.ndarray, float]:
    """Compute annualized (per annum) return with geometric compounding.

    Args:
        prices: Price time series
        annualize_less_1y: If True, annualize periods <1 year linearly;
                          if False, return total return without annualization

    Returns:
        Array of annualized returns for DataFrame, float for Series
    """
    total_return = compute_total_return(prices=prices)
    num_years = compute_num_years(prices=prices)

    if num_years > 0.0:
        ratio = total_return + 1.0
        ratio = np.where(np.greater(ratio, 0.0), ratio, np.nan)
        if num_years > 1.0:
            # Geometric compounding for periods >1 year
            # NumPy 2.x: explicit out= so non-finite positions are deterministic nan.
            compounded_return_pa = np.power(
                ratio, 1.0 / num_years,
                out=np.full_like(ratio, np.nan, dtype=float),
                where=np.isfinite(ratio),
            ) - 1
        else:
            if annualize_less_1y:
                # Linear annualization for periods <1 year
                compounded_return_pa = total_return / num_years
            else:
                # Use total return without annualization
                compounded_return_pa = ratio - 1.0
    else:
        n = len(prices.columns) if isinstance(prices, pd.DataFrame) else 1
        # Note: np.zeros_like(n) where n is an int returns a 0-d scalar 0,
        # not a vector of zeros — that was the previous bug. Use np.zeros(n)
        # so DataFrame callers get a column-aligned vector back.
        compounded_return_pa = np.zeros(n)

    return compounded_return_pa


def compute_returns_dict(prices: Union[pd.DataFrame, pd.Series],
                         perf_params: PerfParams = None,
                         annualize_less_1y: bool = False
                         ) -> Dict[str, np.ndarray]:
    """Compute comprehensive dictionary of return metrics.

    Args:
        prices: Price time series
        perf_params: Performance parameters including risk-free rates
        annualize_less_1y: Annualize returns for periods <1 year

    Returns:
        Dictionary with keys for total return, PA return, excess returns,
        log returns, dates, and price levels
    """
    if not isinstance(prices, pd.Series) and not isinstance(prices, pd.DataFrame):
        raise ValueError(f"not supperted type={type(prices)}")

    # Handle empty or all-NaN prices
    if prices.empty:
        warnings.warn(f"in compute_returns_dict(): {prices} is all nans", stacklevel=2)
        if isinstance(prices, pd.Series):
            n = 1
        else:
            n = len(prices.columns)
        if n == 1:
            return_dict = {PerfStat.TOTAL_RETURN.to_str(): np.nan,
                           PerfStat.PA_RETURN.to_str(): np.nan,
                           PerfStat.AN_LOG_RETURN.to_str(): np.nan,
                           PerfStat.AN_LOG_EXCESS_RETURN.to_str(): np.nan,
                           PerfStat.NUM_YEARS.to_str(): np.nan}
        else:
            return_dict = {PerfStat.TOTAL_RETURN.to_str(): np.full(n, fill_value=np.nan),
                           PerfStat.PA_RETURN.to_str(): np.full(n, fill_value=np.nan),
                           PerfStat.AN_LOG_RETURN.to_str(): np.full(n, fill_value=np.nan),
                           PerfStat.AN_LOG_EXCESS_RETURN.to_str(): np.full(n, fill_value=np.nan),
                           PerfStat.NUM_YEARS.to_str(): np.full(n, fill_value=np.nan)}
        return return_dict

    if perf_params is None:
        perf_params = PerfParams()

    # Compute core return metrics
    compounded_return_pa = compute_pa_return(prices=prices, annualize_less_1y=annualize_less_1y)
    total_return = compute_total_return(prices=prices)
    num_years = compute_num_years(prices=prices)
    num_days = compute_num_days(prices=prices)

    # Compute excess returns if risk-free rates provided
    if perf_params.rates_data is not None:
        excess_return_pa = compute_pa_excess_compounded_returns(returns=to_returns(prices,
                                                                        return_type=ReturnTypes.RELATIVE,
                                                                        is_first_zero=True),
                                                     rates_data=perf_params.rates_data,
                                                     first_date=prices.index[0],
                                                     annualize_less_1y=annualize_less_1y)
    else:
        excess_return_pa = compounded_return_pa

    # Extract start and end values

    if isinstance(prices, pd.DataFrame):
        start_value = prices.iloc[0, :].to_numpy()
        end_value = prices.iloc[-1, :].to_numpy()
    else:
        start_value = prices.iloc[0]
        end_value = prices.iloc[-1]

    # NumPy 2.x: use helper lambdas with explicit out= to avoid uninitialized memory on masked positions.
    def _safe_log1p(x):
        x_arr = np.asarray(x, dtype=float)
        return np.log(
            1.0 + x_arr,
            out=np.full_like(x_arr, np.nan, dtype=float),
            where=np.greater(x_arr, -1.0),
        )

    # Build return dictionary
    return_dict = {PerfStat.TOTAL_RETURN.to_str(): total_return,
                   PerfStat.PA_RETURN.to_str(): compounded_return_pa,
                   PerfStat.PA_EXCESS_RETURN.to_str(): excess_return_pa,
                   PerfStat.AN_LOG_RETURN.to_str(): _safe_log1p(compounded_return_pa),
                   PerfStat.AN_LOG_EXCESS_RETURN.to_str(): _safe_log1p(excess_return_pa),
                   PerfStat.NAV1.to_str(): (1.0+total_return),
                   PerfStat.NUM_YEARS.to_str(): num_years,
                   PerfStat.START_DATE.to_str(): prices.index[0],
                   PerfStat.END_DATE.to_str(): prices.index[-1],
                   PerfStat.START_PRICE.to_str(): start_value,
                   PerfStat.END_PRICE.to_str(): end_value}

    return return_dict


def compute_excess_return_navs(prices: Union[pd.Series, pd.DataFrame],
                               rates_data: pd.Series,
                               first_date: pd.Timestamp = None
                               ) -> Union[pd.Series, pd.DataFrame]:
    """Compute NAVs for excess returns over risk-free rate.

    Args:
        prices: Price time series
        rates_data: Risk-free rate time series (annualized)
        first_date: Start date for NAV (default: first price date)

    Returns:
        Excess return NAV time series
    """
    returns = to_returns(prices=prices, is_first_zero=True)
    excess_returns = compute_excess_returns(returns=returns, rates_data=rates_data)
    navs = returns_to_nav(returns=excess_returns, first_date=first_date)
    return navs


def compute_excess_returns(returns: Union[pd.Series, pd.DataFrame],
                           rates_data: pd.Series
                           ) -> Union[pd.Series, pd.DataFrame]:
    """Subtract risk-free rate from returns.

    Args:
        returns: Return time series
        rates_data: Risk-free rate time series (annualized)

    Returns:
        Excess return time series (returns minus risk-free rate)
    """
    # Use lag=1 on the rate series: funding cost at time t reflects the
    # rate that was set at t-1 (the rate the manager could observe and
    # plan around). Previously this used lag=None (contemporaneous rate),
    # which introduces a small look-ahead bias relative to
    # get_excess_returns_nav() that uses lag=1. The two functions now
    # agree on convention.
    rates_dt = dfo.multiply_df_by_dt(df=rates_data, dates=returns.index, lag=1)
    returns0 = returns.copy()
    if isinstance(returns, pd.Series):
        returns0 = returns0.to_frame(name=returns.name)
    excess_returns = returns0.subtract(rates_dt.to_numpy(), axis=0)
    if isinstance(returns, pd.Series):
        excess_returns = excess_returns.iloc[:, 0]
    return excess_returns


def compute_pa_excess_compounded_returns(returns: Union[pd.Series, pd.DataFrame],
                                         rates_data: pd.Series,
                                         first_date: pd.Timestamp = None,
                                         annualize_less_1y: bool = False
                                         ) -> Union[np.ndarray, float]:
    """Compute annualized excess returns with geometric compounding.

    Args:
        returns: Return time series
        rates_data: Risk-free rate time series (annualized)
        first_date: Start date for NAV calculation
        annualize_less_1y: Annualize periods <1 year

    Returns:
        Array of annualized excess returns for DataFrame input, float for Series input
    """
    excess_returns = compute_excess_returns(returns=returns, rates_data=rates_data)
    prices = returns_to_nav(returns=excess_returns, first_date=first_date)
    compounded_return_pa = compute_pa_return(prices=prices, annualize_less_1y=annualize_less_1y)
    return compounded_return_pa


def estimate_vol(sampled_returns: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    """Estimate volatility from return samples.

    For columns with 20 or more finite observations, uses standard deviation with ddof=1.
    For smaller finite samples, uses RMS to avoid mean adjustment bias.

    Args:
        sampled_returns: Return time series. NumPy arrays must have shape `(observations,)` or
            `(observations, columns)`. Each column selects its estimator from its own finite
            observation count; missing rows do not count toward the threshold

    Returns:
        Volatility estimate for a one-dimensional input or one estimate per two-dimensional input
        column. A column is missing when it cannot supply the observations required by its
        selected estimator

    Raises:
        ValueError: If a NumPy array is not one- or two-dimensional
    """
    if isinstance(sampled_returns, np.ndarray) and sampled_returns.ndim not in (1, 2):
        # Higher ranks have no unambiguous observation/column mapping for this estimator.
        raise ValueError("sampled_returns must be a 1- or 2-dimensional NumPy array")

    if isinstance(sampled_returns, (pd.Series, pd.DataFrame)):
        sampled_values = sampled_returns.to_numpy(dtype=float, na_value=np.nan)
    else:
        sampled_values = np.asarray(sampled_returns, dtype=float)

    def estimate_column(values: np.ndarray) -> np.float64:
        """Apply the finite-sample estimator to one return column."""
        finite_values = values[np.isfinite(values)]
        num_observations = finite_values.size
        # Missing rows carry no sample information, so each column selects independently.
        if num_observations >= 20:
            return np.float64(np.std(finite_values, ddof=1))
        # The small-window RMS convention is defined as soon as one return is observed.
        if num_observations >= 1:
            return np.float64(np.sqrt(np.mean(np.square(finite_values))))
        return np.float64(np.nan)

    if sampled_values.ndim == 1:
        return estimate_column(sampled_values)
    return np.asarray([estimate_column(values) for values in sampled_values.T], dtype=float)


def compute_sampled_vols(prices: Union[pd.DataFrame, pd.Series],
                         freq_vol: str = 'ME',
                         freq_return: Optional[str] = None,
                         include_start_date: bool = False,
                         include_end_date: bool = False,
                         ) -> Union[pd.DataFrame, pd.Series]:
    """Compute annualized realized volatility from sampled returns.

    Args:
        prices: Price time series
        freq_vol: Frequency for volatility estimation window (e.g., 'ME' for monthly)
        freq_return: Frequency for return calculation (e.g., 'D' for daily)
        include_start_date: Include period start in resampling
        include_end_date: Include period end in resampling

    Returns:
        Annualized volatility time series. Windows without enough observed returns remain missing
        without affecting eligible Series or DataFrame columns
    """
    # Compute returns at specified frequency
    sampled_returns = to_returns(prices=prices, freq=freq_return,
                                 include_start_date=include_start_date, include_end_date=include_end_date)

    # Split returns by volatility estimation window
    sampled_returns_at_vol_freq = da.split_df_by_freq(df=sampled_returns,
                                                      freq=freq_vol,
                                                      overlap_frequency=None,
                                                      include_start_date=include_start_date,
                                                      include_end_date=include_end_date)

    # Compute volatility for each window
    vol_samples = {}
    for key, df in sampled_returns_at_vol_freq.items():
        # Normalize nullable pandas values before applying NumPy reducers column by column.
        sampled_values = df.to_numpy(dtype=float, na_value=np.nan)
        vol_samples[key] = estimate_vol(sampled_returns=sampled_values)

    # Convert to time series and annualize
    if isinstance(prices, pd.Series):
        vols = pd.DataFrame.from_dict(vol_samples, orient='index').iloc[:, 0].rename(prices.name)
    else:
        vols = pd.DataFrame.from_dict(vol_samples, orient='index', columns=prices.columns)
    vols = vols.multiply(np.sqrt(infer_annualisation_factor_from_df(sampled_returns)))

    return vols


def portfolio_navs_to_additive(grouped_nav: pd.DataFrame,
                               portfolio_name: str
                               ) -> pd.DataFrame:
    """Adjust asset NAVs to make PA returns additive to portfolio total.

    Args:
        grouped_nav: DataFrame with portfolio and asset NAVs
        portfolio_name: Column name for portfolio total

    Returns:
        DataFrame with adjusted asset NAVs
    """
    portfolio_nav = grouped_nav[portfolio_name]
    ac_nav_adj = adjust_component_navs_to_portfolio(portfolio_nav=portfolio_nav,
                                                    component_navs=grouped_nav.drop(columns=[portfolio_name]))
    grouped_nav = pd.concat([portfolio_nav, ac_nav_adj], axis=1, sort=True)
    return grouped_nav


def adjust_component_navs_to_portfolio(portfolio_nav: pd.Series,
                                       component_navs: pd.DataFrame
                                       ) -> pd.DataFrame:
    """Rescale component NAVs so their PA returns sum to the portfolio PA return.

    Used for portfolio NAV decomposition: when a portfolio's total return is
    expressed as a sum of additive components (carry types, fundamental return
    sources, gross vs net vs costs, etc.), the corresponding component NAVs
    don't *automatically* sum back to the portfolio NAV — geometric compounding
    introduces a small residual gap from the linear additivity of period
    returns. This function rescales each component NAV by a common
    time-weighted factor that closes the gap, so the visualised stacked NAVs
    add up to the portfolio total.

    Formula::

        c_m(t) = ((portfolio_pa / n + 1) / (mean(component_pa) + 1)) ** t

    where ``n`` is the number of components and ``t`` is years-from-start.
    For truly additive components, ``mean(component_pa) ≈ portfolio_pa / n``
    and ``c_m`` is close to 1 — only a small adjustment is applied.

    Each component is rescaled by the same ``c_m(t)``, preserving the
    *relative* contributions of components to the portfolio while making
    them sum to the portfolio NAV exactly at the terminal point. Intended
    use is display / stacked-area charting, not formal attribution.

    Args:
        portfolio_nav: Portfolio NAV time series.
        component_navs: DataFrame of component NAVs (one column per
            additive return component — e.g. total return, dividend
            yield, funding cost).

    Returns:
        DataFrame of rescaled component NAVs, same columns as input.
    """
    portfolio_pa = compute_pa_return(prices=portfolio_nav)
    components_pa = compute_pa_return(prices=component_navs)
    n = len(component_navs.columns)
    component_navs_adj = component_navs.copy()

    # Time-weighted adjustment factor. For truly additive components,
    # mean(component_pa) ≈ portfolio_pa / n, so c_m ≈ 1 and adjustment
    # is small. The small deviation closes the compounding gap.
    t = (portfolio_nav.index - portfolio_nav.index[0]).days.to_numpy() / CALENDAR_DAYS_PER_YEAR_SHARPE
    c_m = ((portfolio_pa / n + 1.0) / (np.nanmean(components_pa) + 1.0)) ** t
    ratio = npo.np_array_to_df_columns(a=c_m, ncols=n)

    component_navs_adj = component_navs_adj.multiply(ratio)
    component_navs_adj = component_navs_adj[component_navs.columns]
    return component_navs_adj


def compute_net_return_ex_perf_man_fees(gross_return: pd.Series,
                       man_fee: float = 0.01,
                       perf_fee: float = 0.2,
                       perf_fee_frequency: str = 'YE'
                       ) -> pd.Series:
    """Compute net returns after management and performance fees.

    Args:
        gross_return: Gross return time series. Unique dates are interpreted chronologically
            regardless of physical row order.
        man_fee: Annual management fee (e.g., 0.01 for 1%)
        perf_fee: Performance fee rate on profits above HWM (e.g., 0.2 for 20%)
        perf_fee_frequency: Frequency for performance fee crystallization (e.g., 'YE'). Calendar
            boundaries use the latest available observation on or before the period end.

    Returns:
        Net return time series after fees in increasing date order.

    Raises:
        ValueError: If gross_return contains duplicate dates.
    """
    # Fee state is path-dependent, so physical row order cannot define its chronology.
    if not gross_return.index.is_unique:
        raise ValueError("gross_return index must not contain duplicate dates")
    gross_return = gross_return.sort_index()

    # Generate performance fee crystallization dates
    perf_fee_cristalization_schedule = da.generate_dates_schedule(
        time_period=da.TimePeriod(gross_return.index[0], gross_return.index[-1]),
        freq=perf_fee_frequency)

    # Map off-grid calendar ends backward so later returns never enter the prior fee period.
    perf_cris_positions = np.asarray(
        gross_return.index.searchsorted(perf_fee_cristalization_schedule, side='right'),
        dtype=int,
    ) - 1
    perf_cris_positions = perf_cris_positions[perf_cris_positions >= 0]
    perf_cris_dates = np.zeros(len(gross_return.index), dtype=bool)
    perf_cris_dates[perf_cris_positions] = True

    # Initialize tracking DataFrame
    nav_data = pd.DataFrame(data=0.0,
                            index=gross_return.index,
                            columns=['Net Return', 'NAV', 'GAV', 'PF', 'HWM', 'CPF'])
    nav_data.insert(loc=0, column='gross return', value=gross_return.to_numpy())
    nav_data = nav_data.copy()

    # Set initial values
    nav_data.loc[nav_data.index[0], 'GAV'] = 100.0
    nav_data.loc[nav_data.index[0], 'NAV'] = 100.0
    nav_data.loc[nav_data.index[0], 'HWM'] = 100.0

    # Iterate through dates to compute fees
    for date, last_date, perf_cris_date in zip(nav_data.index[1:], nav_data.index[0:], perf_cris_dates[1:]):
        # Accrue management fee
        man_fee_dt = man_fee * (date-last_date).days/365.0
        nav_data.loc[date, 'GAV'] = (1.0+nav_data.loc[date, 'gross return']-man_fee_dt)*nav_data.loc[last_date, 'GAV']

        # Compute performance fee on profits above HWM
        nav_data.loc[date, 'PF'] = perf_fee*np.maximum(nav_data.loc[date, 'GAV']-nav_data.loc[last_date, 'HWM'], 0.0)
        nav_data.loc[date, 'NAV'] = nav_data.loc[date, 'GAV']-nav_data.loc[date, 'PF']
        nav_data.loc[date, 'HWM'] = nav_data.loc[last_date, 'HWM']

        # Crystallize performance fee at period end.
        # On crystallization day, CPF (crystallized perf fee) is recorded as
        # the accrued PF, HWM is bumped up to max(NAV, prior HWM), and GAV is
        # reduced by CPF so that GAV on the next iteration represents the
        # capital actually carried forward (= NAV before the next period's
        # gross return is applied). NAV is intentionally NOT recomputed here;
        # the relevant invariant is `nav_data[last_date, 'GAV'] == NAV after
        # crystallization` going into the next iteration. The displayed
        # GAV column therefore has a discontinuity at crystallization dates,
        # which is the audit-trail intent (fee paid out of GAV).
        if perf_cris_date:
            nav_data.loc[date, 'CPF'] = nav_data.loc[date, 'PF']
            nav_data.loc[date, 'HWM'] = np.maximum(nav_data.loc[date, 'NAV'], nav_data.loc[last_date, 'HWM'])
            nav_data.loc[date, 'GAV'] = nav_data.loc[date, 'GAV'] - nav_data.loc[date, 'CPF']

    # Convert NAV to returns
    net_return = nav_data['NAV'] / nav_data['NAV'].shift(1)-1.0
    net_return = net_return.copy()
    net_return.iloc[0] = 0.0
    net_return = net_return.rename(gross_return.name)

    return net_return


def compute_net_navs_ex_perf_man_fees(navs: Union[pd.Series, pd.DataFrame],
                                      man_fee: float = 0.01,
                                      perf_fee: float = 0.2,
                                      perf_fee_frequency: str = 'YE'
                                      ) -> Union[pd.Series, pd.DataFrame]:
    """Compute net NAVs after management and performance fees.

    Args:
        navs: Gross NAV time series
        man_fee: Annual management fee
        perf_fee: Performance fee rate on profits above HWM
        perf_fee_frequency: Performance fee crystallization frequency

    Returns:
        Net NAV time series after fees
    """
    gross_returns = navs.pct_change()
    net_returns = []
    if isinstance(navs, pd.Series):
        net_returns = compute_net_return_ex_perf_man_fees(gross_return=gross_returns,
                                         man_fee=man_fee,
                                         perf_fee=perf_fee,
                                         perf_fee_frequency=perf_fee_frequency)
    else:
        for column in gross_returns.columns:
            net = compute_net_return_ex_perf_man_fees(gross_return=gross_returns[column],
                                     man_fee=man_fee,
                                     perf_fee=perf_fee,
                                     perf_fee_frequency=perf_fee_frequency)
            net_returns.append(net)
        net_returns = pd.concat(net_returns, axis=1, sort=True)
    net_nav = returns_to_nav(returns=net_returns)
    return net_nav


def returns_to_nav(returns: Union[np.ndarray, pd.Series, pd.DataFrame],
                   init_period: Optional[int] = 0,
                   terminal_value: np.ndarray = None,
                   init_value: Union[np.ndarray, float] = None,
                   first_date: pd.Timestamp = None,
                   freq: Optional[str] = None,
                   constant_trade_level: bool = False,
                   ffill_between_nans: bool = True,
                   is_log_returns: bool = False
                   ) -> Union[pd.Series, pd.DataFrame]:
    """Convert returns to NAV time series.

    Args:
        returns: Return time series
        init_period: If 1, set first non-NaN return to zero; if 0, set previous value to zero
        terminal_value: Target terminal NAV value for scaling
        init_value: Target initial NAV value (default: 1.0)
        first_date: Inclusive date through which observed returns are set to zero; missing
            observations remain missing. Takes precedence over init_period
        freq: Resampling frequency
        constant_trade_level: If True, use arithmetic cumsum; if False, use geometric compounding
        ffill_between_nans: Forward-fill NAV between NaN returns
        is_log_returns: If True, convert from log returns

    Returns:
        NAV time series starting at init_value (or 1.0)
    """
    # Apply an explicit pandas cutoff before the fallback initialization convention.
    if first_date is not None and isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = returns.copy(deep=True)
        cutoff_returns = returns.loc[:first_date]
        returns.loc[:first_date] = cutoff_returns.mask(cutoff_returns.notna(), 0.0)
    elif init_period is not None and isinstance(returns, np.ndarray) is False:
        returns = to_zero_first_nonnan_returns(returns=returns, init_period=init_period)

    # Convert log returns to simple returns
    if is_log_returns:
        returns = np.expm1(returns)

    # Compute NAV using arithmetic or geometric compounding
    if constant_trade_level:
        if isinstance(returns, np.ndarray):
            strategy_nav = np.nancumsum(returns, axis=0) + 1.0
            strategy_nav = np.where(np.isnan(returns), np.nan, strategy_nav)
        else:
            strategy_nav = returns.cumsum(skipna=True, axis=0).add(1.0)
    else:
        if isinstance(returns, np.ndarray):
            strategy_nav = np.cumprod(1.0+returns, axis=0)
        else:
            strategy_nav = returns.add(1.0).cumprod(skipna=True)

    # Scale to target terminal or initial value
    if terminal_value is not None:
        terminal_value_last = dfo.get_last_nonnan_values(strategy_nav)
        strategy_nav = strategy_nav*(terminal_value/terminal_value_last)
    elif init_value is not None:
        initial_value_first = dfo.get_first_nonnan_values(df=strategy_nav)
        strategy_nav = strategy_nav*(init_value / initial_value_first)

    # Resample and forward-fill if needed
    if freq is not None and isinstance(returns, np.ndarray) is False:
        strategy_nav = strategy_nav.asfreq(freq, method='ffill').ffill()

    if ffill_between_nans and isinstance(returns, np.ndarray) is False:
        strategy_nav = df_price_ffill_between_nans(prices=strategy_nav, method='ffill')

    return strategy_nav


def prices_to_scaled_nav(prices: Union[pd.Series, pd.DataFrame], scale=0.5):
    """Scale price returns by constant factor.

    Args:
        prices: Price time series
        scale: Scaling factor for returns (e.g., 0.5 for half leverage)

    Returns:
        Scaled NAV time series
    """
    returns = scale * to_returns(prices=prices, is_log_returns=False, is_first_zero=True)
    navs = returns_to_nav(returns=returns)
    return navs


def prices_at_freq(prices: Union[pd.Series, pd.DataFrame],
                   freq: Optional[str] = None,
                   include_start_date: bool = False,
                   include_end_date: bool = False,
                   ffill_nans: bool = True,
                   fill_na_method: Optional[str] = 'ffill'
                   ) -> Union[pd.Series, pd.DataFrame]:
    """Resample prices to specified frequency.

    Args:
        prices: Price time series
        freq: Target frequency (e.g., 'D', 'W', 'M')
        include_start_date: Include period start in resampling
        include_end_date: Include period end in resampling
        ffill_nans: Forward-fill NaN values
        fill_na_method: Method for filling NaN ('ffill', 'bfill', None)

    Returns:
        Resampled price time series

    Note:
        A prices index that is not in chronological order is sorted first: the fill below runs
        in row order, and every return computed downstream differences adjacent rows.
    """
    # freq is not None delegates to df_asfreq, which sorts for its own reasons; this branch
    # ffills in place and would otherwise carry a later price onto an earlier date
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()

    if freq is not None:
        if ffill_nans:
            fill_na_method = 'ffill'
        else:
            fill_na_method = None
        prices = dff.df_asfreq(df=prices,
                               freq=freq,
                               include_start_date=include_start_date,
                               include_end_date=include_end_date,
                               fill_na_method=fill_na_method)
    else:
        # Previously this branch ignored `ffill_nans` entirely and gated
        # only on `fill_na_method` (which defaults to 'ffill'). Result:
        # callers passing ffill_nans=False without also overriding
        # fill_na_method got ffilled prices anyway — opposite of what the
        # parameter name promises. Mirror the freq-is-not-None branch:
        # ffill_nans=False disables fill regardless of fill_na_method.
        if not ffill_nans:
            fill_na_method = None
        if fill_na_method is not None:
            if fill_na_method == 'ffill':
                prices = prices.ffill()
            elif fill_na_method == 'bfill':
                prices = prices.bfill()
            else:
                raise NotImplementedError(f"fill_na_method={fill_na_method}")

    return prices


def log_returns_to_nav(log_returns: Union[np.ndarray, pd.Series, pd.DataFrame],
                       init_period: Optional[int] = None,
                       terminal_value: np.ndarray = None,
                       init_value: Union[np.ndarray, float] = None
                       ) -> Union[pd.Series, pd.DataFrame]:
    """Convert log returns to NAV using exponential compounding.

    Args:
        log_returns: Log return time series
        init_period: If specified, set first non-NaN return to zero
        terminal_value: Target terminal NAV value
        init_value: Target initial NAV value

    Returns:
        NAV time series computed as exp(cumsum(log_returns))
    """
    if init_period is not None and isinstance(log_returns, np.ndarray) is False:
        log_returns = to_zero_first_nonnan_returns(returns=log_returns, init_period=init_period)

    strategy_nav = np.exp(log_returns.cumsum(axis=0, skipna=True))

    # Scale to target values
    if terminal_value is not None:
        strategy_nav = strategy_nav.multiply(terminal_value/dfo.get_last_nonnan_values(df=strategy_nav))
    elif init_value is not None:
        strategy_nav = strategy_nav.divide(dfo.get_first_nonnan_values(df=strategy_nav) / init_value)

    return strategy_nav


def long_short_to_relative_nav(long_price: pd.Series, short_price: pd.Series) -> pd.Series:
    """Compute NAV for long-short strategy.

    Args:
        long_price: Price series for long position
        short_price: Price series for short position

    Returns:
        Relative NAV series (long return minus short return)
    """
    returns = to_returns(pd.concat([long_price, short_price],
                                   axis=1, sort=True).ffill(), is_first_zero=True)
    relative_returns = np.subtract(returns[long_price.name], returns[short_price.name])
    relative_nav = returns_to_nav(returns=relative_returns, init_period=1)
    return relative_nav


def to_portfolio_returns(weights: pd.DataFrame,
                         returns: pd.DataFrame,
                         portfolio_name: str = 'portfolios'
                         ) -> pd.Series:
    """Compute portfolio returns from asset weights and returns.

    Uses lagged weights (rebalanced at prior period close).

    NaN handling — IMPORTANT:
        This function aggregates ``returns * lagged_weights`` via ``nansum``
        across the asset axis. A NaN return contributes ``0`` to that day's
        portfolio PnL — equivalent to "asset held its notional but earned 0%".
        It does NOT renormalize remaining asset weights.

        Example: weights = [0.5, 0.5], returns = [+0.02, NaN]. Output =
        ``nansum([0.5*0.02, 0.5*NaN]) = 0.01``, not 0.02.

        This is the right convention if NaN means "asset wasn't tradable
        that day, position held in cash earning 0%". It is wrong if NaN
        means "data missing, treat the position as fully invested in the
        remaining assets". For the latter, drop NaN rows or renormalize
        weights yourself before calling.

        A date where ALL asset returns are NaN produces NaN portfolio
        return (rather than 0), so fully-NaN periods propagate correctly.

    Args:
        weights: Portfolio weight DataFrame (columns = assets)
        returns: Asset return DataFrame (columns = assets)
        portfolio_name: Name for output series

    Returns:
        Portfolio return time series
    """
    weights_1 = weights.shift(1)
    portfolio_pnl = returns.multiply(weights_1).to_numpy()
    pnl = np.nansum(portfolio_pnl, axis=1)
    is_all_nan = np.all(np.isnan(portfolio_pnl), axis=1)
    portfolio_pnl = np.where(is_all_nan, np.nan, pnl)
    portfolio_returns = pd.Series(data=portfolio_pnl,
                                  index=returns.index,
                                  name=portfolio_name)
    return portfolio_returns


def portfolio_returns_to_nav(returns: pd.DataFrame,
                             init_period: Optional[int] = 1,
                             init_value: Union[np.ndarray, float] = None,
                             freq: Optional[str] = None
                             ) -> Union[pd.Series, pd.DataFrame]:
    """Aggregate portfolio returns across assets to single NAV.

    NaN handling: uses ``nansum`` across columns — a NaN contribution
    on a given date is treated as 0 PnL, equivalent to "this asset held
    its notional but earned 0% that period". See ``to_portfolio_returns``
    docstring for the full discussion.

    Args:
        returns: Return DataFrame with assets as columns
        init_period: Set first non-NaN return to zero
        init_value: Initial NAV value
        freq: Resampling frequency

    Returns:
        Aggregate portfolio NAV series
    """
    agg_pnl = pd.Series(np.nansum(returns.to_numpy(), axis=1), index=returns.index)
    nav = returns_to_nav(returns=agg_pnl, init_period=init_period, init_value=init_value, freq=freq)
    return nav


def to_zero_first_nonnan_returns(returns: Union[pd.Series, pd.DataFrame],
                                 init_period: Union[int, None] = 0
                                 ) -> Union[pd.Series, pd.DataFrame]:
    """Set first non-NaN return(s) to zero for NAV initialization.

    Args:
        returns: Return time series
        init_period:
            - 0: Set value before first non-NaN to zero (if exists and is NaN)
            - 1: Set first non-NaN value to zero
            - None: No modification

    Returns:
        Modified return time series
    """
    if init_period is None:
        return returns

    if not isinstance(init_period, int):
        raise ValueError(f"init_period must be integer")

    returns: Union[pd.Series, pd.DataFrame] = returns.copy(deep=True)
    if init_period not in (0, 1):
        warnings.warn(f"in to_zero_first_nonnan_returns init_period={init_period} is not supported")
        return returns

    if isinstance(returns, pd.Series) and returns.isna().all():
        return returns

    if init_period == 0:
        # Set previous value to zero if it's NaN
        first_nonnan_index = dfo.get_nonnan_index(df=returns, position='first')

        if isinstance(returns, pd.Series):
            idx_pos = returns.index.get_loc(first_nonnan_index)
            if idx_pos > 0:
                prev_idx = returns.index[idx_pos - 1]
                if pd.isna(returns.loc[prev_idx]):
                    returns.loc[prev_idx] = 0.0
        else:
            for first_nonnan_index_, column in zip(first_nonnan_index, returns.columns):
                if returns[column].isna().all():
                    continue
                idx_pos = returns.index.get_loc(first_nonnan_index_)
                if idx_pos > 0:
                    prev_idx = returns.index[idx_pos - 1]
                    if pd.isna(returns.loc[prev_idx, column]):
                        returns.loc[prev_idx, column] = 0.0

    elif init_period == 1:
        # Set first non-NaN value to zero. Previously there was a guard
        # `if first_nonnan_index >= first_date` here, but since
        # `first_date = returns.index[0]`, any non-NaN index is by
        # definition >= the first index — the check was always True.
        # Removed.
        first_nonnan_index = dfo.get_nonnan_index(df=returns, position='first')
        if isinstance(returns, pd.Series):
            returns.loc[first_nonnan_index] = 0.0
        else:
            for first_nonnan_index_, column in zip(first_nonnan_index, returns.columns):
                if returns[column].isna().all():
                    continue
                returns.loc[first_nonnan_index_, column] = 0.0
    return returns


def get_excess_returns_nav(prices: Union[pd.DataFrame, pd.Series],
                           funding_rate: pd.Series,
                           freq: str = 'B'
                           ) -> Union[pd.DataFrame, pd.Series]:
    """Compute excess return NAV after funding costs.

    Args:
        prices: Price time series
        funding_rate: Funding rate time series (annualized)
        freq: Return calculation frequency

    Returns:
        Excess return NAV scaled to match terminal price
    """
    if not isinstance(funding_rate, pd.Series):
        raise ValueError(f"funding_rate must be series")

    nav_returns = to_returns(prices=prices, freq=freq)

    # Convert annualized funding rate to period rate
    funding_rate_dt = dfo.multiply_df_by_dt(df=funding_rate, dates=nav_returns.index, lag=1)

    # Subtract funding costs from returns
    if isinstance(prices, pd.DataFrame):
        ncols = len(prices.columns)
        excess_returns = nav_returns.subtract(npo.np_array_to_df_columns(a=funding_rate_dt.to_numpy(), ncols=ncols),
                                              axis=1)
    else:
        data = nav_returns.to_numpy()-funding_rate_dt.to_numpy()
        excess_returns = pd.Series(data, index=nav_returns.index, name=nav_returns.name)

    # Scale to match terminal price
    terminal_value = dfo.get_last_nonnan_values(prices)
    excess_nav = returns_to_nav(returns=excess_returns,
                                    terminal_value=terminal_value,
                                    init_period=1)
    return excess_nav


# =============================================================================
# LEVERAGE ADJUSTMENT
# =============================================================================
# Paste these three functions into qis/perfstats/returns.py alongside
# compute_excess_returns. The annualisation import at the top of that file
# (from qis.utils.annualisation import infer_annualisation_factor_from_df)
# is already present, so no additional imports are needed.


def _validate_leverage_parameters(leverage: object,
                                  periods_per_year: object) -> None:
    """Validate the shared scalar domain of the public leverage transforms.

    Args:
        leverage: Candidate debt-to-equity ratio.
        periods_per_year: Candidate annualization factor or ``None``.

    Raises:
        ValueError: If leverage is not a finite nonnegative real scalar, or if
            periods_per_year is neither ``None`` nor a positive integer.
    """
    if (isinstance(leverage, (bool, np.bool_))
            or not isinstance(leverage, Real)
            or not isfinite(leverage)
            or leverage < 0.0):
        raise ValueError(
            f"leverage must be a finite non-negative real number, got {leverage!r}"
        )

    if (periods_per_year is not None
            and (isinstance(periods_per_year, (bool, np.bool_))
                 or not isinstance(periods_per_year, Integral)
                 or periods_per_year <= 0)):
        raise ValueError(
            f"periods_per_year must be a positive integer or None, got {periods_per_year!r}"
        )


def _get_periodic_financing_rate(financing_rate: Union[float, pd.Series],
                                 returns_index: pd.Index,
                                 periods_per_year: int
                                 ) -> Union[float, pd.Series]:
    """Convert annual financing to a chronologically aligned periodic rate.

    Args:
        financing_rate: Constant annual rate or a date-indexed Series of annual rates.
        returns_index: Return dates to which a financing Series is aligned.
        periods_per_year: Annualization factor used for periodic conversion.

    Returns:
        Constant periodic rate or a Series aligned to returns_index.

    Raises:
        ValueError: If a scalar rate is not finite and real, or if a Series contains an
            invalid observed value, an incompatible date axis, ``NaT``, or duplicate dates.
    """
    if isinstance(financing_rate, pd.Series):
        # Reject undefined or incompatible chronology before point-in-time alignment.
        financing_index = financing_rate.index
        if (not isinstance(returns_index, pd.DatetimeIndex)
                or not isinstance(financing_index, pd.DatetimeIndex)):
            raise ValueError(
                "returns and financing_rate must use DatetimeIndex "
                "when financing_rate is a Series"
            )
        if returns_index.hasnans or financing_index.hasnans:
            raise ValueError("returns and financing_rate indexes must not contain NaT")
        if not financing_rate.index.is_unique:
            raise ValueError("financing_rate index must not contain duplicate dates")
        if ((returns_index.tz is None)
                != (financing_index.tz is None)):
            raise ValueError(
                "returns and financing_rate indexes must both be "
                "timezone-naive or timezone-aware"
            )

        # Missing values remain unavailable; every observed rate must be finite and real.
        observed_financing = financing_rate.dropna()
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not isfinite(value)
            for value in observed_financing.array
        ):
            raise ValueError(
                "financing_rate Series must contain only finite real values "
                "or missing values"
            )

        financing_rate = financing_rate.sort_index()
        if financing_index.tz is not None:
            financing_rate = financing_rate.tz_convert(returns_index.tz)
        return financing_rate.reindex(returns_index, method='ffill') / periods_per_year

    # Reject invalid constants before they can silently produce non-finite returns.
    if (isinstance(financing_rate, (bool, np.bool_))
            or not isinstance(financing_rate, Real)
            or not isfinite(financing_rate)):
        raise ValueError(
            f"financing_rate must be a finite real number or a Series, got {financing_rate!r}"
        )
    return float(financing_rate) / periods_per_year


def delever_returns(returns: Union[pd.Series, pd.DataFrame],
                    leverage: float,
                    financing_rate: Union[float, pd.Series] = 0.0,
                    periods_per_year: int = None
                    ) -> Union[pd.Series, pd.DataFrame]:
    """Recover unlevered asset-level returns from levered portfolio returns.

    Inverts the standard constant-leverage identity:

        r_portfolio = (1 + L) * r_asset - L * r_financing

    to recover:

        r_asset = (r_portfolio + L * r_financing) / (1 + L)

    Useful for comparing levered vehicles (BDCs, levered ETFs like TQQQ, levered
    loan funds) to their unlevered analogues on a common-risk basis.

    Args:
        returns: Period returns of the levered portfolio (Series or DataFrame).
        leverage: Finite, nonnegative leverage ratio L (debt / equity). Booleans are invalid.
            For a 1.5x levered fund pass 0.5; for a 3x ETF pass 2.0; for the typical BDC at
            1.0x debt-to-equity pass 1.0.
        financing_rate: Annualised financing rate. Pass a finite real scalar for constant cost
            (e.g. risk-free rate as a crude proxy), or a Series indexed by date for
            time-varying financing (recommended for accuracy). Observed Series values must be
            finite and real; ordinary or nullable missing values remain unavailable. For nonzero
            leverage, both date axes must be ``DatetimeIndex`` objects without ``NaT`` and must
            both be timezone-naive or timezone-aware. Different aware zones align by absolute
            instant. Series observations are sorted chronologically, aligned to return dates, and
            forward-filled from prior observations only; duplicate funding dates are invalid.
            Default 0.0 ignores financing — only correct if the portfolio earns the financing
            rate on its borrowed capital, which is rarely the case.
        periods_per_year: Strictly positive integer annualisation factor used to convert the
            financing rate to per-period. If None, inferred from the index frequency.

    Returns:
        De-levered returns matching the input shape and pandas labels. Zero leverage
        returns an independent copy, even when financing data is unavailable.

    Raises:
        ValueError: If leverage is not a finite nonnegative real scalar; if an explicit
            periods_per_year is not a positive integer; or, for nonzero leverage, if financing
            values or date axes violate the contract described above.

    Note:
        This assumes constant leverage and a single-tier financing structure. Real
        BDCs and levered funds have time-varying leverage, multiple debt tranches at
        different rates, and credit spreads above the risk-free rate. For a precise
        treatment, use the realised interest expense from filings rather than this
        approximation.

    Example:
        >>> # De-lever OCSL using its actual weighted average debt rate of 6.1%
        >>> ocsl_unlev = delever_returns(ocsl_returns, leverage=1.07,
        ...                              financing_rate=0.061)
    """
    # Validate explicit inputs before zero leverage can bypass an invalid annualization factor.
    _validate_leverage_parameters(leverage=leverage, periods_per_year=periods_per_year)

    # Preserve an exact identity before unavailable financing can propagate NaNs.
    if leverage == 0.0:
        return returns.copy()

    if periods_per_year is None:
        periods_per_year = int(round(infer_annualisation_factor_from_df(
            returns if isinstance(returns, pd.DataFrame) else returns.to_frame()
        )))

    # Apply the shared chronological funding contract before the inverse equation.
    rf_per_period = _get_periodic_financing_rate(financing_rate=financing_rate,
                                                 returns_index=returns.index,
                                                 periods_per_year=periods_per_year)

    if isinstance(returns, pd.DataFrame) and isinstance(rf_per_period, pd.Series):
        # Apply each financing observation across all assets on the same date.
        return returns.add(leverage * rf_per_period, axis=0) / (1.0 + leverage)

    # Restore the return label that Series alignment can otherwise discard.
    result = (returns + leverage * rf_per_period) / (1.0 + leverage)
    if isinstance(result, pd.Series) and isinstance(returns, pd.Series):
        result.name = returns.name
    return result


def lever_returns(returns: Union[pd.Series, pd.DataFrame],
                  leverage: float,
                  financing_rate: Union[float, pd.Series] = 0.0,
                  periods_per_year: int = None
                  ) -> Union[pd.Series, pd.DataFrame]:
    """Apply constant leverage to an unlevered asset return series.

    Forward direction of ``delever_returns``:

        r_portfolio = (1 + L) * r_asset - L * r_financing

    Useful for stress-testing unlevered strategies under hypothetical leverage,
    or for comparing managed-account performance against a levered benchmark.

    Args:
        returns: Period returns of the unlevered asset (Series or DataFrame).
        leverage: Finite, nonnegative leverage ratio L (debt / equity). Booleans are invalid.
        financing_rate: Annualised financing rate. A scalar must be finite and real. Observed
            Series values must be finite and real; ordinary or nullable missing values remain
            unavailable. For nonzero leverage, both date axes must be ``DatetimeIndex`` objects
            without ``NaT`` and must both be timezone-naive or timezone-aware. Different aware
            zones align by absolute instant. Series observations are sorted chronologically,
            aligned to return dates, and forward-filled from prior observations only; duplicate
            funding dates are invalid.
        periods_per_year: Strictly positive integer annualisation factor. If None, inferred from
            the index.

    Returns:
        Levered returns matching the input shape and pandas labels. Zero leverage
        returns an independent copy, even when financing data is unavailable.

    Raises:
        ValueError: If leverage is not a finite nonnegative real scalar; if an explicit
            periods_per_year is not a positive integer; or, for nonzero leverage, if financing
            values or date axes violate the contract described above.

    Example:
        >>> # Show what GCF would look like at 1x leverage with BDC-like financing
        >>> gcf_levered = lever_returns(gcf_returns, leverage=1.0,
        ...                             financing_rate=0.061)
    """
    # Validate explicit inputs before zero leverage can bypass an invalid annualization factor.
    _validate_leverage_parameters(leverage=leverage, periods_per_year=periods_per_year)

    # Preserve an exact identity before unavailable financing can propagate NaNs.
    if leverage == 0.0:
        return returns.copy()

    if periods_per_year is None:
        periods_per_year = int(round(infer_annualisation_factor_from_df(
            returns if isinstance(returns, pd.DataFrame) else returns.to_frame()
        )))

    # Apply the shared chronological funding contract before the forward equation.
    rf_per_period = _get_periodic_financing_rate(financing_rate=financing_rate,
                                                 returns_index=returns.index,
                                                 periods_per_year=periods_per_year)

    if isinstance(returns, pd.DataFrame) and isinstance(rf_per_period, pd.Series):
        # Apply each financing observation across all assets on the same date.
        return ((1.0 + leverage) * returns).sub(leverage * rf_per_period, axis=0)

    # Restore the return label that Series alignment can otherwise discard.
    result = (1.0 + leverage) * returns - leverage * rf_per_period
    if isinstance(result, pd.Series) and isinstance(returns, pd.Series):
        result.name = returns.name
    return result


def implied_leverage(levered_returns: Union[pd.Series, pd.DataFrame],
                     unlevered_returns: pd.Series,
                     ) -> Union[float, pd.Series]:
    """Estimate the implicit leverage ratio between two return series via OLS.

    Regresses levered returns on unlevered returns and extracts the slope, which
    under the constant-leverage model equals (1 + L). Useful for inferring the
    effective leverage of a vehicle when not explicitly disclosed.

    Args:
        levered_returns: Returns of the levered vehicle (Series or DataFrame).
        unlevered_returns: Returns of the unlevered analogue (Series).

    Returns:
        Implied leverage ratio L = slope - 1. For Series input, returns a float;
        for DataFrame input, returns a Series indexed by column name.

    Note:
        The estimate is contaminated by any non-leverage differences between the
        two vehicles (security selection, sector tilts, financing spread). Use
        with caution and only when both vehicles target the same underlying exposure.

    Example:
        >>> # Estimate OCSL's implied leverage vs Oaktree GCF
        >>> L = implied_leverage(ocsl_returns, gcf_returns)
        >>> print(f"Implied leverage: {L:.2f}x")
    """
    if isinstance(levered_returns, pd.Series):
        joint = pd.concat([levered_returns, unlevered_returns], axis=1, sort=True).dropna()
        joint.columns = ['y', 'x']
        if len(joint) < 10:
            return np.nan
        slope = np.cov(joint['x'], joint['y'], ddof=1)[0, 1] / np.var(joint['x'], ddof=1)
        return float(slope - 1.0)

    if isinstance(levered_returns, pd.DataFrame):
        return pd.Series(
            {col: implied_leverage(levered_returns[col], unlevered_returns)
             for col in levered_returns.columns},
            name='implied_leverage',
        )

    raise TypeError(f"levered_returns must be Series or DataFrame, got {type(levered_returns)}")


def to_quarterly_returns(returns: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """Compound returns to quarter-end via NAV round-trip.

    Masks any trailing partial quarter — i.e., a quarter whose end-of-quarter
    date falls in a calendar month later than the input's last observed
    month. This matters at the current quarter boundary where monthly-
    reporting funds have posted Jan/Feb returns but not yet the full Q1 —
    without this drop, the resample would forward-fill a 2-month return as
    if it were a 3-month return.

    The completeness check uses calendar months, not exact QE timestamps, so
    weekly (W-FRI) and business-month-end series compound correctly even
    though their stamps don't land on calendar QE. (A previous implementation
    used ``returns.reindex(QE).notna()``, which silently masked the entire
    output for any input whose stamps did not align with calendar QE.)

    Used uniformly across all funds for schema consistency; for series that
    are already quarterly, this is effectively identity (single-obs quarters
    compound to themselves).
    """
    q_returns = to_returns(returns_to_nav(returns), freq='QE')

    # Per column, find the last observed month-end. A quarter ending at QE
    # is complete iff the input has at least one non-NaN observation in
    # that quarter's last calendar month — equivalently,
    # QE <= (last_dt rounded forward to month-end).
    def _last_complete_month_end(s):
        clean = s.dropna()
        if clean.empty:
            return None
        return clean.index.max() + pd.offsets.MonthEnd(0)

    if isinstance(returns, pd.Series):
        last_me = _last_complete_month_end(returns)
        if last_me is None:
            q_returns[:] = float('nan')
        else:
            q_returns = q_returns.where(q_returns.index <= last_me, other=float('nan'))
    else:  # DataFrame — compute per column to handle ragged end dates
        for col in returns.columns:
            last_me = _last_complete_month_end(returns[col])
            if last_me is None:
                q_returns[col] = float('nan')
            else:
                q_returns.loc[q_returns.index > last_me, col] = float('nan')

    return q_returns
