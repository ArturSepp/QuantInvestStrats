"""
Core performance attribution and return calculations.

This module provides fundamental functions for computing returns, volatilities,
and performance metrics from price data.
"""
# packages
import warnings
import numpy as np
import pandas as pd
from typing import Union, Dict, Optional

# qis
import qis.utils.dates as da
import qis.utils.df_freq as dff
import qis.utils.np_ops as npo
import qis.utils.df_ops as dfo
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
        prices: Price time series
        is_log_returns: If True, compute log returns (overrides return_type)
        return_type: Type of return calculation (LOG, RELATIVE, DIFFERENCE, LEVEL, LEVEL0)
        freq: Resampling frequency (e.g., 'D', 'W', 'M')
        include_start_date: Include period start date when resampling
        include_end_date: Include period end date when resampling
        ffill_nans: Forward-fill NaN prices before computing returns
        drop_first: Drop first return observation
        is_first_zero: Set first non-NaN return to zero
        **kwargs: Additional arguments

    Returns:
        Return time series with NaN for invalid observations
    """
    # Resample prices to specified frequency
    prices = prices_at_freq(prices=prices, freq=freq,
                            include_start_date=include_start_date,
                            include_end_date=include_end_date,
                            ffill_nans=ffill_nans)

    # Identify valid prices (non-NaN and positive)
    prices_np = prices.to_numpy()
    ind_good = np.logical_and(np.isnan(prices_np) == False, np.greater(prices_np, 0.0))

    # Compute returns based on type
    if return_type == ReturnTypes.LOG or is_log_returns:
        returns = np.log(prices).diff(1)
    elif return_type == ReturnTypes.RELATIVE:
        returns = np.divide(prices, prices.shift(1)).add(-1.0)
    elif return_type == ReturnTypes.DIFFERENCE:
        returns = prices - prices.shift(1)
    elif return_type == ReturnTypes.LEVEL:
        returns = prices
    elif return_type == ReturnTypes.LEVEL0:
        returns = prices.shift(1)
    else:
        raise NotImplementedError(f"{return_type}")

    # Mask invalid observations
    returns = returns.where(ind_good, other=np.nan)

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
                               returns_freqs: pd.Series,
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
    group_freqs = get_group_dict(group_data=returns_freqs)
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
        price_end = prices.iloc[-1, :].to_numpy()

    elif isinstance(prices, pd.Series):
        price_0 = prices.iloc[0]
        if np.isnan(price_0):
            price_0 = dfo.get_first_nonnan_values(df=prices)
            warnings.warn(f"detected nan price for prices at date = {prices.index[0]},"
                          f" using first non nan price = {price_0} for {prices.name}")
        price_end = prices.iloc[-1]
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
            compounded_return_pa = np.power(ratio, 1.0 / num_years, where=np.isfinite(ratio)) - 1
        else:
            if annualize_less_1y:
                # Linear annualization for periods <1 year
                compounded_return_pa = total_return / num_years
            else:
                # Use total return without annualization
                compounded_return_pa = ratio - 1.0
    else:
        n = len(prices.columns) if isinstance(prices, pd.DataFrame) else 1
        compounded_return_pa = np.zeros_like(n)

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
                           PerfStat.AN_LOG_RETURN_EXCESS.to_str(): np.nan,
                           PerfStat.APR.to_str(): np.nan,
                           PerfStat.NUM_YEARS.to_str(): np.nan}
        else:
            return_dict = {PerfStat.TOTAL_RETURN.to_str(): np.full(n, fill_value=np.nan),
                           PerfStat.PA_RETURN.to_str(): np.full(n, fill_value=np.nan),
                           PerfStat.AN_LOG_RETURN.to_str(): np.full(n, fill_value=np.nan),
                           PerfStat.AN_LOG_RETURN_EXCESS.to_str(): np.full(n, fill_value=np.nan),
                           PerfStat.APR.to_str(): np.full(n, fill_value=np.nan),
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

    # Build return dictionary
    return_dict = {PerfStat.TOTAL_RETURN.to_str(): total_return,
                   PerfStat.PA_RETURN.to_str(): compounded_return_pa,
                   PerfStat.PA_EXCESS_RETURN.to_str(): excess_return_pa,
                   PerfStat.AN_LOG_RETURN.to_str(): np.log(1.0 + compounded_return_pa, where=np.greater(compounded_return_pa, -1.0)),
                   PerfStat.AN_LOG_RETURN_EXCESS.to_str(): np.log(1.0 + excess_return_pa, where=np.greater(excess_return_pa, -1.0)),
                   PerfStat.AVG_AN_RETURN.to_str(): np.divide(total_return, num_years),
                   PerfStat.APR.to_str(): CALENDAR_DAYS_PER_YEAR_SHARPE*total_return/num_days if num_days > 0 else CALENDAR_DAYS_PER_YEAR_SHARPE*total_return,
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
    # Convert annualized rates to period returns
    rates_dt = dfo.multiply_df_by_dt(df=rates_data, dates=returns.index, lag=None)
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
        Annualized excess return (scalar)
    """
    excess_returns = compute_excess_returns(returns=returns, rates_data=rates_data)
    prices = returns_to_nav(returns=excess_returns, first_date=first_date)
    compounded_return_pa = compute_pa_return(prices=prices, annualize_less_1y=annualize_less_1y)
    if isinstance(compounded_return_pa, np.ndarray):
        compounded_return_pa = compounded_return_pa[0]
    return compounded_return_pa


def estimate_vol(sampled_returns: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    """Estimate volatility from return samples.

    For samples >=20, uses standard deviation with ddof=1.
    For smaller samples, uses RMS to avoid mean adjustment bias.

    Args:
        sampled_returns: Return time series

    Returns:
        Array of volatility estimates
    """
    if isinstance(sampled_returns, np.ndarray):
        n = sampled_returns.shape[0]
    else:
        n = len(sampled_returns.index)

    if n >= 20:
        # Use sample standard deviation for larger samples
        vol = np.nanstd(sampled_returns, axis=0, ddof=1)
    else:
        # Use RMS for small samples to avoid mean adjustment
        vol = np.sqrt(np.nanmean(np.power(sampled_returns, 2), axis=0))
    return vol


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
        Annualized volatility time series
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
        vol_samples[key] = estimate_vol(sampled_returns=df.to_numpy())

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
    ac_nav_adj = adjust_navs_to_portfolio_pa(portfolio_nav=portfolio_nav,
                                             asset_prices=grouped_nav.drop(columns=[portfolio_name]))
    grouped_nav = pd.concat([portfolio_nav, ac_nav_adj], axis=1)
    return grouped_nav


def adjust_navs_to_portfolio_pa(portfolio_nav: pd.Series,
                                asset_prices: pd.DataFrame
                                ) -> pd.DataFrame:
    """Adjust asset NAVs so PA returns are additive to portfolio.

    Uses time-weighted adjustment to match terminal value while
    maintaining relative performance characteristics.

    Args:
        portfolio_nav: Portfolio NAV time series
        asset_prices: Asset NAV DataFrame

    Returns:
        Adjusted asset NAV DataFrame
    """
    portfolio_pa = compute_pa_return(prices=portfolio_nav)
    assets_pa = compute_pa_return(prices=asset_prices)
    n = len(asset_prices.columns)
    asset_prices_adj = asset_prices.copy()

    # Compute time-weighted adjustment factor
    t = (portfolio_nav.index - portfolio_nav.index[0]).days.to_numpy() / CALENDAR_DAYS_PER_YEAR_SHARPE
    c_m = ((portfolio_pa / n + 1.0) / (np.nanmean(assets_pa) + 1.0)) ** t
    ratio = npo.np_array_to_df_columns(a=c_m, ncols=n)

    asset_prices_adj = asset_prices_adj.multiply(ratio)
    asset_prices_adj = asset_prices_adj[asset_prices.columns]
    return asset_prices_adj


def compute_net_return_ex_perf_man_fees(gross_return: pd.Series,
                       man_fee: float = 0.01,
                       perf_fee: float = 0.2,
                       perf_fee_frequency: str = 'YE'
                       ) -> pd.Series:
    """Compute net returns after management and performance fees.

    Args:
        gross_return: Gross return time series
        man_fee: Annual management fee (e.g., 0.01 for 1%)
        perf_fee: Performance fee rate on profits above HWM (e.g., 0.2 for 20%)
        perf_fee_frequency: Frequency for performance fee crystallization (e.g., 'YE')

    Returns:
        Net return time series after fees
    """
    # Generate performance fee crystallization dates
    perf_fee_cristalization_schedule = da.generate_dates_schedule(
        time_period=da.TimePeriod(gross_return.index[0], gross_return.index[-1]),
        freq=perf_fee_frequency)

    perf_cris_dates = np.isin(element=gross_return.index,
                              test_elements=perf_fee_cristalization_schedule,
                              assume_unique=True)

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

        # Crystallize performance fee at period end
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
        net_returns = pd.concat(net_returns, axis=1)
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
        first_date: Date to start NAV at 1.0
        freq: Resampling frequency
        constant_trade_level: If True, use arithmetic cumsum; if False, use geometric compounding
        ffill_between_nans: Forward-fill NAV between NaN returns
        is_log_returns: If True, convert from log returns

    Returns:
        NAV time series starting at init_value (or 1.0)
    """
    # Set first return to zero if needed
    if init_period is not None and isinstance(returns, np.ndarray) is False:
        returns = to_zero_first_nonnan_returns(returns=returns, init_period=init_period)
    elif first_date is not None:
        # Set returns to zero up to first_date
        if isinstance(returns, pd.DataFrame):
            returns.loc[:first_date, :] = 0.0
        elif isinstance(returns, pd.Series):
            returns.loc[:first_date] = 0.0

    # Convert log returns to simple returns
    if is_log_returns:
        returns = np.expm1(returns)

    # Compute NAV using arithmetic or geometric compounding
    if constant_trade_level:
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
    """
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
    returns = to_returns(pd.concat([long_price, short_price], axis=1).ffill(), is_first_zero=True)
    relative_returns = np.subtract(returns[long_price.name], returns[short_price.name])
    relative_nav = returns_to_nav(returns=relative_returns, init_period=1)
    return relative_nav


def to_portfolio_returns(weights: pd.DataFrame,
                         returns: pd.DataFrame,
                         portfolio_name: str = 'portfolios'
                         ) -> pd.Series:
    """Compute portfolio returns from asset weights and returns.

    Uses lagged weights (rebalanced at prior period close) and handles
    NaN returns properly by excluding them from aggregation.

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
                idx_pos = returns.index.get_loc(first_nonnan_index_)
                if idx_pos > 0:
                    prev_idx = returns.index[idx_pos - 1]
                    if pd.isna(returns.loc[prev_idx, column]):
                        returns.loc[prev_idx, column] = 0.0

    elif init_period == 1:
        # Set first non-NaN value to zero
        first_nonnan_index = dfo.get_nonnan_index(df=returns, position='first')
        first_date = returns.index[0]
        if isinstance(returns, pd.Series):
            if first_nonnan_index >= first_date:
                returns.loc[first_nonnan_index] = 0.0
        else:
            for first_nonnan_index_, column in zip(first_nonnan_index, returns.columns):
                if first_nonnan_index_ >= first_date:
                    returns.loc[first_nonnan_index_, column] = 0.0
    else:
        warnings.warn(f"in to_zero_first_nonnan_returns init_period={init_period} is not supported")

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


def df_price_ffill_between_nans(prices: Union[pd.Series, pd.DataFrame],
                                method: Optional[str] = 'ffill'
                                ) -> Union[pd.Series, pd.DataFrame]:
    """Forward-fill prices only between first and last non-NaN dates.

    Preserves leading and trailing NaN values while filling gaps.

    Args:
        prices: Price time series
        method: Fill method ('ffill', 'bfill', or None)

    Returns:
        Price series with gaps filled between first and last valid observations
    """
    is_series_out = False
    if isinstance(prices, pd.Series):
        is_series_out = True
        prices = prices.to_frame()

    # Get first and last valid dates for each column
    first_date = dfo.get_nonnan_index(df=prices, position='first')
    last_date = dfo.get_nonnan_index(df=prices, position='last')

    # Fill only between valid date ranges
    good_parts = []
    for idx, column in enumerate(prices.columns):
        good_price = prices.loc[first_date[idx]:last_date[idx], column]
        if method is not None:
            good_price = good_price.infer_objects(copy=False).ffill()
        good_parts.append(good_price)

    bfilled_data = pd.concat(good_parts, axis=1)
    if bfilled_data.index[0] > prices.index[0]:
        bfilled_data = bfilled_data.reindex(index=prices.index)

    if is_series_out:
        bfilled_data = bfilled_data.iloc[:, 0]
    return bfilled_data
