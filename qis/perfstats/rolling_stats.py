"""
module for computing rolling performance stats
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from enum import Enum
from typing import Union, Optional

# qis
from qis.perfstats import returns as ret
from qis.utils import dates as da


class RollingPerfStat(str, Enum):
    TOTAL_RETURNS = 'Total returns'
    PA_RETURNS = 'Pa returns'
    VOL = 'Volatility'
    SHARPE = 'Sharp ratio'
    SKEW = 'Skeweness'


def compute_rolling_perf_stat(prices: Union[pd.DataFrame, pd.Series],
                              rolling_perf_stat: RollingPerfStat = RollingPerfStat.TOTAL_RETURNS,
                              roll_freq: Optional[str] = None,
                              roll_periods: int = 260
                              ) -> Union[pd.DataFrame, pd.Series]:
    """
    compute rolling performance
    default stats is 1y rolling for prices with roll_freq = 'B'
    for monthly returns anf 5y rolling use roll_freq='ME' and roll_periods=5*12
    """
    if rolling_perf_stat == RollingPerfStat.TOTAL_RETURNS:
        perf_stat = compute_rolling_returns(prices=prices, roll_periods=roll_periods)
    elif rolling_perf_stat == RollingPerfStat.PA_RETURNS:
        perf_stat = compute_rolling_pa_returns(prices=prices, roll_periods=roll_periods)
    elif rolling_perf_stat == RollingPerfStat.VOL:
        perf_stat = compute_rolling_vols(prices=prices, roll_freq=roll_freq, roll_periods=roll_periods)
    elif rolling_perf_stat == RollingPerfStat.SHARPE:
        perf_stat = compute_rolling_sharpes(prices=prices, roll_freq=roll_freq, roll_periods=roll_periods)
    elif rolling_perf_stat == RollingPerfStat.SKEW:
        perf_stat = compute_rolling_skew(prices=prices, roll_freq=roll_freq, roll_periods=roll_periods)
    else:
        raise NotImplementedError(f"rolling_perf_stats")
    return perf_stat


def compute_rolling_returns(prices: Union[pd.DataFrame, pd.Series],
                            roll_periods: int = 260
                            ) -> Union[pd.DataFrame, pd.Series]:
    """
    compute rolling returns
    """
    returns = prices.divide(prices.shift(periods=roll_periods)) - 1.0
    return returns


def compute_rolling_pa_returns(prices: Union[pd.DataFrame, pd.Series],
                               roll_periods: int = 260
                               ) -> Union[pd.DataFrame, pd.Series]:
    """
    compute rolling pa returns
    """
    pa_returns = prices.rolling(roll_periods).apply(lambda x: ret.compute_pa_return(x))
    return pa_returns


def compute_rolling_vols(prices: Union[pd.Series, pd.DataFrame],
                        roll_freq: Optional[str] = None,
                        roll_periods: int = 260
                        ) -> Union[pd.Series, pd.DataFrame]:
    log_returns = ret.to_returns(prices=prices, freq=roll_freq, is_log_returns=True, drop_first=False)
    saf = np.sqrt(da.infer_an_from_data(data=log_returns))
    vols = saf * log_returns.rolling(roll_periods).apply(lambda x: np.nanstd(x, ddof=1))
    return vols


def compute_rolling_sharpes(prices: Union[pd.Series, pd.DataFrame],
                            roll_freq: Optional[str] = None,
                            roll_periods: int = 260
                            ) -> Union[pd.Series, pd.DataFrame]:
    log_returns = ret.to_returns(prices=prices, freq=roll_freq, is_log_returns=True, drop_first=False)
    saf = np.sqrt(da.infer_an_from_data(data=log_returns))
    sharpes = log_returns.rolling(roll_periods).apply(lambda x: compute_sharpe(x, saf=saf))
    return sharpes


def compute_sharpe(log_returns: Union[pd.Series, pd.DataFrame], saf: float = None) -> np.ndarray:
    if saf is None:
        saf = np.sqrt(da.infer_an_from_data(data=log_returns))
    mean = np.expm1(np.nanmean(log_returns.to_numpy()))
    vol = np.nanstd(log_returns.to_numpy(), ddof=1)
    if np.greater(vol, 0.0):
        sharpe = saf * mean / vol
    else:
        sharpe = np.nan
    return sharpe


def compute_rolling_skew(prices: Union[pd.Series, pd.DataFrame],
                         roll_freq: Optional[str] = None,
                         roll_periods: int = 120,
                         ) -> Union[pd.Series, pd.DataFrame]:
    log_returns = ret.to_returns(prices=prices, freq=roll_freq, is_log_returns=True, drop_first=False)
    skw = log_returns.rolling(roll_periods).apply(lambda x: compute_skew(x))
    return skw


def compute_skew(log_returns: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
    skw = skew(log_returns.to_numpy(), axis=0, nan_policy='omit')
    return skw


class UnitTests(Enum):
    ROLLING_STATS = 1


def run_unit_test(unit_test: UnitTests):

    import qis.plots.time_series as pts
    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    if unit_test == UnitTests.ROLLING_STATS:

        for rolling_perf_stat in RollingPerfStat:
            stats = compute_rolling_perf_stat(prices=prices, rolling_perf_stat=rolling_perf_stat,
                                              roll_freq='W-WED',
                                              roll_periods=5*52)
            pts.plot_time_series(df=stats,
                                 var_format='{:.2f}',
                                 title=f"{rolling_perf_stat.name}")

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.ROLLING_STATS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
