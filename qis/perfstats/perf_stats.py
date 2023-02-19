"""
compute risk-adjusted performance tables
"""
# packages
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from typing import Callable, Union, Dict, Tuple, Any, Optional
from collections import OrderedDict
from enum import Enum

# qis
import qis.utils.dates as da
import qis.utils.ols as ols
import qis.perfstats.returns as ret
from qis.perfstats.config import PerfStat, PerfParams

STANDARD_TABLE_COLUMNS = (PerfStat.START_DATE,
                          PerfStat.END_DATE,
                          PerfStat.PA_RETURN,
                          PerfStat.VOL,
                          PerfStat.SHARPE,
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
                    PerfStat.SHARPE,
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
                              PerfStat.SHARPE,
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
                                    PerfStat.SHARPE,
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
                          PerfStat.SHARPE,
                          PerfStat.SHARPE_EXCESS,
                          PerfStat.MAX_DD,
                          PerfStat.MAX_DD_VOL,
                          PerfStat.SKEWNESS,
                          PerfStat.KURTOSIS)

COMPACT_TABLE_COLUMNS = (PerfStat.TOTAL_RETURN,
                         PerfStat.PA_RETURN,
                         PerfStat.VOL,
                         PerfStat.SHARPE,
                         PerfStat.MAX_DD,
                         PerfStat.MAX_DD_VOL,
                         PerfStat.SKEWNESS)

SMALL_TABLE_COLUMNS = (PerfStat.TOTAL_RETURN,
                       PerfStat.PA_RETURN,
                       PerfStat.VOL,
                       PerfStat.SHARPE,
                       PerfStat.MAX_DD)

BENCHMARK_TABLE_COLUMNS = (PerfStat.PA_RETURN,
                           PerfStat.VOL,
                           PerfStat.SHARPE,
                           PerfStat.MAX_DD,
                           PerfStat.MAX_DD_VOL,
                           PerfStat.SKEWNESS,
                           PerfStat.ALPHA_AN,
                           PerfStat.BETA,
                           PerfStat.R2)

BENCHMARK_TABLE_COLUMNS2 = (PerfStat.TOTAL_RETURN,
                            PerfStat.PA_RETURN,
                            PerfStat.VOL,
                            PerfStat.SHARPE,
                            PerfStat.MAX_DD,
                            PerfStat.MAX_DD_VOL,
                            PerfStat.SKEWNESS,
                            PerfStat.ALPHA_AN,
                            PerfStat.BETA,
                            PerfStat.R2)


def compute_performance_table(prices: Union[pd.DataFrame, pd.Series],
                              perf_params: PerfParams,
                              ) -> pd.DataFrame:
    """
    get performances for assets in prices
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"must be pd.Dataframe")

    if perf_params is None:
        perf_params = PerfParams()

    dict_data = OrderedDict()
    for asset in prices:
        asset_data = prices[asset].dropna()  # force drop na
        return_dict = ret.compute_returns_dict(prices=asset_data, perf_params=perf_params)
        dict_data[asset] = return_dict
        # keys will be rows = asset, column = keys in return_dict
    data = pd.DataFrame.from_dict(data=dict_data, orient='index')
    return data


def compute_risk_table(prices: pd.DataFrame,
                       perf_params: PerfParams = None
                       ) -> pd.DataFrame:
    """
    compute price data for returns statistics
    prices is limited to [sample_date_0,...,sample_date_N]
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"prices must be dataframe.")

    if perf_params is None:
        perf_params = PerfParams()

    sampled_prices = ret.prices_at_freq(prices=prices, freq=perf_params.freq_vol)

    # drawdowns are computed using its own freq
    if perf_params.freq_vol == perf_params.freq_drawdown:
        dd_sampled_prices = sampled_prices
    else:
        dd_sampled_prices = ret.prices_at_freq(prices=prices, freq=perf_params.freq_drawdown)

    vol_dt = np.sqrt(da.infer_an_from_data(data=sampled_prices))
    dict_data = OrderedDict()
    for asset in sampled_prices:
        sampled_price = sampled_prices[asset].dropna()
        if len(sampled_price.index) > 1:
            sampled_returns = ret.to_returns(prices=sampled_price, return_type=perf_params.return_type, drop_first=True)
            nd_sampled_returns = sampled_returns.to_numpy()
            vol = vol_dt * np.std(nd_sampled_returns, ddof=1)

            asset_dict = {PerfStat.VOL.to_str(): vol,
                          PerfStat.AVG_LOG_RETURN.to_str(): np.nanmean(sampled_returns),
                          PerfStat.START_DATE.to_str(): sampled_price.index[0],
                          PerfStat.END_DATE.to_str(): sampled_price.index[-1],
                          PerfStat.NUM_OBS.to_str(): len(nd_sampled_returns)
                          }
            # compute max dd on business day schedule
            max_dd = compute_max_dd(prices=dd_sampled_prices[asset].dropna())
            rel_returns = sampled_price.pct_change().dropna()
            asset_dict.update({
                PerfStat.MAX_DD.to_str(): max_dd,
                PerfStat.MAX_DD_VOL.to_str(): max_dd / vol if vol > 0.0 else 0.0,
                PerfStat.WORST.to_str(): np.min(rel_returns),
                PerfStat.BEST.to_str(): np.max(rel_returns),
                PerfStat.SKEWNESS.to_str(): skew(nd_sampled_returns, bias=False),
                PerfStat.KURTOSIS.to_str(): kurtosis(nd_sampled_returns, bias=False)
            })
        else:  # no required data
            asset_dict = {PerfStat.VOL.to_str(): np.nan,
                          PerfStat.AVG_LOG_RETURN.to_str(): np.nan,
                          PerfStat.START_DATE.to_str(): np.nan,
                          PerfStat.END_DATE.to_str(): np.nan,
                          PerfStat.NUM_OBS.to_str(): 0
                          }
        dict_data[asset] = asset_dict

        # keys will be rows = asset, column = keys in return_dict
    data = pd.DataFrame.from_dict(data=dict_data, orient='index')

    return data


def compute_ra_perf_table(prices: Union[pd.DataFrame, pd.Series],
                          perf_params: PerfParams = None
                          ) -> pd.DataFrame:
    if perf_params is None:
        perf_params = PerfParams(freq=pd.infer_freq(prices.index))

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    perf_table = compute_performance_table(prices=prices, perf_params=perf_params)

    # is we only need sharpe we only comptute vol without higher order risk
    risk_table = compute_risk_table(prices=prices, perf_params=perf_params)

    # get vol and compute risk adjusted performance
    vol = risk_table[PerfStat.VOL.to_str()]
    perf_table[PerfStat.SHARPE.to_str()] = perf_table[PerfStat.PA_RETURN.to_str()] / vol
    perf_table[PerfStat.SHARPE_LOG_AN.to_str()] = perf_table[PerfStat.AN_LOG_RETURN.to_str()] / vol
    perf_table[PerfStat.SHARPE_AVG.to_str()] = perf_table[PerfStat.AVG_AN_RETURN.to_str()] / vol  # computed in risk
    perf_table[PerfStat.SHARPE_EXCESS.to_str()] = perf_table[PerfStat.PA_EXCESS_RETURN.to_str()] / vol

    # merge the two meta on dates
    ra_perf_table = pd.merge(left=perf_table, right=risk_table,
                             left_index=True, right_index=True, how='inner',
                             suffixes=(None, "_y"))
    return ra_perf_table


def compute_ra_perf_table_with_benchmark(prices: pd.DataFrame,
                                         benchmark: str,
                                         perf_params: PerfParams = None,
                                         is_log_returns: bool = False,
                                         **kwargs
                                         ) -> pd.DataFrame:
    assert benchmark in prices.columns
    if perf_params is None:
        perf_params = PerfParams(freq=pd.infer_freq(prices.index))

    ra_perf_table = compute_ra_perf_table(prices=prices, perf_params=perf_params)

    returns = ret.to_returns(prices=prices, freq=perf_params.freq_reg, is_log_returns=is_log_returns)
    alphas, betas, r2 = {}, {}, {}
    for column in returns.columns:
        joint_data = returns[[benchmark, column]].dropna()
        if joint_data.empty or len(joint_data.index) < 2:
            alphas[column], betas[column], r2[column] = np.nan, np.nan, np.nan
        else:
            alphas[column], betas[column], r2[column] = ols.estimate_ols_alpha_beta(x=joint_data.iloc[:, 0],
                                                                                    y=joint_data.iloc[:, 1])

            # get vol and compute risk adjusted performance
    ra_perf_table[PerfStat.ALPHA.to_str()] = pd.Series(alphas)
    ra_perf_table[PerfStat.ALPHA_AN.to_str()] = pd.Series(alphas)
    ra_perf_table[PerfStat.BETA.to_str()] = pd.Series(betas)
    ra_perf_table[PerfStat.R2.to_str()] = pd.Series(r2)

    return ra_perf_table


def compute_desc_freq_table(df: pd.DataFrame,
                            freq: str = 'A',
                            agg_func: Callable = np.sum
                            ) -> pd.DataFrame:
    """
    for time series pandas, aggregate by freq and produce descriprive data
    """
    freq_data = df.resample(freq).agg(agg_func)

    # drop na rows for all
    freq_data = freq_data.dropna(axis=0, how='any')

    data_values = freq_data.to_numpy()
    data_table = pd.DataFrame(index=freq_data.columns)
    data_table[PerfStat.AVG.to_str()] = np.nanmean(data_values, axis=0)
    data_table[PerfStat.STD.to_str()] = np.nanstd(data_values, ddof=1, axis=0)
    data_table[PerfStat.QUANT_M_1STD.to_str()] = np.nanquantile(data_values, q=0.16, axis=0)
    data_table[PerfStat.MEDIAN.to_str()] = np.mean(data_values, axis=0)
    data_table[PerfStat.QUANT_P1_STD.to_str()] = np.nanquantile(data_values, q=0.84, axis=0)

    return data_table


def compute_te_ir_errors(return_diffs: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    compute information ratio from return diffs
    """
    vol_dt = np.sqrt(da.infer_an_from_data(return_diffs))
    avg = np.nanmean(return_diffs, axis=0)
    vol = np.nanstd(return_diffs, axis=0, ddof=1)
    ir = vol_dt * np.divide(avg, vol, where=np.greater(vol, 0.0))
    te = pd.Series(vol_dt * vol, index=return_diffs.columns, name=PerfStat.TE.to_str())
    ir = pd.Series(ir, index=return_diffs.columns, name=PerfStat.IR.to_str())
    return te, ir


def compute_info_ratio_table(return_diffs_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    compute information ratio from return diffs
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


def compute_drawdown(price: pd.Series) -> pd.Series:
    if not isinstance(price, pd.Series):
        print(price)
        raise TypeError(f"in compute_max_dd: path_data must be series")

    max_dd = np.zeros_like(price)
    last_peak = -np.inf

    for idx, price_ in enumerate(price):
        if not np.isnan(price_):
            if price_ > last_peak:
                last_peak = price_
            if not np.isclose(last_peak, 0.0):
                max_dd[idx] = price_ / last_peak - 1.0
        else:
            max_dd[idx] = np.nan

    max_dd = pd.Series(data=max_dd, index=price.index, name=price.name)

    return max_dd


def compute_time_under_water(price: pd.Series) -> Tuple[pd.Series, pd.Series]:
    if not isinstance(price, pd.Series):
        raise TypeError(f"in compute_time_under_water: path_data must be series")

    max_dd = np.zeros_like(price)
    dd_times = np.zeros_like(price.index)

    last_peak = -np.inf
    last_dd, last_dd_time = 0.0, price.index[0]
    dd_times[0] = last_dd_time
    for idx, (index, price_) in enumerate(price.items()):
        if not np.isnan(price_):
            if price_ > last_peak:
                last_peak = price_
                last_dd_time = index
            last_dd = price_ / last_peak - 1.0
            max_dd[idx] = last_dd
            dd_times[idx] = last_dd_time
        else:
            max_dd[idx] = last_dd
            dd_times[idx] = last_dd_time

    max_dd = pd.Series(data=max_dd, index=price.index, name=price.name)
    dd_times = (price.index - pd.DatetimeIndex(dd_times)).days
    time_under_water = pd.Series(data=dd_times, index=price.index, name=price.name)

    return max_dd, time_under_water


def compute_drawdown_data(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    if isinstance(prices, pd.Series):
        drawdown = compute_drawdown(price=prices)
    elif isinstance(prices, pd.DataFrame):
        if len(prices.columns) > 1 and prices.columns.duplicated().any():
            raise ValueError(f"dublicated columns = {prices[prices.columns.duplicated()]}")
        drawdowns = []
        for asset_ in prices:
            drawdowns.append(compute_drawdown(price=prices[asset_]))
        drawdown = pd.concat(drawdowns, axis=1)
    else:
        raise ValueError(f"unsuported type {type(prices)}")
    return drawdown


def compute_drawdown_time_under_water(prices: Union[pd.DataFrame, pd.Series]
                                      ) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
    """
    compute joint data of drawdown and time under water
    """
    if isinstance(prices, pd.Series):
        drawdown, time_under_water = compute_time_under_water(price=prices)
    else:
        drawdowns = []
        time_under_waters = []
        for asset in prices.columns:
            drawdown, time_under_water = compute_time_under_water(price=prices[asset])
            drawdowns.append(drawdown)
            time_under_waters.append(time_under_water)
        drawdown = pd.concat(drawdowns, axis=1)
        time_under_water = pd.concat(time_under_waters, axis=1)

    return drawdown, time_under_water


def compute_max_dd(prices: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
    """
    compute realized max drawdown
    """
    max_dd_data = compute_drawdown_data(prices=prices)
    max_dds = np.min(max_dd_data.to_numpy(), axis=0)
    return max_dds


def compute_avg_max_dd(ds: pd.Series,
                       is_max: bool = True,
                       q: float = 0.1
                       ) -> (float, float, float, float):
    """
    compute dd statistics
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
    """
    compute drawdown statistics table
    split drawdown time series into block
    rank block by max dradown and compute stats for the n-top blocks
    """
    if freq is not None:
        price = price.asfreq(freq, method='ffill')
    max_dd, time_under_water = compute_time_under_water(price=price)
    max_dd = max_dd.replace({0.0: np.nan})
    time_under_water = time_under_water.replace({0.0: np.nan})

    # Convert to sparse then query index to find block locations
    joint = pd.concat([max_dd.rename('max_dd'), time_under_water.rename('days'), price], axis=1)

    def process_bslice(bslice: pd.DataFrame) -> Dict[str, Any]:
        max_idx = np.argmin(bslice['max_dd'].to_numpy())
        out = dict(start=bslice.index[0],
                   trough=bslice.index[max_idx],
                   end=bslice.index[-1],
                   max_dd=bslice['max_dd'].iloc[max_idx],
                   days_dd=bslice['days'].iloc[-1],
                   days_to_trough=bslice['days'].iloc[max_idx],
                   days_recovery=bslice['days'].iloc[-1]-bslice['days'].iloc[max_idx],
                   peak=bslice[price.name].iloc[0],
                   bottom=bslice[price.name].iloc[max_idx],
                   recovery=bslice[price.name].iloc[-1])
        return out

    # split max_dd to series according to nan blocs
    sparse_ts = max_dd.astype(pd.SparseDtype('float'))
    # we need to use .values.sp_index.to_block_index() in this version of pandas
    outputs = {}
    for bstart, blength in zip(sparse_ts.values.sp_index.to_block_index().blocs,
                              sparse_ts.values.sp_index.to_block_index().blengths):
        bslice = joint.iloc[bstart: (bstart + blength - 1)]
        if not bslice.empty:
            outputs[bstart] = process_bslice(bslice=bslice)

    df = pd.DataFrame.from_dict(outputs, orient='index')
    df = df.sort_values(by='max_dd')
    if max_num is not None:
        df = df.iloc[:max_num, :]
    return df


class UnitTests(Enum):
    RA_PERF_TABLE = 1
    DRAWDOWN = 2
    DRAWDOWN_STATS_TABLE = 3


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    if unit_test == UnitTests.RA_PERF_TABLE:
        perf_params = PerfParams(freq='B')
        table = compute_ra_perf_table(prices=prices, perf_params=perf_params)
        print(table)
        print(table.columns)

    elif unit_test == UnitTests.DRAWDOWN:
        dd_data = compute_drawdown(price=prices['SPY'])
        print(dd_data)

    elif unit_test == UnitTests.DRAWDOWN_STATS_TABLE:
        df = compute_drawdowns_stats_table(price=prices['SPY'])
        print(df)


if __name__ == '__main__':

    unit_test = UnitTests.DRAWDOWN_STATS_TABLE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

