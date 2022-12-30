# built in
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional
from enum import Enum

# qis
import qis.plots.time_series as pts
import qis.plots.utils as put


def compute_drawdown(prices: pd.Series) -> pd.Series:

    if not isinstance(prices, pd.Series):
        print(prices)
        raise TypeError(f"in compute_max_dd: path_data must be series")

    max_dd = np.zeros_like(prices)
    last_peak = -np.inf

    for idx, price in enumerate(prices):
        if not np.isnan(price):
            if price > last_peak:
                last_peak = price
            if not np.isclose(last_peak, 0.0):
                max_dd[idx] = price / last_peak - 1.0
        else:
            max_dd[idx] = np.nan

    max_dd = pd.Series(data=max_dd, index=prices.index, name=prices.name)

    return max_dd


def compute_time_under_water(prices: pd.Series) -> Tuple[pd.Series, pd.Series]:

    if not isinstance(prices, pd.Series):
        raise TypeError(f"in compute_time_under_water: path_data must be series")

    max_dd = np.zeros_like(prices)
    dd_times = np.zeros_like(prices.index)

    last_peak = -np.inf
    last_dd_time = prices.index[0]
    dd_times[0] = last_dd_time
    for idx, (index, price) in enumerate(prices.items()):
        if not np.isnan(price):
            if price > last_peak:
                last_peak = price
                last_dd_time = index
            max_dd[idx] = price / last_peak - 1.0
            dd_times[idx] = last_dd_time
        else:
            max_dd[idx] = 0.0
            dd_times[idx] = index

    max_dd = pd.Series(data=max_dd, index=prices.index, name=prices.name)
    dd_times = (prices.index - pd.DatetimeIndex(dd_times)).days
    time_under_water = pd.Series(data=dd_times, index=prices.index, name=prices.name)

    return max_dd, time_under_water


def compute_drawdown_data(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    if isinstance(prices, pd.Series):
        drawdown = compute_drawdown(prices=prices)
    elif isinstance(prices, pd.DataFrame):
        if len(prices.columns) > 1 and prices.columns.duplicated().any():
            raise ValueError(f"dublicated columns = {prices[prices.columns.duplicated()]}")
        drawdowns = []
        for asset_ in prices:
            drawdowns.append(compute_drawdown(prices=prices[asset_]))
        drawdown = pd.concat(drawdowns, axis=1)
    else:
        raise ValueError(f"unsuported type {type(prices)}")
    return drawdown


def compute_drawdown_time_data(prices: Union[pd.DataFrame, pd.Series]
                               ) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:

    if isinstance(prices, pd.Series):
        drawdown, time_under_water = compute_time_under_water(prices=prices)

    else:
        drawdowns = []
        time_under_waters = []
        for asset in prices:
            drawdown, time_under_water = compute_time_under_water(prices=prices[asset])
            drawdowns.append(drawdown)
            time_under_waters.append(time_under_water)

        drawdown = pd.concat(drawdowns, axis=1)
        time_under_water = pd.concat(time_under_waters, axis=1)

    return drawdown, time_under_water


def compute_max_dd(prices: Union[pd.DataFrame, pd.Series]
                   ) -> np.ndarray:
    max_dd_data = compute_drawdown_data(prices=prices)
    max_dds = np.min(max_dd_data.to_numpy(), axis=0)
    return max_dds


def compute_avg_max(ds: pd.Series,
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
        quant = np.nanquantile(nan_data, 1.0-q)
        nmax = np.nanmax(nan_data)
    else:
        quant = np.nanquantile(nan_data, q)
        nmax = np.nanmin(nan_data)

    last = ds.iloc[-1]

    return avg, quant, nmax, last


class DdLegendType(Enum):
    NONE = 1
    SIMPLE = 2
    DETAILED = 3


def plot_drawdown(prices: pd.DataFrame,
                  title: Optional[str] = None,
                  var_format: str = '{:.0%}',
                  dd_legend_type: DdLegendType = DdLegendType.DETAILED,
                  legend_loc: str = 'lower left',
                  y_limits: Tuple[Optional[float], Optional[float]] = (None, 0.0),
                  ax: plt.Subplot = None,
                  **kwargs
                  ) -> plt.Figure:

    max_dd_data = compute_drawdown_data(prices=prices)

    if dd_legend_type == DdLegendType.NONE:
        legend_loc = None
        legend_labels = None
    else:
        legend_labels = []
        for column in max_dd_data.columns:
            avg, quant, nmax, last = compute_avg_max(ds=max_dd_data[column], is_max=False)
            if dd_legend_type == DdLegendType.SIMPLE:
                legend_labels.append(f"{column}, max dd={var_format.format(nmax)}")
            elif dd_legend_type == DdLegendType.DETAILED:
                legend_labels.append(f"{column}, mean={var_format.format(avg)},"
                                     f" quantile_10%={var_format.format(quant)}, max={var_format.format(nmax)},"
                                     f" last={var_format.format(last)}")
            else:
                raise NotImplementedError(f"{dd_legend_type}")

    fig = pts.plot_time_series(df=max_dd_data,
                               var_format=var_format,
                               legend_loc=legend_loc,
                               legend_labels=legend_labels,
                               title=title,
                               y_limits=y_limits,
                               ax=ax,
                               **kwargs)
    return fig


def plot_rolling_time_under_water(prices: pd.DataFrame,
                                  title: Union[str, None] = None,
                                  var_format: str = '{:,.0f}',
                                  y_limits: Tuple[Optional[float], Optional[float]] = (0.0, None),
                                  ax: plt.Subplot = None,
                                  **kwargs
                                  ) -> plt.Figure:
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    max_dd_data, time_under_water = compute_drawdown_time_data(prices=prices)

    legend_labels = []
    for column in time_under_water.columns:
        avg, quant, nmax, last = compute_avg_max(ds=time_under_water[column], is_max=True)
        legend_labels.append(f"{column}, mean={var_format.format(avg)}, "
                             f"quantile_10%={var_format.format(quant)}, max={var_format.format(nmax)},"
                             f" last={var_format.format(last)}")

    fig = pts.plot_time_series(df=time_under_water,
                               var_format=var_format,
                               legend_loc='upper left',
                               legend_labels=legend_labels,
                               title=title,
                               y_limits=y_limits,
                               ax=ax,
                               **kwargs)
    return fig


def plot_drawdown_lengths(price: pd.Series,
                          cut: int = 7,
                          date_format: str = '%d%b%Y',
                          title: Union[str, None] = None,
                          var_format: str = '{:.0%}',
                          legend_loc: str = 'upper center',
                          x_limits: Tuple[Optional[float], Optional[float]] = (0.0, None),
                          y_limits: Tuple[Optional[float], Optional[float]] = (None, 0.0),
                          ax: plt.Subplot = None,
                          **kwargs
                          ) -> plt.Figure:
    """
    plot drowdowns with x-being the days in dd
    """
    max_dd, time_under_water = compute_time_under_water(prices=price)

    # compute_run_data
    val0 = 0
    index0 = index00 = max_dd.index[0]
    dds_dict = {}
    for index, val in time_under_water.items():
        if val0 > 0 and val == 0:  # dd finished
            dds_dict[index] = dict(price_run=price[index00:index], dd_lenth=val0, is_current=False)
            index00 = index
        elif val0==0 and val == 0:  # ath
            index00 = index
        else:  # in dd
            pass
        val0 = val
        index0 = index
    # add last index
    if time_under_water[index0] > 0:
        dds_dict[index0] = dict(price_run=price[index00:index0], dd_lenth=time_under_water[index0], is_current=True)

    # sort by lenths
    sorted_dds_dict = dict(sorted(dds_dict.items(), key=lambda item: item[1]['dd_lenth'], reverse=True))

    dd_datas = []
    colors = put.get_n_colors(n=cut)
    linestyles = ['dotted']*cut
    for idx, (key, dd_data) in enumerate(sorted_dds_dict.items()):
        if idx < cut:
            price_run = dd_data['price_run']
            perf = price_run/price_run[0] - 1.0
            name = f"{price_run.index[0].strftime(date_format)}-{price_run.index[-1].strftime(date_format)}," \
                   f" days={dd_data['dd_lenth']}, max={np.nanmin(perf):.0%}"
            if dd_data['is_current']:
                name = f"{name}-ongoing"
                colors[idx] = 'black'
                linestyles[idx] = 'solid'
            dd_datas.append(pd.Series(perf.to_numpy(), name=name))
        else:
            break
    dd_datas = pd.concat(dd_datas, axis=1)
    fig = pts.plot_time_series(df=dd_datas,
                               var_format=var_format,
                               legend_loc=legend_loc,
                               linestyles=linestyles,
                               legend_line_type=pts.LegendLineType.NONE,
                               x_limits=x_limits,
                               y_limits=y_limits,
                               xlabel='Days in drawdown',
                               ylabel='% performance from the last peak',
                               title=title,
                               colors=colors,
                               ax=ax,
                               **kwargs)
    return fig


class UnitTests(Enum):
    DD_DATA = 1
    DRAWDOWN_TS = 2
    ROLLING_TIME = 3
    PLOT_DD_LENTHS = 4


def run_unit_test(unit_test: UnitTests):

    from qis.data.yf_data import load_etf_data
    prices = load_etf_data().dropna()

    if unit_test == UnitTests.DD_DATA:
        prices = prices.iloc[:, 0]
        max_dd, time_under_water = compute_time_under_water(prices=prices)
        print(max_dd)
        print(time_under_water)

    elif unit_test == UnitTests.DRAWDOWN_TS:
        plot_drawdown(prices=prices)

    elif unit_test == UnitTests.ROLLING_TIME:
        plot_rolling_time_under_water(prices=prices)

    elif unit_test == UnitTests.PLOT_DD_LENTHS:
        plot_drawdown_lengths(price=prices.iloc[:, 0])

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.DRAWDOWN_TS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
