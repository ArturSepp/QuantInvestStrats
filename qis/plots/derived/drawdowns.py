# packages
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional
from enum import Enum

# qis
import qis.plots.utils as put
import qis.plots.time_series as pts
import qis.perfstats.perf_stats as pt


class DdLegendType(Enum):
    NONE = 1
    SIMPLE = 2
    DETAILED = 3


def plot_rolling_drawdowns(prices: Union[pd.Series, pd.DataFrame],
                           title: Optional[str] = None,
                           var_format: str = '{:.0%}',
                           dd_legend_type: DdLegendType = DdLegendType.DETAILED,
                           legend_loc: str = 'lower left',
                           y_limits: Tuple[Optional[float], Optional[float]] = (None, 0.0),
                           ax: plt.Subplot = None,
                           **kwargs
                           ) -> plt.Figure:

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    max_dd_data = pt.compute_rolling_drawdowns(prices=prices)

    if dd_legend_type == DdLegendType.NONE:
        legend_loc = None
        legend_labels = None
    else:
        legend_labels = []
        for column in max_dd_data.columns:
            avg, quant, nmax, last = pt.compute_avg_max_dd(ds=max_dd_data[column], is_max=False)
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

    max_dd_data, time_under_water = pt.compute_rolling_drawdown_time_under_water(prices=prices)

    legend_labels = []
    for column in time_under_water.columns:
        avg, quant, nmax, last = pt.compute_avg_max_dd(ds=time_under_water[column], is_max=True)
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


def plot_top_drawdowns_paths(price: pd.Series,
                             freq: Optional[str] = 'D',
                             max_num: int = 10,
                             date_format: str = '%d%b%Y',
                             title: Union[str, None] = None,
                             var_format: str = '{:.0%}',
                             legend_loc: str = 'lower left',
                             highlight_ongoing: bool = False,
                             x_limits: Tuple[Optional[float], Optional[float]] = (0.0, None),
                             y_limits: Tuple[Optional[float], Optional[float]] = (None, 0.0),
                             ax: plt.Subplot = None,
                             **kwargs
                             ) -> plt.Figure:

    if freq is not None:
        price = price.asfreq(freq, method='ffill')  # it will have nans
    df = pt.compute_drawdowns_stats_table(price=price, max_num=max_num)
    price_slices = {}
    points = {}
    for start, trough, end, max_dd, peak, days_dd in zip(df['start'], df['trough'], df['end'], df['max_dd'], df['peak'],
                                                         df['days_dd']):
        name = f"{start:{date_format}}-{end:{date_format}}: max_dd={max_dd:0.0%}, days_dd={days_dd:0.0f}"
        price_slices[name] = (price.loc[start:end] / peak - 1.0).reset_index(drop=True)
        points[name] = {trough: max_dd}
    price_slices = pd.DataFrame.from_dict(price_slices, orient='columns')
    # price_slices = price_slices.dropna(axis=0, how='all')  # drop rows with all nans = business days

    n = len(price_slices.columns)
    colors, linestyles = put.get_n_colors(n=n), None
    if highlight_ongoing:
        linestyles = ['dotted'] * n
        last_time = price.index[-2]  # shouldbe one tick back
        for idx, end in enumerate(df['end']):
            if end == last_time:
                dd_slice = price_slices.columns[idx]
                name = f"{dd_slice}-ongoing"
                colors[idx] = 'black'
                linestyles[idx] = 'solid'
                price_slices = price_slices.rename({dd_slice: name}, axis=1)
                break

    fig = pts.plot_time_series(df=price_slices,
                               var_format=var_format,
                               legend_loc=legend_loc,
                               linestyles=linestyles,
                               legend_stats=pts.LegendStats.NONE,
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
    DRAWDOWN_TS = 1
    ROLLING_TIME = 2
    PLOT_TOP_DRAWDOWNS = 3


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data() # .dropna()

    if unit_test == UnitTests.DRAWDOWN_TS:
        plot_rolling_drawdowns(prices=prices)

    elif unit_test == UnitTests.ROLLING_TIME:
        plot_rolling_time_under_water(prices=prices)

    elif unit_test == UnitTests.PLOT_TOP_DRAWDOWNS:
        # plot_top_drawdowns_ts(price=prices['TLT'], freq='D')
        plot_top_drawdowns_paths(price=prices['TLT'], highlight_ongoing=True, freq='D')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.DRAWDOWN_TS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
