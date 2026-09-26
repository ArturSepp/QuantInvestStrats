"""
drawdown panels: the running loss from the prior peak, and how long it lasted.

``plot_rolling_drawdowns`` draws p_t / max_{s<=t} p_s - 1 through time, a series that is
non-positive and whose axis is therefore capped at zero by default.
``plot_rolling_time_under_water`` draws the consecutive calendar days spent below the prior peak,
on levels rebased to calendar days, and ``plot_top_drawdowns_paths`` overlays the deepest
episodes re-indexed to grid steps since their own start (calendar days on its default 'D'
grid), so episodes of different dates are compared on one horizontal axis; with
``highlight_ongoing`` the episode still under water at the last date is drawn solid.

``DdLegendType`` selects what the legend reports - nothing, the extreme and the last value, or
the mean and the 10% quantile as well - all from ``compute_avg_max_dd``. The drawdown series
themselves are computed by ``compute_rolling_drawdowns`` in ``qis/perfstats/perf_stats.py``,
which is where a number quoted in a table comes from; arguments in ``qis/docs/plotting_kwargs.md``.
"""
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
                legend_labels.append(f"{column}, max dd={var_format.format(nmax)}, last={var_format.format(last)}")
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
                                  dd_legend_type: DdLegendType = DdLegendType.SIMPLE,
                                  var_format: str = '{:,.0f}',
                                  y_limits: Tuple[Optional[float], Optional[float]] = (0.0, None),
                                  legend_loc: str = 'lower left',
                                  ax: plt.Subplot = None,
                                  **kwargs
                                  ) -> plt.Figure:
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    max_dd_data, time_under_water = pt.compute_rolling_drawdown_time_under_water(prices=prices)

    if dd_legend_type == DdLegendType.NONE:
        legend_loc = None
        legend_labels = None
    else:
        legend_labels = []
        for column in max_dd_data.columns:
            avg, quant, nmax, last = pt.compute_avg_max_dd(ds=time_under_water[column], is_max=True)
            if dd_legend_type == DdLegendType.SIMPLE:
                legend_labels.append(f"{column}, max={var_format.format(nmax)}, last={var_format.format(last)}")
            elif dd_legend_type == DdLegendType.DETAILED:
                legend_labels.append(f"{column}, mean={var_format.format(avg)}, "
                                     f"quantile_10%={var_format.format(quant)}, max={var_format.format(nmax)},"
                                     f" last={var_format.format(last)}")
            else:
                raise NotImplementedError(f"{dd_legend_type}")

    fig = pts.plot_time_series(df=time_under_water,
                               var_format=var_format,
                               legend_labels=legend_labels,
                               legend_loc=legend_loc,
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
    """
    overlay the deepest drawdown episodes, each re-indexed to the number of grid steps since its
    start.

    The episodes come from ``compute_drawdowns_stats_table`` on the same ``freq`` grid as the
    plotted paths. Each path is price / peak - 1 from the episode's start to its end, plotted
    against the number of ``freq`` observations since the start: calendar days on the default
    'D' grid, native observations with ``freq=None``. The legend's ``days_dd`` is the episode
    table's duration: calendar days for any non-None ``freq``, observations for None.

    Args:
        price: level series of one asset
        freq: grid to rebase the levels to before finding episodes; None keeps the native grid
        max_num: number of deepest episodes to draw
        date_format: format of the start and end dates in the legend
        highlight_ongoing: draw the episode that is still under water at the last date
            (``is_recovered=False``) solid black and the others dotted

    Returns:
        the figure
    """
    if freq is not None:
        price = price.asfreq(freq, method='ffill')  # it will have nans
    # the episode table uses the plotted grid, so its dates and durations match the paths
    df = pt.compute_drawdowns_stats_table(price=price, max_num=max_num, freq=freq)
    price_slices = {}
    points = {}
    for start, trough, end, max_dd, peak, days_dd in zip(df['start'], df['trough'], df['end'], df['max_dd'], df['peak'],
                                                         df['days_dd']):
        name = f"{start:{date_format}}-{end:{date_format}}: max_dd={max_dd:0.0%}, days_dd={days_dd:0.0f}"
        price_slices[name] = (price.loc[start:end] / peak - 1.0).reset_index(drop=True)
        points[name] = {trough: max_dd}
    price_slices = pd.DataFrame.from_dict(price_slices, orient='columns')

    n = len(price_slices.columns)
    colors, linestyles = put.get_n_colors(n=n), None
    if highlight_ongoing:
        linestyles = ['dotted'] * n
        # an episode is ongoing when it has not recovered by the last observation; its end is
        # then the final grid date
        for idx, is_recovered in enumerate(df['is_recovered']):
            if not is_recovered:
                dd_slice = price_slices.columns[idx]
                name = f"{dd_slice}-ongoing"
                colors[idx] = 'black'
                linestyles[idx] = 'solid'
                price_slices = price_slices.rename({dd_slice: name}, axis=1)
                break

    xlabel = 'Days in drawdown' if freq == 'D' else 'Observations in drawdown'
    fig = pts.plot_time_series(df=price_slices,
                               var_format=var_format,
                               legend_loc=legend_loc,
                               linestyles=linestyles,
                               legend_stats=pts.LegendStats.NONE,
                               x_limits=x_limits,
                               y_limits=y_limits,
                               xlabel=xlabel,
                               ylabel='% performance from the last peak',
                               title=title,
                               colors=colors,
                               ax=ax,
                               **kwargs)
    return fig
