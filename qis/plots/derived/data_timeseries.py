"""
useful as interface when data can be either price or level time series
"""
import pandas as pd
from typing import Optional, Union
import matplotlib.pyplot as plt

# qis
import qis.utils.dates as da
from qis.perfstats.config import PerfParams
import qis.plots.time_series as pts
import qis.plots.derived.prices as ppd


PERF_PARAMS = PerfParams(freq_reg='B', freq_vol='B', freq_drawdown='B')


def plot_data_timeseries(data: Union[pd.DataFrame, pd.Series],
                         is_price_data: bool = False,
                         legend_stats: pts.LegendStats = pts.LegendStats.FIRST_AVG_LAST,
                         var_format: Optional[str] = None,
                         time_period: da.TimePeriod = None,
                         title: str = '',
                         title_add_date: bool = True,
                         perf_params: PerfParams = PERF_PARAMS,
                         start_to_one: bool = False,
                         ax: plt.Subplot = None,
                         **kwargs
                         ) -> None:
    """
    define plot time series
    """
    if time_period is not None:
        data = time_period.locate(data)

    if title_add_date:
        title = f"{title} {da.get_time_period(df=data.dropna()).to_str()}"
    else:
        title = title

    if var_format is None:
        if is_price_data:
            var_format = '{:,.2f}'
        else:
            var_format = '{:.2%}'

    if is_price_data:
        ppd.plot_prices(prices=data,
                        perf_params=perf_params,
                        title=title,
                        start_to_one=start_to_one,
                        is_log=False,
                        var_format=var_format,
                        perf_stats_labels=ppd.PerfStatsLabels.DETAILED_WITH_DDVOL.value,
                        ax=ax,
                        **kwargs)
    else:
        pts.plot_time_series(df=data,
                             title=title,
                             legend_stats=legend_stats,
                             var_format=var_format,
                             ax=ax,
                             **kwargs)
