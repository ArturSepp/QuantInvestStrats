# packages
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
from typing import Union, Dict, Tuple, Optional
from enum import Enum
from qis import PerfParams

DATE_FORMAT = '%d%b%Y'  # 31Jan2020 - common across all reporting
WEEK_DAYS_PER_YEAR = 260  # calendar days excluding weekends in a year

PERF_PARAMS = PerfParams(freq_reg='B', freq_vol='B', freq_drawdown='B')


class ReportType(Enum):
    SingleTimeSeries = 1  # ax = 1
    SingleTimeSeriesWithPDF = 2  # ax = 2
    WithInSampleTimeSeries = 3  # ax = 2
    WithInSampleTimeSeriesPDF = 4  # ax = 3


def set_x_date_freq(data: Union[pd.Series, pd.DataFrame],
                    kwargs: Dict
                    ) -> Dict:
    if len(data.index) / WEEK_DAYS_PER_YEAR > 30:  # increase freq
        local_kwargs = kwargs.copy()
        local_kwargs.update({'x_date_freq': '2YE'})
    elif len(data.index) / WEEK_DAYS_PER_YEAR < 1.0:  # increase freq
        local_kwargs = kwargs.copy()
        local_kwargs.update({'x_date_freq': 'ME'})
    else:
        local_kwargs = kwargs
    return local_kwargs


def get_summary_table_fig(data_start_dates: Union[pd.Series, pd.DataFrame],
                          descriptive_data: Union[pd.Series, pd.DataFrame] = None,
                          figsize: Tuple[float, Optional[float]] = (3.7, None),
                          first_column_name: str = 'Instrument',
                          **kwargs
                          ) -> plt.Subplot:

    if isinstance(data_start_dates, pd.Series):
        summary_data = qis.series_to_str(ds=data_start_dates, var_format=DATE_FORMAT).to_frame()
    elif isinstance(data_start_dates, pd.DataFrame):
        summary_data = qis.df_to_str(df=data_start_dates, var_format=DATE_FORMAT)
    else:
        raise TypeError(f"unsupported type {type(data_start_dates)}")

    if descriptive_data is not None:
        summary_data = pd.concat([descriptive_data, summary_data], axis=1)
        index_column_name = first_column_name
        # summary_data = summary_data.reset_index()
        # summary_data.index = np.arange(1, len(summary_data) + 1)
    else:
        index_column_name = first_column_name

    height = qis.calc_table_height(num_rows=len(summary_data.index))
    fig, ax = plt.subplots(1, 1, figsize=(figsize[0], height))
    qis.plot_df_table(df=summary_data,
                      index_column_name=index_column_name,
                      ax=ax,
                      **kwargs)
    return fig
