"""
plot aggregate report on pandas data and plot time series of columns
each time series plot = data[column]
 # = # columns
"""
# packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
from enum import Enum
from tqdm import tqdm

# qis
import qis as qis
import qis.utils.dates as da
import qis.utils.df_agg as dfa
import qis.utils.df_groups as dfg
import qis.utils.df_ops as dfo
from qis.perfstats.desc_table import DescTableType
from qis.plots import histogram as pdf
from qis.plots.reports.utils import get_summary_table_fig, set_x_date_freq, ReportType
from qis.plots.derived.data_timeseries import plot_data_timeseries


def econ_data_report(data: pd.DataFrame,
                     complimentary_datas: Dict[str, pd.DataFrame] = None,
                     group_data: pd.Series = None,
                     descriptive_data: pd.DataFrame = None,
                     inst_names: pd.Series = None,
                     feature_name: str = None,
                     report_type: ReportType = ReportType.SingleTimeSeries,
                     is_add_summary_table: bool = True,
                     file_name: str = 'econ_data',
                     var_format: Optional[str] = None,
                     is_price_data: bool = False,
                     desc_table_type: DescTableType = DescTableType.WITH_KURTOSIS,
                     x_min_max_quantiles: Tuple[Optional[float], Optional[float]] = (0.005, 0.995),
                     background: str = 'darkgrid',
                     **kwargs
                     ) -> List[plt.Figure]:
    """
    report for economic data
    """
    local_kwargs = {'fontsize': 6,
                    'linewidth': 0.5,
                    'weight': 'normal',
                    'framealpha': 0.75}
    qis.update_kwargs(local_kwargs, kwargs)

    figsize_table = (8.0, 2.9)  # table for 1 column
    figsize = (14, 10)

    if var_format is None:
        if is_price_data:
            var_format = '{:,.2f}'
        else:
            var_format = '{:.2%}'

    # get summary table
    price_start_dates = {}
    price_end_dates = {}
    insample_prices = {}
    for inst in data.columns:
        insample_price = dfo.drop_first_nan_data(df=data[inst]).dropna()

        if insample_price.empty:
            bbg_start = np.nan
            bbg_finish = np.nan
        else:
            bbg_start = insample_price.index[0]
            bbg_finish = insample_price.index[-1]

        price_start_dates.update({inst: bbg_start})
        price_end_dates.update({inst: bbg_finish})
        insample_prices.update({inst: insample_price})
    price_start_dates = pd.Series(price_start_dates).rename('StartDate')
    price_end_dates = pd.Series(price_end_dates).rename('EndDate')
    price_start_dates = pd.concat([price_start_dates, price_end_dates], axis=1)

    figs = []
    if is_add_summary_table:
        fig = get_summary_table_fig(data_start_dates=price_start_dates,
                                    descriptive_data=descriptive_data,
                                    figsize=figsize_table,
                                    **kwargs)
        figs.append(fig)

    group_means = None
    if group_data is not None:
        group_means = dfg.agg_df_by_groups(df=data,
                                           group_data=group_data,
                                           agg_func=dfa.nanmean,
                                           total_column=None)
    figure_num = 0
    for inst, insample_price in tqdm(insample_prices.items()):
        if np.all(insample_price.isnull()):  # skip for all nans
            continue
        figure_num += 1
        figure_caption = f"Figure {figure_num}. {inst}"
        if inst_names is not None:
            figure_caption += f", {inst_names[inst]}"

        if descriptive_data is not None:
            for k, v in descriptive_data.loc[inst].to_dict().items():
                figure_caption += f", {k}: {v}"

        local_kwargs = set_x_date_freq(data=insample_price, kwargs=kwargs)

        # change names
        if feature_name is None:
            inst_name = f"{inst} "
        else:
            insample_price = insample_price.rename(feature_name)
            inst_name = ""

        if complimentary_datas is not None:
            for name, data in complimentary_datas.items():
                if inst in data.columns:
                    comp_name = f"{inst_name}{name}"
                    insample_price = pd.concat([insample_price, data[inst].rename(comp_name)], axis=1)

        if group_means is not None:
            inst_group = group_data[inst]
            insample_price = pd.concat([insample_price, group_means[inst_group].rename(f"{inst_group} ac avg")], axis=1)

        if report_type == ReportType.SingleTimeSeries:
            with sns.axes_style(background):
                fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
                plot_data_timeseries(data=insample_price,
                                     ax=ax,
                                     title=f"In-sample period = ",
                                     is_price_data=is_price_data,
                                     var_format=var_format,
                                     **local_kwargs)

        elif report_type == ReportType.SingleTimeSeriesWithPDF:
            fig, axs = plt.subplots(1, 2, figsize=figsize, tight_layout=True)
            plot_data_timeseries(data=insample_price,
                                 ax=axs[0],
                                 title=f"In-sample period = ",
                                 is_price_data=is_price_data,
                                 var_format=var_format,
                                 **local_kwargs)

            pdf.plot_histogram(df=insample_price,
                               xvar_format=var_format,
                               x_min_max_quantiles=x_min_max_quantiles,
                               add_last_value=False,
                               desc_table_type=desc_table_type,
                               title=f"Distribution: {da.get_time_period(df=data).to_str()}",
                               bbox_to_anchor=None,
                               ax=axs[1],
                               **kwargs)

        else:
            raise TypeError(f"report_type={report_type} is not implemented")

        qis.set_suptitle(fig, title=f"{file_name}_{figure_num}_{inst}")
        figs.append(fig)

    return figs


class UnitTests(Enum):
    UPDATE_DATA = 1
    RUN_REPORT = 2


def run_unit_test(unit_test: UnitTests):

    import matplotlib.pyplot as plt
    import quant_strats.local_path as lp
    local_path = lp.get_resource_path()

    if unit_test == UnitTests.UPDATE_DATA:
        from bbg_fetch import fetch_field_timeseries_per_tickers
        vix_tickers = ['VIX1D Index', 'VIX9D Index', 'VIX Index', 'VIX3M Index', 'VIX6M Index', 'VIX1Y Index']
        vols = 0.01 * fetch_field_timeseries_per_tickers(tickers=vix_tickers, field='PX_LAST', CshAdjNormal=True)
        print(vols)
        qis.save_df_to_csv(df=vols, file_name='vix_indices', local_path=local_path)

    elif unit_test == UnitTests.RUN_REPORT:
        df = qis.load_df_from_csv(file_name='vix_indices', local_path=local_path)
        figs = econ_data_report(data=df)
        qis.save_figs_to_pdf(figs, file_name='vix_indices', local_path=local_path)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.RUN_REPORT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
