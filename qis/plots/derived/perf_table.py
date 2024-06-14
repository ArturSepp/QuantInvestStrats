# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional, Dict, Union
from enum import Enum

# qis
import qis.utils.dates as da
import qis.utils.df_str as dfs
import qis.utils.struct_ops as sop
import qis.perfstats.returns as ret
from qis.perfstats.config import PerfStat, PerfParams
import qis.perfstats.perf_stats as rpt
import qis.plots.scatter as psc
import qis.plots.table as ptb
import qis.plots.utils as put
from qis.plots.bars import plot_bars, plot_vbars


def get_ra_perf_columns(prices: Union[pd.DataFrame, pd.Series],
                        perf_params: PerfParams = None,
                        perf_columns: List[PerfStat] = rpt.STANDARD_TABLE_COLUMNS,
                        column_header: str = 'Asset',
                        df_to_add: pd.DataFrame = None,
                        is_to_str: bool = True,
                        **kwargs
                        ) -> pd.DataFrame:
    """
    compute ra perf table and get ra performance columns with data as string for tables
    """
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    ra_perf_table = rpt.compute_ra_perf_table(prices=prices, perf_params=perf_params)
    data = pd.DataFrame(index=ra_perf_table.index)
    for perf_column in perf_columns:
        if is_to_str:
            data[perf_column.to_str()] = dfs.series_to_str(ds=ra_perf_table[perf_column.to_str()],
                                                           var_format=perf_column.to_format(**kwargs))
        else:
            data[perf_column.to_str()] = ra_perf_table[perf_column.to_str()]

    if df_to_add is not None:
        for idx, column in enumerate(df_to_add.columns):
            data.insert(idx+1, column=column, value=df_to_add[column].to_numpy())
    data.index.name = column_header
    return data


def plot_ra_perf_table(prices: Union[pd.DataFrame, pd.Series],
                       perf_params: PerfParams = None,
                       perf_columns: List[PerfStat] = rpt.STANDARD_TABLE_COLUMNS,
                       special_columns_colors: List[Tuple[int, str]] = ((0, 'blue'),),
                       rows_edge_lines: List[int] = None,
                       special_rows_colors: List[Tuple[int, str]] = None,
                       column_header: str = 'Asset',
                       fontsize: int = 10,
                       transpose: bool = False,
                       df_to_add: pd.DataFrame = None,
                       ax: plt.Subplot = None,
                       **kwargs
                       ) -> Optional[plt.Figure]:
    """
    plot ra perf table columns
    """
    df = get_ra_perf_columns(prices=prices,
                             perf_params=perf_params,
                             perf_columns=perf_columns,
                             column_header=column_header,
                             df_to_add=df_to_add,
                             **kwargs)
    fig = ptb.plot_df_table(df=df,
                            add_index_as_column=True,
                            transpose=transpose,
                            special_columns_colors=special_columns_colors,
                            special_rows_colors=special_rows_colors,
                            rows_edge_lines=rows_edge_lines,
                            fontsize=fontsize,
                            ax=ax,
                            **kwargs)
    return fig


def get_ra_perf_benchmark_columns(prices: pd.DataFrame,
                                  benchmark: str,
                                  drop_benchmark: bool = False,
                                  perf_params: PerfParams = None,
                                  perf_columns: List[PerfStat] = rpt.BENCHMARK_TABLE_COLUMNS,
                                  column_header: str = 'Asset',
                                  is_convert_to_str: bool = True,
                                  **kwargs
                                  ) -> pd.DataFrame:
    """
    compute ra perf table and get ra performance columns with data as string for tables
    """
    ra_perf_table = rpt.compute_ra_perf_table_with_benchmark(prices=prices,
                                                             benchmark=benchmark,
                                                             perf_params=perf_params,
                                                             **kwargs)
    df = pd.DataFrame(index=ra_perf_table.index)
    for perf_column in perf_columns:
        if is_convert_to_str:
            # here we can shorten the performance var for outputs
            df[perf_column.to_str(**kwargs)] = dfs.series_to_str(ds=ra_perf_table[perf_column.to_str()],
                                                                 var_format=perf_column.to_format(**kwargs))
        else:
            df[perf_column.to_str(**kwargs)] = ra_perf_table[perf_column.to_str()]

    if drop_benchmark:
        df = df.drop(benchmark, axis=0)
    df.index.name = column_header
    return df


def plot_ra_perf_table_benchmark(prices: pd.DataFrame,
                                 benchmark: str,
                                 drop_benchmark: bool = False,
                                 perf_params: PerfParams = None,
                                 perf_columns: List[PerfStat] = rpt.BENCHMARK_TABLE_COLUMNS,
                                 special_columns_colors: List[Tuple[int, str]] = ((0, 'skyblue'),),
                                 column_header: str = 'Asset',
                                 fontsize: int = 10,
                                 transpose: bool = False,
                                 alpha_an_factor: float = None,
                                 is_fig_out: bool = True,
                                 ax: plt.Subplot = None,
                                 **kwargs
                                 ) -> Union[plt.Figure, pd.DataFrame]:
    """
    plot ra perf table and get ra performance columns with data as string for tables
    """
    ra_perf_table = get_ra_perf_benchmark_columns(prices=prices,
                                                  benchmark=benchmark,
                                                  drop_benchmark=drop_benchmark,
                                                  perf_params=perf_params,
                                                  perf_columns=perf_columns,
                                                  column_header=column_header,
                                                  alpha_an_factor=alpha_an_factor,
                                                  **kwargs)
    if is_fig_out:
        if not drop_benchmark:
            special_rows_colors = [(1, 'skyblue')]  # for benchmarl separation
            kwargs = sop.update_kwargs(kwargs, dict(special_rows_colors=special_rows_colors))
        return ptb.plot_df_table(df=ra_perf_table,
                                 transpose=transpose,
                                 special_columns_colors=special_columns_colors,
                                 fontsize=fontsize,
                                 ax=ax,
                                 **kwargs)
    else:
        return ra_perf_table


def plot_ra_perf_bars(prices: pd.DataFrame,
                      benchmark: Optional[str] = None,
                      drop_benchmark: bool = False,
                      perf_column: PerfStat = PerfStat.SHARPE_RF0,
                      perf_params: PerfParams = None,
                      legend_loc: Optional[str] = None,
                      ax: plt.Subplot = None,
                      **kwargs
                      ) -> plt.Figure:

    if benchmark is None:
        ra_perf_table = rpt.compute_ra_perf_table(prices=prices, perf_params=perf_params)
    else:
        ra_perf_table = get_ra_perf_benchmark_columns(prices=prices,
                                                      benchmark=benchmark,
                                                      drop_benchmark=drop_benchmark,
                                                      perf_params=perf_params,
                                                      is_convert_to_str=False,
                                                      **kwargs)

    df = ra_perf_table[perf_column.to_str()].to_frame()
    colors = put.compute_heatmap_colors(a=df.to_numpy())
    fig = plot_vbars(df=df,
                     var_format=perf_column.to_format(**kwargs),
                     legend_loc=legend_loc,
                     colors=colors,
                     is_category_names_colors=False,
                     ax=ax,
                     **kwargs)
    return fig


def plot_ra_perf_scatter(prices: pd.DataFrame,
                         benchmark: str = None,
                         perf_params: PerfParams = None,
                         x_var: PerfStat = PerfStat.MAX_DD,
                         y_var: PerfStat = PerfStat.PA_RETURN,
                         hue_data: pd.Series = None,
                         yvar_format: str = None,
                         x_filters: Tuple[Optional[float], Optional[float]] = None,
                         order: int = 1,
                         add_universe_model_label: bool = True,
                         ci: Optional[int] = 95,
                         ax: plt.Subplot = None,
                         **kwargs
                         ) -> plt.Figure:
    """
    scatter plot of performance stats
    """
    if benchmark is not None:
        ra_perf_table = rpt.compute_ra_perf_table_with_benchmark(prices=prices,
                                                                 benchmark=benchmark,
                                                                 perf_params=perf_params,
                                                                 **kwargs)
    else:
        ra_perf_table = rpt.compute_ra_perf_table(prices=prices, perf_params=perf_params)

    xy = ra_perf_table[[x_var.to_str(), y_var.to_str()]]
    if x_filters is not None:
        if x_filters[0] is not None:
            xy = xy.iloc[xy[x_var.to_str()].to_numpy()>x_filters[0], :]
        if x_filters[1] is not None:
            xy = xy.iloc[xy[x_var.to_str()].to_numpy()<x_filters[1], :]

    # fu.save_df_to_excel(xy, file_name='xy')
    if hue_data is not None:
        xy = pd.concat([xy, hue_data], axis=1)
        hue = hue_data.name
    else:
        hue = None
    fig = psc.plot_scatter(df=xy,
                           hue=hue,
                           xvar_format=x_var.to_format(**kwargs),
                           yvar_format=yvar_format or y_var.to_format(**kwargs),
                           add_universe_model_label=add_universe_model_label,
                           order=order,
                           ci=ci,
                           ax=ax,
                           **kwargs)

    return fig


def plot_ra_perf_by_dates(prices: pd.DataFrame,
                          time_period_dict: Dict[str, da.TimePeriod],
                          perf_column: PerfStat = PerfStat.SHARPE_RF0,
                          perf_params: PerfParams = None,
                          fontsize: int = 12,
                          title: str = None,
                          ax: plt.Subplot = None,
                          **kwargs
                          ) -> plt.Figure:
    """
    compute table of performance var for a dict of strat end periods
    """
    datas = []
    for key, time_period in time_period_dict.items():
        prices_ = time_period.locate(prices)
        ra_perf_table = rpt.compute_ra_perf_table(prices=prices_,
                                                  perf_params=perf_params)
        this = dfs.series_to_str(ra_perf_table[perf_column.to_str()], var_format=perf_column.to_format(**kwargs))
        datas.append(this.rename(key))
    df = pd.concat(datas, axis=1)

    fig = ptb.plot_df_table(df=df,
                            add_index_as_column=True,
                            fontsize=fontsize,
                            title=title,
                            ax=ax,
                            **kwargs)
    return fig


def plot_ra_perf_annual_matrix(price: pd.Series,
                               min_lag: int = 1,
                               date_format: str = '%d%b%Y',
                               perf_column: PerfStat = PerfStat.SHARPE_RF0,
                               perf_params: PerfParams = None,
                               fontsize: int = 12,
                               ax: plt.Subplot = None,
                               is_fig_out: bool = True,
                               is_shift_by_day: bool = False,
                               **kwargs
                               ) -> Union[plt.Figure, pd.DataFrame]:
    """
    compute annual matrix of performance var for a dict of strat end periods
    """
    if not isinstance(price, pd.Series):
        raise TypeError(f"must be pd.Series not {type(price)}")

    # extract years
    dates_schedule = da.generate_dates_schedule(time_period=da.get_time_period(df=price),
                                                freq='YE',
                                                include_start_date=True,
                                                include_end_date=True)
    yearly_dfs = {}
    for idx, start in enumerate(dates_schedule[:-min_lag]):
        yearly_df = {}
        for idxx, end in enumerate(dates_schedule[idx+min_lag:]):
            price_ = da.TimePeriod(start, end).locate(price)
            ra_perf_table = rpt.compute_ra_perf_table(prices=price_, perf_params=perf_params)
            #print(f"{start} to {end}")
            # yearly_df[end.strftime(date_format)] = f"{start} to {end}"
            if is_shift_by_day:
                this = da.shift_date_by_day(date=end, backward=False)
            else:
                this = end
            yearly_df[this.strftime(date_format)] = ra_perf_table[perf_column.to_str()].iloc[0]

        yearly_dfs[start.strftime(date_format)] = pd.Series(yearly_df)
    yearly_dfs = pd.DataFrame.from_dict(yearly_dfs, orient='index')

    if is_fig_out:
        data_colors = put.compute_heatmap_colors(a=yearly_dfs.to_numpy(), **kwargs)
        df = dfs.df_to_str(yearly_dfs, var_format=perf_column.to_format(**kwargs))
        fig = ptb.plot_df_table(df=df,
                                add_index_as_column=True,
                                index_column_name='Start \ End',
                                fontsize=fontsize,
                                data_colors=data_colors,
                                ax=ax,
                                **kwargs)
        return fig
    else:
        return yearly_dfs


def plot_desc_freq_table(df: pd.DataFrame,
                         freq: str = 'YE',
                         agg_func: Callable = np.sum,
                         var_format: str = '{:.2f}',
                         special_columns_colors: List[Tuple[int, str]] = None,
                         special_rows_colors: List[Tuple[int, str]] = None,
                         column_header: str = 'Variable',
                         fontsize: int = 12,
                         title: Optional[str] = None,
                         ax: plt.Subplot = None,
                         **kwargs
                         ) -> plt.Figure:

    data_table = rpt.compute_desc_freq_table(df=df,
                                             freq=freq,
                                             agg_func=agg_func)
    plot_data = pd.DataFrame(data=data_table.index.T, index=data_table.index, columns=[column_header])

    for column in data_table:
        plot_data[column] = data_table[column].apply(lambda x: var_format.format(x))

    fig = ptb.plot_df_table(df=plot_data,
                            special_columns_colors=special_columns_colors,
                            special_rows_colors=special_rows_colors,
                            fontsize=fontsize,
                            title=title,
                            ax=ax,
                            **kwargs)
    return fig


def plot_top_bottom_performers(prices: pd.DataFrame,
                               yvar_format: str = '{:.0%}',
                               num_assets: int = None,
                               ax: plt.Subplot = None,
                               **kwargs
                               ) -> Optional[plt.Subplot]:
    returns = ret.to_total_returns(prices=prices).sort_values().dropna()

    if num_assets is not None and len(returns.index) > 2*num_assets:
        returns = pd.concat([returns.iloc[:num_assets], returns.iloc[-num_assets:]])

    fig = plot_bars(df=returns,
                    stacked=False,
                    skip_y_axis=True,
                    legend_loc=None,
                    x_rotation=90,
                    yvar_format=yvar_format,
                    ax=ax,
                    **kwargs)
    return fig


class UnitTests(Enum):
    PLOT_RA_PERF_TABLE = 1
    PLOT_RA_PERF_SCATTER = 2
    PLOT_RA_PERF_TABLE_BENCHMARKS = 3
    PLOT_DESC_FREQ_TABLE = 4
    PLOT_SHARPE_BARPLOT = 5
    PLOT_SHARPE_BY_DATES = 6
    PLOT_PERF_FOR_START_END_PERIOD = 7
    PLOT_TOP_BOTTOM_PERFORMERS = 8


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()
    print(prices)

    if unit_test == UnitTests.PLOT_RA_PERF_TABLE:

        perf_params = PerfParams(freq='B')
        prices = prices.iloc[:, :5]
        plot_ra_perf_table(prices=prices,
                           perf_columns=rpt.COMPACT_TABLE_COLUMNS,
                           perf_params=perf_params)

    elif unit_test == UnitTests.PLOT_RA_PERF_SCATTER:

        perf_params = PerfParams(freq='B')
        plot_ra_perf_scatter(prices=prices,
                             perf_params=perf_params)

    elif unit_test == UnitTests.PLOT_RA_PERF_TABLE_BENCHMARKS:
        perf_params = PerfParams(freq='ME')
        plot_ra_perf_table_benchmark(prices=prices,
                                     benchmark='SPY',
                                     perf_params=perf_params,
                                     transpose=False)

    elif unit_test == UnitTests.PLOT_DESC_FREQ_TABLE:
        freq_data = plot_desc_freq_table(df=prices,
                                         freq='YE',
                                         agg_func=np.mean)
        print(freq_data)

    elif unit_test == UnitTests.PLOT_SHARPE_BARPLOT:
        plot_ra_perf_bars(prices=prices, perf_column=PerfStat.MAX_DD)

    elif unit_test == UnitTests.PLOT_SHARPE_BY_DATES:
        prices = prices

        time_period_dict = {'1y': da.TimePeriod(start='30Jun2019', end='30Jun2020'),
                            '3y': da.TimePeriod(start='30Jun2017', end='30Jun2020'),
                            '5y': da.TimePeriod(start='30Jun2015', end='30Jun2020')}
        plot_ra_perf_by_dates(prices=prices,
                              time_period_dict=time_period_dict)

    elif unit_test == UnitTests.PLOT_PERF_FOR_START_END_PERIOD:
        plot_ra_perf_annual_matrix(price=prices.iloc[:, 0])

    elif unit_test == UnitTests.PLOT_TOP_BOTTOM_PERFORMERS:
        plot_top_bottom_performers(prices=prices, num_assets=2)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_RA_PERF_TABLE_BENCHMARKS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
