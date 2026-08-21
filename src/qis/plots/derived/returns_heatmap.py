"""
periodic returns laid out as a table of coloured cells rather than a chart.

``plot_returns_heatmap`` is the calendar grid - years down, months or quarters across, built by
``compute_periodic_returns_table`` from one ``pd.Series`` at a time and rejecting a frame. The
many-asset forms put assets on one axis and periods on the other, and which way round is
``transpose``: ``plot_periodic_returns_table`` defaults it to True, ``plot_returns_table``
leaves it False. The latter gives the total return over each window of a ``TimePeriod`` dict, and
``plot_sorted_periodic_returns`` ranks assets within each period, one colour held per name.

Cells are simple period returns from ``to_returns``; the trailing column named by ``ytd_name``
is the return over the whole row period, which compounds the cells rather than summing them.
The colour scale and cell drawing are ``qis/plots/heatmap.py``; the statistics table, with its
Sharpe and drawdown columns, is ``qis/plots/derived/perf_table.py``.
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Union, List, Optional, Tuple, Dict
# qis
import qis.utils.dates as da
import qis.perfstats.returns as ret
import qis.plots.utils as put
import qis.plots.heatmap as phe
import qis.plots.table as ptb
from qis.plots.heatmap import plot_heatmap

MONTH_MAP = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}


def compute_periodic_returns_by_row_table(prices: pd.Series,
                                          heatmap_freq: str = 'YE',
                                          column_period: str = 'ME'
                                          ) -> pd.DataFrame:

    periodic_prices_by_row_period = da.split_df_by_freq(df=prices.to_frame(), freq=heatmap_freq)
    returns_table = None
    for key, prices in periodic_prices_by_row_period.items(): # fill table
        returns = ret.to_returns(prices=prices, freq=column_period, include_start_date=True, include_end_date=True, drop_first=True)
        if column_period in ['ME', 'QE']:
            year_returns_by_period = pd.DataFrame(data=np.column_stack(returns.to_numpy()), columns=[returns.index.month], index=[returns.index[-1]])
        elif 'W' in column_period:  # Weeklies
            year_returns_by_period = pd.DataFrame(data=np.column_stack(returns.to_numpy()), columns=[returns.index.week], index=[returns.index[-1]])
        else:
            raise TypeError(f"column_period={column_period} not implemented")
        returns_table = pd.concat([returns_table, year_returns_by_period], axis=0) if returns_table is not None else year_returns_by_period

    returns_table = returns_table.reindex(sorted(returns_table.columns), axis=1)

    # map months column: concat create multiindex columns
    if column_period == 'ME':

        returns_table.columns = returns_table.columns.get_level_values(0).map(MONTH_MAP)

        if heatmap_freq == 'A-Mar':
            cols = returns_table.columns.tolist()
            cols = cols[3:] + cols[:3]
            returns_table = returns_table[cols]

    elif column_period == 'QE':
        returns_table.columns = returns_table.columns.get_level_values(0).map({1: 'Q1', 2: 'Q1', 3: 'Q1',
                                                                              4: 'Q2', 5: 'Q2', 6: 'Q2',
                                                                              7: 'Q3', 8: 'Q3', 9: 'Q3',
                                                                              10: 'Q4', 11: 'Q4', 12: 'Q4'})
    else:
        print(f"map for column_period={column_period} is not implemented")

    return returns_table


def compute_periodic_returns_table(prices: pd.Series,
                                   heatmap_freq: str = 'YE',
                                   column_period: Optional[str] = 'ME',
                                   row_date_format: str = '%Y',
                                   is_inverse_order: bool = False,
                                   is_add_annual_column: bool = True,
                                   ytd_name: str = 'YTD'
                                   ) -> pd.DataFrame:
    """
    compute table for heatmap: columns are monthly returns, rows are years
    implemented only for single asset with price passed as series
    """
    if not isinstance(prices, pd.Series):
        raise ValueError("prices must be pd.Series")

    if column_period is None:
        # insert returns returns table
        # compute annual returns
        row_periodic_returns = ret.to_returns(prices=prices,
                                              freq=heatmap_freq,
                                              include_start_date=True,
                                              include_end_date=True)
        
        row_periodic_returns = row_periodic_returns.set_index(
            row_periodic_returns.index.strftime(row_date_format).to_numpy())
        periodic_returns_table = row_periodic_returns
    else:
        periodic_returns_table = compute_periodic_returns_by_row_table(prices=prices,
                                                                       heatmap_freq=heatmap_freq,
                                                                       column_period=column_period)
        row_periodic_returns = ret.to_returns(prices=prices, freq=heatmap_freq,
                                              include_start_date=True, include_end_date=True, drop_first=True)
        # need to change to unique index
        if isinstance(prices, pd.Series):
            row_periodic_returns.index = row_periodic_returns.index.strftime(row_date_format)
        else:
            row_periodic_returns = row_periodic_returns.set_index(row_periodic_returns.index.strftime(row_date_format))
        # change index to year for merging
        periodic_returns_table = periodic_returns_table.set_index(periodic_returns_table.index.strftime(row_date_format))

        if is_add_annual_column:
            if isinstance(prices, pd.Series):
                row_periodic_returns = row_periodic_returns.rename(ytd_name)
            else:
                row_periodic_returns.columns = [ytd_name]
            periodic_returns_table = pd.concat([periodic_returns_table, row_periodic_returns],
                                               axis=1, sort=False, join='inner')

    if is_inverse_order:
        periodic_returns_table = periodic_returns_table.reindex(index=periodic_returns_table.index[::-1])

    return periodic_returns_table


def _scale_returns_heatmap_colors(returns_table: pd.DataFrame,
                                  ytd_name: str = 'YTD'
                                  ) -> pd.DataFrame:
    """Scale periodic cells and the annual column on separate symmetric colour ranges.

    Args:
        returns_table: actual returns, with the annual return in ``ytd_name``
        ytd_name: name of the annual-return column

    Returns:
        values scaled to [-1, 1] within the periodic and annual blocks; NaNs are preserved
    """
    color_table = returns_table.astype(float).copy()
    column_groups = [[column for column in color_table.columns if column != ytd_name]]
    if ytd_name in color_table.columns:
        column_groups.append([ytd_name])

    for columns in column_groups:
        if not columns:
            continue
        values = color_table.loc[:, columns].to_numpy()
        finite_values = np.abs(values[np.isfinite(values)])
        scale = np.max(finite_values) if finite_values.size > 0 else 0.0
        if scale > 0.0:
            color_table.loc[:, columns] = color_table.loc[:, columns] / scale
    return color_table


def plot_returns_heatmap(prices: pd.Series,
                         heatmap_freq: str = 'YE',
                         heatmap_column_freq: Optional[str] = 'ME',  # colums for pivot
                         date_format: str = '%Y',
                         is_inverse_order: bool = False,
                         is_add_annual_column: bool = True,
                         cmap: Union[str, ListedColormap] = 'RdYlGn',
                         alpha: float = 1.0,
                         ytd_name: str = 'YTD',
                         fontsize: int = 5,
                         vline_columns: List[int] = None,
                         hline_rows: List[int] = None,
                         figsize: Tuple[float, float] = None,
                         ax: plt.Subplot = None,
                         max_years: Optional[int] = None,
                         is_ytd_color_scale_independent: bool = False,
                         **kwargs
                         ) -> plt.Figure:
    """Plot periodic returns as a colour-coded calendar table.

    Styling arguments shared by exported plot functions are documented in
    ``qis/docs/plotting_kwargs.md``.

    Args:
        prices: price or NAV series used to compute periodic returns.
        heatmap_freq: frequency defining table rows.
        heatmap_column_freq: frequency defining periodic-return columns.
        date_format: row-label date format.
        is_inverse_order: whether to show the latest row first.
        is_add_annual_column: whether to append the annual return column.
        ytd_name: label for the annual return column.
        max_years: maximum number of calendar rows to show; None keeps all rows.
        is_ytd_color_scale_independent: whether periodic cells and the annual column use
            separate symmetric colour ranges while retaining the original annotations.
        **kwargs: forwarded to ``plot_heatmap``.

    Returns:
        The created figure, or None when drawing on a supplied axis.

    Raises:
        ValueError: If ``max_years`` is not positive or None.
    """

    if max_years is not None and max_years <= 0:
        raise ValueError(f"max_years must be positive or None, got {max_years}")

    periodic_returns_table = compute_periodic_returns_table(prices=prices,
                                                            heatmap_freq=heatmap_freq,
                                                            column_period=heatmap_column_freq,
                                                            row_date_format=date_format,
                                                            is_inverse_order=is_inverse_order,
                                                            is_add_annual_column=is_add_annual_column,
                                                            ytd_name=ytd_name)
    if max_years is not None and len(periodic_returns_table.index) > max_years:
        if is_inverse_order:
            periodic_returns_table = periodic_returns_table.iloc[:max_years, :]
        else:
            periodic_returns_table = periodic_returns_table.iloc[-max_years:, :]
    if is_add_annual_column:
        shift = 4 if heatmap_column_freq == 'QE' else 12
        vline_columns_ = [0, shift]
        if vline_columns is not None:
            vline_columns_.append(vline_columns)
        else:
            vline_columns_ = vline_columns_
    else:
        vline_columns_ = vline_columns

    if ax is None:
        if figsize is not None:
            height = put.calc_table_height(num_rows=len(periodic_returns_table.index), scale=0.225)
            fig, ax = plt.subplots(1, 1, figsize=(figsize[0], height))
        else:
            fig, ax = plt.subplots()
    else:
        fig = None

    if periodic_returns_table.size == 0:
        return fig

    heatmap_table = periodic_returns_table
    var_format = "0.1%"
    heatmap_kwargs = kwargs.copy()
    if is_ytd_color_scale_independent and ytd_name in periodic_returns_table.columns:
        heatmap_table = _scale_returns_heatmap_colors(returns_table=periodic_returns_table,
                                                      ytd_name=ytd_name)
        annot = heatmap_kwargs.pop('annot', True)
        if annot is True:
            formatted_returns = periodic_returns_table.apply(
                lambda column: column.map(lambda value: f"{value:0.1%}" if pd.notna(value) else '')
            )
            heatmap_kwargs['annot'] = formatted_returns.to_numpy()
            var_format = None
        elif annot is not False:
            heatmap_kwargs['annot'] = annot
            var_format = None
        else:
            heatmap_kwargs['annot'] = False
        heatmap_kwargs['vmin'] = -1.0
        heatmap_kwargs['vmax'] = 1.0

    phe.plot_heatmap(df=heatmap_table,
                     cmap=cmap,
                     var_format=var_format,
                     alpha=alpha,
                     fontsize=fontsize,
                     vline_columns=vline_columns_,
                     hline_rows=hline_rows,
                     ax=ax,
                     **heatmap_kwargs)

    return fig


def plot_returns_table(prices: pd.DataFrame,
                       time_period_dict: Dict[str, da.TimePeriod],
                       vline_columns: List[int] = None,
                       hline_rows: List[int] = None,
                       transpose: bool = False,
                       var_format: str = '{:.1%}',
                       ax: plt.Subplot = None,
                       **kwargs
                       ) -> plt.Figure:
    """
    plot returns at specified dates dict
    """
    period_returns = []
    for period, time_period in time_period_dict.items():
        period_data = time_period.locate(prices)
        if len(period_data.index) > 1:
            period_return = period_data.iloc[-1, :] / period_data.iloc[0, :] - 1
            period_returns.append(period_return.rename(period))
    data = pd.concat(period_returns, axis=1, sort=False)
    fig = plot_heatmap(df=data,
                       vline_columns=vline_columns,
                       hline_rows=hline_rows,
                       transpose=transpose,
                       var_format=var_format,
                       ax=ax,
                       **kwargs)
    return fig


def compute_periodic_returns(prices: pd.DataFrame,
                             freq: str = 'ME',
                             time_period: da.TimePeriod = None,
                             total_name: str = None,
                             add_total: bool = True,
                             date_format: str = None,
                             **kwargs
                             ) -> pd.DataFrame:
    """
    compute returns at specified frequency for datadrfame
    index are periods, columns are prices.columns
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be dataframe")
    if time_period is not None:
        prices = time_period.locate(prices)

    # the fill below and to_total_returns() further down both read the panel in row order, so
    # the panel has to be chronological first. pd.concat(axis=1, sort=False) leaves the union
    # of two DatetimeIndexes in appearance order from pandas 3.0 - see df_asfreq - and filling
    # such a panel carries the terminal price backwards onto the dates a column does not carry
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()

    # make sure there are no gaps for heterogeneous price data
    prices = prices.ffill().bfill()
    data = ret.to_returns(prices=prices, freq=freq, include_start_date=True, include_end_date=True, drop_first=True)

    if add_total:
        if freq == 'ME':
            total_name = total_name or 'YTD'
        elif freq == 'YE':
            total_name = total_name or 'Total'
        else:
            total_name = total_name or 'total'
        total_return = ret.to_total_returns(prices=prices).rename(total_name).to_frame().T
        data = pd.concat([data, total_return], axis=0)

    if date_format is not None:  # index may include 'Total'
        data.index = [date.strftime(date_format) if isinstance(date, pd.Timestamp) else date for date in data.index]

    return data


def plot_periodic_returns_table(prices: pd.DataFrame,
                                freq: str = 'ME',
                                date_format: str = None,
                                time_period: da.TimePeriod = None,
                                transpose: bool = True,
                                var_format: str = '{:.0%}',
                                total_name: str = None,
                                add_total: bool = True,
                                ax: plt.Subplot = None,
                                **kwargs
                                ) -> plt.Figure:
    """
    plot returns at specified frequency for datadrfame
    columns are periods, rows are prices.columns
    """
    if time_period is not None:
        prices = time_period.locate(prices)

    if freq == 'ME':
        date_format = date_format or '%b'
    elif freq == 'YE':
        date_format = date_format or '%Y'
    else:
        date_format = date_format or '%d%b%Y'

    data = compute_periodic_returns(prices=prices,
                                    freq=freq,
                                    time_period=time_period,
                                    total_name=total_name,
                                    add_total=add_total,
                                    **kwargs)

    if len(data.columns) > 1:
        np_data = data.to_numpy()[:-1, :-1]  # exclude last row from cmap
    else:
        np_data = data.to_numpy()

    if add_total:
        vline_columns = [len(data.index) - 1]
    else:
        vline_columns = None
    data = data.replace({0.0: np.nan})
    fig = plot_heatmap(df=data,
                       vline_columns=vline_columns,
                       transpose=transpose,
                       date_format=date_format,
                       var_format=var_format,
                       vmin=np.nanmin(np_data),
                       vmax=np.nanmax(np_data),
                       ax=ax,
                       **kwargs)
    return fig


def plot_sorted_periodic_returns(prices: pd.DataFrame,
                                 freq: str = 'ME',
                                 date_format: str = '%d%b%Y',
                                 time_period: da.TimePeriod = None,
                                 transpose: bool = True,
                                 var_format: str = '{:.0%}',
                                 total_name: str = None,
                                 add_total: bool = True,
                                 ax: plt.Subplot = None,
                                 **kwargs
                                 ) -> plt.Figure:
    """
    plot returns at specified frequency
    """
    if time_period is not None:
        prices = time_period.locate(prices)

    if freq == 'ME':
        date_format = date_format or '%b'
        total_name = total_name or 'Total'
    elif freq == 'YE':
        date_format = date_format or '%Y'
        total_name = total_name or 'Total'
    else:
        raise NotImplementedError(f"{freq}")

    data = ret.to_returns(prices=prices, freq=freq, include_start_date=True, drop_first=True)
    data.index = [pd.Timestamp(x).strftime(date_format) for x in data.index]
    if add_total:
        total_return = ret.to_total_returns(prices=prices).rename(total_name).to_frame().T
        data = pd.concat([data, total_return], axis=0)

    fixed_colors = pd.Series(put.get_n_colors(n=len(data.columns), is_fixed_n_colors=False), index=data.columns)
    data_colors = pd.DataFrame(index=data.index, columns=data.columns)
    sorted_returns = []
    sorted_colors = []
    for idx, date in enumerate(data.index):
        current_period = data.loc[date, :].sort_values(ascending=False)
        # data_colors.loc[date, :] = fixed_colors[current_period.index].to_list()
        data_colors.loc[date, :] = [fixed_colors[x] for x in current_period.index]
        # data_colors.loc[date, :] = fixed_colors.to_list()
        entries = pd.Series([f"{key.split('_')[0]}\n{var_format.format(v)}" for key, v in current_period.to_dict().items()], name=date)
        sorted_returns.append(entries)
        entries_colors = pd.Series([fixed_colors[key] for key, v in current_period.to_dict().items()], name=date)
        sorted_colors.append(entries_colors)

    sorted_returns = pd.concat(sorted_returns, axis=1, sort=False)
    sorted_colors = pd.concat(sorted_colors, axis=1, sort=False)

    fig = ptb.plot_df_table(df=sorted_returns,
                            first_column_width=None,
                            add_index_as_column=False,
                            data_colors=list(sorted_colors.to_numpy()),
                            ax=ax,
                            **kwargs)
    return fig
