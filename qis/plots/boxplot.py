"""
boxplot
"""
# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, List, Dict, Optional, Tuple
from enum import Enum

# qis
import qis.utils.df_cut as dfc
import qis.utils.df_melt as dfm
import qis.plots.utils as put


def plot_box(df: Union[pd.Series, pd.DataFrame],
             x: str,
             y: str,
             xlabel: Optional[Union[str, bool]] = True,
             ylabel: Optional[Union[str, bool]] = True,
             hue: Optional[str] = None,
             hue_order: List[str] = None,
             xvar_format: str = '{:.2f}',
             yvar_format: str = '{:.2f}',
             colors: Union[List[str], List[Tuple[float, float, float]]] = None,
             title: str = None,
             meanline: bool = False,
             showfliers: bool = False,
             showmeans: bool = False,
             showmedians: bool = False,
             add_y_mean_labels: bool = False,
             add_xy_mean_labels: bool = False,
             add_xy_med_labels: bool = False,
             add_y_med_labels: bool = False,
             x_rotation: Optional[int] = 0,
             legend_loc: Optional[str] = 'upper right',
             fontsize: int = 10,
             linewidth: float = 1.0,
             y_lines: List[Dict] = None,
             labels: List[str] = None,
             original_index: Union[pd.DatetimeIndex, pd.Index] = None,
             x_date_freq: Union[str, None] = 'YE',
             date_format: str = '%b-%y',
             num_obs_for_ci: np.ndarray = None,
             add_zero_line: bool = False,
             add_mean_std_bound: bool = False,
             add_autocorr_std_bound: bool = False,
             continuous_x_col: str = None,
             y_limits: Tuple[Optional[float], Optional[float]] = None,
             whis: Optional[float] = 1.5,  # sns default
             add_hue_to_legend_title: bool = True,
             ax: plt.Subplot = None,
             **kwargs
             ) -> Optional[plt.Figure]:
    """
    plot boxplot of df[[x, y]]
    original_index use for melted df in term of original index
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if colors is not None:
        palette = colors
    else:
        if hue is None:
            palette = put.get_data_group_colors(df=df, x=x, y=y)
        else:
            palette = put.get_n_colors(n=len(df[x].unique()))

    sns.boxplot(x=x, y=y, data=df,
                hue=hue,
                hue_order=hue_order,
                palette=palette,
                linewidth=linewidth,
                meanline=meanline,
                showfliers=showfliers,
                medianprops={'visible': showmedians, 'color': 'black'},
                showmeans=showmeans or meanline,
                whis=whis,
                ax=ax)

    if num_obs_for_ci is not None:
        se = 1.96 / np.sqrt(num_obs_for_ci)
        mean = np.zeros_like(se)
        xx = np.arange(0, len(num_obs_for_ci))
        ax.plot(xx, -se+mean, color='red', lw=linewidth, linestyle='--')
        ax.plot(xx, mean, color='red', lw=linewidth, linestyle='-')
        ax.plot(xx, se + mean, color='red', lw=linewidth, linestyle='--')

    elif add_mean_std_bound:
        mean = np.nanmean(df[y].to_numpy())
        stdev = np.nanstd(df[y].to_numpy())
        se = 1.96 * stdev / np.sqrt(len(df.index))
        y_lines = [dict(x=-se+mean, color='blue', linestyle='--'),
                   dict(x=mean, color='blue', linestyle='-'),
                   dict(x=se+mean, color='blue', linestyle='--')]
    elif add_autocorr_std_bound:
        se = 1.96 * 2.0 / np.sqrt(len(df.index))
        y_lines = [dict(x=-se, color='blue', linestyle='--'),
                   dict(x=0, color='blue', linestyle='-'),
                   dict(x=se, color='blue', linestyle='--')]

    elif add_zero_line:
        y_lines = [dict(x=0, color='black', linestyle='-')]

    if y_lines is not None:
        for y_line in y_lines:
            ax.axhline(y=y_line['x'], color=y_line['color'], linestyle=y_line['linestyle'], lw = 1)

    # x ticks
    if original_index is not None:
        dates = pd.Series(data=original_index, index=original_index)
        re_indexed_data, datalables = put.map_dates_index_to_str(data=dates,
                                                                 x_date_freq=x_date_freq,
                                                                 date_format=date_format)

        current_ticks = ax.get_xticks()
        xticks = list(np.linspace(current_ticks[0], current_ticks[-1], len(datalables)))
        put.set_ax_tick_labels(ax=ax, x_labels=datalables, x_rotation=x_rotation, xticks=xticks, fontsize=fontsize)

    else:

        ax.set_xticklabels(ax.get_xticklabels(),
                           fontsize=fontsize,
                           rotation=x_rotation,
                           minor=False)
        put.set_ax_tick_params(ax=ax, fontsize=fontsize)
        put.set_ax_tick_labels(ax=ax, x_rotation=x_rotation, fontsize=fontsize)

    if len(ax.get_yticks()) > 0:
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([yvar_format.format(x) for x in ax.get_yticks()],
                           fontsize=fontsize, minor=False)

    if add_y_mean_labels or add_xy_mean_labels:
        if hue is not None:
            df_by_x = df.sort_values(hue).groupby(hue, sort=False)
            x_col = x
        else:
            df_by_x = df.sort_values(x).groupby(x, sort=False)
            if continuous_x_col is None:
                raise ValueError(f"continuous_x_col must be given")
            x_col = continuous_x_col

        for x_idx, (key, g_data) in enumerate(df_by_x):
            y_val = g_data[y].mean()
            x_val = g_data[x_col].mean()
            if add_y_mean_labels:
                ax.text(x_idx, y_val, f"{yvar_format.format(y_val)}\n",
                        ha='center', va='center',
                        color='black', fontsize=fontsize)
            else:
                ax.text(x_idx, y_val, f"avg:\ny={yvar_format.format(y_val)}\nx={xvar_format.format(x_val)}",
                        ha='center', va='center',
                        color='black', fontsize=fontsize)

    elif add_y_med_labels or add_xy_med_labels:
        if continuous_x_col is None:
            raise ValueError(f"continuous_x_col must be given")
        df_by_x = df.groupby(x, sort=False)

        for x_idx, (key, g_data) in enumerate(df_by_x):
            y_val = g_data[y].median()
            x_val = g_data[continuous_x_col].median()
            if add_y_med_labels:
                ax.text(x_idx, y_val, f"{yvar_format.format(y_val)}\n\n",
                        ha='center', va='center',
                        color='black', fontsize=fontsize)
            else:
                ax.text(x_idx, y_val, f"med:\ny={yvar_format.format(y_val)}\nx={xvar_format.format(x_val)}",
                        ha='center', va='center',
                        color='black', fontsize=fontsize)

    # labels
    if xlabel is not None:
        if isinstance(xlabel, bool):
            if xlabel is True:
                xlabel = f"x = {x}"
            else:
                xlabel = None

    if ylabel is not None:
        if isinstance(ylabel, bool):
            if ylabel is True:
                ylabel = f"y = {y}"
            else:
                ylabel = None

    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize)

    if y_limits is not None:
        put.set_y_limits(ax=ax, y_limits=y_limits)

    # need to add line for legend with hues
    lines = None
    if labels is not None:
        lines = []
        for label, color in zip(labels, palette):
            lines.append((label, {'color': color}))

    if legend_loc is not None:
        put.set_legend(ax=ax,
                       legend_title=hue if add_hue_to_legend_title else None,
                       legend_loc=legend_loc,
                       fontsize=fontsize,
                       lines=lines,
                       **kwargs)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize)
    put.set_spines(ax=ax)

    return fig


def df_boxplot_by_index(df: Union[pd.Series, pd.DataFrame],
                        index_var_name: str = 'date',
                        ylabel: str = 'returns',
                        xlabel: str = None,
                        show_ylabel: bool = True,
                        title: str = None,
                        colors: Optional[List[str]] = None,
                        ax: plt.Subplot = None,
                        **kwargs
                        ) -> Optional[plt.Figure]:
    """
    plot data frame by aggregating by columns
    xvar is the index
    appropriate when y data is uniform conditional n regimes
    """
    # make sure it is given
    df.index.name = index_var_name
    box_data = dfm.melt_df_by_columns(df=df, x_index_var_name=index_var_name, y_var_name=ylabel)
    if colors is None:
        colors = put.compute_heatmap_colors(a=np.nanmean(df.to_numpy(), axis=1))

    fig = plot_box(df=box_data,
                   x=index_var_name,
                   y=ylabel,
                   continuous_x_col=ylabel,
                   ylabel=ylabel if show_ylabel else None,
                   original_index=df.index,
                   colors=colors,
                   xlabel=xlabel or False,
                   title=title,
                   ax=ax,
                   **kwargs)
    return fig


def df_boxplot_by_columns(df: Union[pd.Series, pd.DataFrame],
                          hue_var_name: str = 'instruments',
                          y_var_name: str = 'weights',
                          ylabel: str = 'weights',
                          show_ylabel: bool = True,
                          title: str = None,
                          colors: Optional[List[str]] = None,
                          ax: plt.Subplot = None,
                          **kwargs
                          ) -> Optional[plt.Figure]:
    """
    plot boxplot using data by columns
    x-axis will be colum names
    """
    box_data = dfm.melt_df_by_columns(df=df, x_index_var_name=None, hue_var_name=hue_var_name, y_var_name=y_var_name)

    if colors is None:
        colors = put.compute_heatmap_colors(a=np.nanmean(df.to_numpy(), axis=0), **kwargs)

    fig = plot_box(df=box_data,
                   x=hue_var_name,
                   y=y_var_name,
                   continuous_x_col=y_var_name,
                   ylabel=ylabel if show_ylabel else None,
                   xlabel=hue_var_name,
                   original_index=None,
                   colors=colors,
                   title=title,
                   ax=ax,
                   **kwargs)

    return fig


def df_dict_boxplot_by_columns(dfs: Dict[str, Union[pd.Series, pd.DataFrame]],
                               hue_var_name: str = 'instruments',
                               y_var_name: str = 'weights',
                               ylabel: str = 'weights',
                               hue: str = 'Portfolio',
                               show_ylabel: bool = True,
                               title: str = None,
                               colors: Optional[List[str]] = None,
                               ax: plt.Subplot = None,
                               **kwargs
                               ) -> Optional[plt.Figure]:
    """
    dict keys are added as hue
    plot boxplot using data by columns
    x-axis will be colum names
    """
    box_datas = []
    for key, df in dfs.items():
        df = df.dropna()  # important
        box_data = dfm.melt_df_by_columns(df=df, x_index_var_name=None, hue_var_name=hue_var_name, y_var_name=y_var_name)
        box_data[hue] = key
        box_datas.append(box_data)
    box_datas = pd.concat(box_datas, axis=0)
    #if colors is None:
    #    colors = put.compute_heatmap_colors(a=np.nanmean(df.to_numpy(), axis=0))

    fig = plot_box(df=box_datas,
                   x=hue_var_name,
                   y=y_var_name,
                   hue=hue,
                   labels=list(dfs.keys()),
                   continuous_x_col=y_var_name,
                   ylabel=ylabel if show_ylabel else None,
                   xlabel=hue_var_name,
                   original_index=None,
                   colors=colors,
                   title=title,
                   ax=ax,
                   **kwargs)

    return fig


def df_boxplot_by_hue_var(df: Union[pd.Series, pd.DataFrame],
                          hue_var_name: str = 'date',
                          x_index_var_name: str = None,
                          y_var_name: str = 'returns',
                          is_heatmap_colors: bool = True,
                          hue_order: List[str] = None,
                          title: str = None,
                          colors: List[str] = None,
                          ax: plt.Subplot = None,
                          **kwargs
                          ) -> Optional[plt.Figure]:
    """
    plot time series of data frame by melting columns
    hue_var_name is hue index
    """
    box_data = dfm.melt_df_by_columns(df=df,
                                      x_index_var_name=x_index_var_name,
                                      y_var_name=y_var_name,
                                      hue_var_name=hue_var_name,
                                      hue_order=hue_order)
    if colors is None:
        if is_heatmap_colors:
            a = np.nanmedian(df.to_numpy(), axis=0)
            colors = put.compute_heatmap_colors(a=np.where(np.isfinite(a), a, 0.0))
        else:
            n = len(df.columns) if isinstance(df, pd.DataFrame) else 1
            colors = put.get_n_colors(n=n, **kwargs)

    fig = plot_box(df=box_data,
                   x=x_index_var_name,
                   y=y_var_name,
                   hue=hue_var_name,
                   colors=colors,
                   title=title,
                   ax=ax,
                   **kwargs)

    return fig


def df_boxplot_by_classification_var(df: pd.DataFrame,
                                     x: str,  # classification variables
                                     y: str,
                                     num_buckets: int = 6,
                                     x_hue_name: str = 'x_regimes',
                                     hue: Optional[str] = None,
                                     hue_order: List[str] = None,
                                     title: str = None,
                                     is_add_xlabel: bool = True,
                                     xvar_format: str = '{:.2f}',
                                     yvar_format: str = '{:.2f}',
                                     showfliers: bool = False,
                                     showmeans: bool = False,
                                     meanline: bool = False,
                                     medianline: bool = True,
                                     add_xy_mean_labels: bool = False,
                                     add_xy_med_labels: bool = False,
                                     is_value_labels: bool = True,
                                     colors: Union[List[str], List[Tuple[float, float, float]]] = None,
                                     ax: plt.Subplot = None,
                                     **kwargs
                                     ) -> None:
    """
    use x as classification var and plod box plot relative to quatiles of x
    """
    scatter_data, _ = dfc.add_quantile_classification(df=df, x_column=x,
                                                      hue_name=x_hue_name,
                                                      num_buckets=num_buckets,
                                                      bins=None,
                                                      xvar_format=xvar_format,
                                                      is_value_labels=is_value_labels)

    plot_box(df=scatter_data,
             x=x_hue_name,
             y=y,
             hue=hue,
             continuous_x_col=x,
             xlabel=x_hue_name if is_add_xlabel else None,
             hue_order=hue_order,
             title=title,
             xvar_format=xvar_format,
             yvar_format=yvar_format,
             showfliers=showfliers,
             showmeans=showmeans,
             meanline=meanline,
             medianline=medianline,
             add_xy_mean_labels=add_xy_mean_labels,
             add_xy_med_labels=add_xy_med_labels,
             colors=colors,
             ax=ax,
             **kwargs)


def df_dict_boxplot_by_classification_var(data_dict: Dict[Tuple[str, str], pd.DataFrame],
                                          num_buckets: int = 6,
                                          x_hue_name: str = 'x_regimes',
                                          var_hue_name: str = 'vars',
                                          y_var_name: str = 'y_var',
                                          title: str = None,
                                          is_add_xlabel: bool = True,
                                          xvar_format: str = '{:.2f}',
                                          yvar_format: str = '{:.2f}',
                                          showfliers: bool = False,
                                          showmeans: bool = False,
                                          meanline: bool = False,
                                          medianline: bool = True,
                                          add_xy_mean_labels: bool = False,
                                          add_xy_med_labels: bool = False,
                                          colors: Union[List[str], List[Tuple[float, float, float]]] = None,
                                          is_value_labels: bool = False,
                                          is_add_last_value: bool = False,
                                          ax: plt.Subplot = None,
                                          **kwargs
                                          ) -> None:
    """
    get dict of dfs and add regime vars, key = (x_column, y_column)
    use x as classification var and plod box plot relative to quatiles of x
    is_value_labels will be same across different df and x
    """
    scatter_datas = None
    hue_order = []
    for key, df in data_dict.items():
        scatter_data, _ = dfc.add_quantile_classification(df=df, x_column=key[0],
                                                          hue_name=x_hue_name,
                                                          num_buckets=num_buckets,
                                                          bins=None,
                                                          xvar_format=xvar_format,
                                                          is_value_labels=is_value_labels,
                                                          **kwargs)
        scatter_data[var_hue_name] = key[1]  # y is id
        scatter_data = scatter_data.rename({key[1]: y_var_name}, axis=1)
        if scatter_datas is None:
            scatter_datas = scatter_data
        else:
            scatter_datas = pd.concat([scatter_datas, scatter_data], axis=0)
        hue_order.append(key[1])
    plot_box(df=scatter_datas,
             x=x_hue_name,
             y=y_var_name,
             hue=var_hue_name,
             xlabel=x_hue_name if is_add_xlabel else None,
             hue_order=hue_order,
             title=title,
             xvar_format=xvar_format,
             yvar_format=yvar_format,
             showfliers=showfliers,
             showmeans=showmeans,
             meanline=meanline,
             medianline=medianline,
             add_xy_mean_labels=add_xy_mean_labels,
             add_xy_med_labels=add_xy_med_labels,
             labels=hue_order,
             colors=colors,
             ax=ax,
             **kwargs)
    if is_add_last_value:
        scatter_datas = scatter_datas.sort_index().ffill()
        last = scatter_datas.iloc[-1, :]
        for idx, key in enumerate(data_dict.keys()):
            ax.annotate('Last',
                        xy=(last[x_hue_name], last[key[1]]), xytext=(1, 1),
                        textcoords='offset points', ha='left', va='bottom',
                        fontsize=12)
            ax.scatter(x=last[x_hue_name], y=last[key[1]], s=20)


class UnitTests(Enum):
    RETURNS_BOXPLOT = 1
    DF_BOXPLOT = 2
    DF_BOXPLOT_INDEX = 3
    DF_WEIGHTS = 4
    DF_DICT = 5


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()
    returns = prices.asfreq('QE', method='ffill').pct_change().dropna()

    if unit_test == UnitTests.RETURNS_BOXPLOT:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        global_kwargs = dict(fontsize=8, linewidth=0.5, weight='normal', markersize=1)

        index_name = 'date'
        value_name = 'returns'

        box_data = dfm.melt_df_by_columns(df=returns, x_index_var_name=index_name, y_var_name=value_name)
        print(box_data)
        colors = put.compute_heatmap_colors(a=np.nanmean(returns.to_numpy(), axis=1))

        plot_box(df=box_data,
                 x=index_name,
                 y=value_name,
                 original_index=returns.index,
                 colors=colors,
                 xlabel=False,
                 ax=ax,
                 **global_kwargs)

    elif unit_test == UnitTests.DF_BOXPLOT:
        # returns by the quantiles of the first variable
        var = returns.columns[0]
        df_boxplot_by_classification_var(df=returns[var].to_frame(), x=var, y=var)

    elif unit_test == UnitTests.DF_BOXPLOT_INDEX:
        df_boxplot_by_index(df=returns)

    elif unit_test == UnitTests.DF_WEIGHTS:
        df_boxplot_by_columns(df=prices,
                              hue_var_name='instruments',
                              y_var_name='weights',
                              ylabel='weights',
                              showmedians=True,
                              add_y_med_labels=True)

    elif unit_test == UnitTests.DF_DICT:
        dfs = {'alts': prices, 'bal': 0.5*prices}
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            df_dict_boxplot_by_columns(dfs=dfs,
                                       hue_var_name='instruments',
                                       y_var_name='weights',
                                       ylabel='weights',
                                       legend_loc='upper center',
                                       showmedians=True,
                                       add_y_med_labels=True,
                                       ncol=2,
                                       ax=ax)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.DF_DICT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
