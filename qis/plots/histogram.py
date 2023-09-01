"""
plot histogram
"""
# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as stats
from scipy.stats import norm
from statsmodels import api as sm
from typing import Union, List, Optional, Tuple
from enum import Enum

# qis
import qis
import qis.plots.utils as put
import qis.perfstats.desc_table as dsc


class PdfType(Enum):
    KDE = 1
    KDE_NORM = 2
    HISTOGRAM = 3
    TRUNCETED_PDF = 4
    KDE_WITH_HISTOGRAM = 5


def plot_histogram(df: Union[pd.DataFrame, pd.Series],
                   pdf_type: PdfType = PdfType.KDE,
                   is_drop_na: bool = True,
                   title: str = None,
                   colors: List[str] = None,
                   legend_stats: put.LegendStats = put.LegendStats.NONE,
                   desc_table_type: Optional[dsc.DescTableType] = dsc.DescTableType.SHORT,
                   add_norm_std_pdf: bool = False,
                   add_data_std_pdf: bool = False,
                   bbox_to_anchor: Optional[Tuple[float, float]] = None,
                   legend_loc: Optional[str] = 'upper left',
                   xlabel: str = None,
                   ylabel: str = None,
                   xvar_format: str = '{:.2f}',
                   yvar_format: Optional[str] = None,
                   fontsize: int = 10,
                   linewidth: float = 1.0,
                   x_limits: Tuple[Optional[float], Optional[float]] = None,
                   y_limits: Tuple[Optional[float], Optional[float]] = (0.0, None),
                   x_min_max_quantiles: Tuple[Optional[float], Optional[float]] = None,
                   clip: Tuple[Optional[float], Optional[float]] = None,
                   add_last_value: bool = False,
                   add_total_sample_pdf: bool = False,  # concat total
                   total_sample_name: str = 'Universe',
                   annualize_vol: bool = False,
                   first_color_fixed: bool = False,
                   fill: bool = False,
                   cumulative: bool = False,
                   bins: Optional[Union[np.ndarray, str]] = "auto",
                   ax: plt.Subplot = None,
                   **kwargs
                   ) -> Optional[plt.Figure]:

    df = df.copy()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if isinstance(df, pd.DataFrame):
        n = len(df.columns)
    elif isinstance(df, pd.Series):
        df = df.to_frame()
        n = 1
    else:
        raise TypeError(f"unsupported data type = {type(df)}")

    if add_total_sample_pdf:
        first_color_fixed = True
        n = n + 1
        # unstack and remove all index data
        universe_data = df.unstack().reset_index(drop=True).rename(total_sample_name)
        # index is not encessary, nan data will be removed in loop
        df = pd.concat([universe_data,
                        df.reset_index(drop=True)
                        ], axis=1)

    if colors is None:
        colors = put.get_n_colors(n=n,
                                  first_color_fixed=first_color_fixed,
                                  **kwargs)

    for column, color in zip(df.columns, colors):
        column_data = df[column]
        if is_drop_na:  # only drop column-wise
            column_data = column_data.dropna()
        if column_data.empty:
            print(f"in plot_pdf: column{column} is empty")
            continue

        column_data = column_data.to_numpy()

        if x_min_max_quantiles is not None:
            if x_min_max_quantiles[0] is not None:
                x_min = np.nanquantile(column_data, q= x_min_max_quantiles[0])
                column_data = column_data[column_data >x_min]
            if x_min_max_quantiles[1] is not None:
                x_max = np.nanquantile(column_data, q=x_min_max_quantiles[1])
                column_data = column_data[column_data < x_max]

        if pdf_type == PdfType.KDE:
            sns.kdeplot(column_data, fill=fill, color=color, linewidth=linewidth,
                        clip=clip,
                        ax=ax)

        elif pdf_type == PdfType.KDE_NORM:
            x, y = kde_dens(column_data)
            ax.plot(x, y, color=color, linewidth=linewidth)

        elif pdf_type == PdfType.HISTOGRAM:
            sns.histplot(data=column_data, kde=False, color=color,
                         stat="probability",
                         cumulative=cumulative,
                         bins=bins,
                         ax=ax)

        elif pdf_type == PdfType.KDE_WITH_HISTOGRAM:
            sns.histplot(data=column_data, kde=False, color=color,
                         stat="probability",
                         bins=bins,
                         ax=ax)
            sns.kdeplot(column_data, fill=fill, color=color, linewidth=linewidth,
                        clip=clip,
                        ax=ax)

        elif pdf_type == PdfType.TRUNCETED_PDF:
            x, y = trunc_dens(column_data, **kwargs)
            ax.plot(x, y, color=color, linewidth=linewidth)

    if add_last_value:
        ymin, ymax = ax.get_ylim()
        hatches = put.get_n_hatch(n=len(df.columns))
        dy = (ymax-ymin) / (len(df.columns) + 1.0)
        y_positions = [0.9 * ymax - n * dy for n in range(len(df.columns))]
        xmin, xmax = ax.get_xlim()
        dx = 0.5 / (xmax-xmin)
        for idx, column in enumerate(df.columns):
            column_data = df[column].to_numpy()
            rects = ax.bar(column_data[-1], y_positions[idx], dx, color=colors[idx], hatch=hatches[idx], alpha=0.5)
            label = f"{column}\nlast={xvar_format.format(column_data[-1])}"
            percentile = stats.percentileofscore(a=column_data, score=column_data[-1], kind='rank')
            label = f"{label}\nrank={'{:.0%}'.format(0.01*percentile)}"
            put.autolabel(ax=ax, rects=rects, xpos='right', label0=label, color=colors[idx], fontsize=fontsize)

    if add_norm_std_pdf or add_data_std_pdf:
        xmin, xmax = ax.get_xlim()
        if add_data_std_pdf:
            norm_mean = np.mean(np.nanmean(df.to_numpy(), axis=0))
            norm_std = np.sqrt(np.mean(np.nanvar(df.to_numpy(), axis=0)))
        else:
            norm_mean = 0.0
            norm_std = 1.0

        dx = 0.1 * norm_std
        x_axis = np.arange(xmin, xmax, dx)
        ax.plot(x_axis, norm.pdf(x_axis, norm_mean, norm_std),
                label=f"Normal PDF ({xvar_format.format(norm_mean)},  {xvar_format.format(norm_std)})",
                color='black',
                lw=1, linestyle='--')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([0.0, ymax])

    if y_limits is not None:
        put.set_y_limits(ax=ax, y_limits=y_limits)
    if x_limits is not None:
        put.set_x_limits(ax=ax, x_limits=x_limits)

    if desc_table_type is not None and desc_table_type != dsc.DescTableType.NONE:
        stats_table = dsc.compute_desc_table(df=df,
                                         annualize_vol=annualize_vol,
                                         desc_table_type=desc_table_type,
                                         **qis.update_kwargs(kwargs, dict(var_format=xvar_format)))
        put.set_legend_with_stats_table(stats_table=stats_table,
                                        ax=ax,
                                        colors=colors,
                                        legend_loc=legend_loc,
                                        bbox_to_anchor=bbox_to_anchor,
                                        fontsize=fontsize,
                                        **kwargs)
    else:
        legend_labels = put.get_legend_lines(data=df,
                                             legend_stats=legend_stats,
                                             var_format=xvar_format)
        put.set_legend(ax=ax,
                       labels=legend_labels,
                       colors=colors,
                       legend_loc=legend_loc,
                       bbox_to_anchor=bbox_to_anchor,
                       fontsize=fontsize,
                       **kwargs)
    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    # cannot use for sns.distplot
    # put.set_ax_tick_labels(ax=ax, fontsize=fontsize, skip_y_axis=True, **kwargs)
    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, **kwargs)
    put.set_ax_ticks_format(ax=ax, xvar_format=xvar_format, yvar_format=yvar_format, fontsize=fontsize, **kwargs)
    if yvar_format is None:
        ax.set_yticklabels('')
    put.set_spines(ax=ax, **kwargs)

    return fig


def kde_dens(x: np.ndarray, bandwidth: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    bandwidth = bandwidth*np.nanstd(x)
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit(bw=bandwidth)
    h = kde.bw
    d = sm.nonparametric.KDEUnivariate(x)
    d = d.fit(bw=h, fft=False)
    d_support = d.support
    d_dens = d.density
    d_dens = d_dens / np.nansum(d_dens)
    return d_support, d_dens


def trunc_dens(x: np.ndarray,
               bandwidth: float = 0.5,
               truncate_below_zero: bool = True,
               **kwargs
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    with truncation at zero:
    """
    bandwidth = bandwidth*np.nanstd(x)
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit(bw=bandwidth)
    h = kde.bw
    if truncate_below_zero:
        w = 1/(1-norm.cdf(0, loc=x, scale=h))
    else:
        w = 1 / norm.cdf(0, loc=x, scale=h)
    d = sm.nonparametric.KDEUnivariate(x)
    d = d.fit(bw=h, weights=w / len(x), fft=False)
    d_support = d.support
    d_dens = d.density
    if truncate_below_zero:
        d_dens[d_support < 0] = 0
    else:
        d_dens[d_support > 0] = 0
    d_dens = d_dens / np.nansum(d_dens)
    return d_support, d_dens


class UnitTests(Enum):
    TEST = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.TEST:
        np.random.seed(1)
        n_instruments = 100
        m_samples = 2
        exposures_nm = np.random.normal(0.0, 1.0, size=(n_instruments, m_samples))
        data = pd.DataFrame(data=exposures_nm, columns=[f"id{n+1}" for n in range(m_samples)])

        fig, ax = plt.subplots(1, 1, figsize=(3.9, 3.4), tight_layout=True)
        global_kwargs = dict(fontsize=6, linewidth=0.5, weight='normal', first_color_fixed=True)

        plot_histogram(df=data,
                       add_data_std_pdf=True,
                       add_last_value=True,
                       ax=ax,
                       **global_kwargs)
        # ax.locator_params(nbins=10, axis='x')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.TEST

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
