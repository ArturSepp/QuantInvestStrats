"""
import qis.plots.derived as put
"""

# packages
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib._color_data import CSS4_COLORS as mcolors
from matplotlib.colors import rgb2hex, LinearSegmentedColormap
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from scipy import stats as stats
from scipy.stats import skew, kurtosis
from enum import Enum
from typing import List, Union, Tuple, Optional, Dict, Any

# qis
import qis.utils.df_ops as dfo
import qis.utils.df_str as dfs
import qis.utils.df_freq as dff
import qis.utils.struct_ops as sop
from qis.plots.table import ROW_HIGHT, COLUMN_WIDTH, FIRST_COLUMN_WIDTH


class FixedColors(Enum):
    NAVY = mcolors['navy']
    DARKGREEN = mcolors['darkgreen']
    DARKRED = mcolors['darkred']
    CHOCOLATE = mcolors['chocolate']
    DODGERBLUE = mcolors['slateblue']
    DIMGRAY = mcolors['dimgray']
    SANDYBROWN = mcolors['deepskyblue']
    SADDLEBROWN = mcolors['saddlebrown']
    PURPLE = mcolors['purple']
    INDIANRED = mcolors['indianred']
    FIREBRICK = mcolors['firebrick']
    DARKORANGE = mcolors['darkorange']
    MAGENTA = mcolors['magenta']
    PEACHPUFF = mcolors['deepskyblue']
    SIENNA = mcolors['sienna']
    PERU = mcolors['peru']
    LIGHTCORAL = mcolors['lightcoral']
    CRIMSON = mcolors['crimson']


class TrendLine(Enum):
    NONE = 0
    AVERAGE = 1
    AVERAGE_SHADOWS = 2
    ZERO = 3
    ZERO_SHADOWS = 4
    TREND_LINE = 5
    TREND_LINE_SHADOWS = 6
    ABOVE_ZERO_SHADOWS = 7


class LastLabel(Enum):
    NONE = 0
    LAST_VALUE = 1
    LAST_VALUE_SORTED = 2
    AVERAGE_VALUE = 3
    AVERAGE_VALUE_SORTED = 4


def create_dummy_line(**kwargs) -> Line2D:
    return Line2D([], [], **kwargs)


def set_title(ax: plt.Subplot,
              title: str,
              fontsize: int = 12,
              title_color: str = 'darkblue',
              pad: int = None,
              **kwargs
              ) -> None:
    ax.set_title(label=title, color=title_color, fontsize=fontsize, pad=pad)


def set_suptitle(fig: plt.Figure,
                 title: str,
                 fontsize: int = 12,
                 color: str = 'dodgerblue',
                 **kwargs
                 ) -> None:
    fig.suptitle(title, color=color, fontsize=fontsize)


def set_ax_tick_params(ax: plt.Subplot,
                       fontsize: int = None,
                       labelbottom: bool = True,
                       labelleft: bool = True,
                       labeltop: bool = False,
                       pad: int = 0,
                       **kwargs
                       ) -> None:
    ax.tick_params(axis='both',
                   which='both',
                   bottom=False, left=False, top=False, right=False,  # disable ticks
                   labelbottom=labelbottom,
                   labelleft=labelleft,
                   labeltop=labeltop,
                   labelsize=fontsize,
                   pad=pad)


def set_ax_ticks_format(ax: plt.Subplot,
                        xvar_format: Optional[str] = None,
                        yvar_format: str = None,
                        fontsize: int = 10,
                        set_ticks: bool = True,
                        xvar_numticks: int = None,
                        xvar_major_ticks: np.ndarray = None,
                        yvar_numticks: int = None,
                        yvar_major_ticks: np.ndarray = None,
                        x_rotation: int = 0,
                        **kwargs
                        ) -> None:
    if set_ticks:
        set_ax_tick_params(ax=ax, **kwargs)  # always remove minor ticks
    # set x - ticks
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())  # always remove minor ticks

    # Tick locators
    if xvar_numticks is not None:
        ax.xaxis.set_major_locator(mticker.LinearLocator(numticks=xvar_numticks))
    elif xvar_major_ticks is not None:
        ax.xaxis.set_major_locator(mticker.FixedLocator(xvar_major_ticks))
    else:
        ticks_loc = ax.get_xticks()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))

    if yvar_numticks is not None:
        ax.yaxis.set_major_locator(mticker.LinearLocator(numticks=yvar_numticks))
    elif yvar_major_ticks is not None:
        ax.yaxis.set_major_locator(mticker.FixedLocator(yvar_major_ticks))

    # Tick formatters
    if xvar_format is not None:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: xvar_format.format(z)))

    for z in ax.get_xticklabels():
        z.set_fontsize(fontsize=fontsize)

    # set y - ticks
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())  # always remove minor ticks
    if yvar_format is not None:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: yvar_format.format(z)))
    for z in ax.get_yticklabels():
        z.set_fontsize(fontsize=fontsize)
    ax.tick_params(axis='x', rotation=x_rotation)


def set_ax_xy_labels(ax: plt.Subplot,
                     fontsize: int = 10,
                     xlabel: str = None,
                     ylabel: str = None,
                     labelpad: float = None,
                     **kwargs
                     ) -> None:

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    else:
        ax.set_xlabel('')

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=labelpad)
    else:
        ax.set_ylabel('')


def set_ax_tick_labels(ax: plt.Subplot,
                       x_labels: list = None,
                       y_labels: list = None,
                       x_rotation: int = 90,
                       y_rotation: int = 0,
                       is_remove_labels: bool = False,
                       xticks: List[float] = None,
                       fontsize: int = 10,
                       skip_y_axis: bool = False,
                       **kwargs
                       ) -> None:

    if is_remove_labels:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    else:

        if xticks is not None:
            ax.set_xticks(xticks)
        else:  # to stop matplotlib 3.3.1 warning
            ax.set_xticks(ax.get_xticks())

        if x_labels is None:
            x_labels = ax.get_xticklabels()

        ax.set_xticklabels(x_labels,
                           fontsize=fontsize,
                           rotation=x_rotation,
                           minor=False)

        if not skip_y_axis:
            # to stop matplotlib 3.3.1 warning
            ax.set_yticks(ax.get_yticks())

            if y_labels is None:
                y_labels = ax.get_yticklabels()
            ax.set_yticklabels(y_labels,
                               fontsize=fontsize,
                               rotation=y_rotation,
                               minor=False)
        # else:
        #    ax.yaxis.set_major_formatter(plt.NullFormatter())
        #    ax.set_yticklabels([])


def validate_returns_plot(prices: Union[pd.DataFrame, pd.Series],
                          min_number: int = 20,  # 4 years of q returns
                          freq: str = 'QE',
                          fontsize: int = 8,
                          ax: plt.Figure = None,
                          **kwargs
                          ) -> Tuple[bool, Optional[plt.Figure]]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = None

    # num_returns = len(prices.asfreq(freq, method='ffill').dropna().index)
    num_returns = len(prices.asfreq(freq, method='ffill').index)
    if num_returns < min_number:
        is_good = False
        text = f"the number of available returns = {str(num_returns)}\n is not sufficient to make figure"
        ax.text(0.1, 0.5, text,
                transform=ax.transAxes,
                style='italic',
                fontsize=fontsize,
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    else:
        is_good = True

    return is_good, fig


def set_spines(ax: plt.Subplot,
               top_spine: bool = False,
               bottom_spine: bool = True,
               left_spine: bool = True,
               right_spine: bool = False,
               **kwargs
               ) -> None:
    ax.spines['top'].set_visible(top_spine)
    ax.spines['bottom'].set_visible(bottom_spine)
    ax.spines['left'].set_visible(left_spine)
    ax.spines['right'].set_visible(right_spine)


def set_ax_linewidth(ax: plt.Subplot, linewidth: float = 2.0):
    for line in ax.get_lines():
        line.set_linewidth(linewidth)
    for line in ax.get_legend().get_lines():
        line.set_linewidth(linewidth)


def remove_spines(ax: plt.Subplot) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)


def subplot_border(fig: plt.Figure,
                   nrows: int = 1,
                   ncols: int = 1,
                   color: str = 'navy'
                   ) -> None:
    """
    draw border for a figure with multiple subplots
    """

    n_ax1 = nrows
    n_ax2 = ncols

    rects = []
    height = 1.0 / n_ax1
    for r in range(n_ax1):
        rects.append(plt.Rectangle((0.0, r*height), 1.0, height,  # (lower-left corner), width, height
                                   fill=False,
                                   color=color,
                                   lw=1,
                                   zorder=1000,
                                   transform=fig.transFigure, figure=fig))
    width = 1.0 / n_ax2
    for r in range(n_ax2):
        rects.append(plt.Rectangle((r*width, 0), width, 1.0,  # (lower-left corner), width, height
                                   fill=False,
                                   color=color,
                                   lw=1,
                                   zorder=1000,
                                   transform=fig.transFigure, figure=fig))
    fig.patches.extend(rects)


def autolabel(ax: plt.Subplot,
              rects: List[BarContainer],
              xpos: str = 'center',
              name: str = '',
              label0: str = None,
              color: str = None,
              shapecolor: str = None,
              fontsize: int = 10,
              y_offset: int = 8,
              **kwargs
              ) -> None:
    """
    Attach a text label above each bar in *rects*, displaying its height.
    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        y_offset = y_offset if height > 0.0 else -2.0*y_offset

        if shapecolor is not None:
            bbox = dict(boxstyle='round,pad=0.5', fc=shapecolor, alpha=0.5)
        else:
            bbox = None

        if label0 is None:
            label = f"{name} {'{0:.1%}'.format(height)}"
        else:
            label = label0

        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos], y_offset),
                    textcoords="offset points",  # in both directions
                    fontsize=fontsize,
                    color=color,
                    ha=ha[xpos], va='bottom',
                    bbox=bbox)


def set_legend_colors(ax: plt.Subplot,
                      text_weight: str = None,
                      colors: List[str] = None,
                      fontsize: int = 12,
                      **kwargs
                      ) -> None:

    leg = ax.get_legend()
    if colors is None:
        colors = [line.get_color() for line in leg.get_lines()]

    for text, color in zip(leg.get_texts(), colors):
        text.set_color(color)
        text.set_size(fontsize)
        if text_weight is not None:
            text.set_weight(text_weight)


def set_y_limits(ax: plt.Subplot,
                 y_limits: Tuple[Optional[float], Optional[float]]
                 ) -> None:

    ymin, ymax = ax.get_ylim()
    if y_limits[0] is not None:
        ymin = y_limits[0]
    if y_limits[1] is not None:
        ymax = y_limits[1]
    ax.set_ylim([ymin, ymax])


def set_x_limits(ax: plt.Subplot,
                 x_limits: Tuple[Optional[float], Optional[float]]
                 ) -> None:
    xmin, xmax = ax.get_xlim()
    if x_limits[0] is not None:
        xmin = x_limits[0]
    if x_limits[1] is not None:
        xmax = x_limits[1]
    ax.set_xlim([xmin, xmax])


def align_xy_limits(ax: plt.Subplot) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xy_min = np.minimum(xmin, ymin)
    xy_max = np.maximum(xmax, ymax)
    ax.set_xlim([xy_min, xy_max])
    ax.set_ylim([xy_min, xy_max])


def align_y_limits_ax12(ax1: plt.Subplot,
                        ax2: plt.Subplot,
                        is_invisible_x_ax1: bool = False,
                        is_invisible_y_ax2: bool = False,
                        ymin: float = None,
                        ymax: float = None,
                        **kwargs
                        ) -> None:
    ymin1, ymax1 = ax1.get_ylim()
    ymin2, ymax2 = ax2.get_ylim()
    if ymin is None:
        ymin = np.minimum(ymin1, ymin2)
    if ymax is None:
        ymax = np.maximum(ymax1, ymax2)

    ax1.set_ylim([ymin, ymax])
    ax2.set_ylim([ymin, ymax])

    if is_invisible_x_ax1:
        ax1.axes.get_xaxis().set_visible(False)
    if is_invisible_y_ax2:
        ax2.axes.get_yaxis().set_visible(False)


def align_y_limits_axs(axs: List[plt.Subplot],
                       is_invisible_ys: bool = False
                       ) -> None:
    ymins = []
    ymaxs = []
    for ax in axs:
        ymin, ymax = ax.get_ylim()
        ymins.append(ymin)
        ymaxs.append(ymax)
    ymin = np.min(ymins)
    ymax = np.max(ymaxs)
    for ax in axs:
        ax.set_ylim([ymin, ymax])

    if is_invisible_ys:
        for idx, ax in enumerate(axs):
            if idx > 0:
                ax.axes.get_yaxis().set_visible(False)


def align_x_limits_ax12(ax1: plt.Subplot,
                        ax2: plt.Subplot,
                        is_invisible_x_ax1: bool = False,
                        is_invisible_y_ax2: bool = False,
                        **kwargs
                        ) -> None:

    xmin1, xmax1 = ax1.get_xlim()
    xmin2, xmax2 = ax2.get_xlim()
    xmin = np.minimum(xmin1, xmin2)
    xmax = np.maximum(xmax1, xmax2)
    ax1.set_xlim([xmin, xmax])
    ax2.set_xlim([xmin, xmax])

    if is_invisible_x_ax1:
        ax1.axes.get_xaxis().set_visible(False)
    if is_invisible_y_ax2:
        ax2.axes.get_yaxis().set_visible(False)


def align_x_limits_axs(axs: List[plt.Subplot],
                       is_invisible_xs: bool = False
                       ) -> None:
    xmins = []
    xmaxs = []
    for ax in axs:
        xmin, xmax = ax.get_xlim()
        xmins.append(xmin)
        xmaxs.append(xmax)
    xmin = np.min(xmins)
    xmax = np.max(xmaxs)
    for ax in axs:
        ax.set_xlim([xmin, xmax])

    if is_invisible_xs:
        for idx, ax in enumerate(axs):
            if idx < len(axs)-1:
                ax.axes.get_xaxis().set_visible(False)


def set_date_on_axis(data: Union[pd.DataFrame, pd.Series],
                     ax: plt.Subplot,
                     x_date_freq: str = None,
                     is_set_date_minmax: bool = False,
                     date_format: str = '%b-%y',
                     x_date_rotation: int = 90,
                     fontsize: int = 10,
                     **kwargs
                     ) -> None:

    if is_set_date_minmax and isinstance(data.index, pd.DatetimeIndex):
        datemin = np.datetime64(data.index[0], 'Y')
        datemax = np.datetime64(data.index[-1], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)

    if isinstance(data.index, pd.DatetimeIndex):
        if x_date_freq is None:
            labels = data.index.strftime(date_format).to_list()  # specified by index converted to strings
        else:
            ticks = pd.date_range(start=data.index[0], end=data.index[-1], freq=x_date_freq)
            if len(ticks) == 1:
                # try to find optimal freq:
                choices = ['ME', 'W-WED', 'B']
                for choice in choices:
                    ticks = pd.date_range(start=data.index[0], end=data.index[-1], freq=choice)
                    if len(ticks) > 1:
                        break
            labels = ticks.strftime(date_format).to_list()
            ax.set_xticks(ticks)

    else:
        labels = data.index
        x_date_rotation = 0
        ax.set_xticks(labels)

    # remove ticks
    # set_ax_ticks(ax=ax)
    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_xticklabels(labels,
                       fontsize=fontsize,
                       rotation=x_date_rotation,
                       weight='normal',
                       minor=False)
    # remove minor tick labels put by matplotlib
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def map_dates_index_to_str(data: Union[pd.DataFrame, pd.Series],
                           x_date_freq: str = 'YE',
                           date_format: str = '%b-%y'
                           ) -> Tuple[pd.DataFrame, List[str]]:
    """
    get labels for barplot: needed when index is datetime and we want to display in dateformat
    """
    re_indexed_data = data.copy()
    datalables = data.index

    if x_date_freq is None:
        # x_date_freq = None corresponds to string labelled data
        pass
    elif data.index.dtype == 'str':
        # no map is needed for string index
        pass
    elif isinstance(data.index, pd.DatetimeIndex):
        # map dates to to str
        dates_index = list(pd.to_datetime(data.index))

        ticks = pd.date_range(start=data.index[0], end=data.index[-1], freq=x_date_freq)
        if len(ticks) == 1:
            # try to find optimal freq for less than annual
            choices = ['ME', 'W-WED', 'B']
            for choice in choices:
                ticks = pd.date_range(start=data.index[0], end=data.index[-1], freq=choice)
                if len(ticks) > 1:
                    x_date_freq = choice
                    break

        if x_date_freq == 'YE':
            re_indexed_data.index = [t.strftime(date_format) for t in dates_index]
            if pd.infer_freq(data.index) is not None:
                datalables = [t.strftime(date_format) if t.month == 12 else '' for t in dates_index]
            else:  # uneven dates so get 1y apart
                datalables = [t0.strftime(date_format) if t.year - t0.year == 1 and t.year % 2 == 0 else '' for
                              t, t0 in zip(dates_index[1:], dates_index[:-1])] + [dates_index[-1].strftime(date_format)]
        elif x_date_freq == 'A-Mar':
            re_indexed_data.index = [t.strftime(date_format) for t in dates_index]
            datalables = [t.strftime(date_format) if t.month == 3 else '' for t in dates_index]  # and t.year % 2 == 0

        elif x_date_freq == 'QE':
            re_indexed_data.index = [t.strftime(date_format) for t in dates_index]
            if pd.infer_freq(data.index) is not None:
                datalables = [t.strftime(date_format) if t.month % 3 == 0 else '' for t in dates_index]
            else:  # uneven dates so get 1y apart
                datalables = [t0.strftime(date_format) if t.month - t0.month == 1 and t0.month % 3 == 0 else '' for
                              t, t0 in zip(dates_index[1:], dates_index[:-1])] + [dates_index[-1].strftime(date_format)]

        elif x_date_freq == 'QS':
            indices = pd.Series(pd.to_datetime(data.index), index=data.index)
            indices = dff.df_asfreq(indices, freq=x_date_freq, inclusive='right', include_end_date=False).to_list()
            re_indexed_data.index = [t.strftime(date_format) for t in dates_index]
            datalables = [t.strftime(date_format) if t in indices else '' for t in dates_index]

        elif x_date_freq == 'B':
            re_indexed_data.index = [t.strftime('%b-%d') for t in dates_index]
            datalables = [t0.strftime('%b-%d') if t.day - t0.day > 0 else '' for
             t, t0 in zip(dates_index[1:], dates_index[:-1])] + [dates_index[-1].strftime('%b-%d')]

        else:  # does not matter
            datalables = [t.strftime('%d-%b-%y') for t in dates_index]
            re_indexed_data.index = datalables

            if x_date_freq == 'ME':
                datalables = [t0.strftime(date_format) if t.month - t0.month == 1 or t.month - t0.month == -11 else '' for
                              t, t0 in zip(dates_index[1:], dates_index[:-1])] + [dates_index[-1].strftime(date_format)]

            elif x_date_freq == '5A':
                datalables = [t.strftime(date_format) if t.month == 12 and t.year % 5 == 0 else '' for t in dates_index]

            elif list(x_date_freq)[-1] == 'YE': # frequncy of type 1A, 2A, 3A,...
                n_years = int(sop.separate_number_from_string(x_date_freq)[0])
                datalables = [t.strftime(date_format) if t.month == 12 and t.year % n_years == 0 else '' for t in dates_index]

        # remove dublicates
        val0 = ''
        for ind, val in enumerate(datalables):
            if val == val0:
                datalables[ind] = ''
            val0 = val

    return re_indexed_data, datalables


def set_legend(ax: plt.Subplot,
               labels: Union[List[str], pd.Index] = None,
               lines: List[Tuple] = None,
               markers: List[str] = None,
               colors: Optional[List[str]] = None,
               legend_loc: str = 'upper left',
               reverse_columns: bool = False,
               bbox_to_anchor: Tuple[float, float] = None,
               text_weight: str = 'light',
               legend_title: Optional[str] = None,
               fontsize: int = 12,
               ncol: int = 1,
               framealpha: float = 0.0,
               handlelength: float = 1.0,
               facecolor: str = None,
               numpoints: int = None,
               **kwargs
               ) -> None:

    if legend_loc is None:
        ax.legend([]).set_visible(False)
        return

    if labels is None:  # this are tuples of 3
        leg = ax.get_legend()
        if leg is not None:
            labels = [label.get_text() for label in leg.get_texts()]
        elif lines is None:
            print('in set_legend: cannot put labels from empty line')
            return

    if colors is None:
        leg = ax.get_legend()
        if leg is not None:
            colors = [line.get_color() for line in leg.get_lines()]

    if lines is None:
        lines = []
        is_markers_list = False
        if markers is not None and isinstance(markers, list):
            is_markers_list = True
        if colors is None:
            for label in labels:
                lines.append((label, {}))
        elif colors is not None and not is_markers_list:
            for label, color in zip(labels, colors):
                lines.append((label, {'color': color}))
        elif colors is not None and is_markers_list:
            for label, color, marker in zip(labels, colors, markers):
                lines.append((label, {'color': color, 'marker': marker}))

    if reverse_columns:
        lines = lines[::-1]

    # new legend
    ax.legend(
        [create_dummy_line(**l[1]) for l in lines],  # Line handles
        [l[0] for l in lines],  # Line titles
        loc=legend_loc,
        labelspacing=0.2,  # The vertical space between the legend entries.
        prop={'size': fontsize},
        bbox_to_anchor=bbox_to_anchor,
        framealpha=framealpha,
        handlelength=handlelength,
        title=legend_title,
        ncol=ncol,
        numpoints=numpoints,
        title_fontsize=fontsize,
        facecolor=facecolor,
    )

    set_legend_colors(ax,
                      text_weight=text_weight,
                      fontsize=fontsize,
                      **kwargs)

    ax.get_legend().get_frame().set_linewidth(0.0)


def set_legend_with_stats_table(stats_table: pd.DataFrame,
                                ax: plt.Subplot,
                                colors: List[str],
                                fontsize: int = 10,
                                legend_loc: Optional[str] = 'upper left',
                                bbox_to_anchor: Tuple[float, float] = None,
                                handlelength: float = 1.0,
                                **kwargs
                                ) -> None:
    """
    convert summary table to string and iterate over lines
    """
    # stats_str = stats.to_string(index_names=False)
    stats_str = dfs.df_all_to_str(df=stats_table)
    legend_title = stats_str.splitlines()[0]  # column names will be titles
    lines = []
    for line, color in zip(stats_str.splitlines()[2:], colors):  # var data is from 2-nd line
        lines.append((line, {'color': color}))
    set_legend(ax=ax,
               legend_title=legend_title,
               lines=lines,
               legend_loc=legend_loc,
               fontsize=fontsize,
               handlelength=handlelength,
               bbox_to_anchor=bbox_to_anchor,
               **kwargs)


def set_linestyles(ax: plt.Subplot,
                   linestyles: List[str]
                   ) -> None:
    leg = ax.get_legend()
    leg_lines = leg.get_lines()
    for idx, linestyle in enumerate(linestyles):
        ax.lines[idx].set_linestyle(linestyle)
        leg_lines[idx].set_linestyle(linestyle)


class LegendStats(Enum):
    NONE = 1
    LAST = 2
    AVG = 3
    AVG_LAST = 4
    AVG_STD = 5
    AVG_STD_SKEW_KURT = 6
    AVG_STD_LAST = 7
    AVG_NONNAN_LAST = 8
    NONZERO_AVG_LAST = 81
    NONZERO_AVG_STD_LAST = 82
    MEDIAN_NONNAN_LAST = 9
    AVG_MEDIAN_STD_NONNAN_LAST = 10
    AVG_LAST_SCORE = 11
    AVG_STD_LAST_SCORE = 12
    FIRST_LAST = 13
    FIRST_LAST_NON_ZERO = 14
    FIRST_AVG_LAST = 15
    FIRST_MEDIAN_LAST = 16
    FIRST_AVG_LAST_SHORT = 17
    AVG_STD_MISSING_ZERO = 18
    MISSING_AVG_LAST = 19
    TOTAL = 20
    MEDIAN = 21
    MEDIAN_MAD = 22
    TSTAT = 23
    AVG_STD_TSTAT = 24
    LAST_NONNAN = 25
    AVG_MIN_MAX_LAST = 26
    FIRST_MIN_MAX_LAST = 27


def get_legend_lines(data: Union[pd.DataFrame, pd.Series],
                     legend_stats: LegendStats = LegendStats.NONE,
                     var_format: str = '{:.0f}',
                     tstat_format: str = '{:,.2f}',
                     nan_display: float = np.nan,  # or zero
                     **kwargs
                     ) -> List[str]:

    data = data.copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    if isinstance(data, pd.Series):
        data = data.to_frame()

    if np.any(data.columns.duplicated()):
        print(data)
        raise ValueError(f"dataframe with dublicated columns not supported:\n{print(data.columns)}")

    if legend_stats == LegendStats.NONE:
        legend_lines = data.columns.to_list()

    elif legend_stats == LegendStats.LAST:
        legend_lines = []
        for column in data.columns:
            legend_lines.append(f"{column}: last={var_format.format(data[column].iloc[-1])}")

    elif legend_stats == LegendStats.LAST_NONNAN:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                last = nan_display
            else:
                last = data_column.dropna().iloc[-1]
            legend_lines.append(f"{column} = {var_format.format(last)}")

    elif legend_stats == LegendStats.AVG:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                avg = nan_display
            else:
                avg = np.nanmean(data_column)
            legend_lines.append(f"{column}: avg={var_format.format(avg)}")

    elif legend_stats == LegendStats.MEDIAN:  # specific for fx amm
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                median = nan_display
            else:
                median = np.nanmedian(data_column)
            legend_lines.append(f"{column}: {var_format.format(median)}")

    elif legend_stats == LegendStats.MEDIAN_MAD:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                median = nan_display
                mad = nan_display
            else:
                median = np.nanmedian(data_column)
                mad = stats.median_abs_deviation(data_column.dropna().to_numpy(), scale='normal')
            legend_lines.append(f"{column}: median={var_format.format(median)}, mad={var_format.format(mad)}")

    elif legend_stats == LegendStats.AVG_LAST:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                avg = nan_display
                last = np.nan
            else:
                avg = np.nanmean(data_column)
                last = data_column.iloc[-1]
            legend_lines.append(f"{column}: avg={var_format.format(avg)}, last={var_format.format(last)}")

    elif legend_stats == LegendStats.AVG_STD:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                avg = np.nan
                std = np.nan
            else:
                avg = np.nanmean(data_column)
                std = np.nanstd(data_column, ddof=1)
            legend_lines.append(f"{column}: avg={var_format.format(avg)}, std={var_format.format(std)}")

    elif legend_stats == LegendStats.AVG_STD_SKEW_KURT:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                avg, std, skw, krt = np.nan, np.nan, np.nan, np.nan
            else:
                data_column = data_column.dropna()
                avg = np.mean(data_column)
                std = np.std(data_column, ddof=1)
                skw = skew(data_column, nan_policy='omit')
                krt = kurtosis(data_column, nan_policy='omit')

            legend_lines.append(f"{column}: avg={var_format.format(avg)}, std={var_format.format(std)}, "
                                f"skew={'{:.2f}'.format(skw)}, kurtosis={'{:.2f}'.format(krt)}")

    elif legend_stats == LegendStats.AVG_STD_LAST:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                avg, std, last = nan_display, nan_display, nan_display
            else:
                avg = np.nanmean(data_column)
                std = np.nanstd(data_column, ddof=1)
                last = data_column.iloc[-1]
            legend_lines.append(f"{column}: avg={var_format.format(avg)}, "
                                f"std={var_format.format(std)}, "
                                f"last={var_format.format(last)}")

    elif legend_stats == LegendStats.TSTAT:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                tstat = nan_display
            else:
                avg = np.nanmean(data_column)
                std = np.nanstd(data_column, ddof=1)
                tstat = avg / std
            legend_lines.append(f"{column}: t-stat={tstat_format.format(tstat)}")

    elif legend_stats == LegendStats.AVG_STD_TSTAT:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                avg, std, tstat = nan_display, nan_display, nan_display
            else:
                avg = np.nanmean(data_column)
                std = np.nanstd(data_column, ddof=1)
                tstat = avg / std
            legend_lines.append(f"{column}: avg={var_format.format(avg)}, "
                                f"std={var_format.format(std)}, "
                                f"t-stat={tstat_format.format(tstat)}")

    elif legend_stats == LegendStats.AVG_NONNAN_LAST:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                avg = nan_display
                last = nan_display
            else:
                avg = np.nanmean(data_column)
                last = data_column.dropna().iloc[-1]
            legend_lines.append(f"{column}: avg={var_format.format(avg)}, last={var_format.format(last)}")

    elif legend_stats == LegendStats.NONZERO_AVG_LAST:
        legend_lines = []
        for column in data.columns:
            data_column = data[column].replace({0.0: np.nan})
            if np.all(np.isnan(data_column)):
                avg = nan_display
                last = nan_display
            else:
                avg = np.nanmean(data_column)
                last = data_column.dropna().iloc[-1]
            legend_lines.append(f"{column}: avg={var_format.format(avg)}, last={var_format.format(last)}")

    elif legend_stats == LegendStats.NONZERO_AVG_STD_LAST:
        legend_lines = []
        for column in data.columns:
            data_column = data[column].replace({0.0: np.nan})
            if np.all(np.isnan(data_column)):
                avg, std, last = nan_display, nan_display, nan_display
            else:
                avg = np.nanmean(data_column)
                std = np.nanstd(data_column)
                last = data_column.dropna().iloc[-1]
            legend_lines.append(f"{column}: avg={var_format.format(avg)}, std={var_format.format(std)}, last={var_format.format(last)}")

    elif legend_stats == LegendStats.MEDIAN_NONNAN_LAST:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                med = nan_display
                last = nan_display
            else:
                med = np.nanmedian(data_column)
                last = data_column.dropna().iloc[-1]
            legend_lines.append(f"{column}: median={var_format.format(med)}, last={var_format.format(last)}")

    elif legend_stats == LegendStats.AVG_MEDIAN_STD_NONNAN_LAST:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                avg = nan_display
                med = nan_display
                std = nan_display
                last = nan_display
            else:
                avg = np.nanmean(data_column)
                med = np.nanmedian(data_column)
                std = np.nanstd(data_column)
                last = data_column.dropna().iloc[-1]
            legend_lines.append(f"{column}: avg={var_format.format(avg)}, median={var_format.format(med)}, std={var_format.format(std)}, last={var_format.format(last)}")

    elif legend_stats == LegendStats.AVG_LAST_SCORE:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                avg = nan_display
                last = nan_display
                score = nan_display
            else:
                data_np = data_column.dropna()
                avg = np.nanmean(data_np)
                last = data_np.iloc[-1]
                score = 0.01 * stats.percentileofscore(a=data_np, score=last, kind='rank')

            legend_lines.append(f"{column}: avg={var_format.format(avg)},"
                                f" last={var_format.format(last)}, "
                                f"last score={'{:.0%}'.format(score)}")

    elif legend_stats == LegendStats.AVG_STD_LAST_SCORE:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                avg = nan_display
                std = nan_display
                last = nan_display
                score = nan_display
            else:
                data_np = data_column.dropna()
                avg = np.nanmean(data_np)
                std = np.nanstd(data_np)
                last = data_np.iloc[-1]
                score = 0.01 * stats.percentileofscore(a=data_np, score=last, kind='rank')

            legend_lines.append(f"{column}: avg={var_format.format(avg)},"
                                f" std={var_format.format(std)}, "
                                f" last={var_format.format(last)}, "
                                f"last score={'{:.0%}'.format(score)}")

    elif legend_stats == LegendStats.FIRST_LAST:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                first = nan_display
                last = nan_display
            else:
                nonnan_data = data_column.dropna()
                first = nonnan_data.iloc[0]
                last = nonnan_data.iloc[-1]
            legend_lines.append(f"{column}: first={var_format.format(first)}, last={var_format.format(last)}")

    elif legend_stats == LegendStats.FIRST_LAST_NON_ZERO:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                first = nan_display
                last = nan_display
            else:
                data_column = data_column.replace({0.0: np.nan})
                nonnan_data = data_column.dropna()
                first = nonnan_data.iloc[0]
                last = nonnan_data.iloc[-1]
            legend_lines.append(f"{column}: first={var_format.format(first)}, last={var_format.format(last)}")

    elif legend_stats in [LegendStats.FIRST_AVG_LAST, LegendStats.FIRST_AVG_LAST_SHORT]:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                first = nan_display
                avg = nan_display
                last = nan_display
            else:
                nonnan_data = data_column.dropna()
                first = nonnan_data.iloc[0]
                avg = np.nanmean(nonnan_data)
                last = nonnan_data.iloc[-1]
            if legend_stats == LegendStats.FIRST_AVG_LAST_SHORT:
                legend_lines.append(f"{column}: [{var_format.format(first)}, "
                                    f"{var_format.format(avg)}, "
                                    f"{var_format.format(last)}]")
            else:
                legend_lines.append(f"{column}: first={var_format.format(first)}, "
                                    f"avg={var_format.format(avg)}, "
                                    f"last={var_format.format(last)}")

    elif legend_stats == LegendStats.FIRST_MEDIAN_LAST:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                first = nan_display
                med = nan_display
                last = nan_display
            else:
                nonnan_data = data_column.dropna()
                first = nonnan_data.iloc[0]
                med = np.nanmedian(nonnan_data)
                last = nonnan_data.iloc[-1]
            legend_lines.append(f"{column}: first={var_format.format(first)}, "
                                f"median={var_format.format(med)}, "
                                f"last={var_format.format(last)}")

    elif legend_stats == LegendStats.AVG_STD_MISSING_ZERO:
        legend_lines = []
        missing_ratio, zeros_ratio = dfo.compute_nans_zeros_ratio_after_first_non_nan(df=data)
        for idx, column in enumerate(data.columns):
            column_data = data[column]
            if np.all(np.isnan(column_data)):
                avg = nan_display
                std = nan_display
                # missing = 1.0
                zeros = nan_display
            else:
                avg = np.nanmean(column_data)
                std = np.nanstd(column_data, ddof=1)
                # missing = missing_ratio[idx]
                zeros = zeros_ratio[idx]
            legend_lines.append(f"{column}: avg={var_format.format(avg)}, "
                                f"std={var_format.format(std)}, "
                                # f"missing%={'{:0.2%}'.format(missing)}, "
                                f"missing%={'{:0.2%}'.format(zeros)}")

    elif legend_stats == LegendStats.MISSING_AVG_LAST:
        legend_lines = []
        missing_ratio, zeros_ratio = dfo.compute_nans_zeros_ratio_after_first_non_nan(df=data)
        for idx, column in enumerate(data.columns):
            column_data = data[column]
            if np.all(np.isnan(column_data)):
                avg = nan_display
                last = nan_display
                zeros = nan_display
            else:
                nonnan_data = column_data.dropna()
                avg = np.nanmean(nonnan_data)
                last = nonnan_data.iloc[-1]
                zeros = zeros_ratio[idx]
            legend_lines.append(f"{column}: "
                                f"missing%={'{:0.2%}'.format(zeros)}, "
                                f"avg={var_format.format(avg)}, "
                                f"last={var_format.format(last)}")

    elif legend_stats == LegendStats.TOTAL:
        legend_lines = []
        for column in data.columns:
            column_data = data[column]
            if np.all(np.isnan(column_data)):
                total = nan_display
            else:
                total = np.nansum(column_data)
            legend_lines.append(f"{column}: total={var_format.format(total)}")
    
    elif legend_stats == LegendStats.AVG_MIN_MAX_LAST:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                avg = np.nan
                min = np.nan
                max = np.nan
                last = np.nan
            else:
                nonnan_data = data_column.dropna()
                avg = np.nanmean(nonnan_data)
                min = np.min(nonnan_data)
                max = np.max(nonnan_data)
                last = nonnan_data.iloc[-1]

            legend_lines.append(f"{column}: avg={var_format.format(avg)}, min={var_format.format(min)}, "
                                f"max={var_format.format(max)}, last={var_format.format(last)}")

    elif legend_stats == LegendStats.FIRST_MIN_MAX_LAST:
        legend_lines = []
        for column in data.columns:
            data_column = data[column]
            if np.all(np.isnan(data_column)):
                first = np.nan
                min = np.nan
                max = np.nan
                last = np.nan
            else:
                nonnan_data = data_column.dropna()
                first = nonnan_data.iloc[0]
                min = np.min(nonnan_data)
                max = np.max(nonnan_data)
                last = nonnan_data.iloc[-1]

            legend_lines.append(f"{column}: first={var_format.format(first)}, min={var_format.format(min)}, "
                                f"max={var_format.format(max)}, last={var_format.format(last)}")
        
    else:
        raise TypeError(f"{legend_stats} not implemented")

    return legend_lines


def get_n_colors(n: int,
                 first_color_fixed: bool = False,
                 last_color_fixed: bool = False,
                 fixed_color: str = 'orangered',
                 type: str = 'soft',
                 is_fixed_n_colors: bool = True,
                 **kwargs
                 ) -> List[str]:
    if is_fixed_n_colors:
        colors = get_n_fixed_colors(n=n,
                                    first_color_fixed=first_color_fixed,
                                    last_color_fixed=last_color_fixed,
                                    fixed_color=fixed_color,  # oragne
                                    type=type,
                                    **kwargs)
    else:
        colors = get_n_cmap_colors(n=n, type=type)
        """
        colors = get_n_mlt_colors(n=n,
                                  first_color_fixed=first_color_fixed,
                                  last_color_fixed=last_color_fixed,
                                  fixed_color=fixed_color,  # oragne
                                  type=type,
                                  **kwargs)
        """
    return colors


def get_n_fixed_colors(n: int,
                       first_color_fixed: bool = False,
                       last_color_fixed: bool = False,
                       fixed_color: str = mcolors['darkorange'],
                       type: str = 'soft',
                       is_hex: bool = False,
                       **kwargs
                       ) -> List[str]:
    """
    colors as '#00284A', '#008B75', ...
    """
    if first_color_fixed:
        colors = [fixed_color]
        n -= 1
    else:
        colors = []

    n_clip = np.minimum(n, len(FixedColors))
    n_to_extend = n-n_clip
    for idx, color in enumerate(FixedColors):
        if idx == n_clip:
            break
        colors.append(color.value)

    if n_to_extend > 0:
        colors.extend(get_n_cmap_colors(n=n_to_extend, type=type, is_hex=is_hex))
    if last_color_fixed:
        colors[-1] = fixed_color
    return colors


def get_n_mlt_colors(n: int,
                     first_color_fixed: bool = False,
                     last_color_fixed: bool = False,
                     fixed_color: str = '#EF5A13',  # orange
                     **kwargs
                     ) -> List[str]:
    """
    colors as '#00284A', '#008B75', ...
    """
    if first_color_fixed:
        colors = [fixed_color]
        n -= 1
    else:
        colors = []
    # colors.extend(get_n_cmap_colors(n=n, type=type, is_hex=is_hex))
    colors.extend(get_n_sns_colors(n=n))

    if last_color_fixed:
        colors[-1] = fixed_color
    return colors


def get_n_hatch(n: int) -> List[str]:
    all_hatch = ["//", "\ \\", "-", "+", "x", "o", "O", ".", "*","|"]
    return all_hatch[:n]


def get_n_markers(n: int) -> List[str]:
    all_markers = ["o", "v",  "D", "p",  "^", "<", "s", ">"]
    return all_markers[:n]


def get_n_cmap_colors(n: int,
                      type: str = 'soft',
                      first_color_fixed: bool = False,
                      is_hex: bool = True
                      ) -> List[str]:
    if n == 1:
        colors = ['gray']
    else:
        cmap = rand_cmap(n, first_color_fixed=first_color_fixed, type=type)
        colors = [cmap(i) for i in range(n)]
        if is_hex:
            colors = [rgb2hex(color) for color in colors]
    return colors


def rand_cmap(nlabels: int,
              type: str = 'bright',
              first_color_fixed: bool = True,
              last_color_black: bool = False,
              verbose: bool = False
              ) -> Optional[LinearSegmentedColormap]:
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_fixed: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    np.random.seed(1)
    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for _ in range(nlabels)]

        # Convert HSV list to RGB
        rand_rg_bcolors = []
        for HSVcolor in randHSVcolors:
            rand_rg_bcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_fixed:
            rand_rg_bcolors[0] = [0, 0, 0]

        if last_color_black:
            rand_rg_bcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', rand_rg_bcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    elif type == 'soft':
        low = 0.6
        high = 0.95
        rand_rg_bcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for _ in range(nlabels)]

        if first_color_fixed:
            rand_rg_bcolors[0] = [0, 0, 0]

        if last_color_black:
            rand_rg_bcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', rand_rg_bcolors, N=nlabels)

    else:
        raise ValueError(f"unknown type = {type}")

    return random_colormap


def get_cmap_colors(n: int, cmap: str = 'RdYlGn') -> List[str]:
    """
    Returns a list of matplotlib cmap colors
    """
    cmap = plt.cm.get_cmap(cmap, n)
    colors = [cmap(n_) for n_ in range(n)]
    return colors


def get_n_sns_colors(n: int,
                     palette:str = 'bright',
                     desat: Optional[float] = None,
                     as_cmap: bool = False,
                     **kwargs) -> List[str]:
    return sns.color_palette(palette=palette, n_colors=n, desat=desat, as_cmap=as_cmap)


def compute_heatmap_colors(a: np.ndarray,
                           axis: int = None,
                           cmap: str = 'RdYlGn',
                           alpha: float = 0.75,  # color scaler
                           exclude_max: bool = False,
                           **kwargs
                           ) -> List[Tuple[float, float, float]]:
    """
    for usage in meta
    axis defines maps by total, columns, rows Literal[None, 0, 1] = None
    """
    lower = np.nanmin(a, axis=axis, keepdims=True)
    upper = np.nanmax(a, axis=axis, keepdims=True)
    if exclude_max:  # avoid too extreeme values
        max_idx = a.argmax()
        a[max_idx] = np.nan
        upper = np.nanmax(a, axis=axis, keepdims=True)
        a[max_idx] = upper

    diffs = upper - lower
    scaler = np.reciprocal(diffs, where=np.greater(diffs, 0.0))
    cond = np.logical_and(np.isfinite(scaler), np.isfinite(scaler))
    z = alpha*np.where(cond, scaler * (a - lower), np.nan)

    if a.ndim == 1:
        # colorise = [(1 - x, x, left) if np.isnan(x) == False else (1, 1, 1) for x in z]
        colorise = [x for x in plt.get_cmap(cmap)(z)]

    elif a.ndim == 2:  # list of lists
        colors = plt.get_cmap(cmap)(z)
        colorise = []
        for idx, row in enumerate(z):
            colorise.append(colors[idx])
    else:
        raise ValueError(f"unsupported {a.ndim}")

    return colorise


def get_data_group_colors(df: pd.DataFrame,
                          x: str,
                          y: str,
                          hue: str = None,
                          is_bullish: bool = True
                          ) -> List[Tuple[float, float, float]]:

    avg = df.groupby(x, sort=False).mean().sort_values(y)[y]
    colorise = compute_heatmap_colors(a=avg.to_numpy())
    if not is_bullish:
        colorise = colorise[::-1]

    if hue is not None:
        n_hue = len(df[hue].unique())
        # colorise = [elem for _ in range(n_hue) for elem in colorise]  # extend colors n times
        colorise = [elem for elem in colorise for _ in range(n_hue) ]  # extend colors n times
    return colorise


def add_scatter_points(ax: plt.Subplot,
                       label_x_y: Union[Dict[str, Tuple[float, float]], List[Tuple[float, float]]],
                       fontsize: int = 12,
                       color: str = 'steelblue',
                       colors: List[str] = None,
                       linewidth: int = 3,
                       marker: str = '*',
                       size: int = 3,
                       **kwargs
                       ) -> None:
    """
    add scatter points to ax
    """
    if colors is None:
        if isinstance(label_x_y, dict):
            colors = len(label_x_y.keys()) * [color]
        else:
            colors = len(label_x_y) * [color]
    if isinstance(label_x_y, dict):
        for (key, (x, y)), color in zip(label_x_y.items(), colors):
            ax.annotate(key, xy=(x, y), xytext=(2, 2), color=color,
                        textcoords='offset points', ha='left', va='bottom', fontsize=fontsize)
            ax.scatter(x=x, y=y, marker=marker, color=color, s=size, linewidth=linewidth)
    else:
        for (x, y), color in zip(label_x_y, colors):
            ax.scatter(x=x, y=y, marker=marker, color=color, s=size, linewidth=linewidth)


def calc_table_height(num_rows: int,
                      row_height: float = ROW_HIGHT,
                      first_row_height: float = None,
                      scale: float = 0.225
                      ) -> float:
    """
    core function for sizing of table and heatmaplot
    """
    if first_row_height is None:
        first_row_height = row_height
    height = (scale*num_rows*row_height + scale*first_row_height)
    return height


def calc_table_width(num_col: int,
                     column_width: float = COLUMN_WIDTH,
                     first_column_width: Optional[float] = FIRST_COLUMN_WIDTH,
                     scale: float = 0.225
                     ) -> float:
    """
    core function for sizing of table and heatmaplot
    """
    if first_column_width is None:
        first_column_width = first_column_width
    width = (scale*num_col*column_width + scale*first_column_width)
    return width


def calc_df_table_size(df: pd.DataFrame,
                       min_rows: Optional[int] = None,
                       min_cols: Optional[int] = None,
                       scale_rows: float = 0.225,
                       scale_cols: float = 0.225
                       ) -> Tuple[float, float, int, int]:
    """
    calc optimal table size
    """
    if min_rows is not None:
        num_rows = np.maximum(len(df.index), min_rows)
    else:
        num_rows = len(df.index)
    if min_cols is not None:
        num_cols = np.maximum(len(df.columns), min_cols)
    else:
        num_cols = len(df.columns)
    width = calc_table_width(num_col=num_cols, scale=scale_cols)
    height = calc_table_height(num_rows=num_rows, scale=scale_rows)
    return width, height, num_cols, num_rows


def get_df_table_size(df: pd.DataFrame,
                      min_rows: Optional[int] = None,
                      min_cols: Optional[int] = None,
                      scale_rows: float = 0.225,
                      scale_cols: float = 0.225
                      ) -> Tuple[float, float]:
    """
    calc optimal table size
    """
    width, height, num_cols, num_rows = calc_df_table_size(df=df,
                                                           min_rows=min_rows,
                                                           min_cols=min_cols,
                                                           scale_rows=scale_rows,
                                                           scale_cols=scale_cols)
    return width, height


def reset_xticks(ax: plt.Axes, data: np.ndarray, nbins: int = 20, var_format: str = '{:.2%}') -> None:
    """
    useful for barplot to reduce the number of x ticks
    """
    current_ticks = ax.get_xticks()
    ax.set_xticks(np.linspace(current_ticks[0], current_ticks[-1], nbins))
    x_datalables = np.linspace(data[0], data[-1], nbins)
    ax.set_xticklabels([var_format.format(x) for x in x_datalables], rotation=0, fontsize=10)


class UnitTests(Enum):
    DUMMY_LINE = 1
    LEGEND_LINES = 2
    CMAP_COLORS = 3
    SNS_COLORS = 4
    HEATMAP_COLORS = 5
    GET_COLORS = 6


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.DUMMY_LINE:
        print(create_dummy_line())

    elif unit_test == UnitTests.LEGEND_LINES:
        from qis.test_data import load_etf_data
        prices = load_etf_data().dropna()

        for legend_stats in LegendStats:
            legend_lines = get_legend_lines(data=prices, legend_stats=legend_stats)
            print(legend_lines)

    elif unit_test == UnitTests.CMAP_COLORS:
        cmap_colors = get_cmap_colors(n=100)
        print(cmap_colors)

    elif unit_test == UnitTests.SNS_COLORS:
        cmap_colors = get_n_sns_colors(n=3)
        print(cmap_colors)

    elif unit_test == UnitTests.HEATMAP_COLORS:
        data = np.array([1.0, 2.0, 3.0])
        print(data.ndim)
        heatmap_colors = compute_heatmap_colors(a=data)
        print(heatmap_colors)

        data = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        print(data.ndim)
        heatmap_colors = compute_heatmap_colors(a=data)
        print(heatmap_colors)

    elif unit_test == UnitTests.GET_COLORS:
        n_colors = get_n_colors(n=10)
        print(n_colors)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.GET_COLORS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
