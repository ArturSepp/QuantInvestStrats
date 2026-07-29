"""
a DataFrame drawn as a colour-coded grid, one cell per value.

``plot_heatmap`` is the only entry point. Colour carries the magnitude and the cell annotation
carries the number, so a panel is readable both at a glance and exactly; no colour bar is
drawn, since ``annot`` makes one redundant. Passing an array of strings to ``annot`` lays a grid
of formatted values over unformatted data, and requires ``var_format=None``: otherwise
``var_format`` reaches seaborn as ``fmt`` and raises on the string cells.

The scale is centred on zero and diverging by default, which suits a signed quantity. ``vmin``
and ``vmax`` are what make two heatmaps comparable: without them each panel scales to its own
range and a pair drawn side by side misleads. Shared arguments are in
``qis/docs/plotting_kwargs.md``. Text cells with per-cell control over fill and edges are
``qis.plots.table``; the monthly and annual returns grid is in ``qis.plots.derived``.
"""
# packages
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Optional, Union

# qis
import qis.plots.utils as put


def plot_heatmap(df: pd.DataFrame,
                 transpose: bool = False,
                 inverse: bool = False,
                 date_format: Optional[str] = '%Y',
                 cmap: Union[str, ListedColormap] = 'RdYlGn',
                 var_format: Optional[str] = '{:.1%}',
                 alpha: float = 1.0,
                 fontsize: int = 10,
                 title: Optional[str] = None,
                 top_x_label: bool = True,
                 square: bool = False,
                 vline_columns: List[int] = None,
                 hline_rows: List[int] = None,
                 vmin: float = None,
                 vmax: float = None,
                 labelpad: int = 50,
                 ylabel: str = '',
                 annot: Union[bool, np.ndarray] = True,
                 ax: plt.Subplot = None,
                 **kwargs
                 ) -> Optional[plt.Figure]:
    """
    draw a DataFrame as a colour-coded table.

    The colour carries the magnitude and the annotation carries the number, so the panel is
    readable both at a glance and exactly. ``vmin`` and ``vmax`` are what make several heatmaps
    comparable: without them each panel scales its colours to its own range and two charts side
    by side mislead.

    Arguments shared with every ``plot_*`` function are documented in
    ``qis/docs/plotting_kwargs.md``.

    Args:
        df: values to colour, drawn in the orientation of the frame
        transpose: swap rows and columns before drawing
        inverse: reverse the colour map, for a quantity where low is good
        date_format: strftime format for a DatetimeIndex on either axis
        cmap: matplotlib colour map name, or an explicit map. The default is diverging, which
            suits a signed quantity centred near zero
        alpha: cell opacity
        top_x_label: place the column labels above the table rather than below
        square: force square cells, which a correlation matrix wants
        vline_columns: positions after which to draw a vertical separator
        hline_rows: positions after which to draw a horizontal separator
        vmin: value mapped to the low end of the colour scale. None takes the data minimum,
            so panels drawn separately are not comparable unless this is set
        vmax: value mapped to the high end of the colour scale
        labelpad: padding in points between the axis and its label
        annot: write the value in each cell. An array of the same shape writes those strings
            instead, which is how a table of formatted values is drawn over unformatted colours

    Returns:
        the figure drawn on, or None when ``ax`` was supplied
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if df.empty:
        warnings.warn('df is empty: no data to plot')
        return fig

    df = df.copy()

    if date_format is not None:  # index may include 'Total'
        df.index = [date.strftime(date_format) if isinstance(date, pd.Timestamp) else date for date in df.index]

    if transpose:
        df = df.T
        inverse = False

    if inverse:
        df = df.reindex(index=df.index[::-1])

    if var_format is not None:
        var_format = var_format.replace('{:', '').replace('}', '')  # no {}
    else:
        var_format = ""

    sns.heatmap(data=df,
                center=0,
                annot=annot,
                fmt=var_format,
                cmap=cmap,
                alpha=alpha,
                cbar_kws={'size': fontsize},
                cbar=False,
                annot_kws={'size': fontsize},
                xticklabels=True,  # important for full display of labels
                yticklabels=True,  # important for full display of labels
                square=square,
                vmin=vmin,
                vmax=vmax,
                ax=ax)  # ,"ha": 'right' #cbar_kws={'format': '%0.2f%%'}

    if top_x_label:
        ax.xaxis.tick_top()

    if not transpose:
        pass
        # bottom, top = ax.get_ylim()
        # ax.set_ylim(bottom + 0.5, top - 0.5)
    else:
        ax.xaxis.labelpad = labelpad

    put.set_ax_tick_params(ax=ax, fontsize=fontsize, labelbottom=not top_x_label, labeltop=top_x_label, **kwargs)
    put.set_ax_tick_labels(ax=ax, fontsize=fontsize, **kwargs)

    if vline_columns is not None:
        for vline_column in vline_columns:
            ax.vlines([vline_column], *ax.get_ylim(), lw=1)

    if hline_rows is not None:
        for hline_row in hline_rows:
            ax.hlines([hline_row], *ax.get_xlim(), lw=1)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    ax.set_ylabel(ylabel)
    ax.set_xlabel('')

    return fig
