"""
filled contour plots of a value ``z`` sampled on an ``x`` by ``y`` grid, with a formatted colour
bar. ``plot_contour`` draws a single panel and ``contour_multi`` a row of panels sharing axes,
one per entry of ``zs``. ``plot_contour`` takes a ``fig`` rather than an ``ax``, since the colour
bar is a figure-level artist; ``contour_multi`` builds its own figure at ``figsize``.
"""

# packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import List, Tuple, Optional

# qis
import qis.plots.utils as put


def plot_contour(x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 xvar_format: str = '{:.0%}',
                 yvar_format: str = '{:.0%}',
                 zvar_format: str = '{:.1f}',
                 fontsize: int = 10,
                 num_ranges: int = 7,
                 cmap: str = 'RdYlGn',
                 xlabel: str = 'x',
                 ylabel: str = 'y',
                 title: str = None,
                 fig: plt.Figure = None,
                 **kwargs
                 ) -> Optional[plt.Figure]:

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]

    X, Y = np.meshgrid(x, y)
    Z = z.T  # need to transpose

    cbar = fig.axes[0].contourf(X, Y, Z, num_ranges, cmap=cmap)

    fmt = lambda x, pos: zvar_format.format(x)
    fig.colorbar(cbar, format=FuncFormatter(fmt))
    # cbar.ax.tick_params(labelsize=fontsize)

    put.set_ax_ticks_format(ax=ax, fontsize=fontsize, xvar_format=xvar_format, yvar_format=yvar_format)
    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, **kwargs)

    if title is not None:
        ax.set_title(label=title, fontsize=fontsize)

    return fig


def contour_multi(x: np.ndarray,
                  y: np.ndarray,
                  zs: List[np.ndarray],
                  xvar_format: str = '{:.0%}',
                  yvar_format: str = '{:.0%}',
                  zvar_format: str = '{:.1f}',
                  fontsize: int = 10,
                  num_ranges: int = 7,
                  cmap: str = 'RdYlGn',
                  xlabel: str = 'x',
                  ylabel: str = 'y',
                  titles: List[str] = None,
                  figsize: Tuple[float, float] = (11, 6),
                  **kwargs
                  ) -> plt.Figure:

    fig, axs = plt.subplots(figsize=figsize, nrows=1, ncols=len(zs), sharex=True, sharey=True)

    X, Y = np.meshgrid(x, y)
    for idx, ax in enumerate(axs.flat):
        Z = zs[idx].T
        cbar = fig.axes[idx].contourf(X, Y, Z, num_ranges, cmap=cmap)
        fmt = lambda x, pos: zvar_format.format(x)
        fig.colorbar(cbar, ax=ax, format=FuncFormatter(fmt))
        put.set_ax_ticks_format(ax=ax, fontsize=fontsize, xvar_format=xvar_format, yvar_format=yvar_format)

        ylabel = ylabel if idx == 0 else None
        put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, **kwargs)

        if titles[idx] is not None:
            ax.set_title(label=titles[idx], fontsize=fontsize)

    return fig

