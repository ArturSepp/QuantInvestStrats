"""
2-d countrur plot
"""

# packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import List, Tuple, Optional
from enum import Enum

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


class UnitTests(Enum):
    SHARPE_VOL = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.SHARPE_VOL:

        global_kwargs = {'fontsize': 12}

        n = 41
        vol_ps = np.linspace(0.04, 0.20, n)
        vol_xys = np.linspace(0.00, 0.20, n)
        sharpes = np.zeros((n, n))
        for n1, vol_p in enumerate(vol_ps):
            for n2, vol_xy in enumerate(vol_xys):
                sharpes[n1, n2] = (2.0*vol_xy*vol_xy-0.25*vol_p*vol_p)/vol_p

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
        plot_contour(x=vol_ps,
                     y=vol_xys,
                     z=sharpes,
                     fig=fig,
                     **global_kwargs)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.SHARPE_VOL

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
