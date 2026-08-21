"""Development runner extracted from ``qis.plots.utils``."""

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

from qis.plots.utils import (
    LegendStats,
    compute_heatmap_colors,
    create_dummy_line,
    get_cmap_colors,
    get_legend_lines,
    get_n_colors,
    get_n_sns_colors,
)

class Locals(Enum):
    DUMMY_LINE = 1
    LEGEND_LINES = 2
    CMAP_COLORS = 3
    SNS_COLORS = 4
    HEATMAP_COLORS = 5
    GET_COLORS = 6

def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local == Locals.DUMMY_LINE:
        print(create_dummy_line())

    elif local == Locals.LEGEND_LINES:
        from qis.run_local.price_data_run import load_etf_data
        prices = load_etf_data().dropna()

        for legend_stats in LegendStats:
            legend_lines = get_legend_lines(data=prices, legend_stats=legend_stats)
            print(legend_lines)

    elif local == Locals.CMAP_COLORS:
        cmap_colors = get_cmap_colors(n=100)
        print(cmap_colors)

    elif local == Locals.SNS_COLORS:
        cmap_colors = get_n_sns_colors(n=3)
        print(cmap_colors)

    elif local == Locals.HEATMAP_COLORS:
        data = np.array([1.0, 2.0, 3.0])
        print(data.ndim)
        heatmap_colors = compute_heatmap_colors(a=data)
        print(heatmap_colors)

        data = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        print(data.ndim)
        heatmap_colors = compute_heatmap_colors(a=data)
        print(heatmap_colors)

    elif local == Locals.GET_COLORS:
        n_colors = get_n_colors(n=10)
        print(n_colors)

    plt.show()

if __name__ == "__main__":
    run_local(local=Locals.GET_COLORS)
