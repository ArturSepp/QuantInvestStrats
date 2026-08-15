import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from qis.plots.utils import (get_n_colors, get_n_sns_colors, create_dummy_line,
                            LegendStats, get_legend_lines, get_cmap_colors, compute_heatmap_colors)


class LocalTests(Enum):
    DUMMY_LINE = 1
    LEGEND_LINES = 2
    CMAP_COLORS = 3
    SNS_COLORS = 4
    HEATMAP_COLORS = 5
    GET_COLORS = 6


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.DUMMY_LINE:
        print(create_dummy_line())

    elif local_test == LocalTests.LEGEND_LINES:
        from qis.tests.price_data_test import load_etf_data
        prices = load_etf_data().dropna()

        for legend_stats in LegendStats:
            legend_lines = get_legend_lines(data=prices, legend_stats=legend_stats)
            print(legend_lines)

    elif local_test == LocalTests.CMAP_COLORS:
        cmap_colors = get_cmap_colors(n=100)
        print(cmap_colors)

    elif local_test == LocalTests.SNS_COLORS:
        cmap_colors = get_n_sns_colors(n=3)
        print(cmap_colors)

    elif local_test == LocalTests.HEATMAP_COLORS:
        data = np.array([1.0, 2.0, 3.0])
        print(data.ndim)
        heatmap_colors = compute_heatmap_colors(a=data)
        print(heatmap_colors)

        data = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        print(data.ndim)
        heatmap_colors = compute_heatmap_colors(a=data)
        print(heatmap_colors)

    elif local_test == LocalTests.GET_COLORS:
        n_colors = get_n_colors(n=10)
        print(n_colors)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.GET_COLORS)
