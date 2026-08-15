import matplotlib.pyplot as plt
from enum import Enum
import qis.plots.utils as put
from qis.plots.utils import TrendLine, LastLabel
from qis.plots.time_series import plot_time_series, plot_time_series_2ax


class LocalTests(Enum):
    PRICES = 1
    PRICES_2AX = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.tests.price_data_test import load_etf_data
    prices = load_etf_data().dropna()

    if local_test == LocalTests.PRICES:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), tight_layout=True)
        global_kwargs = {'fontsize': 8,
                         'linewidth': 0.5,
                         'weight': 'normal',
                         'markersize': 1}
        plot_time_series(df=prices,
                         legend_stats=put.LegendStats.AVG_LAST,
                         last_label=LastLabel.AVERAGE_VALUE_SORTED,
                         trend_line=TrendLine.AVERAGE_SHADOWS,
                         ax=axs[0],
                         **global_kwargs)
        plot_time_series(df=prices,
                         legend_stats=put.LegendStats.AVG_LAST,
                         last_label=LastLabel.LAST_VALUE,
                         trend_line=TrendLine.AVERAGE_SHADOWS,
                         ax=axs[1],
                         **global_kwargs)

    elif local_test == LocalTests.PRICES_2AX:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3), tight_layout=True)
        global_kwargs = {'fontsize': 6,
                         'linewidth': 0.5,
                         'weight': 'normal',
                         'markersize': 1}

        plot_time_series_2ax(df1=prices.iloc[:, -1],
                             df2=prices.iloc[:, :-1],
                             legend_stats=put.LegendStats.AVG_LAST,
                             var_format_yax2='{:.0f}',
                             ax=ax,
                             **global_kwargs)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.PRICES)
