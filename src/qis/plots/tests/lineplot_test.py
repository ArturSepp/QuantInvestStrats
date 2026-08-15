
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
import qis.plots.utils as put
from qis.plots.lineplot import plot_line


class LocalTests(Enum):
    LINEPLOT = 1
    MOVE_DATA = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.LINEPLOT:
        x = np.linspace(0, 14, 100)
        y = np.sin(x)
        data = pd.Series(y, index=x, name='data')
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        global_kwargs = dict(fontsize=12, linewidth=2.0, weight='normal', markersize=2)
        plot_line(df=data, legend_stats=put.LegendStats.AVG_LAST, ax=axs[0], **global_kwargs)
        plot_line(df=data, legend_stats=put.LegendStats.AVG_LAST,
                  linestyle='dotted',
                  ax=axs[1], **global_kwargs)

    elif local_test == LocalTests.MOVE_DATA:
        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(2, 1, figsize=(12, 6))
            global_kwargs = dict(fontsize=12, linewidth=2.0, weight='normal', markersize=2)
            index = ['06-07JUN22', '10-17JUN22', '17-24JUN22', '24JUN-01JUL22', '26JUN-30SEP22', '30SEP-30DEC22']
            data = [0.7644, 0.7602, 0.7306, 0.7524, 0.8192, 0.8204]
            df = pd.DataFrame(data, index=index, columns=['Expected Move % annualized'])
            print(df)
            markers = put.get_n_markers(n=1)
            plot_line(df=df,
                      title='ATM forward volatilities Implied from BTC MOVE contracts on 06-Jun-2022',
                      legend_stats=put.LegendStats.NONE,
                      yvar_format='{:.0%}',
                      xvar_format=None,
                      markers=markers,
                      ax=axs[0],
                      **global_kwargs)

            plot_line(df=df,
                      title='ATM forward volatilities Implied from BTC MOVE contracts on 06-Jun-2022',
                      legend_stats=put.LegendStats.NONE,
                      yvar_format='{:.0%}',
                      xvar_format=None,
                      linewidth=0.,
                      markers=['s'] * len(df.columns),
                      ax=axs[1])

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.MOVE_DATA)
