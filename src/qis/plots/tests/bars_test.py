import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

# qis
import qis.plots.utils as put
from qis.plots.bars import plot_bars, plot_vbars


class LocalTests(Enum):
    BARS = 1
    BARS2 = 2
    TOP_BOTTOM_RETURNS = 3
    VBAR_WEIGHTS = 4
    MONTHLY_RETURNS_BARS = 5


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.BARS:
        n = 11
        index = [f"id{x+1} {x**2}" for x in range(n)]
        df1 = pd.DataFrame(np.linspace(0.0, 1.0, n), index=index, columns=['part1'])
        df2 = pd.DataFrame(np.linspace(0.0, 0.5, n), index=index, columns=['part2'])
        df = pd.concat([df1, df2], axis=1)
        plot_bars(df=df, stacked=True)

    elif local_test == LocalTests.BARS2:

        n = 11
        index = [f"id{x+1} {x**2}" for x in range(n)]
        df1 = pd.DataFrame(np.linspace(0.0, 1.0, n), index=index, columns=['data1'])
        df2 = pd.DataFrame(np.linspace(0.0, 0.5, n), index=index, columns=['data2'])

        fig, axs = plt.subplots(1, 2, figsize=(8, 6), tight_layout=True)
        datas = [df1, df2]
        titles = ['Group Lasso', 'Lasso']
        for data, ax, title in zip(datas, axs, titles):
            plot_bars(df=data,
                      stacked=False,
                      skip_y_axis=True,
                      title=title,
                      legend_loc=None,
                      x_rotation=90,
                      ax=ax)
        put.align_y_limits_ax12(ax1=axs[0], ax2=axs[1], is_invisible_y_ax2=True)

    elif local_test == LocalTests.TOP_BOTTOM_RETURNS:

        from qis.tests.price_data_test import load_etf_data
        import qis.perfstats.returns as ret

        prices = load_etf_data().dropna().loc['2021', :]
        returns = ret.to_total_returns(prices=prices).sort_values()
        print(returns)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

        plot_bars(df=returns,
                  stacked=False,
                  skip_y_axis=True,
                  legend_loc=None,
                  x_rotation=90,
                  ax=ax)

    elif local_test == LocalTests.VBAR_WEIGHTS:
        desc_dict = {'f1': (0.5, 0.5),
                     'f2': (0.7, 0.3),
                     'f3': (1.0, 0.0),
                     'f4': (0.8, 0.2),
                     'f5': (0.35, 0.65),
                     'f6': (0.0, 1.0)}

        df = pd.DataFrame.from_dict(desc_dict, orient='index', columns=['as1', 'as2'])
        print(df)
        plot_vbars(df=df,
                   colors=put.get_n_colors(n=len(df.columns)),
                   bbox_to_anchor=(0.5, 1.25),
                   add_bar_values=False,
                   add_bar_value_at_mid=False,
                   add_total_bar=False)

    elif local_test == LocalTests.MONTHLY_RETURNS_BARS:
        from qis.tests.price_data_test import load_etf_data
        import qis.perfstats.returns as ret

        prices = load_etf_data().dropna().loc['2020':, :].iloc[:, :3]
        returns = ret.to_returns(prices=prices, freq='ME', drop_first=True)
        print(returns)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

        plot_bars(df=returns,
                  stacked=False,
                  skip_y_axis=True,
                  x_rotation=90,
                  yvar_format='{:,.0%}',
                  date_format='%b-%y',
                  fontsize=6,
                  ax=ax)
    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.VBAR_WEIGHTS)
