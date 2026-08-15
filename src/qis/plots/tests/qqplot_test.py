import matplotlib.pyplot as plt
from enum import Enum
import qis.perfstats.returns as ret
import qis.perfstats.desc_table as dsc
from qis.plots.qqplot import plot_qq, plot_xy_qq


class LocalTests(Enum):
    RETURNS = 1
    XY_PLOT = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.tests.price_data_test import load_etf_data
    prices = load_etf_data().dropna()

    df = ret.to_returns(prices=prices, drop_first=True)

    if local_test == LocalTests.RETURNS:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        global_kwargs = dict(fontsize=8, linewidth=0.5, weight='normal', markersize=1)

        plot_qq(df=df,
                desc_table_type=dsc.DescTableType.SKEW_KURTOSIS,
                ax=ax,
                **global_kwargs)

    elif local_test == LocalTests.XY_PLOT:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        global_kwargs = dict(fontsize=8, linewidth=0.5, weight='normal', markersize=1)
        plot_xy_qq(x=df.iloc[:, 1],
                   y=df.iloc[:, 0],
                   ax=ax,
                   **global_kwargs)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.RETURNS)
