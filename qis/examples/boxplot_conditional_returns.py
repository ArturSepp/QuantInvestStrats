# packages
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import yfinance as yf
import qis


class LocalTests(Enum):
    RETURNS_BOXPLOT = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.RETURNS_BOXPLOT:
        asset = 'SPY'
        regime_benchmark = '^VIX'
        prices = yf.download([asset, regime_benchmark], start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].dropna()
        prices = prices.asfreq('W-FRI', method='ffill')
        data = pd.concat([prices[asset].pct_change(),  # use returns over (t_0, t_1]
                          prices[regime_benchmark].shift(1)  # use level at t_0
                          ], axis=1).dropna()

        qis.df_boxplot_by_classification_var(df=data,
                                             x=regime_benchmark,
                                             y=asset,
                                             num_buckets=4,
                                             x_hue_name='VIX bucket',
                                             title=f"{asset} returns conditional on {regime_benchmark}",
                                             xvar_format='{:.2f}',
                                             yvar_format='{:.2%}',
                                             add_xy_mean_labels=True)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.RETURNS_BOXPLOT)
