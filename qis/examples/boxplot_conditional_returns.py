# packages
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import yfinance as yf
import qis


class UnitTests(Enum):
    RETURNS_BOXPLOT = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.RETURNS_BOXPLOT:
        asset = 'SPY'
        regime_benchmark = '^VIX'
        prices = yf.download([asset, regime_benchmark], start=None, end=None)['Adj Close'].dropna()
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

    unit_test = UnitTests.RETURNS_BOXPLOT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

