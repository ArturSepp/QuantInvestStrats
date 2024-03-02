# biult in
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

# qis
import qis.file_utils as fu
import qis.utils.dates as da
import qis.models.linear.corr_cov_matrix as cor
import qis.plots.time_series as pts

# data
import yfinance as yf


def get_compute_corrs(prices: pd.DataFrame,
                      time_period: da.TimePeriod = None,
                      span: int = 33
                      ) -> pd.DataFrame:
    returns = prices.pct_change()
    corrs = cor.compute_ewm_corr_df(df=returns,
                                    corr_matrix_output=cor.CorrMatrixOutput.TOP_ROW,
                                    ewm_lambda=1.0-2.0/(span+1.0))

    if time_period is not None:
        corrs = time_period.locate(corrs)

    return corrs


class UnitTests(Enum):
    CORRS_SPY = 1
    CORRS_ALL = 2


def run_unit_test(unit_test: UnitTests):
    tickers = ['SPY', 'BTC-USD']
    prices = yf.download(tickers, start=None, end=None)['Adj Close'][tickers].dropna()
    prices = prices.rename({'BTC-USD': 'BTC'}, axis=1)
    time_period = da.TimePeriod('01Jan2021', None)

    if unit_test == UnitTests.CORRS_SPY:
        corrs = get_compute_corrs(prices=prices, span=33, time_period=time_period)
        trend = corrs.rolling(21).mean()
        df = pd.concat([corrs.iloc[:, 0].rename('Correlation'), trend.iloc[:, 0].rename('1m rolling average')], axis=1)
        fig = pts.plot_time_series(df=df,
                                   trend_line=pts.TrendLine.AVERAGE,
                                   legend_stats=pts.LegendStats.AVG_LAST,
                                   var_format='{:.2%}')

    elif unit_test == UnitTests.CORRS_ALL:

        returns = prices.pct_change()
        # returns = np.log(prices).diff()
        corr_1m = returns['BTC'].rolling(21).corr(returns['SPY']).rename('1m')
        corr_3m = returns['BTC'].rolling(63).corr(returns['SPY']).rename('3m')
        corr_1y = returns['BTC'].rolling(252).corr(returns['SPY']).rename('1y')
        corr_3y = returns['BTC'].rolling(3*252).corr(returns['SPY']).rename('3y')

        corrs = pd.concat([corr_1m, corr_3m, corr_1y, corr_3y], axis=1).dropna()
        corrs = time_period.locate(corrs)
        fu.save_df_to_excel(data=corrs, file_name='btc_spy_corr')

        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(15, 8), tight_layout=True)

            pts.plot_time_series(df=corrs,
                                 trend_line=pts.TrendLine.AVERAGE,
                                 legend_stats=pts.LegendStats.AVG_LAST_SCORE,
                                 var_format='{:.0%}',
                                 fontsize=14,
                                 title='Rolling correlation of daily returns between BTC-S&P500 as function of rolling window',
                                 ax=ax)
        fu.save_fig(fig=fig, file_name='btc_all_corr')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.CORRS_ALL

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
        