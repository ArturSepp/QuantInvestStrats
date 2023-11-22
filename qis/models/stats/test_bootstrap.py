# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from typing import Tuple, List

# stats
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, acf

# qis
import qis.file_utils as fu
import qis.perfstats.returns as ret
import qis.plots.time_series as pts
from qis.plots.derived.prices import plot_prices

# internal
import qis.models.stats.bootstrap as bts


def plot_price_bootrstrap(prices: pd.Series,
                          seed: int = 1,
                          block_size: int = 20,
                          **kwargs):
    bt_data = bts.bootstrap_price_data(prices=prices,
                                       bootsrap_type=bts.BootsrapType.STATIONARY,
                                       bootsrap_output=bts.BootsrapOutput.SERIES_TO_DF,
                                       index_length=len(prices),
                                       seed=seed,
                                       block_size=block_size)
    bt_data.index = prices.index
    all_data = pd.concat([prices, bt_data], axis=1)
    plot_prices(all_data, **kwargs)


def plot_ew_index_bootrstrap(prices: pd.DataFrame):
    log_returns = ret.to_returns(prices=prices, is_log_returns=True, drop_first=True)
    bt_data = bts.bootstrap_data(data=log_returns,
                                 bootsrap_type=bts.BootsrapType.STATIONARY,
                                 bootsrap_output=bts.BootsrapOutput.DF_TO_LIST_ARRAYS,
                                 index_length=len(prices))
    ew_data = []
    for idx, sample in enumerate(bt_data):
        ew_data.append(pd.Series(np.nanmean(sample, axis=1), name=f"path_{idx+1}"))
    ew_data = pd.concat(ew_data, axis=1)
    ew_data.index = prices.index
    ew_prices = ret.returns_to_nav(ew_data)
    original = ret.returns_to_nav(
        ret.to_returns(prices=prices, is_log_returns=True).mean(1)).rename('realized')
    all_data = pd.concat([original, ew_prices], axis=1)
    plot_prices(all_data)


def estimate_ar_model(data: pd.Series,
                      title: str = None,
                      axs: List[plt.Subplot] = None,
                      **kwargs):
    ar_model = AutoReg(data, lags=1).fit()
    print(ar_model.summary())

    intercept, beta = ar_model.params[0], ar_model.params[1]
    prediction = np.zeros(len(data.index))
    prediction[0] = data[0]
    for t in np.arange(1, len(data.index)):
        prediction[t] = beta*data[t-1] + intercept

    prediction = pd.Series(prediction, index=data.index, name='prediction')
    joint_data = pd.concat([data, prediction], axis=1)

    if axs is None:
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 1, figsize=(12, 9), tight_layout=True)

    plot_pacf(data, lags=10, title=f"{title}: Partial Auto-Correlation", ax=axs[0])
    out = acf(data)
    print(out)

    pts.plot_time_series(df=joint_data,
                         legend_stats=pts.LegendStats.AVG_STD,
                         var_format='{:,.0f}',
                         title=f"{title}: One-Day Prediction",
                         ax=axs[1],
                         **kwargs)


def plot_ar_bootstrap(data: pd.Series,
                      title: str = None,
                      ax: plt.Subplot = None,
                      num_samples: int = 10,
                      seed: int = 2,
                      **kwargs
                      ):

    df_bts = bts.bootstrap_ar_process(data=data,
                                      bootsrap_output=bts.BootsrapOutput.SERIES_TO_DF,
                                      index_length=len(data.index),
                                      num_samples=num_samples,
                                      seed=seed)
    df_bts.index = data.index
    joint_data = pd.concat([data.rename('realized'), df_bts], axis=1)
    pts.plot_time_series(df=joint_data,
                         legend_stats=pts.LegendStats.AVG_STD,
                         var_format='{:,.0f}',
                         title=title,
                         y_limits=(0.0, None),
                         first_color_fixed=False,
                         ax=ax,
                         **kwargs)


def plot_joint_bootstrap():
    prices, fundamental_data = get_test_data()
    prices, fundamental_data = prices.iloc[:, 0], fundamental_data.iloc[:, 0]
    price_datas = {'price': prices.iloc[:, 0]}
    fundamental_datas = {'fundamental': fundamental_data}
    bootstrap_prices, bootstrap_fundamentals = bts.bootstrap_price_fundamental_data(price_datas=price_datas,
                                                                                    fundamental_datas=fundamental_datas,
                                                                                    bootsrap_output=bts.BootsrapOutput.SERIES_TO_DF,
                                                                                    index_length=len(prices.index),
                                                                                    num_samples=10,
                                                                                    block_size=10)
    bootstrap_prices = bootstrap_prices['price']
    bootstrap_prices.index = prices.index
    bootstrap_fundamentals = bootstrap_fundamentals['fundamental']
    bootstrap_fundamentals.index = prices.index
    join_price_data = pd.concat([prices, bootstrap_prices], axis=1)
    joint_fund_data = pd.concat([fundamental_data.rename('realized'), bootstrap_fundamentals], axis=1)

    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), tight_layout=True)

        pts.plot_time_series(df=joint_fund_data,
                             legend_stats=pts.LegendStats.AVG_STD,
                             var_format='{:,.0f}',
                             first_color_fixed=True,
                             ax=axs[0])
        plot_prices(prices=join_price_data,
                        is_log=True,
                        ax=axs[1])


def get_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    tickers = ['uniswap_msr', 'terra_msr']
    price_datas = []
    fund_datas = []
    for ticker in tickers:
        data = fu.load_df_from_csv(file_name=ticker, folder_name='crypto')
        price_datas.append(data['close'].rename(ticker))
        fund_datas.append(np.divide(data['volume'], data['close']).rename(ticker))
    prices = pd.concat(price_datas, axis=1).dropna()
    fund_data = pd.concat(fund_datas, axis=1).dropna()
    prices = prices.reindex(index=fund_data.index, method='ffill')
    return prices, fund_data


class UnitTests(Enum):
    PLOT_PRICE_BOOTSTRAP = 1
    PLOT_EW_INDEX = 2
    FIT_AR_MODEL = 3
    AR_RESIDUALS = 4
    PLOT_AR_BOOTSTRAP = 5
    PLOT_JOINT_BOOTSTRAP = 6


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    if unit_test == UnitTests.PLOT_PRICE_BOOTSTRAP:
        plot_price_bootrstrap(prices['SPY'])

    elif unit_test == UnitTests.PLOT_EW_INDEX:
        plot_ew_index_bootrstrap(prices)

    elif unit_test == UnitTests.FIT_AR_MODEL:
        data = fu.load_df_from_csv(file_name='UNI_Uniswap_cmc', folder_name='crypto/instruments')
        unit_volume = np.divide(data['volume'], data['close']).rename('unit volume')
        estimate_ar_model(data=unit_volume)

    elif unit_test == UnitTests.AR_RESIDUALS:
        prices, fund_data = get_test_data()
        residuals, intercept, beta = bts.compute_ar_residuals(data=fund_data)
        residuals = pd.DataFrame(residuals, index=fund_data.index, columns=fund_data.columns)
        pts.plot_time_series(residuals,
                             legend_stats=pts.LegendStats.AVG_STD,
                             var_format='{:,.0f}')

    elif unit_test == UnitTests.PLOT_AR_BOOTSTRAP:
        prices, fund_data = get_test_data()
        data = fund_data.iloc[:, 1]
        plot_ar_bootstrap(data=data)

    elif unit_test == UnitTests.PLOT_JOINT_BOOTSTRAP:
        plot_joint_bootstrap()

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.FIT_AR_MODEL

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
