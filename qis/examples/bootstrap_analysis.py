
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from enum import Enum

# qis
import yfinance as yf
import qis

SEED = 1


def plot_bootsrap_paths(prices: pd.Series,
                        num_samples: int = 10,
                        block_size: int = 20,
                        nlags: int = 20,
                        ) -> List[plt.Figure]:
    """
    create analysis figure
    """
    figs = []
    bootstrap_prices = qis.bootstrap_price_data(prices=prices,
                                                bootsrap_type=qis.BootsrapType.STATIONARY,
                                                bootsrap_output=qis.BootsrapOutput.SERIES_TO_DF,
                                                num_samples=num_samples,
                                                block_size=block_size,
                                                index_length=len(prices.index),
                                                seed=SEED,
                                                init_to_end=False)
    bootstrap_prices.index = prices.index
    prices1 = pd.concat([bootstrap_prices, prices], axis=1)

    colors = num_samples * ['gray']
    colors1 = colors + ['red']
    kwargs = dict(x_date_freq='YE', legend_loc=None)
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(10, 7))
        qis.set_suptitle(fig, title='Log-performance and drawdowns of realized (red) and bootsrapped paths (grey)')
        figs.append(fig)
        qis.plot_prices_with_dd(prices=prices1,
                                is_log=True,
                                colors=colors1,
                                axs=axs,
                                **kwargs)

    log_returns = qis.to_returns(prices1, is_log_returns=True, is_first_zero=True)
    log_returns2 = np.square(log_returns)
    span = np.maximum(block_size, 5)
    ewma_vols = qis.compute_ewm_vol(data=log_returns,
                                    span=span,
                                    af=252)
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        qis.set_suptitle(fig, title=f"EWMA-{span} span volatility of realized (red) and bootsrapped paths (gray)")
        figs.append(fig)
        qis.plot_time_series(df=ewma_vols,
                             colors=colors1,
                             var_format='{:,.0%}',
                             ax=ax,
                             **kwargs)

    acfs, m_acf, std_acf = qis.estimate_path_acf(log_returns, is_pacf=True, nlags=nlags)
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        qis.set_suptitle(fig, title=f"Auto-correlation of returns of realized (red) and bootsrapped paths (grey)")
        figs.append(fig)
        qis.plot_line(df=acfs,
                      colors=colors1,
                      ax=ax,
                      **kwargs)

    acfs, m_acf, std_acf = qis.estimate_path_acf(log_returns2, is_pacf=True, nlags=nlags)
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        qis.set_suptitle(fig, title=f"Auto-correlation of squared returns of realized (red) and bootsrapped paths (grey)")
        figs.append(fig)
        qis.plot_line(df=acfs,
                      colors=colors1,
                      xlabel='lag',
                      ax=ax,
                      **kwargs)

    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        qis.set_suptitle(fig, title=f"Realized auto-correlation of squared returns (red) and Boxplot of auto-correlation of bootsrapped paths (grey)")
        figs.append(fig)
        qis.df_boxplot_by_index(df=acfs.drop(prices.name, axis=1),
                                legend_loc=None,
                                colors=colors,
                                xlabel='lag',
                                showmedians=True,
                                ax=ax)
        # add reasized labels
        qis.add_scatter_points(label_x_y=[(lag, v) for lag, v in acfs[prices.name].to_dict().items()],
                               color='red',
                               ax=ax)
        qis.set_legend(ax=ax,
                       labels=[prices.name],
                       markers=["*"],
                       colors=['red'],
                       handlelength=0)

        return figs


def plot_autocorr_in_block_size(prices: pd.Series,
                                block_sizes: List[int] = (5, 10, 20, 60),
                                num_samples: int = 1000,
                                nlags: int = 20,
                                ) -> List[plt.Figure]:
    """
    use different block_size to check for autocorrelation
    """
    figs = []
    for idx, block_size in enumerate(block_sizes):
        bootstrap_prices = qis.bootstrap_price_data(prices=prices,
                                                    bootsrap_type=qis.BootsrapType.STATIONARY,
                                                    bootsrap_output=qis.BootsrapOutput.SERIES_TO_DF,
                                                    num_samples=num_samples,
                                                    block_size=block_size,
                                                    index_length=len(prices.index),
                                                    seed=SEED,
                                                    init_to_end=False)
        bootstrap_prices.index = prices.index
        prices1 = pd.concat([bootstrap_prices, prices], axis=1)
        log_returns = qis.to_returns(prices1, is_log_returns=True, is_first_zero=True)
        log_returns2 = np.square(log_returns)
        acfs, m_acf, std_acf = qis.estimate_path_acf(log_returns2, is_pacf=True, nlags=nlags)

        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            qis.set_suptitle(fig, title=f"Realized auto-correlation of squared returns (red) and Boxplot of "
                                        f"auto-correlation of bootsrapped paths (grey) for block_size={block_size:0.0f}")
            figs.append(fig)
            qis.df_boxplot_by_index(df=acfs.drop(prices.name, axis=1),
                                    legend_loc=None,
                                    colors=num_samples * ['gray'],
                                    title=f"block_size={block_size:0.0f}",
                                    xlabel='lag',
                                    showmedians=True,
                                    ax=ax)
            # add reasized labels
            qis.add_scatter_points(label_x_y=[(lag, v) for lag, v in acfs[prices.name].to_dict().items()],
                                   color='red',
                                   ax=ax)
            qis.set_legend(ax=ax,
                           labels=[prices.name],
                           markers=["*"],
                           colors=['red'],
                           handlelength=0)
    return figs


class UnitTests(Enum):
    PLOT_BOOTSRAPPED_PRICES = 1
    PLOT_AUTOCORR_BLOCKSIZES = 2


def run_unit_test(unit_test: UnitTests):
    # to save pds
    LOCAL_PATH = "C://Users//artur//OneDrive//analytics//outputs//"

    # download spy prices
    prices = yf.download(tickers=['SPY'], start=None, end=None, ignore_tz=True)['Adj Close'].rename('Realised')

    if unit_test == UnitTests.PLOT_BOOTSRAPPED_PRICES:
        # use small number of num_samples for illustration
        figs = plot_bootsrap_paths(prices=prices,
                                   block_size=30,
                                   num_samples=50)
        qis.save_figs_to_pdf(figs, file_name='bootstrap_illustrations', local_path=LOCAL_PATH)

    elif unit_test == UnitTests.PLOT_AUTOCORR_BLOCKSIZES:
        # block_size = 1 corresponds to iid sampling
        figs = plot_autocorr_in_block_size(prices=prices,
                                           block_sizes=[1, 2, 5, 10, 20, 40, 60, 120, 180],
                                           num_samples=1000)
        qis.save_figs_to_pdf(figs, file_name='bootstrap_block_sizes', local_path=LOCAL_PATH)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_AUTOCORR_BLOCKSIZES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
