# packages
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
from bbg_fetch import fetch_field_timeseries_per_tickers
from enum import Enum

BENCHMARK = 'SPTGGUT Index'


def run_hedge_funds() -> pd.DataFrame:
    # 50/50
    tickers = {'HFRXGL Index': 'HFRXGL Index', 'NEIXCTA Index': 'NEIXCTA Index'}
    prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='PX_LAST').ffill()
    nav1 = qis.backtest_model_portfolio(prices=prices, weights=[0.5, 0.5], rebalancing_freq='QE', ticker=f"50/50 HFRXGL/NEIXCTA").get_portfolio_nav()

    tickers = {'HFRIFWI Index': 'HFRIFWI Index', 'NEIXCTA Index': 'NEIXCTA Index'}
    prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='PX_LAST').ffill()
    nav2 = qis.backtest_model_portfolio(prices=prices, weights=[0.5, 0.5], rebalancing_freq='QE', ticker=f"50/50 HFRIFWI/NEIXCTA").get_portfolio_nav()

    tickers = {
            BENCHMARK: BENCHMARK,
            'HFRXGL Index': 'HFRXGL Index',
            'HFRIFWI Index': 'HFRIFWI Index',
            'NEIXCTA Index': 'NEIXCTA Index'
        }

    prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='PX_LAST').ffill()
    prices = pd.concat([prices, nav1, nav2], axis=1)
    return prices


def run_ils() -> pd.DataFrame:
    tickers = {BENCHMARK: BENCHMARK,
               'SRGLTRR Index': 'Swiss re Index',
               'LGTIPBU LX Equity': 'LGT ILS Fund',
               'EHFI804 Index': 'Eurekahedge Index'}
    prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='PX_LAST').ffill()
    return prices


def run_re() -> pd.DataFrame:
    tickers = {BENCHMARK: BENCHMARK,
               'MXWO0RI Index': 'MXWO0RI Index',
               'SWIIT Index': 'SWIIT Index'}
    prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='PX_LAST').ffill()
    return prices


def run_em() -> pd.DataFrame:
    tickers = {BENCHMARK: BENCHMARK,
               'EMUSTRUU Index': 'EMUSTRUU Index',
               'LG20TRUH Index': 'LG20TRUH Index'}
    prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='PX_LAST').ffill()
    return prices


class LocalTests(Enum):
    HEDGE_FUNDS = 1
    ILS = 2
    RE = 3
    EM = 4


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


    if local_test == LocalTests.HEDGE_FUNDS:
        prices = run_hedge_funds()
        time_period = qis.TimePeriod('31Dec2004', '30Sep2025')

    elif local_test == LocalTests.ILS:
        prices = run_ils()
        time_period = qis.TimePeriod('31Dec2014', '30Sep2025')

    elif local_test == LocalTests.RE:
        prices = run_re()
        time_period = qis.TimePeriod('31Dec2014', '30Sep2025')

    elif local_test == LocalTests.EM:
        prices = run_em()
        time_period = qis.TimePeriod('31Dec2014', '30Sep2025')

    else:
        raise NotImplementedError

    kwargs = qis.fetch_factsheet_config_kwargs(factsheet_config=qis.FACTSHEET_CONFIG_MONTHLY_DATA_LONG_PERIOD,
                                               add_rates_data=False,
                                               override=dict(digits_to_show=1, sharpe_digits=2))

    fig = qis.generate_multi_asset_factsheet(prices=prices,
                                             benchmark=BENCHMARK,
                                             time_period=time_period,
                                             **kwargs)
    qis.save_figs_to_pdf(figs=[fig],
                         file_name=f"bbg_multiasset_report", orientation='landscape',
                         local_path=qis.get_output_path())

    plt.close('all')


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.EM)
