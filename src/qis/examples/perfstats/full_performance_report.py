"""
Five-figure performance report on a price panel: risk-adjusted table,
periodic returns table, cumulative price, price + drawdown, regime-conditional
scatter.

The layout itself is reused across examples and lives in
``qis.examples._helpers.reporting_helpers.generate_performance_report``.
This file just shows how to drive it on several universe configurations
(ETFs, crypto, trend-following ETFs, vol ETFs, etc.).

Run with:
    python -m qis.examples.perfstats.full_performance_report
"""
import matplotlib.pyplot as plt
import yfinance as yf
from enum import Enum

import qis as qis
from qis.examples._helpers.reporting_helpers import generate_performance_report


class Universe(Enum):
    ETFS = 1
    CRYPTO = 2
    TREND_FOLLOWING_ETFS = 3
    DIVERSIFIED_ETFS = 4
    COMMODITY_ETFS = 5
    VOL_ETFS = 6


def run(universe: Universe = Universe.ETFS,
        is_long_period: bool = True) -> None:
    """Pull a yfinance universe, define perf params, render the report."""
    ust_3m_rate = yf.download('^IRX', start="2003-12-31", end=None,
                              ignore_tz=True, auto_adjust=True)['Close'].dropna() / 100.0

    if universe == Universe.ETFS:
        regime_benchmark = 'SPY'
        tickers = [regime_benchmark, 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'SHY', 'GLD']
    elif universe == Universe.CRYPTO:
        regime_benchmark = 'BTC-USD'
        tickers = [regime_benchmark, 'SPY', 'TLT', 'ETH-USD', 'SOL-USD']
    elif universe == Universe.TREND_FOLLOWING_ETFS:
        regime_benchmark = 'SPY'
        tickers = [regime_benchmark, 'DBMF', 'WTMF', 'CTA']
    elif universe == Universe.DIVERSIFIED_ETFS:
        regime_benchmark = 'AOR'
        tickers = [regime_benchmark, 'SPY', 'PEX', 'PSP', 'GSG', 'COMT', 'REET', 'REZ']
    elif universe == Universe.COMMODITY_ETFS:
        regime_benchmark = 'AOR'
        tickers = [regime_benchmark, 'SPY', 'GLD', 'GSG', 'COMT', 'PDBC']
    elif universe == Universe.VOL_ETFS:
        regime_benchmark = 'SPY'
        tickers = [regime_benchmark, 'SVOL']
    else:
        raise NotImplementedError(f"unknown universe {universe}")

    if is_long_period:
        time_period = qis.TimePeriod('16Oct2014', None)
        perf_params = qis.PerfParams(freq='W-WED', freq_reg='ME',
                                     freq_drawdown='B', rates_data=ust_3m_rate)
        kwargs = dict(x_date_freq='YE', heatmap_freq='YE', date_format='%Y',
                      perf_params=perf_params)
    else:
        time_period = qis.TimePeriod('31Dec2022', None)
        perf_params = qis.PerfParams(freq='W-WED', freq_reg='W-WED',
                                     freq_drawdown='B', rates_data=ust_3m_rate)
        kwargs = dict(x_date_freq='ME', heatmap_freq='ME', date_format='%b-%y',
                      perf_params=perf_params)

    prices = yf.download(tickers, start="2003-12-31", end=None,
                         ignore_tz=True, auto_adjust=True)['Close'][tickers].dropna()
    prices = time_period.locate(prices)

    generate_performance_report(prices=prices,
                                regime_benchmark=regime_benchmark,
                                **kwargs)
    plt.show()


if __name__ == '__main__':
    run(universe=Universe.ETFS)
