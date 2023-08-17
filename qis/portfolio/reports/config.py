"""
configuration for performance reports
"""
from typing import Dict, Any, Tuple
from qis import PerfParams, BenchmarkReturnsQuantileRegimeSpecs, TimePeriod
import yfinance as yf

# default params have no risk-free rate
PERF_PARAMS = PerfParams(freq='W-WED', freq_reg='W-WED', rates_data=None)
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='Q')


def fetch_default_perf_params() -> Tuple[PerfParams, BenchmarkReturnsQuantileRegimeSpecs]:
    """
    by default we use 3m US rate
    """
    rates_data = yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0
    perf_params = PerfParams(freq='W-WED', freq_reg='W-WED', rates_data=rates_data)
    regime_params = BenchmarkReturnsQuantileRegimeSpecs(freq='Q')

    return perf_params, regime_params


def fetch_default_report_kwargs(time_period: TimePeriod, long_threshold: float = 5.0) -> Dict[str, Any]:

    rates_data = yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0

    # use for number years > 5
    if time_period.get_time_period_an() > long_threshold:
        report_kwargs = dict(perf_params=PerfParams(freq='W-WED', freq_reg='Q', rates_data=rates_data),
                             regime_params=BenchmarkReturnsQuantileRegimeSpecs(freq='Q'),
                             x_date_freq='A',
                             date_format='%b-%y')

    else:
        report_kwargs = dict(perf_params=PerfParams(freq='W-WED', freq_reg='M', rates_data=rates_data),
                             regime_params=BenchmarkReturnsQuantileRegimeSpecs(freq='M'),
                             x_date_freq='Q',
                             date_format='%b-%y')

    return report_kwargs





