"""
configuration for performance reports
"""
from typing import Dict, Any, Tuple
from qis import PerfParams, BenchmarkReturnsQuantileRegimeSpecs, TimePeriod, PerfStat, PerformanceLabel
import yfinance as yf

# default params have no risk-free rate
PERF_PARAMS = PerfParams(freq='W-WED', freq_reg='W-WED', alpha_an_factor=52, rates_data=None)
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='QE')

PERF_COLUMNS = (PerfStat.TOTAL_RETURN,
                PerfStat.PA_RETURN,
                PerfStat.VOL,
                PerfStat.SHARPE_EXCESS,
                PerfStat.MAX_DD,
                PerfStat.SKEWNESS,
                PerfStat.ALPHA_AN,
                PerfStat.BETA,
                PerfStat.R2)


def fetch_default_perf_params() -> Tuple[PerfParams, BenchmarkReturnsQuantileRegimeSpecs]:
    """
    by default we use 3m US rate
    """
    rates_data = yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0
    if rates_data.empty:  # if online
        rates_data = None
    perf_params = PerfParams(freq='W-WED', freq_reg='W-WED', rates_data=rates_data)
    regime_params = BenchmarkReturnsQuantileRegimeSpecs(freq='QE')

    return perf_params, regime_params


def fetch_default_report_kwargs(time_period: TimePeriod,
                                long_threshold_years: float = 5.0
                                ) -> Dict[str, Any]:

    rates_data = yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0
    if rates_data.empty:  # if online
        rates_data = None

    # use for number years > 5
    if time_period.get_time_period_an() > long_threshold_years:
        report_kwargs = dict(perf_params=PerfParams(freq='W-WED', freq_reg='QE', rates_data=rates_data, alpha_an_factor=4.0),
                             regime_params=BenchmarkReturnsQuantileRegimeSpecs(freq='QE'),
                             perf_columns=PERF_COLUMNS,
                             short=True,  # ra columns
                             performance_label=PerformanceLabel.DETAILED_EXCESS_SHARPE,
                             heatmap_freq='YE',
                             x_date_freq='YE',
                             date_format='%b-%y')

    else:
        report_kwargs = dict(perf_params=PerfParams(freq='W-WED', freq_reg='ME', rates_data=rates_data, alpha_an_factor=12.0),
                             regime_params=BenchmarkReturnsQuantileRegimeSpecs(freq='ME'),
                             perf_columns=PERF_COLUMNS,
                             short=True,  # ra columns
                             performance_label=PerformanceLabel.DETAILED_EXCESS_SHARPE,
                             heatmap_freq='QE',
                             x_date_freq='QE',
                             date_format='%b-%y')

    return report_kwargs

