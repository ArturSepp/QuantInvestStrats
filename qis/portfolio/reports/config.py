"""
configuration for performance reports
"""
from typing import Dict, Any, Tuple
from qis import PerfParams, BenchmarkReturnsQuantileRegimeSpecs, TimePeriod, PerfStat, PerfStatsLabels
import yfinance as yf

# default params have no risk-free rate
PERF_PARAMS = PerfParams(freq='W-WED', freq_reg='W-WED', alpha_an_factor=52, rates_data=None)
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='QE')

PERF_COLUMNS = (PerfStat.TOTAL_RETURN,
                PerfStat.PA_RETURN,
                PerfStat.VOL,
                PerfStat.SHARPE_EXCESS,
                PerfStat.MAX_DD,
                PerfStat.MAX_DD_VOL,
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
                             perf_stats_labels=(PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_EXCESS, ),
                             heatmap_freq='YE',
                             x_date_freq='YE',
                             date_format='%b-%y')

    else:
        report_kwargs = dict(perf_params=PerfParams(freq='W-WED', freq_reg='ME', rates_data=rates_data, alpha_an_factor=12.0),
                             regime_params=BenchmarkReturnsQuantileRegimeSpecs(freq='ME'),
                             perf_columns=PERF_COLUMNS,
                             short=True,  # ra columns
                             perf_stats_labels=(PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_EXCESS, ),
                             heatmap_freq='QE',
                             x_date_freq='QE',
                             date_format='%b-%y')

    return report_kwargs

# for pybloqs
margin_top = 1.0
margin_bottom = 1.0
line_height = 0.99
font_family = 'Calibri'
KWARGS_SUPTITLE = {'title_wrap': True, 'text_align': 'center', 'color': 'blue', 'font_size': "12px", 'font-weight': 'normal',
                   'title_level': 1, 'line_height': 0.7, 'inherit_cfg': False,
                   'margin_top': 0, 'margin_bottom': 0,
                   'font-family': 'sans-serif'}
KWARGS_TITLE = {'title_wrap': True, 'text_align': 'left', 'color': 'blue', 'font_size': "12px",
                'title_level': 1, 'line_height': line_height, 'inherit_cfg': False,
                'margin_top': margin_top,  'margin_bottom': margin_bottom,
                'font-family': font_family}
KWARGS_DESC = {'title_wrap': True, 'text_align': 'left', 'font_size': "12px", 'font-weight': 'normal',
               'title_level': 2, 'line_height': line_height, 'inherit_cfg': False,
               'margin_top': margin_top, 'margin_bottom': margin_bottom,
               'font-family': font_family}
KWARGS_TEXT = {'title_wrap': True, 'text_align': 'left', 'font_size': "12px", 'font-weight': 'normal',
               'title_level': 2, 'line_height': line_height, 'inherit_cfg': False,
               'margin_top': margin_top, 'margin_bottom': margin_bottom,
               'font-family': font_family}
KWARGS_FIG = {'title_wrap': True, 'text_align': 'left', 'font_size': "12px",
              'title_level': 2, 'line_height': line_height, 'inherit_cfg': False,
              'margin_top': margin_top, 'margin_bottom': margin_bottom,
              'font-family': font_family}
KWARGS_FOOTNOTE = {'title_wrap': True, 'text_align': 'left', 'font_size': "12px", 'font-weight': 'normal',
                   'title_level': 3, 'line_height': line_height, 'inherit_cfg': False,
                   'margin_top': margin_top, 'margin_bottom': margin_bottom,
                   'font-family': font_family}