"""
configuration for performance reports
"""
from typing import Dict, Any, Tuple, NamedTuple, Optional, List
from qis import PerfParams, BenchmarkReturnsQuantileRegimeSpecs, TimePeriod, PerfStat, PerfStatsLabels
import yfinance as yf

# default params have no risk-free rate
PERF_PARAMS = PerfParams(freq='W-WED', freq_reg='W-WED', alpha_an_factor=52, rates_data=None)
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='QE')

PERF_COLUMNS = (PerfStat.TOTAL_RETURN,
                PerfStat.PA_RETURN,
                PerfStat.VOL,
                PerfStat.SHARPE_RF0,
                PerfStat.SHARPE_EXCESS,
                PerfStat.MAX_DD,
                PerfStat.MAX_DD_VOL,
                PerfStat.SKEWNESS,
                PerfStat.ALPHA_AN,
                PerfStat.BETA,
                PerfStat.R2)


class FactsheetConfig(NamedTuple):
    """
    enumerate key parameters for report comps and visuals on different time period or data frequency
    default is for daily data with at least 10y
    """
    freq: str = 'W-WED'  # for vols
    freq_drawdown: str = 'D'
    freq_reg: str = 'W-WED'  # for beta regressions
    alpha_an_factor: float = 52 # for W-WED returns
    regime_freq: str = 'QE'  # for regime frequency
    sharpe_rolling_window: int = 156 # 3y of weekly returns
    sharpe_freq: str = 'W-WED'
    sharpe_title: Optional[str] = None
    turnover_rolling_period: int = 260  # turnover = turnover.rolling(turnover_roll_period).sum()
    turnover_title: Optional[str] = None
    cost_rolling_period: int = 260  # turnover = turnover.rolling(turnover_roll_period).sum()
    cost_title: Optional[str] = None
    is_norm_costs: bool = True  # for long-only protfolio use nrmalised costs
    factor_beta_span: int = 52  # to compute rolling beta
    beta_freq: str = 'W-WED'
    factor_beta_title: Optional[str] = None
    exposures_freq: str = 'W-WED'  # for plotting strategy exposures
    # general data
    perf_columns: List[PerfStat] = PERF_COLUMNS
    perf_stats_labels: List[PerfStat] = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0,)
    short: bool = True  # ra columns
    # next depend on report time period
    heatmap_freq: str = 'YE'  # for heatmap
    x_date_freq: str = 'YE'  # for time series plots
    date_format: str = '%b-%y' # for dates


# create enumerations
FACTSHEET_CONFIG_DAILY_DATA_LONG_PERIOD = FactsheetConfig()

FACTSHEET_CONFIG_DAILY_DATA_SHORT_PERIOD = FactsheetConfig(heatmap_freq='QE',
                                                           x_date_freq='QE')


FACTSHEET_CONFIG_MONTHLY_DATA_LONG_PERIOD = FactsheetConfig(freq='ME',
                                                            freq_drawdown='ME',
                                                            freq_reg='ME',
                                                            alpha_an_factor=12,
                                                            regime_freq='QE',
                                                            sharpe_rolling_window=36,
                                                            sharpe_freq='ME',
                                                            sharpe_title=None,
                                                            turnover_rolling_period=12,
                                                            turnover_title=None,
                                                            cost_rolling_period=12,
                                                            cost_title=None,
                                                            is_norm_costs=True,
                                                            factor_beta_span=36,
                                                            beta_freq='ME',
                                                            factor_beta_title=None,
                                                            exposures_freq='ME')

FACTSHEET_CONFIG_MONTHLY_DATA_SHORT_PERIOD = FACTSHEET_CONFIG_MONTHLY_DATA_LONG_PERIOD


def fetch_factsheet_config_kwargs(factsheet_config: FactsheetConfig = FACTSHEET_CONFIG_DAILY_DATA_LONG_PERIOD,
                                  add_rates_data: bool = True
                                  ) -> Dict[str, Any]:
    """
    return kwargs for factsheet reports
    """
    rates_data = None
    if add_rates_data:
        rates_data = yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0
        if rates_data.empty:  # if online
            rates_data = None

    perf_params = PerfParams(freq=factsheet_config.freq,
                             freq_drawdown=factsheet_config.freq_drawdown,
                             freq_reg=factsheet_config.freq_reg,
                             alpha_an_factor=factsheet_config.alpha_an_factor,
                             rates_data=rates_data)
    regime_params = BenchmarkReturnsQuantileRegimeSpecs(freq=factsheet_config.regime_freq)
    kwargs = factsheet_config._asdict()
    kwargs.pop('freq')  # remove frequency as some methods have default freq
    kwargs.update(dict(perf_params=perf_params, regime_params=regime_params))
    return kwargs


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
                                is_daily: bool = True,
                                long_threshold_years: float = 5.0,
                                add_rates_data: bool = True
                                ) -> Dict[str, Any]:

    # use for number years > 5
    if time_period.get_time_period_an() > long_threshold_years:
        if is_daily:
            factsheet_config = FACTSHEET_CONFIG_DAILY_DATA_LONG_PERIOD
        else:
            factsheet_config = FACTSHEET_CONFIG_MONTHLY_DATA_LONG_PERIOD
    else:
        if is_daily:
            factsheet_config = FACTSHEET_CONFIG_DAILY_DATA_SHORT_PERIOD
        else:
            factsheet_config = FACTSHEET_CONFIG_MONTHLY_DATA_SHORT_PERIOD

    kwargs = fetch_factsheet_config_kwargs(factsheet_config=factsheet_config,
                                           add_rates_data=add_rates_data)
    return kwargs


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
