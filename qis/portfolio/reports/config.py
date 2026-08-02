"""
configuration for performance reports.

Factsheet settings are driven by two independent axes:

  1. data sampling frequency (ReportingFrequency) -> every freq_* field uses the frequency's own
     pandas grid; rolling windows / EWM spans are annualised from periods-per-year.
  2. reported time span (long vs short) -> window lengths, heatmap_freq, x_date_freq and freq_regime:
       - long  : vol/var/sharpe 3y (DAILY 1y, a 3y span is too slow on daily returns), beta 3y,
                 turnover/cost 1y; heatmap / x-axis 'YE', regime 'QE'
       - short : ALL rolling windows 1y (so ~3y reports still populate); turnover/cost 1y;
                 heatmap / x-axis 'ME', regime 'ME'

Resulting presets (counts are in periods of the base grid; LONG | SHORT where they differ):

    Frequency   grid     vol/var/sharpe   factor beta    turnover/cost   regime
    DAILY       B        260 | 260        780 | 260      260             QE | ME
    WEEKLY      W-WED    156 |  52        156 |  52       52             QE | ME
    MONTHLY     ME        36 |  12         36 |  12       12             QE | ME
    QUARTERLY   QE        12 |   4         12 |   4        4             QE | ME

vol_rolling_window and var_span are EWM spans; sharpe_rolling_window is a rolling window;
turnover/cost_rolling_period are rolling-sum windows.

The FactsheetConfig field schema is the public kwargs contract: fetch_factsheet_config_kwargs()
spreads FactsheetConfig._asdict() into the generate_*_factsheet(...) calls, so field names stay stable.

Two page-level guards live here beside the settings, both called from the generate_* entry
points: validate_reporting_frequency() on the data, and validate_legend_capacity() on the page
geometry - a panel legend is a fixed height per series and overflowing it collapses the layout of
the whole page, not only of the panel that overflows.
"""
import math
import warnings
from enum import Enum
from typing import Dict, Any, Tuple, NamedTuple, Optional, List, Union
import pandas as pd
from qis import PerfParams, BenchmarkReturnsQuantilesRegime, TimePeriod, PerfStat, update_kwargs
from qis.utils.annualisation import (get_annualization_factor,
                                     infer_data_periods_per_year, infer_data_frequency_label)

# default params have no risk-free rate
PERF_PARAMS = PerfParams(freq='W-WED', freq_reg='W-WED', rates_data=None)
regime_classifier = BenchmarkReturnsQuantilesRegime(freq='QE')

PERF_COLUMNS_RF0 = (PerfStat.TOTAL_RETURN,
                    PerfStat.PA_RETURN,
                    PerfStat.VOL,
                    PerfStat.SHARPE_RF0,
                    PerfStat.MAX_DD,
                    PerfStat.MAX_DD_VOL,
                    PerfStat.SKEWNESS,
                    PerfStat.ALPHA_AN,
                    PerfStat.BETA,
                    PerfStat.R2,
                    PerfStat.ALPHA_PVALUE)


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
                PerfStat.R2,
                PerfStat.ALPHA_PVALUE)


class ReportingFrequency(Enum):
    """
    the grid a factsheet is computed on.

    Selecting this recalibrates the whole report, not just the x axis: rolling windows, EWM
    spans, regression frequency, regime classification and annualisation are all derived from
    it, so the same portfolio reported daily and monthly yields different volatilities and
    Sharpe ratios by construction rather than by error. Each panel states the frequency it was
    computed at. See ``qis/docs/reporting_frequencies.md``.

    The value of each member is the pandas resampling rule.

    Attributes:
        DAILY: business days, 'B', 260 periods per year
        WEEKLY: Wednesdays, 'W-WED', 52 per year
        MONTHLY: month ends, 'ME', 12 per year
        QUARTERLY: quarter ends, 'QE', 4 per year
    """
    DAILY = 'B'
    WEEKLY = 'W-WED'
    MONTHLY = 'ME'
    QUARTERLY = 'QE'


# periods per year per reporting frequency (for annualising rolling windows / EWM spans)
_PERIODS_PER_YEAR: Dict[ReportingFrequency, int] = {
    ReportingFrequency.DAILY: 260,
    ReportingFrequency.WEEKLY: 52,
    ReportingFrequency.MONTHLY: 12,
    ReportingFrequency.QUARTERLY: 4,
}

# tier -> human name, used only for the validate_reporting_frequency error message
# (the periods-per-year classifier and label helper now live in qis.utils.annualisation)
_TIER_NAME: Dict[float, str] = {260.0: 'daily', 52.0: 'weekly', 12.0: 'monthly',
                                4.0: 'quarterly', 1.0: 'annual'}


def validate_reporting_frequency(data: Union[pd.Series, pd.DataFrame, pd.DatetimeIndex],
                                 reporting_frequency: Union[ReportingFrequency, str]
                                 ) -> None:
    """
    raise ValueError if `reporting_frequency` is FINER than the native sampling frequency of
    `data` (e.g. WEEKLY or DAILY reporting on MONTHLY prices). Up-sampling returns to a higher
    frequency than the data carries no information and yields meaningless statistics. Equal or
    coarser reporting (e.g. QUARTERLY on MONTHLY) is allowed and simply down-samples.

    reporting_frequency may be a ReportingFrequency or a pandas freq string ('B', 'W-WED', ...).
    """
    if isinstance(reporting_frequency, ReportingFrequency):
        report_freq_str, report_name = reporting_frequency.value, reporting_frequency.name
    else:
        report_freq_str = report_name = str(reporting_frequency)
    report_ppy = get_annualization_factor(report_freq_str)
    data_ppy = infer_data_periods_per_year(data)
    if report_ppy > data_ppy:
        data_name = _TIER_NAME.get(data_ppy, f"~{data_ppy:g}/yr")
        raise ValueError(
            f"reporting_frequency={report_name} (~{report_ppy:g} periods/year) is finer than the input "
            f"data sampling frequency (~{data_name}, ~{data_ppy:g} periods/year). Up-sampling returns to a "
            f"higher frequency than the data produces meaningless statistics. Request {data_name} reporting "
            f"or coarser, or supply data sampled at {report_name} frequency or finer."
        )


# legend geometry of a factsheet page. A panel legend is a FIXED-height decoration carrying one
# row per series, measured on the labelspacing=0.2 used by qis.plots.utils.set_legend:
#     legend height (inches) ~ fontsize * (0.0086 + 0.01576 * n_entries)
# the intercept is frame padding, under a tenth of a row, and is dropped below.
LEGEND_ROW_HEIGHT_PER_FONTSIZE = 0.01576  # inches per legend entry per point of fontsize
PANEL_DECORATION_HEIGHT = 0.43            # inches: panel title plus x tick labels


def estimate_legend_capacity(figsize: Tuple[float, float],
                             fontsize: float,
                             panel_rows: int,
                             gridspec_rows: int
                             ) -> int:
    """
    number of legend entries a factsheet panel carries before the page layout degrades.

    matplotlib's ``constrained_layout`` counts the part of a legend that spills out of its axes
    as a layout margin. A panel legend is a fixed height and grows with the number of series, so
    once it exceeds the panel's cell the solver drives the axes height to zero, warns, and
    disables itself for the WHOLE figure: every panel on the page then reverts to the raw
    gridspec and the page reads as collapsed, not only the panel that overflowed.

    Args:
        figsize: figure size in inches, (width, height)
        fontsize: base font size of the report, in points
        panel_rows: gridspec rows spanned by one time-series panel
        gridspec_rows: total gridspec rows on the page

    Returns:
        largest number of legend entries leaving the panel axes at full height; 0 when the panel
        is too small to carry a legend at all

    Raises:
        ValueError: if fontsize, panel_rows or gridspec_rows is not positive
    """
    if fontsize <= 0.0:
        raise ValueError(f"fontsize must be positive, got {fontsize!r}")
    if panel_rows <= 0 or gridspec_rows <= 0:
        raise ValueError(f"panel_rows and gridspec_rows must be positive, "
                         f"got {panel_rows!r} and {gridspec_rows!r}")
    usable_height = figsize[1] * panel_rows / gridspec_rows - PANEL_DECORATION_HEIGHT
    if usable_height <= 0.0:
        return 0
    return int(usable_height / (LEGEND_ROW_HEIGHT_PER_FONTSIZE * fontsize))


def validate_legend_capacity(n_legend_entries: int,
                             figsize: Tuple[float, float],
                             fontsize: float,
                             panel_rows: int,
                             gridspec_rows: int,
                             report_name: str = 'factsheet'
                             ) -> int:
    """
    warn when a page carries more series than its panel legends hold.

    A warning rather than a ValueError: the page still renders, the caller may not care on a
    screening run, and pages between the capacity and the point where constrained_layout gives
    up render acceptably today. Nothing is dropped and no number changes - see
    :func:`estimate_legend_capacity` for the geometry.

    Args:
        n_legend_entries: series drawn in one time-series panel, benchmarks included
        figsize: figure size in inches, (width, height)
        fontsize: base font size of the report, in points
        panel_rows: gridspec rows spanned by one time-series panel
        gridspec_rows: total gridspec rows on the page
        report_name: report named in the warning message

    Returns:
        the capacity, so a caller can act on it rather than only warn
    """
    capacity = estimate_legend_capacity(figsize=figsize, fontsize=fontsize,
                                        panel_rows=panel_rows, gridspec_rows=gridspec_rows)
    if n_legend_entries > capacity:
        usable_height = figsize[1] * panel_rows / gridspec_rows - PANEL_DECORATION_HEIGHT
        # round the two suggestions AWAY from the constraint: a suggestion that still trips this
        # warning when the caller applies it is worse than no suggestion
        fontsize_fit = math.floor(10.0 * usable_height
                                  / (LEGEND_ROW_HEIGHT_PER_FONTSIZE * n_legend_entries)) / 10.0
        figsize_fit = math.ceil(10.0 * (n_legend_entries * LEGEND_ROW_HEIGHT_PER_FONTSIZE * fontsize
                                        + PANEL_DECORATION_HEIGHT)
                                * gridspec_rows / panel_rows) / 10.0
        warnings.warn(
            f"{report_name}: {n_legend_entries} series exceed the panel legend capacity of "
            f"{capacity} at figsize={figsize} and fontsize={fontsize}. The legends will squeeze "
            f"the panel axes and matplotlib will collapse the page layout. Report fewer series, "
            f"or pass fontsize={fontsize_fit:.1f}, or pass figsize=({figsize[0]}, "
            f"{figsize_fit:.1f}).",
            stacklevel=3
        )
    return capacity


# horizons (in years) used to calibrate the rolling windows / EWM spans below.
# LONG period: vol/var/sharpe use 3y everywhere except DAILY (1y, a 3y span is too slow on daily
#              returns); factor beta uses 3y.
# SHORT period: every rolling window collapses to 1y so that ~3y reports still populate
#              (a 3y window on 3y of data yields no full window).
# Turnover/cost are always annual (1y) on both.
_RISK_WINDOW_YEARS_LONG: Dict[ReportingFrequency, int] = {
    ReportingFrequency.DAILY: 1,
    ReportingFrequency.WEEKLY: 3,
    ReportingFrequency.MONTHLY: 3,
    ReportingFrequency.QUARTERLY: 3,
}
_BETA_WINDOW_YEARS_LONG = 3      # factor beta span (long period)
_SHORT_WINDOW_YEARS = 1         # short period: all rolling windows -> 1y
_TURNOVER_WINDOW_YEARS = 1      # turnover and cost -> annual numbers (both spans)

# regime classification frequency depends on the reported span (long vs short)
_FREQ_REGIME_LONG = 'QE'
_FREQ_REGIME_SHORT = 'ME'


class FactsheetConfig(NamedTuple):
    """
    key parameters for report comps and visuals on different time period or data frequency.
    See the module header for the full calibration table and the two axes (data frequency,
    reported span) that drive these settings. Presets are built by make_factsheet_config().

    The field schema below MUST stay stable: fetch_factsheet_config_kwargs() spreads
    FactsheetConfig._asdict() straight into the downstream generate_*_factsheet(...) calls,
    so every field name here is part of the public kwargs contract.

    The defaults are the DAILY / long-period preset, i.e.
        FactsheetConfig() == make_factsheet_config(ReportingFrequency.DAILY, is_long_period=True)
                          == FACTSHEET_CONFIG_DAILY_DATA_LONG_PERIOD
    """
    freq: str = 'B'  # for trading stats
    vol_freq: str = 'B'
    vol_rolling_window: int = 260  # DAILY: 1y of daily returns (EWM span)
    freq_drawdown: str = 'B'
    freq_reg: str = 'B'  # for beta regressions
    freq_var: str = 'B'  # for var computations
    var_span: float = 260.0  # DAILY: 1y of daily returns (EWM span)
    freq_regime: str = 'QE'  # for regime frequency (long period)
    sharpe_rolling_window: int = 260  # DAILY: 1y of daily returns
    freq_sharpe: str = 'B'  # for rolling sharpe
    turnover_rolling_period: int = 260  # annual: turnover.rolling(turnover_rolling_period).sum()
    freq_turnover: str = 'B'   # daily freq
    cost_rolling_period: int = 260  # annual: cost.rolling(cost_rolling_period).sum()
    freq_cost: Optional[str] = 'B'  # for rolling costs
    is_unit_based_traded_volume: bool = True  # for long-only protfolio use nrmalised costs
    factor_beta_span: int = 780  # 3y of daily returns (EWM span)
    freq_beta: str = 'B'  # for scatter plot
    weights_freq: str = 'B'  # for plotting strategy exposures
    corr_freq: str = 'B'  # for correlations table
    # general data
    perf_columns: List[PerfStat] = PERF_COLUMNS_RF0
    perf_stats_labels: List[PerfStat] = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0,)
    short: bool = True  # ra columns
    # next depend on report time period
    heatmap_freq: str = 'YE'  # for heatmap
    x_date_freq: str = 'YE'  # for time series plots
    date_format: str = '%b-%y'  # for dates
    digits_to_show: int = 2  # for pa table
    sharpe_digits: int = 2


# ----------------------------------------------------------------------------------------------------
# single source of truth: data-frequency driven statistic settings
# ----------------------------------------------------------------------------------------------------
def _frequency_stat_spec(reporting_frequency: ReportingFrequency,
                         is_long_period: bool
                         ) -> Dict[str, Any]:
    """
    all settings that depend on the data sampling frequency and the reported span.
    every freq_* field uses the frequency's own grid; windows are annualised from _PERIODS_PER_YEAR.
    LONG period uses the per-frequency risk horizon (3y, DAILY 1y) and a 3y beta span;
    SHORT period collapses every rolling window (vol/var/sharpe/beta) to 1y so ~3y reports populate.
    display/binning fields (heatmap_freq, x_date_freq) and freq_regime are NOT set here;
    they belong to the long/short axis and are added by _display_overrides().
    """
    grid = reporting_frequency.value  # 'B' / 'W-WED' / 'ME' / 'QE'
    n = _PERIODS_PER_YEAR[reporting_frequency]
    if is_long_period:
        risk_window = _RISK_WINDOW_YEARS_LONG[reporting_frequency] * n  # vol / var / sharpe
        beta_span = _BETA_WINDOW_YEARS_LONG * n
    else:  # short period: all rolling windows -> 1y
        risk_window = beta_span = _SHORT_WINDOW_YEARS * n
    turnover_period = _TURNOVER_WINDOW_YEARS * n
    return dict(
        freq=grid, vol_freq=grid,
        vol_rolling_window=risk_window,
        freq_drawdown=grid, freq_reg=grid,
        freq_var=grid, var_span=float(risk_window),
        sharpe_rolling_window=risk_window, freq_sharpe=grid,
        turnover_rolling_period=turnover_period, freq_turnover=grid,
        cost_rolling_period=turnover_period, freq_cost=grid,
        factor_beta_span=beta_span, freq_beta=grid,
        weights_freq=grid, corr_freq=grid,
    )


def _display_overrides(is_long_period: bool) -> Dict[str, Any]:
    """
    long/short axis: heatmap_freq, x_date_freq and freq_regime depend on the reported span.
    long period  -> annual heatmap / x-axis ('YE'), quarterly regime ('QE').
    short period -> monthly heatmap / x-axis ('ME'), monthly regime ('ME').
    """
    binning = 'YE' if is_long_period else 'ME'
    regime = _FREQ_REGIME_LONG if is_long_period else _FREQ_REGIME_SHORT
    return dict(heatmap_freq=binning, x_date_freq=binning, freq_regime=regime)


def make_factsheet_config(reporting_frequency: ReportingFrequency = ReportingFrequency.DAILY,
                          is_long_period: bool = True,
                          **overrides: Any
                          ) -> FactsheetConfig:
    """
    build a FactsheetConfig from the (data frequency, reported span) axes.

    reporting_frequency : sampling frequency of the input data (sets all freq_*/window/span fields).
    is_long_period      : True for multi-year reports (3y risk windows, annual heatmap);
                          False for short reports (all rolling windows 1y, monthly heatmap).
    overrides           : any explicit FactsheetConfig field override, applied last.

    freq_regime follows the long/short axis ('QE' long, 'ME' short). Static fields
    (perf_columns, perf_stats_labels, short, digits_to_show, sharpe_digits,
    is_unit_based_traded_volume, date_format) fall back to the FactsheetConfig defaults.
    """
    if reporting_frequency not in _PERIODS_PER_YEAR:
        raise ValueError(f"unsupported reporting_frequency={reporting_frequency}, "
                         f"expected one of {list(_PERIODS_PER_YEAR.keys())}")
    fields: Dict[str, Any] = _frequency_stat_spec(reporting_frequency, is_long_period)
    fields.update(_display_overrides(is_long_period))
    fields.update(overrides)  # caller-supplied overrides win
    return FactsheetConfig(**fields)


# ----------------------------------------------------------------------------------------------------
# module-level presets (derived from the single source of truth above)
# ----------------------------------------------------------------------------------------------------
FACTSHEET_CONFIG_DAILY_DATA_LONG_PERIOD = make_factsheet_config(ReportingFrequency.DAILY, is_long_period=True)
FACTSHEET_CONFIG_DAILY_DATA_SHORT_PERIOD = make_factsheet_config(ReportingFrequency.DAILY, is_long_period=False)

FACTSHEET_CONFIG_WEEKLY_DATA_LONG_PERIOD = make_factsheet_config(ReportingFrequency.WEEKLY, is_long_period=True)
FACTSHEET_CONFIG_WEEKLY_DATA_SHORT_PERIOD = make_factsheet_config(ReportingFrequency.WEEKLY, is_long_period=False)
# back-compat alias: the old name lacked the _DATA_ infix.
FACTSHEET_CONFIG_WEEKLY_SHORT_PERIOD = FACTSHEET_CONFIG_WEEKLY_DATA_SHORT_PERIOD

FACTSHEET_CONFIG_MONTHLY_DATA_LONG_PERIOD = make_factsheet_config(ReportingFrequency.MONTHLY, is_long_period=True)
# genuine short-period preset (monthly heatmap + 'ME' regime), per the long/short regime rule.
FACTSHEET_CONFIG_MONTHLY_DATA_SHORT_PERIOD = make_factsheet_config(ReportingFrequency.MONTHLY, is_long_period=False)

FACTSHEET_CONFIG_QUARTERLY_DATA_LONG_PERIOD = make_factsheet_config(ReportingFrequency.QUARTERLY, is_long_period=True)
FACTSHEET_CONFIG_QUARTERLY_DATA_SHORT_PERIOD = make_factsheet_config(ReportingFrequency.QUARTERLY, is_long_period=False)


# lookup tables used by fetch_default_report_kwargs (replaces the repeated if/elif ladders).
# long span -> long presets, short span -> short presets (uniform across frequencies).
_DEFAULT_CONFIG_LONG: Dict[ReportingFrequency, FactsheetConfig] = {
    ReportingFrequency.DAILY: FACTSHEET_CONFIG_DAILY_DATA_LONG_PERIOD,
    ReportingFrequency.WEEKLY: FACTSHEET_CONFIG_WEEKLY_DATA_LONG_PERIOD,
    ReportingFrequency.MONTHLY: FACTSHEET_CONFIG_MONTHLY_DATA_LONG_PERIOD,
    ReportingFrequency.QUARTERLY: FACTSHEET_CONFIG_QUARTERLY_DATA_LONG_PERIOD,
}
_DEFAULT_CONFIG_SHORT: Dict[ReportingFrequency, FactsheetConfig] = {
    ReportingFrequency.DAILY: FACTSHEET_CONFIG_DAILY_DATA_SHORT_PERIOD,
    ReportingFrequency.WEEKLY: FACTSHEET_CONFIG_WEEKLY_DATA_SHORT_PERIOD,
    ReportingFrequency.MONTHLY: FACTSHEET_CONFIG_MONTHLY_DATA_SHORT_PERIOD,
    ReportingFrequency.QUARTERLY: FACTSHEET_CONFIG_QUARTERLY_DATA_SHORT_PERIOD,
}


def fetch_factsheet_config_kwargs(factsheet_config: FactsheetConfig = FACTSHEET_CONFIG_DAILY_DATA_LONG_PERIOD,
                                  rates_data: pd.Series = None,
                                  add_rates_data: bool = True,
                                  override: Dict[str, Any] = None
                                  ) -> Dict[str, Any]:
    """
    Expand a FactsheetConfig into the keyword arguments consumed by the generate_*_factsheet(...)
    functions.

    The FactsheetConfig fields are spread verbatim (via ._asdict()) and augmented with the two
    objects the generators expect to be built once: a PerfParams (constructed from the config's
    frequency fields) and a BenchmarkReturnsQuantilesRegime (from freq_regime). The 'freq' field is
    popped so it does not override the per-method default frequency downstream.

    Risk-free rate handling:
      - rates_data (a periodic rate series as a decimal, e.g. the 3M US T-bill) is passed straight
        into PerfParams and drives the excess-return / excess-Sharpe statistics.
      - if rates_data is None and add_rates_data is True, the 3M US T-bill ('^IRX') is downloaded
        via yfinance and divided by 100; if that download is empty (e.g. offline) rates_data falls
        back to None and the report uses zero-rate statistics.
      - add_rates_data also selects the displayed performance columns: the excess-return set and
        labels when True, the rf=0 set when False.

    Args:
        factsheet_config: the (frequency, span) preset whose fields become the report kwargs
        rates_data: external risk-free rate series as a decimal. Supplied, it is used as-is and
            no download occurs
        add_rates_data: when ``rates_data`` is None, download the default 3M US rate and show
            the excess-return columns. Needs the ``[data]`` extra
        override: final overrides merged into the result, which win over the preset

    Returns:
        keyword arguments ready to spread into ``generate_*_factsheet(...)``, including the
        constructed ``perf_params`` and ``regime_classifier``
    """
    if rates_data is None:
        if add_rates_data:
            try:
                import yfinance as yf
            except ImportError as exc:
                raise ImportError(
                    "add_rates_data=True downloads the 3M US T-bill ('^IRX') via the optional "
                    "dependency yfinance: run `pip install qis[data]`, or pass rates_data directly, "
                    "or set add_rates_data=False for zero-rate statistics"
                ) from exc
            rates_data = yf.download('^IRX', start="1959-12-31", end=None, auto_adjust=True)['Close'].dropna() / 100.0
            if rates_data.empty:  # if online
                rates_data = None

    perf_params = PerfParams(freq=factsheet_config.freq,
                             freq_vol=factsheet_config.vol_freq,
                             freq_skewness=factsheet_config.vol_freq,
                             freq_drawdown=factsheet_config.freq_drawdown,
                             freq_reg=factsheet_config.freq_reg,
                             rates_data=rates_data)
    regime_classifier = BenchmarkReturnsQuantilesRegime(freq=factsheet_config.freq_regime)
    kwargs = factsheet_config._asdict()

    if add_rates_data:
        kwargs['perf_columns'] = PERF_COLUMNS
        kwargs['perf_stats_labels'] = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0, PerfStat.SHARPE_EXCESS,)
    else:
        kwargs['perf_columns'] = PERF_COLUMNS_RF0
        kwargs['perf_stats_labels'] = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0, )

    kwargs.pop('freq')  # remove frequency as some methods have default freq
    kwargs.update(dict(perf_params=perf_params, regime_classifier=regime_classifier))
    if override is not None:
        kwargs = update_kwargs(kwargs, override)
    return kwargs


def fetch_default_perf_params(rates_data: pd.Series = None,
                              add_rates_data: bool = True
                              ) -> Tuple[PerfParams, BenchmarkReturnsQuantilesRegime]:
    """
    Build the default (PerfParams, BenchmarkReturnsQuantilesRegime) pair used outside the
    factsheet-config path.

    The 3M US T-bill ('^IRX') is downloaded via yfinance and divided by 100 to give a decimal rate
    series; if the download is empty (e.g. offline) the rate falls back to None (zero-rate stats).
    PerfParams use weekly ('W-WED') return and regression frequencies; the regime classifier uses a
    quarterly ('QE') grid.

    Args:
        rates_data: external risk-free rate series as a decimal. Supplied, it is used as-is and
            no download occurs
        add_rates_data: when ``rates_data`` is None, download the default 3M US rate. Needs the
            ``[data]`` extra; an empty download falls back to zero-rate statistics

    Returns:
        the performance parameters and the regime classifier, in that order
    """
    if rates_data is None:
        if add_rates_data:
            try:
                import yfinance as yf
            except ImportError as exc:
                raise ImportError(
                    "add_rates_data=True downloads the 3M US T-bill ('^IRX') via the optional "
                    "dependency yfinance: run `pip install qis[data]`, or pass rates_data directly, "
                    "or set add_rates_data=False for zero-rate statistics"
                ) from exc
            rates_data = yf.download('^IRX', start="1959-12-31", end=None, auto_adjust=True)['Close'].dropna() / 100.0
            if rates_data.empty:  # if online
                rates_data = None
    perf_params = PerfParams(freq='W-WED', freq_reg='W-WED', rates_data=rates_data)
    regime_classifier = BenchmarkReturnsQuantilesRegime(freq='QE')

    return perf_params, regime_classifier


def fetch_default_report_kwargs(time_period: Optional[TimePeriod] = None,
                                reporting_frequency: ReportingFrequency = ReportingFrequency.MONTHLY,
                                long_threshold_years: float = 5.0,
                                add_rates_data: bool = False,
                                is_unit_based_traded_volume: bool = True,
                                override: Dict[str, Any] = None
                                ) -> Dict[str, Any]:
    """
    the report keyword arguments for a given reporting frequency and reported span.

    The reported-span axis is selected automatically: the long-period presets apply when no
    ``time_period`` is given or its length exceeds ``long_threshold_years``, otherwise the
    short-period presets. ``reporting_frequency`` then selects the matching preset, which
    ``fetch_factsheet_config_kwargs`` expands. An unknown frequency falls back to MONTHLY.

    Spread the result into a factsheet generator: it carries the window lengths, grid spacing and
    regime presets that make a report's panels agree with one another and with the data frequency.

    Args:
        time_period: reporting window; its length drives the long / short preset choice. None
            selects the long-period presets
        reporting_frequency: data sampling frequency and the report's base grid
        long_threshold_years: span in years above which the long-period presets are used
        add_rates_data: download the default 3M US rate and show excess-return columns. Needs the
            ``[data]`` extra, since the download goes through yfinance
        is_unit_based_traded_volume: report turnover and cost on normalised units rather than
            notional
        override: final overrides merged into the result, which win over every preset

    Returns:
        keyword arguments ready to spread into ``generate_*_factsheet(...)``
    """
    # long span when no time_period is given or its length exceeds the threshold (number of years > 5)
    is_long_period = time_period is None or time_period.get_time_period_an() > long_threshold_years
    config_by_frequency = _DEFAULT_CONFIG_LONG if is_long_period else _DEFAULT_CONFIG_SHORT
    # unknown frequencies fall back to the monthly preset (matches the previous `else` branch)
    factsheet_config = config_by_frequency.get(reporting_frequency,
                                               config_by_frequency[ReportingFrequency.MONTHLY])

    kwargs = fetch_factsheet_config_kwargs(factsheet_config=factsheet_config,
                                           add_rates_data=add_rates_data)
    kwargs = update_kwargs(kwargs, dict(is_unit_based_traded_volume=is_unit_based_traded_volume))
    if override is not None:
        kwargs = update_kwargs(kwargs, override)
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