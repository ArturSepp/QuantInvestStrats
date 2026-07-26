"""
qis.factsheet - a one-call facade over the four factsheet generators.

This is *additive* convenience sugar: it wraps generate_strategy_factsheet,
generate_strategy_benchmark_factsheet_plt, generate_multi_portfolio_factsheet and
generate_multi_asset_factsheet without changing any of them. All of those remain available and
unchanged for full control; this entry point only removes boilerplate for the common case.

It picks the report archetype from the input type, calibrates every window / regression / regime /
annualisation for the requested reporting frequency via fetch_default_report_kwargs (long vs short
horizon is auto-selected from the reporting span), renders, and optionally writes a PDF.

    import qis
    # one strategy vs benchmark, monthly reporting, full history -> PDF
    qis.factsheet(prices, benchmark_prices=spy, reporting_frequency='monthly', file_name='book')

All qis imports are deferred into the function bodies so importing this module never depends on
qis being fully initialised (keeps the top-level `qis.factsheet` export circular-import safe).
"""
import matplotlib.pyplot as plt
import pandas as pd
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:  # resolved by typing.get_type_hints and by sphinx autodoc, never at runtime
    from qis.portfolio.reports.config import ReportingFrequency
    from qis.portfolio.multi_portfolio_data import MultiPortfolioData
    from qis.portfolio.portfolio_data import PortfolioData
    from qis.utils.dates import TimePeriod

# report-archetype identifiers (also accepted via the `kind=` override)
KIND_STRATEGY = 'strategy'
KIND_STRATEGY_BENCHMARK = 'strategy_benchmark'
KIND_MULTI_STRATEGY = 'multi_strategy'
KIND_MULTI_ASSET = 'multi_asset'

# keys the facade passes explicitly to the generators - popped from the spread kwargs so that a
# caller-supplied value can never collide into a duplicate keyword argument
_RESERVED_KWARGS = ('portfolio_data', 'multi_portfolio_data', 'prices', 'benchmark',
                    'benchmark_prices', 'time_period', 'factsheet_name', 'backtest_name')


def _parse_reporting_frequency(reporting_frequency):
    """accept a ReportingFrequency or a friendly string ('daily'/'weekly'/'monthly'/'quarterly')."""
    from qis.portfolio.reports.config import ReportingFrequency
    if isinstance(reporting_frequency, ReportingFrequency):
        return reporting_frequency
    aliases = {'daily': ReportingFrequency.DAILY, 'b': ReportingFrequency.DAILY, 'd': ReportingFrequency.DAILY,
               'weekly': ReportingFrequency.WEEKLY, 'w': ReportingFrequency.WEEKLY, 'w-wed': ReportingFrequency.WEEKLY,
               'monthly': ReportingFrequency.MONTHLY, 'm': ReportingFrequency.MONTHLY, 'me': ReportingFrequency.MONTHLY,
               'quarterly': ReportingFrequency.QUARTERLY, 'q': ReportingFrequency.QUARTERLY, 'qe': ReportingFrequency.QUARTERLY}
    key = str(reporting_frequency).strip().lower()
    if key in aliases:
        return aliases[key]
    raise ValueError(f"unsupported reporting_frequency={reporting_frequency!r}; use one of "
                     f"daily/weekly/monthly/quarterly or a ReportingFrequency member")


def _as_fig_list(result) -> List[plt.Figure]:
    """normalise a generator return (single Figure or list) to a list of figures."""
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        return [f for f in result if f is not None]
    return [result]


def _infer_time_period(data, prices):
    """full-history reporting span when the caller does not pass time_period."""
    import qis as qis
    from qis import PortfolioData, MultiPortfolioData
    if prices is not None:
        return qis.get_time_period(df=prices)
    if isinstance(data, PortfolioData):
        return qis.get_time_period(df=data.get_portfolio_nav())
    if isinstance(data, MultiPortfolioData):
        return qis.get_time_period(df=data.get_navs())
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return qis.get_time_period(df=data)
    raise TypeError(f"cannot infer time_period from {type(data)!r}; pass time_period explicitly")


def factsheet(data: Union[pd.Series, pd.DataFrame, "PortfolioData", "MultiPortfolioData"],
              benchmark_prices: Optional[Union[pd.Series, pd.DataFrame]] = None,
              benchmark: Optional[str] = None,
              reporting_frequency: Union[str, "ReportingFrequency"] = 'monthly',
              time_period: Optional["TimePeriod"] = None,
              kind: Optional[str] = None,
              data_is_returns: bool = False,
              long_threshold_years: float = 5.0,
              add_rates_data: bool = False,
              file_name: Optional[str] = None,
              local_path: Optional[str] = None,
              factsheet_name: Optional[str] = None,
              **kwargs
              ) -> Union[str, List[plt.Figure]]:
    """
    Render the appropriate qis factsheet across a chosen reporting frequency in a single call.

    Archetype is auto-detected from `data` (override with `kind=`):
      - pd.Series / pd.DataFrame of prices (or returns, if data_is_returns=True) -> multi-asset
      - PortfolioData                                                            -> strategy
      - MultiPortfolioData                                                       -> multi-strategy
        (pass kind='strategy_benchmark' to render the strategy-vs-benchmark report instead)

    The reporting frequency ('daily'/'weekly'/'monthly'/'quarterly' or a ReportingFrequency)
    calibrates every rolling window, regression frequency, regime-classification frequency and
    annualisation consistently, via fetch_default_report_kwargs; long vs short horizon is selected
    automatically from the reporting span against long_threshold_years.

    This is additive: the underlying ``generate_*_factsheet`` functions are unchanged and remain
    the full-control API.

    Args:
        data: prices or returns, PortfolioData, or MultiPortfolioData; drives archetype
            selection
        benchmark_prices: reference prices; required for the strategy (PortfolioData) report
        benchmark: reference column name on the multi-asset path; defaults to the first column
            of ``data``
        reporting_frequency: 'daily', 'weekly', 'monthly', 'quarterly', or a ReportingFrequency
        time_period: reporting span; defaults to the full history of ``data``
        kind: force an archetype identifier; None auto-detects from the type of ``data``
        data_is_returns: treat ``data`` and ``benchmark_prices`` as returns and compound them
            to navs
        long_threshold_years: spans of at least this length use the long-horizon preset
        add_rates_data: download the risk-free rate for the excess-return statistics; needs the
            [data] extra
        file_name: write the report to a PDF of this name and return its path; without it the
            figures are returned
        local_path: directory for the PDF; ignored when ``file_name`` is None
        factsheet_name: report title, mapped to ``backtest_name`` for the multi-portfolio
            reports
        **kwargs: forwarded to the underlying generator; caller values override the preset

    Returns:
        the PDF path when ``file_name`` is given, otherwise the list of figures
    """
    import qis as qis
    from qis import PortfolioData, MultiPortfolioData
    from qis.portfolio.reports.config import fetch_default_report_kwargs
    from qis.portfolio.reports.strategy_factsheet import generate_strategy_factsheet
    from qis.portfolio.reports.strategy_benchmark_factsheet import generate_strategy_benchmark_factsheet_plt
    from qis.portfolio.reports.multi_strategy_factsheet import generate_multi_portfolio_factsheet
    from qis.portfolio.reports.multi_assets_factsheet import generate_multi_asset_factsheet

    rf = _parse_reporting_frequency(reporting_frequency)

    # resolve the archetype from the input type unless explicitly forced
    if kind is None:
        if isinstance(data, MultiPortfolioData):
            kind = KIND_MULTI_STRATEGY
        elif isinstance(data, PortfolioData):
            kind = KIND_STRATEGY
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            kind = KIND_MULTI_ASSET
        else:
            raise TypeError(f"unsupported data type {type(data)!r}; pass prices/returns, "
                            f"PortfolioData or MultiPortfolioData (or set kind=)")

    # normalise raw price/return input for the multi-asset path
    prices = None
    if kind == KIND_MULTI_ASSET:
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError(f"kind='{KIND_MULTI_ASSET}' expects prices/returns, got {type(data)!r}")
        prices = qis.returns_to_nav(data) if data_is_returns else data
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        if data_is_returns and benchmark_prices is not None:
            benchmark_prices = qis.returns_to_nav(benchmark_prices)
        # default the regime/beta reference to the first column when none is supplied
        if benchmark is None and benchmark_prices is None and prices.shape[1] >= 1:
            benchmark = str(prices.columns[0])

    if time_period is None:
        time_period = _infer_time_period(data, prices)

    # frequency-calibrated preset kwargs; caller-supplied kwargs win, then drop any keys the facade
    # passes explicitly so they cannot collide into duplicate keyword arguments
    report_kwargs = fetch_default_report_kwargs(time_period=time_period,
                                                reporting_frequency=rf,
                                                long_threshold_years=long_threshold_years,
                                                add_rates_data=add_rates_data)
    report_kwargs = qis.update_kwargs(report_kwargs, kwargs)
    report_kwargs = {k: v for k, v in report_kwargs.items() if k not in _RESERVED_KWARGS}

    if kind == KIND_STRATEGY:
        if benchmark_prices is None:
            raise ValueError("kind='strategy' (PortfolioData) requires benchmark_prices")
        result = generate_strategy_factsheet(portfolio_data=data,
                                             benchmark_prices=benchmark_prices,
                                             time_period=time_period,
                                             factsheet_name=factsheet_name,
                                             **report_kwargs)
    elif kind == KIND_STRATEGY_BENCHMARK:
        if not isinstance(data, MultiPortfolioData):
            raise TypeError("kind='strategy_benchmark' requires a MultiPortfolioData (strategy + benchmark)")
        result = generate_strategy_benchmark_factsheet_plt(multi_portfolio_data=data,
                                                           time_period=time_period,
                                                           backtest_name=factsheet_name,
                                                           **report_kwargs)
    elif kind == KIND_MULTI_STRATEGY:
        if not isinstance(data, MultiPortfolioData):
            raise TypeError("kind='multi_strategy' requires a MultiPortfolioData")
        result = generate_multi_portfolio_factsheet(multi_portfolio_data=data,
                                                    time_period=time_period,
                                                    backtest_name=factsheet_name,
                                                    **report_kwargs)
    elif kind == KIND_MULTI_ASSET:
        result = generate_multi_asset_factsheet(prices=prices,
                                                benchmark_prices=benchmark_prices,
                                                benchmark=benchmark,
                                                time_period=time_period,
                                                factsheet_name=factsheet_name,
                                                **report_kwargs)
    else:
        raise ValueError(f"unsupported kind={kind!r}; use one of "
                         f"{(KIND_STRATEGY, KIND_STRATEGY_BENCHMARK, KIND_MULTI_STRATEGY, KIND_MULTI_ASSET)}")

    figs = _as_fig_list(result)

    if file_name is not None:
        if local_path is None:
            local_path = qis.local_path.get_output_path()
        return qis.save_figs_to_pdf(figs=figs, file_name=file_name, local_path=local_path)
    return figs
