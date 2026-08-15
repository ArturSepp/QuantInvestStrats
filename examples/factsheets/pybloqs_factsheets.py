"""
Optional: pybloqs-rendered factsheets.

``pybloqs`` is an optional dependency — the qis top-level ``__init__`` does
not import it. To run this example, install pybloqs and apply the small
jinja patch documented below.

This file demonstrates three pybloqs report types:

  ``Mode.RA_PERFORMANCE``       — five-figure performance report wrapped in
                                  a pybloqs ``VStack``.
  ``Mode.MULTI_PORTFOLIO``      — multi-strategy factsheet (volparity sweep
                                  across spans) via
                                  ``generate_multi_portfolio_factsheet_with_pybloqs``.
  ``Mode.STRATEGY_BENCHMARK``   — single strategy vs benchmark via
                                  ``generate_strategy_benchmark_factsheet_with_pybloqs``.

────────────────────────────────────────────────────────────────────────────
pybloqs jinja patch (required for pandas >= 2.x)
────────────────────────────────────────────────────────────────────────────
Locate the file ``...\\Lib\\site-packages\\pybloqs\\jinja\\table.html``
and change line 44 from::

    {% for col_name, cell in row.iteritems() %}

to::

    {% for col_name, cell in row.items() %}
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
from enum import Enum

import yfinance as yf
import qis
from qis import (PerfStat, TimePeriod, MultiPortfolioData, local_path)
from qis.portfolio.reports.config import fetch_default_report_kwargs
from qis.plots.utils import calc_table_height

# pybloqs is optional — only required to run this file
try:
    import pybloqs as p
    from qis.portfolio.reports.multi_strategy_factsheet_pybloqs import (
        generate_multi_portfolio_factsheet_with_pybloqs,
    )
    from qis.portfolio.reports.strategy_benchmark_factsheet_pybloqs import (
        generate_strategy_benchmark_factsheet_with_pybloqs,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pybloqs is required to run this example. Install it with "
        "`pip install pybloqs` and apply the jinja patch documented in "
        "this file's docstring."
    ) from exc


# Performance columns shown in the pybloqs RA-perf table.
RA_TABLE_COLUMNS = (PerfStat.TOTAL_RETURN,
                    PerfStat.PA_RETURN,
                    PerfStat.VOL,
                    PerfStat.SHARPE_RF0,
                    PerfStat.SHARPE_EXCESS,
                    PerfStat.MAX_DD,
                    PerfStat.MAX_DD_VOL,
                    PerfStat.SKEWNESS,
                    PerfStat.KURTOSIS)


# ──────────────────────────────────────────────────────────────────────
# Mode 1: pybloqs-wrapped 5-figure performance report
# ──────────────────────────────────────────────────────────────────────

def generate_ra_performance_pybloqs(prices: pd.DataFrame,
                                    regime_benchmark: str,
                                    perf_params: qis.PerfParams = None,
                                    heatmap_freq: str = 'YE',
                                    **kwargs
                                    ) -> None:
    """Build the same 5-figure layout as
    ``examples/perfstats/full_performance_report.py``, then assemble it
    into a pybloqs ``VStack`` and save as PDF.
    """
    local_kwargs = dict(digits_to_show=1, framealpha=0.75,
                        perf_stats_labels=qis.PerfStatsLabels.DETAILED_WITH_DD.value)
    kwargs = qis.update_kwargs(kwargs, local_kwargs)

    fig1, ax = plt.subplots(1, 1, figsize=(14, 1.1), tight_layout=True)
    qis.plot_ra_perf_table(prices=prices, perf_columns=RA_TABLE_COLUMNS,
                           perf_params=perf_params, fontsize=8, ax=ax, **kwargs)

    ra_table = qis.get_ra_perf_columns(prices=prices, perf_columns=RA_TABLE_COLUMNS,
                                       perf_params=perf_params, drop_index=True, **kwargs)

    fig2, ax = plt.subplots(
        1, 1,
        figsize=(7, calc_table_height(num_rows=len(prices.columns) + 5, scale=0.5)),
        tight_layout=True)
    qis.plot_periodic_returns_table(
        prices=prices, freq=heatmap_freq, ax=ax,
        title=f"Periodic performance: {qis.get_time_period_label(prices, date_separator='-')}",
        total_name='Total',
        **qis.update_kwargs(kwargs, dict(square=False, x_rotation=90)))

    with sns.axes_style("darkgrid"):
        fig3, ax = plt.subplots(1, 1, figsize=(8, 6))
        qis.plot_prices(
            prices=prices, regime_benchmark=regime_benchmark,
            perf_params=perf_params,
            title=f"Time series of $1 invested: {qis.get_time_period_label(prices, date_separator='-')}",
            ax=ax, **kwargs)

        fig4, axs = plt.subplots(2, 1, figsize=(7, 7))
        qis.plot_prices_with_dd(
            prices=prices, regime_benchmark=regime_benchmark,
            perf_params=perf_params,
            title=f"Time series of $1 invested: {qis.get_time_period_label(prices, date_separator='-')}",
            axs=axs, **kwargs)

        fig5, ax = plt.subplots(1, 1, figsize=(8, 6))
        regime_classifier = qis.BenchmarkReturnsQuantilesRegime(freq=perf_params.freq_reg)
        qis.plot_scatter_regression(
            prices=prices, regime_benchmark=regime_benchmark,
            regime_classifier=regime_classifier, perf_params=perf_params,
            title=f"Regime Conditional Regression: {qis.get_time_period_label(prices, date_separator='-')}",
            ax=ax, **kwargs)

    bkwargs_title = {'title_wrap': True, 'text-align': 'left', 'color': 'blue',
                     'font_size': "12px", 'title_level': 3, 'line-height': 0.1,
                     'inherit_cfg': False}
    bkwargs_text = {'title_wrap': True, 'text-align': 'left', 'font_size': "10px",
                    'title_level': 3, 'line-height': 0.1}
    bkwargs_fig = bkwargs_text

    b_fig_ra = p.Block(
        [p.Paragraph("Block with Risk-adjusted performance", **bkwargs_title),
         p.Paragraph("Description", **bkwargs_text),
         p.Block(fig1, **bkwargs_fig)],
        **bkwargs_text)

    r = p.VStack([
        p.Block("Strategy description", title="Backtest strategy"),
        b_fig_ra,
        p.Block(fig2, title='Periodic returns'),
        p.Block(ra_table),
        p.Block(fig3),
        p.Block(fig4),
        p.Block(fig5),
    ])

    out = f"{local_path.get_output_path()}PyBloqs_Factsheet_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
    r.save(out)
    print(f"saved to {out}")


# ──────────────────────────────────────────────────────────────────────
# Modes 2 & 3: multi-portfolio and strategy-vs-benchmark pybloqs reports
# ──────────────────────────────────────────────────────────────────────

def fetch_riskparity_universe_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Define a simple multi-asset universe with asset-class grouping."""
    universe_data = dict(SPY='Equities', QQQ='Equities', EEM='Equities',
                         TLT='Bonds', IEF='Bonds',
                         LQD='Credit', HYG='HighYield',
                         GLD='Gold')
    tickers = list(universe_data.keys())
    group_data = pd.Series(universe_data)
    prices = yf.download(tickers=tickers, start="2003-12-31", end=None,
                         ignore_tz=True, auto_adjust=True)['Close'][tickers]
    prices = prices.asfreq('B', method='ffill').loc['2003':]
    benchmark_prices = prices[['SPY', 'TLT']]
    return prices, benchmark_prices, group_data


def generate_volparity_multi_strategy(prices: pd.DataFrame,
                                      benchmark_prices: pd.DataFrame,
                                      group_data: pd.Series,
                                      time_period: TimePeriod,
                                      spans: List[int] = (5, 10, 20, 40, 60, 120),
                                      vol_target: float = 0.15,
                                      rebalancing_costs: float = 0.0010,
                                      ) -> MultiPortfolioData:
    """Generate a span sweep of volparity portfolios as a MultiPortfolioData."""
    returns = qis.to_returns(prices=prices, is_log_returns=True)

    portfolio_datas = []
    for span in spans:
        ra_returns, weights, ewm_vol = qis.compute_ra_returns(
            returns=returns, span=span, vol_target=vol_target)
        weights = weights.divide(weights.sum(axis=1), axis=0)
        portfolio_data = qis.backtest_model_portfolio(
            prices=prices,
            weights=time_period.locate(weights),
            rebalancing_costs=rebalancing_costs,
            ticker=f"VP span={span}")
        portfolio_data.set_group_data(group_data=group_data,
                                      group_order=list(group_data.unique()))
        portfolio_datas.append(portfolio_data)

    return MultiPortfolioData(portfolio_datas, benchmark_prices=benchmark_prices)


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

class Mode(Enum):
    RA_PERFORMANCE = 1
    MULTI_PORTFOLIO = 2
    STRATEGY_BENCHMARK = 3


def run(mode: Mode = Mode.RA_PERFORMANCE) -> None:
    if mode == Mode.RA_PERFORMANCE:
        ust_3m_rate = yf.download('^IRX', start="2003-12-31", end=None,
                                  ignore_tz=True, auto_adjust=True)['Close'].dropna() / 100.0
        regime_benchmark = 'SPY'
        tickers = [regime_benchmark, 'QQQ', 'EEM', 'TLT', 'IEF',
                   'LQD', 'HYG', 'SHY', 'GLD']
        time_period = qis.TimePeriod('31Dec2021', '23Jan2023')
        perf_params = qis.PerfParams(freq='W-WED', freq_reg='W-WED',
                                     freq_drawdown='B', rates_data=ust_3m_rate)
        kwargs = dict(x_date_freq='ME', heatmap_freq='ME',
                      date_format='%b-%y', perf_params=perf_params)
        prices = yf.download(tickers, start="2003-12-31", end=None,
                             ignore_tz=True, auto_adjust=True)['Close'].dropna()
        prices = time_period.locate(prices)
        generate_ra_performance_pybloqs(prices=prices,
                                        regime_benchmark=regime_benchmark,
                                        **kwargs)
        plt.show()
        return

    # MULTI_PORTFOLIO and STRATEGY_BENCHMARK both need a MultiPortfolioData
    time_period = qis.TimePeriod('31Dec2005', '21Apr2025')
    prices, benchmark_prices, group_data = fetch_riskparity_universe_data()
    multi = generate_volparity_multi_strategy(
        prices=prices,
        benchmark_prices=benchmark_prices,
        group_data=group_data,
        time_period=time_period,
        vol_target=0.15,
        rebalancing_costs=0.0010,
    )

    if mode == Mode.MULTI_PORTFOLIO:
        report = generate_multi_portfolio_factsheet_with_pybloqs(
            multi_portfolio_data=multi,
            time_period=time_period,
            **fetch_default_report_kwargs(time_period=time_period))
        out = f"{qis.local_path.get_output_path()}_volparity_span_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
        report.save(out)
        print(f"saved multi-portfolio report to {out}")

    elif mode == Mode.STRATEGY_BENCHMARK:
        report = generate_strategy_benchmark_factsheet_with_pybloqs(
            multi_portfolio_data=multi,
            strategy_idx=-1,
            benchmark_idx=0,
            time_period=time_period,
            **fetch_default_report_kwargs(time_period=time_period))
        out = f"{qis.local_path.get_output_path()}_volparity_pybloq_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
        report.save(out)
        print(f"saved strategy-benchmark report to {out}")


if __name__ == '__main__':
    run(mode=Mode.RA_PERFORMANCE)