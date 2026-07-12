"""
Reusable performance report layout used by several examples.

Extracted from the former ``examples/core/price_plots.py``. The function
``generate_performance_report`` produces a 5-figure performance summary on a
price DataFrame:

  1. Risk-adjusted performance table vs benchmark
  2. Periodic (e.g. yearly) returns table
  3. Cumulative price plot with regime shadows
  4. Cumulative price + drawdown plot
  5. Regime-conditional scatter regression vs the benchmark

It is used by ``examples/perfstats/full_performance_report.py`` and
``examples/factsheets/pybloqs_factsheets.py``.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

import qis as qis
from qis import PerfStat
from qis.plots.utils import calc_table_height


# Default column set for risk-adjusted performance vs benchmark.
DEFAULT_RA_TABLE_COLUMNS = (PerfStat.START_DATE,
                            PerfStat.END_DATE,
                            PerfStat.TOTAL_RETURN,
                            PerfStat.PA_RETURN,
                            PerfStat.VOL,
                            PerfStat.SHARPE_RF0,
                            PerfStat.SHARPE_EXCESS,
                            PerfStat.MAX_DD,
                            PerfStat.MAX_DD_VOL,
                            PerfStat.SKEWNESS,
                            PerfStat.KURTOSIS,
                            PerfStat.ALPHA,
                            PerfStat.BETA,
                            PerfStat.R2)


def generate_performance_report(prices: pd.DataFrame,
                                regime_benchmark: str,
                                perf_params: qis.PerfParams = None,
                                perf_columns: List[PerfStat] = DEFAULT_RA_TABLE_COLUMNS,
                                heatmap_freq: str = 'YE',
                                **kwargs
                                ) -> None:
    """Generate a 5-figure performance summary on a price panel.

    Args:
        prices: Price level DataFrame, columns = assets.
        regime_benchmark: Column name in ``prices`` to use as the regime benchmark.
        perf_params: Optional PerfParams; defaults to qis defaults.
        perf_columns: Columns to show in the risk-adjusted performance table.
        heatmap_freq: Frequency for the periodic returns heatmap (e.g. 'YE', 'ME').
        **kwargs: Forwarded to qis plotting routines.
    """
    local_kwargs = dict(digits_to_show=1,
                        framealpha=0.75,
                        perf_stats_labels=qis.PerfStatsLabels.DETAILED_WITH_DD.value)
    kwargs = qis.update_kwargs(kwargs, local_kwargs)

    # 1. risk-adjusted performance table vs benchmark
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), tight_layout=True)
    qis.plot_ra_perf_table_benchmark(
        prices=prices,
        benchmark=regime_benchmark,
        perf_params=perf_params,
        perf_columns=perf_columns,
        title=f"Risk-adjusted performance: {qis.get_time_period_label(prices, date_separator='-')}",
        ax=ax,
        **kwargs)

    # 2. periodic returns table
    fig, ax = plt.subplots(
        1, 1,
        figsize=(7, calc_table_height(num_rows=len(prices.columns) + 5, scale=0.5)),
        tight_layout=True)
    qis.plot_periodic_returns_table(
        prices=prices,
        freq=heatmap_freq,
        ax=ax,
        title=f"Periodic performance: {qis.get_time_period_label(prices, date_separator='-')}",
        total_name='Total',
        **qis.update_kwargs(kwargs, dict(square=False, x_rotation=90)))

    with sns.axes_style("darkgrid"):
        # 3. cumulative price with regime shadows
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        qis.plot_prices(
            prices=prices,
            regime_benchmark=regime_benchmark,
            perf_params=perf_params,
            title=f"Time series of $1 invested: {qis.get_time_period_label(prices, date_separator='-')}",
            ax=ax,
            **kwargs)

        # 4. price + drawdown
        fig, axs = plt.subplots(2, 1, figsize=(7, 7))
        qis.plot_prices_with_dd(
            prices=prices,
            regime_benchmark=regime_benchmark,
            perf_params=perf_params,
            title=f"Time series of $1 invested: {qis.get_time_period_label(prices, date_separator='-')}",
            axs=axs,
            **kwargs)

        # 5. regime-conditional scatter regression
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        regime_classifier = qis.BenchmarkReturnsQuantilesRegime(freq=perf_params.freq_reg)
        qis.plot_scatter_regression(
            prices=prices,
            regime_benchmark=regime_benchmark,
            regime_classifier=regime_classifier,
            perf_params=perf_params,
            title=f"Regime Conditional Regression: {qis.get_time_period_label(prices, date_separator='-')}",
            ax=ax,
            **kwargs)