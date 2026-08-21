"""Development runner extracted from ``qis.plots.derived.perf_table``."""

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import qis.utils.dates as da
from qis.perfstats.config import PerfStat, PerfParams
import qis.perfstats.perf_stats as rpt

from qis.plots.derived.perf_table import (
    plot_best_worst_returns,
    plot_desc_freq_table,
    plot_ra_perf_annual_matrix,
    plot_ra_perf_bars,
    plot_ra_perf_by_dates,
    plot_ra_perf_scatter,
    plot_ra_perf_table,
    plot_ra_perf_table_benchmark,
    plot_top_bottom_performers,
)

class Locals(Enum):
    PLOT_RA_PERF_TABLE = 1
    PLOT_RA_PERF_SCATTER = 2
    PLOT_RA_PERF_TABLE_BENCHMARK = 3
    PLOT_DESC_FREQ_TABLE = 4
    PLOT_SHARPE_BARPLOT = 5
    PLOT_SHARPE_BY_DATES = 6
    PLOT_PERF_FOR_START_END_PERIOD = 7
    PLOT_TOP_BOTTOM_PERFORMERS = 8
    PLOT_TOP_BOTTOM_RETURNS = 9

def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data().dropna()
    print(prices)

    if local == Locals.PLOT_RA_PERF_TABLE:

        perf_params = PerfParams(freq='B')
        prices = prices.iloc[:, :5]
        plot_ra_perf_table(prices=prices,
                           perf_columns=rpt.COMPACT_TABLE_COLUMNS,
                           perf_params=perf_params)

    elif local == Locals.PLOT_RA_PERF_SCATTER:

        perf_params = PerfParams(freq='B')
        plot_ra_perf_scatter(prices=prices,
                             perf_params=perf_params)

    elif local == Locals.PLOT_RA_PERF_TABLE_BENCHMARK:
        perf_params = PerfParams(freq='ME')
        plot_ra_perf_table_benchmark(prices=prices,
                                     benchmark='SPY',
                                     perf_params=perf_params,
                                     transpose=False)

    elif local == Locals.PLOT_DESC_FREQ_TABLE:
        freq_data = plot_desc_freq_table(df=prices,
                                         freq='YE',
                                         agg_func=np.mean)
        print(freq_data)

    elif local == Locals.PLOT_SHARPE_BARPLOT:
        plot_ra_perf_bars(prices=prices, perf_column=PerfStat.MAX_DD)

    elif local == Locals.PLOT_SHARPE_BY_DATES:
        prices = prices

        time_period_dict = {'1y': da.TimePeriod(start='30Jun2019', end='30Jun2020'),
                            '3y': da.TimePeriod(start='30Jun2017', end='30Jun2020'),
                            '5y': da.TimePeriod(start='30Jun2015', end='30Jun2020')}
        plot_ra_perf_by_dates(prices=prices,
                              time_period_dict=time_period_dict)

    elif local == Locals.PLOT_PERF_FOR_START_END_PERIOD:
        plot_ra_perf_annual_matrix(price=prices.iloc[:, 0])

    elif local == Locals.PLOT_TOP_BOTTOM_PERFORMERS:
        plot_top_bottom_performers(prices=prices, num_assets=2)

    elif local == Locals.PLOT_TOP_BOTTOM_RETURNS:
        plot_best_worst_returns(price=prices.iloc[:, 0])

    plt.show()

if __name__ == "__main__":
    run_local(local=Locals.PLOT_RA_PERF_TABLE_BENCHMARK)
