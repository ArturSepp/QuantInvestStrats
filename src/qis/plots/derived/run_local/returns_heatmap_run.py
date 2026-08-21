"""Development runner extracted from ``qis.plots.derived.returns_heatmap``."""

import matplotlib.pyplot as plt
from enum import Enum
import qis.utils.dates as da

from qis.plots.derived.returns_heatmap import (
    compute_periodic_returns_by_row_table,
    compute_periodic_returns_table,
    plot_periodic_returns_table,
    plot_returns_heatmap,
    plot_returns_table,
    plot_sorted_periodic_returns,
)

class Locals(Enum):
    PERIODIC_RETURNS_BY_ROW = 1
    RETURNS_HEATMAP = 2
    RETURNS_TABLE = 3
    PERIODIC_RETURNS_TABLE = 4
    PERIODIC_RETURNS_TABLE_A = 5
    SORTED_PERIODIC_RETURNS_TABLE = 6

def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data().dropna()

    if local == Locals.PERIODIC_RETURNS_BY_ROW:
        periodic_returns_table = compute_periodic_returns_by_row_table(prices=prices['SPY'],
                                                                       heatmap_freq='YE',
                                                                       column_period='ME')
        print(periodic_returns_table)

    elif local == Locals.RETURNS_HEATMAP:
        periodic_returns_table = compute_periodic_returns_table(prices=prices['SPY'],
                                                                column_period='ME',
                                                                is_add_annual_column=True,
                                                                is_inverse_order=True)
        print(periodic_returns_table)

        plot_returns_heatmap(prices=prices['SPY'],
                             heatmap_column_freq='ME',
                             heatmap_freq='YE',
                             #date_format='%b-%Y',
                             is_add_annual_column=True,
                             is_inverse_order=True)

    elif local == Locals.RETURNS_TABLE:
        time_period_dict = {'Q1': da.TimePeriod(start='31Dec2019', end='31Mar2020'),
                            'Q2': da.TimePeriod(start='31Mar2020', end='30Jun2020'),
                            'YTD': da.TimePeriod(start='31Dec2019', end='30Jun2020')}
        plot_returns_table(prices=prices.iloc[:, :10],
                           time_period_dict=time_period_dict,
                           vline_columns=[2],
                           hline_rows=[1],
                           transpose=False,
                           is_inverse_order=True)

    elif local == Locals.PERIODIC_RETURNS_TABLE:

        time_period = None

        plot_periodic_returns_table(prices=prices,
                                                time_period=time_period,
                                                date_format='%b-%y',
                                                freq='YE',
                                                x_rotation=90,
                                                df_out_name='heatmap1y')

    elif local == Locals.PERIODIC_RETURNS_TABLE_A:

        time_period = da.TimePeriod(start='28Feb2010', end='31Jan2021')
        plot_periodic_returns_table(prices=prices,
                                                time_period=time_period,
                                                date_format='%b-%y',
                                                freq='YE',
                                                x_rotation=90,
                                                df_out_name='heatmap1y')

    elif local == Locals.SORTED_PERIODIC_RETURNS_TABLE:
        time_period = da.TimePeriod(start='28Feb2010', end='31Jan2021')
        plot_sorted_periodic_returns(prices=prices.iloc[:, :20],
                                     time_period=time_period,
                                     date_format='%b-%y',
                                     freq='YE',
                                     x_rotation=90,
                                     add_total=False)

    plt.show()

if __name__ == "__main__":
    run_local(local=Locals.PERIODIC_RETURNS_BY_ROW)
