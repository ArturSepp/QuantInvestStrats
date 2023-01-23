

import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum

from qis.data.ust_rates import load_ust_rates
import qis


def plot_ust_rates_data():
    df = load_ust_rates()

    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
        qis.plot_time_series(df=df,
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                             var_format='{:,.2f}',
                             ax=ax)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
        qis.plot_time_series(df=df[['3m', '10y']],
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                             var_format='{:,.2f}',
                             ax=ax)


class UnitTests(Enum):
    PLOT_RATES = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.PLOT_RATES:
        plot_ust_rates_data()

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_RATES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
