
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import qis as qis
from qis.plots.histogram import plot_histogram


class LocalTests(Enum):
    TEST = 1
    RETURNS = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.TEST:
        np.random.seed(1)
        n_instruments = 100
        m_samples = 2
        exposures_nm = np.random.normal(0.0, 1.0, size=(n_instruments, m_samples))
        data = pd.DataFrame(data=exposures_nm, columns=[f"id{n+1}" for n in range(m_samples)])

        fig, ax = plt.subplots(1, 1, figsize=(3.9, 3.4), tight_layout=True)
        global_kwargs = dict(fontsize=6, linewidth=0.5, weight='normal', first_color_fixed=True)

        plot_histogram(df=data,
                       add_data_std_pdf=True,
                       add_last_value=True,
                       ax=ax,
                       **global_kwargs)
        # ax.locator_params(nbins=10, axis='x')

    elif local_test == LocalTests.RETURNS:
        from qis.tests.price_data_test import load_etf_data
        prices = load_etf_data().dropna()
        returns = qis.to_returns(prices=prices[['EEM', 'SPY']], freq='QE')
        plot_histogram(df=returns,
                       xvar_format='{:.0%}',
                       add_bar_at_peak=True,
                       desc_table_type=qis.DescTableType.NONE
                       )

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.RETURNS)
