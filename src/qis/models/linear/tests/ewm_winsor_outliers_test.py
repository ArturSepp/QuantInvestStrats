
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from qis.models.linear.ewm_winsor_outliers import ewm_insample_winsorising, ReplacementType, ewm_winsdor_markovian_score



class LocalTests(Enum):
    TEST1 = 1
    MARKOVIAN_WINDSOR = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    np.random.seed(2) #freeze seed

    import qis as qis

    dates = pd.date_range(start='12/31/2018', end='12/31/2019', freq='B')
    n = 2

    data = pd.DataFrame(data=np.random.standard_t(df=2, size=(len(dates), n)),
                        index=dates,
                        columns=['x'+str(m+1) for m in range(n)])
    # data = pd.Series(data=data_np, index=dates, name='data')

    if local_test == LocalTests.TEST1:
        winsor_data = ewm_insample_winsorising(data=data,
                                               ewm_lambda=0.94,
                                               nan_replacement_type=ReplacementType.EWMA_MEAN,
                                               quantile_cut=0.05)

        winsor_data.columns = [x + ' winsor' for x in data.columns]

        plot_data = pd.concat([data, winsor_data], axis=1)
        title = 'Data Winsor'

        qis.plot_time_series(df=plot_data,
                             title=title,
                             legend_loc='upper left',
                             legend_stats=qis.LegendStats.AVG,
                             last_label=qis.LastLabel.AVERAGE_VALUE,
                             trend_line=qis.TrendLine.AVERAGE,
                             var_format='{:.2f}')

    elif local_test == LocalTests.MARKOVIAN_WINDSOR:
        clean_x, ewm, ewm2, score = ewm_winsdor_markovian_score(a=data.to_numpy(), span=7,
                                                                init_value=0.0,
                                                                init_var=0.1*np.ones(len(data.columns)))
        clean_x = pd.DataFrame(clean_x, index=dates, columns=[x + ' winsor' for x in data.columns])
        plot_data = pd.concat([data, clean_x], axis=1)
        qis.plot_time_series(df=plot_data,
                             title='Windsor',
                             legend_loc='upper left',
                             legend_stats=qis.LegendStats.AVG_STD_LAST,
                             var_format='{:.2f}')

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.MARKOVIAN_WINDSOR)
