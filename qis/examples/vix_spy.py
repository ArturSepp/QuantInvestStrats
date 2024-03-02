# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

# qis
import qis.utils.dates as da
from qis.perfstats.config import PerfParams
import qis.plots.derived.prices as pdp
import qis.plots.scatter as psc

# project
import yfinance as yf


class UnitTests(Enum):
    VIX_SPY = 1


def run_unit_test(unit_test: UnitTests):

    perf_params = PerfParams(freq_drawdown='B', freq='B')
    kwargs = dict(fontsize=12, digits_to_show=1, sharpe_digits=2,
                  alpha_format='{0:+0.0%}',
                  beta_format='{:0.1f}',
                  performance_label=pdp.PerformanceLabel.TOTAL_DETAILED,
                  framealpha=0.9)

    time_period = da.TimePeriod('31Dec2016', None)

    if unit_test == UnitTests.VIX_SPY:
        prices = time_period.locate(yf.download(['SPY', '^VIX'], start=None, end=None)['Adj Close'])

        df1 = prices['SPY'].pct_change()
        df2 = 0.01*prices['^VIX'].rename('VIX').diff(1)
        df = pd.concat([df1, df2], axis=1).dropna()

        hue = 'year'
        df[hue] = [x.year for x in df.index]
        print(df)

        with sns.axes_style('darkgrid'):
            fig1, ax = plt.subplots(1, 1, figsize=(15, 8), constrained_layout=True)
            psc.plot_scatter(df=df,
                             x='SPY',
                             y='VIX',
                             xlabel='S&P 500 daily return',
                             ylabel='VIX daily change',
                             title='Daily change in the VIX predicted by daily return of S&P 500 index split by years',
                             xvar_format='{:.0%}',
                             yvar_format='{:.0%}',
                             hue=hue,
                             order=2,
                             fit_intercept=False,
                             add_hue_model_label=True,
                             #add_universe_model_ci=True,
                             ci=95,
                             # annotation_labels=annotation_labels,
                             #full_sample_label='Universe: ',
                             ax=ax,
                             **kwargs)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.VIX_SPY

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
