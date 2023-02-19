# packages
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from enum import Enum

# qis
import qis


class UnitTests(Enum):
    PLOT_MIXURE = 1


def run_unit_test(unit_test: UnitTests):

    perf_params = qis.PerfParams(freq='W-WED')
    kwargs = dict(fontsize=12, digits_to_show=1, sharpe_digits=2,
                  alpha_format='{0:+0.0%}',
                  beta_format='{:0.1f}',
                  alpha_an_factor=12)

    time_periods = [qis.TimePeriod('31Aug2002', '31Dec2019'), qis.TimePeriod('31Dec2019', '16Dec2022')]

    if unit_test == UnitTests.PLOT_MIXURE:

        prices = yf.download(tickers=['SPY', 'TLT'], start=None, end=None)['Adj Close'].dropna()

        n_components = 3

        with sns.axes_style('white'):
            fig1, axs = plt.subplots(1, len(time_periods), figsize=(15, 5), constrained_layout=True)

        for idx, time_period in enumerate(time_periods):
            prices_ = time_period.locate(prices)
            rets = qis.to_returns(prices=prices_, is_log_returns=True, drop_first=True, freq=perf_params.freq)
            params = qis.fit_gaussian_mixture(x=rets.to_numpy(), n_components=n_components, idx=1)
            qis.plot_mixure2(x=rets.to_numpy(),
                            n_components=n_components,
                            columns=prices.columns,
                            title=f"({idx+1}) Returns and ellipsoids of Gaussian clusters for period {time_period.to_str()}",
                            ax=axs[idx],
                            **kwargs)

            means, vols, corrs = params.get_all_params(columns=prices.columns, vol_scaler=12.0)
            print(f"means=\n{means}")
            print(f"vols=\n{vols}")
            print(f"corrs=\n{corrs}")

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_MIXURE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
