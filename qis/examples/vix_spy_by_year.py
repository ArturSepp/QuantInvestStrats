"""
plot changes in VIX or ATM volatility predicted by the underlying asset
data is split by years
"""
# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import yfinance as yf
import qis as qis


def plot_vol_vs_underlying(spot: pd.Series, vol: pd.Series, time_period: qis.TimePeriod = None) -> plt.Figure:
    """
    plot scatter plot of changes in the volatility predicted by returns in the underlying
    """
    df = pd.concat([spot.pct_change(), 0.01 * vol.diff(1)], axis=1).dropna()

    if time_period is not None:
        df = time_period.locate(df)

    # insert year classifier
    hue = 'year'
    df[hue] = [x.year for x in df.index]
    vol1 = vol.reindex(index=df.index, method='ffill')
    df2 = 0.01 * vol1.rename(f"{vol1.name} avg by year").to_frame()
    df2[hue] = [x.year for x in df2.index]
    df2_avg_by_year = df2.groupby(hue).mean()

    df2_avg_by_year = pd.concat([pd.Series(np.nanmean(0.01 * vol1), index=['Full sample']),
                                 df2_avg_by_year.iloc[:, 0]], axis=0)

    kwargs = dict(fontsize=12, digits_to_show=1, sharpe_digits=2,
                  alpha_format='{0:+0.0%}',
                  beta_format='{:+0.1f}',
                  perf_stats_labels=qis.PerfStatsLabels.TOTAL_DETAILED.value,
                  framealpha=0.75,
                  is_fixed_n_colors=False)

    with sns.axes_style('darkgrid'):
        fig = plt.figure(figsize=(18, 8), constrained_layout=True)
        gs = fig.add_gridspec(nrows=1, ncols=4, wspace=0.0, hspace=0.0)
        qis.plot_scatter(df=df,
                         x=str(spot.name),
                         y=str(vol.name),
                         xlabel=f'{spot.name} daily return',
                         ylabel=f'{vol.name} daily change',
                         title=f'Daily change in the {vol.name}  predicted by daily return of {spot.name} index split by years',
                         xvar_format='{:.0%}',
                         yvar_format='{:.0%}',
                         hue=hue,
                         order=2,
                         fit_intercept=False,
                         add_hue_model_label=True,
                         ci=95,
                         ax=fig.add_subplot(gs[0, :3]),
                         **kwargs)

        qis.plot_bars(df2_avg_by_year,
                      yvar_format='{:.0f}',
                      xvar_format='{:,.0%}',
                      title=f'Average {vol.name} by year',
                      xlabel=f'Average {vol.name}',
                      legend_loc=None,
                      x_rotation=0,
                      is_horizontal=True,
                      ax=fig.add_subplot(gs[0, 3]),
                      **kwargs)
    return fig


class UnitTests(Enum):
    VIX_SPY = 1
    USDJPY = 2


def run_unit_test(unit_test: UnitTests):

    time_period = qis.TimePeriod('01Jan1996', None)

    if unit_test == UnitTests.VIX_SPY:
        prices = yf.download(['SPY', '^VIX'], start=None, end=None)['Adj Close']
        fig = plot_vol_vs_underlying(spot=prices['SPY'].rename('S&P500'),
                                     vol=prices['^VIX'].rename('VIX'),
                                     time_period=time_period)
        qis.save_fig(fig, file_name='spx_vix')

    elif unit_test == UnitTests.USDJPY:
        # need to use bloomberg data
        from bbg_fetch import fetch_fields_timeseries_per_ticker
        spot = fetch_fields_timeseries_per_ticker(ticker='USDJPY Curncy', fields=['PX_LAST']).iloc[:, 0].rename('USDJPY')
        vol = fetch_fields_timeseries_per_ticker(ticker='USDJPYV1M BGN Curncy', fields=['PX_LAST']).iloc[:, 0].rename('USDJPY 1M ATM')
        fig = plot_vol_vs_underlying(spot=spot, vol=vol, time_period=time_period)
        qis.save_fig(fig, file_name='usdjpy')
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.VIX_SPY

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
