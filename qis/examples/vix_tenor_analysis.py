import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import qis as qis
from bbg_fetch import fetch_field_timeseries_per_tickers

vix_tickers = ['VIX1D Index', 'VIX9D Index', 'VIX Index', 'VIX3M Index', 'VIX6M Index', 'VIX1Y Index']

vols = 0.01*fetch_field_timeseries_per_tickers(tickers=vix_tickers, field='PX_LAST', CshAdjNormal=True).dropna()
benchmark = 'SPX Index'
snp = fetch_field_timeseries_per_tickers(tickers=[benchmark], field='PX_LAST', CshAdjNormal=True).dropna()
snp = snp.reindex(index=vols.index, method='ffill')
df = pd.concat([snp.pct_change(), vols.diff(1)], axis=1).dropna()

with sns.axes_style("darkgrid"):
    # qis.plot_time_series(df=vols, ax=axs[0])
    # qis.plot_prices(prices=snp, ax=axs[0])
    kwargs = dict(fontsize=12, framealpha=0.75)

    corr = qis.compute_masked_covar_corr(data=df, is_covar=False)
    fig1, ax = plt.subplots(1, 1, figsize=(4, 4))
    qis.plot_heatmap(df=corr,
                     var_format='{:.0%}',
                     cmap='PiYG',
                     title=f"Correlation between daily changes for period {qis.get_time_period_label(df, date_separator='-')}",
                     fontsize=12,
                     ax=ax)
    """
    fig1, axs = plt.subplots(2, 1, figsize=(18, 9))
    qis.plot_scatter(df=df,
                     x=benchmark,
                     xlabel=f'{benchmark} daily return',
                     ylabel=f'Vix daily change',
                     title=f'Daily change in VIX indices predicted by daily return of benchmark',
                     xvar_format='{:.0%}',
                     yvar_format='{:.0%}',
                     order=2,
                     fit_intercept=True,
                     add_hue_model_label=True,
                     ci=95,
                     ax=axs[0],
                     **kwargs)
    """
    fig2, axs = plt.subplots(2, 3, figsize=(18, 9))
    axs = qis.to_flat_list(axs)
    for idx, vol in enumerate(vols.columns):
        qis.plot_classification_scatter(df=df[[benchmark, vol]],
                                        full_sample_order=1,
                                        num_buckets=2,
                                        xvar_format='{:.0%}',
                                        yvar_format='{:.0%}',
                                        fit_intercept=True,
                                        title=f"{vol}",
                                        ax=axs[idx],
                                        **kwargs)

plt.show()
