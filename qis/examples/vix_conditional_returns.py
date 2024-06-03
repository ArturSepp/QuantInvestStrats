"""
compute and display conditial returns on short front month VIX future strategy SPVXSPI conditioned on
different predictors: VIX and vols
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import qis as qis
from qis import PerfStat
from bbg_fetch import fetch_field_timeseries_per_tickers

# define names for convenience
benchmark_name = 'SPX'
strategy_name = 'ShortVix'
predictor_name = 'VIX'
benchmark_vol_name = f"{benchmark_name} real vol"
vix_vol_spread_name = f"Spread = {predictor_name} - real vol"
strategy_vol_name = f"{strategy_name} real vol"

# get index from bloomberg (run with open bloomberg terminal)
assets = {'SPXT Index': benchmark_name, 'SPVXSPI Index': strategy_name, 'VIX Index': predictor_name}
prices = fetch_field_timeseries_per_tickers(tickers=list(assets.keys()), field='PX_LAST', CshAdjNormal=True).dropna()
prices = prices.rename(assets, axis=1)

# define explanatory vars on monthly frequency, vol is multiplied by 100.0
monthly_return = prices[strategy_name].asfreq('ME', method='ffill').pct_change()
predictor_1 = prices[predictor_name].asfreq('ME', method='ffill').shift(1)
monthly_benchmark_vol_1 = 100.0*np.sqrt(252)*prices[benchmark_name].pct_change().rolling(21).std().asfreq('ME', method='ffill').shift(1).rename(benchmark_vol_name)
vix_vol_spread = predictor_1.subtract(monthly_benchmark_vol_1).rename(vix_vol_spread_name)
strategy_vol = 100.0*np.sqrt(252)*prices[strategy_name].pct_change().rolling(21).std().asfreq('ME', method='ffill').shift(1).rename(strategy_vol_name)
monthly_df = pd.concat([monthly_return, predictor_1, monthly_benchmark_vol_1, vix_vol_spread, strategy_vol], axis=1).dropna()

with sns.axes_style("darkgrid"):
    fig1, axs = plt.subplots(2, 3, figsize=(18, 9))
    qis.plot_prices_with_dd(prices=prices[[benchmark_name, strategy_name]],
                            regime_benchmark_str=benchmark_name,
                            perf_stats_labels=[PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0, PerfStat.MAX_DD],
                            title=f"Performances of ShortVix (SPVXSPI Index) and SPX (SPXT Index)",
                            x_date_freq='YE',
                            framealpha=0.9, fontsize=8,
                            axs=axs[:, 0])

    kwargs = dict(y=strategy_name,
                  num_buckets=6, ylabel=f"{strategy_name} monthly return",
                  yvar_format='{:.0%}', xvar_format='{:.0f}', showfliers=True, fontsize=10)
    qis.df_boxplot_by_classification_var(df=monthly_df,
                                         x=predictor_name,
                                         title=f"Monthly returns conditional on {predictor_name} at month start",
                                         x_hue_name=f"{predictor_name} month start",
                                         ax=axs[0, 1], **kwargs)

    qis.df_boxplot_by_classification_var(df=monthly_df,
                                         x=benchmark_vol_name,
                                         title=f"Monthly returns conditional on {benchmark_vol_name} at month start",
                                         x_hue_name=f"{benchmark_vol_name} month start",
                                         ax=axs[1, 1], **kwargs)

    qis.df_boxplot_by_classification_var(df=monthly_df,
                                         x=vix_vol_spread_name,
                                         title=f"Monthly returns conditional on {vix_vol_spread_name} at month start",
                                         x_hue_name=f"{vix_vol_spread_name} month start",
                                         ax=axs[0, 2], **kwargs)

    qis.df_boxplot_by_classification_var(df=monthly_df,
                                         x=strategy_vol_name,
                                         title=f"Monthly returns conditional on {strategy_vol_name} at month start",
                                         x_hue_name=f"{strategy_vol_name} month start",
                                         ax=axs[1, 2], **kwargs)
plt.show()
