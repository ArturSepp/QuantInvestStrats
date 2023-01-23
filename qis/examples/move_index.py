# python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# qis
import qis
import yfinance as yf


FIG_SIZE21 = (15, 12)
FIG_SIZE12 = (15, 7)
LOCAL_PATH = "C://Users//artur//OneDrive//analytics//LognormalStochVol//cheyette//draft//figures//"

MOVE_NAME = 'Move volatility index'
fixed_tenor = '10y'
RVOL = 'Realized 10y rate Vol'

move = yf.download(tickers=['^MOVE'], start=None, end=None)['Adj Close'].rename(MOVE_NAME)

tickers = {'^IRX': '3m', '^FVX': '5y', '^TNX': '10y', '^TYX': '30y'}
tenors = list(tickers.values())

rates = yf.download(tickers=list(tickers.keys()), start=None, end=None)['Adj Close'].rename(tickers, axis=1)

rates = rates.reindex(index=move.index, method='ffill')*100.0

move = move.loc[:'2022']
rates = rates.loc[:'2022', :]

freq = 'W-MON'
move_change = move.resample(freq).last().diff(1).loc[:'2022']
rates_change = rates.resample(freq).last().diff(1).loc[:'2022', :]
real_vol = np.sqrt(52.0*qis.compute_ewm(data=np.square(rates_change[fixed_tenor]), span=26)).rename(RVOL)
real_vol_change = real_vol.diff(1)

joint = pd.concat([rates, move, real_vol], axis=1).iloc[10:, :]
joint_change = pd.concat([rates_change, move_change, real_vol_change], axis=1).dropna()

vol_beta = qis.compute_one_factor_ewm_betas(x=joint_change[fixed_tenor], y=joint_change[MOVE_NAME].to_frame(), span=26
                                            ).iloc[:, 0].rename('Realized Volatility Beta')
vol_vol = np.sqrt(260.0*qis.compute_ewm(data=np.square(joint_change[MOVE_NAME]), span=26, init_type=qis.InitType.MEAN)
                  ).rename('Realized Volatility-of-Volatility')

hue = 'year'
fixed_years = [2001, 2007, 2010, 2017, 2020, 2022]
colors = ['orchid', 'green', 'grey', 'olive', 'red']

with sns.axes_style('darkgrid'):

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    qis.set_suptitle(fig=fig, title='Tenor rates vs Move Index')
    for idx, tenor in enumerate(tenors):
        df = qis.add_hue_fixed_years(df=joint[[tenor, MOVE_NAME]].copy(), hue=hue, fixed_years=fixed_years)
        qis.plot_scatter(df=df,
                         hue=hue,
                         xvar_format='{:.0%}',
                         yvar_format='{:.0%}',
                         order=2,
                         ci=None,
                         title=f"{tenor}",
                         ax=axs[idx // 2][idx % 2])

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    qis.set_suptitle(fig=fig, title='Tenor rates change vs Change in Move Index')
    for idx, tenor in enumerate(tenors):
        df = qis.add_hue_fixed_years(df=joint_change[[tenor, MOVE_NAME]].copy(), hue=hue, fixed_years=fixed_years)
        qis.plot_scatter(df=df,
                         hue=hue,
                         xvar_format='{:.2%}',
                         yvar_format='{:.2%}',
                         order=2,
                         ci=None,
                         title=f"{tenor}",
                         ax=axs[idx//2][idx%2])


    kwargs = dict(fontsize=14, legend_loc='upper center', linewidth=1.5, framealpha=0.75, x_date_freq='A')

    def add_shadows(ax: plt.Subplot):
        df = qis.add_hue_fixed_years(df=rates.copy(), hue=hue, fixed_years=fixed_years)
        for idx, (key, df_) in enumerate(df.groupby(hue)):
            ax.axvspan(xmin=df_.index[0], xmax=df_.index[-1], alpha=0.2, color=colors[idx], lw=0)
        ax.set_xlim([rates.index[0], rates.index[-1]])


    #################################
    # time series figures for article
    ################################
    fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE21, tight_layout=True)
    qis.plot_time_series_2ax(df1=joint[[fixed_tenor]], df2=joint[[MOVE_NAME, RVOL]],
                             var_format='{:,.0f}',
                             var_format_yax2='{:,.0f}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                             legend_stats2=qis.LegendStats.FIRST_AVG_LAST,
                             y_limits=(0.0, None), y_limits_ax2=(0.0, None),
                             title=f"(A) Dynamics of {fixed_tenor} rate and {MOVE_NAME} in bp",
                             x_rotation=90,
                             ax=axs[0],
                             **kwargs)
    add_shadows(ax=axs[0])

    # beta and vol vol
    new_kwargs = dict(yvar_major_ticks1=np.linspace(-0.8, 0.8, 5), yvar_major_ticks2=np.linspace(0, 500.0, 6))
    qis.plot_time_series_2ax(df1=vol_beta, df2=vol_vol,
                             var_format='{:,.2f}',
                             var_format_yax2='{:,.0f}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                             legend_stats2=qis.LegendStats.FIRST_AVG_LAST,
                             #y_limits=(0.0, None), y_limits_ax2=(0.0, None),
                             title=f"(B) Realized rates beta and volatility-of-volatility of {MOVE_NAME}",
                             x_rotation=90,
                             ax=axs[1],
                             **qis.update_kwargs(kwargs, new_kwargs))
    add_shadows(ax=axs[1])
    qis.save_fig(fig=fig, file_name='timeseries', local_path=LOCAL_PATH)

    # pdf of vols
    fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE12, tight_layout=True)
    qis.plot_histogram(df=joint[[MOVE_NAME, RVOL]],
                       desc_table_type=qis.DescTableType.WITH_KURTOSIS,
                       title=f"Vols",
                       #x_min_max_quantiles=(0.01, 0.99),
                       ax=axs[0],
                       **kwargs)
    qis.plot_histogram(df=np.log(joint[[MOVE_NAME, RVOL]]),
                       desc_table_type=qis.DescTableType.WITH_KURTOSIS,
                       title=f"Log Vols",
                       #x_min_max_quantiles=(0.01, 0.99),
                       ax=axs[1],
                       **kwargs)

    # qqplots
    fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE12, tight_layout=True)
    qis.plot_qq(df=joint[[MOVE_NAME, RVOL]],
                title=f"Vols",
                #x_min_max_quantiles=(0.01, 0.99),
                ax=axs[0],
                **kwargs)
    qis.plot_qq(df=np.log(joint[[MOVE_NAME, RVOL]]),
                title=f"Log Vols",
                #x_min_max_quantiles=(0.01, 0.99),
                ax=axs[1],
                **kwargs)

    # move and real vols vs rate
    fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE12, tight_layout=True)
    df = qis.add_hue_fixed_years(df=joint[[fixed_tenor, MOVE_NAME]].copy(), hue=hue, fixed_years=fixed_years)
    qis.plot_scatter(df=df,
                     xvar_format='{:,.0f}',
                     yvar_format='{:,.0f}',
                     hue=hue,
                     order=1,
                     fit_intercept=True,
                     add_hue_model_label=True,
                     full_sample_order=0,
                     markersize=12,
                     colors=colors,
                     ci=None,
                     title=f"(A) {MOVE_NAME} vs {fixed_tenor} rate",
                     ax=axs[0],
                     **kwargs)

    df = qis.add_hue_fixed_years(df=joint[[fixed_tenor, RVOL]].copy(), hue=hue, fixed_years=fixed_years)
    qis.plot_scatter(df=df,
                     xvar_format='{:,.0f}',
                     yvar_format='{:,.0f}',
                     hue=hue,
                     order=1,
                     fit_intercept=True,
                     add_hue_model_label=True,
                     full_sample_order=0,
                     markersize=12,
                     colors=colors,
                     ci=None,
                     title=f"(B) Realized volatility vs {fixed_tenor} rate",
                     ax=axs[1],
                     **kwargs)

    # change in move and real vol vs change in rate
    fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE12, tight_layout=True)
    df = qis.add_hue_fixed_years(df=joint_change[[fixed_tenor, MOVE_NAME]].copy(), hue=hue, fixed_years=fixed_years)
    qis.plot_scatter(df=df,
                     xlabel=f"Change in {fixed_tenor} rate",
                     ylabel=f"Change in {MOVE_NAME}",
                     xvar_format='{:,.0f}',
                     yvar_format='{:,.0f}',
                     hue=hue,
                     order=2,
                     full_sample_order=2,
                     markersize=12,
                     colors=colors,
                     ci=None,
                     fit_intercept=True,
                     add_universe_model_label=False,
                     title=f"(A) Change in {MOVE_NAME} vs change in {fixed_tenor} rate",
                     ax=axs[0],
                     **kwargs)
    df = qis.add_hue_fixed_years(df=joint_change[[fixed_tenor, RVOL]].copy(), hue=hue, fixed_years=fixed_years)
    qis.plot_scatter(df=df,
                     xlabel=f"Change in {fixed_tenor} rate",
                     ylabel=f"Change in realized volatility",
                     xvar_format='{:.0%}',
                     yvar_format='{:.0%}',
                     hue=hue,
                     order=2,
                     full_sample_order=2,
                     markersize=12,
                     colors=colors,
                     ci=None,
                     fit_intercept=True,
                     add_universe_model_label=False,
                     title=f"(B) Change in realized volatility vs change in {fixed_tenor} rate",
                     ax=axs[1],
                     **kwargs)

    #################################
    # move vs rate and move changes vs rate figures for article
    ################################
    fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE12, tight_layout=True)
    df = qis.add_hue_fixed_years(df=joint[[fixed_tenor, MOVE_NAME]].copy(), hue=hue, fixed_years=fixed_years)
    qis.plot_scatter(df=df,
                     xvar_format='{:,.0f}',
                     yvar_format='{:,.0f}',
                     hue=hue,
                     order=1,
                     fit_intercept=False,
                     add_hue_model_label=True,
                     full_sample_order=0,
                     markersize=12,
                     colors=colors,
                     ci=None,
                     title=f"(A) {MOVE_NAME} vs {fixed_tenor} rate",
                     ax=axs[0],
                     **kwargs)

    df = qis.add_hue_fixed_years(df=joint_change[[fixed_tenor, MOVE_NAME]].copy(), hue=hue, fixed_years=fixed_years)
    qis.plot_scatter(df=df,
                     xlabel=f"Change in {fixed_tenor} rate",
                     ylabel=f"Change in {MOVE_NAME}",
                     xvar_format='{:,.0f}',
                     yvar_format='{:,.0f}',
                     hue=hue,
                     order=2,
                     full_sample_order=2,
                     markersize=12,
                     colors=colors,
                     ci=None,
                     fit_intercept=True,
                     add_universe_model_label=False,
                     title=f"(B) Change in {MOVE_NAME} vs change in {fixed_tenor} rate",
                     ax=axs[1],
                     **kwargs)

    qis.save_fig(fig=fig, file_name='a_move_b_change_vs_rate', local_path=LOCAL_PATH)

    #################################
    # realized vol-vol and vol-beta vs move index for article
    ################################
    fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE12, tight_layout=True)
    df = qis.add_hue_fixed_years(df=pd.concat([move, vol_vol], axis=1).dropna(), hue=hue, fixed_years=fixed_years)
    qis.plot_scatter(df=df,
                     xvar_format='{:,.0f}',
                     yvar_format='{:,.0f}',
                     hue=hue,
                     order=1,
                     full_sample_order=0,
                     fit_intercept=False,
                     markersize=12,
                     colors=colors,
                     ci=None,
                     title=f"(A) Realized volatility-of-volatility vs {MOVE_NAME}",
                     ax=axs[0],
                     **kwargs)
    # volbeta vs move
    df = qis.add_hue_fixed_years(df=pd.concat([move, vol_beta], axis=1).dropna(), hue=hue, fixed_years=fixed_years)
    qis.plot_scatter(df=df,
                     xvar_format='{:.0f}',
                     yvar_format='{:.2f}',
                     beta_format='{0:+0.4f}',
                     hue=hue,
                     order=1,
                     full_sample_order=0,
                     fit_intercept=False,
                     markersize=12,
                     colors=colors,
                     ci=None,
                     title=f"(B) Realized rates beta vs {MOVE_NAME}",
                     ax=axs[1],
                     **kwargs)
    qis.save_fig(fig=fig, file_name='move_volvol', local_path=LOCAL_PATH)

    # realized volvol and beta vs rate
    fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE12, tight_layout=True)
    df = qis.add_hue_fixed_years(df=pd.concat([rates[fixed_tenor], vol_vol], axis=1).dropna(), hue=hue, fixed_years=fixed_years)
    qis.plot_scatter(df=df,
                     xvar_format='{:,.0f}',
                     yvar_format='{:,.0f}',
                     hue=hue,
                     order=1,
                     full_sample_order=0,
                     fit_intercept=False,
                     markersize=12,
                     colors=colors,
                     ci=95,
                     title=f"Realized Volvol vs {fixed_tenor} rate",
                     ax=axs[0],
                     **kwargs)
    df = qis.add_hue_fixed_years(df=pd.concat([rates[fixed_tenor], vol_beta], axis=1).dropna(), hue=hue, fixed_years=fixed_years)
    qis.plot_scatter(df=df,
                     xvar_format='{:,.0f}',
                     yvar_format='{:,.2f}',
                     hue=hue,
                     order=1,
                     full_sample_order=0,
                     fit_intercept=False,
                     markersize=12,
                     colors=colors,
                     ci=95,
                     title=f"Realized beta vs {fixed_tenor} rate",
                     ax=axs[1],
                     **kwargs)

plt.show()
