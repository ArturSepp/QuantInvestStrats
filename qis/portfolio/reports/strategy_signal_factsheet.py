"""
report for PortfolioData.strategy_signal_data
"""
import numpy as np
# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import qis as qis
from qis import TimePeriod, BenchmarkReturnsQuantileRegimeSpecs
from qis.portfolio.portfolio_data import PortfolioData, StrategySignalData


def generate_weight_change_report(portfolio_data: PortfolioData,
                                  time_period: TimePeriod,
                                  is_grouped: bool = True,
                                  sample_size: int = 20,
                                  figsize: Tuple[float, float] = (8.3, 11.7),
                                  verbose: bool = False,
                                  is_add_residual_to_momentum: bool = True,
                                  group_deflator: Optional[Dict[str, float]] = dict(STIR=0.1),
                                  **kwargs
                                  ) -> plt.Figure:
    strategy_signal_data = portfolio_data.strategy_signal_data
    if strategy_signal_data is None:
        raise ValueError(f"portfolio_data.strategy_signal_data must be provided")

    if is_grouped:
        predictions_g, fitted_models_g, prediction_period = strategy_signal_data.estimate_signal_changes_by_groups(
            group_data=portfolio_data.group_data,
            group_order=portfolio_data.group_order,
            time_period=time_period,
            sample_size=sample_size)

        group_preds = {}
        agg_preds = {}
        for idx, (group, df) in enumerate(predictions_g.items()):
            if is_add_residual_to_momentum:
                df_mom = df['momentum_change'].add(df['residual']).rename('momentum_change')
                df_ac = pd.concat([df_mom, df[['carry_change', 'target_vol_change', 'port_leverage_change']]], axis=1)
            else:
                df_ac = df[['momentum_change', 'carry_change', 'target_vol_change', 'port_leverage_change', 'residual']]
            group_preds[group] = df_ac.copy()
            if group_deflator is not None and group in group_deflator.keys():
                df_ac *= group_deflator[group]
            agg_preds[group] = df_ac.sum(0)
        agg_preds = pd.DataFrame.from_dict(agg_preds, orient='index')

        post_agg = ''
        if group_deflator is not None:
            this = ''
            for key, deflator in group_deflator.items():
                this += f"{key}={deflator:0.2f}"
            post_agg = f" (using deflator for {this})"
        preds_dict = {f"Total by groups {post_agg}": agg_preds}
        preds_dict.update(group_preds)

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(len(preds_dict.keys())//2+len(preds_dict.keys())%2, 2, figsize=figsize, tight_layout=True)
            axs = qis.to_flat_list(axs)
            weight_freq = pd.infer_freq(strategy_signal_data.weights.index)

            qis.set_suptitle(fig, title=f"{portfolio_data.ticker} Attribution of changes in weights for period "
                                        f"{prediction_period.to_str()}"
                                        f" using last {sample_size} {weight_freq}-freq periods",
                             fontweight="bold", fontsize=8, color='blue')

        for idx, (group, df_ac) in enumerate(preds_dict.items()):
            if verbose:
                print(f"{group}")
                print(df_ac)
                print(f"{fitted_models_g[group].summary()}")
            qis.plot_bars(df=df_ac,
                          stacked=True,
                          totals=df_ac.sum(1).to_list(),
                          annotate_totals=False,
                          title=f"{group}",
                          ncols=len(df_ac.columns)//2,
                          yvar_format='{:,.1%}',
                          ax=axs[idx],
                          **kwargs)
    else:
        raise NotImplementedError
    return fig


def generate_current_signal_report(portfolio_data: PortfolioData,
                                   y_limits: Tuple[Optional[float], Optional[float]] = (-1.0, 1.0),
                                   figsize: Tuple[float, float] = (8.3, 11.7),
                                   **kwargs
                                   ) -> plt.Figure:
    strategy_signal_data = portfolio_data.strategy_signal_data
    if strategy_signal_data is None:
        raise ValueError(f"portfolio_data.strategy_signal_data must be provided")

    agg_by_group_dict = strategy_signal_data.get_current_signal_by_groups(group_data=portfolio_data.group_data,
                                                                          group_order=portfolio_data.group_order)

    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(len(agg_by_group_dict.keys())//2+len(agg_by_group_dict.keys())%2, 2,
                                figsize=figsize, tight_layout=True)
        axs = qis.to_flat_list(axs)
        qis.set_suptitle(fig, title=(f"{portfolio_data.ticker} Signals for period "
                                     f"{qis.date_to_str(strategy_signal_data.signal.index[-21])} and "
                                     f"{qis.date_to_str(strategy_signal_data.signal.index[-1])}; "
                                     f"min, max = [{y_limits[0]:0.2f}, {y_limits[1]:0.2f}]"),
                         fontweight="bold", fontsize=8, color='blue')

    for idx, (group, df_ac) in enumerate(agg_by_group_dict.items()):
        if y_limits is not None:
            qis.set_y_limits(ax=axs[idx], y_limits=y_limits)
        qis.plot_bars(df=df_ac,
                      stacked=False,
                      title=f"{group}",
                      ncols=2,
                      yvar_format='{:,.1f}',
                      ax=axs[idx],
                      **kwargs)
    return fig


def generate_strategy_signal_factsheet_by_instrument(strategy_signal_data: StrategySignalData,
                                                     time_period: TimePeriod,
                                                     figsize: Tuple[float, float] = (8.3, 11.7),  # A4 for portrait
                                                     fontsize: int = 4,
                                                     regime_params: BenchmarkReturnsQuantileRegimeSpecs = None,
                                                     **kwargs
                                                     ) -> List[plt.Figure]:

    if strategy_signal_data is None:
        raise ValueError(f"portfolio_data.strategy_signal_data must be provided")

    navs = qis.returns_to_nav(returns=np.expm1(strategy_signal_data.log_returns))
    if time_period is not None:
        strategy_signal_data = strategy_signal_data.locate_period(time_period=time_period)
        navs = time_period.locate(navs)

    plot_kwargs = dict(fontsize=fontsize,
                       linewidth=0.5,
                       digits_to_show=1, sharpe_digits=2,
                       weight='normal',
                       markersize=1,
                       framealpha=0.75)
    kwargs = qis.update_kwargs(kwargs, plot_kwargs)
    figs = []
    for instrument in strategy_signal_data.signal:
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(7, 2, figsize=figsize, tight_layout=True)
            qis.set_suptitle(fig, title=f"{instrument}")
            figs.append(fig)

        datas = dict(nav=navs[instrument],
                     signal=strategy_signal_data.signal[instrument],
                     vol=strategy_signal_data.instrument_vols[instrument],
                     target_vol=strategy_signal_data.instrument_target_vols[instrument],
                     target_signal_vol_weight=strategy_signal_data.instrument_target_signal_vol_weights[instrument],
                     portfolio_leverages=strategy_signal_data.instrument_portfolio_leverages[instrument],
                     weights=strategy_signal_data.weights[instrument])
        var_formats = ['{:.2%}', '{:.2f}', '{:.2%}', '{:.2%}', '{:.2%}', '{:.2f}', '{:.2%}']

        for idx, (key, df) in enumerate(datas.items()):

            if idx == 0:
                df0 = df / df.iloc[0]
                df1 = df.pct_change()
                title1 = 'Log-returns'
                qis.plot_prices(prices=df0,
                                title='Performance',
                                var_format='{:,.2f}',
                                ax=axs[idx][0],
                                **kwargs)
            else:
                df1 = df
                title1 = key
                qis.plot_time_series(df=df,
                                     # trend_line=qis.TrendLine.ZERO_SHADOWS,
                                     var_format=var_formats[idx],
                                     title=key,
                                     legend_stats=qis.LegendStats.AVG_MIN_MAX_LAST,
                                     ax=axs[idx][0],
                                     **kwargs)
            qis.add_bnb_regime_shadows(ax=axs[idx][0],
                                       pivot_prices=navs[instrument].reindex(index=df.index, method='ffill'),
                                       regime_params=regime_params)

            qis.plot_histogram(df=df1,
                               # trend_line=qis.TrendLine.ZERO_SHADOWS,
                               var_format=var_formats[idx],
                               title=title1,
                               legend_stats=qis.LegendStats.AVG_MIN_MAX_LAST,
                               desc_table_type=None,
                               ax=axs[idx][1],
                               **kwargs)
            plt.close('all')
    return figs
