"""
FX hedging research reports.

Single-asset and multi-asset hedging tearsheets built on ``qis.plots``. These
are research/reporting helpers, not part of the data container. No seaborn
dependency: the prior ``sns.axes_style`` styling wrapper has been removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
from typing import Dict, Optional, Tuple, Union

from qis.market_data.fx_rates_data import FxRatesData
from qis.market_data.fx_hedging import (compute_fx_optimal_hedge,
                                        compute_fx_vol_beta,
                                        compute_performance_of_local_ccy_asset_in_reference_ccy)


def run_asset_fx_hedging_report(asset_price_local_ccy: pd.Series,
                                fx_rates_data: FxRatesData,
                                local_ccy: str = 'USD',
                                reference_ccy: str = 'CHF',
                                time_period: qis.TimePeriod = None,
                                freq: str = 'ME',
                                span: int = 3 * 12,
                                risk_aversion_lambda: float = 4.0 / 3.0,
                                min_max_hedge: Optional[Tuple[float, float]] = (0.0, 1.0)
                                ) -> plt.Figure:
    """Single-asset FX-hedging tearsheet for a local-currency asset in a reference ccy.

    Builds the reference-currency NAVs at hedge ratios ``h in {0, 0.5, 1}`` plus the
    beta-, carry-, and optimal-hedge variants, and plots a risk-adjusted performance
    table, the NAV/drawdown panel, a Sharpe bar, the hedge ratios over time, and the
    EWMA FX beta/vol. The optimizer knobs (``span``, ``risk_aversion_lambda``,
    ``min_max_hedge``) are passed explicitly to ``compute_fx_optimal_hedge`` and
    ``compute_fx_vol_beta`` rather than relying on their defaults.

    Args:
        asset_price_local_ccy: Asset price quoted in ``local_ccy``.
        fx_rates_data: FX rates container providing the cross rate and CIP forward.
        local_ccy: Currency the asset is denominated in.
        reference_ccy: Reporting / investor currency.
        time_period: Optional date filter for the plotted series.
        freq: Resampling frequency (e.g. ``'ME'``).
        span: EWMA span (periods of ``freq``) for the FX vol/beta estimates.
        risk_aversion_lambda: Mean-variance risk aversion for the carry tilt.
        min_max_hedge: Optional ``(min, max)`` clip on the hedge ratios.

    Returns:
        The assembled matplotlib Figure.
    """
    # Extract FX market universe
    local_to_reference_fx_rate = fx_rates_data.get_local_to_reference_fx_rate(local_ccy=local_ccy,
                                                                              reference_ccy=reference_ccy)
    forward_rate_for_local_ccy = fx_rates_data.get_forward_rate_for_local_ccy(local_ccy=local_ccy,
                                                                              reference_ccy=reference_ccy, freq=freq)
    carry_fx_nav = fx_rates_data.get_carry_fx_return_nav(local_ccy=local_ccy, reference_ccy=reference_ccy,
                                                         freq=freq,
                                                         time_period=time_period)

    kwargs = dict(asset_price_local_ccy=asset_price_local_ccy,
                  local_to_reference_fx_rate=local_to_reference_fx_rate,
                  forward_rate_for_local_ccy=forward_rate_for_local_ccy, freq=freq)

    # Calculate optimal hedge strategies
    optimal_hedge, max_carry, beta_hedged = compute_fx_optimal_hedge(**kwargs, span=span,
                                                                     risk_aversion_lambda=risk_aversion_lambda,
                                                                     min_max_hedge=min_max_hedge)
    hedges = pd.concat([optimal_hedge, max_carry, beta_hedged], axis=1)

    # Generate NAVs for different hedge ratios
    nav0, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=0.0, **kwargs)
    nav05, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=0.5, **kwargs)
    nav1, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=1.0, **kwargs)
    nav_beta, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=beta_hedged, **kwargs)
    nav_carry, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=max_carry, **kwargs)
    nav_optimal, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=optimal_hedge, **kwargs)

    navs = pd.concat([asset_price_local_ccy,
                      local_to_reference_fx_rate.rename(f"{carry_fx_nav.name} spot return"),
                      carry_fx_nav.rename(f"{carry_fx_nav.name} carry return"),
                      nav0.rename('h=0.0'), nav05.rename('h=0.5'), nav1.rename('h=1.0'),
                      nav_beta.rename('Beta-Hedged'),
                      nav_carry.rename('Carry-Hedged'),
                      nav_optimal.rename('Optimal-Hedged')], axis=1)

    fx_vol, fx_beta = compute_fx_vol_beta(asset_price_local_ccy=asset_price_local_ccy,
                                          local_to_reference_fx_rate=local_to_reference_fx_rate, freq=freq,
                                          span=span)

    # Create visualization
    perf_params = qis.PerfParams(freq='ME')
    kwargs = dict(digits_to_show=1, framealpha=0.9)
    fig, axs = plt.subplots(3, 2, figsize=(18, 10))

    if time_period is not None:
        navs = time_period.locate(navs)
    qis.set_suptitle(fig, title=f"{asset_price_local_ccy.name} local in {local_ccy} reference in {reference_ccy}:"
                                f" {qis.get_time_period(navs).to_str()}")
    qis.plot_ra_perf_table_benchmark(prices=navs,
                                     benchmark=str(asset_price_local_ccy.name),
                                     heatmap_columns=[1, 3],
                                     rows_edge_lines=[1, 2, 3, 6],
                                     ax=axs[0, 0], **kwargs)
    qis.plot_prices_with_dd(prices=navs, perf_params=perf_params, axs=axs[1:, 0], **kwargs)

    perf_column = qis.PerfStat.SHARPE_RF0
    qis.plot_ra_perf_bars(prices=navs,
                          perf_column=perf_column,
                          perf_params=perf_params,
                          title= f"{perf_column.to_str()}",
                          ax=axs[0, 1],
                          **kwargs)

    if time_period is not None:
        hedges = time_period.locate(hedges)
    qis.plot_time_series(hedges, title='hedges', ax=axs[1, 1], **kwargs)

    if time_period is not None:
        fx_beta = time_period.locate(fx_beta)
        fx_vol = time_period.locate(fx_vol)
    qis.plot_time_series_2ax(df1=fx_beta.rename('FX-beta'), df2=fx_vol.rename('Fx-vol'),
                             legend_stats=qis.LegendStats.AVG_LAST,
                             legend_stats2=qis.LegendStats.AVG_LAST,
                             var_format='{:,.2f}',
                             var_format_yax2='{:,.2%}',
                             ax=axs[2, 1],
                             **kwargs)
    return fig


def compute_multi_asset_fx_hedging(asset_prices: pd.DataFrame,
                                   fx_rates_data: FxRatesData,
                                   time_period: qis.TimePeriod = None,
                                   local_ccys: Union[str, pd.Series] = 'USD',
                                   reference_ccy: str = 'CHF',
                                   freq: str = 'ME',
                                   span: int = 3 * 12,
                                   risk_aversion_lambda: float = 4.0 / 3.0,
                                   min_max_hedge: Optional[Tuple[float, float]] = (0.0, 1.0)
                                   ) -> Dict[str, pd.DataFrame]:
    """Hedging performance metrics for a panel of local-currency assets in a reference ccy.

    For each asset, builds the reference-currency NAVs at ``h in {0, 0.5, 1}`` plus the
    beta/carry/optimal variants and reads the risk-adjusted stats off a benchmark perf
    table (benchmark = the unconverted local asset). The optimizer knobs (``span``,
    ``risk_aversion_lambda``, ``min_max_hedge``) are passed explicitly to
    ``compute_fx_optimal_hedge`` rather than relying on its defaults.

    ``local_ccys`` may be a single string (all assets share one currency, so the cross
    rate and forward are computed once in the outer scope) or a per-asset Series
    (resolved inside the loop).

    Args:
        asset_prices: Native-currency price panel (columns are assets).
        fx_rates_data: FX rates container.
        time_period: Optional date filter applied to each asset's NAV panel.
        local_ccys: One currency for all assets (str) or per-asset currencies (Series).
        reference_ccy: Reporting / investor currency.
        freq: Resampling frequency (e.g. ``'ME'``).
        span: EWMA span (periods of ``freq``) for the FX vol/beta estimates.
        risk_aversion_lambda: Mean-variance risk aversion for the carry tilt.
        min_max_hedge: Optional ``(min, max)`` clip on the hedge ratios.

    Returns:
        Dict of DataFrames ``pas``, ``sharpes``, ``betas``, ``vols`` (asset x strategy)
        and ``last_hedges`` (asset x {Optimal, Max Carry, Beta Hedge}).
    """
    # Initialize FX universe if common currency
    if isinstance(local_ccys, str):
        local_to_reference_fx_rate = fx_rates_data.get_local_to_reference_fx_rate(local_ccy=local_ccys,
                                                                                  reference_ccy=reference_ccy)
        forward_rate_for_local_ccy = fx_rates_data.get_forward_rate_for_local_ccy(local_ccy=local_ccys,
                                                                                  reference_ccy=reference_ccy, freq=freq)

    perf_params = qis.PerfParams(freq='ME')

    # Iterate through assets and calculate metrics
    pas, sharpes, betas, last_hedges, vols = {}, {}, {}, {}, {}
    for asset in asset_prices.columns:
        if isinstance(local_ccys, pd.Series):
            local_ccy = local_ccys.loc[asset]
        else:
            local_ccy = local_ccys

        if local_ccy != reference_ccy:
            asset_price_local_ccy = asset_prices[asset]
            # Per-asset FX universe when local_ccys is a Series; otherwise reuse the
            # common-currency rate and forward computed once in the outer scope above.
            if not isinstance(local_ccys, str):
                local_to_reference_fx_rate = fx_rates_data.get_local_to_reference_fx_rate(local_ccy=local_ccy,
                                                                                          reference_ccy=reference_ccy)
                forward_rate_for_local_ccy = fx_rates_data.get_forward_rate_for_local_ccy(local_ccy=local_ccy,
                                                                                          reference_ccy=reference_ccy,
                                                                                          freq=freq)

            kwargs = dict(asset_price_local_ccy=asset_price_local_ccy,
                          local_to_reference_fx_rate=local_to_reference_fx_rate,
                          forward_rate_for_local_ccy=forward_rate_for_local_ccy, freq=freq)

            # Calculate hedge strategies
            optimal_hedge, max_carry, beta_hedged = compute_fx_optimal_hedge(**kwargs, span=span,
                                                                             risk_aversion_lambda=risk_aversion_lambda,
                                                                             min_max_hedge=min_max_hedge)
            hedges = pd.concat([optimal_hedge, max_carry, beta_hedged], axis=1)

            # Generate NAVs for comparison
            nav0, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=0.0, **kwargs)
            nav05, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=0.5, **kwargs)
            nav1, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=1.0, **kwargs)
            nav_beta, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=beta_hedged, **kwargs)
            nav_carry, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=max_carry, **kwargs)
            nav_optimal, _ = compute_performance_of_local_ccy_asset_in_reference_ccy(hedge_ratio=optimal_hedge, **kwargs)

            navs = pd.concat([asset_price_local_ccy.rename('LocalAsset'),
                              nav0.rename('h=0.0'), nav05.rename('h=0.5'), nav1.rename('h=1.0'),
                              nav_beta.rename('Beta-Hedged'),
                              nav_carry.rename('Carry-Hedged'),
                              nav_optimal.rename('Optimal-Hedged')], axis=1)

            if time_period is not None:
                navs = time_period.locate(navs)

            # Extract performance metrics
            perf_table = qis.compute_ra_perf_table_with_benchmark(prices=navs,
                                                     benchmark='LocalAsset',
                                                     perf_params=perf_params)
            pas[asset] = perf_table[qis.PerfStat.PA_RETURN.to_str()]
            sharpes[asset] = perf_table[qis.PerfStat.SHARPE_RF0.to_str()]
            betas[asset] = perf_table[qis.PerfStat.BETA.to_str()]
            vols[asset] = perf_table[qis.PerfStat.VOL.to_str()]
            last_hedges[asset] = hedges.iloc[-1, :]

    # Consolidate results into dataframes
    pas = pd.DataFrame.from_dict(pas, orient='index')
    sharpes = pd.DataFrame.from_dict(sharpes, orient='index')
    betas = pd.DataFrame.from_dict(betas, orient='index')
    vols = pd.DataFrame.from_dict(vols, orient='index')
    last_hedges = pd.DataFrame.from_dict(last_hedges, orient='index')
    out = dict(pas=pas, sharpes=sharpes, betas=betas, vols=vols, last_hedges=last_hedges)
    return out


def plot_multi_asset_fx_hedging_report(asset_prices: pd.DataFrame,
                                      fx_rates_data: FxRatesData,
                                      time_period: qis.TimePeriod = None,
                                      local_ccy: str = 'USD',
                                      reference_ccy: str = 'CHF',
                                      freq: str = 'ME',
                                      span: int = 3 * 12,
                                      risk_aversion_lambda: float = 4.0 / 3.0,
                                      min_max_hedge: Optional[Tuple[float, float]] = (0.0, 1.0)
                                      ) -> plt.Figure:
    """Multi-asset FX-hedging comparison report (heatmap tables).

    Runs ``compute_multi_asset_fx_hedging`` over the panel and renders p.a. return,
    volatility, and Sharpe tables (asset x hedge strategy) as heatmaps. The optimizer
    knobs are forwarded explicitly to ``compute_multi_asset_fx_hedging``.

    Args:
        asset_prices: Native-currency price panel (columns are assets).
        fx_rates_data: FX rates container.
        time_period: Optional date filter applied to each asset's NAV panel.
        local_ccy: Common currency of denomination for all assets.
        reference_ccy: Reporting / investor currency.
        freq: Resampling frequency (e.g. ``'ME'``).
        span: EWMA span (periods of ``freq``) for the FX vol/beta estimates.
        risk_aversion_lambda: Mean-variance risk aversion for the carry tilt.
        min_max_hedge: Optional ``(min, max)`` clip on the hedge ratios.

    Returns:
        The assembled matplotlib Figure.
    """
    out = compute_multi_asset_fx_hedging(asset_prices=asset_prices,
                                         fx_rates_data=fx_rates_data,
                                         time_period=time_period,
                                         local_ccys=local_ccy,
                                         reference_ccy=reference_ccy,
                                         freq=freq,
                                         span=span,
                                         risk_aversion_lambda=risk_aversion_lambda,
                                         min_max_hedge=min_max_hedge)
    pas = out['pas']
    sharpes = out['sharpes']
    betas = out['betas']
    vols = out['vols']

    fig, axs = plt.subplots(3, 1, figsize=(18, 10))
    if time_period is not None:
        title = f"FX hedging report: {time_period.to_str()}"
    else:
        title = f"FX hedging report"
    qis.set_suptitle(fig, title=title)
    heatmap_rows = [n for n in np.arange(len(pas.index))]
    kwargs = dict(heatmap_rows=heatmap_rows)
    qis.plot_df_table(df=qis.df_to_str(df=pas, var_format='{:.1%}'), title='P.a. return', ax=axs[0], **kwargs)
    qis.plot_df_table(df=qis.df_to_str(df=vols, var_format='{:.1%}'), title='Vols', ax=axs[1], **kwargs)
    qis.plot_df_table(df=qis.df_to_str(df=sharpes, var_format='{:.2f}'), title='Sharpe ratios', ax=axs[2], **kwargs)

    return fig