"""
two delta-one strategy simulations, used as examples and as reference implementations:
``simulate_vol_target_strats`` scales each asset by ``vol_target`` over its EWM volatility and
returns weights and navs; ``simulate_trend_strats`` multiplies that by a unit-variance EWM trend
signal and returns weights, navs and the signal. Weights are applied lagged one period. The
``*_range`` variants sweep one span each - ``vol_spans``, and ``tf_spans`` with ``vol_span`` held
fixed - and prepend the underlying prices to the navs frame unless ``add_asset`` is off. Neither
returns a portfolio object, so a report over them goes through ``qis.backtest_model_portfolio``.

``vol_target`` is annual. ``vol_af`` converts it to the per-period target of daily data and
defaults to 252, qis's business-day annualisation factor (``qis.get_annualization_factor('B')``),
so a 15% target yields 15% realised volatility as qis reports it.
"""

# packages
import numpy as np
import pandas as pd
from typing import Tuple, Union, List

# qis
import qis.utils as qu
import qis.perfstats.returns as ret
import qis.models.linear.ewm as ewm


def simulate_vol_target_strats(prices: Union[pd.DataFrame, pd.Series],
                               vol_span: int = 21,
                               vol_target: float = 0.15,
                               constant_trade_level: bool = False,
                               vol_af: float = 252
                               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    simulate weights and returns on vol target.

    The weight is ``vol_target / (sqrt(vol_af) * sigma_t)`` with ``sigma_t`` the per-period EWM
    volatility of log returns, applied to simple returns one period later: the same as
    ``qis.compute_ra_returns`` with the per-period target ``vol_target / sqrt(vol_af)``.

    Args:
        prices: asset prices
        vol_span: EWM span of the volatility
        vol_target: annual volatility target
        constant_trade_level: passed to ``returns_to_nav``
        vol_af: periods per year of the price grid; 252 for business days, as in qis

    Returns:
        the pair ``(nav_weights, vt_navs)`` of unlagged target weights and strategy navs
    """
    log_returns = ret.to_returns(prices=prices, is_log_returns=True)
    returns = ret.to_returns(prices=prices, is_log_returns=False)
    ewm_vol = ewm.compute_ewm_vol(data=log_returns,
                                  span=vol_span,
                                  mean_adj_type=ewm.MeanAdjType.NONE,
                                  annualization_factor=vol_af)
    # vol target weights
    weights_100 = qu.to_finite_reciprocal(data=ewm_vol, fill_value=0.0, is_gt_zero=True)
    nav_weights = weights_100.multiply(vol_target)
    vt_returns = returns.multiply(nav_weights.shift(1))
    vt_navs = ret.returns_to_nav(returns=vt_returns, constant_trade_level=constant_trade_level)
    return nav_weights, vt_navs


def simulate_vol_target_strats_range(prices: Union[pd.DataFrame, pd.Series],
                                     vol_spans: List[int] = (21, 31),
                                     vol_target: float = 0.15,
                                     constant_trade_level: bool = False,
                                     vol_af: float = 252,
                                     add_asset: bool = True
                                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vt_nav_weights, vt_navs = [], []
    for vol_span in vol_spans:
        vt_nav_weights_, vt_navs_ = simulate_vol_target_strats(prices=prices, vol_span=vol_span, vol_target=vol_target,
                                                               constant_trade_level=constant_trade_level, vol_af=vol_af)
        if isinstance(prices, pd.Series):
            name = f"{prices.name} vol_span={vol_span}"
            vt_nav_weights_.name, vt_navs_.name = name, name
        else:
            names = [f"{x} vol_span={vol_span}" for x in prices.columns]
            vt_nav_weights_.columns, vt_navs_.columns = names, names
        vt_nav_weights.append(vt_nav_weights_)
        vt_navs.append(vt_navs_)
    vt_nav_weights = pd.concat(vt_nav_weights, axis=1, sort=True)
    vt_navs = pd.concat(vt_navs, axis=1, sort=True)
    if add_asset:
        vt_navs = pd.concat([prices, vt_navs], axis=1, sort=True)
    return vt_nav_weights, vt_navs


def simulate_trend_strats(prices: Union[pd.DataFrame, pd.Series],
                          vol_span: int = 33,
                          tf_span: int = 63,
                          vol_target: float = 0.15,
                          constant_trade_level: bool = False,
                          vol_af: float = 252
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    simulate weights and returns on tf strats
    """
    log_returns = ret.to_returns(prices=prices, is_log_returns=True)
    returns = ret.to_returns(prices=prices)

    ewm_vol = ewm.compute_ewm_vol(data=log_returns, span=vol_span, mean_adj_type=ewm.MeanAdjType.NONE,
                                  annualize=False)

    # vol target weights
    weights_100 = qu.to_finite_reciprocal(data=ewm_vol, fill_value=0.0, is_gt_zero=True)
    vt_return_100 = returns.multiply(weights_100.shift(1))
    # signal is unit var
    signals = ewm.compute_ewm(data=vt_return_100, span=tf_span, is_unit_vol_scaling=True)
    # normalized to target vol
    weights = signals.multiply(weights_100).multiply(vol_target/np.sqrt(vol_af))
    vt_returns = returns.multiply(weights.shift(1))
    vt_navs = ret.returns_to_nav(returns=vt_returns, constant_trade_level=constant_trade_level)
    return weights, vt_navs, signals


def simulate_trend_strats_range(prices: Union[pd.DataFrame, pd.Series],
                                vol_span: int = 33,
                                tf_spans: List[int] = (21, 63),
                                vol_target: float = 0.15,
                                constant_trade_level: bool = False,
                                vol_af: float = 252,
                                add_asset: bool = True
                                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tf_nav_weights, tf_navs, signals = [], [], []
    for tf_span in tf_spans:
        tf_nav_weights_, tf_navs_, signals_ = simulate_trend_strats(prices=prices, tf_span=tf_span, vol_span=vol_span,
                                                                    vol_target=vol_target, vol_af=vol_af,
                                                                    constant_trade_level=constant_trade_level)
        if isinstance(prices, pd.Series):
            name = f"{prices.name} tf_span={tf_span}"
            tf_nav_weights_.name, tf_navs_.name, signals_.name = name, name, name
        else:
            names = [f"{x} tf_span={tf_span}" for x in prices.columns]
            tf_nav_weights_.columns, tf_navs_.columns, signals_.columns = names, names, names
        tf_nav_weights.append(tf_nav_weights_)
        tf_navs.append(tf_navs_)
        signals.append(signals_)
    tf_nav_weights = pd.concat(tf_nav_weights, axis=1, sort=True)
    tf_navs = pd.concat(tf_navs, axis=1, sort=True)
    signals = pd.concat(signals, axis=1, sort=True)
    if add_asset:
        tf_navs = pd.concat([prices, tf_navs], axis=1, sort=True)
    return tf_nav_weights, tf_navs, signals
