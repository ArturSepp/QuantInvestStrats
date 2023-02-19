
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
                               vol_af: float = 260
                               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    simulate weights and returns on vol target
    """
    log_returns = ret.to_returns(prices=prices, is_log_returns=True)
    returns = ret.to_returns(prices=prices, is_log_returns=False)
    ewm_vol = ewm.compute_ewm_vol(data=log_returns,
                                  span=vol_span,
                                  mean_adj_type=ewm.MeanAdjType.NONE,
                                  af=vol_af)
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
                                     vol_af: float = 260,
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
    vt_nav_weights, vt_navs = pd.concat(vt_nav_weights, axis=1), pd.concat(vt_navs, axis=1)
    if add_asset:
        vt_navs = pd.concat([prices, vt_navs], axis=1)
    return vt_nav_weights, vt_navs


def simulate_trend_starts(prices: Union[pd.DataFrame, pd.Series],
                          vol_span: int = 31,
                          tf_span: int = 63,
                          vol_target: float = 0.15,
                          constant_trade_level: bool = False,
                          vol_af: float = 260
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


def simulate_trend_starts_range(prices: Union[pd.DataFrame, pd.Series],
                                vol_span: int = 31,
                                tf_spans: List[int] = (21, 63),
                                vol_target: float = 0.15,
                                constant_trade_level: bool = False,
                                vol_af: float = 260,
                                add_asset: bool = True
                                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tf_nav_weights, tf_navs, signals = [], [], []
    for tf_span in tf_spans:
        tf_nav_weights_, tf_navs_, signals_ = simulate_trend_starts(prices=prices, tf_span=tf_span, vol_span=vol_span,
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
    tf_nav_weights, tf_navs, signals = pd.concat(tf_nav_weights, axis=1), pd.concat(tf_navs, axis=1), pd.concat(signals, axis=1)
    if add_asset:
        tf_navs = pd.concat([prices, tf_navs], axis=1)
    return tf_nav_weights, tf_navs, signals
