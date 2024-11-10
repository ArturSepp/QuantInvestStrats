"""
define transform for ra returns
"""
# packages
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import laplace
from typing import Union, Optional, Tuple
from enum import Enum
# qis
import qis.utils.dates as da
import qis.utils.np_ops as npo
import qis.models.linear.ewm as ewm


def compute_ra_returns(returns: Union[pd.Series, pd.DataFrame],
                       span: Union[float, np.ndarray] = None,
                       ewm_lambda: Union[float, np.ndarray] = 0.94,
                       vol_target: Optional[float] = None,  # if need to target vol
                       mean_adj_type: ewm.MeanAdjType = ewm.MeanAdjType.NONE,
                       init_value: Optional[Union[float, np.ndarray]] = None,
                       vol_floor_quantile: Optional[float] = None,  # to floor the volatility = 0.16
                       vol_floor_quantile_roll_period: int = 5 * 260,  # 5y for daily returns
                       warmup_period: Optional[int] = None,
                       is_log_returns_to_arithmetic: bool = False,  # typically log-return are passed to vol computations
                       weight_lag: Optional[int] = 1
                       ) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:

    if span is not None:
        ewm_lambda = 1.0-2.0/(span+1.0)

    if vol_target is None:  # non-dimensional 100% vol target
        annualize = False
        vol_target = 1.0
    else:
        annualize = True

    ewm_vol = ewm.compute_ewm_vol(data=returns,
                                  ewm_lambda=ewm_lambda,
                                  mean_adj_type=mean_adj_type,
                                  init_value=init_value,
                                  vol_floor_quantile=vol_floor_quantile,
                                  vol_floor_quantile_roll_period=vol_floor_quantile_roll_period,
                                  warmup_period=warmup_period,
                                  annualize=annualize)

    weights = npo.to_finite_reciprocal(data=ewm_vol, fill_value=np.nan, is_gt_zero=True)
    weights = weights.multiply(vol_target)
    if weight_lag is not None:
        weights = weights.shift(weight_lag)

    if is_log_returns_to_arithmetic:  # convert returns back to arithmetic = exp(r)-1.0
        returns = np.expm1(returns)

    ra_returns = returns.multiply(weights)

    # alignment in case
    if isinstance(ra_returns, pd.DataFrame):
        ra_returns = ra_returns[returns.columns]
        weights = weights[returns.columns]
        ewm_vol = ewm_vol[returns.columns]

    return ra_returns, weights, ewm_vol


def compute_ewm_long_short_filtered_ra_returns(returns: pd.DataFrame,
                                               vol_span: Optional[Union[int, np.ndarray]] = 31,
                                               long_span: Union[int, np.ndarray] = 63,
                                               short_span: Optional[Union[int, np.ndarray]] = 5,
                                               warmup_period: Optional[Union[int, np.ndarray]] = 21,
                                               weight_lag: Optional[int] = 1,
                                               mean_adj_type: ewm.MeanAdjType = ewm.MeanAdjType.NONE
                                               ) -> pd.DataFrame:
    if vol_span is not None:
        ra_returns, _, _ = compute_ra_returns(returns=returns,
                                              span=vol_span,
                                              vol_target=None,
                                              mean_adj_type=mean_adj_type,
                                              weight_lag=weight_lag)
    else:
        ra_returns = returns
    filter = ewm.compute_ewm_long_short_filter(data=ra_returns,
                                               long_span=long_span,
                                               short_span=short_span,
                                               warmup_period=warmup_period)
    return filter


class SignalMapType(Enum):
    NormalCDF = 1
    LaplaceCDF = 2
    ExpCDF = 3


def map_signal_to_weight(signals: pd.DataFrame, signal_map_type: SignalMapType = SignalMapType.NormalCDF,
                         loc: Union[float, pd.DataFrame] = 0.0,
                         scale: Union[float, np.ndarray] = 1.0,
                         tail_level: Union[float, np.ndarray] = 1.0,
                         slope_right: Union[float, np.ndarray] = 0.5,
                         slope_left: Union[float, np.ndarray] = 0.5,
                         tail_decay_right: Optional[Union[float, np.ndarray]] = None,
                         tail_decay_left: Optional[Union[float, np.ndarray]] = None
                         ) -> pd.DataFrame:
    x = signals.to_numpy()
    if isinstance(loc, pd.DataFrame):
        loc = loc.to_numpy()
    if isinstance(scale, np.ndarray) and scale.shape[0] != x.shape[1]:
        raise ValueError(f"{scale.shape[0]} != {x.shape[1]}")
    if isinstance(slope_right, np.ndarray) and slope_right.shape[0] != x.shape[1]:
        raise ValueError(f"{slope_right.shape[0]} != {x.shape[1]}")
    if isinstance(slope_left, np.ndarray) and slope_left.shape[0] != x.shape[1]:
        raise ValueError(f"{slope_left.shape[0]} != {x.shape[1]}")
    if isinstance(tail_level, np.ndarray) and tail_level.shape[0] != x.shape[1]:
        raise ValueError(f"{tail_level.shape[0]} != {x.shape[1]}")

    if signal_map_type == SignalMapType.NormalCDF:
        weight = 2.0*norm.cdf(x=x, loc=loc, scale=scale) - 1.0

    elif signal_map_type == SignalMapType.LaplaceCDF:
        weight = 2.0*laplace.cdf(x=x, loc=loc, scale=scale) - 1.0

    elif signal_map_type == SignalMapType.ExpCDF:
        if np.any(np.less_equal(tail_level, slope_right)) or np.any(np.less_equal(tail_level, slope_left)):
            raise ValueError(f"must be tail>slope_positive and tail > slope_negative")
        scale_negative = 1.5625 * scale / np.log(tail_level / (tail_level - slope_left))
        scale_positive = 1.5625 * scale / np.log(tail_level / (tail_level - slope_right))
        s_negative = - tail_level * (1.0 - np.exp(-np.square(x - loc) / scale_negative))
        s_positive = tail_level * (1.0 - np.exp(-np.square(x - loc) / scale_positive))
        weight = np.where(np.less(x, loc, where=np.isfinite(x)), s_negative, s_positive)

        if tail_decay_right is not None and tail_decay_left is not None:  # take min(loc,0.0) and max(loc, 0.0)
            if isinstance(tail_decay_right, np.ndarray) and tail_decay_right.shape[0] != x.shape[1]:
                raise ValueError(f"{tail_decay_right.shape[0]} != {x.shape[1]}")
            if isinstance(tail_decay_left, np.ndarray) and tail_decay_left.shape[0] != x.shape[1]:
                raise ValueError(f"{tail_decay_left.shape[0]} != {x.shape[1]}")

            x_left_tail = x + tail_level - np.where(np.less(loc, 0.0), loc, 0.0)
            f_left_tail = np.where(np.less(x_left_tail, 0.0), np.exp(x_left_tail / tail_decay_left), 1.0)

            x_right_tail = x - tail_level - np.where(np.greater(loc, 0.0), loc, 0.0)
            f_right_tail = np.where(np.greater(x_right_tail, 0.0), np.exp(-x_right_tail / tail_decay_right), 1.0)

            tails = np.where(np.greater(x_right_tail, 0.0), f_right_tail, f_left_tail)
            weight = weight * tails

    else:
        raise NotImplementedError(f"signal_map_type={signal_map_type}")
    weight = pd.DataFrame(weight, index=signals.index, columns=signals.columns)
    return weight


def compute_rolling_ra_returns(returns: pd.DataFrame,
                               span: int = 1,
                               ewm_lambda_eod: float = 0.94,
                               vol_target: Optional[float] = None,
                               weight_shift: Optional[int] = 1,
                               is_log_returns_to_arithmetic: bool = True  # typically log-return are passed to vol
                               ) -> pd.DataFrame:
    """
    span = 1: daily ra returns
    otherwise compute sum of returns and then their vols
    interpretation: voltargeting for returns over span
    ewm_lambda is vol over the span too
    """
    if span > 1:
        rolling_returns = returns.rolling(span).sum()
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    else:
        ewm_lambda = ewm_lambda_eod
        rolling_returns = returns

    ra_returns, weights, _ = compute_ra_returns(returns=rolling_returns,
                                                ewm_lambda=ewm_lambda,
                                                vol_target=vol_target,
                                                weight_lag=weight_shift,
                                                is_log_returns_to_arithmetic=is_log_returns_to_arithmetic)
    return ra_returns


def compute_sum_rolling_ra_returns(returns: pd.DataFrame,
                                   span: int = 1,
                                   ewm_lambda: float = 0.94,
                                   vol_target: Optional[float] = None,
                                   weight_shift: Optional[int] = 1,
                                   is_log_returns_to_arithmetic: bool = True,  # typically log-return are passed to vol
                                   is_norm: bool = True
                                   ) -> pd.DataFrame:
    """
    span = 1: daily ra returns
    otherwise compute sum of daily ra-returns
    interpretation: pnl of daily voltargeting returns over span
    """
    ra_returns, weights, _ = compute_ra_returns(returns=returns,
                                                ewm_lambda=ewm_lambda,
                                                vol_target=vol_target,
                                                weight_lag=weight_shift,
                                                is_log_returns_to_arithmetic=is_log_returns_to_arithmetic)

    if span > 1:
        sum_rolling_ra_returns = ra_returns.rolling(span).sum()

        if is_norm:
            sum_rolling_ra_returns = sum_rolling_ra_returns.divide(np.sqrt(span))
    else:
        sum_rolling_ra_returns = ra_returns

    return sum_rolling_ra_returns


def compute_sum_freq_ra_returns(returns: Union[pd.Series, pd.DataFrame],
                                freq: str = 'B',
                                span: int = None,
                                ewm_lambda: float = 0.94,
                                vol_target: Optional[float] = None,
                                weight_shift: Optional[int] = 1,
                                is_log_returns_to_arithmetic: bool = True,  # typically log-return are passed to vol
                                is_norm: bool = True,
                                warmup_period: Optional[int] = None
                                ) -> Union[pd.Series, pd.DataFrame]:
    """
    span = 1: daily ra returns
    otherwise compute freq sum of daily ra-returns
    interpretation: pnl of daily voltargeting returns over non-overlap freq
    """
    ra_returns, _, _ = compute_ra_returns(returns=returns,
                                          span=span,
                                          ewm_lambda=ewm_lambda,
                                          vol_target=vol_target,
                                          weight_lag=weight_shift,
                                          is_log_returns_to_arithmetic=is_log_returns_to_arithmetic,
                                          warmup_period=warmup_period)

    if freq not in ['B', 'D']:
        sum_rolling_ra_returns = ra_returns.resample(freq).sum()

        if is_norm:
            n_daily, _ = da.get_period_days(freq=freq, is_calendar=False)
            sum_rolling_ra_returns = sum_rolling_ra_returns.divide(np.sqrt(n_daily))
    else:
        sum_rolling_ra_returns = ra_returns

    return sum_rolling_ra_returns


def compute_ewm_ra_returns_momentum(returns: Union[pd.Series, pd.DataFrame],
                                    momentum_span: int = 63,
                                    momentum_lambda: Optional[Union[float, np.ndarray]] = None,
                                    vol_span: Union[float, np.ndarray] = 31,
                                    vol_lambda: Optional[Union[float, np.ndarray]] = None,
                                    weight_shift: Optional[int] = 1
                                    ) -> Union[pd.Series, pd.DataFrame]:
    """
    span = 1: daily ra returns
    """
    if momentum_lambda is None:
        momentum_lambda = 1.0 - 2.0 / (momentum_span + 1.0)
    if vol_lambda is None:
        vol_lambda = 1.0 - 2.0 / (vol_span + 1.0)

    ra_returns, _, _ = compute_ra_returns(returns=returns,
                                          ewm_lambda=vol_lambda,
                                          vol_target=None,
                                          weight_lag=weight_shift)

    ewm_signal = ewm.ewm_recursion(a=ra_returns.to_numpy(),
                                   ewm_lambda=momentum_lambda,
                                   init_value=0.0 if isinstance(returns, pd.Series) else np.zeros(len(returns.columns)),
                                   is_unit_vol_scaling=True)

    if isinstance(returns, pd.DataFrame):
        ewm_ra_returns_momentum = pd.DataFrame(data=ewm_signal, index=returns.index, columns=returns.columns)
    else:
        ewm_ra_returns_momentum = pd.Series(data=ewm_signal, index=returns.index, name=returns.name)

    return ewm_ra_returns_momentum


def get_paired_rareturns_signals(returns: Union[pd.Series, pd.DataFrame],
                                 signal: Union[pd.Series, pd.DataFrame],
                                 freq: str = 'BQ',
                                 span: int = 63,
                                 is_nonoverlapping: bool = True,
                                 ra_returns_ewm_vol_lambda: float = 0.94,
                                 is_mean_adjust_returns: bool = False
                                 ) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:

    if is_nonoverlapping:
        sum_freq_ra_returns = compute_sum_freq_ra_returns(returns=returns,
                                                              freq=freq,
                                                              ewm_lambda=ra_returns_ewm_vol_lambda,
                                                              is_norm=True)
        ra_return = sum_freq_ra_returns
        indicator = signal.resample(freq).last().shift(1)
    else:
        sum_rolling_ra_returns = compute_sum_rolling_ra_returns(returns=returns,
                                                                    span=span,
                                                                    ewm_lambda=ra_returns_ewm_vol_lambda,
                                                                    is_norm=True)
        ra_return = sum_rolling_ra_returns
        indicator = signal.shift(1)

    # into two column dataframe
    if is_mean_adjust_returns:
        ra_returns_mean = ra_return.expanding(min_periods=1, axis=0).mean()  # apply pandas expanding
        ra_return = ra_return.subtract(ra_returns_mean)

    return ra_return, indicator


class ReturnsTransform(Enum):
    ROLLING_RA_RETURNS = 1
    EWMA_RETURNS_MOMENTUM = 2


def compute_returns_transform(returns: pd.DataFrame,
                              returns_transform: ReturnsTransform = ReturnsTransform.ROLLING_RA_RETURNS,
                              momentum_span: int = 31,
                              vol_span: int = 31,
                              rolling_ra_returns_span: int = 31
                              ) -> pd.DataFrame:

    if returns_transform == ReturnsTransform.ROLLING_RA_RETURNS:
        returns_transform = compute_rolling_ra_returns(returns=returns,
                                                       span=rolling_ra_returns_span,
                                                       weight_shift=1)
    elif returns_transform == ReturnsTransform.EWMA_RETURNS_MOMENTUM:
        returns_transform = compute_ewm_ra_returns_momentum(returns=returns,
                                                             momentum_span=momentum_span,
                                                             vol_span=vol_span,
                                                             weight_shift=1)
    else:
        raise TypeError(f"returns_transform {returns_transform} of {type(returns_transform)} not implemented")
    return returns_transform


class UnitTests(Enum):
    RA_RETURNS = 1
    TRANSFORM = 2


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()
    returns = prices.pct_change()

    if unit_test == UnitTests.RA_RETURNS:
        df = compute_ra_returns(returns=returns)
        print(df)

    elif unit_test == UnitTests.TRANSFORM:
        df = compute_returns_transform(returns=returns)
        print(df)


if __name__ == '__main__':

    unit_test = UnitTests.RA_RETURNS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
