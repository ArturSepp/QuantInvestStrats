"""
define transform for ra returns
"""

# packages
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from enum import Enum

# qis
import qis.utils.dates as da
import qis.utils.np_ops as npo
import qis.models.linear.ewm as ewm

PD_UNION = Union[pd.Series, pd.DataFrame]


def compute_ra_returns(returns: Union[pd.Series, pd.DataFrame],
                       ewm_lambda: Union[float, np.ndarray] = 0.94,
                       vol_target: Optional[float] = 0.12,
                       span: Optional[int] = None,
                       weight_shift: Optional[int] = 1,
                       is_log_returns_to_arithmetic: bool = True  # typically log-return are passed to vol computations
                       ) -> Tuple[PD_UNION, PD_UNION, PD_UNION]:

    if span is not None:
        ewm_lambda = 1.0-2.0/(span+1.0)

    if vol_target is None:  # non-dimensional 100% vol target
        annualize = False
        vol_target = 1.0
    else:
        annualize = True

    ewm_vol = ewm.compute_ewm_vol(data=returns,
                                  ewm_lambda=ewm_lambda,
                                  mean_adj_type=ewm.MeanAdjType.NONE,
                                  annualize=annualize)

    weights = npo.to_finite_reciprocal(data=ewm_vol, fill_value=0.0, is_gt_zero=True)
    weights = weights.multiply(vol_target)
    if weight_shift is not None:
        weights = weights.shift(weight_shift)

    if is_log_returns_to_arithmetic:  # convert returns back to arithmetic = exp(r)-1.0
        returns = np.expm1(returns)

    ra_returns = returns.multiply(weights)

    # alignment in case
    if isinstance(ra_returns, pd.DataFrame):
        ra_returns = ra_returns[returns.columns]
        weights = weights[returns.columns]

    return ra_returns, weights, ewm_vol


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
                                                weight_shift=weight_shift,
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
                                                weight_shift=weight_shift,
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
                                ewm_lambda: float = 0.94,
                                vol_target: Optional[float] = None,
                                weight_shift: Optional[int] = 1,
                                is_log_returns_to_arithmetic: bool = True,  # typically log-return are passed to vol
                                is_norm: bool = True
                                ) -> Union[pd.Series, pd.DataFrame]:
    """
    span = 1: daily ra returns
    otherwise compute freq sum of daily ra-returns
    interpretation: pnl of daily voltargeting returns over non-overlap freq
    """
    ra_returns, weights, _ = compute_ra_returns(returns=returns,
                                                ewm_lambda=ewm_lambda,
                                                vol_target=vol_target,
                                                weight_shift=weight_shift,
                                                is_log_returns_to_arithmetic=is_log_returns_to_arithmetic)

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
                                    vol_span: int = 31,
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
                                          weight_shift=weight_shift)

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

