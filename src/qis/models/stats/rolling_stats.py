"""
performance statistics computed on a rolling window of a price series.

``compute_rolling_perf_stat`` is the dispatcher: given a ``RollingPerfStat`` it returns the
statistic and the title describing the window, which is what the plotting layer draws. Behind it
sit ``compute_rolling_returns`` and ``compute_rolling_pa_returns``, ``compute_rolling_vols`` and
``compute_ewma_vols``, ``compute_rolling_sharpes`` and ``compute_rolling_skew``.

``roll_periods`` counts rows after the series is resampled to ``roll_freq``, so 260 with
``roll_freq='B'`` is one year and a five-year monthly window is ``roll_freq='ME'`` with 60; the
exceptions are ``RollingPerfStat.PA_RETURNS``, which never resamples, and ``EWMA_VOL``, where
``roll_periods`` is the EWM ``span``. Volatility and Sharpe are taken on log returns and
annualised by the factor inferred from the index, the skew on log returns without any
annualisation, while the two return statistics work on the prices directly. ``compute_sharpe``
subtracts no financing rate; the excess-return conventions live in ``qis.perfstats``.
"""
# packages
import numpy as np
import pandas as pd
from scipy.stats import skew
from typing import Union, Optional, Tuple
from enum import Enum

# qis
import qis.perfstats.returns as ret
from qis.utils.annualisation import (
    CALENDAR_DAYS_PER_YEAR_SHARPE,
    infer_annualisation_factor_from_df,
)
import qis.models.linear.ewm as ewm


class RollingPerfStat(Enum):
    """
    enumerated tuple for name and format
    """
    TOTAL_RETURNS = ('Total returns', '{:.2%}')
    PA_RETURNS = ('Pa returns', '{:.2%}')
    VOL = ('Volatility', '{:.2%}')
    SHARPE = ('Sharpe ratio', '{:.2f}')
    SKEW = ('Skeweness', '{:.2f}')
    EWMA_VOL = ('EWMA vol', '{:.2%}')


def compute_rolling_perf_stat(prices: Union[pd.DataFrame, pd.Series],
                              rolling_perf_stat: RollingPerfStat = RollingPerfStat.TOTAL_RETURNS,
                              roll_freq: Optional[str] = 'B',
                              roll_periods: int = 260
                              ) -> Tuple[Union[pd.DataFrame, pd.Series], str]:
    """
    compute rolling performance
    default stats is 1y rolling for prices with roll_freq = 'B'
    for monthly returns anf 5y rolling use roll_freq='ME' and roll_periods=5*12
    """
    if roll_freq is None:
        roll_freq1 = pd.infer_freq(index=prices.index)
    else:
        roll_freq1 = roll_freq

    if rolling_perf_stat == RollingPerfStat.TOTAL_RETURNS:
        perf_stat = compute_rolling_returns(prices=prices, roll_freq=roll_freq, roll_periods=roll_periods)
        title = f"Rolling total returns with {roll_freq1}-freq and roll_period={roll_periods}"
    elif rolling_perf_stat == RollingPerfStat.PA_RETURNS:
        perf_stat = compute_rolling_pa_returns(prices=prices, roll_periods=roll_periods)
        title = f"P.a. returns with roll_period={roll_periods}"
    elif rolling_perf_stat == RollingPerfStat.VOL:
        perf_stat = compute_rolling_vols(prices=prices, roll_freq=roll_freq, roll_periods=roll_periods)
        title = f"Rolling Vol with {roll_freq1}-freq and roll_period={roll_periods}"
    elif rolling_perf_stat == RollingPerfStat.SHARPE:
        perf_stat = compute_rolling_sharpes(prices=prices, roll_freq=roll_freq, roll_periods=roll_periods)
        title = f"Rolling Sharpe with {roll_freq1}-freq and roll_period={roll_periods}"
    elif rolling_perf_stat == RollingPerfStat.SKEW:
        perf_stat = compute_rolling_skew(prices=prices, roll_freq=roll_freq, roll_periods=roll_periods)
        title = f"Rolling skew with {roll_freq1}-freq and roll_period={roll_periods}"
    elif rolling_perf_stat == RollingPerfStat.EWMA_VOL:
        perf_stat = compute_ewma_vols(prices=prices, roll_freq=roll_freq, roll_periods=roll_periods)
        title = f"Rolling EWMA vol with {roll_freq1}-freq and span={roll_periods}"
    else:
        raise NotImplementedError(f"rolling_perf_stats")
    return perf_stat, title


def compute_rolling_returns(prices: Union[pd.DataFrame, pd.Series],
                            roll_freq: Optional[str] = None,
                            roll_periods: int = 260
                            ) -> Union[pd.DataFrame, pd.Series]:
    """
    compute rolling returns
    """
    if roll_freq is not None:
        prices = prices.asfreq(roll_freq, method='ffill')
    returns = prices.divide(prices.shift(periods=roll_periods)) - 1.0
    return returns


def compute_rolling_pa_returns(prices: Union[pd.DataFrame, pd.Series],
                               roll_periods: int = 260
                               ) -> Union[pd.DataFrame, pd.Series]:
    """Compute per-annum returns on trailing windows of native price rows.

    Args:
        prices: Price Series or DataFrame. Complete positive float64 histories with ordered,
            unique dates share endpoint/calendar preparation; other inputs retain the scalar path.
        roll_periods: Number of price observations per window, without implicit resampling.

    Returns:
        Rolling returns with the native input labels and initial unavailable windows. As in
        ``compute_pa_return``, spans of at most one year are not annualised; longer spans use
        geometric compounding on elapsed calendar days divided by 365.25.
    """
    # Unsigned windows stay native: 1 - window can wrap before calendar slicing.
    ordinary = (
        type(prices) in (pd.DataFrame, pd.Series)
        and isinstance(roll_periods, (int, np.integer))
        and not isinstance(roll_periods, (bool, np.bool_, np.unsignedinteger))
        and 2 <= roll_periods <= len(prices)
        and type(prices.index) is pd.DatetimeIndex
        and not prices.index.hasnans
        and prices.index.is_unique
        and prices.index.is_monotonic_increasing
        and prices.size > 0
    )
    if ordinary:
        dtypes = prices.dtypes if isinstance(prices, pd.DataFrame) else [prices.dtype]
        if all(dtype == np.dtype('float64') for dtype in dtypes):
            values = prices.to_numpy()
            if np.isfinite(values).all() and (values > 0.0).all():
                panel = values[:, None] if isinstance(prices, pd.Series) else values
                try:
                    # Every complete column uses the same window dates. Keep native whole-day
                    # truncation and its one-day floor, including subday and DST-spanning input.
                    days = (prices.index[roll_periods - 1:] - prices.index[:1 - roll_periods]).days
                    years = (
                        np.maximum(days, 1).to_numpy(dtype=float) / CALENDAR_DAYS_PER_YEAR_SHARPE
                    )
                    with np.errstate(all='raise'):
                        # Do not simplify to an endpoint ratio: subtract/add rounding is part
                        # of the scalar helper's result, including ratios rounded down to zero.
                        total = panel[roll_periods - 1:] / panel[:1 - roll_periods] - 1.0
                        ratio = total + 1.0
                        ratio = np.where(ratio > 0.0, ratio, np.nan)
                        result = ratio - 1.0
                        annualised = years > 1.0
                        result[annualised] = np.power(
                            ratio[annualised], 1.0 / years[annualised, None]
                        ) - 1.0
                except (FloatingPointError, OverflowError, ValueError):
                    # Let the untouched scalar path emit its own diagnostics under the
                    # caller's NumPy policy; a batch must not introduce or suppress warnings.
                    pass
                else:
                    out = np.full(panel.shape, np.nan)
                    out[roll_periods - 1:] = result
                    if isinstance(prices, pd.Series):
                        return pd.Series(out[:, 0], index=prices.index, name=prices.name)
                    # Rolling isolates column-name metadata, but does not deep-copy labels
                    # or the row index. Preserve that ownership boundary with a shallow copy.
                    columns = prices.columns.copy(deep=False)
                    return pd.DataFrame(out, index=prices.index, columns=columns)
    pa_returns = prices.rolling(roll_periods).apply(lambda x: ret.compute_pa_return(x))
    return pa_returns


def compute_rolling_vols(prices: Union[pd.Series, pd.DataFrame],
                        roll_freq: Optional[str] = None,
                        roll_periods: int = 260
                        ) -> Union[pd.Series, pd.DataFrame]:
    log_returns = ret.to_returns(prices=prices, freq=roll_freq, is_log_returns=True, drop_first=False)
    saf = np.sqrt(infer_annualisation_factor_from_df(data=log_returns))
    # Raw windows avoid constructing a labeled Series for every unchanged NumPy reduction.
    vols = saf * log_returns.rolling(roll_periods).apply(
        lambda x: np.nanstd(x, ddof=1), raw=True
    )
    return vols


def compute_ewma_vols(prices: Union[pd.Series, pd.DataFrame],
                      roll_freq: Optional[str] = None,
                      roll_periods: int = 260
                      ) -> Union[pd.Series, pd.DataFrame]:
    log_returns = ret.to_returns(prices=prices, freq=roll_freq, is_log_returns=True, drop_first=False)
    vols = ewm.compute_ewm_vol(data=log_returns, span=roll_periods, annualize=True)
    return vols


def compute_rolling_sharpes(prices: Union[pd.Series, pd.DataFrame],
                            roll_freq: Optional[str] = None,
                            roll_periods: int = 260
                            ) -> Union[pd.Series, pd.DataFrame]:
    log_returns = ret.to_returns(prices=prices, freq=roll_freq, is_log_returns=True, drop_first=False)
    saf = np.sqrt(infer_annualisation_factor_from_df(data=log_returns))
    sharpes = log_returns.rolling(roll_periods).apply(
        lambda x: _compute_sharpe_from_array(x, saf=saf), raw=True
    )
    return sharpes


def compute_sharpe(log_returns: Union[pd.Series, pd.DataFrame], saf: float = None) -> np.ndarray:
    if saf is None:
        saf = np.sqrt(infer_annualisation_factor_from_df(data=log_returns))
    return _compute_sharpe_from_array(log_returns=log_returns.to_numpy(), saf=saf)


def _compute_sharpe_from_array(
    log_returns: np.ndarray, saf: float
) -> Union[np.float64, float]:
    """Compute the established rolling Sharpe formula without rebuilding pandas objects.

    Args:
        log_returns: One NumPy window of periodic log returns.
        saf: Square root of the inferred annualisation factor.

    Returns:
        Annualised rolling Sharpe, or ``nan`` when volatility is not positive.
    """
    mean = np.expm1(np.nanmean(log_returns))
    vol = np.nanstd(log_returns, ddof=1)
    if np.greater(vol, 0.0):
        sharpe = saf * mean / vol
    else:
        sharpe = np.nan
    return sharpe


def compute_rolling_skew(prices: Union[pd.Series, pd.DataFrame],
                         roll_freq: Optional[str] = None,
                         roll_periods: int = 120,
                         ) -> Union[pd.Series, pd.DataFrame]:
    log_returns = ret.to_returns(prices=prices, freq=roll_freq, is_log_returns=True, drop_first=False)
    skw = log_returns.rolling(roll_periods).apply(lambda x: compute_skew(x))
    return skw


def compute_skew(log_returns: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
    skw = skew(log_returns.to_numpy(), axis=0, nan_policy='omit')
    return skw
