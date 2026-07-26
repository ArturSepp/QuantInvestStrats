
# packages
import numpy as np
import pandas as pd
from enum import Enum
from typing import Optional
from qis.utils.annualisation import infer_annualisation_factor_from_df


class OhlcEstimatorType(Enum):
    PARKINSON = 'Parkinson'
    GARMAN_KLASS = 'Garman-Klass'
    ROGERS_SATCHELL = 'Rogers-Satchell'
    CLOSE_TO_CLOSE = 'Close-to-Close'


def estimate_ohlc_var(ohlc_data: pd.DataFrame,  # must contain ohlc columnes
                      ohlc_estimator_type: OhlcEstimatorType = OhlcEstimatorType.PARKINSON,
                      min_size: int = 2
                      ) -> pd.Series:

    if ohlc_data.empty or len(ohlc_data.index) < min_size:
        return np.nan

    log_ohlc = np.log(ohlc_data[['open', 'high', 'low', 'close']].to_numpy())
    open, high, low, close = log_ohlc[:, 0], log_ohlc[:, 1], log_ohlc[:, 2], log_ohlc[:, 3]

    hc = high - close
    ho = high - open
    lc = low - close
    lo = low - open
    hl = high - low
    co = close - open

    if ohlc_estimator_type == OhlcEstimatorType.CLOSE_TO_CLOSE:
        sample_var = np.square(close[1:]-close[:-1])
        sample_var = np.concatenate((np.array([np.nan]), sample_var), axis=0)

    elif ohlc_estimator_type == OhlcEstimatorType.PARKINSON:
        multiplier = 1.0 / (4.0 * np.log(2.0))
        sample_var = multiplier * np.square(hl)

    elif ohlc_estimator_type == OhlcEstimatorType.GARMAN_KLASS:
        multiplier = 2.0 * np.log(2.0) - 1.0
        sample_var = 0.5 * np.square(hl) - multiplier * np.square(co)

    elif ohlc_estimator_type == OhlcEstimatorType.ROGERS_SATCHELL:
        sample_var = hc*ho + lc*lo

    else:
        raise TypeError(f"unknown ohlc_estimator_type={ohlc_estimator_type}")

    sample_var = pd.Series(sample_var, index=ohlc_data.index)
    return sample_var


def estimate_hf_ohlc_vol(ohlc_data: pd.DataFrame,
                         ohlc_estimator_type: OhlcEstimatorType = OhlcEstimatorType.PARKINSON,
                         annualization_factor: float = None,  # annualisation factor highly recomended
                         is_exclude_weekends: bool = False,  # for crypto
                         agg_freq: Optional[str] = 'B'
                         ) -> pd.Series:
    """
    annualised volatility from open-high-low-close bars using a range estimator.

    A range estimator uses the high and low as well as the close, so it extracts several times more
    information per bar than a close-to-close estimate at the same sample size. Bars are aggregated
    to ``agg_freq`` and the per-bar variance is averaged before annualising.

    Args:
        ohlc_data: bars with open, high, low and close columns
        ohlc_estimator_type: which range estimator to apply; see :class:`OhlcEstimatorType`
        annualization_factor: periods per year. None infers it from the aggregated index, which is
            a guess worth overriding - it rescales every reported number
        is_exclude_weekends: drop Saturdays and Sundays from the output. Off by default because a
            24/7 market trades through them
        agg_freq: frequency the per-bar variances are averaged to. None leaves them per bar

    Returns:
        annualised volatility, indexed at ``agg_freq``
    """
    sample_var = estimate_ohlc_var(ohlc_data=ohlc_data, ohlc_estimator_type=ohlc_estimator_type)
    if agg_freq is not None:
        sample_var = sample_var.resample(agg_freq).mean()

    if annualization_factor is None:
        annualization_factor = infer_annualisation_factor_from_df(data=sample_var)

    vols = np.sqrt(annualization_factor*sample_var)
    if is_exclude_weekends:
        vols = vols[vols.index.dayofweek < 5]
    return vols
