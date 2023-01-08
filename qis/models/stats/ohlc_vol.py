
# buil in
import numpy as np
import pandas as pd
from enum import Enum
from typing import NamedTuple


class OhlcEstimatorType(Enum):
    PARKINSON = 'Parkinson'
    GARMAN_KLASS = 'Garman-Klass'
    ROGERS_SATCHELL = 'Rogers-Satchell'


class FreqAn(NamedTuple):
    freq: str
    an: float


class FreqAns(FreqAn, Enum):
    HOUR = FreqAn('1h', an=24*260)
    DAY = FreqAn('D', an=260)


def estimate_ohlc_vol(ohlc_data: pd.DataFrame,  # must contain ohlc columnes
                      ohlc_estimator_type: OhlcEstimatorType = OhlcEstimatorType.PARKINSON,
                      min_size: int = 2
                      ) -> float:

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

    if ohlc_estimator_type == OhlcEstimatorType.PARKINSON:
        multiplier = 1.0 / (4.0 * np.log(2.0))
        sample_var = multiplier * np.square(hl)

    elif ohlc_estimator_type == OhlcEstimatorType.GARMAN_KLASS:
        multiplier = 2.0 * np.log(2.0) - 1.0
        sample_var = 0.5 * np.square(hl) - multiplier * np.square(co)

    elif ohlc_estimator_type == OhlcEstimatorType.ROGERS_SATCHELL:
        sample_var = hc*ho + lc*lo

    else:
        raise TypeError(f"unknown ohlc_estimator_type={ohlc_estimator_type}")

    vol = np.sqrt(np.nansum(sample_var))
    return vol


def estimate_intra_ohlc_vol_data(ohlc_data: pd.DataFrame,
                                 ohlc_estimator_type: OhlcEstimatorType = OhlcEstimatorType.PARKINSON,
                                 is_exclude_weekends: bool = True,
                                 freq_an: FreqAn = FreqAns.DAY
                                 ) -> pd.Series:
    """
    sample vol at freq and annualize at an
    """
    vols = ohlc_data.groupby(pd.Grouper(freq=freq_an.freq)).apply(estimate_ohlc_vol, ohlc_estimator_type)
    vols = np.sqrt(freq_an.an) * vols  # annualize
    if is_exclude_weekends:
        vols = vols[vols.index.dayofweek < 5]
    return vols
