"""
implement winsorizing of time series data using ewm
1. compute mean_t and vol_t
2. select x% of outliers defined by normalized score (x_t-mean_t) / vol_t
3. replace or trim outliers as specified
"""
# packages
import numpy as np
import pandas as pd
from enum import Enum

# qis
import qis.utils.dates as da
import qis.models.linear.ewm as ewm


class ConvolutionType(Enum):
    AUTO_CORR = 1
    SIGNAL_CORR = 2
    SIGNAL_BETA = 3


class SignalAggType(Enum):
    LAST_VALUE = 1
    MEAN = 2


def ewm_xy_convolution(returns: pd.DataFrame,
                       freq: str,
                       signals: pd.DataFrame = None,
                       convolution_type: ConvolutionType = ConvolutionType.AUTO_CORR,
                       signal_agg_type: SignalAggType = SignalAggType.LAST_VALUE,
                       is_ra_return: bool = False,
                       estimates_smoothing_lambda: float = None,
                       mean_adj_type: ewm.MeanAdjType = ewm.MeanAdjType.NONE
                       ) -> pd.DataFrame:
    """
    ewm convolution, typical case:
    y is return, x is signal
    span deines the
    assumed frequency is daily
    """

    signal_span, _ = da.get_period_days(freq=freq, is_calendar=True)

    if not np.isclose(signal_span, 1):
        ewm_lambda = 1.0 - 2.0 / (signal_span + 1.0)
    else:  # take 1.5 for frequency of business day
        ewm_lambda = 0.5 / 2.5

    if is_ra_return:
        ewm_vol = ewm.compute_ewm_vol(data=returns,
                                         ewm_lambda=ewm_lambda,
                                         annualize=False)
        returns = np.divide(returns, ewm_vol, where=np.isclose(ewm_vol, 0.0)==False)

    # rolling returns by the span
    if not np.isclose(signal_span, 1):
        # norm_factor = np.sqrt(signal_span)
        rolling_returns = returns.rolling(signal_span).sum()
    else:
        rolling_returns = returns

    if signals is not None:
        if signal_agg_type == SignalAggType.LAST_VALUE:
            agg_signal = signals
        elif signal_agg_type == SignalAggType.MEAN:
            agg_signal = signals.rolling(signal_span).mean()
        else:
            raise TypeError(f"unknown {signal_agg_type}")

        agg_signal = agg_signal.reindex(index=rolling_returns.index, method='ffill')
        agg_signal = agg_signal.shift(signal_span)  # shift backrard by the span
    else:
        agg_signal = None

    if convolution_type == ConvolutionType.AUTO_CORR:
        x_data = rolling_returns.shift(signal_span) # shift backward by the span
        y_data = rolling_returns
        cross_xy_type = ewm.CrossXyType.CORR

    elif convolution_type == ConvolutionType.SIGNAL_CORR:
        x_data = agg_signal
        y_data = rolling_returns
        cross_xy_type = ewm.CrossXyType.CORR

    elif convolution_type == ConvolutionType.SIGNAL_BETA:
        x_data = agg_signal
        y_data = rolling_returns
        cross_xy_type = ewm.CrossXyType.BETA

    else:
        raise ValueError(f"{convolution_type} is not implemented")

    # compute ewm cross
    corr = ewm.compute_ewm_cross_xy(x_data=x_data,
                                      y_data=y_data,
                                      ewm_lambda=ewm_lambda,
                                      cross_xy_type=cross_xy_type,
                                      mean_adj_type=mean_adj_type)

    if estimates_smoothing_lambda is not None:
        corr = ewm.compute_ewm(data=corr, ewm_lambda=estimates_smoothing_lambda)

    return corr
