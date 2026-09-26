"""
EWM convolution of returns with their own lag or with a signal, at a horizon set by ``freq``.

``ewm_xy_convolution`` is the single entry point. ``ConvolutionType`` selects what x is: the
rolling return lagged by the horizon (``AUTO_CORR``), or the signal (``SIGNAL_CORR``,
``SIGNAL_BETA``), y always being the rolling return. ``SignalAggType`` says whether a signal
enters at its last value or as its mean over the horizon, and ``estimates_smoothing_lambda``
smooths the resulting series of estimates.

The horizon is ``get_annualization_factor`` of ``freq`` - 252 for ``'B'``, 12 for ``'ME'`` - taken
as an integer number of rows, and that one number is used three ways: rows summed into the rolling
return, lag applied to x, and span behind the EWM decay. It counts rows of the input, which is
assumed daily, not periods of ``freq``; where the factor is one (``'YE'``) the decay falls back to
0.2 and the returns are left unsummed. ``is_ra_returns`` divides returns by the EWM volatility
lagged one period first.

The cross moment and both second moments are EWM recursions seeded at zero from their first
finite row, so an estimate dated t uses rows up to t only.
"""
# packages
import numpy as np
import pandas as pd
from enum import Enum

# qis
import qis.models.linear.ewm as ewm
from qis.utils.annualisation import get_annualization_factor


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
                       is_ra_returns: bool = False,
                       estimates_smoothing_lambda: float = None,
                       mean_adj_type: ewm.MeanAdjType = ewm.MeanAdjType.NONE,
                       var_init_type: ewm.InitType = ewm.InitType.ZERO
                       ) -> pd.DataFrame:
    """
    EWM correlation or beta of rolling h-row returns with their own lag or with a signal.

    The horizon h is ``get_annualization_factor(freq)`` as an integer number of rows of
    ``returns``, which are assumed daily: 252 for ``'B'``, 12 for ``'ME'``, 4 for ``'QE'``. y is
    the rolling sum of the last h returns; x is y lagged by h rows (``AUTO_CORR``) or the signal
    lagged by h rows (``SIGNAL_CORR``, ``SIGNAL_BETA``). The EWM decay is
    ``1 - 2 / (h + 1)``. When h is one the returns are not summed and the decay is 0.2.

    Args:
        returns: returns, rows are dates and columns are assets
        freq: frequency whose annualisation factor sets the horizon h
        signals: signals aligned to ``returns``; required for the signal convolution types
        convolution_type: what x is, and whether a correlation or a beta is returned
        signal_agg_type: a signal enters at its last value or as its mean over h rows
        is_ra_returns: divide returns by their EWM volatility (lambda 0.94) lagged one row first
        estimates_smoothing_lambda: if given, smooth the output with a further EWM of this decay
        mean_adj_type: mean removed from x and y before the moments; none by default
        var_init_type: seed of the EWM second moments of x and y. The default
            ``InitType.ZERO`` is point in time, like the zero seed of the cross moment;
            ``InitType.MEAN`` seeds them with full-sample means, which looks ahead

    Returns:
        the EWM estimates, indexed like ``returns``; NaN until x and y are both available

    Raises:
        ValueError: if ``freq`` does not give a whole number of rows of at least one, or
            ``convolution_type`` is not implemented
    """
    signal_span = get_annualization_factor(freq=freq)
    horizon = int(np.round(signal_span))
    if horizon < 1 or not np.isclose(signal_span, horizon):
        raise ValueError(f"freq={freq} gives a horizon of {signal_span} rows; "
                         f"a whole number of at least one row is required")

    if horizon > 1:
        ewm_lambda = 1.0 - 2.0 / (horizon + 1.0)
    else:  # take span 1.5 for a one-row horizon
        ewm_lambda = 0.5 / 2.5

    if is_ra_returns:
        ewm_vol = ewm.compute_ewm_vol(data=returns,
                                      ewm_lambda=0.94,
                                      annualize=False)
        # NumPy 2.x: work on ndarrays with explicit out=; rebuild DataFrame from result.
        returns_np = returns.to_numpy(dtype=float) if isinstance(returns, pd.DataFrame) else np.asarray(returns, dtype=float)
        ewm_vol_np = ewm_vol.shift(1).to_numpy(dtype=float) if isinstance(ewm_vol, pd.DataFrame) else np.asarray(ewm_vol.shift(1), dtype=float)
        returns_np = np.divide(
            returns_np, ewm_vol_np,
            out=np.full_like(returns_np, np.nan),
            where=~np.isclose(ewm_vol_np, 0.0),
        )
        if isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns_np, index=returns.index, columns=returns.columns)
        elif isinstance(returns, pd.Series):
            returns = pd.Series(returns_np, index=returns.index, name=returns.name)
        else:
            returns = returns_np

    # rolling returns by the horizon, an integer number of rows
    if horizon > 1:
        rolling_returns = returns.rolling(horizon).sum()
    else:
        rolling_returns = returns

    if signals is not None:
        if signal_agg_type == SignalAggType.LAST_VALUE:
            agg_signal = signals
        elif signal_agg_type == SignalAggType.MEAN:
            agg_signal = signals.rolling(horizon).mean()
        else:
            raise TypeError(f"unknown {signal_agg_type}")

        agg_signal = agg_signal.reindex(index=rolling_returns.index, method='ffill')
        agg_signal = agg_signal.shift(horizon)  # shift backward by the horizon
    else:
        agg_signal = None

    if convolution_type == ConvolutionType.AUTO_CORR:
        x_data = rolling_returns.shift(horizon)  # shift backward by the horizon
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

    # compute ewm cross; the second moments are seeded with var_init_type, ZERO by default
    corr = ewm.compute_ewm_cross_xy(x_data=x_data,
                                    y_data=y_data,
                                    ewm_lambda=ewm_lambda,
                                    cross_xy_type=cross_xy_type,
                                    mean_adj_type=mean_adj_type,
                                    var_init_type=var_init_type)

    if estimates_smoothing_lambda is not None:
        corr = ewm.compute_ewm(data=corr, ewm_lambda=estimates_smoothing_lambda)

    return corr
