"""
risk-adjusted returns: divide by an EWM volatility, then build signals on the result.

The primitive is ``compute_ra_returns``:

    w_t = vol_target / sigma_t,   ra_t = w_t r_t

with sigma an EWM volatility at ``span`` or ``ewm_lambda``. Three conventions hold throughout.
The weight is lagged by ``weight_lag``, 1 by default, so the scaling applied over [t, t+1] uses
the volatility known at t and the construction carries no look-ahead. Sigma is never annualised:
it is a volatility per period of the return grid, so ``vol_target`` is a target per period too,
and an annual target enters as ``sigma_annual / sqrt(AN)`` (for example ``0.15 / sqrt(252)`` on
business days). And ``vol_target=None`` means a non-dimensional unit target rather than no
scaling: the output is then in units of risk, comparable across assets and summable across a
panel, but not a tradeable return stream. Pass an explicit target to size a position. The
function returns the triple (ra_returns, weights, ewm_vol), since the weights and the vol are
usually wanted alongside.

``vol_floor_quantile`` floors sigma at a rolling quantile of itself, so a quiet sample does not
produce an unbounded weight; ``is_log_returns_to_arithmetic`` converts back with expm1 before
scaling, for the common case where the vol was estimated on log returns.

The sum functions aggregate risk-adjusted returns and divide by the square root of the number of
terms summed, so that uncorrelated unit-variance terms give unit-variance sums:
``compute_sum_rolling_ra_returns`` over a rolling window of ``span`` rows and
``compute_sum_freq_ra_returns`` over calendar periods of ``freq``, dividing each period's sum by
the root of its own observation count. ``get_paired_rareturns_signals`` pairs such sums with a
signal known before the window starts.

``compute_ewm_long_short_filtered_ra_returns`` is the trend primitive built on top: normalise by
vol, then take the difference of a fast and a slow EWM to isolate the medium-frequency
component. Every span drives λ = 1 - 2/(span + 1) and must be at least 1; below that the
recursion alternates sign instead of smoothing, which on the vol leg yields a negative variance
and NaN, so the spans are validated rather than trusted.

``map_signal_to_weight`` turns a signal into a bounded weight: ``SignalMapType`` selects a
normal or Laplace CDF, or ``ExpCDF``, a Gaussian-shaped saturating map with separate anchor
levels for the two sides and optional tail fading. ``compute_returns_transform`` dispatches over
``ReturnsTransform`` for the rolling and momentum variants.

Turning weights into a portfolio is ``qis.backtest_model_portfolio``; the EWM estimators
themselves are ``qis/models/linear/ewm.py``.
"""
# packages
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import laplace
from typing import Union, Optional, Tuple
from enum import Enum
# qis
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
    """
    divide returns by a lagged EWM volatility: ``ra_t = vol_target * r_t / sigma_(t - weight_lag)``.

    The volatility is never annualised. ``sigma_t`` is the per-period EWM volatility of the
    supplied returns, so ``vol_target`` is a volatility per period of the same grid: an annual
    target ``sigma_annual`` is passed as ``sigma_annual / sqrt(AN)``, for example
    ``0.15 / np.sqrt(252)`` on business-day returns. ``vol_target=0.15`` on daily returns targets
    15% per day.

    Args:
        returns: periodic returns, one column per asset
        span: EWM span of the volatility; overrides ``ewm_lambda`` via ``1 - 2 / (span + 1)``.
            May be an array with one entry per column
        ewm_lambda: EWM decay of the volatility, 0.94 (RiskMetrics) by default
        vol_target: volatility target per period of the return grid. None means a unit target,
            so the output is in units of risk, not unscaled returns
        mean_adj_type: mean removed before the second moment; ``MeanAdjType.NONE`` by default,
            so ``sigma`` is the root of an EWM second moment about zero
        init_value: explicit seed of the variance recursion; the first squared return by default
        vol_floor_quantile: floor ``sigma`` at this rolling quantile of itself; None disables it
        vol_floor_quantile_roll_period: rolling window of the floor quantile, in rows
        warmup_period: number of leading finite volatility estimates set to missing
        is_log_returns_to_arithmetic: map the returns to ``exp(r) - 1`` before scaling, while the
            volatility is still estimated on the supplied (log) returns
        weight_lag: rows between the volatility estimate and the return it scales; 1 by default,
            so the scale is known before the return. None or 0 divides a return by a volatility
            that already contains it

    Returns:
        the triple ``(ra_returns, weights, ewm_vol)``: the scaled returns, the weights
        ``vol_target / ewm_vol`` already lagged by ``weight_lag`` (dated at the return they
        scale), and the unlagged per-period EWM volatility. The target weight to execute at t is
        ``vol_target / ewm_vol`` at t, not the lagged ``weights``
    """
    if span is not None:
        ewm_lambda = 1.0-2.0/(span+1.0)

    if vol_target is None:  # non-dimensional unit target: output in units of risk
        vol_target = 1.0

    ewm_vol = ewm.compute_ewm_vol(data=returns,
                                  ewm_lambda=ewm_lambda,
                                  mean_adj_type=mean_adj_type,
                                  init_value=init_value,
                                  vol_floor_quantile=vol_floor_quantile,
                                  vol_floor_quantile_roll_period=vol_floor_quantile_roll_period,
                                  warmup_period=warmup_period,
                                  annualize=False)  # per-period vol: vol_target is per period

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
    """
    vol-normalise returns, then band-pass them with a long/short EWM filter.

    The trend-following signal primitive: dividing by EWM vol puts every asset on comparable risk,
    and the difference of a short and a long EWM isolates the medium-frequency component that a
    trend signal is built from. The result is in units of risk, so it can be summed across assets.

    Every span drives a decay ``lambda = 1 - 2/(span + 1)`` and so must be at least 1: ``span = 1``
    passes the data through unsmoothed, and below 1 the recursion alternates sign rather than
    smoothing, which on the vol leg can produce a negative variance and hence NaN.

    Timing: the output is not shifted. The value dated t is a signal formed at t from the
    risk-adjusted returns ``x_s = r_s / sigma_(s - weight_lag)`` with s <= t, and is to be applied
    over (t, t+1] by the caller (for example through ``qis.backtest_model_portfolio``). The
    two-leg filter puts zero weight on ``x_t``, so its value at t uses ``x`` only through t-1;
    the single-leg filter (``short_span=None``) loads on ``x_t`` with weight
    ``sqrt(1 - lambda_long^2)``. Neither uses a return after t.

    Args:
        returns: returns panel, one column per asset
        vol_span: EWM span of the volatility used to normalise. None skips the normalisation
        long_span: EWM span of the slow leg
        short_span: EWM span of the fast leg, strictly less than ``long_span``. None applies the
            long leg alone
        warmup_period: leading periods blanked, before which the EWM state is still converging
        weight_lag: lag of the volatility normaliser, passed to ``compute_ra_returns``: the return
            at t is divided by the volatility estimated at t - ``weight_lag``, so the scale is
            known before the return. It does not shift the filter output
        mean_adj_type: mean subtracted before the vol estimate; see :class:`MeanAdjType`

    Returns:
        the filtered risk-adjusted returns, in the shape of ``returns``, dated at formation

    Raises:
        ValueError: if any span is below 1, or if ``short_span`` is not strictly less than
            ``long_span``
    """
    if vol_span is not None and np.any(np.asarray(vol_span, dtype=float) < 1.0):
        raise ValueError(f"compute_ewm_long_short_filtered_ra_returns: vol_span must be >= 1 "
                         f"(lambda = 1 - 2/(span+1) is negative below span 1); got vol_span={vol_span}")
    ewm._validate_long_short_spans(long_span=long_span, short_span=short_span)

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
    """
    shape of the signal-to-weight map in ``map_signal_to_weight``.

    With ``eta = (y - loc) / scale``:

    Attributes:
        NormalCDF: ``2 * Phi(eta) - 1``; odd about ``loc``, bounded in (-1, 1), slope
            ``sqrt(2 / pi) / scale`` at the centre. Uses ``loc`` and ``scale`` only
        LaplaceCDF: ``sign(eta) * (1 - exp(-|eta|))``; slope ``1 / scale`` at the centre,
            exponential approach to the bound. Uses ``loc`` and ``scale`` only
        ExpCDF: not a CDF of an exponential law despite its name. A Gaussian-shaped saturating
            map ``+-tail_level * (1 - exp(-(y - loc)^2 / omega))``, quadratic and so of zero slope
            at ``loc``, which damps small signals. ``scale`` acts as a variance: the weight is
            ``+slope_right`` at ``y - loc = 1.25 sqrt(scale)`` and ``-slope_left`` at
            ``y - loc = -1.25 sqrt(scale)``, and tends to ``+-tail_level``; optional tail decays
            fade extreme signals
    """
    NormalCDF = 1
    LaplaceCDF = 2
    ExpCDF = 3


# arguments that only SignalMapType.ExpCDF reads, with their defaults
_EXP_CDF_ONLY_DEFAULTS = {'tail_level': 1.0, 'slope_right': 0.5, 'slope_left': 0.5,
                          'tail_decay_right': None, 'tail_decay_left': None}
# the anchor of ExpCDF sits at |y - loc| = _EXP_CDF_ANCHOR * sqrt(scale)
_EXP_CDF_ANCHOR = 1.25


def _differs_from_default(value, default) -> bool:
    """Whether an argument was set away from its default (elementwise for arrays)."""
    if value is None:
        return False
    if default is None:
        return True
    return bool(np.any(np.asarray(value) != default))


def map_signal_to_weight(signals: pd.DataFrame,
                         signal_map_type: SignalMapType = SignalMapType.NormalCDF,
                         loc: Union[float, pd.DataFrame] = 0.0,
                         scale: Union[float, np.ndarray] = 1.0,
                         tail_level: Union[float, np.ndarray] = 1.0,
                         slope_right: Union[float, np.ndarray] = 0.5,
                         slope_left: Union[float, np.ndarray] = 0.5,
                         tail_decay_right: Optional[Union[float, np.ndarray]] = None,
                         tail_decay_left: Optional[Union[float, np.ndarray]] = None
                         ) -> pd.DataFrame:
    """
    map a signal to a bounded weight through the shape selected by ``signal_map_type``.

    ``NormalCDF`` returns ``2 * Phi((y - loc) / scale) - 1`` and ``LaplaceCDF`` the Laplace
    analogue; both read only ``loc`` and ``scale`` and emit a UserWarning when any
    ``ExpCDF``-only argument differs from its default.

    ``ExpCDF`` returns ``+-q * (1 - exp(-(y - loc)^2 / omega))`` with ``q = tail_level``, the ``+``
    branch for ``y >= loc`` and ``omega = 1.5625 * scale / ln(q / (q - p))``, where ``p`` is
    ``slope_right`` on the right and ``slope_left`` on the left. The constant
    ``1.5625 = 1.25^2`` places the anchor at ``|y - loc| = 1.25 sqrt(scale)``, where the weight
    equals exactly ``+-p``; equivalently the weight is ``+-q * (1 - (1 - p / q)^(v^2))`` with
    ``v = (y - loc) / (1.25 sqrt(scale))``. So the "slopes" are weight levels at the anchor, not
    derivatives, and ``scale`` acts as a variance. The map is quadratic near ``loc`` (zero slope
    there) and tends to ``+-q``.

    Tail fading (``ExpCDF`` only): with ``tail_decay_right = d+`` the weight is multiplied by
    ``exp(-(y - q - max(loc, 0)) / d+)`` for ``y > q + max(loc, 0)``, and with
    ``tail_decay_left = d-`` by ``exp((y + q - min(loc, 0)) / d-)`` for
    ``y < -q + min(loc, 0)``. Each side is faded only when its own decay is given. ``tail_level``
    therefore plays two roles: the weight cap and the signal threshold beyond which fading
    starts.

    Args:
        signals: signal panel, one column per asset
        signal_map_type: shape of the map; see :class:`SignalMapType`
        loc: centre of the map, a scalar or a frame shaped like ``signals``
        scale: scale of the map, a scalar or one entry per column: the standard deviation of the
            normal map, the scale parameter of the Laplace map, a variance for ``ExpCDF``
        tail_level: ``ExpCDF`` only: the weight cap ``q`` and the fading threshold
        slope_right: ``ExpCDF`` only: weight at ``y - loc = 1.25 sqrt(scale)``; below
            ``tail_level``
        slope_left: ``ExpCDF`` only: absolute weight at ``y - loc = -1.25 sqrt(scale)``; below
            ``tail_level``
        tail_decay_right: ``ExpCDF`` only: decay length, in signal units, of the right-tail
            fading; None disables it
        tail_decay_left: ``ExpCDF`` only: decay length of the left-tail fading; None disables it

    Returns:
        weights in the shape of ``signals``

    Raises:
        ValueError: if an array argument does not have one entry per column, or, for ``ExpCDF``,
            if ``tail_level`` does not exceed both slopes
        NotImplementedError: for an unknown ``signal_map_type``
    """
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

    if signal_map_type in (SignalMapType.NormalCDF, SignalMapType.LaplaceCDF):
        passed = {'tail_level': tail_level, 'slope_right': slope_right, 'slope_left': slope_left,
                  'tail_decay_right': tail_decay_right, 'tail_decay_left': tail_decay_left}
        ignored = [name for name, value in passed.items()
                   if _differs_from_default(value, _EXP_CDF_ONLY_DEFAULTS[name])]
        if ignored:
            warnings.warn(f"map_signal_to_weight: {', '.join(ignored)} ignored by "
                          f"{signal_map_type}, which reads only loc and scale; these arguments "
                          f"apply to SignalMapType.ExpCDF", UserWarning, stacklevel=2)

    if signal_map_type == SignalMapType.NormalCDF:
        weight = 2.0*norm.cdf(x=x, loc=loc, scale=scale) - 1.0

    elif signal_map_type == SignalMapType.LaplaceCDF:
        weight = 2.0*laplace.cdf(x=x, loc=loc, scale=scale) - 1.0

    elif signal_map_type == SignalMapType.ExpCDF:
        if np.any(np.less_equal(tail_level, slope_right)) or np.any(np.less_equal(tail_level, slope_left)):
            raise ValueError("must be tail>slope_positive and tail > slope_negative")
        # omega = 1.25^2 * scale / ln(q / (q - p)) puts the weight p at |x - loc| = 1.25 sqrt(scale)
        anchor2 = _EXP_CDF_ANCHOR ** 2  # = 1.5625
        scale_negative = anchor2 * scale / np.log(tail_level / (tail_level - slope_left))
        scale_positive = anchor2 * scale / np.log(tail_level / (tail_level - slope_right))
        s_negative = - tail_level * (1.0 - np.exp(-np.square(x - loc) / scale_negative))
        s_positive = tail_level * (1.0 - np.exp(-np.square(x - loc) / scale_positive))
        # NumPy 2.x: comparison with `where=` needs `out=` so masked positions are False,
        # causing np.where to select s_positive (the safer default for non-finite x).
        finite_mask = np.isfinite(x)
        less_mask = np.less(x, loc, out=np.zeros_like(finite_mask, dtype=bool), where=finite_mask)
        weight = np.where(less_mask, s_negative, s_positive)

        # tail fading beyond +-tail_level (shifted by max(loc, 0) and min(loc, 0)); each side is
        # faded only when its own decay is given
        if tail_decay_right is not None or tail_decay_left is not None:
            if isinstance(tail_decay_right, np.ndarray) and tail_decay_right.shape[0] != x.shape[1]:
                raise ValueError(f"{tail_decay_right.shape[0]} != {x.shape[1]}")
            if isinstance(tail_decay_left, np.ndarray) and tail_decay_left.shape[0] != x.shape[1]:
                raise ValueError(f"{tail_decay_left.shape[0]} != {x.shape[1]}")

            x_left_tail = x + tail_level - np.where(np.less(loc, 0.0), loc, 0.0)
            if tail_decay_left is not None:
                f_left_tail = np.where(np.less(x_left_tail, 0.0),
                                       np.exp(x_left_tail / tail_decay_left), 1.0)
            else:
                f_left_tail = np.ones_like(x_left_tail, dtype=float)

            x_right_tail = x - tail_level - np.where(np.greater(loc, 0.0), loc, 0.0)
            if tail_decay_right is not None:
                f_right_tail = np.where(np.greater(x_right_tail, 0.0),
                                        np.exp(-x_right_tail / tail_decay_right), 1.0)
            else:
                f_right_tail = np.ones_like(x_right_tail, dtype=float)

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
    sum risk-adjusted returns within non-overlapping calendar periods of ``freq``.

    The terms are ``x_t`` from ``compute_ra_returns`` (``span``, when given, is its volatility
    span). For each calendar period J of ``freq`` the output is ``sum_{t in J} x_t``, divided with
    ``is_norm=True`` by ``sqrt(n_J)``, where ``n_J`` is the number of finite ``x_t`` in J. For
    uncorrelated unit-variance terms the normalised sums then have unit variance whatever the
    frequency and however many observations a period holds; a period without observations is
    missing. Interpretation: the P&L over each period of the per-period volatility-targeted
    position, in units of its own period volatility.

    Args:
        returns: periodic returns, usually business-daily
        freq: calendar frequency of the sums. ``'B'`` and ``'D'`` return ``x_t`` unchanged
        span: EWM span of the volatility; overrides ``ewm_lambda``
        ewm_lambda: EWM decay of the volatility
        vol_target: per-period volatility target; None is a unit target
        weight_shift: lag of the volatility normaliser, ``weight_lag`` of ``compute_ra_returns``
        is_log_returns_to_arithmetic: map log returns to simple returns before scaling
        is_norm: divide each period's sum by the root of its observation count; False returns
            the plain sums, with zero for a period without observations
        warmup_period: number of leading volatility estimates set to missing

    Returns:
        one row per calendar period of ``freq``, labelled as ``resample(freq)`` labels it
    """
    ra_returns, _, _ = compute_ra_returns(returns=returns,
                                          span=span,
                                          ewm_lambda=ewm_lambda,
                                          vol_target=vol_target,
                                          weight_lag=weight_shift,
                                          is_log_returns_to_arithmetic=is_log_returns_to_arithmetic,
                                          warmup_period=warmup_period)

    if freq not in ['B', 'D']:
        resampled = ra_returns.resample(freq)
        sum_rolling_ra_returns = resampled.sum()

        if is_norm:
            # the number of terms in each period, so unit-variance terms give unit-variance sums
            n_obs = resampled.count()
            sum_rolling_ra_returns = sum_rolling_ra_returns.divide(np.sqrt(n_obs.where(n_obs > 0)))
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
                                 freq: str = 'BQE',
                                 span: int = 63,
                                 is_nonoverlapping: bool = True,
                                 ra_returns_ewm_vol_lambda: float = 0.94,
                                 is_mean_adjust_returns: bool = False
                                 ) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:
    """
    pair normalised sums of risk-adjusted returns with the signal known before each window.

    Both modes are forward looking: the signal paired with a window is the last value observed
    before the window starts, so a predictive diagnostic on the pairs carries no look-ahead.

    With ``is_nonoverlapping=True`` the returns are ``compute_sum_freq_ra_returns`` over the
    calendar periods of ``freq`` (unit-variance sums, log-to-simple map on) and the signal is
    the last value of the previous period, ``signal.resample(freq).last().shift(1)``. With
    ``is_nonoverlapping=False`` the returns are ``compute_sum_rolling_ra_returns`` over the
    rolling window of ``span`` rows ending at t, which covers rows (t - span, t], and the signal is
    the value at t - span, ``signal.shift(span)``. Consecutive overlapping windows share
    ``span - 1`` rows, so inference on them needs an autocorrelation-robust standard error.

    Args:
        returns: periodic returns, usually business-daily
        signal: signal on the same grid, dated when it is formed
        freq: calendar frequency of the non-overlapping mode; ``'BQE'`` (business quarter-end)
            by default, a pandas alias valid under pandas 2.2 and 3
        span: rolling window, in rows, of the overlapping mode
        is_nonoverlapping: calendar periods of ``freq`` (True) or rolling windows of ``span``
        ra_returns_ewm_vol_lambda: EWM decay of the volatility normaliser
        is_mean_adjust_returns: subtract the expanding mean of the paired returns through each
            date, which is point in time

    Returns:
        the pair ``(ra_return, indicator)``, dated at the end of each return window; each keeps
        the index of its own input, so align them by date before use
    """
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
        # the window ending at t covers (t - span, t]: the signal must predate it
        indicator = signal.shift(span)

    if is_mean_adjust_returns:
        ra_returns_mean = ra_return.expanding(min_periods=1).mean()  # point in time
        ra_return = ra_return.subtract(ra_returns_mean)

    return ra_return, indicator


class ReturnsTransform(Enum):
    ROLLING_RA_RETURNS = 1
    EWMA_RETURNS_MOMENTUM = 2


def compute_returns_transform(returns: pd.DataFrame,
                              returns_transform: ReturnsTransform = ReturnsTransform.ROLLING_RA_RETURNS,
                              momentum_span: int = 31,
                              vol_span: int = 33,
                              rolling_ra_returns_span: int = 31
                              ) -> pd.DataFrame:
    """
    apply the returns transform selected by ``returns_transform``.

    ``ROLLING_RA_RETURNS`` calls ``compute_rolling_ra_returns(span=rolling_ra_returns_span,
    weight_shift=1)``; ``EWMA_RETURNS_MOMENTUM`` calls ``compute_ewm_ra_returns_momentum(
    momentum_span, vol_span, weight_shift=1)``.

    The defaults of this dispatcher, ``momentum_span=31`` and ``vol_span=33``, differ on purpose
    from those of ``compute_ewm_ra_returns_momentum`` (63 and 31): they make the two transforms
    comparable. An EWM of span 31 has mean age (31 - 1) / 2 = 15 rows, the same as the 31-row
    rolling window of ``ROLLING_RA_RETURNS``, and span 33 gives the decay 1 - 2/34 = 0.941 of the
    RiskMetrics daily volatility (0.94). Pass the spans explicitly to reproduce the underlying
    function's defaults.

    Args:
        returns: periodic returns, one column per asset
        returns_transform: the transform to apply
        momentum_span: EWM span of the momentum filter of ``EWMA_RETURNS_MOMENTUM``
        vol_span: EWM span of the volatility normaliser of ``EWMA_RETURNS_MOMENTUM``
        rolling_ra_returns_span: summation window, in rows, of ``ROLLING_RA_RETURNS``

    Returns:
        the transformed returns, in the shape of ``returns``

    Raises:
        TypeError: for a ``returns_transform`` that is not implemented
    """
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
