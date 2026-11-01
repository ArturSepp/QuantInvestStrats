"""Point-in-time samples for equal-weighted and EWMA-weighted HAC regressions.

At input row i, fit the prefix through i or the last W observations, including i.
``span`` weights observations within that sample; it never smooths fitted coefficients.
All estimation and Bartlett-HAC inference belong to ``utils.regression``. Window, span
and lags are in input observations, irrespective of the output schedule. Coefficients
retain the input units and no annualisation or trading implementation lag is applied.

Inputs must be complete on an explicitly declared calendar. Outputs are pointwise
diagnostics, with no simultaneous, sequential or selected-window error control.
"""

# packages
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

# qis
from qis.utils import regression
from qis.utils.dates import TimePeriod


class RollingHacStatus(str, Enum):
    """Endpoint state; invalid eligible fits raise instead of becoming missing rows.

    Attributes:
        WARMUP: Fewer than the requested minimum input observations are available.
        FITTED: The shared estimator returned a valid fit.
    """

    WARMUP = 'warmup'
    FITTED = 'fitted'


@dataclass(frozen=True)
class RollingHacRegressionResult:
    """Scheduled HAC estimates and per-endpoint sample metadata.

    Attributes:
        coefficients: Estimates, with dates as rows and (equation, parameter) columns.
        hac_se: Matching HAC standard errors in coefficient units.
        pvalues: Matching two-sided, normal-reference, pointwise p-values.
        confidence_lower: Lower bounds of unadjusted pointwise confidence intervals.
        confidence_upper: Upper bounds of unadjusted pointwise confidence intervals.
        diagnostics: Requested/actual sample bounds, nobs, effective_nobs, window, span,
            actual hac_lags and RollingHacStatus for each output date. A requested start
            before supplied history is NaT; actual bounds describe the available sample.
            Effective size and lag count are NaN during warmup, when no fit is made.
        parameter_covariance: Optional mapping from fitted dates to joint covariance
            matrices, with (equation, parameter) axes; no warmup entries.
        confidence_level: Confidence level of the pointwise intervals.
    """

    coefficients: pd.DataFrame
    hac_se: pd.DataFrame
    pvalues: pd.DataFrame
    confidence_lower: pd.DataFrame
    confidence_upper: pd.DataFrame
    diagnostics: pd.DataFrame
    parameter_covariance: Optional[Dict[pd.Timestamp, pd.DataFrame]]
    confidence_level: float


def _validate_period_count(value: int, name: str, n_parameters: int) -> int:
    """Require an integer observation count greater than the design dimension."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer greater than {n_parameters}, got {value!r}')
    if value <= n_parameters:
        raise ValueError(f'{name} must exceed {n_parameters} fitted parameters, got {value!r}')
    return int(value)


def _validate_calendar(index: pd.Index,
                       sampling_frequency: Optional[str],
                       expected_index: Optional[pd.DatetimeIndex]) -> None:
    """Reject gaps instead of inferring a coarser frequency or compressing time."""
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError('y must have a DatetimeIndex')
    if index.empty or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError('input dates must be nonempty, ordered, unique and without NaT')
    if sampling_frequency is not None and expected_index is not None:
        raise ValueError('specify only one of sampling_frequency and expected_index')
    if expected_index is not None:
        if not isinstance(expected_index, pd.DatetimeIndex) or not index.equals(expected_index):
            raise ValueError('input dates must exactly equal expected_index')
        return
    frequency = sampling_frequency if sampling_frequency is not None else index.freq
    if frequency is None:
        raise ValueError('declare sampling_frequency or expected_index, or attach index.freq')
    try:
        expected = pd.date_range(start=index[0], end=index[-1], freq=frequency)
    except (TypeError, ValueError) as exception:
        raise ValueError(f'invalid sampling_frequency, got {frequency!r}') from exception
    if not index.equals(expected):
        raise ValueError(f'input dates must be complete on sampling_frequency={frequency!r}')


def estimate_rolling_hac_regression(
        y: Union[pd.Series, pd.DataFrame],
        x: Optional[Union[pd.Series, pd.DataFrame]] = None,
        window: Optional[int] = None,
        span: Optional[float] = None,
        min_periods: Optional[int] = None,
        hac_lags: int = 3,
        confidence_level: float = 0.95,
        fit_dates: Optional[pd.DatetimeIndex] = None,
        time_period: Optional[TimePeriod] = None,
        sampling_frequency: Optional[str] = None,
        expected_index: Optional[pd.DatetimeIndex] = None,
        include_covariance: bool = False,
) -> RollingHacRegressionResult:
    """Fit historical samples by delegating each eligible date to the shared HAC core.

    ``window=None`` uses an expanding prefix; a finite window uses the last W input
    observations. ``span=None`` gives equal weights, otherwise each sample uses geometric
    observation weights. Both finite arguments give truncated EWMA weighting. An estimate
    at t includes the observation ending at t and becomes available only afterwards.

    Args:
        y: Finite response Series or DataFrame on ordered, unique observation dates.
        x: Optional common factors with exactly the same dates; None fits means only.
        window: Trailing input-row count, greater than the number k of fitted parameters.
        span: EWMA decay span; None selects exactly equal weights.
        min_periods: Minimum rows before fitting; defaults to window, or max(24, k+1)
            for an expanding sample. Must exceed k and cannot exceed a finite window.
        hac_lags: Requested Bartlett lag count; the core reports its per-sample cap.
        confidence_level: Two-sided pointwise confidence level.
        fit_dates: Ordered, unique subset of input dates; None requests every date.
        time_period: Restrict output dates only, retaining all earlier estimation history.
        sampling_frequency: Declared regular observation calendar, e.g. 'ME'. If neither
            calendar argument is supplied, use an explicitly attached index.freq.
        expected_index: Exact observation dates for a custom calendar; mutually exclusive
            with sampling_frequency. No dates are inferred, filled or dropped.
        include_covariance: Retain each successful fit's full joint covariance if True.

    Returns:
        Scheduled coefficient/inference frames and sample diagnostics. Warmup estimates
        remain NaN. Covariance storage does not affect the statistics.

    Raises:
        TypeError: If response/factor containers or HAC settings have invalid types.
        ValueError: If labels, dates, calendar, finite values or configuration are invalid,
            or an eligible fit fails. Fit errors identify the endpoint and original reason.
    """
    span = regression._validate_hac_settings(span, hac_lags, confidence_level)
    y_frame, x_frame = regression._prepare_hac_inputs(y, x)
    _validate_calendar(y_frame.index, sampling_frequency, expected_index)
    if not np.isfinite(y_frame.to_numpy()).all() or (
            x_frame is not None and not np.isfinite(x_frame.to_numpy()).all()
    ):
        raise ValueError('rolling HAC inputs must be finite; rows cannot be dropped or filled')
    if not isinstance(include_covariance, (bool, np.bool_)):
        raise ValueError(f'include_covariance must be boolean, got {include_covariance!r}')
    if time_period is not None and not isinstance(time_period, TimePeriod):
        raise ValueError('time_period must be a TimePeriod or None')

    parameters = ['Intercept'] + ([] if x_frame is None else x_frame.columns.tolist())
    n_parameters = len(parameters)
    if window is not None:
        window = _validate_period_count(window, 'window', n_parameters)
    if min_periods is None:
        min_periods = window if window is not None else max(24, n_parameters + 1)
    min_periods = _validate_period_count(min_periods, 'min_periods', n_parameters)
    if window is not None and min_periods > window:
        raise ValueError(f'min_periods={min_periods} must not exceed window={window}')

    dates = y_frame.index
    if fit_dates is None:
        fit_dates = dates
    elif (not isinstance(fit_dates, pd.DatetimeIndex) or fit_dates.hasnans
          or not fit_dates.is_unique or not fit_dates.is_monotonic_increasing
          or not fit_dates.isin(dates).all() or fit_dates.tz != dates.tz):
        raise ValueError('fit_dates must be an ordered, unique subset of input dates')
    if time_period is not None:
        fit_dates = time_period.locate(pd.Series(index=fit_dates, dtype=float)).index

    columns = pd.MultiIndex.from_product(
        [y_frame.columns, parameters], names=['equation', 'parameter'],
    )
    statistic_names = ('coefficients', 'hac_se', 'pvalues', 'confidence_lower', 'confidence_upper')
    statistics = {
        name: pd.DataFrame(np.nan, index=fit_dates, columns=columns) for name in statistic_names
    }
    diagnostic_columns = [
        'requested_start', 'requested_end', 'actual_start', 'actual_end', 'nobs',
        'effective_nobs', 'window', 'span', 'hac_lags', 'status',
    ]
    records = []
    covariances = {} if include_covariance else None
    for output_row, input_row in enumerate(dates.get_indexer(fit_dates)):
        endpoint = dates[input_row]
        requested_start_row = 0 if window is None else input_row - window + 1
        start_row = max(0, requested_start_row)
        stop_row = input_row + 1
        nobs = stop_row - start_row
        record = dict(
            requested_start=dates[requested_start_row] if requested_start_row >= 0 else pd.NaT,
            requested_end=endpoint,
            actual_start=dates[start_row],
            actual_end=endpoint,
            nobs=nobs,
            effective_nobs=np.nan,
            window=window,
            span=span,
            hac_lags=np.nan,
            status=RollingHacStatus.WARMUP,
        )
        if nobs >= min_periods:
            try:
                result = regression.estimate_hac_regression(
                    y=y_frame.iloc[start_row:stop_row],
                    x=None if x_frame is None else x_frame.iloc[start_row:stop_row],
                    span=span, hac_lags=hac_lags, confidence_level=confidence_level,
                )
                if result.r_squared.isna().any():
                    constant_columns = result.r_squared.index[result.r_squared.isna()].tolist()
                    raise ValueError(f'constant response sample has undefined R-squared: '
                                     f'{constant_columns}')
            except ValueError as exception:
                raise ValueError(
                    f'HAC fit at {endpoint.isoformat()} failed: {exception}'
                ) from exception
            for name in statistic_names:
                statistics[name].iloc[output_row] = getattr(result, name).to_numpy().T.ravel()
            record.update(effective_nobs=result.effective_nobs, hac_lags=result.hac_lags,
                          status=RollingHacStatus.FITTED)
            if covariances is not None:
                covariances[endpoint] = result.parameter_covariance
        records.append(record)
    diagnostics = pd.DataFrame(records, index=fit_dates, columns=diagnostic_columns).astype({
        'nobs': 'int64', 'effective_nobs': 'float64', 'window': 'Int64',
        'span': 'float64', 'hac_lags': 'float64',
    })
    for name in ('requested_start', 'requested_end', 'actual_start', 'actual_end'):
        diagnostics[name] = pd.Series(
            [record[name] for record in records], index=fit_dates, dtype=dates.dtype,
        )
    # Explicit object dtype retains enum members under pandas' string inference as well.
    diagnostics['status'] = pd.Series(
        [record['status'] for record in records], index=fit_dates, dtype=object,
    )
    return RollingHacRegressionResult(
        **statistics,
        diagnostics=diagnostics,
        parameter_covariance=covariances,
        confidence_level=confidence_level,
    )
