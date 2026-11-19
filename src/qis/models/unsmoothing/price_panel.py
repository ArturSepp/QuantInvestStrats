"""Reconstruct native price panels through the existing AR and joint return engines."""
from collections.abc import Mapping
from enum import Enum

import numpy as np
import pandas as pd

from qis.models.linear.ewm import MeanAdjType
from qis.models.unsmoothing.ar_lag import adjust_returns_with_ar
from qis.models.unsmoothing.joint_lag import adjust_returns_with_joint_unsmoothing
from qis.perfstats.returns import log_returns_to_nav, returns_to_nav, to_returns


class UnsmoothingMethod(str, Enum):
    """Return engine used by native-price reconstruction.

    Attributes:
        AR: Own-lag AR(q), with the supplied order and coefficient-sum bounds.
        JOINT: Joint own-lag AR(1) and lagged-factor regression.
    """

    AR = 'ar'
    JOINT = 'joint'


_DIAGNOSTIC_COLUMNS = [
    'date', 'ticker', 'frequency', 'method', 'fitted_ar', 'fitted_lagged_beta',
    'applied_ar', 'applied_lagged_beta', 'has_adjusted_return',
]
_MIN_OBSERVATIONS_AFTER_WARMUP = 1


def _validate_horizons(table: Mapping[str, int], name: str, *, allow_empty: bool) -> None:
    if not isinstance(table, Mapping) or (not table and not allow_empty):
        raise ValueError(f'{name} must be a per-cadence mapping')
    for frequency, periods in table.items():
        if (isinstance(periods, (bool, np.bool_))
                or not isinstance(periods, (int, np.integer)) or periods <= 0):
            raise ValueError(f'{name}[{frequency!r}] must be a positive integer')


def _observed_in_ending_month(prices: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Keep sampled returns only when the ending calendar month contains a native mark."""
    observed = pd.DataFrame(False, index=dates, columns=prices.columns)
    for asset in prices.columns:
        valid_dates = prices[asset].dropna().index
        if valid_dates.empty:
            continue
        positions = valid_dates.get_indexer(dates, method='pad')
        has_mark = positions >= 0
        if has_mark.any():
            last_mark = valid_dates.take(positions[has_mark])
            observed.loc[dates[has_mark], asset] = (
                last_mark.to_period('M') == dates[has_mark].to_period('M')
            )
    return observed


def compute_unsmoothed_price_panel(
        prices: pd.DataFrame, *, frequency: str | pd.Series,
        method: UnsmoothingMethod, span: Mapping[str, int],
        warmup_period: Mapping[str, int], is_log_returns: bool,
        factor_prices: pd.Series | None = None,
        mean_adj_type: MeanAdjType = MeanAdjType.NONE,
        apply_ewma_mean_smoother: bool = False,
        max_ar_coeff: float = 0.75, min_ar_coeff: float | None = 0.0,
        max_factor_beta: float | None = 1.0, min_factor_beta: float | None = 0.0,
        ar_order: int = 1, ar_max_value_for_beta: float | None = 0.75,
        ar_min_value_for_beta: float | None = 0.0, ar_non_negative: bool = False,
        ar_non_negative_tol: float = 0.0, return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Unsmooth supplied native marks per cadence and align reconstructed NAVs to their grid.

    Asset returns use no forward filling before sampling. A sampled ending date must
    have a native mark in its ending calendar month. Missing returns stay unavailable
    to the regression. Coefficients fitted through t-1 apply at t in the existing QIS
    engines. Log reconstruction uses ``log_returns_to_nav``; simple reconstruction uses
    ``returns_to_nav``. These operations retain their initialization and gap behavior.
    Reconstructed period-end NAVs are then aligned to the original grid with a backward
    as-of lookup, retaining weekend period-end observations. No annualisation is applied.

    This operation deliberately differs from the existing AR price helpers in sampling,
    observation support and alignment. It does not select assets or infer their cadence.

    Args:
        prices: Native price frame with a unique increasing DatetimeIndex and asset columns.
        frequency: Explicit pandas cadence, or Series covering every asset without missing
            values. Keys must match the horizon mappings; normalize aliases before calling.
        method: AR for own-lag regression or JOINT for own-lag plus lagged-factor regression.
        span: Positive EWMA spans in observations, separately specified for each cadence.
        warmup_period: Positive warmup counts per cadence; an empty mapping disables this
            mask. With a nonempty mapping, each asset must have at least warmup + 1 observed
            returns. This guard does not guarantee regression identification.
        is_log_returns: Explicit log or simple return convention, also used for the factor.
        factor_prices: Liquid factor price Series, required for JOINT; unused for AR.
        mean_adj_type: Regression mean adjustment. INSAMPLE uses future sample information.
        apply_ewma_mean_smoother: Apply the engine's additional coefficient smoother.
        max_ar_coeff: JOINT upper own-lag bound.
        min_ar_coeff: JOINT lower own-lag bound, or None for no lower bound.
        max_factor_beta: JOINT upper lagged-factor bound, or None.
        min_factor_beta: JOINT lower lagged-factor bound, or None.
        ar_order: AR lag order; unused for JOINT.
        ar_max_value_for_beta: AR upper coefficient-sum bound, or None.
        ar_min_value_for_beta: AR lower coefficient-sum bound, or None.
        ar_non_negative: Apply the AR nonnegative coefficient-sum policy.
        ar_non_negative_tol: AR nonnegative dead-zone tolerance.
        return_diagnostics: Also return long-form fitted/applied coefficients. AR diagnostics
            contain the coefficient sum and zero factor beta; they do not expose each lag.

    Returns:
        A new NAV frame on the original price axes, or (NAV, diagnostics). Diagnostics
        contain date, ticker, frequency, method, fitted_ar, fitted_lagged_beta, applied_ar,
        applied_lagged_beta and has_adjusted_return. Applied coefficients are shifted by
        one observation within each cadence. Inputs are not mutated.

    Raises:
        ValueError: For invalid axes, missing cadence/horizons/factor, invalid positive
            horizon counts, or a history consumed by its requested warmup.
    """
    method = UnsmoothingMethod(method)
    if (not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique
            or not prices.index.is_monotonic_increasing or not prices.columns.is_unique):
        raise ValueError('prices require unique columns and a unique increasing DatetimeIndex')
    if method is UnsmoothingMethod.JOINT and factor_prices is None:
        raise ValueError('joint unsmoothing requires factor_prices')
    _validate_horizons(span, 'span', allow_empty=False)
    _validate_horizons(warmup_period, 'warmup_period', allow_empty=True)
    frequencies = (pd.Series(frequency, index=prices.columns) if isinstance(frequency, str)
                   else frequency.reindex(prices.columns))
    if frequencies.isna().any():
        raise ValueError('frequency must cover every asset without missing values')
    for name, table in (('span', span), ('warmup_period', warmup_period)):
        missing = set(frequencies) - set(table)
        if missing and (name == 'span' or table):
            raise ValueError(f'{name} has no horizon for {sorted(missing)}')

    result = prices.copy()
    diagnostics = []
    for cadence, group in frequencies.groupby(frequencies):
        columns = group.index.intersection(prices.columns)
        if columns.empty:
            continue
        bucket_span = int(span[cadence])
        warmup = int(warmup_period[cadence]) if warmup_period else None
        returns = to_returns(prices[columns], freq=str(cadence),
                             is_log_returns=is_log_returns, ffill_nans=False, drop_first=False)
        returns = returns.where(_observed_in_ending_month(prices[columns], returns.index))
        if warmup is not None:
            counts = returns.notna().sum(axis=0)
            starved = counts[counts < warmup + _MIN_OBSERVATIONS_AFTER_WARMUP]
            if not starved.empty:
                raise ValueError(f'unsmoothing at cadence {cadence!r} with warmup {warmup} '
                                 f'requires at least {warmup + 1} observed returns: '
                                 f'{starved.to_dict()}')
        diagnostic_kwargs = {'return_diagnostics': True} if return_diagnostics else {}
        if method is UnsmoothingMethod.AR:
            unsmoothed = adjust_returns_with_ar(
                returns=returns, ar_order=ar_order, span=bucket_span,
                mean_adj_type=mean_adj_type, warmup_period=warmup,
                max_value_for_beta=ar_max_value_for_beta,
                min_value_for_beta=ar_min_value_for_beta,
                apply_ewma_mean_smoother=apply_ewma_mean_smoother,
                non_negative=ar_non_negative, non_negative_tol=ar_non_negative_tol,
                **diagnostic_kwargs,
            )
        else:
            factor_returns = to_returns(
                factor_prices.to_frame(), freq=str(cadence), is_log_returns=is_log_returns,
                drop_first=False).iloc[:, 0].reindex(returns.index)
            unsmoothed = adjust_returns_with_joint_unsmoothing(
                returns=returns, factor_returns=factor_returns, span=bucket_span,
                mean_adj_type=mean_adj_type, warmup_period=warmup,
                max_ar_coeff=max_ar_coeff, min_ar_coeff=min_ar_coeff,
                max_factor_beta=max_factor_beta, min_factor_beta=min_factor_beta,
                apply_ewma_mean_smoother=apply_ewma_mean_smoother, **diagnostic_kwargs,
            )
        if return_diagnostics:
            unsmoothed, phi, auxiliary = unsmoothed
            beta = phi * 0.0 if method is UnsmoothingMethod.AR else auxiliary
            for column in columns:
                item = pd.DataFrame({
                    'ticker': column, 'frequency': cadence, 'method': method.value,
                    'fitted_ar': phi[column], 'fitted_lagged_beta': beta[column],
                    'applied_ar': phi[column].shift(1),
                    'applied_lagged_beta': beta[column].shift(1),
                    'has_adjusted_return': unsmoothed[column].notna(),
                })
                item.index.name = 'date'
                diagnostics.append(item.loc[
                    item.fitted_ar.notna() | item.applied_ar.notna()].reset_index())
        nav = log_returns_to_nav(unsmoothed) if is_log_returns else returns_to_nav(unsmoothed)
        result[columns] = nav.reindex(index=prices.index, method='ffill')
    if return_diagnostics:
        details = (pd.concat(diagnostics, ignore_index=True) if diagnostics
                   else pd.DataFrame(columns=_DIAGNOSTIC_COLUMNS))
        return result, details
    return result
