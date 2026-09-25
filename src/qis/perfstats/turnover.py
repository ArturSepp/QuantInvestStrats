"""Two-sided portfolio-turnover computations independent of the reporting layer."""
from __future__ import annotations

import warnings
from enum import Enum
from typing import Optional, TypeVar

import pandas as pd


_PandasObject = TypeVar('_PandasObject', pd.DataFrame, pd.Series)


class TurnoverComputationType(str, Enum):
    """Select the holdings and denominator used to compute two-sided turnover.

    Attributes:
        TARGET_WEIGHTS: Absolute changes in target weights. This is a target-allocation proxy,
            not a reconstruction of executed trades.
        VOLATILITY_NORMALIZED_WEIGHTS: Absolute changes in target weights multiplied by
            contemporaneous annualized volatility. This is the theoretical turnover convention
            of Sepp and Lucic (2026), Definition 4.5.
        EXECUTED_NOTIONAL_NAV: Absolute changes in units valued at current unit notional and
            divided by portfolio NAV.
        EXECUTED_NOTIONAL_GROSS: Absolute changes in units valued at current unit notional and
            divided by current gross exposure.
    """

    TARGET_WEIGHTS = 'target_weights'
    VOLATILITY_NORMALIZED_WEIGHTS = 'volatility_normalized_weights'
    EXECUTED_NOTIONAL_NAV = 'executed_notional_nav'
    EXECUTED_NOTIONAL_GROSS = 'executed_notional_gross'


def resolve_turnover_computation_type(
        default: TurnoverComputationType,
        computation_type: Optional[TurnoverComputationType] = None,
        is_unit_based_traded_volume: Optional[bool] = None,
        ) -> TurnoverComputationType:
    """Resolve the new enum and the deprecated boolean turnover selector.

    Args:
        default: Convention used when neither selector is supplied.
        computation_type: Explicit enum convention.
        is_unit_based_traded_volume: Deprecated selector. True maps to executed notional over
            gross exposure; False maps to target-weight turnover.

    Returns:
        Resolved turnover convention.

    Raises:
        ValueError: If both the enum and deprecated boolean are supplied.
    """
    if computation_type is not None and is_unit_based_traded_volume is not None:
        raise ValueError(
            "pass turnover_computation_type or is_unit_based_traded_volume, not both"
        )
    if is_unit_based_traded_volume is not None:
        warnings.warn(
            "is_unit_based_traded_volume is deprecated for turnover; use "
            "turnover_computation_type instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if is_unit_based_traded_volume:
            return TurnoverComputationType.EXECUTED_NOTIONAL_GROSS
        return TurnoverComputationType.TARGET_WEIGHTS
    if computation_type is None:
        return TurnoverComputationType(default)
    return TurnoverComputationType(computation_type)


def _require_frame(data: Optional[pd.DataFrame], name: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame for this turnover computation")
    return data


def _chronological_dated_input(data: _PandasObject, name: str) -> _PandasObject:
    """Validate and chronologically order one dated turnover input."""
    if not isinstance(data.index, pd.DatetimeIndex):
        return data
    if data.index.hasnans:
        raise ValueError(f"{name} index must not contain NaT")
    if data.index.has_duplicates:
        raise ValueError(f"{name} index must not contain duplicate dates")
    if not data.index.is_monotonic_increasing:
        # Dated changes describe chronology, never the caller's physical row storage.
        return data.sort_index(kind='stable')
    return data


def _align_unit_notional(units: pd.DataFrame,
                         unit_notional: pd.DataFrame
                         ) -> pd.DataFrame:
    missing_columns = units.columns.difference(unit_notional.columns)
    if not missing_columns.empty:
        raise ValueError(
            f"unit_notional is missing unit columns: {missing_columns.to_list()}"
        )
    return unit_notional.reindex(index=units.index, columns=units.columns)


def _validate_vols_alignment(input_weights: pd.DataFrame,
                             vols: pd.DataFrame
                             ) -> None:
    if not vols.index.equals(input_weights.index):
        raise ValueError("vols index must exactly match input_weights index")
    if not vols.columns.equals(input_weights.columns):
        raise ValueError("vols columns must exactly match input_weights columns and order")
    if vols.lt(0.0).any(axis=None):
        raise ValueError("vols must not contain negative values")


def _divide_by_denominator(traded_notional: pd.DataFrame,
                           denominator: pd.Series,
                           denominator_name: str
                           ) -> pd.DataFrame:
    denominator = denominator.reindex(traded_notional.index)
    zero_denominator = denominator.eq(0.0)
    if zero_denominator.any():
        warnings.warn(
            f"{denominator_name} is zero on {int(zero_denominator.sum())} date(s); "
            "turnover is set to NaN on those dates.",
            RuntimeWarning,
            stacklevel=3,
        )
        denominator = denominator.mask(zero_denominator)
    return traded_notional.divide(denominator, axis=0)


def compute_turnover(
        computation_type: TurnoverComputationType = (
            TurnoverComputationType.EXECUTED_NOTIONAL_NAV
        ),
        units: Optional[pd.DataFrame] = None,
        unit_notional: Optional[pd.DataFrame] = None,
        nav: Optional[pd.Series] = None,
        input_weights: Optional[pd.DataFrame] = None,
        vols: Optional[pd.DataFrame] = None,
        ) -> pd.DataFrame:
    """Compute per-instrument two-sided turnover before resampling or rolling aggregation.

    Executed turnover values unit changes with the current date's unit notional. For cash
    securities this is normally the asset price. For derivatives it is the full contract
    notional, including multiplier and currency conversion. Target-weight turnover is retained
    as an explicit proxy for backtests that do not carry executed holdings. Volatility-normalized
    weight turnover is ``annualized_volatility[t] * abs(weight[t] - weight[t-1])``. It uses target
    weights rather than drifted holdings because it is a theoretical signal-turnover measure.
    Dated inputs are ordered chronologically on local objects before changes or alignment;
    non-dated indexes retain their supplied row order.

    Args:
        computation_type: Holdings and denominator convention. The default is executed traded
            notional divided by portfolio NAV.
        units: Executed units or contracts held on each date. Required by both executed modes.
        unit_notional: Current value of one unit or contract. Required by both executed modes.
        nav: Portfolio NAV in the same currency as ``unit_notional``. Required by
            ``EXECUTED_NOTIONAL_NAV``.
        input_weights: Requested target weights. Required by ``TARGET_WEIGHTS`` and
            ``VOLATILITY_NORMALIZED_WEIGHTS``.
        vols: Annualized fractional volatility for each target weight. Required by
            ``VOLATILITY_NORMALIZED_WEIGHTS`` and required to have exactly the same dated index
            after chronological ordering, columns, and column order as ``input_weights``.
            Warm-up NaNs are preserved.

    Returns:
        Per-instrument two-sided turnover on the input index, ordered chronologically for dated
        inputs. The first row is normally missing because no preceding holding is available.

    Raises:
        TypeError: If a required input is not a pandas object of the expected type.
        ValueError: If ``unit_notional`` does not contain every unit column, if ``vols`` is not
            exactly aligned or contains negative values, if an applicable dated input contains
            duplicate or ``NaT`` dates, or if the computation type is unsupported.
    """
    computation_type = TurnoverComputationType(computation_type)
    if computation_type == TurnoverComputationType.TARGET_WEIGHTS:
        input_weights = _require_frame(input_weights, 'input_weights')
        input_weights = _chronological_dated_input(input_weights, 'input_weights')
        return input_weights.diff(1).abs()

    if computation_type == TurnoverComputationType.VOLATILITY_NORMALIZED_WEIGHTS:
        input_weights = _require_frame(input_weights, 'input_weights')
        vols = _require_frame(vols, 'vols')
        input_weights = _chronological_dated_input(input_weights, 'input_weights')
        vols = _chronological_dated_input(vols, 'vols')
        _validate_vols_alignment(input_weights=input_weights, vols=vols)
        return input_weights.diff(1).abs().multiply(vols)

    units = _require_frame(units, 'units')
    unit_notional = _require_frame(unit_notional, 'unit_notional')
    units = _chronological_dated_input(units, 'units')
    unit_notional = _chronological_dated_input(unit_notional, 'unit_notional')
    unit_notional = _align_unit_notional(units=units, unit_notional=unit_notional)
    if computation_type == TurnoverComputationType.EXECUTED_NOTIONAL_NAV:
        if not isinstance(nav, pd.Series):
            raise TypeError(
                "nav must be a pandas Series for EXECUTED_NOTIONAL_NAV turnover"
            )
        nav = _chronological_dated_input(nav, 'nav')

    traded_notional = units.diff(1).abs().multiply(unit_notional)

    if computation_type == TurnoverComputationType.EXECUTED_NOTIONAL_NAV:
        return _divide_by_denominator(
            traded_notional=traded_notional,
            denominator=nav,
            denominator_name='nav',
        )

    if computation_type == TurnoverComputationType.EXECUTED_NOTIONAL_GROSS:
        gross_exposure = units.multiply(unit_notional).abs().sum(axis=1, min_count=1)
        return _divide_by_denominator(
            traded_notional=traded_notional,
            denominator=gross_exposure,
            denominator_name='gross exposure',
        )

    raise ValueError(f"unsupported turnover computation type: {computation_type}")
