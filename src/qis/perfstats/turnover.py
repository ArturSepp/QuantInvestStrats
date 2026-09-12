"""Two-sided portfolio-turnover computations independent of the reporting layer."""
from __future__ import annotations

import warnings
from enum import Enum
from typing import Optional

import pandas as pd


class TurnoverComputationType(str, Enum):
    """Select the holdings and denominator used to compute two-sided turnover.

    Attributes:
        TARGET_WEIGHTS: Absolute changes in target weights. This is a target-allocation proxy,
            not a reconstruction of executed trades.
        EXECUTED_NOTIONAL_NAV: Absolute changes in units valued at current unit notional and
            divided by portfolio NAV.
        EXECUTED_NOTIONAL_GROSS: Absolute changes in units valued at current unit notional and
            divided by current gross exposure.
    """

    TARGET_WEIGHTS = 'target_weights'
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


def _align_unit_notional(units: pd.DataFrame,
                         unit_notional: pd.DataFrame
                         ) -> pd.DataFrame:
    missing_columns = units.columns.difference(unit_notional.columns)
    if not missing_columns.empty:
        raise ValueError(
            f"unit_notional is missing unit columns: {missing_columns.to_list()}"
        )
    return unit_notional.reindex(index=units.index, columns=units.columns)


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
        ) -> pd.DataFrame:
    """Compute per-instrument two-sided turnover before resampling or rolling aggregation.

    Executed turnover values unit changes with the current date's unit notional. For cash
    securities this is normally the asset price. For derivatives it is the full contract
    notional, including multiplier and currency conversion. Target-weight turnover is retained
    as an explicit proxy for backtests that do not carry executed holdings.

    Args:
        computation_type: Holdings and denominator convention. The default is executed traded
            notional divided by portfolio NAV.
        units: Executed units or contracts held on each date. Required by both executed modes.
        unit_notional: Current value of one unit or contract. Required by both executed modes.
        nav: Portfolio NAV in the same currency as ``unit_notional``. Required by
            ``EXECUTED_NOTIONAL_NAV``.
        input_weights: Requested target weights. Required by ``TARGET_WEIGHTS``.

    Returns:
        Per-instrument two-sided turnover on the input index. The first row is normally missing
        because no preceding holding is available.

    Raises:
        TypeError: If a required input is not a pandas object of the expected type.
        ValueError: If ``unit_notional`` does not contain every unit column or the computation
            type is unsupported.
    """
    computation_type = TurnoverComputationType(computation_type)
    if computation_type == TurnoverComputationType.TARGET_WEIGHTS:
        input_weights = _require_frame(input_weights, 'input_weights')
        return input_weights.diff(1).abs()

    units = _require_frame(units, 'units')
    unit_notional = _require_frame(unit_notional, 'unit_notional')
    unit_notional = _align_unit_notional(units=units, unit_notional=unit_notional)
    traded_notional = units.diff(1).abs().multiply(unit_notional)

    if computation_type == TurnoverComputationType.EXECUTED_NOTIONAL_NAV:
        if not isinstance(nav, pd.Series):
            raise TypeError(
                "nav must be a pandas Series for EXECUTED_NOTIONAL_NAV turnover"
            )
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
