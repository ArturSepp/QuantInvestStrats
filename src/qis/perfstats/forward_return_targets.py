"""Realised forward returns on a common decision calendar with frozen factor exposures.

Over a window (t, u], raw return is the sum of end-labelled additive asset increments
on the asset's native grid. The explained return is the dot product of the latest
loadings available at t and the summed factor increments. The residual retains any
intercept. Callers supply the decision dates, increments and dated loading matrices.
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from qis.utils.df_freq import validate_calendar_index


class ForwardTargetStatus(str, Enum):
    """Availability of a target in one asset and decision window.

    Attributes:
        NATIVE_ENDPOINTS: A decision endpoint is off the native grid or absent from the panel.
        MISSING_RETURN: An aligned window lacks a finite asset increment.
        MISSING_SNAPSHOT: No loading snapshot was available at formation.
        MISSING_FACTOR_RETURN: A required factor lacks a complete window of increments.
        MISSING_LOADING: The selected snapshot lacks a required loading for this asset.
        OK: The target is available.
    """

    NATIVE_ENDPOINTS = 'native_endpoints'
    MISSING_RETURN = 'missing_return'
    MISSING_SNAPSHOT = 'missing_snapshot'
    MISSING_FACTOR_RETURN = 'missing_factor_return'
    MISSING_LOADING = 'missing_loading'
    OK = 'ok'


@dataclass(frozen=True)
class ForwardReturnAdjustment:
    """Inputs for one benchmark or multifactor adjustment.

    Attributes:
        factor_increments_by_freq: For each asset cadence, a frame of end-labelled additive
            increments, dates by K named factors. Use the units in which exposures were fitted.
        loadings_by_date: Availability timestamp to asset-by-factor matrix of signed
            exposures. The latest snapshot available at formation supplies the whole matrix.
            Rows are asset identifiers. The K column labels select and order the increments.
            For an EwmLinearModel fitted to K benchmarks, supply get_loadings_at_date(t).T.
            Missing cells in that snapshot propagate. A benchmark uses one factor column.
    """

    factor_increments_by_freq: Mapping[str, pd.DataFrame]
    loadings_by_date: Mapping[pd.Timestamp, pd.DataFrame]


_BASE_COLUMNS = ('horizon', 'formation_date', 'realisation_date', 'asset', 'cadence',
                 'raw', 'raw_status')


def _numeric_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f'{name} must be a DataFrame')
    if not frame.columns.is_unique or frame.columns.to_flat_index().hasnans:
        raise ValueError(f'{name} must have unique, nonmissing column labels')
    if any(dtype.kind in ('M', 'm') for dtype in frame.dtypes):
        raise ValueError(f'{name} must contain numeric increments or loadings')
    numeric = frame.apply(pd.to_numeric, errors='raise')
    if any(pd.api.types.is_complex_dtype(dtype) for dtype in numeric.dtypes):
        raise ValueError(f'{name} must contain real increments or loadings')
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    return pd.DataFrame(np.where(np.isfinite(values), values, np.nan),
                        index=frame.index, columns=frame.columns)


def _increment_panels(panels: Mapping[str, pd.DataFrame], decision_dates: pd.DatetimeIndex,
                      name: str) -> dict:
    result = {}
    for freq, frame in panels.items():
        if not isinstance(freq, str) or not freq or pd.tseries.frequencies.to_offset(freq).n <= 0:
            raise ValueError(f'{name} cadence must be positive, got {freq!r}')
        numeric = _numeric_frame(frame, name)
        validate_calendar_index(numeric, name)
        if not numeric.index.is_unique:
            raise ValueError(f'{name} dates must be unique')
        if len(numeric.index) and isinstance(numeric.index, pd.DatetimeIndex):
            if (numeric.index.tz is None) != (decision_dates.tz is None):
                raise ValueError(f'{name} and decision_dates must agree on timezone awareness')
        result[freq] = numeric
    return result


def _snapshot_schedule(loadings_by_date: Mapping[pd.Timestamp, pd.DataFrame],
                       decision_dates: pd.DatetimeIndex) -> dict:
    snapshots = {}
    for key, matrix in loadings_by_date.items():
        date = pd.Timestamp(key)
        if pd.isna(date) or (date.tz is None) != (decision_dates.tz is None):
            raise ValueError('snapshot dates must be valid and match decision timezone awareness')
        if date.tz is not None:
            date = date.tz_convert(decision_dates.tz)
        if date in snapshots:
            raise ValueError('duplicate snapshot availability date')
        matrix = _numeric_frame(matrix, 'loadings')
        if (not len(matrix.columns) or not matrix.index.is_unique
                or matrix.index.to_flat_index().hasnans):
            raise ValueError('loadings require factor columns and unique, nonmissing asset labels')
        snapshots[date] = matrix
    if not snapshots:
        return {}
    dates = pd.DatetimeIndex(sorted(snapshots))
    positions = dates.searchsorted(decision_dates, side='right') - 1
    return {opening: (dates[pos], snapshots[dates[pos]])
            for opening, pos in zip(decision_dates, positions) if pos >= 0}


def _window_total(panel: pd.DataFrame, ends: pd.DatetimeIndex,
                  columns: Optional[pd.Index] = None) -> np.ndarray:
    values = panel.reindex(index=ends, columns=columns).to_numpy()
    with np.errstate(over='raise'):
        return values.sum(axis=0)


def _adjusted_window(block: pd.DataFrame, name: str, ends: pd.DatetimeIndex, aligned: bool,
                     factors: Optional[pd.DataFrame], snapshot: tuple) -> None:
    date, matrix = snapshot
    block[f'{name}_snapshot_date'] = pd.DatetimeIndex([date] * len(block),
                                                      tz=block['formation_date'].dt.tz)
    explained = np.full(len(block), np.nan)
    reason = ForwardTargetStatus.MISSING_SNAPSHOT.value
    if matrix is not None:
        reason = ForwardTargetStatus.MISSING_FACTOR_RETURN.value
        if aligned and factors is not None:
            payoff = _window_total(factors, ends, matrix.columns)
            if np.isfinite(payoff).all():
                exposures = matrix.reindex(index=block['asset'])
                with np.errstate(over='raise'):
                    explained = exposures.to_numpy() @ payoff
                reason = np.where(np.isfinite(explained), ForwardTargetStatus.OK.value,
                                  ForwardTargetStatus.MISSING_LOADING.value)
    block[f'{name}_explained'] = explained
    with np.errstate(over='raise'):
        block[name] = block['raw'].to_numpy() - explained
    block[f'{name}_status'] = np.where(block['raw_status'] == ForwardTargetStatus.OK.value,
                                       reason, block['raw_status'])


def compute_forward_return_targets(
        decision_dates: pd.DatetimeIndex,
        asset_increments_by_freq: Mapping[str, pd.DataFrame],
        *,
        horizons: Sequence[int] = (1, 2, 3, 6),
        adjustments: Optional[Mapping[str, ForwardReturnAdjustment]] = None,
) -> pd.DataFrame:
    """Aggregate forward asset returns and deduct factor payoffs at formation exposures.

    The first decision sets the phase. For horizon h in decision steps, take consecutive
    windows from positions 0 to h, h to 2h, and so on. Each window requires both endpoint
    labels in the asset frame and on its native grid. Increments at formation are excluded.
    Both raw and factor sums require every expected increment, including interior dates.

    Args:
        decision_dates: Sorted, unique decision timestamps identifying formation and
            realisation. Irregular decision schedules are supported.
        asset_increments_by_freq: Native pandas frequency to end-labelled frame of additive
            increments (native end dates by assets). Asset columns match loading rows.
            Each asset has one cadence. Supply log
            returns, log excess returns, or another explicitly additive return convention.
        horizons: Distinct positive integer holding periods in decision steps.
        adjustments: Residual target name to factor increments and dated loading snapshots.
            The latest snapshot available at or before formation is held across the window.
            Every factor named by that snapshot requires a complete return.

    Returns:
        One row per scheduled window and asset, with horizon, formation_date, realisation_date,
        asset, cadence, raw and raw_status. Each adjustment k adds k, k_explained,
        k_snapshot_date and k_status. The explained component is available independently of a
        missing asset increment on an aligned grid. Residual status follows raw status, then
        missing snapshot, required factor return and asset loading, in that order.
        An insufficient decision history produces an empty table with the output columns.

    Warns:
        UserWarning: A cadence and horizon have no aligned window under the decision schedule.

    Raises:
        TypeError: If the supplied time panels or decision index have an invalid type.
        ValueError: If labels, dates, frequencies, horizons, adjustment names or data are invalid.
        FloatingPointError: If a cumulative return or an attribution exceeds float range.
    """
    if not isinstance(decision_dates, pd.DatetimeIndex):
        raise TypeError('decision_dates must be a DatetimeIndex')
    if (decision_dates.hasnans or not decision_dates.is_unique
            or not decision_dates.is_monotonic_increasing):
        raise ValueError('decision_dates must be ordered, unique and without NaT')
    horizons = tuple(horizons)
    if (not horizons or any(isinstance(h, (bool, np.bool_))
                            or not isinstance(h, (int, np.integer)) or h < 1 for h in horizons)
            or len(set(horizons)) != len(horizons)):
        raise ValueError('horizons must be distinct positive integers')

    assets = _increment_panels(asset_increments_by_freq, decision_dates, 'asset increments')
    names = [asset for frame in assets.values() for asset in frame.columns]
    if len(set(names)) != len(names):
        raise ValueError('an asset can belong to only one native cadence')
    columns = list(_BASE_COLUMNS)
    models = {}
    for name, model in (adjustments or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError('adjustment names must be nonempty strings')
        outputs = [name, f'{name}_explained', f'{name}_snapshot_date', f'{name}_status']
        if set(columns).intersection(outputs):
            raise ValueError(f'adjustment name {name!r} collides with an output column')
        columns.extend(outputs)
        if not isinstance(model, ForwardReturnAdjustment):
            raise TypeError('adjustments must contain ForwardReturnAdjustment values')
        factors = _increment_panels(model.factor_increments_by_freq, decision_dates,
                                    'factor increments')
        models[name] = (factors, _snapshot_schedule(model.loadings_by_date, decision_dates))

    blocks = []
    for h in horizons:
        aligned_freqs = set()
        for j in range(0, len(decision_dates) - h, h):
            opening, closing = decision_dates[j], decision_dates[j + h]
            for freq, frame in assets.items():
                if not len(frame.columns):
                    continue
                grid = pd.date_range(opening, closing, freq=freq)
                aligned = (len(grid) >= 2 and grid[0] == opening and grid[-1] == closing
                           and opening in frame.index and closing in frame.index)
                if aligned:
                    aligned_freqs.add(freq)
                    raw = _window_total(frame, grid[1:])
                    raw_status = np.where(np.isfinite(raw), ForwardTargetStatus.OK.value,
                                          ForwardTargetStatus.MISSING_RETURN.value)
                else:
                    raw = np.full(len(frame.columns), np.nan)
                    raw_status = ForwardTargetStatus.NATIVE_ENDPOINTS.value
                block = pd.DataFrame({'horizon': h, 'formation_date': opening,
                                      'realisation_date': closing, 'asset': frame.columns.to_list(),
                                      'cadence': freq, 'raw': raw, 'raw_status': raw_status})
                for name, (factors, snapshots) in models.items():
                    _adjusted_window(block, name, grid[1:], aligned, factors.get(freq),
                                     snapshots.get(opening, (pd.NaT, None)))
                blocks.append(block)
        for freq, frame in assets.items():
            if len(frame.columns) and freq not in aligned_freqs:
                warnings.warn(f'No aligned native window for cadence {freq!r}, horizon {h}',
                              UserWarning, stacklevel=2)
    if not blocks:
        return pd.DataFrame(columns=columns)
    return pd.concat(blocks, ignore_index=True)[columns]
