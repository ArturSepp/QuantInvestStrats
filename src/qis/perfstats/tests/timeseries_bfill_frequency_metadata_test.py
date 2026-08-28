"""Regression tests for ``bfill_timeseries`` output-frequency metadata.

The public ``freq`` argument defines the returned date grid, so a regular result must advertise
that same cadence through ``DatetimeIndex.freq`` regardless of input construction or row order.
These tests separate metadata from numerical behavior: literal return expectations pin regular,
short, and gapped grids, while a mixed price panel protects finite, ragged, absent, fallback, and
off-grid values simultaneously. They also cover Series/DataFrame shape, nullable returns, labels,
column order, warnings, and caller ownership.
"""

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset

from qis.perfstats.timeseries_bfill import bfill_timeseries


# =============================================================================
# Shared deterministic timelines, labels, and comparison settings
# =============================================================================

_DAILY_DATES = pd.date_range('2024-01-01', periods=6, freq='D')
_TWO_DAY_DATES = pd.date_range('2024-01-01', periods=3, freq='2D')

_ABSENT_ASSET = 'Absent'
_FALLBACK_ASSET = 'Fallback'
_FINITE_ASSET = 'Finite'
_RAGGED_ASSET = 'Ragged'
_SERIES_NAME = 'Asset A'
_SHARED_ASSET = 'Shared'

_TOLERANCE = 1.0e-12

_FloatDtype = Literal['float64', 'Float64']


# =============================================================================
# Independent return-panel fixtures and metadata assertion
# =============================================================================

def _astype_float(frame: pd.DataFrame, dtype: _FloatDtype) -> pd.DataFrame:
    """Apply one of the two explicitly supported floating-point test dtypes.

    Args:
        frame: Return panel to convert.
        dtype: NumPy-backed or pandas nullable floating-point dtype.

    Returns:
        Converted return panel.
    """
    if dtype == 'Float64':
        return frame.astype(pd.Float64Dtype())
    return frame.astype(np.float64)


def _make_regular_return_panels(dtype: _FloatDtype) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct regular providers without stored frequency metadata.

    Args:
        dtype: pandas dtype applied independently to both provider frames.

    Returns:
        Newer and older frames containing finite, ragged, fallback, and absent histories.
    """
    older = _astype_float(pd.DataFrame(
        {
            _ABSENT_ASSET: (np.nan, np.nan, np.nan),
            _RAGGED_ASSET: (np.nan, 0.40, 0.50),
            _FALLBACK_ASSET: (0.60, 0.70, 0.80),
            _FINITE_ASSET: (0.10, 0.20, 0.30),
        },
        index=pd.DatetimeIndex(_DAILY_DATES[:3].to_numpy()),
    ), dtype)
    newer = _astype_float(pd.DataFrame(
        {
            _ABSENT_ASSET: (np.nan, np.nan, np.nan),
            _RAGGED_ASSET: (0.60, np.nan, 0.80),
            _FALLBACK_ASSET: (np.nan, np.nan, np.nan),
            _FINITE_ASSET: (0.40, 0.50, 0.60),
        },
        index=pd.DatetimeIndex(_DAILY_DATES[3:].to_numpy()),
    ), dtype)
    return newer, older


def _make_expected_return_panel(dtype: _FloatDtype) -> pd.DataFrame:
    """Return the literal six-date splice for the regular mixed panel.

    Args:
        dtype: pandas dtype applied to the expected frame.

    Returns:
        Complete expected values on a daily index carrying canonical metadata.
    """
    return _astype_float(pd.DataFrame(
        {
            _ABSENT_ASSET: (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
            _RAGGED_ASSET: (np.nan, 0.40, 0.50, 0.60, np.nan, 0.80),
            _FALLBACK_ASSET: (0.60, 0.70, 0.80, np.nan, np.nan, np.nan),
            _FINITE_ASSET: (0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
        },
        index=_DAILY_DATES,
    ), dtype)


def _assert_requested_frequency(
        actual: pd.Series | pd.DataFrame,
        freq: str,
) -> None:
    """Assert that a pandas result advertises its requested regular grid.

    Args:
        actual: Result whose index metadata is under test.
        freq: Frequency passed to ``bfill_timeseries``.
    """
    assert isinstance(actual.index, pd.DatetimeIndex)
    assert actual.index.freq == to_offset(freq)


# =============================================================================
# Regular Series and DataFrame metadata
# =============================================================================

@pytest.mark.parametrize(
    ('older_positions', 'strip_frequency'),
    (
        pytest.param((0, 1, 2), False, id='sorted-with-metadata'),
        pytest.param((0, 1, 2), True, id='sorted-without-metadata'),
        pytest.param((2, 1, 0), False, id='reversed'),
        pytest.param((0, 2, 1), False, id='interior-shuffled'),
    ),
)
def test_bfill_timeseries_sets_daily_frequency_for_regular_series(
        older_positions: tuple[int, ...],
        strip_frequency: bool,
) -> None:
    """Return the same canonical daily Series for every equivalent input representation.

    The independent result is the exact chronological concatenation ``[1, 2, 3, 4, 5, 6]``.
    Stored input frequency and row order may change neither those values nor the output's
    requested daily metadata.

    Args:
        older_positions: Positional permutation applied to the older provider.
        strip_frequency: Whether to reconstruct its index without pandas frequency metadata.
    """
    older = pd.Series((1.0, 2.0, 3.0), index=_DAILY_DATES[:3], name='Older provider')
    newer = pd.Series((4.0, 5.0, 6.0), index=_DAILY_DATES[3:], name=_SERIES_NAME)
    older = older.iloc[pd.Index(older_positions)]
    if strip_frequency:
        older.index = pd.DatetimeIndex(older.index.to_numpy())
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected = pd.Series(
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        index=_DAILY_DATES,
        name=_SERIES_NAME,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual = bfill_timeseries(df_newer=newer, df_older=older, freq='D')

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    _assert_requested_frequency(actual, 'D')
    pd.testing.assert_series_equal(older, original_older, check_exact=True)
    pd.testing.assert_series_equal(newer, original_newer, check_exact=True)


@pytest.mark.parametrize('dtype', ('float64', 'Float64'))
def test_bfill_timeseries_sets_daily_frequency_for_regular_mixed_dataframe(
        dtype: _FloatDtype,
) -> None:
    """Canonicalize metadata without changing any mixed-panel column state.

    Args:
        dtype: NumPy-backed or pandas nullable dtype used for the same return panel.
    """
    newer, older = _make_regular_return_panels(dtype=dtype)
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual = bfill_timeseries(df_newer=newer, df_older=older, freq='D')

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(
        actual,
        _make_expected_return_panel(dtype=dtype),
        check_exact=True,
    )
    _assert_requested_frequency(actual, 'D')
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)
    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)


def test_bfill_timeseries_sets_daily_frequency_for_regular_dataframe() -> None:
    """Canonicalize a same-schema DataFrame whose regular indexes lack metadata."""
    older = pd.DataFrame(
        {
            _RAGGED_ASSET: (np.nan, 2.0, 3.0),
            _FINITE_ASSET: (1.0, 2.0, 3.0),
        },
        index=pd.DatetimeIndex(_DAILY_DATES[:3].to_numpy()),
    )
    newer = pd.DataFrame(
        {
            _RAGGED_ASSET: (4.0, np.nan, 6.0),
            _FINITE_ASSET: (4.0, 5.0, 6.0),
        },
        index=pd.DatetimeIndex(_DAILY_DATES[3:].to_numpy()),
    )
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected = pd.DataFrame(
        {
            _RAGGED_ASSET: (np.nan, 2.0, 3.0, 4.0, np.nan, 6.0),
            _FINITE_ASSET: (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        },
        index=_DAILY_DATES,
    )

    actual = bfill_timeseries(df_newer=newer, df_older=older, freq='D')

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    _assert_requested_frequency(actual, 'D')
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)
    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)


def test_bfill_timeseries_sets_multiday_frequency_for_regular_series() -> None:
    """Advertise a two-day cadence when regular input indexes carry no metadata."""
    older = pd.Series(
        (1.0, 2.0),
        index=pd.DatetimeIndex(_TWO_DAY_DATES[:2].to_numpy()),
        name='Older provider',
    )
    newer = pd.Series(
        (3.0,),
        index=pd.DatetimeIndex(_TWO_DAY_DATES[2:].to_numpy()),
        name=_SERIES_NAME,
    )
    expected = pd.Series((1.0, 2.0, 3.0), index=_TWO_DAY_DATES, name=_SERIES_NAME)

    actual = bfill_timeseries(df_newer=newer, df_older=older, freq='2D')

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    _assert_requested_frequency(actual, '2D')


# =============================================================================
# Existing short and gapped expansion controls
# =============================================================================

def test_bfill_timeseries_retains_frequency_for_single_observation() -> None:
    """Keep canonical daily metadata when pandas cannot infer a one-row cadence."""
    older = pd.Series((1.0,), index=_DAILY_DATES[[0]], name='Older provider')
    newer = pd.Series((2.0,), index=_DAILY_DATES[[0]], name=_SERIES_NAME)
    expected_index = pd.date_range(_DAILY_DATES[0], periods=1, freq='D')
    expected = pd.Series((2.0,), index=expected_index, name=_SERIES_NAME)

    actual = bfill_timeseries(df_newer=newer, df_older=older, freq='D')

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    _assert_requested_frequency(actual, 'D')


def test_bfill_timeseries_retains_frequency_for_gapped_observations() -> None:
    """Keep canonical daily metadata when existing expansion inserts missing dates."""
    older = pd.Series((1.0, 3.0), index=_DAILY_DATES[[0, 2]], name='Older provider')
    newer = pd.Series((5.0,), index=_DAILY_DATES[[4]], name=_SERIES_NAME)
    expected = pd.Series(
        (1.0, np.nan, 3.0, np.nan, 5.0),
        index=_DAILY_DATES[:5],
        name=_SERIES_NAME,
    )

    actual = bfill_timeseries(df_newer=newer, df_older=older, freq='D')

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    _assert_requested_frequency(actual, 'D')


# =============================================================================
# Mixed off-grid price interaction
# =============================================================================

@pytest.mark.parametrize(
    ('older_positions', 'newer_positions'),
    (
        pytest.param((0, 1), (0, 1), id='sorted'),
        pytest.param((1, 0), (1, 0), id='reversed'),
    ),
)
def test_bfill_timeseries_preserves_mixed_off_grid_prices_and_frequency(
        older_positions: tuple[int, ...],
        newer_positions: tuple[int, ...],
) -> None:
    """Retain every accepted price state while pinning business-day metadata.

    Saturday's shared and fallback prices carry into Monday, fallback remains 55 on Wednesday,
    and the absent asset remains missing. The coincident ragged price anchors Tuesday at 210
    before reaching 231 on Wednesday. These literal values ensure metadata work retains every
    price state while assigning the requested frequency.

    Args:
        older_positions: Positional permutation applied to the older price provider.
        newer_positions: Positional permutation applied to the newer price provider.
    """
    older = pd.DataFrame(
        {
            _RAGGED_ASSET: (np.nan, 210.0),
            _FALLBACK_ASSET: (50.0, 55.0),
            _SHARED_ASSET: (100.0, 110.0),
        },
        index=pd.DatetimeIndex(('2024-01-06', '2024-01-09')),
    ).iloc[list(older_positions)]
    newer = pd.DataFrame(
        {
            _ABSENT_ASSET: (np.nan, np.nan),
            _RAGGED_ASSET: (210.0, 231.0),
            _FALLBACK_ASSET: (np.nan, np.nan),
            _SHARED_ASSET: (110.0, 121.0),
        },
        index=pd.DatetimeIndex(('2024-01-09', '2024-01-10')),
    ).iloc[list(newer_positions)]
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected = pd.DataFrame(
        {
            _ABSENT_ASSET: (np.nan, np.nan, np.nan),
            _RAGGED_ASSET: (np.nan, 210.0, 231.0),
            _FALLBACK_ASSET: (50.0, 55.0, 55.0),
            _SHARED_ASSET: (100.0, 110.0, 121.0),
        },
        index=pd.bdate_range('2024-01-08', '2024-01-10'),
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual = bfill_timeseries(
            df_newer=newer,
            df_older=older,
            freq='B',
            is_prices=True,
        )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    _assert_requested_frequency(actual, 'B')
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)
    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)
