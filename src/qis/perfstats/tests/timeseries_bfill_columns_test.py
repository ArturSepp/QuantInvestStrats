"""Regression coverage for absent older columns in price backfills.

``bfill_timeseries`` defines its DataFrame output schema from the newer provider. When a newer
price column is entirely missing and the older provider has no matching column, there is no price
history from which to construct a NAV. The column should therefore remain entirely missing rather
than raising during terminal-value lookup or receiving an invented price.

The deterministic fixtures pair that boundary with an ordinary shared asset. Expected shared
prices are calculated directly by scaling the complete older path to the newer provider's first
price; no QIS return or NAV helper is used to construct the reference. A separate control retains
the established behavior when the older provider does contain the missing newer history. An
interaction regression combines the absent-column boundary with an off-grid weekend price so
that accepted frequency-conversion behavior cannot be lost while aligning columns.
"""

import warnings

import numpy as np
import pandas as pd

# qis
from qis.perfstats.timeseries_bfill import bfill_timeseries


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_DATES = pd.DatetimeIndex(
    (
        '2024-01-01',
        '2024-01-02',
        '2024-01-03',
        '2024-01-04',
        '2024-01-05',
        '2024-01-06',
    )
)

_ABSENT_ASSET = 'Absent Asset'
_FALLBACK_ASSET = 'Fallback Asset'
_SHARED_ASSET = 'Shared Asset'

_OLDER_SHARED_PRICES = (100.0, 110.0, 121.0)
_NEWER_SHARED_PRICES = (150.0, 165.0, 181.5)
_TOLERANCE = 1e-12


def _price_histories_with_absent_older_column() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a shared price history and an absent all-missing newer column.

    Returns:
        Older and newer price DataFrames. The newer column order deliberately places the absent
        asset before the shared asset so the output-order contract is visible.
    """
    older = pd.DataFrame(
        {_SHARED_ASSET: _OLDER_SHARED_PRICES},
        index=_DATES[:3],
    )
    newer = pd.DataFrame(
        {
            _ABSENT_ASSET: np.full(3, np.nan, dtype=float),
            _SHARED_ASSET: _NEWER_SHARED_PRICES,
        },
        index=_DATES[3:],
    )
    return older, newer


# =============================================================================
# Absent-column regression
# =============================================================================

def test_bfill_prices_preserves_absent_all_missing_newer_column() -> None:
    """Preserve the newer schema when neither provider has an asset's prices.

    The absent asset has no observable terminal value in either provider, so all six expected
    values are missing. The shared asset's older terminal price is 121 and its newer initial price
    is 150. Scaling every older price by ``150 / 121`` gives the independent prefix
    ``[100 * 150 / 121, 110 * 150 / 121, 150]``; the complete newer path then remains
    ``[150, 165, 181.5]``.

    The call must be warning-free, retain the newer order ``[Absent Asset, Shared Asset]``, and
    leave both caller-owned frames unchanged.
    """
    older, newer = _price_histories_with_absent_older_column()
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    scale = _NEWER_SHARED_PRICES[0] / _OLDER_SHARED_PRICES[-1]
    expected = pd.DataFrame(
        {
            _ABSENT_ASSET: np.full(6, np.nan, dtype=float),
            _SHARED_ASSET: (
                _OLDER_SHARED_PRICES[0] * scale,
                _OLDER_SHARED_PRICES[1] * scale,
                _OLDER_SHARED_PRICES[2] * scale,
                *_NEWER_SHARED_PRICES,
            ),
        },
        index=_DATES,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual = bfill_timeseries(
            df_newer=newer,
            df_older=older,
            freq='D',
            is_prices=True,
        )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        check_freq=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(older, original_older)
    pd.testing.assert_frame_equal(newer, original_newer)


# =============================================================================
# Accepted off-grid price interaction
# =============================================================================

def test_bfill_prices_preserves_absent_column_with_off_grid_shared_price() -> None:
    """Preserve both newer columns while carrying an off-grid shared price.

    The shared asset's Saturday price of 100 lies outside the requested business-day grid but is
    the latest available level for Monday. Tuesday and Wednesday are explicitly anchored at 110
    and 121 by the providers. The absent asset has no observation in either provider, so its
    independently expected business-day path is entirely missing.

    This mixed boundary guards two accepted invariants in one public call: frequency conversion
    must carry Saturday's price into Monday, and terminal-value alignment must retain the newer
    ``[Absent Asset, Shared Asset]`` schema without inventing an absent price. The call must also
    remain warning-free and preserve both caller-owned frames.
    """
    older = pd.DataFrame(
        {_SHARED_ASSET: (100.0, 110.0)},
        index=pd.DatetimeIndex(('2024-01-06', '2024-01-09')),
    )
    newer = pd.DataFrame(
        {
            _ABSENT_ASSET: np.full(2, np.nan, dtype=float),
            _SHARED_ASSET: (110.0, 121.0),
        },
        index=pd.DatetimeIndex(('2024-01-09', '2024-01-10')),
    )
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected = pd.DataFrame(
        {
            _ABSENT_ASSET: np.full(3, np.nan, dtype=float),
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
        check_freq=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(older, original_older)
    pd.testing.assert_frame_equal(newer, original_newer)


# =============================================================================
# Existing older-provider fallback control
# =============================================================================

def test_bfill_prices_retains_available_older_fallback() -> None:
    """Continue using older prices when an all-missing newer asset has a match.

    The fallback asset's older prices are ``[50, 55, 60.5]``. Its newer history is entirely
    missing, so the established price behavior carries the terminal 60.5 level over the three
    newer dates. The shared asset provides the same independently scaled control as the absent
    column regression. This distinguishes a genuinely absent history from an available older
    history and guards the existing terminal fallback while checking input ownership.
    """
    older, newer = _price_histories_with_absent_older_column()
    older = pd.DataFrame(
        {
            _FALLBACK_ASSET: (50.0, 55.0, 60.5),
            _SHARED_ASSET: _OLDER_SHARED_PRICES,
        },
        index=older.index,
    )
    newer = newer.rename(columns={_ABSENT_ASSET: _FALLBACK_ASSET})
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    scale = _NEWER_SHARED_PRICES[0] / _OLDER_SHARED_PRICES[-1]
    expected = pd.DataFrame(
        {
            _FALLBACK_ASSET: (50.0, 55.0, 60.5, 60.5, 60.5, 60.5),
            _SHARED_ASSET: (
                _OLDER_SHARED_PRICES[0] * scale,
                _OLDER_SHARED_PRICES[1] * scale,
                _OLDER_SHARED_PRICES[2] * scale,
                *_NEWER_SHARED_PRICES,
            ),
        },
        index=_DATES,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual = bfill_timeseries(
            df_newer=newer,
            df_older=older,
            freq='D',
            is_prices=True,
        )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        check_freq=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(older, original_older)
    pd.testing.assert_frame_equal(newer, original_newer)
