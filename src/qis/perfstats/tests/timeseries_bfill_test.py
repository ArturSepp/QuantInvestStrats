"""Regression coverage for time-series backfill transition boundaries.

``bfill_timeseries`` should retain every older observation strictly before the newer history
begins and let the newer provider win from its first valid observation onward. The deterministic
fixtures below distinguish adjacent histories from overlapping histories so a positional slice
cannot silently discard the older provider's final return.

Price expectations are calculated without a QIS return or NAV helper. The complete older price
path is multiplied by the single scale factor that joins its terminal level to the newer
provider's initial level; the newer price path must then remain exactly unchanged.
"""

import pandas as pd

# qis
from qis.perfstats.timeseries_bfill import bfill_timeseries


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_ADJACENT_DATES = pd.bdate_range('2024-01-01', periods=6)
_OVERLAP_DATES = pd.bdate_range('2024-02-01', periods=7)

_ASSET_NAME = 'Asset A'
_TOLERANCE = 1e-12


def _adjacent_return_histories() -> tuple[pd.Series, pd.Series]:
    """Create older and newer return histories on consecutive business dates.

    Returns:
        Older and newer named Series with no shared date and no calendar gap.
    """
    older = pd.Series(
        (0.00, 0.10, -0.05),
        index=_ADJACENT_DATES[:3],
        name='Older provider',
    )
    newer = pd.Series(
        (0.20, 0.03, -0.02),
        index=_ADJACENT_DATES[3:],
        name=_ASSET_NAME,
    )
    return older, newer


def _adjacent_price_histories() -> tuple[pd.Series, pd.Series]:
    """Create adjacent price histories with independently visible returns.

    Returns:
        Older and newer named Series whose consecutive returns are 10% and -10%.
    """
    older = pd.Series(
        (100.0, 110.0, 99.0),
        index=_ADJACENT_DATES[:3],
        name='Older provider',
    )
    newer = pd.Series(
        (120.0, 132.0, 118.8),
        index=_ADJACENT_DATES[3:],
        name=_ASSET_NAME,
    )
    return older, newer


# =============================================================================
# Adjacent-history boundary regressions
# =============================================================================

def test_bfill_timeseries_preserves_the_final_older_return() -> None:
    """Retain all supplied returns when provider histories are adjacent.

    The older provider ends one business day before the newer provider starts, so there is no
    overlapping observation to remove. The independently expected result is therefore the exact
    chronological concatenation ``[0.00, 0.10, -0.05, 0.20, 0.03, -0.02]``. In particular, the
    January 3 return must remain -5% rather than being discarded and replaced by a forward-filled
    January 2 return.

    The same values are exercised through Series and DataFrame inputs. The output should use the
    newer Series name, and neither public input shape may be modified in place.
    """
    older, newer = _adjacent_return_histories()
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected = pd.Series(
        (0.00, 0.10, -0.05, 0.20, 0.03, -0.02),
        index=_ADJACENT_DATES,
        name=_ASSET_NAME,
    )

    series_result = bfill_timeseries(df_newer=newer, df_older=older, freq='B')
    frame_result = bfill_timeseries(
        df_newer=newer.to_frame(),
        df_older=older.rename(_ASSET_NAME).to_frame(),
        freq='B',
    )

    assert isinstance(series_result, pd.Series)
    assert isinstance(frame_result, pd.DataFrame)
    pd.testing.assert_series_equal(series_result, expected)
    pd.testing.assert_frame_equal(frame_result, expected.to_frame())
    pd.testing.assert_series_equal(older, original_older)
    pd.testing.assert_series_equal(newer, original_newer)


def test_bfill_timeseries_preserves_adjacent_older_price_returns() -> None:
    """Scale the complete older path while preserving the newer price history.

    The older path ends at 99 and the newer path starts at 120, so the independent splice factor
    is ``120 / 99``. Scaling all three older prices produces approximately ``[121.212121,
    133.333333, 120]`` and preserves both older returns: +10% followed by -10%. The newer prices
    ``[120, 132, 118.8]`` must remain exact after the join.
    """
    older, newer = _adjacent_price_histories()
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    scale = 120.0 / 99.0
    expected = pd.Series(
        (100.0 * scale, 110.0 * scale, 120.0, 120.0, 132.0, 118.8),
        index=_ADJACENT_DATES,
        name=_ASSET_NAME,
    )

    actual = bfill_timeseries(
        df_newer=newer,
        df_older=older,
        freq='B',
        is_prices=True,
    )

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_series_equal(actual.loc[newer.index], newer)
    pd.testing.assert_series_equal(older, original_older)
    pd.testing.assert_series_equal(newer, original_newer)


# =============================================================================
# Ordinary overlap precedence
# =============================================================================

def test_bfill_timeseries_uses_newer_dataframe_values_through_the_overlap() -> None:
    """Use only the older prefix and preserve newer column order across overlap.

    Both older columns run through February 7, while the newer provider begins February 5. The
    expected result takes only February 1-2 from the older frame and every newer value from
    February 5 onward. Reversing the newer column order also confirms that output labels and
    order follow the documented newer-panel contract.
    """
    older = pd.DataFrame(
        {
            'Asset A': (0.00, 0.01, 0.02, 0.03, 0.04),
            'Asset B': (0.00, -0.01, -0.02, -0.03, -0.04),
        },
        index=_OVERLAP_DATES[:5],
    )
    newer = pd.DataFrame(
        {
            'Asset B': (-0.20, -0.21, -0.22, -0.23, -0.24),
            'Asset A': (0.20, 0.21, 0.22, 0.23, 0.24),
        },
        index=_OVERLAP_DATES[2:],
    )
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected = pd.DataFrame(
        {
            'Asset B': (0.00, -0.01, -0.20, -0.21, -0.22, -0.23, -0.24),
            'Asset A': (0.00, 0.01, 0.20, 0.21, 0.22, 0.23, 0.24),
        },
        index=_OVERLAP_DATES,
    )

    actual = bfill_timeseries(df_newer=newer, df_older=older, freq='B')

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(older, original_older)
    pd.testing.assert_frame_equal(newer, original_newer)
