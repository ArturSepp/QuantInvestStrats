"""Regression tests for ``bfill_timeseries`` input-date ordering.

A provider history is a mapping from dates to values, so the order in which its rows arrive must
not change provider-boundary selection or missing-value filling. These tests compare deliberately
scrambled inputs with literal chronological expectations. They exercise both provider sides,
Series and DataFrame shapes, an ``ffill`` no-look-ahead boundary, labels, column order, caller
ownership, and the established price-splice calculation.
"""

import numpy as np
import pandas as pd

from qis.perfstats.timeseries_bfill import bfill_timeseries


# =============================================================================
# Shared deterministic timeline and labels
# =============================================================================

_DAILY_DATES = pd.date_range('2024-01-01', periods=6, freq='D')

_ASSET_NAME = 'Asset A'
_TOLERANCE = 1.0e-12


# =============================================================================
# Provider-boundary selection
# =============================================================================

def test_bfill_timeseries_selects_older_series_boundary_in_date_order() -> None:
    """Retain the valid older prefix regardless of its stored row order.

    The older provider contains January 1-3 plus an overlapping January 5 value deliberately set
    to 990%. The newer provider begins January 4 and must win thereafter. The independent splice
    is therefore exactly ``[10%, 20%, 30%, 40%, 50%, 60%]``; putting January 5 first in the older
    input must neither discard January 1-3 nor expose the overlapping 990% value.
    """
    older = pd.Series(
        (9.90, 0.10, 0.20, 0.30),
        index=_DAILY_DATES[[4, 0, 1, 2]],
        name='Older provider',
    )
    newer = pd.Series(
        (0.40, 0.50, 0.60),
        index=_DAILY_DATES[3:],
        name=_ASSET_NAME,
    )
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected = pd.Series(
        (0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
        index=_DAILY_DATES,
        name=_ASSET_NAME,
    )

    actual = bfill_timeseries(df_newer=newer, df_older=older, freq='D')

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected, check_exact=True, check_freq=False)
    pd.testing.assert_series_equal(older, original_older, check_exact=True)
    pd.testing.assert_series_equal(newer, original_newer, check_exact=True)


def test_bfill_timeseries_selects_older_dataframe_boundary_in_date_order() -> None:
    """Apply the chronological splice independently while preserving newer column order.

    Both older columns store their overlapping January 5 rows first. The expected frame takes
    January 1-3 from the older provider and January 4-6 from the newer provider. Reversed newer
    columns pin the output-label and column-order contract alongside the date-order regression.
    """
    older = pd.DataFrame(
        {
            'Asset A': (9.90, 0.10, 0.20, 0.30),
            'Asset B': (-9.90, -0.10, -0.20, -0.30),
        },
        index=_DAILY_DATES[[4, 0, 1, 2]],
    )
    newer = pd.DataFrame(
        {
            'Asset B': (-0.40, -0.50, -0.60),
            'Asset A': (0.40, 0.50, 0.60),
        },
        index=_DAILY_DATES[3:],
    )
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected = pd.DataFrame(
        {
            'Asset B': (-0.10, -0.20, -0.30, -0.40, -0.50, -0.60),
            'Asset A': (0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
        },
        index=_DAILY_DATES,
    )

    actual = bfill_timeseries(df_newer=newer, df_older=older, freq='D')

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_freq=False)
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)
    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)


def test_bfill_timeseries_selects_newer_boundary_in_date_order() -> None:
    """Preserve a trailing newer observation even when it is stored first.

    January 6 is an explicit missing observation rather than an absent date. The newer rows arrive
    as January 6, 4, 5, while their first observed return is January 4. Chronological slicing must
    retain January 6 as missing and produce the same six-date output as sorted input.
    """
    older = pd.Series(
        (0.10, 0.20, 0.30),
        index=_DAILY_DATES[:3],
        name='Older provider',
    )
    newer = pd.Series(
        (np.nan, 0.40, 0.50),
        index=_DAILY_DATES[[5, 3, 4]],
        name=_ASSET_NAME,
    )
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected = pd.Series(
        (0.10, 0.20, 0.30, 0.40, 0.50, np.nan),
        index=_DAILY_DATES,
        name=_ASSET_NAME,
    )

    actual = bfill_timeseries(df_newer=newer, df_older=older, freq='D')

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected, check_exact=True, check_freq=False)
    pd.testing.assert_series_equal(older, original_older, check_exact=True)
    pd.testing.assert_series_equal(newer, original_newer, check_exact=True)


# =============================================================================
# Chronological fill policy
# =============================================================================

def test_bfill_timeseries_ffill_does_not_look_ahead_on_unsorted_rows() -> None:
    """Fill January 2 from January 1 rather than the later January 3 return.

    The older rows arrive as January 1, 3, 2 with January 2 missing. A row-order forward fill
    incorrectly carries January 3's 30% return backward. Chronological filling instead gives the
    independently expected 10% on January 2 and preserves every supplied finite return.
    """
    older = pd.Series(
        (0.10, 0.30, np.nan),
        index=_DAILY_DATES[[0, 2, 1]],
        name='Older provider',
    )
    newer = pd.Series(
        (0.40, 0.50, 0.60),
        index=_DAILY_DATES[3:],
        name=_ASSET_NAME,
    )
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected = pd.Series(
        (0.10, 0.10, 0.30, 0.40, 0.50, 0.60),
        index=_DAILY_DATES,
        name=_ASSET_NAME,
    )

    actual = bfill_timeseries(
        df_newer=newer,
        df_older=older,
        freq='D',
        fill_method='ffill',
    )

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected, check_exact=True, check_freq=False)
    pd.testing.assert_series_equal(older, original_older, check_exact=True)
    pd.testing.assert_series_equal(newer, original_newer, check_exact=True)


# =============================================================================
# Existing price-splice convention
# =============================================================================

def test_bfill_timeseries_preserves_price_splice_for_unsorted_older_rows() -> None:
    """Keep the established price splice while normalizing its provider dates.

    Older prices of 100, 110, and 121 imply two independent 10% returns. Scaling that prefix by
    ``150 / 121`` joins it to the newer January 4 price without changing either return. The older
    January 5 value of 999 overlaps the newer provider and must remain unused.
    """
    older = pd.Series(
        (999.0, 100.0, 110.0, 121.0),
        index=_DAILY_DATES[[4, 0, 1, 2]],
        name='Older provider',
    )
    newer = pd.Series(
        (150.0, 165.0, 181.5),
        index=_DAILY_DATES[3:],
        name=_ASSET_NAME,
    )
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    scale = 150.0 / 121.0
    expected = pd.Series(
        (100.0 * scale, 110.0 * scale, 150.0, 150.0, 165.0, 181.5),
        index=_DAILY_DATES,
        name=_ASSET_NAME,
    )

    actual = bfill_timeseries(
        df_newer=newer,
        df_older=older,
        freq='D',
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
    pd.testing.assert_series_equal(actual.loc[newer.sort_index().index], newer.sort_index())
    pd.testing.assert_series_equal(older, original_older, check_exact=True)
    pd.testing.assert_series_equal(newer, original_newer, check_exact=True)
