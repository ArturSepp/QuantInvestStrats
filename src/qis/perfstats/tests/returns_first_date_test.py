"""Regression coverage for ``first_date`` return-to-NAV initialization.

The public ``returns_to_nav`` API documents ``first_date`` as the date on which a NAV starts at
one. These tests use short, fully observed return paths so the cutoff behavior can be separated
from ragged-start and missing-data policies. Expected NAVs are calculated directly from gross
return factors, without calling another package conversion function.
"""
# packages
from typing import cast
import pandas as pd

# qis
from qis.perfstats.returns import returns_to_nav


# =============================================================================
# Shared deterministic references
# =============================================================================
_DATES = pd.date_range('2024-01-01', periods=4, freq='D')
_FIRST_DATE = cast(pd.Timestamp, pd.Timestamp('2024-01-02'))


def _make_series_returns() -> pd.Series:
    """Create a fully observed return path spanning the initialization cutoff.

    Returns:
        A new Series containing two returns on or before the cutoff and two after it.
    """
    return pd.Series(
        [0.10, 0.20, -0.10, 0.05],
        index=_DATES,
        name='asset_a',
    )


def _make_frame_returns() -> pd.DataFrame:
    """Create two fully observed return paths for DataFrame ownership coverage.

    Returns:
        A new two-column DataFrame spanning the initialization cutoff.
    """
    return pd.DataFrame({
        'asset_a': [0.10, 0.20, -0.10, 0.05],
        'asset_b': [0.05, -0.02, 0.04, 0.01],
    }, index=_DATES)


def _expected_series_nav() -> pd.Series:
    """Return the independently compounded Series reference.

    Returns:
        A NAV that is one through the inclusive cutoff, then compounds later returns.
    """
    return pd.Series(
        [1.0, 1.0, 0.90, 0.945],
        index=_DATES,
        name='asset_a',
    )


def _expected_frame_nav() -> pd.DataFrame:
    """Return independently compounded references for both DataFrame columns.

    Returns:
        A two-column NAV frame initialized at one through the inclusive cutoff.
    """
    return pd.DataFrame({
        'asset_a': [1.0, 1.0, 0.90, 0.945],
        'asset_b': [1.0, 1.0, 1.04, 1.0504],
    }, index=_DATES)


# =============================================================================
# Cutoff semantics
# =============================================================================
def test_returns_to_nav_honors_first_date_with_default_initialization() -> None:
    """Start NAV at one on an explicitly supplied ``first_date`` under API defaults.

    An explicit cutoff should determine initialization even though ``init_period`` has a
    non-``None`` default. Returns through 2024-01-02 are therefore excluded, after which direct
    compounding gives ``1.0 * 0.90 = 0.90`` and ``0.90 * 1.05 = 0.945``.
    """
    returns = _make_series_returns()

    actual_nav = returns_to_nav(
        returns=returns,
        first_date=_FIRST_DATE,
        ffill_between_nans=False,
    )

    assert isinstance(actual_nav, pd.Series)
    pd.testing.assert_series_equal(actual_nav, _expected_series_nav())


# =============================================================================
# Caller ownership
# =============================================================================
def test_returns_to_nav_first_date_does_not_mutate_series() -> None:
    """Preserve a caller-owned Series while applying the explicit cutoff.

    ``init_period=None`` reaches the existing ``first_date`` path, allowing this regression to
    isolate ownership from the separate default-precedence question. Zero-valued initialization
    observations belong only to the derived NAV calculation; the supplied returns may be reused
    by a caller and must remain unchanged.
    """
    returns = _make_series_returns()
    original_returns = returns.copy(deep=True)

    actual_nav = returns_to_nav(
        returns=returns,
        init_period=None,
        first_date=_FIRST_DATE,
        ffill_between_nans=False,
    )

    # Check the numerical result before ownership so a failure identifies which contract broke.
    assert isinstance(actual_nav, pd.Series)
    pd.testing.assert_series_equal(actual_nav, _expected_series_nav())
    pd.testing.assert_series_equal(returns, original_returns)


def test_returns_to_nav_first_date_does_not_mutate_frame() -> None:
    """Preserve every DataFrame column while applying the explicit cutoff.

    Fully observed columns avoid mixing caller ownership with the unresolved policy for a
    later-starting asset's leading NaNs. Asset A compounds to ``0.945`` and asset B independently
    compounds to ``1.04 * 1.01 = 1.0504`` after both are initialized at one.
    """
    returns = _make_frame_returns()
    original_returns = returns.copy(deep=True)

    actual_nav = returns_to_nav(
        returns=returns,
        init_period=None,
        first_date=_FIRST_DATE,
        ffill_between_nans=False,
    )

    # The output calculation is pinned separately from exact preservation of the caller's frame.
    assert isinstance(actual_nav, pd.DataFrame)
    pd.testing.assert_frame_equal(actual_nav, _expected_frame_nav())
    pd.testing.assert_frame_equal(returns, original_returns)
