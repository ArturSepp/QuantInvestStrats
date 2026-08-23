"""Boundary coverage for converting returns to NAVs and back.

Financial panels commonly contain assets that began trading on different dates or temporarily
stopped reporting. These tests use small, hand-calculated examples to pin how those missing
observations interact with initialization, simple and log compounding, and optional gap filling.
"""
# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.perfstats.returns import returns_to_nav, to_returns, to_zero_first_nonnan_returns


# =============================================================================
# Shared fixtures and assertions
# =============================================================================
_DATES = pd.date_range('2024-01-01', periods=5, freq='D')
_TOLERANCE = 1e-12


def _make_ragged_returns() -> pd.DataFrame:
    """Create the standard ragged-start simple-return panel.

    Returns:
        A new DataFrame whose columns have different first-valid dates.
    """
    return pd.DataFrame({
        'early': [np.nan, 0.10, -0.05, 0.02, 0.03],
        'late': [np.nan, np.nan, np.nan, 0.20, -0.10],
    }, index=_DATES)


def _assert_frames_close(actual: pd.Series | pd.DataFrame,
                         expected: pd.DataFrame) -> None:
    """Assert that a result is a DataFrame and agrees within the test tolerance.

    Args:
        actual: Result returned under the library's Series-or-DataFrame annotation.
        expected: Independently calculated reference result.
    """
    # The production annotation cannot express that a DataFrame input produces a DataFrame.
    # Check that part of the contract at runtime while narrowing the type for static analysis.
    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        rtol=_TOLERANCE,
        atol=_TOLERANCE,
    )


def _assert_series_close(actual: pd.Series | pd.DataFrame,
                         expected: pd.Series) -> None:
    """Assert that a result is a Series and agrees within the test tolerance.

    Args:
        actual: Result returned under the library's Series-or-DataFrame annotation.
        expected: Independently calculated reference result.
    """
    # The production annotation cannot express that a Series input produces a Series.
    # Check that part of the contract at runtime while narrowing the type for static analysis.
    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(
        actual,
        expected,
        rtol=_TOLERANCE,
        atol=_TOLERANCE,
    )


# =============================================================================
# Return-convention round trips
# =============================================================================
def test_return_nav_round_trip_preserves_ragged_starts() -> None:
    """Preserve each column's first-valid date through a simple-return round trip.

    Each asset receives its own unit NAV on the observation before its first non-NaN return. The
    later-starting asset must therefore retain its leading NaNs instead of being aligned to the
    earlier asset. Explicit NAV values provide a reference independent of the conversion code.
    """
    returns = _make_ragged_returns()

    # Direct geometric compounding gives, for example, 1.10 * 0.95 = 1.045 for
    # early and 1.20 * 0.90 = 1.08 for late. No package calculation builds this reference.
    expected_nav = pd.DataFrame({
        'early': [1.0, 1.10, 1.045, 1.0659, 1.097877],
        'late': [np.nan, np.nan, 1.0, 1.20, 1.08],
    }, index=_DATES)

    # Disable filling so this test isolates initialization and compounding from gap policy.
    actual_nav = returns_to_nav(
        returns=returns,
        init_period=0,
        ffill_between_nans=False,
    )
    actual_returns = to_returns(
        prices=actual_nav,
        ffill_nans=False,
    )

    _assert_frames_close(actual_nav, expected_nav)
    _assert_frames_close(actual_returns, returns)


def test_log_return_nav_round_trip_preserves_ragged_starts() -> None:
    """Preserve ragged starts when NAVs are compounded from log returns.

    The fixture expresses the same economic path as the simple-return test using logarithms of
    gross return factors. The expected NAV remains a direct multiplication reference, while the
    reverse conversion must recover the original log returns for each asset independently.
    """
    log_returns = pd.DataFrame({
        'early': [np.nan, np.log(1.10), np.log(0.95), np.log(1.02), np.log(1.03)],
        'late': [np.nan, np.nan, np.nan, np.log(1.20), np.log(0.90)],
    }, index=_DATES)
    expected_nav = pd.DataFrame({
        'early': [1.0, 1.10, 1.045, 1.0659, 1.097877],
        'late': [np.nan, np.nan, 1.0, 1.20, 1.08],
    }, index=_DATES)

    # is_log_returns=True must be set in both directions; otherwise the two functions would
    # silently apply different return conventions and a numerically plausible path could result.
    actual_nav = returns_to_nav(
        returns=log_returns,
        init_period=0,
        ffill_between_nans=False,
        is_log_returns=True,
    )
    actual_log_returns = to_returns(
        prices=actual_nav,
        is_log_returns=True,
        ffill_nans=False,
    )

    _assert_frames_close(actual_nav, expected_nav)
    _assert_frames_close(actual_log_returns, log_returns)


# =============================================================================
# Missing-data boundaries
# =============================================================================
def test_returns_to_nav_controls_interior_gap_fill() -> None:
    """Apply the requested display policy to an interior missing return.

    Pandas' skip-NaN cumulative product continues compounding after a missing observation. With
    filling enabled, the NAV should carry its prior value across that observation; with filling
    disabled, only the missing date should remain NaN and the later compounded values should
    agree between both outputs.
    """
    returns = pd.Series(
        [np.nan, 0.10, np.nan, 0.02, 0.03],
        index=_DATES,
        name='asset',
    )

    # The path is 1.0, 1.10, a missing observation, 1.10 * 1.02, then * 1.03.
    expected_filled = pd.Series(
        [1.0, 1.10, 1.10, 1.122, 1.15566],
        index=_DATES,
        name='asset',
    )
    expected_unfilled = pd.Series(
        [1.0, 1.10, np.nan, 1.122, 1.15566],
        index=_DATES,
        name='asset',
    )

    filled_nav = returns_to_nav(
        returns=returns,
        init_period=0,
        ffill_between_nans=True,
    )
    unfilled_nav = returns_to_nav(
        returns=returns,
        init_period=0,
        ffill_between_nans=False,
    )

    _assert_series_close(filled_nav, expected_filled)
    _assert_series_close(unfilled_nav, expected_unfilled)


def test_returns_to_nav_preserves_trailing_nans_when_filling_gaps() -> None:
    """Fill interior gaps without extending an ended asset beyond its final return.

    Forward filling between valid NAV observations is useful for intermittent missing data, but
    applying an unrestricted forward fill would make an ended asset appear to keep reporting.
    This fixture combines one terminated asset with one later-starting live asset to pin both
    boundaries in the same panel.
    """
    returns = pd.DataFrame({
        'ended': [np.nan, 0.10, 0.02, np.nan, np.nan],
        'live': [np.nan, np.nan, 0.20, -0.10, 0.05],
    }, index=_DATES)
    expected_nav = pd.DataFrame({
        'ended': [1.0, 1.10, 1.122, np.nan, np.nan],
        'live': [np.nan, 1.0, 1.20, 1.08, 1.134],
    }, index=_DATES)

    # The fill option is deliberately enabled. Its contract is to fill only between the first
    # and last valid NAV, so the ended column's final two observations must remain missing.
    actual_nav = returns_to_nav(
        returns=returns,
        init_period=0,
        ffill_between_nans=True,
    )

    _assert_frames_close(actual_nav, expected_nav)


@pytest.mark.parametrize(
    ('init_period', 'expected_valid_nav'),
    [
        (0, [1.0, 1.10, 1.045, 1.0659, 1.097877]),
        (1, [np.nan, 1.0, 0.95, 0.969, 0.99807]),
    ],
    ids=['previous-period-initialization', 'first-valid-period-initialization'],
)
def test_returns_to_nav_preserves_all_nan_columns(
        init_period: int,
        expected_valid_nav: list[float]) -> None:
    """Avoid manufacturing a NAV observation for an asset with no return history.

    An all-NaN column represents an asset that never entered the observable sample. It has no
    first valid return and therefore no legitimate initialization date. Both supported
    initialization modes must leave that entire NAV column missing while continuing to compound
    an adjacent valid column normally. The same rule applies when the all-NaN input is a Series.

    Args:
        init_period: Whether NAV initialization occurs before or on the first valid return.
        expected_valid_nav: Independently compounded NAV for the adjacent valid asset.
    """
    returns = pd.DataFrame({
        'valid': [np.nan, 0.10, -0.05, 0.02, 0.03],
        'never_started': [np.nan, np.nan, np.nan, np.nan, np.nan],
    }, index=_DATES)
    original_returns = returns.copy(deep=True)
    expected_nav = pd.DataFrame({
        'valid': expected_valid_nav,
        'never_started': [np.nan, np.nan, np.nan, np.nan, np.nan],
    }, index=_DATES)

    # A fallback index for an empty column must not be mistaken for a real first-valid date.
    actual_nav = returns_to_nav(
        returns=returns,
        init_period=init_period,
        ffill_between_nans=False,
    )

    _assert_frames_close(actual_nav, expected_nav)
    pd.testing.assert_frame_equal(returns, original_returns)

    all_nan_series = returns['never_started']
    assert isinstance(all_nan_series, pd.Series)
    all_nan_series = all_nan_series.copy(deep=True)
    expected_series_nav = all_nan_series.copy(deep=True)
    actual_series_nav = returns_to_nav(
        returns=all_nan_series,
        init_period=init_period,
        ffill_between_nans=False,
    )
    _assert_series_close(actual_series_nav, expected_series_nav)
    pd.testing.assert_series_equal(all_nan_series, expected_series_nav)


@pytest.mark.parametrize('init_period', [0, 1])
def test_zero_first_nonnan_returns_preserves_all_nan_inputs(init_period: int) -> None:
    """Leave Series and DataFrame inputs unchanged when no valid return exists.

    Args:
        init_period: Whether initialization would otherwise occur before or on the first return.
    """
    all_nan_series = pd.Series(np.nan, index=_DATES, name='never_started')
    all_nan_frame = pd.DataFrame({
        'first': np.nan,
        'second': np.nan,
    }, index=_DATES)

    actual_series = to_zero_first_nonnan_returns(
        returns=all_nan_series,
        init_period=init_period,
    )
    actual_frame = to_zero_first_nonnan_returns(
        returns=all_nan_frame,
        init_period=init_period,
    )

    pd.testing.assert_series_equal(actual_series, all_nan_series)
    pd.testing.assert_frame_equal(actual_frame, all_nan_frame)


def test_all_nan_series_warns_for_unsupported_init_period() -> None:
    """Retain the unsupported-mode warning when an entire Series is missing."""
    returns = pd.Series(np.nan, index=_DATES, name='never_started')
    expected_returns = returns.copy(deep=True)

    with pytest.warns(UserWarning, match='init_period=2 is not supported'):
        actual_returns = to_zero_first_nonnan_returns(returns=returns, init_period=2)

    _assert_series_close(actual_returns, expected_returns)
    pd.testing.assert_series_equal(returns, expected_returns)


# =============================================================================
# NAV scaling and accumulation
# =============================================================================
def test_returns_to_nav_scales_each_ragged_column_to_initial_value() -> None:
    """Scale each asset from its own first-valid NAV to the requested initial value.

    Ragged columns do not share an inception date, so a cross-column scaling reference would
    either manufacture early data for the later asset or scale it from the wrong observation.
    The expected values apply a factor of 100 independently to both unit-based NAV paths.
    """
    returns = _make_ragged_returns()

    # Each asset starts at 100 on its own initialization date. The late column must retain
    # leading NaNs rather than borrowing the early column's 2024-01-01 start.
    expected_nav = pd.DataFrame({
        'early': [100.0, 110.0, 104.5, 106.59, 109.7877],
        'late': [np.nan, np.nan, 100.0, 120.0, 108.0],
    }, index=_DATES)

    actual_nav = returns_to_nav(
        returns=returns,
        init_period=0,
        init_value=100.0,
        ffill_between_nans=False,
    )

    _assert_frames_close(actual_nav, expected_nav)


def test_returns_to_nav_scales_each_ragged_column_to_terminal_value() -> None:
    """Scale each asset from its own last-valid NAV to its requested terminal value.

    The two assets begin on different dates and receive different terminal targets. Simple paths
    make the independent scaling factors visible: the early asset is multiplied by 100 and the
    late asset by 50. Neither column may use the other column's final value.
    """
    returns = pd.DataFrame({
        'early': [np.nan, 0.10, 0.0, 0.0, 0.0],
        'late': [np.nan, np.nan, np.nan, 0.20, 0.0],
    }, index=_DATES)
    expected_nav = pd.DataFrame({
        'early': [100.0, 110.0, 110.0, 110.0, 110.0],
        'late': [np.nan, np.nan, 50.0, 60.0, 60.0],
    }, index=_DATES)

    # Unscaled terminal values are 1.10 and 1.20. Targets of 110 and 60 therefore imply
    # independent scaling factors of 100 and 50, respectively.
    actual_nav = returns_to_nav(
        returns=returns,
        init_period=0,
        terminal_value=np.array([110.0, 60.0]),
        ffill_between_nans=False,
    )

    _assert_frames_close(actual_nav, expected_nav)


def test_returns_to_nav_uses_additive_constant_trade_level() -> None:
    """Use additive P&L accumulation when the trade level remains constant.

    Geometric compounding is appropriate for ordinary investment returns, whereas a constant
    trade level accumulates each period's return against the same notional. The expected paths
    are direct cumulative sums from 1.0, calculated separately for each ragged column.
    """
    returns = _make_ragged_returns()

    # Early finishes at 1 + 0.10 - 0.05 + 0.02 + 0.03 = 1.10. Late finishes
    # at 1 + 0.20 - 0.10 = 1.10; neither path multiplies gross return factors.
    expected_nav = pd.DataFrame({
        'early': [1.0, 1.10, 1.05, 1.07, 1.10],
        'late': [np.nan, np.nan, 1.0, 1.20, 1.10],
    }, index=_DATES)

    actual_nav = returns_to_nav(
        returns=returns,
        init_period=0,
        constant_trade_level=True,
        ffill_between_nans=False,
    )

    _assert_frames_close(actual_nav, expected_nav)


# =============================================================================
# Initialization, shape, and ownership contracts
# =============================================================================
def test_returns_to_nav_initializes_each_column_at_first_valid_return() -> None:
    """Apply ``init_period=1`` independently at each asset's first non-NaN return.

    This initialization convention replaces the first observed return with zero, making that date
    the unit-NAV observation. Because the assets start on different dates, a row-wise operation
    would initialize one of them incorrectly; the expected paths pin the per-column behavior.
    """
    returns = _make_ragged_returns()

    # The first 10% and 20% observations establish the respective unit NAVs and are not
    # compounded. Later observations follow ordinary geometric compounding.
    expected_nav = pd.DataFrame({
        'early': [np.nan, 1.0, 0.95, 0.969, 0.99807],
        'late': [np.nan, np.nan, np.nan, 1.0, 0.90],
    }, index=_DATES)

    actual_nav = returns_to_nav(
        returns=returns,
        init_period=1,
        ffill_between_nans=False,
    )

    _assert_frames_close(actual_nav, expected_nav)


def test_returns_to_nav_series_matches_one_column_frame() -> None:
    """Return equivalent values and labels for Series and one-column DataFrame inputs.

    Many public callers accept either pandas shape. A branch that handles per-column ragged starts
    correctly for a DataFrame must not change the numerical path, dates, or name when the same
    observations arrive as a Series. Both outputs are checked against direct multiplication.
    """
    returns = pd.Series(
        [np.nan, 0.10, -0.05, 0.02, 0.03],
        index=_DATES,
        name='asset',
    )
    expected_nav = pd.Series(
        [1.0, 1.10, 1.045, 1.0659, 1.097877],
        index=_DATES,
        name='asset',
    )

    series_nav = returns_to_nav(
        returns=returns,
        init_period=0,
        ffill_between_nans=False,
    )
    frame_nav = returns_to_nav(
        returns=returns.to_frame(),
        init_period=0,
        ffill_between_nans=False,
    )

    _assert_series_close(series_nav, expected_nav)
    assert isinstance(frame_nav, pd.DataFrame)
    _assert_series_close(frame_nav['asset'], expected_nav)


def test_returns_to_nav_does_not_mutate_input() -> None:
    """Leave the caller's ragged return panel unchanged during initialization.

    ``init_period=0`` needs to insert zero-valued initialization observations before compounding.
    Those zeros belong only to the derived NAV calculation; writing them into the supplied frame
    would silently alter data that a caller might reuse for risk or attribution calculations.
    """
    returns = _make_ragged_returns()
    expected_returns = returns.copy(deep=True)

    # The result is intentionally unused: this test protects ownership of the input object, not
    # a particular NAV path. Other tests independently pin the numerical output.
    returns_to_nav(
        returns=returns,
        init_period=0,
        ffill_between_nans=False,
    )

    pd.testing.assert_frame_equal(returns, expected_returns)
