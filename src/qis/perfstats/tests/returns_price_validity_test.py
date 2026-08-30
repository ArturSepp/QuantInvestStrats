"""Return-mode endpoint-validity regressions for ``to_returns``.

The five public return modes consume different price endpoints. Ratio and logarithmic returns
require two finite positive levels, while absolute changes and level observations are also
meaningful for finite signed rates and spreads. These tests use independently constructed literal
references to keep those domains separate.

One mixed panel combines healthy prices, zero and negative signed levels, missing values,
positive and negative infinity, and recovery after each boundary. Running the same panel with
ordinary ``float64`` and nullable ``Float64`` columns verifies that pandas' physical missing-data
representation cannot change the numerical contract or warning behavior.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.config import ReturnTypes
from qis.perfstats.returns import to_returns


# =============================================================================
# Shared deterministic fixtures and independent references
# =============================================================================

_DATES = pd.date_range("2024-01-01", periods=4, freq="D", name="Date")

_HEALTHY = "Healthy price"
_ZERO_RECOVERY = "Zero recovery"
_NEGATIVE_RECOVERY = "Negative recovery"
_MISSING_RECOVERY = "Missing recovery"
_POSITIVE_INFINITY_RECOVERY = "Positive infinity recovery"
_NEGATIVE_INFINITY_RECOVERY = "Negative infinity recovery"
_SIGNED_SPREAD = "Signed spread"

_TOLERANCE = 1.0e-12


def _as_nullable(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert every column to pandas nullable floating point.

    Args:
        frame: NumPy-backed deterministic panel.

    Returns:
        An equivalent panel whose columns use pandas ``Float64``.
    """
    return frame.astype(pd.Float64Dtype())


def _make_price_panel(nullable: bool) -> pd.DataFrame:
    """Create every endpoint state in one deliberately ordered panel.

    Args:
        nullable: Whether to use pandas nullable floating columns.

    Returns:
        Mixed price, rate, and spread histories for all return modes.
    """
    prices = pd.DataFrame(
        {
            _HEALTHY: (100.0, 110.0, 121.0, 133.1),
            _ZERO_RECOVERY: (100.0, 0.0, 110.0, 121.0),
            _NEGATIVE_RECOVERY: (100.0, -100.0, 110.0, 121.0),
            _MISSING_RECOVERY: (100.0, np.nan, 110.0, 121.0),
            _POSITIVE_INFINITY_RECOVERY: (100.0, np.inf, 110.0, 121.0),
            _NEGATIVE_INFINITY_RECOVERY: (100.0, -np.inf, 110.0, 121.0),
            _SIGNED_SPREAD: (-1.0, 0.0, 1.0, -2.0),
        },
        index=_DATES,
    )
    return _as_nullable(prices) if nullable else prices


def _make_expected_returns(return_type: ReturnTypes, nullable: bool) -> pd.DataFrame:
    """Construct literal results for one return convention.

    Args:
        return_type: Return convention whose endpoint domain is under test.
        nullable: Whether expected columns should use pandas nullable floating point.

    Returns:
        Independently calculated results in the fixture's deliberate column order.

    Raises:
        AssertionError: If a new return convention lacks an explicit reference.
    """
    missing = (np.nan, np.nan, np.nan, 0.10)

    if return_type == ReturnTypes.RELATIVE:
        expected = pd.DataFrame(
            {
                _HEALTHY: (np.nan, 0.10, 0.10, 0.10),
                _ZERO_RECOVERY: missing,
                _NEGATIVE_RECOVERY: missing,
                _MISSING_RECOVERY: missing,
                _POSITIVE_INFINITY_RECOVERY: missing,
                _NEGATIVE_INFINITY_RECOVERY: missing,
                _SIGNED_SPREAD: (np.nan, np.nan, np.nan, np.nan),
            },
            index=_DATES,
        )
    elif return_type == ReturnTypes.LOG:
        log_ten_percent = math.log(1.10)
        log_missing = (np.nan, np.nan, np.nan, log_ten_percent)
        expected = pd.DataFrame(
            {
                _HEALTHY: (np.nan, log_ten_percent, log_ten_percent, log_ten_percent),
                _ZERO_RECOVERY: log_missing,
                _NEGATIVE_RECOVERY: log_missing,
                _MISSING_RECOVERY: log_missing,
                _POSITIVE_INFINITY_RECOVERY: log_missing,
                _NEGATIVE_INFINITY_RECOVERY: log_missing,
                _SIGNED_SPREAD: (np.nan, np.nan, np.nan, np.nan),
            },
            index=_DATES,
        )
    elif return_type == ReturnTypes.DIFFERENCE:
        expected = pd.DataFrame(
            {
                _HEALTHY: (np.nan, 10.0, 11.0, 12.1),
                _ZERO_RECOVERY: (np.nan, -100.0, 110.0, 11.0),
                _NEGATIVE_RECOVERY: (np.nan, -200.0, 210.0, 11.0),
                _MISSING_RECOVERY: (np.nan, np.nan, np.nan, 11.0),
                _POSITIVE_INFINITY_RECOVERY: (np.nan, np.nan, np.nan, 11.0),
                _NEGATIVE_INFINITY_RECOVERY: (np.nan, np.nan, np.nan, 11.0),
                _SIGNED_SPREAD: (np.nan, 1.0, 1.0, -3.0),
            },
            index=_DATES,
        )
    elif return_type == ReturnTypes.LEVEL:
        expected = pd.DataFrame(
            {
                _HEALTHY: (100.0, 110.0, 121.0, 133.1),
                _ZERO_RECOVERY: (100.0, 0.0, 110.0, 121.0),
                _NEGATIVE_RECOVERY: (100.0, -100.0, 110.0, 121.0),
                _MISSING_RECOVERY: (100.0, np.nan, 110.0, 121.0),
                _POSITIVE_INFINITY_RECOVERY: (100.0, np.nan, 110.0, 121.0),
                _NEGATIVE_INFINITY_RECOVERY: (100.0, np.nan, 110.0, 121.0),
                _SIGNED_SPREAD: (-1.0, 0.0, 1.0, -2.0),
            },
            index=_DATES,
        )
    elif return_type == ReturnTypes.LEVEL0:
        expected = pd.DataFrame(
            {
                _HEALTHY: (np.nan, 100.0, 110.0, 121.0),
                _ZERO_RECOVERY: (np.nan, 100.0, 0.0, 110.0),
                _NEGATIVE_RECOVERY: (np.nan, 100.0, -100.0, 110.0),
                _MISSING_RECOVERY: (np.nan, 100.0, np.nan, 110.0),
                _POSITIVE_INFINITY_RECOVERY: (np.nan, 100.0, np.nan, 110.0),
                _NEGATIVE_INFINITY_RECOVERY: (np.nan, 100.0, np.nan, 110.0),
                _SIGNED_SPREAD: (np.nan, -1.0, 0.0, 1.0),
            },
            index=_DATES,
        )
    else:
        raise AssertionError(f"Unhandled return type: {return_type}")

    return _as_nullable(expected) if nullable else expected


def _select_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Select one uniquely labeled column with an explicit Series contract.

    Args:
        frame: Deterministic fixture or expected-result panel.
        column: Unique column label to select.

    Returns:
        The selected Series with its original name and index.
    """
    selected = frame[column]
    assert isinstance(selected, pd.Series)
    return selected


def _assert_frame_close(
    actual: pd.Series | pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    """Assert values, missingness, dtype, shape, labels, and column order.

    Args:
        actual: Result under the public Series-or-DataFrame return annotation.
        expected: Independently calculated DataFrame reference.
    """
    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )


# =============================================================================
# Mode-specific endpoint validity
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "Float64"))
@pytest.mark.parametrize("return_type", tuple(ReturnTypes), ids=lambda mode: mode.name)
def test_to_returns_applies_mode_specific_endpoint_validity(
    return_type: ReturnTypes,
    nullable: bool,
) -> None:
    """Apply each public formula only on the endpoints in its documented domain.

    Args:
        return_type: Return convention under test.
        nullable: Whether inputs and expected results use pandas nullable floating point.
    """
    prices = _make_price_panel(nullable=nullable)
    original_prices = prices.copy()
    expected = _make_expected_returns(return_type=return_type, nullable=nullable)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = to_returns(
            prices=prices,
            return_type=return_type,
            ffill_nans=False,
        )

    _assert_frame_close(actual, expected)
    pd.testing.assert_frame_equal(prices, original_prices, check_exact=True)


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "Float64"))
@pytest.mark.parametrize("return_type", tuple(ReturnTypes), ids=lambda mode: mode.name)
def test_to_returns_preserves_series_parity_for_mode_specific_validity(
    return_type: ReturnTypes,
    nullable: bool,
) -> None:
    """Match one-column DataFrame results without losing the Series name.

    Args:
        return_type: Return convention under test.
        nullable: Whether the signed-spread input uses pandas nullable floating point.
    """
    prices = _select_series(_make_price_panel(nullable=nullable), _SIGNED_SPREAD)
    expected = _select_series(
        _make_expected_returns(return_type=return_type, nullable=nullable),
        _SIGNED_SPREAD,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = to_returns(
            prices=prices,
            return_type=return_type,
            ffill_nans=False,
        )

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )


# =============================================================================
# Existing option interactions
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "Float64"))
def test_to_returns_log_override_uses_positive_endpoint_validity(nullable: bool) -> None:
    """Apply the log domain when ``is_log_returns`` overrides a signed mode.

    Args:
        nullable: Whether the mixed panel uses pandas nullable floating point.
    """
    prices = _make_price_panel(nullable=nullable)
    expected = _make_expected_returns(return_type=ReturnTypes.LOG, nullable=nullable)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = to_returns(
            prices=prices,
            is_log_returns=True,
            return_type=ReturnTypes.DIFFERENCE,
            ffill_nans=False,
        )

    _assert_frame_close(actual, expected)


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "Float64"))
@pytest.mark.parametrize(
    ("ffill_nans", "expected_values"),
    (
        pytest.param(True, (np.nan, 0.0, 0.10, 0.10), id="forward-fill"),
        pytest.param(False, (np.nan, np.nan, np.nan, 0.10), id="preserve-gap"),
    ),
)
def test_to_returns_applies_fill_before_endpoint_validity(
    nullable: bool,
    ffill_nans: bool,
    expected_values: tuple[float, ...],
) -> None:
    """Retain the explicit missing-price fill policy before validating endpoints.

    Args:
        nullable: Whether the missing-price Series uses pandas nullable floating point.
        ffill_nans: Whether the missing January 2 level carries January 1's price.
        expected_values: Literal arithmetic-return result after the selected fill policy.
    """
    prices = _select_series(_make_price_panel(nullable=nullable), _MISSING_RECOVERY)
    expected = pd.Series(expected_values, index=_DATES, name=_MISSING_RECOVERY)
    if nullable:
        expected = expected.astype(pd.Float64Dtype())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = to_returns(prices=prices, ffill_nans=ffill_nans)

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "Float64"))
def test_to_returns_initializes_only_after_mode_specific_validation(nullable: bool) -> None:
    """Keep ``is_first_zero`` as a post-validation initialization convention.

    Args:
        nullable: Whether the zero-recovery Series uses pandas nullable floating point.
    """
    prices = _select_series(_make_price_panel(nullable=nullable), _ZERO_RECOVERY)
    expected = pd.Series(
        (np.nan, np.nan, 0.0, 0.10),
        index=_DATES,
        name=_ZERO_RECOVERY,
    )
    if nullable:
        expected = expected.astype(pd.Float64Dtype())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = to_returns(
            prices=prices,
            ffill_nans=False,
            is_first_zero=True,
        )

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
