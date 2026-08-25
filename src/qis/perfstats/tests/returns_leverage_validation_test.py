"""Regression tests for leverage-transform parameter validation.

``lever_returns`` and ``delever_returns`` share two scalar parameters that define their numerical
domain. Leverage is documented as a debt-to-equity ratio and therefore must be finite and
nonnegative. An explicitly supplied ``periods_per_year`` must be a strictly positive integer;
``None`` continues to request frequency inference.

The deterministic monthly panel below keeps every valid result independently calculable. With
12% annual financing and 12 periods per year, financing is 1% per period. At leverage 0.5, the
forward equation is ``1.5 * return - 0.5%`` and the inverse equation is
``(levered return + 0.5%) / 1.5``. Tests cover Series/DataFrame parity, Python and NumPy integer
factors, invalid type and range boundaries, validation before the zero-leverage shortcut,
warnings, labels, and caller ownership.
"""

import warnings
from typing import Protocol, cast

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.returns import delever_returns, lever_returns


# =============================================================================
# Shared deterministic fixtures and typed transform interface
# =============================================================================

_DATES = pd.date_range("2024-01-31", periods=2, freq="ME")

_ANNUAL_FINANCING_RATE = 0.12
_LEVERAGE = 0.5
_PERIODS_PER_YEAR = 12
_TOLERANCE = 1.0e-12

_LEVERAGE_ERROR = r"leverage must be a finite non-negative real number, got"
_PERIODS_ERROR = r"periods_per_year must be a positive integer or None, got"


class _LeverageTransform(Protocol):
    """Test-side interface that permits deliberately invalid runtime arguments."""

    def __call__(
            self,
            *,
            returns: pd.Series | pd.DataFrame,
            leverage: object,
            financing_rate: float | pd.Series,
            periods_per_year: object | None,
    ) -> pd.Series | pd.DataFrame:
        """Apply a leverage transform for validation testing."""
        raise NotImplementedError


_TRANSFORMS: tuple[tuple[str, _LeverageTransform], ...] = (
    ("lever", cast(_LeverageTransform, lever_returns)),
    ("delever", cast(_LeverageTransform, delever_returns)),
)


def _assert_pandas_equal(
        actual: pd.Series | pd.DataFrame,
        expected: pd.Series | pd.DataFrame,
) -> None:
    """Assert exact equality after narrowing both pandas objects to the same shape.

    Args:
        actual: Series or DataFrame produced or retained by a test.
        expected: Independently copied object with the expected matching shape.
    """
    if isinstance(actual, pd.Series):
        assert isinstance(expected, pd.Series)
        pd.testing.assert_series_equal(actual, expected, check_exact=True)
    else:
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(expected, pd.DataFrame)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def _expected_levered_frame() -> pd.DataFrame:
    """Return the leverage-0.5 result calculated directly from the public equation.

    Returns:
        Two-asset monthly levered returns with literal independently calculated values.
    """
    return pd.DataFrame(
        {
            "Asset A": (0.025, -0.020),
            "Asset B": (0.055, -0.005),
        },
        index=_DATES,
    )


def _returns_frame() -> pd.DataFrame:
    """Create the valid two-asset monthly return panel.

    Returns:
        Monthly return DataFrame used by every validation boundary.
    """
    return pd.DataFrame(
        {
            "Asset A": (0.02, -0.01),
            "Asset B": (0.04, 0.00),
        },
        index=_DATES,
    )


def _return_variants() -> tuple[pd.Series, pd.DataFrame]:
    """Create equivalent named-Series and mixed-panel inputs.

    Returns:
        Fresh one-asset Series and two-asset DataFrame inputs.
    """
    frame = _returns_frame()
    series = frame["Asset A"].copy()
    assert isinstance(series, pd.Series)
    return series, frame


# =============================================================================
# Valid boundaries and independently calculated controls
# =============================================================================

def test_lever_returns_preserves_valid_series_and_dataframe_equation() -> None:
    """Match the independently calculated forward equation without mutating inputs."""
    returns = _returns_frame()
    returns_series = returns["Asset A"].copy()
    original_returns = returns.copy(deep=True)
    original_series = returns_series.copy(deep=True)
    expected = _expected_levered_frame()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual_frame = lever_returns(
            returns=returns,
            leverage=_LEVERAGE,
            financing_rate=_ANNUAL_FINANCING_RATE,
            periods_per_year=_PERIODS_PER_YEAR,
        )
        actual_series = lever_returns(
            returns=returns_series,
            leverage=_LEVERAGE,
            financing_rate=_ANNUAL_FINANCING_RATE,
            periods_per_year=_PERIODS_PER_YEAR,
        )

    assert isinstance(actual_frame, pd.DataFrame)
    assert isinstance(actual_series, pd.Series)
    pd.testing.assert_frame_equal(actual_frame, expected, atol=_TOLERANCE)
    pd.testing.assert_series_equal(actual_series, expected["Asset A"], atol=_TOLERANCE)
    pd.testing.assert_frame_equal(returns, original_returns, check_exact=True)
    pd.testing.assert_series_equal(returns_series, original_series, check_exact=True)


def test_delever_returns_preserves_valid_series_and_dataframe_equation() -> None:
    """Invert literal levered values without using the forward function as an oracle."""
    levered = _expected_levered_frame()
    levered_series = levered["Asset A"].copy()
    original_levered = levered.copy(deep=True)
    original_series = levered_series.copy(deep=True)
    expected = _returns_frame()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual_frame = delever_returns(
            returns=levered,
            leverage=_LEVERAGE,
            financing_rate=_ANNUAL_FINANCING_RATE,
            periods_per_year=_PERIODS_PER_YEAR,
        )
        actual_series = delever_returns(
            returns=levered_series,
            leverage=_LEVERAGE,
            financing_rate=_ANNUAL_FINANCING_RATE,
            periods_per_year=_PERIODS_PER_YEAR,
        )

    assert isinstance(actual_frame, pd.DataFrame)
    assert isinstance(actual_series, pd.Series)
    pd.testing.assert_frame_equal(actual_frame, expected, atol=_TOLERANCE)
    pd.testing.assert_series_equal(actual_series, expected["Asset A"], atol=_TOLERANCE)
    pd.testing.assert_frame_equal(levered, original_levered, check_exact=True)
    pd.testing.assert_series_equal(levered_series, original_series, check_exact=True)


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
def test_leverage_transforms_accept_numpy_scalar_parameters(
        transform_name: str,
        transform: _LeverageTransform,
) -> None:
    """Accept NumPy real leverage and integer periods with the same result as Python scalars."""
    returns = _returns_frame()
    original_returns = returns.copy(deep=True)

    expected = transform(
        returns=returns,
        leverage=_LEVERAGE,
        financing_rate=_ANNUAL_FINANCING_RATE,
        periods_per_year=_PERIODS_PER_YEAR,
    )
    actual = transform(
        returns=returns,
        leverage=np.float64(_LEVERAGE),
        financing_rate=_ANNUAL_FINANCING_RATE,
        periods_per_year=np.int64(_PERIODS_PER_YEAR),
    )

    assert transform_name in {"lever", "delever"}
    assert isinstance(actual, pd.DataFrame)
    assert isinstance(expected, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(returns, original_returns, check_exact=True)


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
def test_leverage_transforms_preserve_monthly_annualization_inference(
        transform_name: str,
        transform: _LeverageTransform,
) -> None:
    """Match an explicit monthly factor when a sufficient index requests frequency inference."""
    frame = pd.DataFrame(
        {
            "Asset A": (0.02, -0.01, 0.03, 0.00),
            "Asset B": (0.04, 0.00, -0.02, 0.01),
        },
        index=pd.date_range("2024-01-31", periods=4, freq="ME"),
    )
    series = frame["Asset A"].copy()
    assert isinstance(series, pd.Series)

    for returns in (series, frame):
        original_returns = returns.copy(deep=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            expected = transform(
                returns=returns,
                leverage=_LEVERAGE,
                financing_rate=_ANNUAL_FINANCING_RATE,
                periods_per_year=_PERIODS_PER_YEAR,
            )
            actual = transform(
                returns=returns,
                leverage=_LEVERAGE,
                financing_rate=_ANNUAL_FINANCING_RATE,
                periods_per_year=None,
            )

        assert transform_name in {"lever", "delever"}
        if isinstance(returns, pd.Series):
            assert isinstance(actual, pd.Series)
            assert isinstance(expected, pd.Series)
            pd.testing.assert_series_equal(actual, expected, check_exact=True)
        else:
            assert isinstance(actual, pd.DataFrame)
            assert isinstance(expected, pd.DataFrame)
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        _assert_pandas_equal(returns, original_returns)


# =============================================================================
# Invalid leverage boundaries
# =============================================================================

@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
@pytest.mark.parametrize(
    "invalid_leverage",
    (-1.0, -0.5, np.nan, np.inf, True, 1.0 + 0.0j, "1.0"),
    ids=("singular-minus-one", "negative", "nan", "infinity", "boolean", "complex", "string"),
)
def test_leverage_transforms_reject_invalid_leverage(
        transform_name: str,
        transform: _LeverageTransform,
        invalid_leverage: object,
) -> None:
    """Reject every invalid leverage class consistently for Series and DataFrame inputs."""
    for returns in _return_variants():
        original_returns = returns.copy(deep=True)
        with pytest.raises(ValueError, match=_LEVERAGE_ERROR):
            transform(
                returns=returns,
                leverage=invalid_leverage,
                financing_rate=_ANNUAL_FINANCING_RATE,
                periods_per_year=_PERIODS_PER_YEAR,
            )

        assert transform_name in {"lever", "delever"}
        _assert_pandas_equal(returns, original_returns)


# =============================================================================
# Invalid annualization boundaries and validation order
# =============================================================================

@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
@pytest.mark.parametrize(
    "invalid_periods_per_year",
    (0, -1, 12.0, 12.5, True, np.nan, np.inf, "12"),
    ids=(
        "zero", "negative", "integral-float", "fractional",
        "boolean", "nan", "infinity", "string",
    ),
)
def test_leverage_transforms_reject_invalid_periods_per_year(
        transform_name: str,
        transform: _LeverageTransform,
        invalid_periods_per_year: object,
) -> None:
    """Reject non-integer or non-positive annualization consistently without mutating inputs."""
    for returns in _return_variants():
        original_returns = returns.copy(deep=True)
        with pytest.raises(ValueError, match=_PERIODS_ERROR):
            transform(
                returns=returns,
                leverage=_LEVERAGE,
                financing_rate=_ANNUAL_FINANCING_RATE,
                periods_per_year=invalid_periods_per_year,
            )

        assert transform_name in {"lever", "delever"}
        _assert_pandas_equal(returns, original_returns)


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
def test_leverage_transforms_validate_periods_before_zero_leverage_identity(
        transform_name: str,
        transform: _LeverageTransform,
) -> None:
    """Reject an explicitly invalid factor even when zero leverage makes arithmetic unnecessary."""
    returns = _returns_frame()
    original_returns = returns.copy(deep=True)

    with pytest.raises(ValueError, match=_PERIODS_ERROR):
        transform(
            returns=returns,
            leverage=0.0,
            financing_rate=np.nan,
            periods_per_year=0,
        )

    assert transform_name in {"lever", "delever"}
    pd.testing.assert_frame_equal(returns, original_returns, check_exact=True)


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
def test_leverage_transforms_keep_zero_identity_with_inferred_periods(
        transform_name: str,
        transform: _LeverageTransform,
) -> None:
    """Preserve the independent zero-leverage copy when annualization is intentionally omitted."""
    for returns in _return_variants():
        original_returns = returns.copy(deep=True)
        actual = transform(
            returns=returns,
            leverage=0.0,
            financing_rate=np.nan,
            periods_per_year=None,
        )

        assert transform_name in {"lever", "delever"}
        assert actual is not returns
        _assert_pandas_equal(actual, original_returns)
