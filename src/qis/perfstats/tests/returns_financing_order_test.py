"""Regression tests for chronological leverage-funding alignment.

``lever_returns`` and ``delever_returns`` interpret a financing-rate Series as dated annual
observations that are forward-filled from prior dates. A Series is a date-to-value mapping, so its
storage order must not change which rate was observable at a return date. These tests use literal
monthly equations to distinguish chronological forward-fill from row-order fill and to prove that
no future funding rate is carried backward.

The primary DataFrame combines complete, leading-gap, and interior-gap asset histories in one
public call. Sorted, reverse-sorted, and shuffled funding inputs must produce the same values for
every column. Supplemental boundaries cover a funding history that begins after the return panel,
arbitrary return-index order, named Series parity, duplicate funding dates, zero-leverage identity,
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

_DATES = pd.date_range("2024-01-31", periods=4, freq="ME")

_ANNUAL_FUNDING_LOW = 0.12
_ANNUAL_FUNDING_HIGH = 0.24
_LEVERAGE = 1.0
_PERIODS_PER_YEAR = 12

_DUPLICATE_ERROR = r"financing_rate index must not contain duplicate dates"
_TOLERANCE = 1.0e-12


class _LeverageTransform(Protocol):
    """Test-side interface shared by the forward and inverse public transforms."""

    def __call__(
            self,
            returns: pd.Series | pd.DataFrame,
            leverage: float,
            financing_rate: float | pd.Series = 0.0,
            periods_per_year: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Apply a leverage transform for deterministic regression testing."""
        raise NotImplementedError


_TRANSFORMS: tuple[tuple[str, _LeverageTransform], ...] = (
    ("lever", cast(_LeverageTransform, lever_returns)),
    ("delever", cast(_LeverageTransform, delever_returns)),
)


def _unlevered_returns() -> pd.DataFrame:
    """Create the mixed return panel containing every relevant column state.

    Returns:
        Four monthly observations for complete, leading-gap, and interior-gap assets.
    """
    return pd.DataFrame(
        {
            "Complete": (0.02, -0.01, 0.03, 0.00),
            "Leading gap": (np.nan, 0.04, 0.01, -0.03),
            "Interior gap": (-0.02, np.nan, 0.01, -0.03),
        },
        index=_DATES,
    )


def _expected_levered_returns() -> pd.DataFrame:
    """Return literal values from ``2 * return - periodic funding``.

    Returns:
        Independently calculated leverage-one returns using funding `[1%, 1%, 2%, 2%]`.
    """
    return pd.DataFrame(
        {
            "Complete": (0.03, -0.03, 0.04, -0.02),
            "Leading gap": (np.nan, 0.07, 0.00, -0.08),
            "Interior gap": (-0.05, np.nan, 0.00, -0.08),
        },
        index=_DATES,
    )


def _funding_series(order: str) -> pd.Series:
    """Create equivalent unique funding mappings in a requested storage order.

    Args:
        order: One of ``sorted``, ``reversed``, or ``shuffled``.

    Returns:
        Annual funding observations whose chronological mapping is January 12% and March 24%.

    Raises:
        ValueError: If an unsupported internal fixture order is requested.
    """
    if order == "sorted":
        positions = (0, 2)
        values = (_ANNUAL_FUNDING_LOW, _ANNUAL_FUNDING_HIGH)
    elif order == "reversed":
        positions = (2, 0)
        values = (_ANNUAL_FUNDING_HIGH, _ANNUAL_FUNDING_LOW)
    elif order == "shuffled":
        positions = (2, 0, 1)
        values = (_ANNUAL_FUNDING_HIGH, _ANNUAL_FUNDING_LOW, _ANNUAL_FUNDING_LOW)
    else:
        raise ValueError(f"unsupported funding fixture order: {order}")

    return pd.Series(values, index=_DATES[list(positions)], name="Annual funding")


def _transform_case(transform_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return literal input and expectation for one transform direction.

    Args:
        transform_name: ``lever`` for the forward equation or ``delever`` for its inverse.

    Returns:
        Fresh input and independently calculated expected DataFrames.
    """
    if transform_name == "lever":
        return _unlevered_returns(), _expected_levered_returns()
    return _expected_levered_returns(), _unlevered_returns()


def _pandas_variants(frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Create equivalent named-Series and mixed-DataFrame inputs.

    Args:
        frame: Source panel whose complete column supplies the Series case.

    Returns:
        Independent named Series and owned DataFrame inputs.
    """
    series = frame["Complete"].copy()
    assert isinstance(series, pd.Series)
    return series, frame


def _assert_pandas_equal(
        actual: pd.Series | pd.DataFrame,
        expected: pd.Series | pd.DataFrame,
) -> None:
    """Assert exact-shape pandas equality after narrowing both objects.

    Args:
        actual: Result or retained caller object under test.
        expected: Expected object of the same pandas shape.
    """
    if isinstance(actual, pd.Series):
        assert isinstance(expected, pd.Series)
        pd.testing.assert_series_equal(actual, expected, atol=_TOLERANCE)
    else:
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(expected, pd.DataFrame)
        pd.testing.assert_frame_equal(actual, expected, atol=_TOLERANCE)


# =============================================================================
# Chronological funding-order contract and confirmed regression
# =============================================================================

@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
@pytest.mark.parametrize("funding_order", ("sorted", "reversed", "shuffled"))
def test_lever_and_delever_returns_normalize_unique_funding_order(
        transform_name: str,
        transform: _LeverageTransform,
        funding_order: str,
) -> None:
    """Apply one chronological funding path regardless of unique Series storage order.

    January and February use 1% periodic funding; March and April use 2%. The literal mixed-panel
    expectations prove that reverse storage neither carries March's future rate into February nor
    loses the trailing April rate.
    """
    returns, expected = _transform_case(transform_name)
    funding = _funding_series(funding_order)
    original_returns = returns.copy()
    original_funding = funding.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = transform(
            returns=returns,
            leverage=_LEVERAGE,
            financing_rate=funding,
            periods_per_year=_PERIODS_PER_YEAR,
        )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected, atol=_TOLERANCE)
    pd.testing.assert_frame_equal(returns, original_returns, check_exact=True)
    pd.testing.assert_series_equal(funding, original_funding, check_exact=True)


def test_lever_and_delever_returns_preserve_nullable_mixed_panel_round_trip() -> None:
    """Preserve nullable complete and ragged columns with reversed nullable funding.

    The independently calculated levered values use the same 1%, 1%, 2%, 2% periodic funding
    path as the standard-dtype regression. De-levering those literal values must then recover the
    original nullable panel without replacing ``pd.NA``, changing labels, or mutating either
    caller-owned input.
    """
    nullable_dtype = pd.Float64Dtype()
    returns = _unlevered_returns().astype(nullable_dtype)
    expected_levered = _expected_levered_returns().astype(nullable_dtype)
    funding = pd.Series(
        pd.array((_ANNUAL_FUNDING_HIGH, _ANNUAL_FUNDING_LOW), dtype=nullable_dtype),
        index=_DATES[[2, 0]],
        name="Annual funding",
    )
    original_returns = returns.copy()
    original_expected_levered = expected_levered.copy()
    original_funding = funding.copy()

    actual_levered = lever_returns(
        returns=returns,
        leverage=_LEVERAGE,
        financing_rate=funding,
        periods_per_year=_PERIODS_PER_YEAR,
    )
    assert isinstance(actual_levered, pd.DataFrame)
    pd.testing.assert_frame_equal(actual_levered, expected_levered, atol=_TOLERANCE)

    actual_unlevered = delever_returns(
        returns=expected_levered,
        leverage=_LEVERAGE,
        financing_rate=funding,
        periods_per_year=_PERIODS_PER_YEAR,
    )
    assert isinstance(actual_unlevered, pd.DataFrame)
    pd.testing.assert_frame_equal(actual_unlevered, original_returns, atol=_TOLERANCE)
    pd.testing.assert_frame_equal(returns, original_returns, check_exact=True)
    pd.testing.assert_frame_equal(expected_levered, original_expected_levered, check_exact=True)
    pd.testing.assert_series_equal(funding, original_funding, check_exact=True)


# =============================================================================
# Missing funding boundaries and no-look-ahead behavior
# =============================================================================

@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
def test_lever_and_delever_returns_keep_leading_funding_unavailable(
        transform_name: str,
        transform: _LeverageTransform,
) -> None:
    """Leave every January result missing when the first funding observation is in February."""
    returns, expected = _transform_case(transform_name)
    expected.iloc[0, :] = np.nan
    funding = pd.Series(
        (_ANNUAL_FUNDING_LOW, _ANNUAL_FUNDING_HIGH),
        index=_DATES[[1, 2]],
        name="Annual funding",
    )
    original_returns = returns.copy()
    original_funding = funding.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = transform(
            returns=returns,
            leverage=_LEVERAGE,
            financing_rate=funding,
            periods_per_year=_PERIODS_PER_YEAR,
        )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected, atol=_TOLERANCE)
    pd.testing.assert_frame_equal(returns, original_returns, check_exact=True)
    pd.testing.assert_series_equal(funding, original_funding, check_exact=True)


# =============================================================================
# Series parity, return ordering, and caller ownership
# =============================================================================

@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
def test_lever_and_delever_returns_preserve_named_series_with_reversed_funding(
        transform_name: str,
        transform: _LeverageTransform,
) -> None:
    """Match the complete mixed-panel column while retaining a named Series and both inputs."""
    frame, expected_frame = _transform_case(transform_name)
    returns = frame["Complete"].copy()
    expected = expected_frame["Complete"]
    assert isinstance(returns, pd.Series)
    assert isinstance(expected, pd.Series)
    funding = _funding_series("reversed")
    original_returns = returns.copy()
    original_funding = funding.copy()

    actual = transform(
        returns=returns,
        leverage=_LEVERAGE,
        financing_rate=funding,
        periods_per_year=_PERIODS_PER_YEAR,
    )

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected, atol=_TOLERANCE)
    pd.testing.assert_series_equal(returns, original_returns, check_exact=True)
    pd.testing.assert_series_equal(funding, original_funding, check_exact=True)


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
@pytest.mark.parametrize(
    "return_order",
    ((3, 2, 1, 0), (2, 0, 3, 1), (0, 1, 1, 3)),
    ids=("descending", "shuffled", "duplicate"),
)
def test_lever_and_delever_returns_preserve_return_index_order(
        transform_name: str,
        transform: _LeverageTransform,
        return_order: tuple[int, ...],
) -> None:
    """Align by chronology while retaining descending, shuffled, or duplicate return dates."""
    frame, expected_frame = _transform_case(transform_name)
    returns = frame.iloc[list(return_order)].copy()
    expected = expected_frame.iloc[list(return_order)]
    original_returns = returns.copy()
    funding = _funding_series("sorted")
    original_funding = funding.copy()

    actual = transform(
        returns=returns,
        leverage=_LEVERAGE,
        financing_rate=funding,
        periods_per_year=_PERIODS_PER_YEAR,
    )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected, atol=_TOLERANCE)
    pd.testing.assert_frame_equal(returns, original_returns, check_exact=True)
    pd.testing.assert_series_equal(funding, original_funding, check_exact=True)


# =============================================================================
# Duplicate funding dates and zero-leverage identity
# =============================================================================

@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
def test_lever_and_delever_returns_reject_duplicate_funding_dates(
        transform_name: str,
        transform: _LeverageTransform,
) -> None:
    """Reject ambiguous duplicate funding observations consistently without mutating inputs."""
    frame, _ = _transform_case(transform_name)
    funding = pd.Series(
        (_ANNUAL_FUNDING_LOW, 0.18, _ANNUAL_FUNDING_HIGH),
        index=pd.DatetimeIndex((_DATES[0], _DATES[0], _DATES[2])),
        name="Annual funding",
    )
    original_funding = funding.copy()

    for returns in _pandas_variants(frame):
        original_returns = returns.copy()
        with pytest.raises(ValueError, match=_DUPLICATE_ERROR):
            transform(
                returns=returns,
                leverage=_LEVERAGE,
                financing_rate=funding,
                periods_per_year=_PERIODS_PER_YEAR,
            )

        _assert_pandas_equal(returns, original_returns)
        pd.testing.assert_series_equal(funding, original_funding, check_exact=True)


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
def test_lever_and_delever_returns_keep_zero_identity_with_duplicate_funding(
        transform_name: str,
        transform: _LeverageTransform,
) -> None:
    """Keep funding-index validation irrelevant when zero leverage removes financing entirely."""
    frame, _ = _transform_case(transform_name)
    funding = pd.Series(
        (_ANNUAL_FUNDING_LOW, _ANNUAL_FUNDING_HIGH),
        index=pd.DatetimeIndex((_DATES[0], _DATES[0])),
        name="Annual funding",
    )
    original_funding = funding.copy()

    for returns in _pandas_variants(frame):
        original_returns = returns.copy()
        actual = transform(
            returns=returns,
            leverage=0.0,
            financing_rate=funding,
            periods_per_year=_PERIODS_PER_YEAR,
        )

        assert actual is not returns
        _assert_pandas_equal(actual, original_returns)
        _assert_pandas_equal(returns, original_returns)
        pd.testing.assert_series_equal(funding, original_funding, check_exact=True)
