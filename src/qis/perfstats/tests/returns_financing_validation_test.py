"""Regression coverage for leverage financing-value and index validation.

``lever_returns`` and ``delever_returns`` accept either one annual financing scalar or a dated
Series. For nonzero leverage, scalar funding must be finite and real, while a Series may also use
ordinary or nullable missing values to represent unavailable funding. Series alignment additionally
requires defined, compatible date axes so point-in-time forward filling cannot cross an unknown
``NaT`` chronology or a naive/aware timezone boundary.

The deterministic four-month panel combines complete, leading-gap, and interior-gap assets in each
public call. With leverage one and 12 periods per year, annual funding observations of 12%,
missing, and -12% become monthly funding of 1%, missing, missing, and -1%. Every expected result is
calculated directly from the documented forward and inverse equations rather than through the other
production transform. Supplemental boundaries cover invalid scalar and Series values, empty and
all-missing funding, naive and timezone-aware indexes, Series/DataFrame parity, exact zero-leverage
identity, warnings, labels, nullable dtypes, and caller ownership.
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

_LEVERAGE = 1.0
_PERIODS_PER_YEAR = 12
_TOLERANCE = 1.0e-12

_DATETIME_INDEX_ERROR = (
    r"returns and financing_rate must use DatetimeIndex when financing_rate is a Series"
)
_FINANCING_SCALAR_ERROR = r"financing_rate must be a finite real number or a Series, got"
_FINANCING_SERIES_ERROR = (
    r"financing_rate Series must contain only finite real values or missing values"
)
_NAT_INDEX_ERROR = r"returns and financing_rate indexes must not contain NaT"
_TIMEZONE_ERROR = (
    r"returns and financing_rate indexes must both be timezone-naive or timezone-aware"
)


class _LeverageTransform(Protocol):
    """Test-side interface permitting deliberately invalid financing inputs."""

    def __call__(
        self,
        returns: pd.Series | pd.DataFrame,
        leverage: float,
        financing_rate: object = 0.0,
        periods_per_year: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Apply one public leverage transform for deterministic testing."""
        raise NotImplementedError


_TRANSFORMS: tuple[tuple[str, _LeverageTransform], ...] = (
    ("lever", cast(_LeverageTransform, lever_returns)),
    ("delever", cast(_LeverageTransform, delever_returns)),
)


def _assert_pandas_equal(
    actual: pd.Series | pd.DataFrame,
    expected: pd.Series | pd.DataFrame,
) -> None:
    """Assert shape-matched pandas equality after narrowing ambiguous return types.

    Args:
        actual: Result or retained caller-owned pandas object.
        expected: Independently calculated or copied object with the same shape.
    """
    if isinstance(actual, pd.Series):
        assert isinstance(expected, pd.Series)
        pd.testing.assert_series_equal(actual, expected, atol=_TOLERANCE)
    else:
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(expected, pd.DataFrame)
        pd.testing.assert_frame_equal(actual, expected, atol=_TOLERANCE)


def _case_variants(
    returns: pd.DataFrame,
    expected: pd.DataFrame,
) -> tuple[
    tuple[pd.Series, pd.Series],
    tuple[pd.DataFrame, pd.DataFrame],
]:
    """Return equivalent named-Series and mixed-DataFrame test cases.

    Args:
        returns: Mixed-panel input for one transform direction.
        expected: Independently calculated mixed-panel expectation.

    Returns:
        Complete-column Series pair followed by the full DataFrame pair.
    """
    returns_series = returns["Complete"].copy()
    expected_series = expected["Complete"].copy()
    assert isinstance(returns_series, pd.Series)
    assert isinstance(expected_series, pd.Series)
    return (returns_series, expected_series), (returns, expected)


def _expected_finite_levered_returns() -> pd.DataFrame:
    """Return literal leverage-one values for periodic funding `[1%, 1%, 2%, 2%]`.

    Returns:
        Mixed levered panel used by valid timezone-alignment controls.
    """
    return pd.DataFrame(
        {
            "Complete": (0.03, -0.03, 0.04, -0.02),
            "Leading gap": (np.nan, 0.07, 0.00, -0.08),
            "Interior gap": (-0.05, np.nan, 0.00, -0.08),
        },
        index=_DATES,
    )


def _expected_missing_levered_returns() -> pd.DataFrame:
    """Return literal values for periodic funding `[1%, missing, missing, -1%]`.

    Returns:
        Mixed levered panel retaining results only where financing is available.
    """
    return pd.DataFrame(
        {
            "Complete": (0.03, np.nan, np.nan, 0.01),
            "Leading gap": (np.nan, np.nan, np.nan, -0.05),
            "Interior gap": (-0.05, np.nan, np.nan, -0.05),
        },
        index=_DATES,
    )


def _financing_with_explicit_missing(nullable: bool) -> pd.Series:
    """Create annual funding `[12%, missing, -12%]` on January, February, and April.

    Args:
        nullable: Use pandas nullable ``Float64`` and ``pd.NA`` when true.

    Returns:
        Dated funding whose explicit February gap persists through absent March.
    """
    if nullable:
        values = pd.array((0.12, pd.NA, -0.12), dtype=pd.Float64Dtype())
    else:
        values = np.asarray((0.12, np.nan, -0.12), dtype=float)
    return pd.Series(values, index=_DATES[[0, 1, 3]], name="Annual funding")


def _returns_with_available_funding_only() -> pd.DataFrame:
    """Return asset values masked where the explicit financing observation is missing.

    Returns:
        Original January/April returns with every February/March value unavailable.
    """
    returns = _unlevered_returns()
    returns.iloc[1:3, :] = np.nan
    return returns


def _transform_case(
    transform_name: str,
    nullable: bool = False,
    missing_financing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return literal input and expectation for one transform direction.

    Args:
        transform_name: ``lever`` for the forward equation or ``delever`` for its inverse.
        nullable: Convert both panels to pandas nullable ``Float64``.
        missing_financing: Use the explicit-missing rather than fully finite funding path.

    Returns:
        Fresh transform input and independently calculated expected result.
    """
    if missing_financing:
        levered = _expected_missing_levered_returns()
        unlevered = _returns_with_available_funding_only()
    else:
        levered = _expected_finite_levered_returns()
        unlevered = _unlevered_returns()

    if transform_name == "lever":
        returns, expected = unlevered, levered
    else:
        returns, expected = levered, unlevered

    if nullable:
        nullable_dtype = pd.Float64Dtype()
        return returns.astype(nullable_dtype), expected.astype(nullable_dtype)
    return returns, expected


def _unlevered_returns() -> pd.DataFrame:
    """Create the complete, leading-gap, and interior-gap asset panel.

    Returns:
        Four monthly unlevered observations with every material column state.
    """
    return pd.DataFrame(
        {
            "Complete": (0.02, -0.01, 0.03, 0.00),
            "Leading gap": (np.nan, 0.04, 0.01, -0.03),
            "Interior gap": (-0.02, np.nan, 0.01, -0.03),
        },
        index=_DATES,
    )


# =============================================================================
# Explicit missing financing and independently calculated values
# =============================================================================


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_financing_series_preserves_explicit_missing_intervals(
    transform_name: str,
    transform: _LeverageTransform,
    nullable: bool,
) -> None:
    """Keep an explicit missing rate unavailable until the next finite observation.

    January uses 1% monthly funding, February is explicitly missing, absent March therefore stays
    missing, and April uses -1%. Both public equations must produce the literal mixed-panel values
    and the same complete-column Series without mutating either caller-owned input.
    """
    returns, expected = _transform_case(transform_name, nullable=nullable)
    funding = _financing_with_explicit_missing(nullable=nullable)
    original_funding = funding.copy()

    for transform_returns, transform_expected in _case_variants(returns, expected):
        original_returns = transform_returns.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            actual = transform(
                returns=transform_returns,
                leverage=_LEVERAGE,
                financing_rate=funding,
                periods_per_year=_PERIODS_PER_YEAR,
            )

        _assert_pandas_equal(actual, transform_expected)
        _assert_pandas_equal(transform_returns, original_returns)
        pd.testing.assert_series_equal(funding, original_funding, check_exact=True)


# =============================================================================
# Scalar financing-value domain
# =============================================================================


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
@pytest.mark.parametrize(
    "invalid_financing",
    (np.nan, np.inf, -np.inf, True, np.bool_(False), 0.12 + 0.0j, "0.12", None, pd.NA),
    ids=(
        "nan",
        "positive-infinity",
        "negative-infinity",
        "python-boolean",
        "numpy-boolean",
        "complex",
        "string",
        "none",
        "pandas-na",
    ),
)
def test_leverage_transforms_reject_invalid_scalar_financing(
    transform_name: str,
    transform: _LeverageTransform,
    invalid_financing: object,
) -> None:
    """Reject every non-finite or non-real scalar consistently for both pandas shapes."""
    returns = _unlevered_returns()
    for transform_returns, _ in _case_variants(returns, returns):
        original_returns = transform_returns.copy()
        with pytest.raises(ValueError, match=_FINANCING_SCALAR_ERROR):
            transform(
                returns=transform_returns,
                leverage=_LEVERAGE,
                financing_rate=invalid_financing,
                periods_per_year=_PERIODS_PER_YEAR,
            )

        assert transform_name in {"lever", "delever"}
        _assert_pandas_equal(transform_returns, original_returns)


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
@pytest.mark.parametrize(
    "financing_rate",
    (0, np.int64(0), 0.12, np.float64(0.12), -0.12),
    ids=("python-integer", "numpy-integer", "python-float", "numpy-float", "negative-finite"),
)
def test_leverage_transforms_accept_finite_real_scalar_financing(
    transform_name: str,
    transform: _LeverageTransform,
    financing_rate: int | float | np.integer | np.floating,
) -> None:
    """Preserve finite Python/NumPy real rates, including zero and negative funding."""
    returns = _unlevered_returns()
    original_returns = returns.copy()
    periodic_funding = float(financing_rate) / _PERIODS_PER_YEAR
    expected: pd.DataFrame
    if transform_name == "lever":
        expected = (1.0 + _LEVERAGE) * returns - _LEVERAGE * periodic_funding
    else:
        expected = (returns + _LEVERAGE * periodic_funding) / (1.0 + _LEVERAGE)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = transform(
            returns=returns,
            leverage=_LEVERAGE,
            financing_rate=financing_rate,
            periods_per_year=_PERIODS_PER_YEAR,
        )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected, atol=_TOLERANCE)
    pd.testing.assert_frame_equal(returns, original_returns, check_exact=True)


# =============================================================================
# Financing-Series observed-value domain
# =============================================================================


def _invalid_financing_series(kind: str) -> pd.Series:
    """Create one financing Series containing a requested invalid observed value.

    Args:
        kind: Internal fixture identifier for one invalid observed-value class.

    Returns:
        Dated financing Series combining valid neighbors with the invalid value.

    Raises:
        ValueError: If an unsupported internal fixture identifier is requested.
    """
    index = _DATES[[0, 1, 2]]
    if kind == "positive-infinity":
        values = (0.12, np.inf, 0.24)
    elif kind == "negative-infinity":
        values = (0.12, -np.inf, 0.24)
    elif kind == "boolean":
        values = (0.12, True, 0.24)
    elif kind == "complex":
        values = (0.12, 0.18 + 0.0j, 0.24)
    elif kind == "string":
        values = (0.12, "0.18", 0.24)
    elif kind == "nullable-missing-infinity":
        values = pd.array((0.12, pd.NA, np.inf), dtype=pd.Float64Dtype())
    else:
        raise ValueError(f"unsupported invalid financing fixture: {kind}")
    return pd.Series(values, index=index, name="Annual funding")


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
@pytest.mark.parametrize(
    "invalid_kind",
    (
        "positive-infinity",
        "negative-infinity",
        "boolean",
        "complex",
        "string",
        "nullable-missing-infinity",
    ),
)
def test_leverage_transforms_reject_invalid_observed_financing_values(
    transform_name: str,
    transform: _LeverageTransform,
    invalid_kind: str,
) -> None:
    """Reject invalid observed values globally before calculating any mixed-panel column."""
    returns = _unlevered_returns()
    funding = _invalid_financing_series(invalid_kind)
    original_returns = returns.copy()
    original_funding = funding.copy()

    with pytest.raises(ValueError, match=_FINANCING_SERIES_ERROR):
        transform(
            returns=returns,
            leverage=_LEVERAGE,
            financing_rate=funding,
            periods_per_year=_PERIODS_PER_YEAR,
        )

    assert transform_name in {"lever", "delever"}
    pd.testing.assert_frame_equal(returns, original_returns, check_exact=True)
    pd.testing.assert_series_equal(funding, original_funding, check_exact=True)


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
@pytest.mark.parametrize(
    ("nullable", "empty"),
    ((False, False), (True, False), (False, True), (True, True)),
    ids=("float64-all-missing", "nullable-all-missing", "float64-empty", "nullable-empty"),
)
def test_leverage_transforms_accept_empty_and_all_missing_financing_series(
    transform_name: str,
    transform: _LeverageTransform,
    nullable: bool,
    empty: bool,
) -> None:
    """Preserve unavailable funding as an owned all-missing result without warnings."""
    returns = _unlevered_returns()
    if empty:
        index = pd.DatetimeIndex([])
        values = pd.array((), dtype=pd.Float64Dtype()) if nullable else np.asarray((), dtype=float)
    else:
        index = _DATES[[0, 2]]
        values = (
            pd.array((pd.NA, pd.NA), dtype=pd.Float64Dtype())
            if nullable
            else np.asarray((np.nan, np.nan), dtype=float)
        )
    funding = pd.Series(values, index=index, name="Annual funding")
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

    assert transform_name in {"lever", "delever"}
    assert isinstance(actual, pd.DataFrame)
    assert actual.isna().all(axis=None)
    pd.testing.assert_index_equal(actual.index, returns.index)
    pd.testing.assert_index_equal(actual.columns, returns.columns)
    pd.testing.assert_frame_equal(returns, original_returns, check_exact=True)
    pd.testing.assert_series_equal(funding, original_funding, check_exact=True)


# =============================================================================
# Financing-Series date-axis domain
# =============================================================================


def _invalid_index_case(kind: str) -> tuple[pd.DataFrame, pd.Series, str]:
    """Create one incompatible return/funding index pair and its expected error.

    Args:
        kind: Internal fixture identifier for one invalid index-domain boundary.

    Returns:
        Return panel, financing Series, and stable expected error pattern.

    Raises:
        ValueError: If an unsupported internal fixture identifier is requested.
    """
    returns = _unlevered_returns()
    funding = pd.Series((0.12, 0.24), index=_DATES[[0, 2]], name="Annual funding")

    if kind == "financing-string":
        funding.index = pd.Index(("2024-01-31", "2024-03-31"))
        error = _DATETIME_INDEX_ERROR
    elif kind == "financing-integer":
        funding.index = pd.Index((1, 2))
        error = _DATETIME_INDEX_ERROR
    elif kind == "both-string":
        returns.index = pd.Index(("a", "b", "c", "d"))
        funding.index = pd.Index(("a", "c"))
        error = _DATETIME_INDEX_ERROR
    elif kind == "financing-nat":
        funding.index = pd.DatetimeIndex((_DATES[0], pd.NaT))
        error = _NAT_INDEX_ERROR
    elif kind == "returns-nat":
        returns.index = pd.DatetimeIndex((_DATES[0], pd.NaT, _DATES[2], _DATES[3]))
        error = _NAT_INDEX_ERROR
    elif kind == "aware-returns-naive-financing":
        returns.index = pd.DatetimeIndex(returns.index).tz_localize("UTC")
        error = _TIMEZONE_ERROR
    elif kind == "naive-returns-aware-financing":
        funding.index = pd.DatetimeIndex(funding.index).tz_localize("UTC")
        error = _TIMEZONE_ERROR
    else:
        raise ValueError(f"unsupported invalid index fixture: {kind}")
    return returns, funding, error


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
@pytest.mark.parametrize(
    "invalid_kind",
    (
        "financing-string",
        "financing-integer",
        "both-string",
        "financing-nat",
        "returns-nat",
        "aware-returns-naive-financing",
        "naive-returns-aware-financing",
    ),
)
def test_leverage_transforms_reject_incompatible_financing_index_domains(
    transform_name: str,
    transform: _LeverageTransform,
    invalid_kind: str,
) -> None:
    """Reject undefined or incompatible chronology before point-in-time funding alignment."""
    returns, funding, error = _invalid_index_case(invalid_kind)
    original_returns = returns.copy()
    original_funding = funding.copy()

    with pytest.raises(ValueError, match=error):
        transform(
            returns=returns,
            leverage=_LEVERAGE,
            financing_rate=funding,
            periods_per_year=_PERIODS_PER_YEAR,
        )

    assert transform_name in {"lever", "delever"}
    pd.testing.assert_frame_equal(returns, original_returns, check_exact=True)
    pd.testing.assert_series_equal(funding, original_funding, check_exact=True)


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
@pytest.mark.parametrize(
    "funding_timezone",
    ("UTC", "America/New_York"),
    ids=("same-aware-timezone", "different-aware-timezone"),
)
def test_leverage_transforms_align_compatible_timezone_aware_financing(
    transform_name: str,
    transform: _LeverageTransform,
    funding_timezone: str,
) -> None:
    """Align equal absolute instants while preserving the timezone and labels of returns."""
    returns, expected = _transform_case(transform_name, missing_financing=False)
    aware_returns_index = _DATES.tz_localize("UTC")
    returns.index = aware_returns_index
    expected.index = aware_returns_index
    funding_index = aware_returns_index[[0, 2]].tz_convert(funding_timezone)
    funding = pd.Series((0.12, 0.24), index=funding_index, name="Annual funding")
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
# Zero-leverage validation boundary
# =============================================================================


@pytest.mark.parametrize(("transform_name", "transform"), _TRANSFORMS, ids=("lever", "delever"))
def test_leverage_transforms_keep_zero_identity_before_financing_validation(
    transform_name: str,
    transform: _LeverageTransform,
) -> None:
    """Keep invalid financing irrelevant when leverage makes its contribution exactly zero."""
    returns = _unlevered_returns()
    funding = pd.Series((np.inf, pd.NA), index=("bad", "worse"), name="Annual funding")
    original_funding = funding.copy()

    for transform_returns, _ in _case_variants(returns, returns):
        original_returns = transform_returns.copy()
        actual = transform(
            returns=transform_returns,
            leverage=0.0,
            financing_rate=funding,
            periods_per_year=_PERIODS_PER_YEAR,
        )

        assert transform_name in {"lever", "delever"}
        assert actual is not transform_returns
        _assert_pandas_equal(actual, original_returns)
        _assert_pandas_equal(transform_returns, original_returns)
        pd.testing.assert_series_equal(funding, original_funding, check_exact=True)
