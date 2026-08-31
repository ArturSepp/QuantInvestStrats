"""Regression coverage for finite-observation volatility estimator selection.

``estimate_vol`` uses RMS for small samples so it does not subtract a noisy sample mean, then
switches to sample standard deviation once 20 observations are available. Missing rows carry no
return information and therefore must not select the estimator. The boundary is evaluated per
column so ragged histories in one DataFrame can use different estimators without affecting one
another.

The direct fixtures exercise the exact 0, 1, 2, 19, 20, and 21 observation boundaries, unchanged
fully observed samples, missing-row placement, ordinary and nullable pandas storage, mixed column
states, and caller ownership. Expected values use closed-form sums for the literal sequence
``0.001 * [1, ..., n]``. A public ``compute_sampled_vols`` fixture independently compounds five
literal returns into prices and annualizes their RMS by ``sqrt(252)``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

# qis
from qis.perfstats.returns import compute_sampled_vols, estimate_vol


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_BASE_RETURNS = np.array((0.01, -0.01, 0.02, -0.02, 0.03), dtype=float)
_BOUNDARY_RETURNS = 0.001 * np.arange(1.0, 22.0)
_COLUMN_COUNTS = (21, 20, 19, 2, 1, 0)
_COLUMNS = tuple(f"n{count}" for count in _COLUMN_COUNTS)
_EXPECTED_BY_COUNT = {
    0: np.nan,
    1: 0.001,
    2: 0.0015811388300841897,
    19: 0.011401754250991379,
    20: 0.005916079783099616,
    21: 0.006204836822995429,
}
_JANUARY_DATES = pd.bdate_range("2022-01-03", "2022-01-31")
_MIXED_DATES = pd.bdate_range("2022-01-03", periods=25)
_NAME = "Late Start"
_RTOL = 1.0e-12


def _as_nullable_frame(frame: pd.DataFrame, nullable: bool) -> pd.DataFrame:
    """Optionally convert a frame to pandas nullable floating storage.

    Args:
        frame: Ordinary floating fixture.
        nullable: Whether to use pandas ``Float64`` storage.

    Returns:
        The fixture in the requested floating representation.
    """
    return frame.astype(pd.Float64Dtype()) if nullable else frame


def _as_nullable_series(series: pd.Series, nullable: bool) -> pd.Series:
    """Optionally convert a Series to pandas nullable floating storage.

    Args:
        series: Ordinary floating fixture.
        nullable: Whether to use pandas ``Float64`` storage.

    Returns:
        The fixture in the requested floating representation.
    """
    return series.astype(pd.Float64Dtype()) if nullable else series


def _boundary_values(num_observations: int, total_rows: int) -> np.ndarray:
    """Place the requested finite boundary sample before trailing missing rows.

    Args:
        num_observations: Number of finite returns to include.
        total_rows: Total output length after missing padding.

    Returns:
        One-dimensional return sample with the requested finite count.
    """
    values = np.full(total_rows, np.nan, dtype=float)
    values[:num_observations] = _BOUNDARY_RETURNS[:num_observations]
    return values


def _mixed_count_frame(nullable: bool) -> pd.DataFrame:
    """Create all estimator-selection states in one DataFrame call.

    Args:
        nullable: Whether to use pandas ``Float64`` storage.

    Returns:
        Mixed panel with 21, 20, 19, 2, 1, and 0 finite returns by column.
    """
    values = np.full((len(_MIXED_DATES), len(_COLUMN_COUNTS)), np.nan, dtype=float)
    for column, count in enumerate(_COLUMN_COUNTS):
        values[:count, column] = _BOUNDARY_RETURNS[:count]
    frame = pd.DataFrame(values, index=_MIXED_DATES, columns=_COLUMNS)
    return _as_nullable_frame(frame, nullable)


def _padding_variant(placement: str) -> np.ndarray:
    """Place the same five returns in an unpadded or missing-padded sample.

    Args:
        placement: One of ``none``, ``leading``, ``trailing``, or ``interior``.

    Returns:
        Return sample with unchanged finite observations.

    Raises:
        ValueError: If ``placement`` is unsupported.
    """
    if placement == "none":
        return _BASE_RETURNS.copy()
    values = np.full(25, np.nan, dtype=float)
    if placement == "leading":
        values[-len(_BASE_RETURNS) :] = _BASE_RETURNS
    elif placement == "trailing":
        values[: len(_BASE_RETURNS)] = _BASE_RETURNS
    elif placement == "interior":
        values[[0, 3, 9, 15, 24]] = _BASE_RETURNS
    else:
        raise ValueError(f"unsupported placement={placement!r}")
    return values


def _late_start_prices(nullable: bool) -> pd.Series:
    """Compound five literal returns after fifteen leading missing prices.

    Args:
        nullable: Whether to use pandas ``Float64`` storage.

    Returns:
        January price Series with five finite returns in a 21-row window.
    """
    prices = np.full(len(_JANUARY_DATES), np.nan, dtype=float)
    first_price = len(_JANUARY_DATES) - len(_BASE_RETURNS) - 1
    prices[first_price] = 100.0
    prices[first_price + 1 :] = 100.0 * np.cumprod(1.0 + _BASE_RETURNS)
    series = pd.Series(prices, index=_JANUARY_DATES, name=_NAME)
    return _as_nullable_series(series, nullable)


# =============================================================================
# Direct estimator boundaries
# =============================================================================


@pytest.mark.parametrize(
    ("num_observations", "expected"),
    tuple(_EXPECTED_BY_COUNT.items()),
)
def test_estimate_vol_selects_estimator_by_finite_observation_count(
    num_observations: int,
    expected: float,
) -> None:
    """Select RMS or sample spread from finite count despite 25 total rows.

    For ``x_k = 0.001 * k``, independently summing squares gives
    ``RMS(n) = 0.001 * sqrt((n + 1) * (2n + 1) / 6)``. Subtracting the arithmetic mean gives
    ``sample_std(n) = 0.001 * sqrt(n * (n + 1) / 12)``. The parameterized references apply RMS
    below 20 finite observations and sample standard deviation from 20 onward.

    Args:
        num_observations: Number of finite values before trailing missing rows.
        expected: Independently calculated volatility under the finite-count contract.
    """
    sampled_returns = _boundary_values(num_observations, total_rows=25)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = estimate_vol(sampled_returns)

    assert np.asarray(actual).shape == ()
    np.testing.assert_allclose(actual, expected, rtol=_RTOL, equal_nan=True)


@pytest.mark.parametrize("num_observations", (19, 20, 21))
def test_estimate_vol_preserves_fully_observed_threshold_values(
    num_observations: int,
) -> None:
    """Retain accepted RMS and sample-spread values when no rows are missing.

    Args:
        num_observations: Fully observed sample length around the estimator switch.
    """
    sampled_returns = _BOUNDARY_RETURNS[:num_observations]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = estimate_vol(sampled_returns)

    np.testing.assert_allclose(
        actual,
        _EXPECTED_BY_COUNT[num_observations],
        rtol=_RTOL,
    )


@pytest.mark.parametrize("nullable", (False, True), ids=("ordinary", "nullable"))
def test_estimate_vol_selects_each_mixed_dataframe_column_independently(
    nullable: bool,
) -> None:
    """Apply the finite-count boundary independently across one mixed panel.

    Args:
        nullable: Whether the input uses pandas nullable floating storage.
    """
    sampled_returns = _mixed_count_frame(nullable)
    original_returns = sampled_returns.copy()
    expected = np.array([_EXPECTED_BY_COUNT[count] for count in _COLUMN_COUNTS])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = estimate_vol(sampled_returns)

    assert isinstance(actual, np.ndarray)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=_RTOL, equal_nan=True)
    pd.testing.assert_frame_equal(sampled_returns, original_returns)


@pytest.mark.parametrize("nullable", (False, True), ids=("ordinary", "nullable"))
@pytest.mark.parametrize("placement", ("none", "leading", "trailing", "interior"))
def test_estimate_vol_is_invariant_to_missing_row_placement(
    nullable: bool,
    placement: str,
) -> None:
    """Return the same five-observation RMS for every missing-row placement.

    The direct sum of squared literal returns is ``0.0019``; division by five and square root
    gives ``0.019493588689617928`` independently of any missing labels.

    Args:
        nullable: Whether the input uses pandas nullable floating storage.
        placement: Position of the missing padding around the finite returns.
    """
    values = _padding_variant(placement)
    sampled_returns = pd.Series(values, index=pd.bdate_range("2022-01-03", periods=len(values)))
    sampled_returns = _as_nullable_series(sampled_returns, nullable)
    original_returns = sampled_returns.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = estimate_vol(sampled_returns)

    np.testing.assert_allclose(actual, 0.019493588689617928, rtol=_RTOL)
    pd.testing.assert_series_equal(sampled_returns, original_returns)


# =============================================================================
# Public annualized-volatility integration
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("ordinary", "nullable"))
def test_compute_sampled_vols_annualizes_sparse_window_rms(nullable: bool) -> None:
    """Annualize five finite January returns by RMS rather than total window rows.

    The five-return RMS is ``0.019493588689617928``; multiplying by ``sqrt(252)`` gives
    ``0.3094511269974631``. Fifteen leading missing prices make the January window contain 21 rows
    without adding return observations.

    Args:
        nullable: Whether the input uses pandas nullable floating storage.
    """
    prices = _late_start_prices(nullable)
    original_prices = prices.copy()
    expected = pd.Series(
        (0.3094511269974631,),
        index=pd.DatetimeIndex(("2022-01-31",)),
        name=_NAME,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = compute_sampled_vols(
            prices,
            freq_vol="ME",
            include_start_date=True,
            include_end_date=True,
        )

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected, rtol=_RTOL)
    pd.testing.assert_series_equal(prices, original_prices)
