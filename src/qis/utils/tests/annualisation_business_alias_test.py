"""Regression tests for modern business-period annualization aliases.

Business month- and quarter-end aliases describe the same observation cadence as their calendar
counterparts. Direct, inferred, conversion, and downstream regime paths must therefore share the
independently counted periods-per-year factors without emitting an unknown-frequency warning.
"""

from typing import cast
import warnings

import numpy as np
import pandas as pd
import pytest

import qis


_BUSINESS_END_ALIASES: tuple[tuple[str, str, float], ...] = (
    ("BME", "ME", 12.0),
    ("bme", "ME", 12.0),
    ("2BME", "2ME", 6.0),
    ("3bme", "3ME", 4.0),
    ("BQE", "QE", 4.0),
    ("bqe-dec", "QE-DEC", 4.0),
    ("2BQE", "2QE", 2.0),
    ("3bqe-dec", "3QE-DEC", 4.0 / 3.0),
)

_INFERRED_ALIASES: tuple[tuple[str, str, float], ...] = (
    ("2BME", "2ME", 6.0),
    ("2BQE", "2QE", 2.0),
)


def _returns_at_frequency(frequency: str) -> pd.Series:
    """Return fixed periodic values on a pandas-inferable date grid.

    Args:
        frequency: Pandas offset alias for the constructed index.

    Returns:
        Deterministic returns indexed at the requested frequency.
    """
    return pd.Series(
        (0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.012, -0.008),
        index=pd.date_range("2020-01-01", periods=8, freq=frequency),
        name="Strategy",
    )


@pytest.mark.parametrize(("business", "calendar", "expected"), _BUSINESS_END_ALIASES)
def test_get_annualization_factor_business_end_aliases_match_calendar_cadence(
    business: str,
    calendar: str,
    expected: float,
) -> None:
    """Use the independently counted calendar cadence for each equivalent business alias.

    Args:
        business: Modern business-period alias under test.
        calendar: Equivalent calendar-period construction.
        expected: Independently counted observations per year.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = qis.get_annualization_factor(business)

    assert actual == expected
    assert actual == qis.get_annualization_factor(calendar)


@pytest.mark.parametrize(("business", "calendar", "expected"), _INFERRED_ALIASES)
def test_compute_ewm_vol_business_end_inference_matches_calendar_cadence(
    business: str,
    calendar: str,
    expected: float,
) -> None:
    """Propagate pandas-inferred business multipliers to public annualized volatility.

    Args:
        business: Business-period grid whose inferred alias retains the business token.
        calendar: Equivalent calendar-period grid.
        expected: Independently counted observations per year.
    """
    business_returns = _returns_at_frequency(business)
    calendar_returns = _returns_at_frequency(calendar)
    periodic = qis.compute_ewm_vol(business_returns, span=3, annualize=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = qis.compute_ewm_vol(business_returns, span=3, annualize=True)

    assert isinstance(periodic, pd.Series)
    assert isinstance(actual, pd.Series)
    expected_vol = periodic.multiply(np.sqrt(expected))
    calendar_vol = cast(
        pd.Series,
        qis.compute_ewm_vol(calendar_returns, span=3, annualize=True),
    ).set_axis(business_returns.index)
    pd.testing.assert_series_equal(actual, expected_vol, rtol=1e-14, atol=0.0)
    pd.testing.assert_series_equal(actual, calendar_vol, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize(
    ("business", "calendar"),
    (("BQE", "QE"), ("2BQE", "2QE"), ("2BME", "2ME")),
)
def test_get_annualisation_conversion_factor_business_alias_matches_calendar_peer(
    business: str,
    calendar: str,
) -> None:
    """Return one when converting between equal business and calendar cadences.

    Args:
        business: Modern business-period alias under test.
        calendar: Equivalent calendar-period alias.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = qis.get_annualisation_conversion_factor(business, calendar)

    assert actual == 1.0


def test_compute_regime_avg_uses_business_quarter_end_factor() -> None:
    """Scale public regime contributions with four business-quarter observations per year."""
    sampled_returns = pd.DataFrame(
        {
            "Asset": (0.01, 0.03, -0.02, 0.02),
            "regime": ("Up", "Up", "Down", "Down"),
        }
    )
    original = sampled_returns.copy(deep=True)
    expected_up_contribution = np.mean((0.01, 0.03)) * 4 * 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, actual, _ = qis.compute_regime_avg(
            sampled_returns,
            freq="BQE",
            is_report_pa_returns=False,
            regime_ids=["Up", "Down"],
        )

    actual_asset = cast(pd.Series, actual.loc["Asset"])
    np.testing.assert_allclose(
        actual_asset.to_numpy(dtype=float),
        np.array((expected_up_contribution, 0.0), dtype=float),
        rtol=1e-14,
        atol=0.0,
    )
    pd.testing.assert_frame_equal(sampled_returns, original, check_exact=True)
