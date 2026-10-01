"""Regression tests for modern semi-month annualization aliases.

Semi-month start and end offsets produce two observations per calendar month, but their alternating
day spacing prevents pandas from inferring the cadence from dates alone. Direct aliases and indexes
that retain valid frequency metadata must therefore use the independently counted 24 periods per
year without changing the legacy QIS ``SM`` or metadata-free fallback contracts.
"""

from typing import cast
import warnings

import numpy as np
import pandas as pd
import pytest

import qis


_MONTHS_PER_YEAR = 12
_SEMI_MONTH_PERIODS = 2
_PERIODS_PER_YEAR = _MONTHS_PER_YEAR * _SEMI_MONTH_PERIODS

_DIRECT_ALIASES: tuple[tuple[str, int], ...] = (
    ("SME", 1),
    ("sme-15", 1),
    ("SMS", 1),
    ("sms-15", 1),
    ("2SME-15", 2),
    ("3SMS-10", 3),
)


@pytest.mark.parametrize(("frequency", "multiplier"), _DIRECT_ALIASES)
def test_get_annualization_factor_semimonth_aliases_use_calendar_cadence(
    frequency: str,
    multiplier: int,
) -> None:
    """Divide the independently counted semi-month cadence by the alias multiplier.

    Args:
        frequency: Modern start or end alias, with an optional anchor and multiplier.
        multiplier: Number of semi-month periods represented by one observation.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = qis.get_annualization_factor(frequency)

    assert actual == _PERIODS_PER_YEAR / multiplier


@pytest.mark.parametrize(
    ("frequency", "expected"),
    (("SME-15", 24.0), ("SMS-15", 24.0), ("2SME-15", 12.0)),
)
def test_infer_annualisation_factor_from_df_uses_semimonth_metadata(
    frequency: str,
    expected: float,
) -> None:
    """Retain a valid semi-month cadence that alternating date gaps cannot infer.

    Args:
        frequency: Valid pandas semi-month frequency for the index metadata.
        expected: Independently counted observations per year.
    """
    dates = pd.date_range("2024-01-01", periods=24, freq=frequency)
    data = pd.Series(np.arange(len(dates), dtype=float), index=dates, name="return")

    assert pd.infer_freq(dates) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = qis.infer_annualisation_factor_from_df(data)

    assert actual == expected


def test_compute_ewm_vol_semimonth_inference_uses_calendar_cadence() -> None:
    """Scale public volatility by the independently counted semi-month factor."""
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="SME-15")
    returns = pd.Series(np.linspace(-0.02, 0.03, len(dates)), index=dates, name="Strategy")
    periodic = cast(pd.Series, qis.compute_ewm_vol(returns, span=3, annualize=False))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = cast(pd.Series, qis.compute_ewm_vol(returns, span=3, annualize=True))

    assert len(dates) == _PERIODS_PER_YEAR
    expected = periodic.multiply(np.sqrt(_PERIODS_PER_YEAR))
    pd.testing.assert_series_equal(actual, expected, rtol=1e-14, atol=0.0)


def test_infer_annualisation_factor_from_df_preserves_metadata_free_fallback() -> None:
    """Keep the documented warning and business-day fallback without frequency metadata."""
    dates = pd.date_range("2024-01-01", periods=8, freq="SME-15")
    metadata_free = pd.DatetimeIndex(dates.to_numpy())
    data = pd.Series(np.arange(len(dates), dtype=float), index=metadata_free)

    with pytest.warns(UserWarning, match="cannot infer None"):
        actual = qis.infer_annualisation_factor_from_df(data)

    assert actual == 252


@pytest.mark.parametrize(("frequency", "expected"), (("SM", 26.0), ("2W", 26.0), ("ME", 12.0)))
def test_get_annualization_factor_semimonth_fix_preserves_neighboring_aliases(
    frequency: str,
    expected: float,
) -> None:
    """Preserve the legacy QIS alias and neighboring biweekly and monthly factors.

    Args:
        frequency: Existing neighboring alias.
        expected: Existing periods-per-year factor.
    """
    assert qis.get_annualization_factor(frequency) == expected


@pytest.mark.parametrize("frequency", ("SME-28", "0SME-15"))
def test_get_annualization_factor_rejects_invalid_semimonth_forms(frequency: str) -> None:
    """Keep invalid anchors and non-positive multipliers on the warning fallback.

    Args:
        frequency: Structurally semi-month spelling rejected by the pandas offset contract.
    """
    with pytest.warns(UserWarning, match=f"Unknown frequency '{frequency}'"):
        actual = qis.get_annualization_factor(frequency)

    assert actual == 1.0
