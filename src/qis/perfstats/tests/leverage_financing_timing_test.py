"""Leverage financing over a period uses the rate known at its start.

``lever_returns`` and ``delever_returns`` charge ``L * f`` per period, with ``f`` the annual
financing rate divided by the periods per year. For the period ``(t-1, t]`` the rate is the one
known at the previous return date ``t-1``: the latest quote dated on or before it. The first
return date has no previous return date in the sample, so it uses the latest quote dated strictly
before it and is missing when there is none. A quote dated on a return date applies from the
next period, the convention of ``compute_excess_returns`` and the backtester's funding leg.
"""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.perfstats.returns import delever_returns, lever_returns

_DATES = pd.date_range('2024-01-31', periods=4, freq='ME')
_RETURNS = pd.Series([0.02, -0.01, 0.03, 0.00], index=_DATES, name='fund')


def _funding() -> pd.Series:
    """Annual rates: 12% quoted before the sample, 24% from the second month-end."""
    return pd.Series([0.12, 0.24], index=pd.DatetimeIndex(['2023-12-29', _DATES[1]]))


def test_lever_returns_charges_the_rate_known_at_the_previous_return_date() -> None:
    """February's quote applies to March, not to February."""
    periodic = np.array([0.01, 0.01, 0.02, 0.02])  # Jan uses the December quote
    expected = 2.0 * _RETURNS - periodic
    actual = lever_returns(_RETURNS, leverage=1.0, financing_rate=_funding(), periods_per_year=12)
    pd.testing.assert_series_equal(actual, expected, atol=1e-14)


def test_delever_returns_inverts_with_the_same_timing() -> None:
    """De-levering uses the same lagged rate, so it inverts ``lever_returns`` exactly."""
    levered = lever_returns(_RETURNS, leverage=1.0, financing_rate=_funding(), periods_per_year=12)
    actual = delever_returns(levered, leverage=1.0, financing_rate=_funding(), periods_per_year=12)
    pd.testing.assert_series_equal(actual, _RETURNS, atol=1e-14)


@pytest.mark.parametrize('transform', [lever_returns, delever_returns])
def test_a_quote_dated_on_the_first_return_date_does_not_finance_it(transform) -> None:
    """Without an earlier quote the first period's rate is unknown, so its result is missing."""
    funding = pd.Series([0.12], index=_DATES[[0]])
    actual = transform(_RETURNS, leverage=1.0, financing_rate=funding, periods_per_year=12)
    assert np.isnan(actual.iloc[0])
    assert np.all(np.isfinite(actual.iloc[1:]))


def test_daily_quotes_on_a_monthly_grid_use_the_previous_month_end_quote() -> None:
    """A rate change in the middle of February first applies to March."""
    days = pd.bdate_range('2023-12-01', '2024-04-30')
    funding = pd.Series(np.where(days < pd.Timestamp('2024-02-15'), 0.12, 0.36), index=days)
    actual = lever_returns(_RETURNS, leverage=1.0, financing_rate=funding, periods_per_year=12)
    periodic = np.array([0.01, 0.01, 0.03, 0.03])
    pd.testing.assert_series_equal(actual, 2.0 * _RETURNS - periodic, atol=1e-14)
