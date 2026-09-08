"""Regression tests for minute-frequency annualization consistency.

Minute aliases represent clock intervals within the active days selected by
``get_annualization_factor``. Equivalent spellings and positive integer multipliers must therefore
share one base minute count, including aliases inferred from pandas indexes. The integration
checks keep public Sharpe and EWM-volatility scaling tied to that same independently calculated
factor.
"""

import numpy as np
import pandas as pd
import pytest

import qis
from qis.utils.annualisation import (
    get_annualization_factor,
    infer_annualisation_factor_from_df,
)


_CALENDAR_DAYS = 365
_MINUTES_PER_DAY = 24 * 60
_TRADING_DAYS = 252

_MINUTE_ALIASES: tuple[tuple[str, int], ...] = (
    ("min", 1),
    ("1min", 1),
    ("T", 1),
    ("MIN", 1),
    ("2min", 2),
    ("5min", 5),
    ("5T", 5),
    ("5MIN", 5),
    ("10min", 10),
    ("15min", 15),
    ("15T", 15),
    ("30min", 30),
    ("60min", 60),
)


def _ten_minute_returns() -> pd.Series:
    """Return a finite varying series on a pandas-inferable ten-minute grid."""
    return pd.Series(
        (-0.010, 0.015, -0.005, 0.020, 0.010, -0.015, 0.005, 0.025),
        index=pd.date_range("2026-01-02 09:30", periods=8, freq="10min"),
        name="Strategy",
    )


def _same_returns_at_frequency(returns: pd.Series, frequency: str) -> pd.DataFrame:
    """Relabel the deterministic values on an equivalent comparison grid.

    Args:
        returns: Finite periodic values used by the primary integration path.
        frequency: Comparison grid applied without changing those values.

    Returns:
        One-column return frame on the requested regular grid.
    """
    return pd.DataFrame(
        returns.to_numpy(dtype=float),
        index=pd.date_range(returns.index[0], periods=len(returns), freq=frequency),
        columns=[returns.name],
    )


@pytest.mark.parametrize(("is_calendar", "active_days"), ((False, 252), (True, 365)))
@pytest.mark.parametrize(("frequency", "multiplier"), _MINUTE_ALIASES)
def test_get_annualization_factor_minute_aliases_share_clock_basis(
    frequency: str,
    multiplier: int,
    is_calendar: bool,
    active_days: int,
) -> None:
    """Divide one clock-minute basis by every supported alias multiplier.

    Args:
        frequency: Direct modern, legacy, or case-varied minute spelling.
        multiplier: Number of minutes represented by one observation.
        is_calendar: Whether all calendar days are active.
        active_days: Independently selected number of active days per year.
    """
    expected = active_days * _MINUTES_PER_DAY / multiplier

    actual = get_annualization_factor(frequency, is_calendar=is_calendar)

    assert actual == expected


@pytest.mark.parametrize(
    ("is_calendar", "default_trading_days", "active_days"),
    ((False, 260, 260), (True, 260, _CALENDAR_DAYS)),
)
def test_get_annualization_factor_minute_aliases_respect_selected_active_days(
    is_calendar: bool,
    default_trading_days: int,
    active_days: int,
) -> None:
    """Apply a custom trading-day count only outside calendar mode.

    Args:
        is_calendar: Whether the calendar-day override is active.
        default_trading_days: Caller-supplied active trading days.
        active_days: Independently expected active-day count.
    """
    expected = active_days * _MINUTES_PER_DAY / 10

    actual = get_annualization_factor(
        "10min",
        is_calendar=is_calendar,
        default_trading_days=default_trading_days,
    )

    assert actual == expected


@pytest.mark.parametrize(
    ("frequency", "expected"),
    (("B", 252.0), ("D", 365.0), ("h", 252.0 * 24), ("2h", 252.0 * 12), ("ME", 12.0)),
)
def test_get_annualization_factor_minute_fix_preserves_other_frequency_factors(
    frequency: str,
    expected: float,
) -> None:
    """Keep neighboring business, calendar, hourly, and monthly factors unchanged.

    Args:
        frequency: Unchanged control frequency.
        expected: Existing independently calculated periods per year.
    """
    assert get_annualization_factor(frequency) == expected


@pytest.mark.parametrize(
    ("frequency", "multiplier"),
    (("min", 1), ("2min", 2), ("5min", 5), ("10min", 10), ("30min", 30), ("60min", 60)),
)
def test_infer_annualisation_factor_from_df_uses_minute_clock_basis(
    frequency: str,
    multiplier: int,
) -> None:
    """Propagate pandas' modern inferred aliases through the same minute contract.

    Args:
        frequency: Valid pandas frequency used to construct the index.
        multiplier: Number of minutes represented by one observation.
    """
    data = pd.Series(
        np.arange(8, dtype=float),
        index=pd.date_range("2026-01-02", periods=8, freq=frequency),
        name="return",
    )
    expected = _TRADING_DAYS * _MINUTES_PER_DAY / multiplier

    assert infer_annualisation_factor_from_df(data) == expected


def test_minute_inference_scales_public_ewm_sharpe() -> None:
    """Scale public EWM Sharpe by the independent five-to-ten-minute ratio."""
    returns = _ten_minute_returns()
    five_minute_returns = _same_returns_at_frequency(returns, "5min")
    five_minute_sharpe = qis.compute_ewm_sharpe(five_minute_returns, span=3)
    expected = five_minute_sharpe.multiply(np.sqrt(5 / 10)).set_axis(returns.index)

    actual = qis.compute_ewm_sharpe(returns.to_frame(), span=3)

    pd.testing.assert_frame_equal(actual, expected, rtol=1e-14, atol=0.0)


def test_minute_inference_scales_public_ewm_volatility() -> None:
    """Annualize public EWM volatility from its independently scaled periodic result."""
    returns = _ten_minute_returns()
    expected_factor = _TRADING_DAYS * _MINUTES_PER_DAY / 10
    periodic = qis.compute_ewm_vol(returns, span=3, annualize=False)
    actual = qis.compute_ewm_vol(returns, span=3, annualize=True)

    assert isinstance(periodic, pd.Series)
    assert isinstance(actual, pd.Series)
    expected = periodic.multiply(np.sqrt(expected_factor))
    pd.testing.assert_series_equal(actual, expected, rtol=1e-14, atol=0.0)
