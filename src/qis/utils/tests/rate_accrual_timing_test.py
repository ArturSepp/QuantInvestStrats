"""Timing contract of the internal rate-accrual helper ``multiply_df_by_dt``.

The rate that accrues over the period ``(t-1, t]`` of a target grid is the latest quote dated on
or before the grid date ``t-1``: the lag counts observations of the target grid, not of the rate
series. The first grid date has no elapsed period, so its accrual is exactly zero whatever the
rate. Every expected value below is hand arithmetic on a 36.5% annual rate, which accrues 0.1%
per calendar day under ACT/365.
"""

import numpy as np
import pandas as pd

from qis.utils.df_ops import multiply_df_by_dt


def test_lag_counts_target_grid_observations_not_rate_quotes() -> None:
    """Daily quotes on a month-end grid use the quote of the previous month-end.

    The rate is 36.5% until 14 February and 73% from 15 February. February (29 days in 2024)
    must accrue the rate known on 31 January, 36.5% * 29 / 365 = 2.9%, and March the rate known
    on 29 February. Lagging the daily series by one of its own quotes would instead charge
    February the quote of 28 February, the second-to-last daily quote of the month.
    """
    daily = pd.bdate_range('2024-01-01', '2024-03-29')
    rates = pd.Series(np.where(daily < pd.Timestamp('2024-02-15'), 0.365, 0.73), index=daily)
    grid = pd.DatetimeIndex(['2024-01-31', '2024-02-29', '2024-03-29'])

    accrual = multiply_df_by_dt(df=rates, dates=grid, lag=1)

    expected = pd.Series([0.0, 0.365 * 29 / 365, 0.73 * 29 / 365], index=grid)
    pd.testing.assert_series_equal(accrual, expected, check_exact=False, rtol=0.0, atol=1e-15)


def test_first_grid_date_accrues_zero_even_without_a_prior_quote() -> None:
    """A rate series that starts on the first grid date leaves no missing accrual.

    The first date has no elapsed period and accrues zero; the second period uses the quote of
    the first date.
    """
    grid = pd.date_range('2024-01-01', periods=3, freq='D')
    rates = pd.Series([0.365, 0.73, 1.095], index=grid)

    accrual = multiply_df_by_dt(df=rates, dates=grid, lag=1)

    pd.testing.assert_series_equal(accrual, pd.Series([0.0, 0.001, 0.002], index=grid),
                                   check_exact=False, rtol=0.0, atol=1e-15)


def test_single_quote_series_is_lagged() -> None:
    """A one-quote rate series is lagged like any other.

    The only quote is dated 3 January. The period ending 3 January started on 2 January, when
    no quote was known, so its accrual is missing; the periods after it accrue 0.1% a day.
    """
    grid = pd.date_range('2024-01-01', periods=5, freq='D')
    rates = pd.Series([0.365], index=pd.DatetimeIndex(['2024-01-03']))

    accrual = multiply_df_by_dt(df=rates, dates=grid, lag=1)

    expected = pd.Series([0.0, np.nan, np.nan, 0.001, 0.001], index=grid)
    pd.testing.assert_series_equal(accrual, expected, check_exact=False, rtol=0.0, atol=1e-15)


def test_zero_lag_uses_the_quote_dated_on_the_grid_date() -> None:
    """Without a lag the period ending t accrues the latest quote on or before t."""
    grid = pd.date_range('2024-01-01', periods=3, freq='D')
    rates = pd.Series([0.365, 0.73, 1.095], index=grid)

    accrual = multiply_df_by_dt(df=rates, dates=grid, lag=0)

    pd.testing.assert_series_equal(accrual, pd.Series([0.0, 0.002, 0.003], index=grid),
                                   check_exact=False, rtol=0.0, atol=1e-15)


def test_caller_rate_series_is_not_modified() -> None:
    """Alignment works on a copy, including the timezone conversion of the rate index."""
    grid = pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC')
    rates = pd.Series([0.365, 0.73, 1.095],
                      index=pd.date_range('2024-01-01', periods=3, freq='D', tz='Europe/Zurich'))
    original = rates.copy(deep=True)

    multiply_df_by_dt(df=rates, dates=grid, lag=1)

    pd.testing.assert_series_equal(rates, original)
