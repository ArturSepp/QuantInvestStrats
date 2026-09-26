"""Contract of the Brownian-bridge interpolation ``qis.interpolate_infrequent_returns``.

The interpolated returns are returned on the pivot index; over each report interval they
compound (simple mode) or sum (log mode) exactly to the reported return; they are point in time;
their sum of squares over an interval is the square-root-of-time share of an EWM variance of the
reported returns; and on a Brownian motion observed quarterly they have the motion's volatility
and no serial correlation. Every reference is computed independently with NumPy.
"""

import numpy as np
import pandas as pd
import pytest

import qis


_BUSINESS_DAYS = pd.bdate_range('2019-01-01', '2023-12-29')


def _pivot(seed: int = 20260725) -> pd.Series:
    """A business-day pivot of Gaussian returns."""
    rng = np.random.default_rng(seed)
    return pd.Series(0.01 * rng.standard_normal(len(_BUSINESS_DAYS)), index=_BUSINESS_DAYS,
                     name='pivot')


def _reported(report_dates: pd.DatetimeIndex, seed: int = 3) -> pd.Series:
    """Quarterly reported simple returns."""
    rng = np.random.default_rng(seed)
    return pd.Series(0.02 + 0.04 * rng.standard_normal(len(report_dates)), index=report_dates,
                     name='fund')


def _interval_sums(log_returns: pd.Series, pivot_index: pd.DatetimeIndex,
                   report_dates: pd.DatetimeIndex) -> np.ndarray:
    """Sum of interpolated log returns over each report interval placed on the pivot grid."""
    positions = pivot_index.searchsorted(report_dates, side='right') - 1
    cumulative = np.r_[0.0, np.cumsum(log_returns.fillna(0.0).to_numpy())]
    return np.diff(cumulative[positions + 1])


def test_simple_returns_compound_exactly_to_each_report_on_the_pivot_index() -> None:
    """Quarter-ends on weekends are placed on the preceding business day, none is lost."""
    quarter_ends = pd.date_range('2019-03-31', '2023-12-31', freq='QE')  # 2019-03-31 is a Sunday
    reported = _reported(quarter_ends)
    reported.iloc[5] = 0.0  # an exactly zero report is an ordinary report

    daily = qis.interpolate_infrequent_returns(infrequent_returns=reported, pivot_returns=_pivot())

    assert daily.index.equals(_BUSINESS_DAYS)
    first_pivot_after_start = _BUSINESS_DAYS[_BUSINESS_DAYS > pd.Timestamp('2019-03-29')][0]
    assert daily.first_valid_index() == first_pivot_after_start
    sums = _interval_sums(np.log1p(daily), _BUSINESS_DAYS, quarter_ends)
    np.testing.assert_allclose(np.expm1(sums), reported.to_numpy()[1:], rtol=0.0, atol=1e-14)
    zero_quarter = daily.loc[quarter_ends[4] + pd.Timedelta(days=1):quarter_ends[5]]
    assert zero_quarter.abs().max() > 1e-4  # the bridge still moves inside a flat quarter


def test_log_returns_sum_exactly_to_each_report() -> None:
    """In log mode the output is in log returns that add up to the reported log returns."""
    quarter_ends = pd.date_range('2019-03-29', '2023-12-29', freq='BQE')
    reported = np.log1p(_reported(quarter_ends))

    daily = qis.interpolate_infrequent_returns(infrequent_returns=reported, pivot_returns=_pivot(),
                                               is_to_log_returns=True)

    sums = _interval_sums(daily, _BUSINESS_DAYS, quarter_ends)
    np.testing.assert_allclose(sums, reported.to_numpy()[1:], rtol=0.0, atol=1e-14)


def test_interpolation_is_point_in_time() -> None:
    """Truncating the inputs at a report date leaves the history up to that date unchanged."""
    quarter_ends = pd.date_range('2019-03-29', '2023-12-29', freq='BQE')
    reported, pivot = _reported(quarter_ends), _pivot()
    cutoff = quarter_ends[9]

    full = qis.interpolate_infrequent_returns(infrequent_returns=reported, pivot_returns=pivot)
    truncated = qis.interpolate_infrequent_returns(infrequent_returns=reported.loc[:cutoff],
                                                   pivot_returns=pivot.loc[:cutoff])

    pd.testing.assert_series_equal(full.loc[:cutoff], truncated, rtol=0.0, atol=0.0)


def test_annualization_factor_sets_no_time_scale() -> None:
    """A monthly factor of 12 on a daily pivot gives the same path as the default 260."""
    quarter_ends = pd.date_range('2019-03-29', '2023-12-29', freq='BQE')
    reported, pivot = _reported(quarter_ends), _pivot()

    default = qis.interpolate_infrequent_returns(infrequent_returns=reported, pivot_returns=pivot)
    monthly = qis.interpolate_infrequent_returns(infrequent_returns=reported, pivot_returns=pivot,
                                                 annualization_factor=12)

    pd.testing.assert_series_equal(default, monthly, rtol=0.0, atol=0.0)


@pytest.mark.parametrize('span', (1, 12))
def test_interval_sum_of_squares_matches_the_ewm_variance(span: int) -> None:
    """Sum of squares over an interval is l**2 / n + v * (n - 1) with an EWM of l**2 / n."""
    quarter_ends = pd.date_range('2019-03-29', '2023-12-29', freq='BQE')
    reported = np.log1p(_reported(quarter_ends))

    daily = qis.interpolate_infrequent_returns(infrequent_returns=reported, pivot_returns=_pivot(),
                                               span=span, is_to_log_returns=True)

    positions = _BUSINESS_DAYS.searchsorted(quarter_ends, side='right') - 1
    counts = np.diff(positions)
    log_returns = reported.to_numpy()[1:]
    decay = 1.0 - 2.0 / (span + 1.0)
    variance, expected = np.nan, []
    for count, log_return in zip(counts, log_returns):
        step = log_return ** 2 / count
        variance = step if np.isnan(variance) else decay * variance + (1.0 - decay) * step
        expected.append(log_return ** 2 / count + variance * (count - 1))
    squares = daily.fillna(0.0).to_numpy() ** 2
    actual = [squares[start + 1:end + 1].sum() for start, end in zip(positions[:-1], positions[1:])]
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    if span == 1:
        np.testing.assert_allclose(actual, log_returns ** 2, rtol=1e-12)


def test_brownian_motion_volatility_and_independence_are_recovered() -> None:
    """Quarterly reports of a 15% Brownian motion interpolate to about 15% and no autocorrelation.

    Forty years of business days give about 10,000 increments, so the lag-one autocorrelation
    has a standard error near 0.01. The previous implementation, which used the standardised
    pivot return as a level deviation, had a lag-one autocorrelation near -0.5.
    """
    rng = np.random.default_rng(1)
    days = pd.bdate_range('1990-01-01', periods=252 * 40)
    true_log = pd.Series(0.15 / np.sqrt(252.0) * rng.standard_normal(len(days)), index=days)
    quarter_ends = pd.Series(1.0, index=days).resample('BQE').last().index
    quarter_ends = quarter_ends[quarter_ends <= days[-1]]
    reported = true_log.cumsum().reindex(quarter_ends).diff().iloc[1:]
    pivot = pd.Series(rng.standard_normal(len(days)), index=days)

    daily = qis.interpolate_infrequent_returns(infrequent_returns=reported, pivot_returns=pivot,
                                               is_to_log_returns=True).dropna()

    assert abs(daily.std() * np.sqrt(252.0) - 0.15) < 0.01
    assert abs(daily.autocorr(1)) < 0.04
