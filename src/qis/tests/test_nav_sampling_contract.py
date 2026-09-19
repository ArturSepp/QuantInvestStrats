"""Shared closed-period sampling contract for public NAV entry points."""

import numpy as np
import pandas as pd

import qis


def test_df_asfreq_short_window_requires_explicit_partial_period() -> None:
    """A sub-period history has no completed boundary unless its end is requested."""
    index = pd.bdate_range("2024-08-05", periods=5, name="date")
    nav = pd.Series(np.arange(1.0, 6.0), index=index, name="NAV")

    completed = qis.df_asfreq(nav, freq="ME")
    with_partial_end = qis.df_asfreq(nav, freq="ME", include_end_date=True)

    expected_completed = nav.iloc[0:0]
    expected_partial = pd.Series(
        [nav.iloc[-1]],
        index=pd.DatetimeIndex([nav.index[-1]], name=nav.index.name),
        name=nav.name,
    )
    pd.testing.assert_series_equal(completed, expected_completed)
    pd.testing.assert_series_equal(with_partial_end, expected_partial)


def test_df_asfreq_same_frequency_preserves_missing_values() -> None:
    """Already-periodic levels retain intentional missing observations."""
    index = pd.date_range("2024-01-31", periods=3, freq="ME", name="date")
    nav = pd.Series((1.0, np.nan, 3.0), index=index, name="NAV")

    actual = qis.df_asfreq(nav, freq="ME")

    pd.testing.assert_series_equal(actual, nav)


def test_public_nav_samplers_share_closed_period_values() -> None:
    """Return, portfolio, and multi-portfolio NAV paths use one level sampler."""
    index = pd.bdate_range("2024-01-02", "2024-03-01", name="date")
    returns = pd.Series(0.01, index=index, name="Strategy")
    returns.iloc[0] = 0.0
    returns.loc[pd.Timestamp("2024-01-31")] = np.nan
    daily_nav = qis.returns_to_nav(returns)
    target_index = pd.date_range(index[0], index[-1], freq="ME", name="date")
    expected = daily_nav.ffill().reindex(target_index, method="ffill")

    from_returns = qis.returns_to_nav(returns, freq="ME")
    portfolio = qis.PortfolioData(nav=daily_nav)
    from_portfolio = portfolio.get_portfolio_nav(freq="ME")
    multi = qis.MultiPortfolioData(portfolio_datas=[portfolio])
    multi.set_navs(freq="ME")
    from_multi = multi.navs.iloc[:, 0]

    pd.testing.assert_series_equal(from_returns, expected)
    pd.testing.assert_series_equal(from_portfolio, expected)
    pd.testing.assert_series_equal(from_multi, expected)
