"""The principal-overlay mixes are invested from the first price date.

``create_overlay_portfolio_curve`` backtested each mix with ``rebalancing_freq`` only, so the
backtest held no position until the first scheduled rebalancing date: a history that starts
between quarter-ends had a flat nav, a zero return, in its first quarter in every mix, and the
zero-weight mix was not the principal portfolio.
"""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
import qis
from qis.datasets.synthetic import generate_synthetic_universe
from qis.portfolio.smart_diversification import create_overlay_portfolio_curve


@pytest.fixture(scope='module')
def navs():
    """A principal and an overlay nav that start on 2 January 2012, between quarter-ends."""
    universe = generate_synthetic_universe(start='2012-01-02', end='2016-12-30', apply_quirks=False)
    return universe.benchmark_prices['SBM_6040'], universe.prices['SBD_TSY']


def test_zero_weight_mix_is_the_principal_from_the_first_date(navs):
    principal, overlay = navs
    mixes = create_overlay_portfolio_curve(principal_nav=principal, overlay_nav=overlay)
    first = mixes.iloc[:, 0]
    np.testing.assert_allclose((first / first.iloc[0]).to_numpy(),
                               (principal / principal.iloc[0]).to_numpy(), rtol=1e-12)


def test_funded_mix_earns_the_weighted_return_in_its_first_quarter(navs):
    principal, overlay = navs
    mixes = create_overlay_portfolio_curve(principal_nav=principal, overlay_nav=overlay,
                                           is_principal_weight_fixed=False)
    quarter_end = pd.Timestamp('2012-03-30')
    growth = mixes.loc[quarter_end] / mixes.iloc[0]
    expected = [(1.0 - x) * principal.loc[quarter_end] / principal.iloc[0]
                + x * overlay.loc[quarter_end] / overlay.iloc[0] for x in np.linspace(0, 1, 11)]
    np.testing.assert_allclose(growth.to_numpy(), expected, rtol=1e-12)
    assert (mixes.iloc[1] != mixes.iloc[0]).all()


def test_start_on_a_calendar_quarter_end_is_unchanged(navs):
    """A history that starts on a quarter-end business day was invested from its first date."""
    principal, overlay = navs
    # a calendar quarter-end on a weekday; the index's business-quarter flags would also pick a
    # Friday before a weekend quarter-end, which the schedule rolls to the next Monday
    start = next(date for date in principal.index if pd.Timestamp(date.date()).is_quarter_end)
    indicators = qis.generate_rebalancing_indicators(df=principal.loc[start:].to_frame(), freq='QE')
    assert bool(indicators.iloc[0])  # the old schedule already rebalanced on this first date
    mixes = create_overlay_portfolio_curve(principal_nav=principal.loc[start:],
                                           overlay_nav=overlay.loc[start:])
    reference = qis.backtest_model_portfolio(
        prices=pd.concat([principal.loc[start:], overlay.loc[start:]], axis=1),
        weights=np.array([1.0, 0.5]), rebalancing_freq='QE').get_portfolio_nav()
    np.testing.assert_allclose(mixes.iloc[:, 5].to_numpy(), reference.to_numpy(), rtol=1e-12)


def test_curve_ends_at_the_standalone_points(navs):
    """The funded curve runs from the principal's point to the overlay's standalone point."""
    principal, overlay = navs
    report = qis.SmartDiversificationReport(overlay_navs=overlay.to_frame(),
                                            principal_nav=principal)
    curve = report.compute_smart_diversification_curve(principal_nav=principal, overlay_nav=overlay,
                                                       is_principal_weight_fixed=False)
    points = report.get_overlay_points(principal_nav=principal)
    np.testing.assert_allclose(curve.iloc[[0, -1]].to_numpy(), points.to_numpy(), rtol=1e-10)
