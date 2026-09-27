"""Contracts of the smart-diversification report and its principal-overlay mixes."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import qis  # noqa: E402
from qis.datasets.synthetic import generate_synthetic_universe  # noqa: E402
from qis.perfstats.config import PerfStat  # noqa: E402
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantilesRegime  # noqa: E402
from qis.portfolio.smart_diversification import (  # noqa: E402
    SmartDiversificationReport,
    create_overlay_portfolio_curve,
)

OVERLAYS = ['SBD_TSY', 'SCM_GLD']


@pytest.fixture(scope='module')
def navs():
    """Return a clean synthetic principal nav and two overlay navs on ten years."""
    universe = generate_synthetic_universe(start='2012-01-02', end='2021-12-31', apply_quirks=False)
    principal = universe.benchmark_prices
    if isinstance(principal, pd.DataFrame):
        principal = principal.iloc[:, 0]
    return principal, universe.prices[OVERLAYS]


def test_every_import_path_resolves_to_the_same_objects():
    """The package root, the subpackage and the pre-5.32 module path share one implementation."""
    from qis.portfolio.reports import overlays_smart_diversification as legacy

    assert qis.SmartDiversificationReport is SmartDiversificationReport
    assert legacy.SmartDiversificationReport is SmartDiversificationReport
    assert qis.create_overlay_portfolio_curve is create_overlay_portfolio_curve
    assert legacy.create_overlay_portfolio_curve is create_overlay_portfolio_curve


def test_supplied_regime_classifier_is_kept(navs):
    """A classifier passed by the caller is the one the report uses."""
    principal, overlays = navs
    classifier = BenchmarkReturnsQuantilesRegime(freq='ME')
    report = SmartDiversificationReport(overlay_navs=overlays, principal_nav=principal,
                                        regime_classifier=classifier)
    assert report.regime_classifier is classifier


def test_default_regime_classifier_is_quarterly_one_sigma(navs):
    """Without a classifier the report uses quarterly regimes at the 16%/84% quantiles."""
    principal, overlays = navs
    report = SmartDiversificationReport(overlay_navs=overlays, principal_nav=principal)
    assert isinstance(report.regime_classifier, BenchmarkReturnsQuantilesRegime)
    assert report.regime_classifier.freq == 'QE'
    np.testing.assert_array_equal(report.regime_classifier.q, [0.0, 0.16, 0.84, 1.0])


def test_overlay_curve_has_eleven_mixes_and_the_first_tracks_the_principal(navs):
    """Once invested, the zero-weight mix earns the principal's returns."""
    principal, overlays = navs
    mixes = create_overlay_portfolio_curve(principal_nav=principal, overlay_nav=overlays['SCM_GLD'])
    assert mixes.shape[1] == 11
    assert mixes.columns[0] == 'SCM_GLD 0.00%' and mixes.columns[-1] == 'SCM_GLD 100.00%'
    first = mixes.iloc[:, 0]
    # the backtest holds cash until its first rebalancing date, so compare from the first move
    invested = first.index[(first != first.iloc[0]).to_numpy().argmax() - 1]
    returns = pd.concat([first.pct_change(), principal.pct_change()], axis=1)
    returns = returns.loc[invested:].iloc[1:]
    np.testing.assert_allclose(returns.iloc[:, 0], returns.iloc[:, 1], rtol=0.0, atol=1e-12)


def test_curve_and_points_cover_the_mixes_and_the_overlays(navs):
    """The curve has one row per mix; the points have the principal and each overlay."""
    principal, overlays = navs
    report = SmartDiversificationReport(overlay_navs=overlays, principal_nav=principal)
    curve = report.compute_smart_diversification_curve(principal_nav=principal,
                                                       overlay_nav=overlays['SBD_TSY'])
    points = report.get_overlay_points(principal_nav=principal)
    assert curve.shape == (11, 2)
    assert list(curve.columns) == [PerfStat.BEAR_SHARPE.to_str(), PerfStat.SHARPE_RF0.to_str()]
    assert list(points.index) == [principal.name] + OVERLAYS


def test_curve_rejects_the_same_statistic_on_both_axes(navs):
    """The curve plot needs two different statistics."""
    principal, overlays = navs
    report = SmartDiversificationReport(overlay_navs=overlays, principal_nav=principal)
    with pytest.raises(ValueError, match='cannot be the same'):
        report.plot_smart_diversification_curve(x_var=PerfStat.SHARPE_RF0,
                                                y_var=PerfStat.SHARPE_RF0)


@pytest.mark.parametrize('method', ['plot_smart_diversification_curve',
                                    'plot_smart_diversification_scatter',
                                    'plot_conditional_sharpes'])
def test_drawing_methods_return_a_figure(navs, method):
    """The three frontier exhibits draw on a new figure."""
    principal, overlays = navs
    report = SmartDiversificationReport(overlay_navs=overlays, principal_nav=principal)
    try:
        fig = getattr(report, method)()
        assert isinstance(fig, plt.Figure)
    finally:
        plt.close('all')
