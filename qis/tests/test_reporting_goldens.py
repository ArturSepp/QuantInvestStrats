"""
Visual-regression (golden-image) tier for the multi-asset factsheet, via pytest-mpl.

These render the multi-asset report across the reporting-frequency grid and compare the full A4
figure against committed baseline PNGs - catching layout / overlap / clipping / colour drift that
the structural title assertions in test_reporting_conventions.py cannot see.

Baseline images are environment-specific (matplotlib and freetype versions change text rendering),
so generate them in YOUR environment before enabling comparison:

    # 1) create the baselines once (renders and saves them; performs no comparison):
    pytest qis/tests/test_reporting_goldens.py --mpl-generate-path=qis/tests/baseline

    # 2) thereafter, compare future renders against them:
    pytest qis/tests/test_reporting_goldens.py --mpl

Without the --mpl flag (a plain `pytest` run) these tests still execute and pass but perform no
image comparison, so they are safe to leave in the default suite. The RMS `tolerance` below is set
generously for a dense A4 figure; tighten it once baselines are stable in your environment.

Requires:  pip install pytest-mpl
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # deterministic headless rendering
import pytest

import qis
from qis import ReportingFrequency
from qis.portfolio.reports.config import fetch_default_report_kwargs
from qis.portfolio.reports.multi_assets_factsheet import generate_multi_asset_factsheet


def _make_synthetic_prices(n_assets: int = 6, n_years: int = 14, seed: int = 7) -> pd.DataFrame:
    """reproducible correlated-GBM price panel (identical to test_reporting_conventions)."""
    rng = np.random.default_rng(seed)
    n = n_years * 260
    idx = pd.bdate_range(end=pd.Timestamp('2025-12-31'), periods=n)
    mu = rng.uniform(0.03, 0.12, n_assets) / 260.0
    sig = rng.uniform(0.10, 0.30, n_assets) / np.sqrt(260.0)
    rets = mu + sig * (0.5 * rng.standard_normal((n, 1)) + 0.85 * rng.standard_normal((n, n_assets)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f'A{i}' for i in range(n_assets)])


def _multi_asset_figure(reporting_frequency: ReportingFrequency):
    prices = _make_synthetic_prices()
    tp = qis.get_time_period(df=prices)
    kw = fetch_default_report_kwargs(time_period=tp, reporting_frequency=reporting_frequency)
    return generate_multi_asset_factsheet(prices=prices, benchmark='A0', time_period=tp, **kw)


# common pytest-mpl settings: baselines under qis/tests/baseline, generous RMS tolerance, fixed dpi
_MPL_KW = dict(baseline_dir='baseline', tolerance=25, savefig_kwargs={'dpi': 80})


@pytest.mark.mpl_image_compare(filename='multi_asset_daily.png', **_MPL_KW)
def test_golden_multi_asset_daily():
    return _multi_asset_figure(ReportingFrequency.DAILY)


@pytest.mark.mpl_image_compare(filename='multi_asset_weekly.png', **_MPL_KW)
def test_golden_multi_asset_weekly():
    return _multi_asset_figure(ReportingFrequency.WEEKLY)


@pytest.mark.mpl_image_compare(filename='multi_asset_monthly.png', **_MPL_KW)
def test_golden_multi_asset_monthly():
    return _multi_asset_figure(ReportingFrequency.MONTHLY)


@pytest.mark.mpl_image_compare(filename='multi_asset_quarterly.png', **_MPL_KW)
def test_golden_multi_asset_quarterly():
    return _multi_asset_figure(ReportingFrequency.QUARTERLY)
