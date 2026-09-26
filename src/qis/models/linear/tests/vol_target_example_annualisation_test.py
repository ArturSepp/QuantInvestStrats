"""The delta-one example helpers annualise business-day volatility with qis's 252, not 260."""
# packages
import inspect
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# qis
import qis

EXAMPLE = Path(__file__).resolve().parents[5] / 'examples/portfolios/strats/qis_delta1.py'


@pytest.fixture(scope='module')
def example():
    if not EXAMPLE.exists():
        pytest.skip('Repository-only example is not distributed with an installed wheel')
    return runpy.run_path(str(EXAMPLE))


@pytest.mark.parametrize('name', ['simulate_vol_target_strats', 'simulate_vol_target_strats_range',
                                  'simulate_trend_strats', 'simulate_trend_strats_range'])
def test_default_annualisation_is_the_business_day_factor(example, name):
    default = inspect.signature(example[name]).parameters['vol_af'].default
    assert default == qis.get_annualization_factor('B') == 252


def test_vol_target_strategy_matches_compute_ra_returns_per_period_target(example):
    rng = np.random.default_rng(3)
    index = pd.bdate_range('2010-01-01', periods=2000)
    log_returns = pd.DataFrame(0.012 * rng.standard_normal((2000, 2)), index=index,
                               columns=['a', 'b'])
    prices = 100.0 * np.exp(log_returns.cumsum())
    weights, navs = example['simulate_vol_target_strats'](prices=prices, vol_span=21,
                                                         vol_target=0.15)
    # the helper is compute_ra_returns on log returns with the per-period target 0.15 / sqrt(252)
    log_r = qis.to_returns(prices=prices, is_log_returns=True)
    _, _, vol = qis.compute_ra_returns(returns=log_r, span=21, vol_target=0.15 / np.sqrt(252.0))
    np.testing.assert_allclose(weights.iloc[1:], (0.15 / np.sqrt(252.0) / vol).iloc[1:],
                               rtol=1e-12)
    realised = navs.pct_change().iloc[100:].std() * np.sqrt(252.0)
    np.testing.assert_allclose(realised, 0.15, atol=0.01)
