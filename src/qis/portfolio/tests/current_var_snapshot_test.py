"""Maximum-VaR snapshots retain columnwise values with pandas keyword-only reductions."""
import warnings

import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets import generate_synthetic_universe


@pytest.mark.parametrize('is_grouped', [False, True])
def test_max_var_snapshot_matches_columnwise_reference(is_grouped, monkeypatch):
    """Both reporting branches accept current pandas and match an independent array reduction."""
    universe = generate_synthetic_universe(
        start='2020-01-01', end='2021-12-31', apply_quirks=False
    )
    portfolio = qis.backtest_model_portfolio(
        prices=universe.prices[['SEQ_US', 'SBD_TSY']],
        weights={'SEQ_US': 0.6, 'SBD_TSY': 0.4}, rebalancing_freq='ME',
    )
    portfolio.set_group_data(
        group_data=pd.Series({'SEQ_US': 'Equities', 'SBD_TSY': 'Bonds'}),
        group_order=['Equities', 'Bonds'],
    )
    grouped, instruments = portfolio.compute_portfolio_vars(is_correlated=False)
    panel = grouped if is_grouped else instruments
    expected = pd.Series(np.nanmax(panel.to_numpy(), axis=0), index=panel.columns)
    plotted = []
    monkeypatch.setattr(qis, 'plot_bars', lambda df, **kwargs: plotted.append(df.copy()))
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        portfolio.plot_current_var(
            snapshot_period=qis.SnapshotPeriod.MAX,
            is_grouped=is_grouped, is_correlated=False,
        )
    assert len(plotted) == 1
    pd.testing.assert_series_equal(plotted[0], expected, check_exact=True)
