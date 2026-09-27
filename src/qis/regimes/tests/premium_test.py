"""The premium table and its bootstrap against direct computation and the qis decomposition."""

import numpy as np
import pandas as pd
import pytest

from qis.datasets.synthetic import generate_synthetic_universe
from qis.perfstats.regime_classifier import (
    BenchmarkReturnsQuantilesRegime,
    compute_regime_sharpe_decomposition,
)
from qis.regimes import (
    compute_regime_kappa,
    compute_regime_premium_bootstrap,
    compute_regime_premium_table,
    create_sampled_returns_with_regime_id,
)

ASSETS = ['SBM_6040', 'SBD_TSY', 'SCM_GLD', 'SAL_HF']


@pytest.fixture(scope='module')
def sampled():
    """Quarterly returns of a clean synthetic panel with the one-sigma regimes of its benchmark."""
    universe = generate_synthetic_universe(start='2005-01-03', end='2025-12-31', apply_quirks=False)
    prices = pd.concat([universe.benchmark_prices, universe.prices], axis=1)[ASSETS]
    return BenchmarkReturnsQuantilesRegime(freq='QE').compute_sampled_returns_with_regime_id(
        prices=prices, benchmark='SBM_6040')


def test_table_matches_a_direct_computation(sampled):
    """Every column of the table from numpy on the classified periods."""
    table = compute_regime_premium_table(sampled, benchmark='SBM_6040', af=4.0)
    data = sampled.dropna(subset=['regime'])
    labels = data['regime'].astype(str).to_numpy()
    kappa = compute_regime_kappa(af=4.0)
    for asset in ASSETS:
        r, b = data[asset].to_numpy(), data['SBM_6040'].to_numpy()
        sigma = np.std(r, ddof=1)
        sharpe = 2.0 * np.mean(r) / sigma
        rho = np.corrcoef(r, b)[0, 1]
        bear = 2.0 * np.mean(labels == 'Bear') * np.mean(r[labels == 'Bear']) / sigma
        row = table.loc[asset]
        assert np.isclose(row['sharpe'], sharpe, rtol=1e-12)
        assert np.isclose(row['rho'], rho, rtol=1e-12)
        assert np.isclose(row['bear_sharpe'], bear, rtol=1e-12)
        assert np.isclose(row['convexity_premium'], bear - (0.16 * sharpe - kappa * rho),
                          atol=1e-12)
    assert table.loc['SBM_6040', 'cp_star'] == 0.0


def test_contributions_equal_the_qis_decomposition_and_add_up(sampled):
    """On a panel without gaps the table reproduces compute_regime_sharpe_decomposition."""
    table = compute_regime_premium_table(sampled, benchmark='SBM_6040', af=4.0)
    rets = sampled.dropna(subset=['regime']).drop(columns='regime')
    decomposition = compute_regime_sharpe_decomposition(returns=rets,
                                                        benchmark_returns=rets['SBM_6040'], af=4.0)
    for regime in ('Bear', 'Normal', 'Bull'):
        np.testing.assert_allclose(table[f"{regime.lower()}_sharpe"],
                                   decomposition[f"{regime}-Sharpe"], rtol=1e-12)
    parts = table[['bear_sharpe', 'normal_sharpe', 'bull_sharpe']].sum(axis=1)
    np.testing.assert_allclose(parts, table['sharpe'], rtol=1e-12)


def test_columns_of_the_default_and_of_a_five_bucket_partition(sampled):
    """Column names follow the regime ids; the premium columns sit on the lowest bucket."""
    table = compute_regime_premium_table(sampled, benchmark='SBM_6040', af=4.0, nu=6.0)
    assert list(table.columns) == ['sharpe', 'rho', 'ann_vol', 'bear_sharpe', 'normal_sharpe',
                                   'bull_sharpe', 'null_bear_sharpe', 'convexity_premium',
                                   'cp_star', 'bear_return_pa', 'null_bear_sharpe_t',
                                   'convexity_premium_t']
    q = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rets = sampled.drop(columns='regime').dropna()
    quintiles = create_sampled_returns_with_regime_id(rets, benchmark='SBM_6040', q=q)
    table5 = compute_regime_premium_table(quintiles, benchmark='SBM_6040', af=4.0, q=q)
    contributions = [c for c in table5.columns if c.endswith('_sharpe')][:5]
    assert contributions == [f'q{i}_sharpe' for i in range(1, 6)]
    assert 'null_q1_sharpe' in table5.columns and 'q1_return_pa' in table5.columns


def test_labels_of_another_partition_are_rejected(sampled):
    """A table cannot be computed with a q that did not produce the regimes."""
    with pytest.raises(ValueError, match='not the ids'):
        compute_regime_premium_table(sampled, benchmark='SBM_6040', af=4.0, q=[0.0, 0.5, 1.0])


def test_helper_reproduces_the_classifier_labels(sampled):
    """create_sampled_returns_with_regime_id on returns gives the classifier's labels on prices."""
    rets = sampled.drop(columns='regime')
    rebuilt = create_sampled_returns_with_regime_id(rets, benchmark='SBM_6040')
    classified = sampled['regime'].notna()
    assert (rebuilt.loc[classified, 'regime'].astype(str)
            == sampled.loc[classified, 'regime'].astype(str)).all()


def test_bootstrap_is_seeded_and_brackets_its_spread(sampled):
    """Same seed, same draws; a positive standard error inside an ordered interval."""
    rets = sampled.drop(columns='regime').dropna()[['SBM_6040', 'SCM_GLD', 'SAL_HF']]
    first = compute_regime_premium_bootstrap(rets, benchmark='SBM_6040', af=4.0, n_boot=300)
    second = compute_regime_premium_bootstrap(rets, benchmark='SBM_6040', af=4.0, n_boot=300)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.index) == ['SCM_GLD', 'SAL_HF']
    assert (first['premium_se'] > 0.0).all()
    assert (first['premium_ci_low'] < first['premium_ci_high']).all()


def test_bootstrap_needs_five_blocks():
    """Short samples are refused rather than resampled into near copies."""
    rets = pd.DataFrame(np.random.default_rng(1).normal(size=(30, 2)), columns=['B', 'A'])
    with pytest.raises(ValueError, match='common periods'):
        compute_regime_premium_bootstrap(rets, benchmark='B', af=4.0, block_size=8)
