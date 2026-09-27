"""Tests for the regime premium exhibits drawn from qis.regimes tables."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from qis.plots.derived.regime_premium import (  # noqa: E402
    plot_regime_beta_profiles,
    plot_regime_sharpe_decomposition,
)
from qis.regimes import (  # noqa: E402
    compute_regime_betas,
    compute_regime_betas_bootstrap,
    compute_regime_premium_table,
    create_sampled_returns_with_regime_id,
)


@pytest.fixture(scope='module')
def returns() -> pd.DataFrame:
    """Return ten years of monthly benchmark and asset returns with regime-dependent betas."""
    rng = np.random.default_rng(7)
    b = 0.005 + 0.04 * rng.standard_normal(120)
    tail = b < np.quantile(b, 0.16)
    panel = {'BM': b,
             'CTA 1': np.where(tail, -1.2, 0.1) * b + 0.03 * rng.standard_normal(120),
             'CTA 2': np.where(tail, -0.8, 0.2) * b + 0.03 * rng.standard_normal(120),
             'LS 1': 0.6 * b + 0.02 * rng.standard_normal(120),
             'LS 2': 0.9 * b + 0.02 * rng.standard_normal(120)}
    return pd.DataFrame(panel, index=pd.date_range('2015-01-31', periods=120, freq='ME'))


def test_decomposition_draws_one_bar_per_regime_and_row_with_ticks_and_diamonds(returns):
    """Three stacked segments, a total tick and a null diamond for each table row."""
    sampled = create_sampled_returns_with_regime_id(returns, benchmark='BM')
    table = compute_regime_premium_table(sampled, benchmark='BM', af=12.0)
    try:
        fig = plot_regime_sharpe_decomposition(table, row_separators=[0, 2], title='Decomposition')
        ax = fig.axes[0]
        assert len(ax.patches) == 3 * len(table)
        assert len(ax.collections) == len(table)  # the diamonds
        assert [t.get_text() for t in ax.get_yticklabels()] == list(table.index)
        widths = sorted(round(p.get_width(), 12) for p in ax.patches)
        expected = sorted(round(abs(v), 12) for v in
                          table[['bear_sharpe', 'normal_sharpe', 'bull_sharpe']].to_numpy().ravel())
        assert widths == expected
    finally:
        plt.close('all')


def test_decomposition_reads_quintile_columns_and_draws_on_a_given_axis(returns):
    """Q1 to Q5 contributions are found without naming them; an axis returns no figure."""
    q = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sampled = create_sampled_returns_with_regime_id(returns, benchmark='BM', q=q)
    table = compute_regime_premium_table(sampled, benchmark='BM', af=12.0, q=q)
    fig, ax = plt.subplots()
    try:
        assert plot_regime_sharpe_decomposition(table, ax=ax) is None
        assert len(ax.patches) == 5 * len(table)
    finally:
        plt.close('all')


def test_beta_profiles_draw_one_line_per_group_with_error_bars(returns):
    """Each group gets a mean line with error bars, a range band and a total-beta line."""
    betas = compute_regime_betas_bootstrap(returns, benchmark='BM', af=12.0, n_boot=100)
    sampled = create_sampled_returns_with_regime_id(returns, benchmark='BM')
    totals = compute_regime_betas(sampled, benchmark='BM', af=12.0)['beta_total']
    groups = pd.Series({'CTA 1': 'CTA', 'CTA 2': 'CTA', 'LS 1': 'LS', 'LS 2': 'LS'})
    try:
        fig = plot_regime_beta_profiles(betas.assign(beta_total=totals), groups=groups, se=betas,
                                        group_labels={'CTA': 'Trend followers'})
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ['Trend followers', 'LS']
        assert len(ax.containers) == 2  # one errorbar container per group
        cta_mean = betas.loc[['CTA 1', 'CTA 2'], 'beta_bear'].mean()
        assert np.isclose(ax.containers[0].lines[0].get_ydata()[0], cta_mean)
    finally:
        plt.close('all')
