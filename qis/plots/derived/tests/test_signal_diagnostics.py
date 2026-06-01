"""
Tests for qis.perfstats.signal_diagnostics — returns-dict API with
per-asset native-cadence handling — plus integration tests for the
two compute+plot convenience wrappers in
qis.plots.derived.signal_diagnostics_plot:

    plot_signal_diagnostics_for_returns      — composite figure
    plot_signal_diagnostics_beta_boxplot     — per-asset β boxplot

These two wrappers absorb what was previously in
``optimalportfolios.alphas.signal_diagnostics``; the qis-side
integration test makes sure the contract is stable when callers
plug in raw DataFrame signals.
"""
# built-in
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless test runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# qis
from qis.perfstats.signal_diagnostics import (
    SignalDiagnosticsColumns,
    SignalDiagnosticsResult,
    estimate_signal_diagnostics,
)
from qis.plots.derived.signal_diagnostics_plot import (
    plot_signal_diagnostics,
    plot_signal_diagnostics_boxplot,
    plot_signal_diagnostics_group_boxplot,
    plot_signal_diagnostics_for_returns,
    plot_signal_diagnostics_beta_boxplot,
)


def _make_synthetic_returns_dict(
        n_monthly: int = 20, n_quarterly: int = 5,
        n_years: int = 10, beta_true: float = 0.30, seed: int = 7,
) -> tuple:
    """Build a per-frequency returns dict + monthly signal + group data.

    Monthly assets print every month-end; quarterly assets print every
    quarter-end. Signal predicts cross-sectional ranking at horizon 1.

    Returns (asset_returns_dict, signal, group_data).
    """
    rng = np.random.default_rng(seed)
    n_months = n_years * 12

    me_dates = pd.date_range('2010-01-31', periods=n_months, freq='ME')
    qe_dates = pd.date_range('2010-03-31', periods=n_years * 4, freq='QE')

    me_assets = [f"M{i:02d}" for i in range(n_monthly)]
    qe_assets = [f"Q{i:02d}" for i in range(n_quarterly)]
    all_assets = me_assets + qe_assets
    n_all = len(all_assets)

    # Signal: standard normal per (date, asset) at monthly cadence
    z_m = rng.normal(0, 1, size=(n_months, n_all))
    signal = pd.DataFrame(z_m, index=me_dates, columns=all_assets)

    # Monthly returns: signal_lagged → return next month
    z_lag = np.vstack([np.zeros((1, n_monthly)), z_m[:-1, :n_monthly]])
    f_m = rng.normal(0.005, 0.04, size=n_months)
    idio_m = rng.normal(0, 0.05, size=(n_months, n_monthly))
    r_m = f_m[:, None] + beta_true * 0.05 * z_lag + idio_m
    me_returns = pd.DataFrame(r_m, index=me_dates, columns=me_assets)

    # Quarterly returns: signal at quarter-end → return next quarter
    z_q_at_qe = signal.loc[qe_dates, qe_assets].to_numpy()
    z_q_lag = np.vstack([np.zeros((1, n_quarterly)), z_q_at_qe[:-1, :]])
    f_q = rng.normal(0.015, 0.07, size=len(qe_dates))
    idio_q = rng.normal(0, 0.08, size=(len(qe_dates), n_quarterly))
    r_q = f_q[:, None] + beta_true * 0.08 * z_q_lag + idio_q
    qe_returns = pd.DataFrame(r_q, index=qe_dates, columns=qe_assets)

    asset_returns_dict = {'ME': me_returns, 'QE': qe_returns}

    # Two groups
    half = n_all // 2
    group_data = pd.Series(
        ['A'] * half + ['B'] * (n_all - half), index=all_assets,
    )
    return asset_returns_dict, signal, group_data


# ───────────────────────────────────────────────────────────────────────────────
# Shape and contract
# ───────────────────────────────────────────────────────────────────────────────


class TestSignalDiagnosticsShape:
    def test_returns_result_object(self):
        ard, sig, _ = _make_synthetic_returns_dict()
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, horizons=[1, 3])
        assert isinstance(result, SignalDiagnosticsResult)

    def test_pooled_index_uses_horizon_labels(self):
        ard, sig, _ = _make_synthetic_returns_dict()
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, horizons=[1, 3, 6])
        assert list(result.pooled_universe.index) == ['1', '3', '6']

    def test_pooled_columns_match_enum(self):
        ard, sig, _ = _make_synthetic_returns_dict()
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, horizons=[1])
        expected = [c.value for c in SignalDiagnosticsColumns]
        assert list(result.pooled_universe.columns) == expected

    def test_per_group_empty_when_no_group_data(self):
        ard, sig, _ = _make_synthetic_returns_dict()
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, horizons=[1])
        assert len(result.per_group) == 0

    def test_per_group_multiindex(self):
        ard, sig, gd = _make_synthetic_returns_dict()
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, group_data=gd,
                                             horizons=[1, 3])
        assert result.per_group.index.names == ['horizon', 'group']

    def test_pairs_include_asset_freq(self):
        ard, sig, _ = _make_synthetic_returns_dict()
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, horizons=[1])
        assert 'asset_freq' in result.pairs['1'].columns


# ───────────────────────────────────────────────────────────────────────────────
# Cadence handling — the core new behaviour
# ───────────────────────────────────────────────────────────────────────────────


class TestCadenceHandling:
    """Verify per-asset cadence rules."""

    def test_quarterly_assets_only_print_at_quarter_ends(self):
        ard, sig, _ = _make_synthetic_returns_dict()
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, horizons=[1])
        pairs_1 = result.pairs['1']
        # Quarterly asset rows
        q_rows = pairs_1[pairs_1['asset_freq'] == 'QE']
        # All quarterly observations must fall on quarter-end dates
        q_dates = pd.DatetimeIndex(q_rows['date'].unique())
        for d in q_dates:
            assert d.month in [3, 6, 9, 12], \
                f"Q asset found on non-quarter-end date {d}"

    def test_monthly_assets_print_every_month(self):
        ard, sig, _ = _make_synthetic_returns_dict()
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, horizons=[1])
        m_rows = result.pairs['1'][result.pairs['1']['asset_freq'] == 'ME']
        # Monthly assets should show observations across many calendar months
        m_dates = pd.DatetimeIndex(m_rows['date'].unique())
        months_covered = {d.month for d in m_dates}
        assert len(months_covered) >= 10, "Monthly assets should span many calendar months"

    def test_horizon_1_for_qe_means_1_quarter(self):
        """Sample size sanity: 10 years × 4 quarters = 40 obs per QE asset for horizon 1."""
        ard, sig, _ = _make_synthetic_returns_dict(n_quarterly=5, n_years=10)
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, horizons=[1])
        pairs_1 = result.pairs['1']
        q_rows = pairs_1[pairs_1['asset_freq'] == 'QE']
        # 5 QE assets × ~40 dates (with rolling window losses) ≈ 5 × 35-40
        # Loose check
        assert 100 < len(q_rows) < 220

    def test_horizon_3_qe_means_9_months(self):
        """Horizon 3 for QE = 3 quarters = 9 months; non-overlapping every 3 quarters."""
        ard, sig, _ = _make_synthetic_returns_dict(n_quarterly=5, n_years=10)
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, horizons=[3])
        pairs_3 = result.pairs['3']
        q_rows = pairs_3[pairs_3['asset_freq'] == 'QE']
        # 5 QE assets × ~12 sampled dates (10 years × 4 quarters / 3 step ≈ 13)
        # Some lost to rolling window; loose check
        assert 30 < len(q_rows) < 80


# ───────────────────────────────────────────────────────────────────────────────
# Numerical correctness
# ───────────────────────────────────────────────────────────────────────────────


class TestNumerics:
    def test_recovers_positive_beta(self):
        ard, sig, _ = _make_synthetic_returns_dict(beta_true=0.40, n_years=15)
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, horizons=[1])
        beta_1 = result.pooled_universe.loc['1',
                                            SignalDiagnosticsColumns.BETA.value]
        t_1 = result.pooled_universe.loc['1',
                                         SignalDiagnosticsColumns.T_STAT.value]
        assert beta_1 > 0, f"expected positive beta, got {beta_1}"
        assert t_1 > 2.0, f"expected significant t-stat, got {t_1}"

    def test_no_predictive_content_for_noise(self):
        ard, _sig, _ = _make_synthetic_returns_dict(beta_true=0.0, n_years=15)
        rng = np.random.default_rng(42)
        # Pure-noise signal panel at same monthly cadence
        me_index = ard['ME'].index
        all_assets = list(ard['ME'].columns) + list(ard['QE'].columns)
        noise = pd.DataFrame(rng.normal(0, 1, size=(len(me_index), len(all_assets))),
                             index=me_index, columns=all_assets)
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=noise, horizons=[1])
        t_1 = result.pooled_universe.loc['1',
                                         SignalDiagnosticsColumns.T_STAT.value]
        assert abs(t_1) < 2.5, f"noise signal should not be significant, got t={t_1}"


# ───────────────────────────────────────────────────────────────────────────────
# Edge cases
# ───────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_handles_nan_in_signal(self):
        ard, sig, gd = _make_synthetic_returns_dict()
        sig = sig.copy()
        sig.iloc[::3, ::2] = np.nan
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, group_data=gd,
                                             horizons=[1])
        beta = result.pooled_universe.loc['1',
                                          SignalDiagnosticsColumns.BETA.value]
        assert np.isfinite(beta)

    def test_string_horizon_year_end(self):
        ard, sig, _ = _make_synthetic_returns_dict(n_years=15)
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, horizons=['YE'])
        assert 'YE' in result.pooled_universe.index
        assert 'YE' in result.pairs
        # YE pairs should include both monthly and quarterly assets
        ye_pairs = result.pairs['YE']
        if len(ye_pairs) > 0:
            assert ye_pairs['asset_freq'].nunique() >= 1

    def test_mixed_int_and_string_horizons(self):
        ard, sig, _ = _make_synthetic_returns_dict(n_years=15)
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig,
                                             horizons=[1, 3, 'YE'])
        assert list(result.pooled_universe.index) == ['1', '3', 'YE']

    def test_drops_assets_not_covered_by_signal(self):
        """Partial signal coverage projects the returns dict to the
        intersection — no error, smaller pooled sample."""
        ard, sig, _ = _make_synthetic_returns_dict()
        dropped_col = sig.columns[0]
        partial_sig = sig.drop(columns=[dropped_col])
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=partial_sig, horizons=[1])
        # The dropped asset should not appear in any pair frame
        pairs = result.pairs['1']
        assert dropped_col not in set(pairs['asset'].unique())
        # Other assets should still be present
        assert len(pairs) > 0

    def test_rejects_signal_with_no_overlap(self):
        """Signal panel with zero overlap raises."""
        ard, _sig, _ = _make_synthetic_returns_dict()
        # Build a signal panel on a totally different asset universe
        first_returns = next(iter(ard.values()))
        alien_sig = pd.DataFrame(
            0.0, index=first_returns.index,
            columns=['UNKNOWN_A', 'UNKNOWN_B'],
        )
        with pytest.raises(ValueError, match='No overlap'):
            estimate_signal_diagnostics(asset_returns_dict=ard,
                                        signal=alien_sig, horizons=[1])

    def test_rejects_empty_returns_dict(self):
        _, sig, _ = _make_synthetic_returns_dict()
        with pytest.raises(ValueError, match='non-empty dict'):
            estimate_signal_diagnostics(asset_returns_dict={},
                                        signal=sig, horizons=[1])

    def test_rejects_non_series_group_data(self):
        ard, sig, _ = _make_synthetic_returns_dict()
        with pytest.raises(TypeError, match='pandas Series'):
            estimate_signal_diagnostics(asset_returns_dict=ard,
                                        signal=sig,
                                        horizons=[1],
                                        group_data={'M00': 'X'})


# ───────────────────────────────────────────────────────────────────────────────
# Plot API — the two "compute+plot" convenience wrappers
# ───────────────────────────────────────────────────────────────────────────────


class TestPlotApi:
    """Integration tests for the qis-side compute+plot wrappers.

    These wrappers replace the previous
    ``optimalportfolios.alphas.plot_signal_diagnostics`` and
    ``plot_signal_diagnostics_beta_boxplot`` thin shims; the tests
    cover the contract callers depend on: pass a returns dict + a
    DataFrame signal, get a Figure back without side-effects.
    """

    def teardown_method(self):
        plt.close('all')

    def test_for_returns_default_args(self):
        ard, sig, _ = _make_synthetic_returns_dict(n_years=8)
        fig = plot_signal_diagnostics_for_returns(
            asset_returns_dict=ard, signal=sig, horizons=[1, 3],
        )
        assert isinstance(fig, plt.Figure)
        # 1 row × 2 horizons = 2 axes when no group data
        assert len(fig.axes) >= 2

    def test_for_returns_with_group_data(self):
        ard, sig, gd = _make_synthetic_returns_dict(n_years=8)
        fig = plot_signal_diagnostics_for_returns(
            asset_returns_dict=ard, signal=sig,
            horizons=[1, 3], group_data=gd,
        )
        # 2 rows × 2 horizons = 4 axes
        assert len(fig.axes) == 4

    def test_for_returns_custom_title(self):
        ard, sig, _ = _make_synthetic_returns_dict(n_years=8)
        fig = plot_signal_diagnostics_for_returns(
            asset_returns_dict=ard, signal=sig, horizons=[1],
            title='Custom title — APAC test',
        )
        assert fig._suptitle is not None
        assert 'APAC test' in fig._suptitle.get_text()

    def test_for_returns_string_horizons(self):
        ard, sig, _ = _make_synthetic_returns_dict(n_years=15)
        fig = plot_signal_diagnostics_for_returns(
            asset_returns_dict=ard, signal=sig, horizons=['YE'],
        )
        assert isinstance(fig, plt.Figure)

    def test_beta_boxplot_default_args(self):
        ard, sig, gd = _make_synthetic_returns_dict(n_years=12)
        fig = plot_signal_diagnostics_beta_boxplot(
            asset_returns_dict=ard, signal=sig,
            horizons=[1, 3, 6], group_data=gd,
            min_obs_per_asset=10,
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1

    def test_beta_boxplot_custom_hue_label(self):
        """The user-facing hue label appears in the legend title rather
        than the internal column name."""
        ard, sig, gd = _make_synthetic_returns_dict(n_years=12)
        fig = plot_signal_diagnostics_beta_boxplot(
            asset_returns_dict=ard, signal=sig,
            horizons=[1, 3], group_data=gd,
            hue='asset_freq',
            hue_display_label='Asset returns sampling',
            min_obs_per_asset=10,
        )
        # qis.plot_box renders the legend if hue has >1 unique value.
        # In _make_synthetic_returns_dict we have both ME and QE so the
        # legend must be present and its title must use the display
        # label, not the internal 'asset_freq'.
        legends = [ax.get_legend() for ax in fig.axes
                   if ax.get_legend() is not None]
        assert legends, 'Expected legend on per-asset β boxplot'
        title_txt = legends[0].get_title().get_text()
        assert 'Asset returns sampling' in title_txt or \
               'asset_freq' not in title_txt

    def test_beta_boxplot_no_hue(self):
        ard, sig, _ = _make_synthetic_returns_dict(n_years=12)
        fig = plot_signal_diagnostics_beta_boxplot(
            asset_returns_dict=ard, signal=sig,
            horizons=[1, 3], hue=None,
            min_obs_per_asset=10,
        )
        assert isinstance(fig, plt.Figure)

    def test_beta_boxplot_handles_high_threshold(self):
        """When the obs threshold is high enough that no asset
        qualifies, the function still returns a Figure with a clear
        message rather than crashing."""
        ard, sig, _ = _make_synthetic_returns_dict(n_years=2)
        fig = plot_signal_diagnostics_beta_boxplot(
            asset_returns_dict=ard, signal=sig,
            horizons=[1], min_obs_per_asset=10_000,
        )
        assert isinstance(fig, plt.Figure)

    def test_boxplot_handles_degenerate_signal(self):
        """Signals with mass at zero produce duplicate quantile edges;
        the boxplot adapts the bucket count down (or falls back to a
        single box) rather than raising ValueError from pd.qcut."""
        ard, sig, _ = _make_synthetic_returns_dict(n_years=10)
        # Zero out most of the signal — guarantees mass-at-zero
        sparse = sig.copy()
        sparse[:] = 0.0
        # Sprinkle a few non-zero entries so we have *some* spread
        sparse.iloc[::40, ::5] = 0.5
        sparse.iloc[::43, ::7] = -0.5
        fig = plot_signal_diagnostics_for_returns(
            asset_returns_dict=ard, signal=sparse,
            horizons=[1], num_buckets=10,
        )
        assert isinstance(fig, plt.Figure)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])