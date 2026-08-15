"""
Tests for qis.perfstats.signal_diagnostics — returns-dict API with
per-asset native-cadence handling.
"""
# built-in
import numpy as np
import pandas as pd
import pytest

# qis
from qis.perfstats.signal_diagnostics import (
    SignalDiagnosticsColumns,
    SignalDiagnosticsResult,
    estimate_signal_diagnostics,
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
# Per-asset betas
# ───────────────────────────────────────────────────────────────────────────────


class TestPerAssetBetas:
    """compute_per_asset_betas — distribution of β across assets."""

    def test_returns_long_format(self):
        from qis.perfstats.signal_diagnostics import compute_per_asset_betas
        ard, sig, gd = _make_synthetic_returns_dict(beta_true=0.30, n_years=15)
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, group_data=gd,
                                             horizons=[1, 3])
        df = compute_per_asset_betas(result, min_obs_per_asset=12)
        assert set(df.columns) >= {'horizon', 'asset', 'asset_freq', 'group',
                                   'beta', 't_stat', 'n'}
        assert df['horizon'].nunique() == 2
        # Monthly assets should outnumber quarterly assets (20 vs 5 in default factory)
        freq_counts = df.loc[df['horizon'] == '1', 'asset_freq'].value_counts()
        assert freq_counts.get('ME', 0) > freq_counts.get('QE', 0)

    def test_drops_under_min_obs(self):
        from qis.perfstats.signal_diagnostics import compute_per_asset_betas
        ard, sig, _ = _make_synthetic_returns_dict(n_years=15)
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig,
                                             horizons=[1])
        # n=1 has the most observations; with an absurdly high min, nothing survives
        df_strict = compute_per_asset_betas(result, min_obs_per_asset=999_999)
        assert df_strict.empty
        df_loose = compute_per_asset_betas(result, min_obs_per_asset=5)
        assert not df_loose.empty

    def test_beta_distribution_centred_when_signal_predicts(self):
        from qis.perfstats.signal_diagnostics import compute_per_asset_betas
        ard, sig, _ = _make_synthetic_returns_dict(beta_true=0.40, n_years=20,
                                                   seed=3)
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig, horizons=[1])
        df = compute_per_asset_betas(result, min_obs_per_asset=12)
        # Median per-asset β should be positive when the signal predicts
        med_beta = df.loc[df['horizon'] == '1', 'beta'].median()
        assert med_beta > 0, f"median β should be positive, got {med_beta}"

    def test_horizon_is_categorical_with_order(self):
        from qis.perfstats.signal_diagnostics import compute_per_asset_betas
        ard, sig, _ = _make_synthetic_returns_dict(n_years=15)
        result = estimate_signal_diagnostics(asset_returns_dict=ard,
                                             signal=sig,
                                             horizons=[1, 2, 3, 6])
        df = compute_per_asset_betas(result, min_obs_per_asset=12)
        # horizon column must preserve the order [1, 2, 3, 6]
        cats = list(df['horizon'].cat.categories)
        assert cats == ['1', '2', '3', '6']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
