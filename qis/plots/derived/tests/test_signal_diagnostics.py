"""
Tests for qis.perfstats.signal_diagnostics.
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


def _make_synthetic_data(
        n_assets: int = 20, n_months: int = 120,
        beta_true: float = 0.20, seed: int = 7,
) -> tuple:
    """Generate prices + signal where signal predicts cross-sectional return
    with slope `beta_true` plus noise.

    Returns (prices, signal, group_data).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2010-01-31', periods=n_months, freq='ME')
    assets = [f"A{i:02d}" for i in range(n_assets)]
    # Signal = standard normal per (date, asset)
    z = rng.normal(0, 1, size=(n_months, n_assets))
    # Common factor (universe-wide return) per date
    f = rng.normal(0.005, 0.04, size=n_months)
    # Asset return = common factor + beta_true * z_{t-1} (lagged) * 0.05 + idiosyncratic noise
    # z_{t-1}: the row above (and zero for t=0)
    z_lag = np.vstack([np.zeros((1, n_assets)), z[:-1, :]])
    cs_dispersion_scale = 0.05  # std of asset-specific component
    idio = rng.normal(0, cs_dispersion_scale, size=(n_months, n_assets))
    r = f[:, None] + beta_true * cs_dispersion_scale * z_lag + idio
    # Prices = cumulative product
    prices = pd.DataFrame((1 + r).cumprod(axis=0), index=dates, columns=assets)
    signal_df = pd.DataFrame(z, index=dates, columns=assets)
    # Two groups: first half / second half
    half = n_assets // 2
    group_data = pd.Series(['A'] * half + ['B'] * (n_assets - half), index=assets)
    return prices, signal_df, group_data


# ───────────────────────────────────────────────────────────────────────────────
# Basic shape and contract
# ───────────────────────────────────────────────────────────────────────────────


class TestSignalDiagnosticsShape:
    """Output structure invariants."""

    def test_returns_result_object_with_expected_attributes(self):
        prices, signal, _ = _make_synthetic_data()
        result = estimate_signal_diagnostics(prices=prices, signal=signal,
                                             horizons=[1, 3])
        assert isinstance(result, SignalDiagnosticsResult)
        assert isinstance(result.pooled_universe, pd.DataFrame)
        assert isinstance(result.per_group, pd.DataFrame)
        assert isinstance(result.pairs, dict)

    def test_pooled_index_uses_horizon_labels(self):
        prices, signal, _ = _make_synthetic_data()
        result = estimate_signal_diagnostics(prices=prices, signal=signal,
                                             horizons=[1, 3, 6])
        assert list(result.pooled_universe.index) == ['1m', '3m', '6m']
        assert result.horizon_labels == ['1m', '3m', '6m']

    def test_pooled_columns_match_signal_diagnostics_columns(self):
        prices, signal, _ = _make_synthetic_data()
        result = estimate_signal_diagnostics(prices=prices, signal=signal,
                                             horizons=[1])
        expected = [c.value for c in SignalDiagnosticsColumns]
        assert list(result.pooled_universe.columns) == expected

    def test_per_group_empty_when_no_group_data(self):
        prices, signal, _ = _make_synthetic_data()
        result = estimate_signal_diagnostics(prices=prices, signal=signal,
                                             horizons=[1])
        assert len(result.per_group) == 0

    def test_per_group_has_multiindex_when_group_data_provided(self):
        prices, signal, gd = _make_synthetic_data()
        result = estimate_signal_diagnostics(prices=prices, signal=signal,
                                             horizons=[1, 3], group_data=gd)
        assert result.per_group.index.names == ['horizon', 'group']
        # Two groups × two horizons -> 4 rows
        assert len(result.per_group) == 4

    def test_pairs_keyed_by_horizon_label(self):
        prices, signal, _ = _make_synthetic_data()
        result = estimate_signal_diagnostics(prices=prices, signal=signal,
                                             horizons=[1, 3])
        assert set(result.pairs.keys()) == {'1m', '3m'}
        assert {'date', 'asset', 'group', 'z',
                'r_norm_univ', 'r_norm_group'}.issubset(result.pairs['1m'].columns)


# ───────────────────────────────────────────────────────────────────────────────
# Numerical correctness
# ───────────────────────────────────────────────────────────────────────────────


class TestSignalDiagnosticsNumerics:
    """Coefficient recovery and basic sanity."""

    def test_recovers_positive_beta_when_signal_predicts(self):
        prices, signal, _ = _make_synthetic_data(beta_true=0.30, n_months=240)
        result = estimate_signal_diagnostics(prices=prices, signal=signal,
                                             horizons=[1])
        beta_1m = result.pooled_universe.loc['1m',
                                             SignalDiagnosticsColumns.BETA.value]
        t_1m = result.pooled_universe.loc['1m',
                                          SignalDiagnosticsColumns.T_STAT.value]
        assert beta_1m > 0, f"expected positive beta, got {beta_1m}"
        assert t_1m > 2.0, f"expected significant t-stat, got {t_1m}"

    def test_no_predictive_content_when_signal_is_noise(self):
        prices, _, _ = _make_synthetic_data(beta_true=0.0, n_months=240)
        rng = np.random.default_rng(42)
        noise_signal = pd.DataFrame(
            rng.normal(0, 1, size=prices.shape),
            index=prices.index, columns=prices.columns,
        )
        result = estimate_signal_diagnostics(prices=prices, signal=noise_signal,
                                             horizons=[1])
        t_1m = result.pooled_universe.loc['1m',
                                          SignalDiagnosticsColumns.T_STAT.value]
        assert abs(t_1m) < 2.5, f"noise signal should not be significant, got t={t_1m}"

    def test_no_intercept_default(self):
        """Slope under fit_intercept=False should differ from slope with intercept
        unless the residuals happen to centre exactly on zero. We just check both
        branches run and produce finite output."""
        prices, signal, _ = _make_synthetic_data(beta_true=0.20, n_months=180)
        r_no = estimate_signal_diagnostics(prices=prices, signal=signal,
                                           horizons=[1], fit_intercept=False)
        r_yes = estimate_signal_diagnostics(prices=prices, signal=signal,
                                            horizons=[1], fit_intercept=True)
        beta_no = r_no.pooled_universe.loc['1m',
                                           SignalDiagnosticsColumns.BETA.value]
        beta_yes = r_yes.pooled_universe.loc['1m',
                                             SignalDiagnosticsColumns.BETA.value]
        assert np.isfinite(beta_no)
        assert np.isfinite(beta_yes)


# ───────────────────────────────────────────────────────────────────────────────
# Edge cases
# ───────────────────────────────────────────────────────────────────────────────


class TestSignalDiagnosticsEdgeCases:
    """NaN handling, undersized groups, string horizons."""

    def test_handles_nan_in_signal(self):
        prices, signal, gd = _make_synthetic_data()
        # NaN out half of the signal
        signal = signal.copy()
        signal.iloc[::3, ::2] = np.nan
        result = estimate_signal_diagnostics(prices=prices, signal=signal,
                                             horizons=[1], group_data=gd)
        # Must still produce a finite β
        beta = result.pooled_universe.loc['1m',
                                          SignalDiagnosticsColumns.BETA.value]
        assert np.isfinite(beta)

    def test_drops_groups_with_too_few_obs(self):
        prices, signal, _ = _make_synthetic_data(n_assets=20, n_months=60)
        # 3 groups, 'Tiny' has only 2 assets × ~60 dates = ~120 observations.
        gd = pd.Series(['A'] * 9 + ['B'] * 9 + ['Tiny'] * 2,
                       index=prices.columns)
        # Set min_obs_per_group above the 'Tiny' panel size so it gets dropped
        result = estimate_signal_diagnostics(prices=prices, signal=signal,
                                             horizons=[1], group_data=gd,
                                             min_obs_per_group=200)
        present_groups = set(result.per_group.index.get_level_values('group'))
        assert 'Tiny' not in present_groups
        # Larger groups should still be present
        assert 'A' in present_groups
        assert 'B' in present_groups

    def test_string_horizon_year_end(self):
        prices, signal, _ = _make_synthetic_data(n_months=120)
        result = estimate_signal_diagnostics(prices=prices, signal=signal,
                                             horizons=['YE'])
        assert 'YE' in result.pooled_universe.index
        assert 'YE' in result.pairs

    def test_mixed_int_and_string_horizons(self):
        prices, signal, _ = _make_synthetic_data(n_months=120)
        result = estimate_signal_diagnostics(prices=prices, signal=signal,
                                             horizons=[1, 3, 'YE'])
        assert list(result.pooled_universe.index) == ['1m', '3m', 'YE']

    def test_rejects_invalid_signal_columns(self):
        prices, signal, _ = _make_synthetic_data()
        bad_signal = signal.rename(columns={signal.columns[0]: 'INVALID'})
        with pytest.raises(ValueError, match='subset of prices'):
            estimate_signal_diagnostics(prices=prices, signal=bad_signal,
                                        horizons=[1])

    def test_rejects_non_series_group_data(self):
        prices, signal, _ = _make_synthetic_data()
        with pytest.raises(TypeError, match='pandas Series'):
            estimate_signal_diagnostics(prices=prices, signal=signal,
                                        horizons=[1], group_data={'A00': 'X'})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
