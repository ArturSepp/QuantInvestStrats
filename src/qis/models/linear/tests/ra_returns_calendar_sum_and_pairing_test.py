"""Contracts of the risk-adjusted-return aggregation, pairing and signal-map fixes.

Covers ``compute_sum_freq_ra_returns`` normalisation by the per-period observation count,
``get_paired_rareturns_signals`` under pandas 2.2 and 3 and without look-ahead in its
overlapping mode, the timing of ``compute_ewm_long_short_filtered_ra_returns``, and the
argument handling of ``map_signal_to_weight``.
"""
# packages
import warnings

import numpy as np
import pandas as pd
import pytest

# qis
import qis


def _noise_returns(n_rows: int = 252 * 20, seed: int = 11, n_assets: int = 1) -> pd.DataFrame:
    """Daily returns with constant 1% volatility on a business-day grid."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2000-01-03', periods=n_rows)
    columns = [f'a{i}' for i in range(n_assets)]
    return pd.DataFrame(0.01 * rng.standard_normal((n_rows, n_assets)), index=index,
                        columns=columns)


class TestCalendarSumNormalisation:
    """``is_norm=True`` divides each calendar sum by the root of its observation count."""

    @pytest.mark.parametrize('freq', ['W-FRI', 'ME', 'QE'])
    def test_divides_by_root_of_observations_per_period(self, freq):
        returns = _noise_returns()
        x, _, _ = qis.compute_ra_returns(returns=returns, ewm_lambda=0.97)
        expected = x.resample(freq).sum() / np.sqrt(x.resample(freq).count())
        actual = qis.compute_sum_freq_ra_returns(returns=returns, freq=freq, ewm_lambda=0.97,
                                                 is_log_returns_to_arithmetic=False)
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize('freq', ['W-FRI', 'ME', 'QE'])
    def test_normalised_sums_have_unit_scale(self, freq):
        # before the fix the standard deviations were about 0.31, 1.32 and 3.97 (sqrt(n_J/AN_f))
        returns = _noise_returns(n_rows=252 * 40)
        sums = qis.compute_sum_freq_ra_returns(returns=returns, freq=freq, ewm_lambda=0.99,
                                               is_log_returns_to_arithmetic=False)
        # drop the first year: the one-observation EWM seed distorts the leading estimates
        spread = float(sums.iloc[:, 0].loc['2001':].std())
        assert 0.9 < spread < 1.1, spread

    def test_period_without_observations_is_missing_not_zero(self):
        returns = _noise_returns(n_rows=300)
        returns.loc['2000-03-01':'2000-03-31'] = np.nan
        x, _, _ = qis.compute_ra_returns(returns=returns, ewm_lambda=0.94)
        sums = qis.compute_sum_freq_ra_returns(returns=returns, freq='ME', ewm_lambda=0.94,
                                               is_log_returns_to_arithmetic=False)
        assert x.loc['2000-03'].isna().all().all()
        assert np.isnan(sums.loc['2000-03-31', 'a0'])

    def test_is_norm_false_keeps_raw_sums(self):
        returns = _noise_returns(n_rows=300)
        x, _, _ = qis.compute_ra_returns(returns=returns, ewm_lambda=0.94)
        raw = qis.compute_sum_freq_ra_returns(returns=returns, freq='ME', ewm_lambda=0.94,
                                              is_log_returns_to_arithmetic=False, is_norm=False)
        pd.testing.assert_frame_equal(raw, x.resample('ME').sum())

    def test_business_and_calendar_day_frequencies_return_the_terms(self):
        returns = _noise_returns(n_rows=300)
        x, _, _ = qis.compute_ra_returns(returns=returns, ewm_lambda=0.94)
        out = qis.compute_sum_freq_ra_returns(returns=returns, freq='B', ewm_lambda=0.94,
                                              is_log_returns_to_arithmetic=False)
        pd.testing.assert_frame_equal(out, x)


class TestPairedReturnsAndSignals:
    """``get_paired_rareturns_signals`` runs on pandas 2.2 and 3 and pairs forward returns."""

    def test_default_frequency_runs(self):
        # 'BQ' raised ValueError under pandas 3; the default is now the business quarter-end 'BQE'
        returns = _noise_returns(n_rows=600)
        ra, indicator = qis.get_paired_rareturns_signals(returns=returns, signal=returns)
        assert ra.index.equals(indicator.index)
        assert (ra.index.month % 3 == 0).all()

    def test_mean_adjustment_runs_and_is_expanding(self):
        # ``expanding(axis=0)`` raised TypeError under pandas 3
        returns = _noise_returns(n_rows=600)
        base, _ = qis.get_paired_rareturns_signals(returns=returns, signal=returns, freq='ME')
        adjusted, _ = qis.get_paired_rareturns_signals(returns=returns, signal=returns, freq='ME',
                                                       is_mean_adjust_returns=True)
        pd.testing.assert_frame_equal(adjusted, base - base.expanding(min_periods=1).mean())

    def test_nonoverlapping_mode_uses_unit_scale_sums(self):
        returns = _noise_returns(n_rows=600)
        ra, indicator = qis.get_paired_rareturns_signals(returns=returns, signal=returns,
                                                         freq='ME', ra_returns_ewm_vol_lambda=0.9)
        expected = qis.compute_sum_freq_ra_returns(returns=returns, freq='ME', ewm_lambda=0.9,
                                                   is_norm=True)
        pd.testing.assert_frame_equal(ra, expected)
        pd.testing.assert_frame_equal(indicator, returns.resample('ME').last().shift(1))

    def test_overlapping_mode_lags_signal_by_the_window(self):
        returns = _noise_returns(n_rows=600)
        signal = returns.cumsum()
        _, indicator = qis.get_paired_rareturns_signals(returns=returns, signal=signal, span=21,
                                                        is_nonoverlapping=False)
        pd.testing.assert_frame_equal(indicator, signal.shift(21))

    def test_overlapping_mode_has_no_look_ahead(self):
        # a trailing-sum signal paired with the window it summarises: before the fix the signal
        # at t-1 shared span-1 of the span returns and the correlation was about (span-1)/span
        returns = _noise_returns(n_rows=252 * 30, seed=5)
        span = 21
        x, _, _ = qis.compute_ra_returns(returns=returns, ewm_lambda=0.94)
        trailing = x.rolling(span).sum()
        ra, indicator = qis.get_paired_rareturns_signals(returns=returns, signal=trailing,
                                                         span=span, is_nonoverlapping=False,
                                                         ra_returns_ewm_vol_lambda=0.94)
        pairs = pd.concat([ra.iloc[:, 0], indicator.iloc[:, 0]], axis=1).dropna()
        # only non-overlapping pairs: every span-th row
        sampled = pairs.iloc[::span]
        corr = float(np.corrcoef(sampled.iloc[:, 0], sampled.iloc[:, 1])[0, 1])
        assert abs(corr) < 0.1, corr


class TestFilteredRaReturnsTiming:
    """The output is not shifted; ``weight_lag`` lags only the volatility normaliser."""

    def test_single_leg_loads_on_current_row_and_two_leg_does_not(self):
        index = pd.bdate_range('2020-01-01', periods=120)
        x = pd.DataFrame({'a': np.zeros(120)}, index=index)
        x.iloc[60, 0] = 1.0
        single = qis.compute_ewm_long_short_filtered_ra_returns(x, vol_span=None, short_span=None,
                                                               warmup_period=None)
        two_leg = qis.compute_ewm_long_short_filtered_ra_returns(x, vol_span=None,
                                                                warmup_period=None)
        lam = 1.0 - 2.0 / 64.0
        assert single.iloc[59, 0] == 0.0
        np.testing.assert_allclose(single.iloc[60, 0], np.sqrt(1.0 - lam ** 2), rtol=1e-12)
        assert two_leg.iloc[59, 0] == 0.0 and abs(two_leg.iloc[60, 0]) < 1e-15
        assert two_leg.iloc[61, 0] > 0.0

    def test_weight_lag_lags_the_volatility_normaliser(self):
        returns = _noise_returns(n_rows=300)
        for lag in (1, 2):
            x, _, _ = qis.compute_ra_returns(returns=returns, span=31, weight_lag=lag)
            expected = qis.compute_ewm_long_short_filter(x, long_span=63, short_span=5,
                                                         warmup_period=21)
            actual = qis.compute_ewm_long_short_filtered_ra_returns(returns, weight_lag=lag)
            pd.testing.assert_frame_equal(actual, expected)


class TestSignalMapArguments:
    """Tail decays apply per side; arguments a map ignores are reported."""

    signals = pd.DataFrame({'s': [-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0]})

    def test_right_decay_alone_fades_only_the_right_tail(self):
        plain = qis.map_signal_to_weight(self.signals, signal_map_type=qis.SignalMapType.ExpCDF)
        right = qis.map_signal_to_weight(self.signals, signal_map_type=qis.SignalMapType.ExpCDF,
                                         tail_decay_right=1.0)
        y = self.signals['s'].to_numpy()
        fade = np.where(y > 1.0, np.exp(-(y - 1.0)), 1.0)
        np.testing.assert_allclose(right['s'], plain['s'] * fade, atol=1e-15)
        assert (right['s'].iloc[:5] == plain['s'].iloc[:5]).all()

    def test_left_decay_alone_fades_only_the_left_tail(self):
        plain = qis.map_signal_to_weight(self.signals, signal_map_type=qis.SignalMapType.ExpCDF)
        left = qis.map_signal_to_weight(self.signals, signal_map_type=qis.SignalMapType.ExpCDF,
                                        tail_decay_left=2.0)
        y = self.signals['s'].to_numpy()
        fade = np.where(y < -1.0, np.exp((y + 1.0) / 2.0), 1.0)
        np.testing.assert_allclose(left['s'], plain['s'] * fade, atol=1e-15)
        assert (left['s'].iloc[2:] == plain['s'].iloc[2:]).all()

    def test_both_decays_unchanged(self):
        plain = qis.map_signal_to_weight(self.signals, signal_map_type=qis.SignalMapType.ExpCDF)
        both = qis.map_signal_to_weight(self.signals, signal_map_type=qis.SignalMapType.ExpCDF,
                                        tail_decay_right=1.0, tail_decay_left=2.0)
        y = self.signals['s'].to_numpy()
        fade = np.where(y > 1.0, np.exp(-(y - 1.0)), np.where(y < -1.0, np.exp((y + 1.0) / 2.0),
                                                                1.0))
        np.testing.assert_allclose(both['s'], plain['s'] * fade, atol=1e-15)

    @pytest.mark.parametrize('map_type', [qis.SignalMapType.NormalCDF,
                                          qis.SignalMapType.LaplaceCDF])
    @pytest.mark.parametrize('kwargs', [{'tail_level': 2.0}, {'slope_right': 0.8},
                                        {'slope_left': 0.2}, {'tail_decay_right': 1.0},
                                        {'tail_decay_left': 1.0}])
    def test_cdf_maps_warn_on_ignored_arguments(self, map_type, kwargs):
        with pytest.warns(UserWarning, match='ignored'):
            out = qis.map_signal_to_weight(self.signals, signal_map_type=map_type, **kwargs)
        default = qis.map_signal_to_weight(self.signals, signal_map_type=map_type)
        pd.testing.assert_frame_equal(out, default)

    @pytest.mark.parametrize('map_type', list(qis.SignalMapType))
    def test_defaults_do_not_warn(self, map_type):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            qis.map_signal_to_weight(self.signals, signal_map_type=map_type)
