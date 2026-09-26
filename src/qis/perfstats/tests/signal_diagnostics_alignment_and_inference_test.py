"""Pairing, annualisation, inference and input-validation contracts of the signal diagnostics.

Covers string horizons on ragged panels, the annualisation of the IC ratio, the residual degrees
of freedom of the pooled regression, frames indexed off their key's period labels, and the
warnings for dropped or duplicated assets.
"""
# packages
import warnings

import numpy as np
import pandas as pd
import pytest

# qis
import qis
from qis.perfstats.signal_diagnostics import (
    _fit_through_origin,
    estimate_ic_ir,
    estimate_signal_diagnostics,
)


def _panel(n_names: int = 20, n_months: int = 60, seed: int = 20260725,
           start: str = '2020-01-31') -> tuple:
    """Monthly log returns with a weakly predictive, cross-sectionally standardised signal."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_months, freq='ME')
    names = [f'A{i:02d}' for i in range(n_names)]
    raw = rng.standard_normal((n_months, n_names))
    score = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, ddof=1, keepdims=True)
    lagged = np.vstack([np.zeros((1, n_names)), score[:-1]])
    market = rng.normal(0.005, 0.04, size=(n_months, 1))
    log_returns = market + 0.05 * (0.1 * lagged + rng.standard_normal((n_months, n_names)))
    return (pd.DataFrame(log_returns, index=dates, columns=names),
            pd.DataFrame(score, index=dates, columns=names))


class TestStringHorizonCoverage:
    """String horizons keep only periods fully covered by an asset's native returns."""

    @staticmethod
    def _ragged():
        returns, signal = _panel(n_months=60)  # Jan-2020 .. Dec-2024, five full years
        returns = returns.copy()
        returns.loc[:'2021-06-30', 'A00'] = np.nan  # starts in July 2021
        returns.loc['2023-03-31':, 'A01'] = np.nan  # last return in February 2023
        returns.loc['2022-05-31', 'A02'] = np.nan  # one missing month inside the sample
        return returns, signal

    def test_no_zero_returns_outside_an_assets_life(self):
        returns, signal = self._ragged()
        pairs = estimate_signal_diagnostics({'ME': returns}, signal, horizons=('YE',)).pairs['YE']

        def years(asset: str) -> list:
            return sorted(pairs.loc[pairs['asset'] == asset, 'date'].dt.year)

        assert years('A00') == [2022, 2023, 2024]  # 2021 is partial
        assert years('A01') == [2021, 2022]  # 2023 is partial, 2024 after delisting
        assert years('A02') == [2021, 2023, 2024]  # 2022 has a missing month
        assert years('A03') == [2021, 2022, 2023, 2024]
        assert (pairs['r'] != 0.0).all()

    def test_matches_twelve_month_integer_horizon_on_january_phase(self):
        returns, signal = self._ragged()
        result = estimate_signal_diagnostics({'ME': returns}, signal, horizons=(12, 'YE'))
        annual = result.pairs['YE'].assign(year=lambda df: df['date'].dt.year)
        integer = result.pairs['12'].assign(year=lambda df: df['date'].dt.year)
        assert (integer['date'].dt.month == 1).all()
        left = annual.set_index(['asset', 'year'])[['z', 'r']].sort_index()
        right = integer.set_index(['asset', 'year'])[['z', 'r']].sort_index()
        pd.testing.assert_frame_equal(left, right, rtol=1e-12)

    def test_compounds_simple_returns_over_complete_periods(self):
        returns, signal = self._ragged()
        simple = np.expm1(returns)
        pairs = estimate_signal_diagnostics({'ME': simple}, signal, horizons=('YE',),
                                            is_log_returns=False).pairs['YE']
        row = pairs[(pairs['asset'] == 'A03') & (pairs['date'] == '2023-12-31')].iloc[0]
        expected = float((1.0 + simple.loc['2023', 'A03']).prod() - 1.0)
        np.testing.assert_allclose(row['r'], expected, rtol=1e-12)

    def test_partial_last_period_is_dropped(self):
        returns, signal = _panel(n_months=61)  # Jan-2020 .. Jan-2025
        pairs = estimate_signal_diagnostics({'ME': returns}, signal, horizons=('YE',)).pairs['YE']
        assert pairs['date'].max() == pd.Timestamp('2024-12-31')

    def test_business_days_with_holidays_keep_complete_months(self):
        # exchange holidays are rows absent for every asset, not missing returns
        rng = np.random.default_rng(3)
        days = pd.bdate_range('2021-01-01', '2021-12-31')
        holidays = pd.DatetimeIndex(['2021-01-01', '2021-05-31', '2021-07-05', '2021-12-24'])
        days = days.difference(holidays)
        names = [f'A{i}' for i in range(6)]
        returns = pd.DataFrame(0.01 * rng.standard_normal((len(days), 6)), index=days,
                               columns=names)
        signal_days = pd.bdate_range('2020-12-01', '2021-12-31')
        signal = pd.DataFrame(rng.standard_normal((len(signal_days), 6)), index=signal_days,
                              columns=names)
        pairs = estimate_signal_diagnostics({'B': returns}, signal, horizons=('ME',)).pairs['ME']
        # the frame starts on 4 January, after the key's first business day of 1 January, so a
        # late start cannot be told from a holiday and January is conservatively dropped; the
        # months with interior holidays (May, July, December) are complete and pair
        months = sorted(pairs['date'].dt.month.unique())
        assert months == list(range(2, 13))
        dec = pairs[(pairs['asset'] == 'A0') & (pairs['date'] == '2021-12-31')].iloc[0]
        np.testing.assert_allclose(dec['r'], returns.loc['2021-12', 'A0'].sum(), rtol=1e-12)

    def test_business_month_end_frame_starting_before_the_calendar_month_end(self):
        # January 2021 ends on a Sunday, so its business month-end is Friday 29 January
        rng = np.random.default_rng(5)
        bme = pd.date_range('2021-01-29', '2023-12-29', freq='BME')
        names = [f'A{i}' for i in range(6)]
        returns = pd.DataFrame(0.05 * rng.standard_normal((len(bme), 6)), index=bme,
                               columns=names)
        signal = pd.DataFrame(rng.standard_normal((len(bme), 6)), index=bme, columns=names)
        pairs = estimate_signal_diagnostics({'ME': returns}, signal, horizons=('YE',)).pairs['YE']
        # 2021 has no lagged signal; 2022 and 2023, including the last row on 29 December, pair
        assert sorted(pairs['date'].dt.year.unique()) == [2022, 2023]


class TestIcRatioAnnualisation:
    """``IC_IR_an`` uses qis's annualisation factor of the IC grid, scaled per horizon."""

    @staticmethod
    def _scale(table: pd.DataFrame) -> pd.Series:
        return (table['IC_IR_an'] / table['IC_IR']) ** 2

    def test_month_end_horizons_and_calendar_year(self):
        returns, signal = _panel(n_months=120, start='2010-01-31')
        result = estimate_signal_diagnostics({'ME': returns}, signal, horizons=(1, 3, 'YE'))
        scale = self._scale(estimate_ic_ir(result))
        np.testing.assert_allclose(scale.to_numpy(dtype=float), [12.0, 4.0, 1.0], rtol=1e-12)

    def test_business_days_use_252(self):
        rng = np.random.default_rng(4)
        dates = pd.bdate_range('2020-01-01', periods=300)
        names = [f'A{i}' for i in range(10)]
        returns = pd.DataFrame(0.01 * rng.standard_normal((300, 10)), index=dates, columns=names)
        signal = pd.DataFrame(rng.standard_normal((300, 10)), index=dates, columns=names)
        result = estimate_signal_diagnostics({'B': returns}, signal, horizons=(1, 5))
        scale = self._scale(estimate_ic_ir(result))
        np.testing.assert_allclose(scale.to_numpy(dtype=float), [252.0, 252.0 / 5.0], rtol=1e-12)

    def test_user_periods_per_year_is_scaled_per_horizon(self):
        returns, signal = _panel(n_months=120, start='2010-01-31')
        result = estimate_signal_diagnostics({'ME': returns}, signal, horizons=(1, 3, 'YE'))
        scale = self._scale(estimate_ic_ir(result, periods_per_year=12.0))
        np.testing.assert_allclose(scale.to_numpy(dtype=float), [12.0, 4.0, 1.0], rtol=1e-12)

    def test_short_sample_on_month_ends(self):
        returns, signal = _panel(n_months=9, start='2020-01-31')
        scale = self._scale(estimate_ic_ir(estimate_signal_diagnostics(
            {'ME': returns}, signal, horizons=(1,))))
        np.testing.assert_allclose(scale.iloc[0], 12.0, rtol=1e-12)


class TestPooledDegreesOfFreedom:
    """The residual variance charges one degree of freedom per regression date."""

    def test_through_origin_matches_hand_formula(self):
        returns, signal = _panel(n_months=61)
        result = estimate_signal_diagnostics({'ME': returns}, signal, horizons=(1,))
        z = signal.to_numpy()[:-1]
        y = returns.to_numpy()[1:]
        x = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, ddof=1, keepdims=True)
        n, n_dates = z.size, z.shape[0]
        beta = (z * x).sum() / (z * z).sum()
        sigma2 = ((x - beta * z) ** 2).sum() / (n - n_dates - 1)
        se = np.sqrt(sigma2 / (z * z).sum())
        row = result.pooled_universe.loc['1']
        np.testing.assert_allclose(row[['n', 'beta', 'se', 't_stat']].to_numpy(dtype=float),
                                   [n, beta, se, beta / se], rtol=1e-10)

    def test_with_intercept_matches_hand_formula(self):
        returns, signal = _panel(n_months=61)
        result = estimate_signal_diagnostics({'ME': returns}, signal + 1.0, horizons=(1,),
                                             fit_intercept=True)
        z = signal.to_numpy()[:-1] + 1.0
        y = returns.to_numpy()[1:]
        x = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, ddof=1, keepdims=True)
        n, n_dates = z.size, z.shape[0]
        zc = z - z.mean()
        beta = (zc * x).sum() / (zc * zc).sum()
        alpha = x.mean() - beta * z.mean()
        sigma2 = ((x - alpha - beta * z) ** 2).sum() / (n - n_dates - 1)
        se = np.sqrt(sigma2 / (zc * zc).sum())
        row = result.pooled_universe.loc['1']
        np.testing.assert_allclose(row[['beta', 'se', 't_stat']].to_numpy(dtype=float),
                                   [beta, se, beta / se], rtol=1e-10)
        np.testing.assert_allclose(alpha, -beta * z.mean(), atol=1e-12)

    def test_per_group_charges_group_dates(self):
        returns, signal = _panel(n_months=61)
        groups = pd.Series(['G1'] * 10 + ['G2'] * 10, index=returns.columns)
        result = estimate_signal_diagnostics({'ME': returns}, signal, group_data=groups,
                                             horizons=(1,))
        z = signal.to_numpy()[:-1, :10]
        y = returns.to_numpy()[1:, :10]
        x = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, ddof=1, keepdims=True)
        beta = (z * x).sum() / (z * z).sum()
        se = np.sqrt(((x - beta * z) ** 2).sum() / (z.size - z.shape[0] - 1) / (z * z).sum())
        np.testing.assert_allclose(result.per_group.loc[('1', 'G1'), 't_stat'], beta / se,
                                   rtol=1e-10)

    def test_null_t_statistic_has_unit_dispersion_on_five_names(self):
        # 4,000 null panels of 40 dates x 5 names, normalised exactly as the estimator does and
        # fitted by its pooled fitter; before the fix the dispersion was about sqrt(5/4) = 1.12
        n_dates, n_names = 40, 5
        dates = np.repeat(np.arange(n_dates), n_names)
        rng = np.random.default_rng(12)
        t_stats = []
        for _ in range(4000):
            z = rng.standard_normal((n_dates, n_names))
            z = (z - z.mean(axis=1, keepdims=True)) / z.std(axis=1, ddof=1, keepdims=True)
            y = rng.standard_normal((n_dates, n_names))
            x = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, ddof=1, keepdims=True)
            fit = _fit_through_origin(z.ravel(), x.ravel(), dates=dates)
            t_stats.append(fit['t_stat'])
        dispersion = float(np.std(t_stats))
        # t with n - T - 1 = 159 degrees of freedom has standard deviation 1.006
        assert 0.97 < dispersion < 1.04, dispersion

    def test_null_t_statistic_end_to_end(self):
        returns, _ = _panel(n_names=5, n_months=61)
        rng = np.random.default_rng(8)
        noise = pd.DataFrame(rng.standard_normal(returns.shape), index=returns.index,
                             columns=returns.columns)
        row = estimate_signal_diagnostics({'ME': returns}, noise, horizons=(1,)).pooled_universe
        n, n_dates = int(row.loc['1', 'n']), 60
        z = noise.to_numpy()[:-1]
        y = returns.to_numpy()[1:]
        x = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, ddof=1, keepdims=True)
        beta = (z * x).sum() / (z * z).sum()
        se = np.sqrt(((x - beta * z) ** 2).sum() / (n - n_dates - 1) / (z * z).sum())
        np.testing.assert_allclose(row.loc['1', 't_stat'], beta / se, rtol=1e-10)


class TestPeriodLabelAlignment:
    """A frame indexed on business month-ends under the 'ME' key is aligned, not truncated."""

    @staticmethod
    def _bme_panel():
        rng = np.random.default_rng(0)
        bme = pd.date_range('2020-01-31', periods=61, freq='BME')
        names = [f'A{i}' for i in range(8)]
        returns = pd.DataFrame(0.05 * rng.standard_normal((61, 8)), index=bme, columns=names)
        signal = pd.DataFrame(rng.standard_normal((61, 8)), index=bme, columns=names)
        return returns, signal

    def test_business_month_ends_under_month_end_key_keep_every_pair(self):
        returns, signal = self._bme_panel()
        me = pd.date_range('2020-01-31', periods=61, freq='ME')
        aligned = estimate_signal_diagnostics({'ME': returns}, signal, horizons=(1, 3))
        relabelled = estimate_signal_diagnostics({'ME': returns.set_axis(me)},
                                                 signal.set_axis(me), horizons=(1, 3))
        for label in ('1', '3'):
            pd.testing.assert_frame_equal(aligned.pairs[label], relabelled.pairs[label])
        assert len(aligned.pairs['1']) == 480
        pd.testing.assert_frame_equal(aligned.pooled_universe, relabelled.pooled_universe)

    def test_alignment_uses_only_signal_known_at_the_return_date(self):
        returns, _ = self._bme_panel()
        days = pd.date_range('2019-12-01', '2025-02-28', freq='D')
        # the signal is the ordinal day number: its value identifies its date
        signal = pd.DataFrame(np.repeat(np.arange(len(days), dtype=float)[:, None], 8, axis=1),
                              index=days, columns=returns.columns)
        pairs = estimate_signal_diagnostics({'ME': returns}, signal, horizons=(1,)).pairs['1']
        assert len(pairs) == 61 * 8
        # the return labelled at month-end m covers (BME of m-1, BME of m]: its signal must be
        # the value of the previous business month-end, not of a later calendar day
        previous_bme = pd.date_range('2019-12-31', periods=61, freq='BME')
        labels = pd.date_range('2020-01-31', periods=61, freq='ME')
        by_label = pd.Series(previous_bme, index=labels)
        signal_dates = days[pairs['z'].astype(int).to_numpy()]
        assert (signal_dates == by_label.loc[pairs['date']].to_numpy()).all()
        # the check has teeth: many previous business month-ends precede the calendar month-end
        prior_month_ends = labels - pd.offsets.MonthEnd(1)
        assert (previous_bme != prior_month_ends).sum() > 10

    def test_frame_finer_than_its_key_warns(self):
        rng = np.random.default_rng(1)
        days = pd.bdate_range('2020-01-01', periods=200)
        names = [f'A{i}' for i in range(6)]
        returns = pd.DataFrame(0.01 * rng.standard_normal((200, 6)), index=days, columns=names)
        with pytest.warns(UserWarning, match='finer than'):
            estimate_signal_diagnostics({'ME': returns}, returns, horizons=(1,))


class TestInputWarnings:
    """Dropped and duplicated assets are reported; unsupported IC methods are rejected."""

    def test_assets_without_signal_are_dropped_with_a_warning(self):
        returns, signal = _panel()
        with pytest.warns(UserWarning, match='A00'):
            result = estimate_signal_diagnostics({'ME': returns}, signal.drop(columns=['A00']),
                                                 horizons=(1,))
        assert 'A00' not in set(result.pairs['1']['asset'])

    def test_full_coverage_does_not_warn(self):
        returns, signal = _panel()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            estimate_signal_diagnostics({'ME': returns}, signal, horizons=(1, 'YE'))

    def test_asset_in_two_frames_warns_and_keeps_the_first(self):
        returns, signal = _panel()
        quarterly = returns.resample('QE').sum()[['A00']]
        with pytest.warns(UserWarning, match='A00'):
            result = estimate_signal_diagnostics({'ME': returns, 'QE': quarterly}, signal,
                                                 horizons=(1, 'YE'))
        for label in ('1', 'YE'):
            a00 = result.pairs[label]
            assert set(a00.loc[a00['asset'] == 'A00', 'asset_freq']) == {'ME'}
            assert not a00.duplicated(subset=['date', 'asset']).any()

    @pytest.mark.parametrize('method', ['kendall', 'Spearman', 'rank'])
    def test_unsupported_ic_method_raises(self, method):
        returns, signal = _panel()
        result = estimate_signal_diagnostics({'ME': returns}, signal, horizons=(1,))
        with pytest.raises(ValueError, match='method'):
            qis.estimate_ic_ir(result, method=method)
        with pytest.raises(ValueError, match='method'):
            qis.compute_ic_timeseries(result, method=method)
