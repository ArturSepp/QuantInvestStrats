"""HAC table sampling, base selection, precision and report integration contracts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets.synthetic import generate_synthetic_prices


@pytest.fixture(scope='module')
def prices():
    return generate_synthetic_prices(start='2004-12-31', end='2010-07-10',
                                     apply_quirks=False)[['SEQ_US', 'SBD_TSY', 'SEQ_EM']]


def make_multi(prices):
    return qis.MultiPortfolioData([qis.PortfolioData(prices[name].copy()) for name in prices])


@pytest.mark.parametrize('base_id', [0, 1, 2])
@pytest.mark.parametrize('freq,af', [(None, 260.0), ('ME', None)])
def test_table_matches_direct_tests_on_independent_nav_returns(prices, base_id, freq, af):
    multi = make_multi(prices)
    period = qis.TimePeriod('2005-12-30', '2009-12-31')
    actual = qis.compute_sharpe_difference_table(multi, base_id, period, freq, af, n_lags=3)
    sampled = prices.loc['2005-12-30':'2009-12-31']
    if freq is not None:
        calendar = pd.date_range(sampled.index[0], sampled.index[-1], freq='ME')
        sampled = sampled.reindex(calendar, method='ffill')
    returns = sampled.pct_change(fill_method=None).iloc[1:].dropna()
    expected_names = [name for position, name in enumerate(prices) if position != base_id]
    assert list(actual.index) == expected_names
    for name in expected_names:
        expected = qis.sharpe_difference_test(returns[name], returns.iloc[:, base_id],
                                              n_lags=3, af=af or 12.0)
        np.testing.assert_allclose(actual.loc[name].values,
                                   [expected.sr2_an, expected.sr1_an, expected.diff_an,
                                    expected.se_an, expected.t_stat, expected.p_value,
                                    expected.n_obs, expected.n_lags], atol=1e-12)
        # Direct sample moments also pin return convention and sign independently.
        sr = returns.mean() / returns.std(ddof=0) * np.sqrt(af or 12.0)
        assert actual.loc[name, 'Sharpe difference'] == pytest.approx(sr[name] - sr.iloc[base_id])
    assert actual.attrs['sample_start'] == returns.index[0]
    assert actual.attrs['sample_end'] == returns.index[-1]
    assert actual.attrs['base_portfolio'] == prices.columns[base_id]
    assert actual['Base Sharpe'].nunique() == 1
    assert pd.api.types.is_integer_dtype(actual['Observations'])


def test_common_sample_does_not_fill_or_bridge_missing_navs(prices):
    partial = prices.copy()
    partial.iloc[:80, 2] = np.nan
    partial.iloc[110, 1] = np.nan
    multi = make_multi(partial)
    actual = multi.get_sharpe_difference_table(freq=None, af=260)
    returns = partial.pct_change(fill_method=None).iloc[1:].dropna()
    assert partial.index[110] not in returns.index
    assert partial.index[111] not in returns.index
    assert actual['Observations'].eq(len(returns)).all()
    assert actual.attrs['sample_start'] == partial.index[81]
    for name in actual.index:
        expected = qis.sharpe_difference_test(returns[name], returns.iloc[:, 0], af=260)
        assert actual.loc[name, 'HAC SE'] == pytest.approx(expected.se_an)


def test_window_is_nav_baseline_and_outside_prices_do_not_leak(prices):
    period = qis.TimePeriod('2005-12-30', '2009-12-31')
    original = make_multi(prices).get_sharpe_difference_table(time_period=period, freq=None, af=260)
    changed = prices.copy()
    changed.loc[changed.index < period.start] *= 100.0
    changed.loc[changed.index > period.end] *= 0.01
    actual = make_multi(changed).get_sharpe_difference_table(time_period=period, freq=None, af=260)
    pd.testing.assert_frame_equal(actual, original)
    assert actual.attrs['sample_start'] == pd.Timestamp('2006-01-02')
    assert actual['Observations'].eq(len(prices.loc[period.start:period.end]) - 1).all()


def test_no_partial_month_or_external_benchmark(prices):
    multi = make_multi(prices)
    expected = multi.get_sharpe_difference_table()
    multi.set_benchmark_prices(prices.iloc[:, 0].rename('External'))
    actual = multi.get_sharpe_difference_table()
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.attrs['sample_end'] == pd.Timestamp('2010-06-30')
    assert actual.attrs['af'] == 12.0
    assert 'External' not in actual.index


def test_reordered_base_and_sorted_dates(prices):
    initial = make_multi(prices).get_sharpe_difference_table(freq=None, af=260)
    reordered = make_multi(prices.iloc[::-1, [1, 0, 2]])
    actual = reordered.get_sharpe_difference_table(base_portfolio_id=1, freq=None, af=260)
    pd.testing.assert_frame_equal(actual, initial)


def test_formatted_table_and_plot_preserve_numeric_precision(prices):
    multi = make_multi(prices)
    raw = multi.get_sharpe_difference_table(freq=None, af=260)
    display = multi.get_sharpe_difference_table(freq=None, af=260, is_convert_to_str=True)
    for column in raw:
        for name in raw.index:
            expected = (f'{raw.loc[name, column]:d}' if column in ('Observations', 'HAC lags')
                        else f'{raw.loc[name, column]:0.2f}')
            assert display.loc[name, column] == expected
    fig, actual = multi.plot_sharpe_difference_table(freq=None, af=260)
    try:
        pd.testing.assert_frame_equal(actual, raw)
        assert isinstance(fig, plt.Figure)
        cells = fig.axes[0].tables[0].get_celld()
        assert cells[(1, 1)].get_text().get_text() == display.iloc[0, 0]
        assert cells[(1, 6)].get_text().get_text() == display.iloc[0, 5]
    finally:
        plt.close(fig)
    fig, ax = plt.subplots()
    try:
        returned, data = qis.plot_sharpe_difference_table(multi, freq=None, af=260,
                                                          ax=ax, title='Custom title')
        assert returned is None
        assert len(ax.tables) == 1
        pd.testing.assert_frame_equal(data, raw)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('base_id', [-1, 3, True, 'SEQ_US', 0.5])
def test_invalid_base(prices, base_id):
    with pytest.raises(ValueError, match='base_portfolio_id'):
        make_multi(prices).get_sharpe_difference_table(base_portfolio_id=base_id)


@pytest.mark.parametrize('case', ['one', 'duplicate_names', 'empty', 'short', 'constant',
                                 'zero', 'infinite', 'no_af', 'bad_af', 'bad_lags'])
def test_invalid_sample_or_settings(prices, case):
    navs = prices.copy()
    kwargs = dict(freq=None, af=260)
    if case == 'one':
        navs = navs.iloc[:, :1]
    elif case == 'duplicate_names':
        navs.columns = ['same', 'same', 'other']
    elif case == 'empty':
        kwargs['time_period'] = qis.TimePeriod('2025-01-01', '2026-01-01')
    elif case == 'short':
        navs = navs.iloc[:24]
    elif case == 'constant':
        navs.iloc[:, 1] = 1.0
    elif case == 'zero':
        navs.iloc[50, 1] = 0.0
    elif case == 'infinite':
        navs.iloc[50, 1] = np.inf
    elif case == 'no_af':
        kwargs['af'] = None
    elif case == 'bad_af':
        kwargs['af'] = -1
    elif case == 'bad_lags':
        kwargs['n_lags'] = -1
    if case == 'duplicate_names':
        multi = qis.MultiPortfolioData([qis.PortfolioData(navs.iloc[:, i]) for i in range(3)])
    else:
        multi = make_multi(navs)
    with pytest.raises(ValueError):
        multi.get_sharpe_difference_table(**kwargs)
