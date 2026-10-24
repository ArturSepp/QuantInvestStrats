"""Holdings beta keeps native allocation dates and warms up each instrument."""
import warnings

import numpy as np
import pandas as pd
import pytest

import qis
from qis.portfolio.risk.ewm_factor_model import _get_instrument_beta_estimation_start


def _monthly_prices():
    index = pd.date_range('2007-06-30', periods=35, freq='ME')
    factor = 0.02 * np.sin(np.arange(len(index)) + 0.3) + 0.003
    benchmark = pd.DataFrame({'Benchmark': np.exp(np.cumsum(factor))}, index=index)
    prices = pd.DataFrame({
        'High': np.exp(np.cumsum(2.0 * factor)),
        'Low': np.exp(np.cumsum(0.5 * factor)),
    }, index=index)
    return prices, benchmark


def test_month_end_opening_allocation_is_available_for_lagged_alpha():
    prices, benchmark = _monthly_prices()
    index = prices.index
    weights = pd.DataFrame(0.0, index=index, columns=prices.columns)
    start = pd.Timestamp('2008-06-30')  # Monday; Wednesday resampling loses this allocation.
    weights.loc[start:, 'High'] = 1.0
    weights.loc[index[13]:, 'High'] = 0.0
    weights.loc[index[13]:, 'Low'] = 1.0
    # Non-zero fee returns before investment must not supply the title's estimation date.
    nav = pd.Series(np.exp(-0.02 * np.arange(len(index)) / 12.0), index=index, name='TAA')
    nav.loc[start:] = nav.loc[start] * prices.loc[start:, 'High'] / prices.loc[start, 'High']
    portfolio = qis.PortfolioData(nav=nav, prices=prices, weights=weights)

    beta = portfolio.compute_portfolio_benchmark_betas(
        benchmark_prices=benchmark, freq_beta='ME', factor_beta_span=3,
    )['Benchmark']

    assert beta.loc[start] == pytest.approx(2.0)
    assert beta.loc[index[13]] == pytest.approx(0.5)
    multi = qis.MultiPortfolioData(portfolio_datas=[portfolio], benchmark_prices=benchmark)
    alpha = multi.compute_rolling_benchmark_alpha(
        benchmark_price=benchmark['Benchmark'], freq_beta='ME', factor_beta_span=3,
        time_period=qis.TimePeriod(start=start, end=index[-1]),
    )
    assert alpha.loc[start, 'TAA'] == 0.0
    assert alpha.loc[index[13], 'TAA'] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize('span', [3, 12])
@pytest.mark.parametrize('freq', ['ME', 'W-WED'])
def test_beta_warmup_uses_span_and_never_substitutes_unity(span, freq):
    prices, benchmark = _monthly_prices()
    if freq == 'W-WED':
        prices.index = benchmark.index = pd.date_range('2007-06-27', periods=len(prices), freq=freq)
    weights = pd.DataFrame({'High': 1.0, 'Low': 0.0}, index=prices.index)
    beta = qis.compute_portfolio_ewm_benchmark_betas(
        instrument_prices=prices, benchmark_prices=benchmark, weights=weights,
        freq_beta=freq, factor_beta_span=span,
    )['Benchmark']

    # Row zero is a missing return, followed by span finite warm-up observations.
    assert beta.iloc[:span + 1].isna().all()
    assert beta.iloc[span + 1] == pytest.approx(2.0)
    assert beta.iloc[span + 1:].notna().all()


def test_late_instrument_has_its_own_warmup_and_zero_weights_do_not_block():
    prices, benchmark = _monthly_prices()
    prices.loc[prices.index[:15], 'Low'] = np.nan
    weights = pd.DataFrame({'High': 1.0, 'Low': 0.0}, index=prices.index)
    weights.loc[prices.index[15]:, 'High'] = 0.0
    weights.loc[prices.index[15]:, 'Low'] = 1.0
    beta = qis.compute_portfolio_ewm_benchmark_betas(
        instrument_prices=prices, benchmark_prices=benchmark, weights=weights,
        freq_beta='ME', factor_beta_span=3,
    )['Benchmark']

    assert beta.iloc[10] == pytest.approx(2.0)
    assert beta.iloc[15:19].isna().all()
    assert np.isfinite(beta.iloc[19])


def test_negative_and_cash_betas_survive_without_a_unity_prior():
    prices, benchmark = _monthly_prices()
    weights = pd.DataFrame({'High': -1.0, 'Low': 0.0}, index=prices.index)
    weights.iloc[20:] = 0.0
    beta = qis.compute_portfolio_ewm_benchmark_betas(
        instrument_prices=prices, benchmark_prices=benchmark, weights=weights,
        freq_beta='ME', factor_beta_span=3,
    )['Benchmark']

    np.testing.assert_allclose(beta.iloc[4:20], -2.0, atol=1e-12)
    assert beta.iloc[20:].eq(0.0).all()


def test_alpha_remains_missing_while_a_new_held_instrument_warms_up():
    prices, benchmark = _monthly_prices()
    prices.loc[prices.index[:15], 'Low'] = np.nan
    weights = pd.DataFrame({'High': 1.0, 'Low': 0.0}, index=prices.index)
    weights.loc[prices.index[15]:, 'High'] = 0.0
    weights.loc[prices.index[15]:, 'Low'] = 1.0
    portfolio = qis.PortfolioData(nav=prices['High'].rename('TAA'), prices=prices, weights=weights)
    multi = qis.MultiPortfolioData(portfolio_datas=[portfolio], benchmark_prices=benchmark)

    alpha = multi.compute_rolling_benchmark_alpha(
        benchmark_price=benchmark['Benchmark'], freq_beta='ME', factor_beta_span=3,
        time_period=qis.TimePeriod(start=prices.index[0], end=prices.index[-1]),
    )['TAA']

    assert np.isfinite(alpha.iloc[10])
    assert alpha.loc[prices.index[16:20]].isna().all()
    assert np.isfinite(alpha.loc[prices.index[20]])


def test_warmup_beta_prefix_is_point_in_time():
    prices, benchmark = _monthly_prices()
    weights = pd.DataFrame({'High': 0.7, 'Low': 0.3}, index=prices.index)
    full = qis.compute_portfolio_ewm_benchmark_betas(
        instrument_prices=prices, benchmark_prices=benchmark, weights=weights,
        freq_beta='ME', factor_beta_span=3,
    )
    prefix = qis.compute_portfolio_ewm_benchmark_betas(
        instrument_prices=prices.iloc[:25], benchmark_prices=benchmark.iloc[:25],
        weights=weights.iloc[:25], freq_beta='ME', factor_beta_span=3,
    )

    pd.testing.assert_frame_equal(full.iloc[:25], prefix)


def test_short_history_alpha_stays_unavailable_without_a_beta_prior():
    prices, benchmark = _monthly_prices()
    prices, benchmark = prices.iloc[:8], benchmark.iloc[:8]
    weights = pd.DataFrame({'High': 1.0, 'Low': 0.0}, index=prices.index)
    portfolio = qis.PortfolioData(nav=prices['High'].rename('TAA'), prices=prices, weights=weights)
    multi = qis.MultiPortfolioData(portfolio_datas=[portfolio], benchmark_prices=benchmark)

    alpha = multi.compute_rolling_benchmark_alpha(
        benchmark_price=benchmark['Benchmark'], freq_beta='ME', factor_beta_span=12,
        time_period=qis.TimePeriod(start=prices.index[0], end=prices.index[-1]),
    )['TAA']

    assert alpha.iloc[0] == 0.0
    assert alpha.iloc[1:].isna().all()


def test_beta_estimation_title_uses_paired_instrument_history():
    prices, benchmark = _monthly_prices()
    benchmark.iloc[:4] = np.nan

    start = _get_instrument_beta_estimation_start(
        instrument_prices=prices, benchmark_prices=benchmark, freq='ME',
    )

    assert start == prices.index[5]


@pytest.mark.parametrize('known_weight, young_weight', [
    (0.8, 0.2), (1.6, 0.4), (-0.6, 0.4), (1.0, 1e-12),
])
def test_partial_benchmark_beta_keeps_actual_weights(known_weight, young_weight):
    """A young holding cannot erase mature holdings or cause renormalisation."""
    prices, benchmark = _monthly_prices()
    prices.loc[prices.index[:20], 'Low'] = np.nan
    weights = pd.DataFrame({'High': known_weight, 'Low': young_weight},
                           index=prices.index[10:])
    weights.loc[prices.index[30]:] = 0.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        beta = qis.compute_portfolio_ewm_benchmark_betas(
            instrument_prices=prices, benchmark_prices=benchmark, weights=weights,
            freq_beta='ME', factor_beta_span=3,
        )['Benchmark']

    assert beta.iloc[:10].isna().all()  # no allocation yet
    np.testing.assert_allclose(beta.iloc[10:24], 2.0 * known_weight, atol=1e-12)
    assert beta.iloc[24:30].notna().all()
    assert beta.iloc[30:].eq(0.0).all()
    coverage_warnings = [w for w in caught if 'Partial benchmark beta coverage' in str(w.message)]
    assert len(coverage_warnings) == 1
    assert 'Low' in str(coverage_warnings[0].message)
    assert 'not renormalised' in str(coverage_warnings[0].message)


def test_quarterly_alpha_starts_with_backtest_despite_a_young_holding():
    """Prehistory warms mature instruments before the June 2008 opening allocation."""
    index = pd.date_range('1999-12-31', periods=60, freq='QE')
    factor = pd.Series(0.02 * np.sin(np.arange(len(index)) + 0.3) + 0.003, index=index)
    benchmark = np.exp(factor.cumsum()).rename('Benchmark')
    prices = pd.DataFrame({'High': np.exp((2.0 * factor).cumsum()),
                           'Young': np.exp((0.5 * factor).cumsum())})
    start = pd.Timestamp('2008-06-30')
    prices.loc[index < '2007-12-31', 'Young'] = np.nan
    weights = pd.DataFrame({'High': 0.95, 'Young': 0.05}, index=index[index >= start])
    portfolio = qis.PortfolioData(nav=prices['High'].rename('TAA'),
                                  prices=prices, weights=weights)
    multi = qis.MultiPortfolioData(portfolio_datas=[portfolio],
                                   benchmark_prices=benchmark.to_frame())
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        beta = portfolio.compute_portfolio_benchmark_betas(
            benchmark_prices=benchmark.to_frame(), freq_beta='QE', factor_beta_span=12,
        )['Benchmark']
        alpha = multi.compute_rolling_benchmark_alpha(
            benchmark_price=benchmark, freq_beta='QE', factor_beta_span=12,
            time_period=qis.TimePeriod(start=start, end=index[-1]),
        )['TAA']

    assert beta.loc[start] == pytest.approx(1.9)
    assert beta.loc[start:].notna().all()
    assert alpha.loc[start] == 0.0
    assert alpha.iloc[1:].notna().all()
    # Independently calculated log residuals and pandas EWMA, on the partial-coverage window.
    dates = index[(index > start) & (index <= '2010-12-31')]
    expected = (0.1 * factor.loc[dates]).ewm(span=12, adjust=False).mean() * 4.0
    np.testing.assert_allclose(alpha.loc[dates], expected, atol=1e-12)


def test_partial_benchmark_beta_prefix_is_point_in_time():
    """Future observations that warm a young asset do not revise past exposures."""
    prices, benchmark = _monthly_prices()
    prices.loc[prices.index[:20], 'Low'] = np.nan
    weights = pd.DataFrame({'High': 0.7, 'Low': 0.3}, index=prices.index)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        full = qis.compute_portfolio_ewm_benchmark_betas(
            instrument_prices=prices, benchmark_prices=benchmark, weights=weights,
            freq_beta='ME', factor_beta_span=3,
        )
        prefix = qis.compute_portfolio_ewm_benchmark_betas(
            instrument_prices=prices.iloc[:23], benchmark_prices=benchmark.iloc[:23],
            weights=weights.iloc[:23], freq_beta='ME', factor_beta_span=3,
        )
    np.testing.assert_allclose(prefix.iloc[4:], 1.4, atol=1e-12)
    pd.testing.assert_frame_equal(full.iloc[:23], prefix)
