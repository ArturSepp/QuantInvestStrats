"""Holdings beta keeps native allocation dates and warms up each instrument."""
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
