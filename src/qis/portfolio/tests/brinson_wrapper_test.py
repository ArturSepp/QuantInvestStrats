"""Brinson native-date accounting and display-frequency reconciliation."""
import numpy as np
import pandas as pd
import pytest
import qis

from qis.datasets.synthetic import generate_synthetic_prices


def make_portfolios(cost=0.0):
    """Trade within reporting months on the bundled deterministic universe."""
    prices = generate_synthetic_prices(
        start='2020-12-31', end='2021-06-30', apply_quirks=False)
    strategy_prices = prices[['SEQ_US', 'SBD_TSY']]
    dates = strategy_prices.index[::7]
    equity = np.where(np.arange(len(dates)) % 2 == 0, .8, .3)
    weights = pd.DataFrame({'SEQ_US': equity, 'SBD_TSY': 1 - equity}, index=dates)
    strategy = qis.backtest_model_portfolio(
        strategy_prices, weights, rebalancing_costs=cost, ticker='Strategy')
    benchmark = qis.backtest_model_portfolio(
        prices[['SEQ_US', 'SBD_IG']], [.4, .6], rebalancing_freq='ME',
        rebalancing_costs=cost, ticker='Benchmark')
    strategy.set_group_data(pd.Series({'SEQ_US': 'Equity', 'SBD_TSY': 'Bonds'}))
    benchmark.set_group_data(pd.Series({'SEQ_US': 'Equity', 'SBD_IG': 'Bonds'}))
    return strategy, benchmark


@pytest.mark.parametrize('cost', [0.0, .002])
def test_coarser_contributions_compound_within_month(cost):
    """Daily instrument contributions compound to each monthly NAV return."""
    strategy, _ = make_portfolios(cost)
    pnl, _ = strategy.get_brinson_inputs(freq='ME', is_net=True)
    monthly_nav = strategy.nav.resample('ME').last()
    expected = monthly_nav.pct_change().iloc[1:]
    np.testing.assert_allclose(pnl.sum(axis=1), expected, atol=1e-13)


@pytest.mark.parametrize('net', [False, True])
def test_wrapper_preserves_trades_prior_weights_and_report_baseline(net):
    """The standard wrapper agrees with native attribution and independent NAV ratios."""
    strategy, benchmark = make_portfolios(.002 if net else 0.0)
    multi = qis.MultiPortfolioData([strategy, benchmark])
    period = qis.TimePeriod('2021-01-29', '2021-06-30')
    native = multi.compute_brinson_attribution(time_period=period, is_net=net)
    monthly = multi.compute_brinson_attribution(time_period=period, freq='ME', is_net=net)
    quarterly = multi.compute_brinson_attribution(time_period=period, freq='QE', is_net=net)
    assert native[1].index[0] == pd.Timestamp('2021-02-01')
    for result in (monthly, quarterly):
        pd.testing.assert_frame_equal(result[0], native[0])
    for number in range(1, 5):
        pd.testing.assert_frame_equal(monthly[number], native[number].resample('ME').sum())
        pd.testing.assert_frame_equal(quarterly[number], native[number].resample('QE').sum())
    actual = native[1].sum(axis=1).cumsum()
    baseline = pd.Timestamp('2021-01-29')
    expected = (strategy.nav / strategy.nav.loc[baseline]
                - benchmark.nav / benchmark.nav.loc[baseline]).loc[actual.index]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    # Native allocation uses the weights held BEFORE each price movement.
    prices = benchmark.prices
    returns = qis.to_returns(prices, is_log_returns=False)
    old_weights = benchmark.weights.shift(1)
    benchmark_sector_return = returns['SBD_IG']
    strategy_weights = strategy.weights.shift(1)
    raw_bond_allocation = ((strategy_weights['SBD_TSY'] - old_weights['SBD_IG'])
                          * benchmark_sector_return).loc[actual.index]
    if not net:
        raw = multi.compute_brinson_attribution(
            time_period=period, is_linked=False, is_net=False)
        np.testing.assert_allclose(raw[2]['Bonds'], raw_bond_allocation, atol=1e-13)
    assert set(native[0].index) == {'Equity', 'Bonds', 'Total Sum'}


def test_wrapper_default_linking_reconciles_to_total_growth():
    """Calling the established API without new keywords fixes compound reporting."""
    strategy, benchmark = make_portfolios()
    result = qis.MultiPortfolioData([strategy, benchmark]).compute_brinson_attribution(freq='ME')
    total = strategy.nav.iloc[-1] / strategy.nav.iloc[0] - (
        benchmark.nav.iloc[-1] / benchmark.nav.iloc[0])
    assert result[0].loc['Total Sum', 'Total\nActive'] == pytest.approx(total)


def test_offline_brinson_example_reconciles_and_exports(tmp_path, monkeypatch):
    """The documented example executes outside the repository with no network."""
    from pathlib import Path
    import runpy

    example = Path(__file__).resolve().parents[4] / 'examples/portfolios/brinson_attribution.py'
    if not example.exists():
        pytest.skip('Repository-only example is not distributed with an installed wheel')
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(example))
    _, result = namespace['run_example'](output_dir=tmp_path)
    assert (tmp_path / 'brinson_attribution.pdf').stat().st_size > 0
    assert (tmp_path / 'summary.csv').stat().st_size > 0
    assert np.isfinite(result[0].to_numpy()).all()
