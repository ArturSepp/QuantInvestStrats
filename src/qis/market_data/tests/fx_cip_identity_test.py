"""Exact FX-forward wealth and cash-relative return regression tests.

A hedge of beginning-of-period principal does not remove FX exposure on the
asset's return. Covered interest parity is tested exactly by hedging the
known terminal local-cash amount, rather than asserting currency invariance
for an arbitrary risky asset with a principal hedge of one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import qis
from qis.market_data import FxRatesData
from qis.market_data import fx_hedging


def _market() -> tuple[FxRatesData, pd.Series]:
    """Return deterministic monthly prices, unequal rates and a moving cross."""
    dates = pd.date_range('2024-01-31', periods=6, freq='ME')
    spots = pd.DataFrame({'USD': 1.0, 'CHF': [1.1, 1.2, 0.95, 1.25, 1.0, 1.3]},
                         index=dates)
    rates = pd.DataFrame({'USD': [0.24, -0.12, 0.48, 0.06, 0.36, 0.18],
                         'CHF': [0.6, 0.12, -0.24, 0.36, 0.48, 0.06]}, index=dates)
    prices = pd.Series([100.0, 110.0, 97.0, 120.0, 108.0, 115.0],
                       index=dates, name='ASSET')
    return FxRatesData(fx_spots=spots, domestic_rates=rates), prices


def _terminal_wealth_returns(prices: pd.Series,
                            spots: pd.Series,
                            local_cash: pd.Series,
                            reference_cash: pd.Series,
                            hedge: float | pd.Series) -> pd.Series:
    """Price an asset and sold local-currency forward from terminal cash flows.

    Args:
        prices: Local asset prices on consecutive contract dates.
        spots: Reference currency per local currency on those dates.
        local_cash: Simple local deposit return contracted on each date.
        reference_cash: Simple reference deposit return contracted on each date.
        hedge: Sold local units divided by beginning asset units' local value.

    Returns:
        Exact simple returns with the package's synthetic inception zero.
    """
    initial_reference_wealth = prices.shift(1) * spots.shift(1)
    local_units_sold = prices.shift(1) * (
        hedge.shift(1) if isinstance(hedge, pd.Series) else hedge)
    contracted_forward = spots.shift(1) * (
        (1.0 + reference_cash.shift(1)) / (1.0 + local_cash.shift(1)))
    final_reference_wealth = prices * spots + local_units_sold * (contracted_forward - spots)
    result = (final_reference_wealth / initial_reference_wealth - 1.0).rename(prices.name)
    result.iloc[0] = 0.0
    return result


def _cash_at_period_start(data: FxRatesData, dates: pd.DatetimeIndex,
                          currency: str, annualization: float) -> pd.Series:
    """Read only cash quotes known at each preceding asset observation date."""
    result = [np.nan]
    for date in dates[:-1]:
        observed = data.domestic_rates.loc[:date, currency].dropna()
        result.append(np.nan if observed.empty else float(observed.iloc[-1]) / annualization)
    return pd.Series(result, index=dates)


@pytest.mark.parametrize('is_log', [False, True])
@pytest.mark.parametrize('hedge_kind', ['none', 'half', 'principal', 'changing'])
def test_hedged_return_matches_terminal_forward_wealth(is_log: bool, hedge_kind: str) -> None:
    """All hedge choices reproduce independently valued asset-plus-forward wealth."""
    data, prices = _market()
    hedge = {'none': 0.0, 'half': 0.5, 'principal': 1.0,
             'changing': pd.Series([0.0, 0.8, 0.2, 1.0, 0.5, 0.1], index=prices.index)}[hedge_kind]
    spots = data.fx_spots['CHF']
    premium = data.get_forward_rate_for_local_ccy('CHF', 'USD', is_log_returns=is_log)
    nav, actual = fx_hedging.compute_performance_of_local_ccy_asset_in_reference_ccy(
        prices, spots, premium, hedge, is_log_returns=is_log)
    expected_simple = _terminal_wealth_returns(
        prices, spots, data.domestic_rates['CHF'] / 12.0,
        data.domestic_rates['USD'] / 12.0, hedge)
    expected = np.log1p(expected_simple) if is_log else expected_simple
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(nav, (1.0 + expected_simple).cumprod(), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize('frequency', ['ME', 'QE', '2W-WED'])
def test_unhedged_log_return_matches_price_product(frequency: str) -> None:
    """An unhedged log return equals the return of the reference-currency price."""
    dates = pd.date_range('2024-01-03', periods=210, freq='D')
    t = np.arange(len(dates), dtype=float)
    prices = pd.Series(np.exp(0.005 * t + 0.1 * np.sin(t / 11)), index=dates, name='A')
    spots = pd.Series(np.exp(-0.003 * t + 0.1 * np.cos(t / 7)), index=dates)
    _, actual = fx_hedging.compute_performance_of_local_ccy_asset_in_reference_ccy(
        prices, spots, pd.Series(0.0, index=dates), 0.0,
        freq=frequency, is_log_returns=True)
    expected = qis.to_returns(prices * spots, freq=frequency, is_log_returns=True)
    expected.iloc[0] = 0.0
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14, equal_nan=True)


@pytest.mark.parametrize('is_log', [False, True])
@pytest.mark.parametrize('frequency', ['ME', 'QE'])
def test_forward_getter_preserves_reciprocal_quote(is_log: bool, frequency: str) -> None:
    """The public premium getter keeps its existing quote and return convention."""
    data, _ = _market()
    annualization = 12.0 if frequency == 'ME' else 4.0
    expected = ((1.0 + data.domestic_rates['CHF'] / annualization)
                / (1.0 + data.domestic_rates['USD'] / annualization) - 1.0)
    if is_log:
        expected = np.log1p(expected)
    actual = data.get_forward_rate_for_local_ccy(
        'CHF', 'USD', freq=frequency, is_log_returns=is_log)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    missing = data.domestic_rates.copy()
    missing.iloc[0, :] = np.nan
    data = FxRatesData(data.fx_spots, missing)
    assert pd.isna(data.get_forward_rate_for_local_ccy(
        'CHF', 'USD', freq=frequency, is_log_returns=is_log).iloc[0])


@pytest.mark.parametrize('is_log', [False, True])
@pytest.mark.parametrize('currency', ['CHF', 'USD'])
@pytest.mark.parametrize('bad_annual_rate', [-12.0, -13.0])
def test_forward_getter_rejects_nonpositive_rate_gross(
        is_log: bool, currency: str, bad_annual_rate: float) -> None:
    """Neither side of a CIP deposit can have nonpositive terminal wealth."""
    data, _ = _market()
    data.domestic_rates.loc[data.domestic_rates.index[1], currency] = bad_annual_rate
    with pytest.raises(ValueError):
        data.get_forward_rate_for_local_ccy('CHF', 'USD', is_log_returns=is_log)


@pytest.mark.parametrize('is_log', [False, True])
def test_terminal_cash_hedge_satisfies_exact_covered_interest_parity(is_log: bool) -> None:
    """Hedging terminal local cash exactly reproduces a reference-currency deposit."""
    data, prices = _market()
    local_cash = data.domestic_rates['CHF'] / 12.0
    reference_cash = data.domestic_rates['USD'] / 12.0
    prices = 100.0 * (1.0 + local_cash.shift(1).fillna(0.0)).cumprod()
    prices.name = 'CASH'
    premium = data.get_forward_rate_for_local_ccy('CHF', 'USD', is_log_returns=is_log)
    _, actual = fx_hedging.compute_performance_of_local_ccy_asset_in_reference_ccy(
        prices, data.fx_spots['CHF'], premium, 1.0 + local_cash, is_log_returns=is_log)
    expected = reference_cash.shift(1).fillna(0.0)
    if is_log:
        expected = np.log1p(expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    _, principal_only = fx_hedging.compute_performance_of_local_ccy_asset_in_reference_ccy(
        prices, data.fx_spots['CHF'], premium, 1.0, is_log_returns=is_log)
    assert float((principal_only - expected).abs().max()) > 0.001


@pytest.mark.parametrize('local_currency', ['USD', 'CHF'])
@pytest.mark.parametrize('is_log', [False, True])
def test_excess_return_uses_previous_reference_cash_quote(
        local_currency: str, is_log: bool) -> None:
    """Same- and cross-currency excess returns subtract correctly timed reference cash."""
    data, prices = _market()
    _, total = data.compute_performance_of_local_ccy_asset_in_reference_ccy(
        prices, 0.5, local_currency, 'USD', is_log_returns=is_log)
    nav, actual = data.compute_performance_of_local_ccy_asset_in_reference_ccy(
        prices, 0.5, local_currency, 'USD', is_log_returns=is_log, is_excess_returns=True)
    cash = _cash_at_period_start(data, total.index, 'USD', 12.0)
    expected = total - (np.log1p(cash) if is_log else cash)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14, equal_nan=True)
    assert actual.isna().equals(total.isna())
    if is_log:
        relative_nav = np.exp(total.fillna(0.0).cumsum()) / (1.0 + cash.fillna(0.0)).cumprod()
        np.testing.assert_allclose(nav, relative_nav, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize('is_log', [False, True])
@pytest.mark.parametrize('frequency', ['ME', '2W-WED'])
def test_panel_excess_cash_uses_actual_return_grid(is_log: bool, frequency: str) -> None:
    """Panel and local excess paths use prior asset dates, including alternate biweekly phases."""
    dates = pd.date_range('2024-01-03', periods=100, freq='D')
    t = np.arange(len(dates), dtype=float)
    data = FxRatesData(
        pd.DataFrame({'USD': 1.0, 'CHF': np.exp(0.002 * t)}, index=dates),
        pd.DataFrame({'USD': 0.12 + 0.004 * t, 'CHF': -0.10 + 0.002 * t}, index=dates))
    asset_dates = dates[7:]
    prices = pd.DataFrame({'A': np.exp(0.003 * t[7:]),
                           'B': np.exp(0.005 * t[7:])}, index=asset_dates)
    currencies = pd.Series({'A': 'USD', 'B': 'CHF'})
    hedges = pd.Series({'A': 0.0, 'B': 0.0})
    total = data.compute_fx_adjusted_returns(
        prices, hedges, currencies, freq=frequency, is_log_returns=is_log)[frequency]
    actual = data.compute_fx_adjusted_returns(
        prices, hedges, currencies, freq=frequency, is_log_returns=is_log,
        is_excess_returns=True)[frequency]
    annualization = 12.0 if frequency == 'ME' else 26.0
    for asset in prices.columns:
        # Each dispatch can have a different grid before the panel outer join.
        _, asset_total = data.compute_performance_of_local_ccy_asset_in_reference_ccy(
            prices[asset], 0.0, currencies[asset], 'USD', freq=frequency,
            is_log_returns=is_log)
        cash = _cash_at_period_start(data, asset_total.index, 'USD', annualization)
        expected = asset_total - (np.log1p(cash) if is_log else cash)
        expected = expected.reindex(total.index)
        np.testing.assert_allclose(actual[asset], expected, rtol=1e-13, atol=1e-14, equal_nan=True)
    local, excess, reported_rates = data.compute_returns_adjusted_by_local_rate(
        prices, currencies, freq=frequency, is_log_returns=is_log)
    for asset, currency in currencies.items():
        cash = _cash_at_period_start(data, local.index, currency, annualization)
        expected = local[asset] - (np.log1p(cash) if is_log else cash)
        np.testing.assert_allclose(excess[asset], expected, rtol=1e-13, atol=1e-14, equal_nan=True)
    pd.testing.assert_frame_equal(
        reported_rates, data.fetch_local_rates(currencies, freq=frequency, annualise=True))


@pytest.mark.parametrize('is_log', [False, True])
def test_hedge_and_forward_shocks_are_lagged_once(is_log: bool) -> None:
    """Changing a contract-date hedge or rate first affects the following return."""
    data, prices = _market()
    hedge = pd.Series(0.5, index=prices.index)
    premium = data.get_forward_rate_for_local_ccy('CHF', 'USD', is_log_returns=is_log)
    _, baseline = fx_hedging.compute_performance_of_local_ccy_asset_in_reference_ccy(
        prices, data.fx_spots['CHF'], premium, hedge, is_log_returns=is_log)
    hedge.iloc[2] = 0.9
    premium.iloc[2] = np.log1p(0.3) if is_log else 0.3
    _, shocked = fx_hedging.compute_performance_of_local_ccy_asset_in_reference_ccy(
        prices, data.fx_spots['CHF'], premium, hedge, is_log_returns=is_log)
    np.testing.assert_allclose(shocked.iloc[:3], baseline.iloc[:3], rtol=0, atol=0)
    assert abs(shocked.iloc[3] - baseline.iloc[3]) > 0.01
    np.testing.assert_allclose(shocked.iloc[4:], baseline.iloc[4:], rtol=0, atol=0)


@pytest.mark.parametrize('is_log', [False, True])
@pytest.mark.parametrize('currency', ['USD', 'CHF'])
def test_leading_missing_prices_preserve_return_support(is_log: bool, currency: str) -> None:
    """Missing asset inception dates stay missing in both total and excess output."""
    data, prices = _market()
    prices.iloc[:2] = np.nan
    _, total = data.compute_performance_of_local_ccy_asset_in_reference_ccy(
        prices, 0.0, currency, 'USD', is_log_returns=is_log)
    _, excess = data.compute_performance_of_local_ccy_asset_in_reference_ccy(
        prices, 0.0, currency, 'USD', is_log_returns=is_log, is_excess_returns=True)
    assert total.iloc[:3].isna().all()
    assert total.iloc[3:].notna().all()
    assert excess.isna().equals(total.isna())


@pytest.mark.parametrize('hedge', [1.0, 1.5])
def test_nonpositive_hedged_wealth_is_rejected_only_for_log_output(hedge: float) -> None:
    """Simple output retains total loss or negative wealth; log output rejects both."""
    dates = pd.date_range('2024-01-31', periods=2, freq='ME')
    prices = pd.Series([100.0, 50.0], index=dates, name='A')
    spots = pd.Series([1.0, 2.0], index=dates)
    premium = pd.Series(0.0, index=dates)
    _, simple = fx_hedging.compute_performance_of_local_ccy_asset_in_reference_ccy(
        prices, spots, premium, hedge, is_log_returns=False)
    assert simple.iloc[1] == -hedge
    with pytest.raises(ValueError):
        fx_hedging.compute_performance_of_local_ccy_asset_in_reference_ccy(
            prices, spots, premium, hedge, is_log_returns=True)


@pytest.mark.parametrize('bad_annual_rate', [-12.0, -13.0])
def test_log_excess_rejects_nonpositive_cash_wealth(bad_annual_rate: float) -> None:
    """Both excess-return entry points reject cash denominators without a real logarithm."""
    data, prices = _market()
    data.domestic_rates.loc[prices.index[1], 'USD'] = bad_annual_rate
    with pytest.raises(ValueError):
        data.compute_performance_of_local_ccy_asset_in_reference_ccy(
            prices, 0.0, 'USD', 'USD', is_log_returns=True, is_excess_returns=True)
    with pytest.raises(ValueError):
        data.compute_returns_adjusted_by_local_rate(
            prices.to_frame(), pd.Series({'ASSET': 'USD'}), is_log_returns=True)


def test_optimal_hedge_uses_forward_execution_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optimizer's carry tilt uses the same contracted forward payoff as performance."""
    data, prices = _market()
    vol = pd.Series(0.2, index=prices.index)
    beta = pd.Series(0.3, index=prices.index)

    def fixed_vol_beta(**kwargs: object) -> tuple[pd.Series, pd.Series]:
        """Return known risk inputs to isolate the optimizer's carry convention."""
        return vol, beta

    monkeypatch.setattr(fx_hedging, 'compute_fx_vol_beta', fixed_vol_beta)
    premium = data.get_forward_rate_for_local_ccy('CHF', 'USD', is_log_returns=False)
    optimal, carry, beta_hedge = fx_hedging.compute_fx_optimal_hedge(
        prices, data.fx_spots['CHF'], premium, risk_aversion_lambda=2.0, min_max_hedge=None)
    forward_over_spot = ((1.0 + data.domestic_rates['USD'] / 12.0)
                         / (1.0 + data.domestic_rates['CHF'] / 12.0))
    lost_forward_value = 1.0 - forward_over_spot
    expected_carry = 1.0 - lost_forward_value * 12.0 / (2.0 * 2.0 * 0.2 ** 2)
    np.testing.assert_allclose(carry, expected_carry, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(optimal, expected_carry + 0.3, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(beta_hedge, 1.3, rtol=0, atol=0)


def test_forward_and_hedge_history_fill_before_weekend_valuation() -> None:
    """Weekend valuations use prior valid contracts, including gaps in business-day quotes."""
    data, prices = _market()
    quote_dates = pd.date_range('2024-01-02', '2024-06-28', freq='B')
    premium = pd.Series(np.linspace(0.001, 0.06, len(quote_dates)), index=quote_dates)
    hedge = pd.Series(np.linspace(0.2, 0.8, len(quote_dates)), index=quote_dates)
    # Missing ordinary quotes retain the latest contract already known; they
    # must be filled before discarding dates outside the monthly valuation grid.
    premium.loc[['2024-03-29', '2024-05-31']] = np.nan
    hedge.loc[['2024-03-29', '2024-05-31']] = np.nan
    known_premium = pd.Series(
        [premium.loc[:date].dropna().iloc[-1] for date in prices.index], index=prices.index)
    known_hedge = pd.Series(
        [hedge.loc[:date].dropna().iloc[-1] for date in prices.index], index=prices.index)
    _, actual = fx_hedging.compute_performance_of_local_ccy_asset_in_reference_ccy(
        prices, data.fx_spots['CHF'], premium, hedge, is_log_returns=False)
    # A zero reference deposit makes the getter's reciprocal premium equal to
    # the local deposit return, so terminal cash flows provide a separate oracle.
    expected = _terminal_wealth_returns(
        prices, data.fx_spots['CHF'], known_premium,
        pd.Series(0.0, index=prices.index), known_hedge)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    assert prices.index[2].dayofweek == 6
    assert known_premium.iloc[2] == premium.loc['2024-03-28']


def test_optimal_hedge_uses_latest_business_day_forward_quote(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A Sunday month-end hedge decision uses the most recent Friday forward quote."""
    data, prices = _market()
    vol = pd.Series(0.2, index=prices.index)
    beta = pd.Series(0.3, index=prices.index)

    def fixed_vol_beta(**kwargs: object) -> tuple[pd.Series, pd.Series]:
        """Supply monthly risk estimates to isolate business-day quote alignment."""
        return vol, beta

    monkeypatch.setattr(fx_hedging, 'compute_fx_vol_beta', fixed_vol_beta)
    quote_dates = pd.date_range('2024-01-02', '2024-06-28', freq='B')
    premium = pd.Series(np.linspace(0.001, 0.06, len(quote_dates)), index=quote_dates)
    optimal, carry, _ = fx_hedging.compute_fx_optimal_hedge(
        prices, data.fx_spots['CHF'], premium, risk_aversion_lambda=2.0, min_max_hedge=None)
    known_premium = pd.Series(
        [premium.loc[:date].iloc[-1] for date in prices.index], index=prices.index)
    forward_over_spot = 1.0 / (1.0 + known_premium)
    expected_carry = 1.0 - (1.0 - forward_over_spot) * 12.0 / (2.0 * 2.0 * 0.2 ** 2)
    np.testing.assert_allclose(carry, expected_carry, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(optimal, expected_carry + 0.3, rtol=1e-13, atol=1e-14)
    assert prices.index[2].dayofweek == 6
    assert known_premium.iloc[2] == premium.loc['2024-03-29']
