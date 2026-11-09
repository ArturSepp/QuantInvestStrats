"""Independent aggregation, causality and unchanged-contract checks for monthly cash."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import qis
from qis.fx import fx_returns as fx


def sample():
    """Supply changing and negative rates, leap-year dates and a foreign asset."""
    dates = pd.date_range("2023-12-31", periods=16, freq="ME")
    rates = pd.DataFrame({"USD": np.resize([.03, .08, -.01, .04], 16),
                          "EUR": np.resize([.01, -.02, .09, .02], 16)}, index=dates)
    data = qis.FxRatesData(
        fx_spots=pd.DataFrame({"USD": 1., "EUR": 1.1 + .02*np.sin(np.arange(16))},
                              index=dates), domestic_rates=rates)
    prices = pd.DataFrame({"asset": 100*np.exp(.01*np.arange(16))}, index=dates)
    return data, prices


def request(prices, frequency="QE", **kwargs):
    """Make an explicit log excess request against the monthly cash account."""
    return fx.FxAssetReturnsRequest(
        prices=prices, local_ccys="USD", reference_ccy="USD",
        return_type=fx.FxReturnType.REFERENCE_EXCESS,
        coordinate=fx.FxReturnCoordinate.LOG, frequency=frequency,
        cash_policy=fx.FxCashPolicy(accrual=fx.FxCashAccrual.MONTHLY_INDEX),
        diagnostics=True, **kwargs)


def test_quarterly_cash_matches_independent_monthly_product_and_log_sum():
    """A fixed wealth path has additive monthly/quarterly cash-relative log returns."""
    data, prices = sample()
    monthly = fx.compute_fx_returns(data, request(prices, "ME"))["ME"]
    quarterly = fx.compute_fx_returns(data, request(prices))["QE"]
    dt = prices.index.to_series().diff().dt.days/365.
    cm = data.domestic_rates.USD.shift(1)*dt
    index = (1+cm.fillna(0)).cumprod()
    cq = index.resample("QE").last().pct_change(fill_method=None)
    expected = np.log(prices.asset.resample("QE").last()).diff() - np.log1p(cq)
    np.testing.assert_allclose(quarterly.values.asset, expected, rtol=0, atol=1e-14)
    summed = monthly.values.asset.resample("QE").sum(min_count=1)
    np.testing.assert_allclose(quarterly.values.asset, summed, rtol=0, atol=1e-14)
    np.testing.assert_allclose(quarterly.diagnostics["cash"].USD.iloc[1:], cq.iloc[1:])
    legacy = fx.compute_fx_returns(data, replace(request(prices),
        cash_policy=fx.FxCashPolicy()))["QE"]
    assert (quarterly.values-legacy.values).abs().max().max() > 1e-4


def test_monthly_returns_and_hedge_leg_remain_exact():
    """Changing measurement cash leaves monthly values and the quarterly hedge intact."""
    data, prices = sample()
    for frequency in ("ME", "QE"):
        new = replace(request(prices, frequency), local_ccys="EUR", hedge_ratios=.7)
        old = replace(new, cash_policy=fx.FxCashPolicy())
        a = fx.compute_fx_returns(data, old)[frequency]
        b = fx.compute_fx_returns(data, new)[frequency]
        if frequency == "ME":
            pd.testing.assert_frame_equal(a.values, b.values, check_exact=True)
        for key in ("spot", "forward_premium", "hedge_cost"):
            if key in a.diagnostics:
                pd.testing.assert_frame_equal(
                    a.diagnostics[key], b.diagnostics[key], check_exact=True)
        total_request = replace(old, return_type=fx.FxReturnType.REFERENCE_TOTAL)
        total = fx.compute_fx_returns(data, total_request)[frequency].values
        recovered = b.values.add(np.log1p(b.diagnostics["cash"].USD), axis=0)
        np.testing.assert_allclose(recovered, total, rtol=0, atol=1e-14)


def test_index_is_sample_independent_and_partial_months_do_not_reset():
    """Sampling between month ends observes the same simple-within-month account."""
    data, _ = sample()
    dates = pd.to_datetime(["2023-12-31", "2024-01-15", "2024-01-31", "2024-02-15"])
    actual = qis.compute_monthly_cash_index(data, "USD", dates)
    january = 1+.03*31/365
    expected = [1., 1+.03*15/365, january, january*(1+.08*15/365)]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    cropped = qis.compute_monthly_cash_index(data, "USD", dates[1:])
    pd.testing.assert_series_equal(cropped, actual.iloc[1:])
    shifted = data.domestic_rates.copy()
    shifted.loc[shifted.index > "2024-01-31"] += 5
    changed = qis.FxRatesData(fx_spots=data.fx_spots, domestic_rates=shifted)
    pd.testing.assert_series_equal(actual, qis.compute_monthly_cash_index(changed, "USD", dates))


def test_missing_cash_is_not_backfilled_or_skipped():
    """The cash index starts at a known opening and cannot cross an unknown interval."""
    data, prices = sample()
    data.domestic_rates.loc[prices.index[0], "USD"] = np.nan
    index = qis.compute_monthly_cash_index(data, "USD", prices.index)
    assert pd.isna(index.iloc[0]) and index.iloc[1] == 1.
    data.domestic_rates.loc[prices.index[4], "USD"] = np.nan
    index = qis.compute_monthly_cash_index(data, "USD", prices.index)
    assert index.iloc[5:].isna().all()
    result = fx.compute_fx_returns(data, request(prices, "ME"))["ME"].values
    assert result.iloc[5:].isna().all().all()


@pytest.mark.parametrize("kind", [fx.FxReturnType.LOCAL_EXCESS, fx.FxReturnType.REFERENCE_EXCESS])
@pytest.mark.parametrize("coordinate", list(fx.FxReturnCoordinate))
def test_coordinates_use_the_same_cash_account(kind, coordinate):
    """Simple excess and relative coordinates share the measured cash growth."""
    data, prices = sample()
    req = replace(request(prices), return_type=kind, coordinate=coordinate,
                  reference_ccy=None if kind is fx.FxReturnType.LOCAL_EXCESS else "USD")
    result = fx.compute_fx_returns(data, req)["QE"]
    total = prices.asset.resample("QE").last().pct_change(fill_method=None)
    cash = result.diagnostics["cash"].USD
    expected = (total-cash if coordinate is fx.FxReturnCoordinate.SIMPLE else
                (1+total)/(1+cash)-1 if coordinate is fx.FxReturnCoordinate.CASH_RELATIVE_SIMPLE
                else np.log1p(total)-np.log1p(cash))
    np.testing.assert_allclose(result.values.asset, expected, rtol=0, atol=1e-14)


def test_monthly_policy_rejects_term_contracts_and_nonopening_quotes():
    """Realised account growth cannot silently reprice an opening forward contract."""
    with pytest.raises(ValueError, match="opening"):
        fx.FxCashPolicy(accrual=fx.FxCashAccrual.MONTHLY_INDEX, quote_lag=0)
    with pytest.raises(TypeError, match="FxCashAccrual"):
        fx.FxCashPolicy(accrual="monthly_index")
    data, prices = sample()
    for kind in (fx.FxReturnType.REFERENCE_TOTAL, fx.FxReturnType.LOCAL_EXCESS_TRANSLATED):
        with pytest.raises(ValueError, match="measurement"):
            fx.compute_fx_returns(data, replace(request(prices), return_type=kind))
    with pytest.raises(ValueError, match="measurement"):
        fx.compute_fx_returns(data, fx.FxCurrencyReturnsRequest(
            currencies=("EUR",), reference_ccy="USD", return_type=fx.FxReturnType.FX_FORWARD,
            cash_policy=fx.FxCashPolicy(accrual=fx.FxCashAccrual.MONTHLY_INDEX)))


def test_invalid_gross_growth_is_rejected():
    """Cash-relative wealth requires positive account growth."""
    data, prices = sample()
    data.domestic_rates.iloc[0, 0] = -20
    with pytest.raises(ValueError, match="positive"):
        qis.compute_monthly_cash_index(data, "USD", prices.index)


def test_missing_new_reset_quote_preserves_completed_cash_return():
    """The quote at a closing boundary only finances the following month."""
    data, prices = sample()
    dates = prices.index[:5]
    before = qis.compute_monthly_cash_index(data, "USD", dates)
    data.domestic_rates.loc[dates[-1], "USD"] = np.nan
    after = qis.compute_monthly_cash_index(data, "USD", dates)
    pd.testing.assert_series_equal(before, after, check_exact=True)
