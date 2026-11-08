"""Typed FX contracts: independent arithmetic, timing and normalization references."""

import numpy as np
import pandas as pd
import pytest
import qis


def market():
    """Two actual month intervals including leap February and changing opening quotes."""
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    fx = qis.FxRatesData(
        pd.DataFrame({"USD": 1.0, "EUR": [1.1, 1.2, 1.15]}, index=dates),
        pd.DataFrame({"USD": [0.05, 0.06, 0.07], "EUR": [0.03, 0.04, 0.05]}, index=dates),
    )
    prices = pd.DataFrame({"equity": [100.0, 102.0, 99.0]}, index=dates)
    return fx, prices


def test_typed_reference_excess_is_cash_relative_log():
    """The canonical log excess differs from log1p(arithmetic excess)."""
    fx, prices = market()
    request = qis.FxAssetReturnsRequest(
        prices=prices,
        local_ccys="EUR",
        reference_ccy="USD",
        hedge_ratios=0.5,
        return_type=qis.FxReturnType.REFERENCE_EXCESS,
        coordinate=qis.FxReturnCoordinate.LOG,
        frequency="ME",
        diagnostics=True,
    )
    result = qis.compute_fx_returns(fx, request)["ME"]
    c_local, c_ref = 0.03 * 29 / 365, 0.05 * 29 / 365
    x = 1.2 / 1.1 - 1
    f = (1 + c_local) / (1 + c_ref) - 1
    total = 0.02 * (1 + x) + 0.5 * x - 0.5 * f / (1 + f)
    assert result.values.iloc[1, 0] == pytest.approx(np.log1p(total) - np.log1p(c_ref))
    assert result.synthetic_index is None
    assert result.periods.iloc[1]["opening"] == prices.index[0]
    assert result.cash_policy.day_count == "ACT/365F"
    assert result.opening_hedge_ratios.iloc[1, 0] == 0.5
    assert set(result.diagnostics) == {"cash", "spot", "hedge_cost"}


def test_invalid_kind_coordinate_is_not_silently_coerced():
    """Cash-relative simple is an excess coordinate only."""
    fx, prices = market()
    with pytest.raises(ValueError, match="excess"):
        qis.compute_fx_returns(
            fx,
            qis.FxAssetReturnsRequest(
                prices,
                "EUR",
                qis.FxReturnType.LOCAL_TOTAL,
                qis.FxReturnCoordinate.CASH_RELATIVE_SIMPLE,
            ),
        )


@pytest.mark.parametrize("frequency", ["B", "ME", "QE"])
@pytest.mark.parametrize("coordinate", list(qis.FxReturnCoordinate))
@pytest.mark.parametrize("kind", [qis.FxReturnType.REFERENCE_EXCESS, qis.FxReturnType.LOCAL_EXCESS])
def test_coordinates_match_independent_cash_identity(frequency, coordinate, kind):
    """Dated cash identities hold at each reset cadence, including irregular weekends."""
    dates = pd.bdate_range("2023-12-29", "2024-07-01")
    prices = pd.DataFrame({"a": 100 * np.exp(0.001 * np.arange(len(dates)))}, index=dates)
    spots = pd.DataFrame(
        {"EUR": 1.1 * np.exp(0.0002 * np.arange(len(dates))), "USD": 1.0}, index=dates
    )
    rates = pd.DataFrame({"EUR": 0.03, "USD": 0.05}, index=dates)
    fx = qis.FxRatesData(spots, rates)
    reference = "USD" if kind is qis.FxReturnType.REFERENCE_EXCESS else None
    request = qis.FxAssetReturnsRequest(
        prices, "EUR", kind, coordinate, reference_ccy=reference, frequency=frequency
    )
    got = qis.compute_fx_returns(fx, request)[frequency].values.a
    total_prices = prices.a * spots.EUR if reference else prices.a
    total = qis.to_returns(total_prices, freq=frequency, is_log_returns=False)
    cash = (0.05 if reference else 0.03) * got.index.to_series().diff().dt.days / 365
    expected = {
        qis.FxReturnCoordinate.SIMPLE: total - cash,
        qis.FxReturnCoordinate.CASH_RELATIVE_SIMPLE: (1 + total) / (1 + cash) - 1,
        qis.FxReturnCoordinate.LOG: np.log1p(total) - np.log1p(cash),
    }[coordinate]
    np.testing.assert_allclose(got, expected, atol=3e-15, rtol=1e-12, equal_nan=True)


def test_owned_results_and_mixed_cadence():
    """Caller edits and property edits cannot mutate returned values or hedge metadata."""
    fx, prices = market()
    prices["b"] = prices.equity
    req = qis.FxAssetReturnsRequest(
        prices,
        "EUR",
        qis.FxReturnType.REFERENCE_TOTAL,
        qis.FxReturnCoordinate.SIMPLE,
        "USD",
        frequency=pd.Series({"equity": "ME", "b": "QE"}),
    )
    prices.iloc[:, :] = 1
    result = qis.compute_fx_returns(fx, req)
    assert set(result) == {"ME", "QE"}
    saved = result["ME"].values
    returned_values = result["ME"].values
    returned_ccys = result["ME"].local_ccys
    returned_hedges = result["ME"].opening_hedge_ratios
    returned_values.iloc[:, :] = -20
    returned_ccys.iloc[:, :] = "bad"
    returned_hedges.iloc[:, :] = 5
    pd.testing.assert_frame_equal(result["ME"].values, saved)
    assert result["ME"].local_ccys.iloc[0, 0] == "EUR"
    assert result["ME"].opening_hedge_ratios.iloc[1, 0] == 0


def test_dated_hedges_are_known_at_opening():
    """An off-grid hedge update changes the following opening, not the preceding payoff."""
    fx, prices = market()
    h = pd.DataFrame(
        {"equity": [0.0, 1.0, 0.25]},
        index=pd.to_datetime(["2024-01-31", "2024-02-15", "2024-03-31"]),
    )
    req = qis.FxAssetReturnsRequest(
        prices,
        "EUR",
        qis.FxReturnType.REFERENCE_TOTAL,
        qis.FxReturnCoordinate.SIMPLE,
        "USD",
        hedge_ratios=h,
    )
    result = qis.compute_fx_returns(fx, req)["ME"]
    np.testing.assert_allclose(result.opening_hedge_ratios.equity.iloc[1:], [0.0, 1.0])
    h.iloc[-1] = 20
    pd.testing.assert_frame_equal(qis.compute_fx_returns(fx, req)["ME"].values, result.values)


def test_currency_spells_require_explicit_boundary_masking():
    """A unit-of-account change is never mistaken for an investment return."""
    from dataclasses import replace

    fx, prices = market()
    spells = pd.DataFrame({"equity": ["EUR", "USD", "USD"]}, index=prices.index)
    req = qis.FxAssetReturnsRequest(
        prices, spells, qis.FxReturnType.REFERENCE_TOTAL, qis.FxReturnCoordinate.SIMPLE, "USD"
    )
    with pytest.raises(ValueError, match="spell"):
        qis.compute_fx_returns(fx, req)
    result = qis.compute_fx_returns(fx, replace(req, currency_changes="mask_boundary"))["ME"]
    assert pd.isna(result.values.iloc[1, 0])
    assert result.values.iloc[2, 0] == pytest.approx(99 / 102 - 1)
    assert result.local_ccys.iloc[1, 0] == "USD"


def test_currency_panel_has_no_dummy_asset_or_cash_carry():
    """Spot requests use currency IDs directly, with no automatic index or interest."""
    fx, _ = market()
    result = qis.compute_fx_returns(fx, qis.FxCurrencyReturnsRequest(("EUR", "USD"), "USD"))["ME"]
    assert result.values.columns.tolist() == ["EUR", "USD"]
    assert result.values.EUR.iloc[1] == pytest.approx(1.2 / 1.1 - 1)
    assert result.synthetic_index is None
    assert result.values.USD.iloc[1] == 0


def test_optional_index_has_explicit_capital_and_notional():
    """Fixed and capital-proportional sizing are distinct, with no extra cash accrual."""
    from dataclasses import replace

    fx, prices = market()
    request = qis.FxAssetReturnsRequest(
        prices, "EUR", qis.FxReturnType.LOCAL_TOTAL, qis.FxReturnCoordinate.SIMPLE
    )
    result = qis.compute_fx_returns(fx, request)["ME"]
    sizing = qis.FxIndexNormalization(
        100.0, 2.0, "capital_proportional", "ME", "each_interval", "as_returned", "stop"
    )
    np.testing.assert_allclose(
        result.to_index(sizing).equity, [100, 104, 104 * (1 + 2 * (99 / 102 - 1))]
    )
    fixed = replace(sizing, notional_rule="fixed", reinvestment="none")
    requested = qis.compute_fx_returns(fx, replace(request, normalization=fixed))["ME"]
    assert requested.index_kind == "accumulated_pnl_with_initial_capital"
    np.testing.assert_allclose(
        requested.synthetic_index.equity, [100, 104, 104 + 200 * (99 / 102 - 1)]
    )
    with pytest.raises(ValueError, match="settlement"):
        result.to_index(replace(sizing, settlement_frequency="B"))


def test_arithmetic_loss_below_minus_one_survives_without_index():
    """Signed arithmetic excess is meaningful even when a compounded index is invalid."""
    from dataclasses import replace

    fx, prices = market()
    fx.domestic_rates.loc[:, "EUR"] = 20.0
    request = qis.FxAssetReturnsRequest(
        prices, "EUR", qis.FxReturnType.LOCAL_EXCESS, qis.FxReturnCoordinate.SIMPLE
    )
    result = qis.compute_fx_returns(fx, request)["ME"]
    assert result.values.iloc[1, 0] < -1
    assert result.synthetic_index is None
    norm = qis.FxIndexNormalization(
        100, 1, "capital_proportional", "ME", "each_interval", "as_returned", "stop"
    )
    with pytest.raises(ValueError, match="positive"):
        result.to_index(norm)
    assert np.isfinite(
        qis.compute_fx_returns(fx, replace(request, coordinate=qis.FxReturnCoordinate.LOG))[
            "ME"
        ].values.iloc[1, 0]
    )


@pytest.mark.parametrize("hedge", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("is_log", [False, True])
@pytest.mark.parametrize("is_excess", [False, True])
def test_same_policy_matches_retiring_analytics(hedge, is_log, is_excess):
    """Representation migration retains the independently regression-tested Stage 1 path."""
    fx, prices = market()
    with pytest.warns(DeprecationWarning, match="6.0.0"):
        _, expected = fx.compute_returns_in_reference_ccy(
            prices,
            pd.Series({"equity": hedge}),
            "EUR",
            "USD",
            is_log_returns=is_log,
            is_excess_returns=is_excess,
        )
    result = qis.compute_fx_returns(
        fx,
        qis.FxAssetReturnsRequest(
            prices,
            "EUR",
            qis.FxReturnType.REFERENCE_EXCESS if is_excess else qis.FxReturnType.REFERENCE_TOTAL,
            qis.FxReturnCoordinate.LOG if is_log else qis.FxReturnCoordinate.SIMPLE,
            "USD",
            hedge_ratios=hedge,
        ),
    )["ME"]
    pd.testing.assert_frame_equal(result.values, expected, check_exact=True)


def test_off_grid_currency_round_trip_is_not_hidden_by_sampling():
    """Even a return to the initial quote currency masks an intervening unit change."""
    fx, prices = market()
    spells = pd.DataFrame(
        {"equity": ["EUR", "USD", "EUR"]},
        index=pd.to_datetime(["2024-01-31", "2024-02-10", "2024-02-15"]),
    )
    req = qis.FxAssetReturnsRequest(
        prices,
        spells,
        qis.FxReturnType.REFERENCE_TOTAL,
        qis.FxReturnCoordinate.SIMPLE,
        "USD",
        currency_changes="mask_boundary",
    )
    result = qis.compute_fx_returns(fx, req)["ME"]
    assert pd.isna(result.values.equity.iloc[1])
    assert pd.notna(result.values.equity.iloc[2])
    pd.testing.assert_frame_equal(result.currency_spells, spells)
    returned_spells = result.currency_spells
    returned_spells.iloc[:, :] = "bad"
    pd.testing.assert_frame_equal(result.currency_spells, spells)


@pytest.mark.parametrize(
    "method",
    [
        "compute_returns_in_reference_ccy",
        "compute_fx_adjusted_returns",
        "compute_returns_adjusted_by_local_rate",
    ],
)
def test_retiring_methods_warn_with_replacement_and_removal(method):
    """Every superseded public entry warns from the first typed rollout."""
    fx, prices = market()
    common = dict(local_ccys=pd.Series({"equity": "EUR"}))
    if method != "compute_returns_adjusted_by_local_rate":
        common.update(hedge_ratios=pd.Series({"equity": 0.0}), reference_ccy="USD")
    common["prices" if method == "compute_fx_adjusted_returns" else "asset_prices"] = prices
    with pytest.warns(DeprecationWarning, match="compute_fx_returns.*6.0.0"):
        getattr(fx, method)(**common)


def test_single_asset_container_method_warns_without_changing_payoff():
    """The retained scalar container entry is deprecated; the quote kernel is separate."""
    fx, prices = market()
    with pytest.warns(DeprecationWarning, match='compute_fx_returns.*6.0.0'):
        _, values = fx.compute_performance_of_local_ccy_asset_in_reference_ccy(
            prices.equity, 0.0, 'EUR', 'USD')
    assert values.iloc[1] == pytest.approx(1.02 * 1.2 / 1.1 - 1)


def test_public_request_annotation_resolves_at_runtime():
    """Typed tooling can resolve the canonical service's real data-container annotation."""
    from typing import get_type_hints
    assert get_type_hints(qis.compute_fx_returns)['data'] is qis.FxRatesData
