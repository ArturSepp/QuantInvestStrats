"""FX input errors name required panels without rejecting independent calculations."""
from dataclasses import replace
import re

import numpy as np
import pandas as pd
import pytest
import qis


DATES = pd.date_range("2023-12-31", periods=7, freq="ME")
T = qis.FxReturnType
C = qis.FxReturnCoordinate
CASES = [
    (T.LOCAL_TOTAL, "EUR", (), ()),
    (T.LOCAL_EXCESS, "EUR", (), ("EUR",)),
    (T.REFERENCE_TOTAL, "EUR", ("EUR", "USD"), ("EUR", "USD")),
    (T.REFERENCE_TOTAL, "USD", (), ()),
    (T.REFERENCE_EXCESS, "EUR", ("EUR", "USD"), ("EUR", "USD")),
    (T.REFERENCE_EXCESS, "USD", (), ("USD",)),
    (T.LOCAL_EXCESS_TRANSLATED, "EUR", ("EUR", "USD"), ("EUR",)),
    (T.LOCAL_EXCESS_TRANSLATED, "USD", (), ("USD",)),
]


def _market(spots=("EUR", "USD"), rates=("EUR", "USD")):
    """Build only the requested columns, retaining the same observation calendar."""
    return qis.FxRatesData(
        pd.DataFrame({c: (1.0 if c == "USD" else np.linspace(1.1, 1.2, len(DATES)))
                      for c in spots}, index=DATES),
        pd.DataFrame({c: (0.05 if c == "USD" else 0.03) for c in rates}, index=DATES),
    )


def _request(kind, currency, diagnostics=False, coordinate=C.SIMPLE):
    """Use one local asset with a reference only when the economic kind requires it."""
    reference = None if kind in (T.LOCAL_TOTAL, T.LOCAL_EXCESS) else "USD"
    return qis.FxAssetReturnsRequest(
        prices=pd.DataFrame({"a": np.linspace(100., 110., len(DATES))}, index=DATES),
        local_ccys=currency, return_type=kind, coordinate=coordinate,
        reference_ccy=reference, diagnostics=diagnostics,
    )


def _coordinates(kind):
    """Return the economically valid coordinate choices for this test case."""
    if kind is T.LOCAL_EXCESS_TRANSLATED:
        return (C.SIMPLE,)
    if kind in (T.LOCAL_EXCESS, T.REFERENCE_EXCESS):
        return (C.SIMPLE, C.LOG, C.CASH_RELATIVE_SIMPLE)
    return (C.SIMPLE, C.LOG)


def _required(kind, local, spots, rates, diagnostics):
    """Extend the explicit test contract for currency-level diagnostic legs."""
    spots, rates = set(spots), set(rates)
    if diagnostics:
        rates.add(local)
        if kind not in (T.LOCAL_TOTAL, T.LOCAL_EXCESS):
            spots.update((local, "USD"))
            rates.add("USD")
    return spots, rates


@pytest.mark.parametrize("hedge", [0, 1, np.int32(1), np.int64(1), np.float32(.25),
                                    np.float64(.5), 1., -.25, 1.2, np.nan])
@pytest.mark.parametrize("log", [False, True])
def test_numeric_scalar_hedges_equal_float_payoffs(hedge, log):
    """Numeric scalars preserve the exact float kernel, including signed and missing hedges."""
    fx = _market()
    price = _request(T.REFERENCE_TOTAL, "EUR").prices.a
    cross = fx.get_local_to_reference_fx_rate("EUR", "USD")
    premium = fx.get_forward_rate_for_local_ccy("EUR", "USD", is_log_returns=log)
    kernel = qis.compute_performance_of_local_ccy_asset_in_reference_ccy
    actual = kernel(price, cross, premium, hedge, is_log_returns=log)
    expected = kernel(price, cross, premium, float(hedge), is_log_returns=log)
    for a, b in zip(actual, expected):
        pd.testing.assert_series_equal(a, b, check_exact=True)


@pytest.mark.parametrize("hedge", [True, np.bool_(False), "1", None, 1j, [1]])
def test_unsupported_scalar_hedge_names_value(hedge):
    """Unsupported objects fail before calculation with their offending value."""
    fx = _market()
    price = _request(T.REFERENCE_TOTAL, "EUR").prices.a
    with pytest.raises(TypeError, match="hedge_ratio must be a number or a pd.Series") as exc:
        qis.compute_performance_of_local_ccy_asset_in_reference_ccy(
            price, fx.fx_spots.EUR, pd.Series(0., index=DATES), hedge)
    assert repr(hedge) in str(exc.value)


@pytest.mark.parametrize("kind,local,spots,rates", CASES)
@pytest.mark.parametrize("diagnostics", [False, True])
def test_only_required_asset_panels_are_needed(kind, local, spots, rates, diagnostics):
    """Minimal coverage gives identical labelled results to a fully supplied market."""
    spots, rates = _required(kind, local, spots, rates, diagnostics)
    for coordinate in _coordinates(kind):
        request = _request(kind, local, diagnostics, coordinate)
        actual = qis.compute_fx_returns(_market(spots, rates), request)["ME"]
        expected = qis.compute_fx_returns(_market(), request)["ME"]
        pd.testing.assert_frame_equal(actual.values, expected.values, check_exact=True)
        for name, value in actual.diagnostics.items():
            pd.testing.assert_frame_equal(value, expected.diagnostics[name], check_exact=True)


@pytest.mark.parametrize("kind,local,spots,rates", CASES)
@pytest.mark.parametrize("diagnostics", [False, True])
def test_missing_required_asset_column_names_panel(kind, local, spots, rates, diagnostics):
    """Each individually missing required column gets an actionable coverage error."""
    spots, rates = _required(kind, local, spots, rates, diagnostics)
    for coordinate in _coordinates(kind):
        request = _request(kind, local, diagnostics, coordinate)
        for panel, required in (("fx_spots", spots), ("domestic_rates", rates)):
            for currency in required:
                fx = _market(spots - {currency} if panel == "fx_spots" else spots,
                             rates - {currency} if panel == "domestic_rates" else rates)
                message = re.escape(f"{panel} lacks {[currency]!r}")
                with pytest.raises(ValueError, match=message):
                    qis.compute_fx_returns(fx, request)


def test_missing_both_panels_reports_both_currency_sets():
    """Missing local and reference columns are reported together, not as a bare KeyError."""
    expected = "fx_spots lacks ['EUR', 'USD'], domestic_rates lacks ['EUR', 'USD']"
    with pytest.raises(ValueError, match=re.escape(expected)):
        qis.compute_fx_returns(_market((), ()), _request(T.REFERENCE_TOTAL, "EUR"))


@pytest.mark.parametrize("kind", [T.FX_SPOT, T.FX_FORWARD])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_currency_requests_require_only_their_own_legs(kind, diagnostics):
    """Spot-only diagnostics do not demand rates; forwards require both cash currencies."""
    rates = ("EUR", "USD") if kind is T.FX_FORWARD else ()
    request = qis.FxCurrencyReturnsRequest(
        ("EUR", "USD"), "USD", kind, frequency="ME", diagnostics=diagnostics)
    coordinates = (C.SIMPLE, C.LOG) if kind is T.FX_SPOT else (C.SIMPLE,)
    for coordinate in coordinates:
        req = replace(request, coordinate=coordinate)
        actual = qis.compute_fx_returns(_market(rates=rates), req)["ME"]
        expected = qis.compute_fx_returns(_market(), req)["ME"]
        pd.testing.assert_frame_equal(actual.values, expected.values, check_exact=True)
        for panel, currencies in (("fx_spots", ("EUR", "USD")), ("domestic_rates", rates)):
            for currency in currencies:
                fx = _market()
                setattr(fx, panel, getattr(fx, panel).drop(columns=currency))
                with pytest.raises(ValueError, match=re.escape(f"{panel} lacks {[currency]!r}")):
                    qis.compute_fx_returns(fx, req)


@pytest.mark.parametrize("dated", [False, True])
def test_unused_currency_metadata_is_ignored_with_mixed_cadence(dated):
    """Extra metadata assets require no coverage and do not change either cadence panel."""
    request = _request(T.REFERENCE_EXCESS, "EUR", diagnostics=True)
    prices = request.prices.assign(b=request.prices.a * 2)
    ccys = pd.Series({"a": "EUR", "b": "USD", "unused": "CHF"})
    if dated:
        ccys = pd.DataFrame([ccys, ccys], index=[DATES[0], DATES[-1]])
    req = replace(request, prices=prices, local_ccys=ccys,
                  frequency=pd.Series({"a": "ME", "b": "QE"}))
    minimal = ccys.loc[:, ["a", "b"]] if dated else ccys.loc[["a", "b"]]
    actual = qis.compute_fx_returns(_market(), req)
    expected = qis.compute_fx_returns(_market(), replace(req, local_ccys=minimal))
    for freq in actual:
        pd.testing.assert_frame_equal(actual[freq].values, expected[freq].values, check_exact=True)


def test_diagnostics_validate_only_sampled_currency_spells():
    """A local-total intramonth spell needs no rate if it appears at no diagnostic boundary."""
    dates = pd.bdate_range("2024-01-31", "2024-06-28")
    prices = pd.DataFrame({"a": np.linspace(100., 110., len(dates))}, index=dates)
    ccys = pd.DataFrame({"a": ["EUR", "CHF", "EUR"]},
                        index=pd.to_datetime(["2024-01-31", "2024-02-09", "2024-02-20"]))
    request = qis.FxAssetReturnsRequest(
        prices, ccys, T.LOCAL_TOTAL, C.SIMPLE, diagnostics=True, currency_changes="mask_boundary")
    fx = qis.FxRatesData(pd.DataFrame(index=dates),
                        pd.DataFrame({"EUR": .03}, index=dates))
    result = qis.compute_fx_returns(fx, request)["ME"]
    assert result.diagnostics["cash"].columns.tolist() == ["EUR"]
    assert result.values.a.iloc[2:].notna().all()


def test_missing_quotes_and_foreign_hedges_stay_missing():
    """Column validation does not reject or fill a supported observation-level gap."""
    fx = _market()
    fx.fx_spots.loc[DATES[:2], "EUR"] = np.nan
    fx.domestic_rates.loc[DATES[:2], "EUR"] = np.nan
    before = fx.fx_spots.copy(), fx.domestic_rates.copy()
    request = replace(_request(T.REFERENCE_EXCESS, "EUR"), hedge_ratios=np.nan)
    result = qis.compute_fx_returns(fx, request)["ME"]
    assert result.values.a.isna().all()
    pd.testing.assert_frame_equal(fx.fx_spots, before[0])
    pd.testing.assert_frame_equal(fx.domestic_rates, before[1])


@pytest.mark.parametrize("reference", [" ", 7])
@pytest.mark.parametrize("currency_request", [False, True])
def test_reference_currency_must_be_a_nonblank_id(reference, currency_request):
    """Invalid reference metadata fails before constructing a cross or sorting IDs."""
    request = (qis.FxCurrencyReturnsRequest(("EUR",), reference) if currency_request else
               replace(_request(T.REFERENCE_EXCESS, "EUR"), reference_ccy=reference))
    with pytest.raises(ValueError, match="reference_ccy.*nonblank currency ID"):
        qis.compute_fx_returns(_market(), request)
