"""Public cash access and explicit compounded-index construction preserve FX economics."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import qis


def _market():
    """Keep off-grid rate changes, including one observed after the last March spot."""
    spots = pd.DataFrame({"USD": 1., "EUR": 1.1}, index=pd.to_datetime([
        "2023-12-29", "2024-01-31", "2024-02-29", "2024-03-28",
        "2024-04-02", "2024-04-30",
    ]))
    rates = pd.DataFrame({"USD": [.04, .05, .06, .20, .25], "EUR": .02},
                         index=pd.to_datetime([
                             "2023-12-29", "2024-01-15", "2024-02-15",
                             "2024-03-30", "2024-04-15",
                         ]))
    return qis.FxRatesData(spots, rates)


@pytest.mark.parametrize("lag", [0, 1, np.int64(2)])
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("shuffled", [False, True])
def test_period_cash_matches_independent_dated_accrual(lag, log, shuffled):
    """Actual days, lagging and weekend rate projection follow the stored spot calendar."""
    data = _market()
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"])
    # March 30's rate enters the stored panel on April 2, after the March 31 boundary.
    quotes = [.05, .06, .06, .25]
    values = [0.]
    for i in range(1, len(dates)):
        value = quotes[i - lag] * (dates[i] - dates[i - 1]).days / 365 if i >= lag else np.nan
        values.append(value)
    expected = pd.Series(values, index=dates, name="USD")
    if log:
        expected = np.log1p(expected)
    if shuffled:
        dates = dates[[3, 0, 2, 1]]
        expected = expected.reindex(dates)
    before = data.fx_spots.copy(deep=True), data.domestic_rates.copy(deep=True), dates.copy()
    actual = data.get_period_cash_returns(dates, "USD", quote_lag=lag, is_log_returns=log)
    pd.testing.assert_series_equal(actual, expected, rtol=0, atol=1e-15)
    pd.testing.assert_frame_equal(data.fx_spots, before[0])
    pd.testing.assert_frame_equal(data.domestic_rates, before[1])
    pd.testing.assert_index_equal(dates, before[2])


@pytest.mark.parametrize("dates", [pd.DatetimeIndex([]), pd.to_datetime(["2023-01-01"])])
@pytest.mark.parametrize("lag", [0, 1, 3])
def test_empty_or_single_boundary_has_no_earned_interest(dates, lag):
    """No elapsed interval earns cash, including a first boundary before the rate history."""
    actual = _market().get_period_cash_returns(dates, "USD", quote_lag=lag)
    expected = pd.Series(0., index=dates, name="USD")
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("lag", [True, np.bool_(False), -1, 1.5, "1", None])
def test_public_cash_rejects_invalid_quote_lag_even_on_empty_dates(lag):
    """The public spelling names invalid boundary lags before the empty-grid shortcut."""
    with pytest.raises(ValueError, match="quote_lag must be a nonnegative integer"):
        _market().get_period_cash_returns(pd.DatetimeIndex([]), "USD", quote_lag=lag)


def test_default_cash_is_opening_simple_with_unknown_history_preserved():
    """The initial zero does not turn a later unsupported period into free funding."""
    dates = pd.to_datetime(["2023-12-01", "2023-12-29", "2024-01-31"])
    actual = _market().get_period_cash_returns(dates, "USD")
    expected = pd.Series([0., np.nan, .04 * 33 / 365], index=dates, name="USD")
    pd.testing.assert_series_equal(actual, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("log", [False, True])
def test_public_cash_rejects_nonpositive_gross_growth(log):
    """Simple and logarithmic cash retain the same strict-positive gross-factor gate."""
    data = _market()
    data.domestic_rates.loc[:, "USD"] = -20.
    with pytest.raises(ValueError, match="Cash gross factors must be strictly positive"):
        data.get_period_cash_returns(pd.to_datetime(["2024-01-31", "2024-02-29"]),
                                     "USD", is_log_returns=log)


def test_private_cash_alias_is_removed():
    """The promoted method has one public implementation and no compatibility delegate."""
    assert not hasattr(qis.FxRatesData, "_get_period_cash_returns")


def _payoff_result(coordinate):
    """Create a typed result with known increments and one post-inception gap."""
    dates = pd.date_range("2024-01-31", periods=5, freq="ME")
    values = pd.DataFrame({"asset": [np.nan, .10, np.nan, -.20, .05]}, index=dates)
    if coordinate is qis.FxReturnCoordinate.LOG:
        values = np.log1p(values)
    kind = (qis.FxReturnType.FX_FORWARD if coordinate is qis.FxReturnCoordinate.SIMPLE
            else qis.FxReturnType.REFERENCE_EXCESS)
    return qis.FxReturnsData(
        return_type=kind, coordinate=coordinate, frequency="ME", reference_ccy="USD",
        cash_policy=qis.FxCashPolicy(), embedded_hedge_policy="additional",
        normalization=None, _values=values,
        _local_ccys=pd.DataFrame("EUR", index=dates, columns=values.columns),
        _opening_hedge_ratios=pd.DataFrame(0., index=dates, columns=values.columns),
    )


@pytest.mark.parametrize("initial", [1., 100.])
@pytest.mark.parametrize("missing", ["skip", "stop"])
@pytest.mark.parametrize("coordinate", list(qis.FxReturnCoordinate))
def test_compounded_constructor_equals_explicit_policy_and_cash_flows(initial, missing, coordinate):
    """Return the same seven fields and index values for either gap policy."""
    norm = qis.FxIndexNormalization.compounded("ME", initial_value=initial, missing_returns=missing)
    explicit = qis.FxIndexNormalization(
        initial, 1., "capital_proportional", "ME", "each_interval", "as_returned", missing)
    assert norm == explicit
    result = _payoff_result(coordinate)
    before = result.values
    assert result.synthetic_index is None and result.normalization is None
    expected = np.array([1., 1.1, 1.1, 1.1 * .8, 1.1 * .8 * 1.05]) * initial
    if missing == "stop":
        expected[2:] = np.nan
    pd.testing.assert_frame_equal(
        result.to_index(norm), result.to_index(explicit), check_exact=True)
    np.testing.assert_allclose(result.to_index(norm).asset, expected, rtol=0, atol=1e-13)
    requested = replace(result, normalization=norm)
    assert requested.index_kind == "compounded_index"
    pd.testing.assert_frame_equal(
        requested.synthetic_index, result.to_index(norm), check_exact=True)
    pd.testing.assert_frame_equal(result.values, before, check_exact=True)
    assert result.synthetic_index is None


def test_compounded_defaults_and_explicit_nonunit_or_fixed_policies():
    """Unit compounding is the shorthand; nonunit and fixed notional still use explicit fields."""
    norm = qis.FxIndexNormalization.compounded("ME")
    assert norm == qis.FxIndexNormalization(
        1., 1., "capital_proportional", "ME", "each_interval", "as_returned", "skip")
    result = _payoff_result(qis.FxReturnCoordinate.SIMPLE)
    scaled = replace(norm, initial_value=100., notional_to_capital=2.)
    np.testing.assert_allclose(result.to_index(scaled).asset, [100., 120., 120., 72., 79.2])
    fixed = replace(scaled, notional_rule="fixed", reinvestment="none")
    np.testing.assert_allclose(result.to_index(fixed).asset, [100., 120., 120., 80., 90.])
    assert replace(result, normalization=fixed).index_kind == "accumulated_pnl_with_initial_capital"
    with pytest.raises(TypeError):
        qis.FxIndexNormalization.compounded()
    with pytest.raises(TypeError):
        result.to_index()
    with pytest.raises(ValueError, match="settlement_frequency"):
        result.to_index(qis.FxIndexNormalization.compounded("QE"))


@pytest.mark.parametrize("kwargs", [
    {"initial_value": 0}, {"initial_value": -1}, {"initial_value": np.nan},
    {"initial_value": np.inf}, {"missing_returns": "invent_zero"},
])
def test_compounded_constructor_retains_existing_validation(kwargs):
    """The named constructor delegates to the existing normalization validation."""
    with pytest.raises(ValueError):
        qis.FxIndexNormalization.compounded("ME", **kwargs)
