"""Independent economic references for anchoring, FX and shared risk identity."""

from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.stress.instruments import (
    InstrumentLeg,
    InstrumentType,
    KinkPolicy,
    ResponseBasis,
    Underlying,
)
from qis.portfolio.stress.portfolio import PortfolioHolding


def holding(kind, mark, quantity=1.0, strike=None, key="actual", name="holding"):
    """Construct an original holding with one signed primitive."""
    return PortfolioHolding(name, name, mark, (InstrumentLeg(kind, key, quantity, 2.0, strike),))


def shocks(portfolio, equity=-0.2, fx=0.1):
    """Construct complete labelled log shocks for the independent two-driver examples."""
    factors = portfolio.risk_model.factor_loadings[portfolio.risk_date].columns
    return pd.DataFrame(
        [[math.log1p(equity), math.log1p(fx), 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        index=["stress", "flat"],
        columns=factors,
    )


@pytest.mark.parametrize("kind", list(InstrumentType))
@pytest.mark.parametrize("basis", list(ResponseBasis))
def test_fx_and_observed_mark_anchoring_match_scalar_economics(market, kind, basis):
    """A EUR payoff uses local price changes and exactly one stressed USD conversion."""
    option = kind in (InstrumentType.CALL, InstrumentType.PUT)
    mark = -7.0 if option else (0.0 if kind is InstrumentType.FUTURE else -240.0)
    h = holding(kind, mark, -1.0, 95.0 if option else None)
    p = market([h], currency="EUR", basis=basis)
    result = p.evaluate(shocks(p))
    g = math.log(0.8) + 0.4 * math.log(1.1)
    local = 100.0 * math.exp(g - (math.log(1.1) if basis is ResponseBasis.REFERENCE else 0.0))
    fx = 1.2 * 1.1
    if kind is InstrumentType.DELTA_1:
        expected = mark * (local / 100.0 * fx / 1.2 - 1.0)
        baseline = mark
    else:
        if kind is InstrumentType.CALL:
            baseline, stressed = -2.0 * max(100.0 - 95.0, 0.0), -2.0 * max(local - 95.0, 0.0)
        elif kind is InstrumentType.PUT:
            baseline, stressed = -2.0 * max(95.0 - 100.0, 0.0), -2.0 * max(95.0 - local, 0.0)
        else:
            baseline, stressed = 0.0, -2.0 * (local - 100.0)
        baseline *= 1.2
        expected = stressed * fx - baseline
    assert result.pnl.loc["stress", h.holding_id] == pytest.approx(expected)
    assert result.mtm.loc["stress", h.holding_id] == pytest.approx(mark + expected)
    assert result.audit.loc[h.holding_id, "model_baseline"] == pytest.approx(baseline)
    assert result.audit.loc[h.holding_id, "basis_offset"] == pytest.approx(mark - baseline)
    assert result.pnl.loc["flat", h.holding_id] == 0.0
    assert result.mtm.loc["flat", h.holding_id] == mark
    pd.testing.assert_series_equal(p.get_mtm(shocks(p).loc["stress"]), result.mtm.loc["stress"])
    pd.testing.assert_series_equal(p.get_pnl(shocks(p).loc["stress"]), result.pnl.loc["stress"])


@pytest.mark.parametrize("kind", list(InstrumentType))
@pytest.mark.parametrize("basis", list(ResponseBasis))
def test_current_factor_gradient_matches_independent_scenario_differences(market, kind, basis):
    """FX and local-spot derivatives agree away from kinks, even for zero-MTM futures."""
    h = holding(
        kind,
        0.0 if kind is InstrumentType.FUTURE else 25.0,
        3.0,
        90.0 if kind in (InstrumentType.CALL, InstrumentType.PUT) else None,
    )
    p = market([h], currency="EUR", basis=basis)
    betas = p.risk_model.factor_loadings[p.risk_date]
    analytic = p.response_jacobian() @ betas
    epsilon = 1e-6
    for factor in betas.columns:
        f = pd.Series(0.0, index=betas.columns)
        f.loc[factor] = epsilon
        numerical = (p.get_pnl(f).iloc[0] - p.get_pnl(-f).iloc[0]) / (2 * epsilon)
        assert analytic.loc[h.holding_id, factor] == pytest.approx(numerical, abs=1e-7)


@pytest.mark.parametrize(
    "side,kind,quantity,expected",
    [(KinkPolicy.RIGHT, "accumulator", 1.0, 100.0), (KinkPolicy.LEFT, "decumulator", -1.0, -100.0)],
)
def test_synthetic_accumulator_boundary_and_payoff(market, side, kind, quantity, expected):
    """One-sided holding policy reproduces favorable-side quantity at the strike."""
    if kind == "accumulator":
        legs = (
            InstrumentLeg(InstrumentType.CALL, "actual", 1.0, strike=100.0),
            InstrumentLeg(InstrumentType.PUT, "actual", -2.0, strike=100.0),
        )
    else:
        legs = (
            InstrumentLeg(InstrumentType.PUT, "actual", 1.0, strike=100.0),
            InstrumentLeg(InstrumentType.CALL, "actual", -2.0, strike=100.0),
        )
    p = market([PortfolioHolding(kind, kind, -13.0, legs, kink_policy=side)])
    assert p.response_jacobian().loc[kind, "stock"] == expected
    for price in [80.0, 100.0, 120.0]:
        z = pd.Series(
            [math.log(price / 100.0), 0.0, 0.0, 0.0],
            index=p.risk_model.factor_loadings[p.risk_date].columns,
        )
        gear = 2.0 if quantity * (price - 100.0) < 0 else 1.0
        payoff = quantity * gear * (price - 100.0)
        assert p.get_pnl(z).iloc[0] == pytest.approx(payoff, abs=1e-12)


def test_actual_quotes_sharing_proxy_keep_their_own_strikes_and_residual(market):
    """Distinct quote identities do not acquire independent copies of the same residual."""
    quotes = {
        "a": Underlying("a", 100.0, "USD", "stock", ResponseBasis.REFERENCE),
        "b": Underlying("b", 200.0, "USD", "stock", ResponseBasis.REFERENCE),
    }
    stock = holding(InstrumentType.DELTA_1, 100.0, 0.5, key="a", name="stock")
    future = holding(InstrumentType.FUTURE, 0.0, -0.25, key="b", name="future")
    p = market([stock, future], underlyings=quotes)
    np.testing.assert_allclose(p.response_jacobian().sum(), 0.0)
    dollar_risk = p.risk_model.compute_tre_at_date(
        pd.Series(0.0, index=p.response_jacobian().columns),
        p.response_jacobian().sum(),
        p.risk_date,
    )
    assert dollar_risk == 0.0
    result = p.evaluate(shocks(p))
    np.testing.assert_allclose(result.portfolio_pnl, 0.0, atol=1e-12)
    assert list(result.pnl.columns) == ["stock", "future"]


def test_reporting_denominator_does_not_resize_positions(market):
    """Absolute quantities and P&L are independent of the report's ratio denominator."""
    p = market([holding(InstrumentType.FUTURE, 0.0, -4.0)])
    second = replace(p, reporting_denominator=10.0)
    pd.testing.assert_frame_equal(p.evaluate(shocks(p)).pnl, second.evaluate(shocks(p)).pnl)


def test_funded_tiny_shock_uses_expm1(market):
    """A tiny first-order P&L must not disappear in subtracting two marks."""
    p = market([holding(InstrumentType.DELTA_1, 100.0)])
    z = pd.Series([1e-17, 0.0, 0.0, 0.0], index=p.risk_model.factor_loadings[p.risk_date].columns)
    assert p.get_pnl(z).iloc[0] == pytest.approx(1e-15, abs=1e-30)


def test_context_copies_and_custom_payoff_work_without_private_imports(market):
    """Public composites receive defensive copies and explicit shared response mappings."""

    class Basket:
        """Synthetic equal-price basket expressed only through the public protocol."""

        implementation_id = "test.basket.v1"
        coverage = "deterministic basket sum"
        boundary_policy = "smooth"

        def evaluate(self, context):
            """Return a sum of reference-currency quote prices."""
            quotes = context.quotes
            quotes.iloc[:] = 0.0  # This must not modify the shared context.
            return context.quotes.sum(axis=1)

        def response_jacobian(self, context):
            """Differentiate the baseline basket through the labelled response map."""
            return context.baseline_quotes @ context.quote_response_jacobian

    h = PortfolioHolding("basket", "Basket", 3.0, payoff=Basket())
    p = market([h])
    result = p.evaluate(shocks(p, fx=0.0))
    assert result.audit.loc["basket", "model_baseline"] == 180.0
    assert result.pnl.loc["stress", "basket"] == pytest.approx(-20.0 + 80.0 * (0.8**2 - 1.0))
    assert p.response_jacobian().loc["basket", "stock"] == 100.0


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "date",
        "future_date",
        "missing_fx",
        "missing_response",
        "missing_quote",
        "covariance_only",
        "negative_residual",
        "denominator",
    ],
)
def test_invalid_portfolio_inputs_fail(market, change):
    """No missing exposure or invalid model date is silently dropped or filled."""
    p = market([holding(InstrumentType.FUTURE, 0.0)])
    with pytest.raises((ValueError, TypeError)):
        if change == "duplicate":
            replace(p, holdings=p.holdings * 2)
        elif change == "date":
            replace(p, risk_date=pd.Timestamp("2026-08-30"))
        elif change == "future_date":
            replace(p, valuation_date=pd.Timestamp("2026-08-30"))
        elif change == "missing_fx":
            replace(p, underlyings={"actual": replace(p.underlyings["actual"], currency="EUR")})
        elif change == "missing_response":
            replace(p, underlyings={"actual": replace(p.underlyings["actual"], response_id="bad")})
        elif change == "missing_quote":
            replace(p, holdings=(holding(InstrumentType.FUTURE, 0.0, key="unknown"),))
        elif change == "covariance_only":
            replace(p, risk_model=type(p.risk_model)(p.risk_model.covar))
        elif change == "negative_residual":
            p.risk_model.residual_vars[p.risk_date].iloc[0] = -1.0
            replace(p)
        else:
            replace(p, reporting_denominator=0.0)


@pytest.mark.parametrize(
    "change", ["missing_factor", "duplicate_factor", "duplicate_scenario", "nan", "overflow"]
)
def test_invalid_scenario_batches_fail(market, change):
    """Unknown shapes and nonfinite quotes fail instead of generating incomplete reports."""
    p = market([holding(InstrumentType.CALL, -3.0, strike=100.0)])
    z = shocks(p)
    if change == "missing_factor":
        z = z.iloc[:, :-1]
    elif change == "duplicate_factor":
        z.columns = ["Equity"] * 4
    elif change == "duplicate_scenario":
        z.index = ["one", "one"]
    elif change == "nan":
        z.iloc[0, 0] = np.nan
    else:
        z.iloc[0, 0] = 1000.0
    with pytest.raises(ValueError):
        p.evaluate(z)


def test_funded_baseline_is_exact_even_with_nonbinary_fx_rate(market):
    """A funded mark is authoritative, with no floating-point intrinsic basis offset."""
    p = market([holding(InstrumentType.DELTA_1, 2_000_000.)], currency="EUR")
    p = replace(p, fx_rates={"EUR": replace(p.fx_rates["EUR"], spot0=1.15)})
    result = p.evaluate(shocks(p))
    assert result.audit.loc["holding", "model_baseline"] == 2_000_000.
    assert result.audit.loc["holding", "basis_offset"] == 0.
