"""Independent scalar references for signed primitives and validation."""

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


@pytest.mark.parametrize("kind", list(InstrumentType))
@pytest.mark.parametrize("quantity", [-3.0, 2.0])
def test_signed_intrinsic_values_and_quote_derivatives(kind, quantity):
    """Evaluate scalars by their economic payoff, including the futures multiplier."""
    strike = 100.0 if kind in (InstrumentType.CALL, InstrumentType.PUT) else None
    leg = InstrumentLeg(kind, "quote", quantity, 5.0, strike)
    prices = [70.0, 100.0, 130.0]
    expected = []
    for price in prices:
        if kind is InstrumentType.CALL:
            payoff = max(price - 100.0, 0.0)
        elif kind is InstrumentType.PUT:
            payoff = max(100.0 - price, 0.0)
        elif kind is InstrumentType.FUTURE:
            payoff = price - 90.0
        else:
            payoff = price
        expected.append(quantity * 5.0 * payoff)
    np.testing.assert_allclose(leg.get_payoff(pd.Series(prices), 90.0), expected)
    for price in [70.0, 130.0]:
        epsilon = 1e-4
        bumped = leg.get_payoff(pd.Series([price - epsilon, price + epsilon]), 90.0)
        finite_difference = (bumped.iloc[1] - bumped.iloc[0]) / (2 * epsilon)
        assert leg.get_quote_delta(price, KinkPolicy.MIDPOINT) == pytest.approx(finite_difference)


@pytest.mark.parametrize(
    "side,call_delta,put_delta",
    [(KinkPolicy.LEFT, 0.0, -1.0), (KinkPolicy.RIGHT, 1.0, 0.0), (KinkPolicy.MIDPOINT, 0.5, -0.5)],
)
def test_strike_policy_is_explicit(side, call_delta, put_delta):
    """The same one-sided policy can be applied across all synthetic legs."""
    call = InstrumentLeg(InstrumentType.CALL, "q", 1.0, strike=100.0)
    put = InstrumentLeg(InstrumentType.PUT, "q", 1.0, strike=100.0)
    assert call.get_quote_delta(100.0, side) == call_delta
    assert put.get_quote_delta(100.0, side) == put_delta


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(instrument_type="call", underlying_id="q", quantity=1.0, strike=100.0),
        dict(instrument_type=InstrumentType.CALL, underlying_id="q", quantity=1.0),
        dict(instrument_type=InstrumentType.PUT, underlying_id="q", quantity=1.0, strike=-1.0),
        dict(instrument_type=InstrumentType.FUTURE, underlying_id="q", quantity=1.0, strike=100.0),
        dict(instrument_type=InstrumentType.FUTURE, underlying_id="q", quantity=np.nan),
        dict(
            instrument_type=InstrumentType.FUTURE, underlying_id="q", quantity=1.0, multiplier=0.0
        ),
    ],
)
def test_invalid_primitive_terms_fail(kwargs):
    """Invalid types, units and strikes fail before reporting."""
    with pytest.raises(ValueError):
        InstrumentLeg(**kwargs)


@pytest.mark.parametrize("spot", [0.0, -1.0, np.nan, np.inf])
def test_nonpositive_or_missing_quotes_fail(spot):
    """The multiplicative quote policy does not invent log returns for negative futures."""
    with pytest.raises(ValueError, match="spot0"):
        Underlying("q", spot, "USD", "proxy", ResponseBasis.LOCAL)
