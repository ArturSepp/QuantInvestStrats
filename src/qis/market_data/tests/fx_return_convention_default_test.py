"""Arithmetic defaults and explicit return conventions for FX-adjusted panels."""

from inspect import signature

import numpy as np
import pandas as pd
import pytest

import qis
from qis.market_data import FxRatesData


def _inputs(frequency_mode: str, excess: bool, retain_zeros: bool):
    """Return synthetic funded assets with unequal rates and mixed currency exposures."""
    dates = pd.date_range("2024-01-31", periods=6, freq="ME")
    prices = pd.DataFrame(
        {
            "USD_UNHEDGED": [100.0, 103.0, 101.0, 107.0, 109.0, 104.0],
            "USD_HALF": [100.0, 105.0, 110.0, 108.0, 106.0, 112.0],
            "CHF_NATIVE": [100.0, 104.0, 104.0, 95.0, 105.0, 98.0],
        },
        index=dates,
    )
    data = FxRatesData(
        fx_spots=pd.DataFrame(
            {"USD": 1.0, "CHF": [1.1, 1.0, 1.2, 1.05, 1.15, 1.1]}, index=dates),
        domestic_rates=pd.DataFrame(
            {"USD": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
             "CHF": [0.01, 0.02, 0.015, 0.025, 0.02, 0.03]}, index=dates),
    )
    frequencies = {
        "monthly": "ME",
        "quarterly": "QE",
        "mixed": pd.Series(
            {"USD_UNHEDGED": "ME", "USD_HALF": "ME", "CHF_NATIVE": "QE"}),
    }
    options = dict(
        prices=prices,
        hedge_ratios=pd.Series({"USD_UNHEDGED": 0.0, "USD_HALF": 0.5, "CHF_NATIVE": 1.0}),
        local_ccys=pd.Series({"USD_UNHEDGED": "USD", "USD_HALF": "USD", "CHF_NATIVE": "CHF"}),
        reference_ccy="CHF",
        freq=frequencies[frequency_mode],
        is_excess_returns=excess,
        zero_return_to_nan=not retain_zeros,
    )
    return data, options


def test_fx_panel_default_matches_to_returns_default() -> None:
    """Both public return helpers default to arithmetic, not logarithmic, output."""
    assert signature(qis.to_returns).parameters["is_log_returns"].default is False
    assert signature(FxRatesData.compute_fx_adjusted_returns).parameters[
        "is_log_returns"].default is False


@pytest.mark.parametrize("frequency_mode", ["monthly", "quarterly", "mixed"])
@pytest.mark.parametrize("excess", [False, True])
@pytest.mark.parametrize("retain_zeros", [False, True])
def test_fx_panel_default_is_explicit_arithmetic(
        frequency_mode: str, excess: bool, retain_zeros: bool) -> None:
    """Default output equals explicit simple output for every dispatch and cash policy."""
    data, options = _inputs(frequency_mode, excess, retain_zeros)
    actual = data.compute_fx_adjusted_returns(**options)
    arithmetic = data.compute_fx_adjusted_returns(**options, is_log_returns=False)
    logarithmic = data.compute_fx_adjusted_returns(**options, is_log_returns=True)

    assert actual.keys() == arithmetic.keys() == logarithmic.keys()
    for frequency in actual:
        pd.testing.assert_frame_equal(actual[frequency], arithmetic[frequency])
        simple = arithmetic[frequency].to_numpy()
        log = logarithmic[frequency].to_numpy()
        paired = np.isfinite(simple) & np.isfinite(log)
        assert paired.any()
        assert not np.allclose(simple[paired], log[paired])
        if not excess:
            np.testing.assert_allclose(
                simple, np.expm1(log), rtol=1e-13, atol=1e-14, equal_nan=True)

