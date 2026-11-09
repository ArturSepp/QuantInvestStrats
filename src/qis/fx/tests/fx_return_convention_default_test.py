"""Arithmetic defaults and explicit return conventions for FX-adjusted panels."""

from dataclasses import replace
from inspect import signature

import numpy as np
import pandas as pd
import pytest

import qis
from qis.fx import FxRatesData


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
        fx_spots=pd.DataFrame({"USD": 1.0, "CHF": [1.1, 1.0, 1.2, 1.05, 1.15, 1.1]}, index=dates),
        domestic_rates=pd.DataFrame(
            {
                "USD": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                "CHF": [0.01, 0.02, 0.015, 0.025, 0.02, 0.03],
            },
            index=dates,
        ),
    )
    frequencies = {
        "monthly": "ME",
        "quarterly": "QE",
        "mixed": pd.Series({"USD_UNHEDGED": "ME", "USD_HALF": "ME", "CHF_NATIVE": "QE"}),
    }
    options = dict(
        prices=prices,
        hedge_ratios=pd.Series({"USD_UNHEDGED": 0.0, "USD_HALF": 0.5, "CHF_NATIVE": 1.0}),
        local_ccys=pd.Series({"USD_UNHEDGED": "USD", "USD_HALF": "USD", "CHF_NATIVE": "CHF"}),
        reference_ccy="CHF",
        frequency=frequencies[frequency_mode],
        return_type=(
            qis.FxReturnType.REFERENCE_EXCESS if excess else qis.FxReturnType.REFERENCE_TOTAL
        ),
        zero_return_to_nan=not retain_zeros,
    )
    return data, options


def test_fx_asset_coordinate_is_required() -> None:
    """The lean typed API requires an explicit coordinate instead of guessing a convention."""
    assert signature(qis.to_returns).parameters["is_log_returns"].default is False
    assert (
        signature(qis.FxAssetReturnsRequest).parameters["coordinate"].default
        is signature(qis.FxAssetReturnsRequest).empty
    )
    data, options = _inputs("monthly", False, True)
    with pytest.raises(TypeError, match="coordinate"):
        qis.FxAssetReturnsRequest(**options)


@pytest.mark.parametrize("frequency_mode", ["monthly", "quarterly", "mixed"])
@pytest.mark.parametrize("excess", [False, True])
@pytest.mark.parametrize("retain_zeros", [False, True])
def test_fx_panel_coordinates_are_consistent(
    frequency_mode: str, excess: bool, retain_zeros: bool
) -> None:
    """Explicit simple and log coordinates retain each dispatch and cash policy."""
    data, options = _inputs(frequency_mode, excess, retain_zeros)
    request = qis.FxAssetReturnsRequest(**options, coordinate=qis.FxReturnCoordinate.SIMPLE)
    arithmetic = qis.compute_fx_returns(data, request)
    logarithmic = qis.compute_fx_returns(
        data, replace(request, coordinate=qis.FxReturnCoordinate.LOG)
    )

    assert arithmetic.keys() == logarithmic.keys()
    for frequency in arithmetic:
        simple = arithmetic[frequency].values.to_numpy()
        log = logarithmic[frequency].values.to_numpy()
        paired = np.isfinite(simple) & np.isfinite(log)
        assert paired.any()
        assert not np.allclose(simple[paired], log[paired])
        if not excess:
            np.testing.assert_allclose(
                simple, np.expm1(log), rtol=1e-13, atol=1e-14, equal_nan=True
            )
