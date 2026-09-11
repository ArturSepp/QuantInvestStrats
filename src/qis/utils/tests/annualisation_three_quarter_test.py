"""Regression tests for three-quarter annualization factors.

An observation spanning three quarters represents nine months, so four quarterly periods per year
must be divided by three quarters per observation. Explicit aliases, equivalent generic forms,
frequency conversion, and public regime outputs must all use that same four-thirds factor.
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest

import qis


_QUARTERS_PER_YEAR = 4
_QUARTERS_PER_OBSERVATION = 3
_THREE_QUARTER_FACTOR = _QUARTERS_PER_YEAR / _QUARTERS_PER_OBSERVATION

_EXPLICIT_ALIASES = ("3Q", "3QE", "3BQ", "3QS", "3BQS")
_EQUIVALENT_GENERIC_FORMS = (
    ("3Q", "3q"),
    ("3QE", "3QE-DEC"),
    ("3BQ", "3bq"),
    ("3QS", "3QS-JAN"),
    ("3BQS", "3BQS-JAN"),
)


@pytest.mark.parametrize("frequency", _EXPLICIT_ALIASES)
def test_get_annualization_factor_three_quarter_aliases_divide_quarterly_basis(
    frequency: str,
) -> None:
    """Return four-thirds observations per year for each explicit alias.

    Args:
        frequency: Existing explicit alias for one observation every three quarters.
    """
    assert qis.get_annualization_factor(frequency) == _THREE_QUARTER_FACTOR


@pytest.mark.parametrize(("explicit", "generic"), _EQUIVALENT_GENERIC_FORMS)
def test_get_annualization_factor_three_quarter_aliases_match_generic_forms(
    explicit: str,
    generic: str,
) -> None:
    """Keep explicit aliases consistent with the existing generic multiplier path.

    Args:
        explicit: Alias handled by the dedicated quarterly branch.
        generic: Equivalent alias handled by the generic parser.
    """
    generic_factor = qis.get_annualization_factor(generic)

    assert generic_factor == _THREE_QUARTER_FACTOR
    assert qis.get_annualization_factor(explicit) == generic_factor


def test_get_annualisation_conversion_factor_uses_three_quarter_factor() -> None:
    """Convert three-quarter observations to quarterly units by the period-count ratio."""
    assert qis.get_annualisation_conversion_factor("3QE", "QE") == 1 / 3
    assert qis.get_annualisation_conversion_factor("QE", "3QE") == 3.0


@pytest.mark.parametrize("is_report_pa_returns", (False, True))
def test_compute_regime_avg_uses_three_quarter_factor(
    is_report_pa_returns: bool,
) -> None:
    """Scale public regime contributions with the independent four-thirds factor.

    Args:
        is_report_pa_returns: Apply geometric or additive annualized contribution output.
    """
    sampled_returns = pd.DataFrame(
        {
            "Asset": (0.01, 0.03, -0.02, 0.02),
            "regime": ("Up", "Up", "Down", "Down"),
        }
    )
    original = sampled_returns.copy(deep=True)
    regime_weight = 0.5
    expected_up_contribution = 0.02 * _THREE_QUARTER_FACTOR * regime_weight
    if is_report_pa_returns:
        expected_up_contribution = np.expm1(expected_up_contribution)

    _, actual, frequencies = qis.compute_regime_avg(
        sampled_returns,
        freq="3QE",
        is_report_pa_returns=is_report_pa_returns,
        regime_ids=["Up", "Down"],
    )

    actual_asset = cast(pd.Series, actual.loc["Asset"])
    np.testing.assert_allclose(
        actual_asset.to_numpy(dtype=float),
        (expected_up_contribution, 0.0),
        rtol=1e-14,
        atol=0.0,
    )
    pd.testing.assert_series_equal(
        frequencies.reindex(["Up", "Down"]),
        pd.Series((regime_weight, regime_weight), index=pd.Index(["Up", "Down"], name="regime")),
    )
    pd.testing.assert_frame_equal(sampled_returns, original, check_exact=True)
