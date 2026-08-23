"""Assurance tests for management and performance fee return adjustments.

These deterministic tests document established behavior in
``compute_net_return_ex_perf_man_fees`` rather than correcting a production defect. The initial
observation establishes NAV at 100 and therefore has a conventional zero net return. Subsequent
management fees use actual elapsed calendar days divided by 365, while performance fees apply
only to gains above the previously crystallized high-water mark.

Expected values are calculated directly from the fee equations and supplied as literals. No QIS
NAV or return-conversion function is used to construct an expected result.
"""

import pandas as pd

from qis.perfstats.returns import compute_net_return_ex_perf_man_fees


# =============================================================================
# Shared deterministic timeline and comparison helper
# =============================================================================

_MONTH_END_DATES = pd.date_range("2024-01-31", periods=4, freq="ME")
_FUND_NAME = "Fund"
_TOLERANCE = 1.0e-12


def _assert_result_and_input_ownership(
        actual: pd.Series,
        expected: pd.Series,
        supplied: pd.Series,
        original: pd.Series,
) -> None:
    """Compare a result and prove that the caller-owned input was not modified.

    Args:
        actual: Net returns produced by the public function.
        expected: Independently calculated net returns.
        supplied: Gross-return Series passed to the public function.
        original: Deep copy of the gross returns made before the call.
    """
    pd.testing.assert_series_equal(
        actual,
        expected,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_series_equal(supplied, original, check_exact=True)
    assert actual is not supplied


# =============================================================================
# Zero-fee initialization convention
# =============================================================================

def test_compute_net_return_ex_perf_man_fees_zero_fees_preserve_returns() -> None:
    """Preserve observed gross returns after the conventional initial zero.

    The deliberately nonzero first gross return proves that the first observation initializes
    NAV rather than representing an earned return. With both fee rates set to zero, every later
    net return must equal the corresponding gross return exactly, including the Series name.
    """
    gross_returns = pd.Series(
        [0.25, 0.10, -0.10, 0.15],
        index=_MONTH_END_DATES,
        name=_FUND_NAME,
    )
    original = gross_returns.copy(deep=True)
    expected = pd.Series(
        [0.00, 0.10, -0.10, 0.15],
        index=_MONTH_END_DATES,
        name=_FUND_NAME,
    )

    actual = compute_net_return_ex_perf_man_fees(
        gross_return=gross_returns,
        man_fee=0.0,
        perf_fee=0.0,
        perf_fee_frequency="ME",
    )

    _assert_result_and_input_ownership(actual, expected, gross_returns, original)


# =============================================================================
# Actual/365 management-fee accrual
# =============================================================================

def test_compute_net_return_ex_perf_man_fees_accrues_management_fee_by_elapsed_days() -> None:
    """Deduct management fees using the calendar days in each return interval.

    An annual fee of 36.5% produces an exact 1% charge over ten days and an exact 3% charge over
    thirty days under the actual/365 convention. With performance fees disabled, gross returns
    of 2% and -1% therefore become independently calculated net returns of 1% and -4%.
    """
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-11", "2024-02-10"])
    gross_returns = pd.Series(
        [0.00, 0.02, -0.01],
        index=dates,
        name=_FUND_NAME,
    )
    original = gross_returns.copy(deep=True)
    expected = pd.Series(
        [0.00, 0.01, -0.04],
        index=dates,
        name=_FUND_NAME,
    )

    actual = compute_net_return_ex_perf_man_fees(
        gross_return=gross_returns,
        man_fee=0.365,
        perf_fee=0.0,
        perf_fee_frequency="YE",
    )

    _assert_result_and_input_ownership(actual, expected, gross_returns, original)


# =============================================================================
# Performance-fee crystallization and high-water mark
# =============================================================================

def test_compute_net_return_ex_perf_man_fees_crystallizes_only_profits_above_hwm() -> None:
    """Charge performance fees only on gains above the crystallized high-water mark.

    A 10% gain takes GAV from 100 to 110; a 20% fee leaves NAV and the new HWM at 108, producing
    an 8% net return. A subsequent 10% loss takes both GAV and NAV to 97.2 without a fee. The
    final 15% gain takes GAV to 111.78, but only 3.78 exceeds the HWM. The 0.756 fee leaves NAV
    at 111.024, which is a 14.222222% return from 97.2.
    """
    gross_returns = pd.Series(
        [0.00, 0.10, -0.10, 0.15],
        index=_MONTH_END_DATES,
        name=_FUND_NAME,
    )
    original = gross_returns.copy(deep=True)
    expected = pd.Series(
        [0.00, 0.08, -0.10, 0.14222222222222225],
        index=_MONTH_END_DATES,
        name=_FUND_NAME,
    )

    actual = compute_net_return_ex_perf_man_fees(
        gross_return=gross_returns,
        man_fee=0.0,
        perf_fee=0.20,
        perf_fee_frequency="ME",
    )

    _assert_result_and_input_ownership(actual, expected, gross_returns, original)
