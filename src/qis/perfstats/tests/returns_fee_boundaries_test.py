"""Assurance tests for management and performance fee return adjustments.

These deterministic tests document and protect the fee boundaries in
``compute_net_return_ex_perf_man_fees``. The initial observation establishes NAV at 100 and
therefore has a conventional zero net return. Subsequent management fees use actual elapsed
calendar days divided by 365, while performance fees apply only to gains above the previously
crystallized high-water mark. Calendar crystallization boundaries use the latest observation on
or before the boundary so ordinary weekend and holiday period ends are not skipped. Unique return
dates are processed chronologically regardless of physical row order, while duplicate dates are
rejected because path-dependent fee state would otherwise be ambiguous.

Expected values are calculated directly from the fee equations and supplied as literals. No QIS
NAV or return-conversion function is used to construct an expected result.
"""

import warnings

import pandas as pd
import pytest

from qis.perfstats.returns import compute_net_return_ex_perf_man_fees


# =============================================================================
# Shared deterministic timeline and comparison helper
# =============================================================================

_MONTH_END_DATES = pd.date_range("2024-01-31", periods=4, freq="ME")
_FUND_NAME = "Fund"
_DUPLICATE_DATE_ERROR = r"gross_return index must not contain duplicate dates"
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
# Return-index chronology and uniqueness
# =============================================================================

@pytest.mark.parametrize(
    "order",
    ((0, 1, 2, 3), (3, 2, 1, 0), (0, 2, 1, 3)),
    ids=("sorted", "reversed", "interior-shuffled"),
)
def test_compute_net_return_ex_perf_man_fees_normalizes_unique_return_dates(
        order: tuple[int, ...],
) -> None:
    """Apply fee state chronologically regardless of physical Series row order.

    The December 31 year-end is off-grid and therefore maps to December 30. A 36.5%
    management fee produces exact daily accrual of 0.1%, while a 20% performance fee
    crystallizes the first gain. The literal result protects the interaction among
    chronological ordering, actual/365 accrual, and the on-or-before boundary rule.

    Args:
        order: Physical row permutation applied to the same date-to-return mapping.
    """
    dates = pd.DatetimeIndex(
        ["2022-12-29", "2022-12-30", "2023-01-03", "2023-01-10"],
        name="Date",
    )
    values = (0.00, 0.10, 0.10, -0.05)
    gross_returns = pd.Series(
        [values[position] for position in order],
        index=dates[list(order)],
        name=_FUND_NAME,
    )
    original = gross_returns.copy()
    expected = pd.Series(
        [0.00, 0.07920, 0.07680, -0.046413075780089153],
        index=dates,
        name=_FUND_NAME,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = compute_net_return_ex_perf_man_fees(
            gross_return=gross_returns,
            man_fee=0.365,
            perf_fee=0.20,
            perf_fee_frequency="YE",
        )

    _assert_result_and_input_ownership(actual, expected, gross_returns, original)


def test_compute_net_return_ex_perf_man_fees_rejects_duplicate_return_dates() -> None:
    """Reject ambiguous repeated observations before calculating path-dependent fee state."""
    gross_returns = pd.Series(
        [0.00, 0.10, 0.20],
        index=pd.DatetimeIndex(
            ["2024-01-01", "2024-01-02", "2024-01-02"],
            name="Date",
        ),
        name=_FUND_NAME,
    )
    original = gross_returns.copy()

    with pytest.raises(ValueError, match=_DUPLICATE_DATE_ERROR):
        compute_net_return_ex_perf_man_fees(
            gross_return=gross_returns,
            man_fee=0.0,
            perf_fee=0.20,
            perf_fee_frequency="YE",
        )

    pd.testing.assert_series_equal(gross_returns, original, check_exact=True)


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


@pytest.mark.parametrize(
    ("dates", "frequency"),
    [
        (["2022-12-29", "2022-12-30", "2023-01-03"], "YE"),
        (["2024-05-29", "2024-05-30", "2024-06-03"], "ME"),
        (["2024-12-30", "2024-12-31", "2025-01-02"], "YE"),
    ],
    ids=["weekend-year-end", "holiday-month-end", "exact-year-end"],
)
def test_compute_net_return_ex_perf_man_fees_uses_latest_period_end_observation(
        dates: list[str],
        frequency: str,
) -> None:
    """Crystallize on the latest observation at or before each calendar boundary.

    The weekend and synthetic-holiday cases omit their calendar period-end observations, while
    the exact control includes December 31. In every case the 10% pre-boundary gain takes GAV to
    110, crystallizes a fee of 2, and carries NAV/GAV/HWM of 108 forward. The following 10% gain
    then produces GAV 118.8, fee 2.16, and NAV 116.64: an exact 8% net return from 108.

    Args:
        dates: Sorted observation dates surrounding the crystallization boundary.
        frequency: Calendar frequency used to generate that boundary.
    """
    gross_returns = pd.Series(
        [0.00, 0.10, 0.10],
        index=pd.DatetimeIndex(dates),
        name=_FUND_NAME,
    )
    original = gross_returns.copy(deep=True)
    # Literal expectations come from the GAV/NAV/HWM calculation above, not a QIS helper.
    expected = pd.Series(
        [0.00, 0.08, 0.08],
        index=pd.DatetimeIndex(dates),
        name=_FUND_NAME,
    )

    actual = compute_net_return_ex_perf_man_fees(
        gross_return=gross_returns,
        man_fee=0.0,
        perf_fee=0.20,
        perf_fee_frequency=frequency,
    )

    _assert_result_and_input_ownership(actual, expected, gross_returns, original)


def test_compute_net_return_ex_perf_man_fees_maps_multiple_off_grid_year_ends() -> None:
    """Crystallize independently at consecutive off-grid calendar year ends.

    December 31 falls on a weekend in both 2022 and 2023, so the corresponding December 30 and
    December 29 observations must close their respective fee periods. The first crystallization
    carries NAV/GAV/HWM 108 into 2023. The second carries 126.144 into 2024, making the returns
    immediately after both boundaries exactly 8% rather than retaining an uncrystallized fee
    liability from the prior period.
    """
    dates = pd.DatetimeIndex(
        ["2022-12-29", "2022-12-30", "2023-01-03", "2023-12-29", "2024-01-02"]
    )
    gross_returns = pd.Series(
        [0.00, 0.10, 0.10, 0.10, 0.10],
        index=dates,
        name=_FUND_NAME,
    )
    original = gross_returns.copy(deep=True)
    # The 2023 year-end return is 126.144 / 116.64 - 1; the next period restarts at 8%.
    expected = pd.Series(
        [0.00, 0.08, 0.08, 0.08148148148148149, 0.08],
        index=dates,
        name=_FUND_NAME,
    )

    actual = compute_net_return_ex_perf_man_fees(
        gross_return=gross_returns,
        man_fee=0.0,
        perf_fee=0.20,
        perf_fee_frequency="YE",
    )

    _assert_result_and_input_ownership(actual, expected, gross_returns, original)
