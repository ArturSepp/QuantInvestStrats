"""Regression tests for missing gross NAVs in fee-adjusted NAV calculations.

Expected values are derived directly from the historical forward-fill return convention and
zero-fee compounding. The tests cover both public container forms and prove that caller-owned
inputs remain unchanged.
"""

import numpy as np
import pandas as pd

from qis.perfstats.returns import compute_net_navs_ex_perf_man_fees


_DATES = pd.date_range("2024-01-31", periods=4, freq="ME")


def test_compute_net_navs_ex_perf_man_fees_forward_fills_series_gaps() -> None:
    """Treat an interior missing gross NAV as an unchanged price before fee calculation."""
    navs = pd.Series(
        pd.array([100.0, pd.NA, 121.0, 133.1], dtype="Float64"),
        index=_DATES,
        name="Fund",
    )
    original = navs.copy(deep=True)
    expected = pd.Series([1.0, 1.0, 1.21, 1.331], index=_DATES, name="Fund")

    actual = compute_net_navs_ex_perf_man_fees(navs, man_fee=0.0, perf_fee=0.0)

    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(navs, original, check_exact=True)


def test_compute_net_navs_ex_perf_man_fees_forward_fills_mixed_frame_gaps() -> None:
    """Apply the same explicit gap policy independently to complete and gapped columns."""
    navs = pd.DataFrame(
        {
            "Gapped": [100.0, np.nan, 121.0, np.nan],
            "Complete": [100.0, 110.0, 121.0, 133.1],
        },
        index=_DATES,
    )
    original = navs.copy(deep=True)
    expected = pd.DataFrame(
        {
            "Gapped": [1.0, 1.0, 1.21, 1.21],
            "Complete": [1.0, 1.1, 1.21, 1.331],
        },
        index=_DATES,
    )

    actual = compute_net_navs_ex_perf_man_fees(navs, man_fee=0.0, perf_fee=0.0)

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(navs, original, check_exact=True)
