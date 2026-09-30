"""Matrix-path assurances for management and performance fee NAVs.

The multi-asset helper must update independent fee accounts together without dispatching every
column through the scalar public path. Mixed starts and missing values exercise separate state,
while an unanchored frequency proves that each ragged history keeps its own crystallization grid.
"""

import numpy as np
import pandas as pd
import pytest

import qis.perfstats.returns as returns_module


_FEE_DATES = pd.DatetimeIndex(
    ["2022-12-29", "2022-12-30", "2023-01-03", "2023-12-29", "2024-01-02"],
    name="Date",
)


def _mixed_navs(dtype: str = "float64") -> pd.DataFrame:
    """Gross NAVs spanning complete, ragged, gapped, and all-missing fee accounts."""
    return pd.DataFrame(
        {
            "Complete": [100.0, 110.0, 121.0, 133.1, 146.41],
            "Late": [np.nan, np.nan, 50.0, 55.0, 60.5],
            "Boundary gap": [100.0, np.nan, 110.0, 121.0, 133.1],
            "All missing": [np.nan] * 5,
        },
        index=_FEE_DATES,
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", ("float64", "Float64"))
def test_compute_net_navs_ex_perf_man_fees_updates_mixed_accounts_together(
    monkeypatch: pytest.MonkeyPatch,
    dtype: str,
) -> None:
    """Use one matrix path while preserving literal independent fee-account results.

    The expected columns follow the fee equations directly. The boundary gap is a flat return
    on the first crystallization date, and the late and all-missing columns must not borrow state
    from their complete neighbor.

    Args:
        monkeypatch: Pytest fixture used to forbid scalar public-path dispatch.
        dtype: Ordinary or nullable floating-point input representation.
    """
    navs = _mixed_navs(dtype=dtype)
    original = navs.copy(deep=True)
    expected = pd.DataFrame(
        {
            "Complete": [1.0, 1.08, 1.1664, 1.26144, 1.3623552],
            "Late": [np.nan, np.nan, 1.0, 1.08, 1.1664],
            "Boundary gap": [1.0, 1.0, 1.08, 1.168, 1.26144],
            "All missing": [np.nan] * 5,
        },
        index=_FEE_DATES,
    )

    def _reject_scalar_dispatch(*args: object, **kwargs: object) -> pd.Series:
        raise AssertionError("multi-asset fee calculation dispatched through the scalar path")

    monkeypatch.setattr(
        returns_module,
        "compute_net_return_ex_perf_man_fees",
        _reject_scalar_dispatch,
    )
    actual = returns_module.compute_net_navs_ex_perf_man_fees(
        navs,
        man_fee=0.0,
        perf_fee=0.2,
        perf_fee_frequency="YE",
    )

    pd.testing.assert_frame_equal(actual, expected, rtol=0.0, atol=1.0e-14)
    pd.testing.assert_frame_equal(navs, original, check_exact=True)


def test_compute_net_navs_ex_perf_man_fees_preserves_ragged_unanchored_schedules() -> None:
    """Match exact per-column results when a relative frequency depends on each start date."""
    dates = pd.date_range("2024-01-01", periods=12, freq="D", name="Date")
    navs = pd.DataFrame(
        {
            "Early": 100.0 * 1.05 ** np.arange(12),
            "Late": np.r_[[np.nan, np.nan], 50.0 * 1.05 ** np.arange(10)],
        },
        index=dates,
    )

    actual = returns_module.compute_net_navs_ex_perf_man_fees(
        navs,
        man_fee=0.0,
        perf_fee=0.2,
        perf_fee_frequency="7D",
    )
    expected = pd.concat(
        [
            returns_module.compute_net_navs_ex_perf_man_fees(
                navs[column].dropna(),
                man_fee=0.0,
                perf_fee=0.2,
                perf_fee_frequency="7D",
            ).reindex(dates)
            for column in navs.columns
        ],
        axis=1,
    )

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_compute_net_navs_ex_perf_man_fees_rejects_duplicate_columns() -> None:
    """Keep duplicate labels from ambiguously sharing or duplicating fee-account state."""
    navs = _mixed_navs().iloc[:, :2]
    navs.columns = ["Fund", "Fund"]

    with pytest.raises(ValueError):
        returns_module.compute_net_navs_ex_perf_man_fees(navs)
