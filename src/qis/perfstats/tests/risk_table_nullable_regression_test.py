"""Verify benchmark regressions across ordinary and nullable price storage.

The mixed panel is the public path that previously passed a nullable benchmark Series into
statsmodels beside an ordinary asset. Returns are constructed from a known linear relation so the
expected regression statistics do not depend on another QIS reduction.
"""

import warnings
from numbers import Real
from typing import cast

import numpy as np
import pandas as pd

from qis.perfstats.config import PerfParams, PerfStat
from qis.perfstats.perf_stats import (
    compute_ra_perf_table_with_benchmark,  # pyright: ignore[reportUnknownVariableType]
)


_FREQUENCY = "ME"
_ANNUALIZATION_FACTOR = 12.0


def _stat(table: pd.DataFrame, asset: str, perf_stat: PerfStat) -> float:
    """Extract one real-valued statistic using its public table label."""
    label = perf_stat.value.name
    if not isinstance(label, str):
        raise TypeError("expected a string performance-statistic label")
    value = cast(object, table.loc[asset, label])
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = cast(object, value.item())
    if not isinstance(value, Real):
        raise TypeError("expected a real performance statistic")
    return float(value)


def test_benchmark_table_normalizes_a_nullable_benchmark_regressor() -> None:
    """Report the known mixed-storage alpha, beta, and R-squared without warnings."""
    dates = pd.date_range("2020-01-31", periods=13, freq=_FREQUENCY)
    benchmark_returns = np.array(
        [-0.03, 0.01, 0.02, -0.01, 0.04, 0.00, -0.02, 0.03, 0.01, -0.01, 0.02, 0.04],
        dtype=np.float64,
    )
    asset_returns = 0.002 + 1.5 * benchmark_returns
    prices = pd.DataFrame(
        {
            "Benchmark": 100.0 * np.cumprod(np.r_[1.0, 1.0 + benchmark_returns]),
            "Asset": 80.0 * np.cumprod(np.r_[1.0, 1.0 + asset_returns]),
        },
        index=dates,
    )
    prices["Benchmark"] = prices["Benchmark"].astype(pd.Float64Dtype())
    prices_before = prices.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = compute_ra_perf_table_with_benchmark(
            prices=prices,
            benchmark="Benchmark",
            perf_params=PerfParams(
                freq_vol=_FREQUENCY,
                freq_drawdown=_FREQUENCY,
                freq_skewness=_FREQUENCY,
                freq_reg=_FREQUENCY,
            ),
            is_log_returns=False,
        )

    np.testing.assert_allclose(
        [
            _stat(table, "Asset", PerfStat.ALPHA_AN),
            _stat(table, "Asset", PerfStat.BETA),
            _stat(table, "Asset", PerfStat.R2),
        ],
        [_ANNUALIZATION_FACTOR * 0.002, 1.5, 1.0],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert 0.0 <= _stat(table, "Asset", PerfStat.ALPHA_PVALUE) <= 1.0
    pd.testing.assert_frame_equal(prices, prices_before)
