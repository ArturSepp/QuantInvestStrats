"""Regression coverage for benchmark means in regime performance tables.

``is_use_benchmark_means`` replaces the benchmark's per-annum regime values with its conditional
periodic means, for display. The average and per-annum tables use different display suffixes, so
the values must follow the common explicit regime order rather than align on those display
labels. The regime Sharpe values are computed before the substitution.

The deterministic quarterly panel reverses the natural regime order and retains an unobserved
middle regime. Expected means and Sharpe values are calculated directly from literal returns.
Ordinary and nullable inputs, all Sharpe conventions, exact labels, missing placement, the
disabled-flag control, and caller ownership protect the complete assignment boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.config import PerfParams, RegimeData, ReturnTypes, SharpeConvention
from qis.perfstats.regime_classifier import (
    compute_regimes_pa_perf_table_from_sampled_returns,
)


_BENCHMARK = "Benchmark"
_ASSET = "Asset"
_REGIME_ORDER = ("Up", "Flat", "Down")
_BENCHMARK_RETURNS = np.array((-0.01, -0.02, 0.01, 0.03))
_ASSET_RETURNS = np.array((-0.04, 0.02, 0.05, 0.07))
_TOLERANCE = 1.0e-12


def _inputs(dtype: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create classified quarterly returns and their corresponding price panel.

    Args:
        dtype: Pandas storage dtype for the sampled numerical returns.

    Returns:
        Classified returns and prices derived directly from the literal returns.
    """
    price_dates = pd.date_range("2023-12-31", periods=5, freq="QE", name="Date")
    return_dates = price_dates[1:]
    regimes = pd.Categorical(
        ("Down", "Down", "Up", "Up"),
        categories=_REGIME_ORDER,
        ordered=True,
    )
    sampled_returns = pd.DataFrame(
        {
            _BENCHMARK: pd.array(_BENCHMARK_RETURNS, dtype=dtype),
            _ASSET: pd.array(_ASSET_RETURNS, dtype=dtype),
            "regime": regimes,
        },
        index=return_dates,
    )
    prices = pd.DataFrame(
        {
            _BENCHMARK: 100.0 * np.concatenate(((1.0,), np.cumprod(1.0 + _BENCHMARK_RETURNS))),
            _ASSET: 80.0 * np.concatenate(((1.0,), np.cumprod(1.0 + _ASSET_RETURNS))),
        },
        index=price_dates,
    )
    return sampled_returns, prices


def _expected_sharpe(convention: SharpeConvention) -> np.ndarray:
    """Calculate the ordered benchmark regime Sharpe values independently.

    Args:
        convention: Regime Sharpe convention selected by ``PerfParams``.

    Returns:
        Expected Up, Flat, and Down regime Sharpe values.
    """
    up = _BENCHMARK_RETURNS[2:]
    down = _BENCHMARK_RETURNS[:2]
    if convention is SharpeConvention.PA:
        # Expected value changed in the W1a fixes: the PA regime Sharpe divides the patched
        # per-annum contributions by VOL, and the display substitution of the benchmark's
        # periodic means (is_use_benchmark_means) no longer enters it. The previous expectation,
        # periodic means over an annualised volatility, pinned that defect.
        annualized_vol = np.sqrt(4.0) * np.std(_BENCHMARK_RETURNS, ddof=1)
        years = 366.0 / 365.25  # native endpoints 2023-12-31 and 2024-12-31
        pa_return = np.prod(1.0 + _BENCHMARK_RETURNS) ** (1.0 / years) - 1.0
        compounded = np.expm1(4.0 * 0.5 * np.array((np.mean(up), np.mean(down))))
        patched = compounded + 0.5 * (pa_return - compounded.sum())
        return np.array((patched[0], np.nan, patched[1])) / annualized_vol

    values = (
        np.log1p(_BENCHMARK_RETURNS) if convention is SharpeConvention.LOG else _BENCHMARK_RETURNS
    )
    up_values = values[2:]
    down_values = values[:2]
    # Each observed regime contains half the sample, so sqrt(4) * p_s equals one.
    sample_std = np.std(values, ddof=1)
    return np.array((np.mean(up_values) / sample_std, np.nan, np.mean(down_values) / sample_std))


@pytest.mark.parametrize("dtype", ("float64", "Float64"))
@pytest.mark.parametrize("convention", tuple(SharpeConvention))
def test_regime_table_preserves_ordered_benchmark_means(
    dtype: str,
    convention: SharpeConvention,
) -> None:
    """Assign benchmark means by regime order for every Sharpe convention and storage dtype."""
    sampled_returns, prices = _inputs(dtype)
    original_returns = sampled_returns.copy(deep=True)
    original_prices = prices.copy(deep=True)
    expected_means = np.array((0.02, np.nan, -0.015))

    table, components = compute_regimes_pa_perf_table_from_sampled_returns(
        sampled_returns_with_regime_id=sampled_returns,
        prices=prices,
        benchmark=_BENCHMARK,
        perf_params=PerfParams(
            freq="QE",
            return_type=ReturnTypes.RELATIVE,
            sharpe_convention=convention,
        ),
        freq="QE",
        is_use_benchmark_means=True,
        is_add_ra_perf_table=False,
        regime_ids=list(_REGIME_ORDER),
    )

    regime_average = components[RegimeData.REGIME_AVG]
    regime_pa = components[RegimeData.REGIME_PA]
    regime_sharpe = components[RegimeData.REGIME_SHARPE]
    assert regime_average.columns.tolist() == ["Up Average", "Flat Average", "Down Average"]
    assert regime_pa.columns.tolist() == ["Up P.a.", "Flat P.a.", "Down P.a."]
    assert regime_sharpe.columns.tolist() == ["Up-Sharpe", "Flat-Sharpe", "Down-Sharpe"]
    np.testing.assert_allclose(
        regime_average.loc[_BENCHMARK].to_numpy(dtype=float, na_value=np.nan),
        expected_means,
        rtol=0.0,
        atol=_TOLERANCE,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        regime_pa.loc[_BENCHMARK].to_numpy(dtype=float, na_value=np.nan),
        expected_means,
        rtol=0.0,
        atol=_TOLERANCE,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        regime_sharpe.loc[_BENCHMARK].to_numpy(dtype=float, na_value=np.nan),
        _expected_sharpe(convention),
        rtol=0.0,
        atol=_TOLERANCE,
        equal_nan=True,
    )
    pd.testing.assert_frame_equal(
        table.loc[:, regime_pa.columns],
        regime_pa,
    )
    pd.testing.assert_frame_equal(sampled_returns, original_returns)
    pd.testing.assert_frame_equal(prices, original_prices)


def test_regime_table_preserves_annualized_values_when_benchmark_means_are_disabled() -> None:
    """Leave the established annualized benchmark contributions unchanged when the flag is off."""
    sampled_returns, prices = _inputs("float64")
    expected_pa = np.expm1(np.array((0.02, np.nan, -0.015)) * 2.0)

    _, components = compute_regimes_pa_perf_table_from_sampled_returns(
        sampled_returns_with_regime_id=sampled_returns,
        prices=prices,
        benchmark=_BENCHMARK,
        perf_params=PerfParams(freq="QE", return_type=ReturnTypes.RELATIVE),
        freq="QE",
        is_use_benchmark_means=False,
        is_add_ra_perf_table=False,
        additive_pa_returns_to_pa_total=False,
        regime_ids=list(_REGIME_ORDER),
    )

    np.testing.assert_allclose(
        components[RegimeData.REGIME_PA].loc[_BENCHMARK].to_numpy(dtype=float),
        expected_pa,
        rtol=0.0,
        atol=_TOLERANCE,
        equal_nan=True,
    )
