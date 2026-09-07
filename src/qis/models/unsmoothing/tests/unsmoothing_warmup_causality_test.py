"""Verify that rolling AR unsmoothing publishes only point-in-time estimates.

The rolling estimator must not revise an existing prefix when later returns are appended. These
tests separate backward-filled warmup coefficients from the full-sample mean seed, then exercise
the causal availability floor, insufficient-data policies, mixed histories, and public wrappers.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from qis.models.linear.ewm import MeanAdjType
from qis.models.unsmoothing.ar_lag import (
    InsufficientData,
    adjust_returns_with_ar,
    compute_ar_unsmoothed_prices,
    min_obs_for_ar_unsmoothing,
    unsmooth_returns_ar1_ewma,
)


Diagnostics = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def _seeded_ar_returns(num_periods: int = 50, seed: int = 20260907) -> pd.DataFrame:
    """Return a deterministic monthly AR(1) sample with a drifting point-in-time state."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(loc=0.001, scale=0.025, size=num_periods)
    values = np.empty(num_periods, dtype=float)
    values[0] = innovations[0]
    for position in range(1, num_periods):
        values[position] = 0.55 * values[position - 1] + innovations[position]
    index = pd.date_range("2018-01-31", periods=num_periods, freq="ME")
    return pd.DataFrame({"asset": values}, index=index)


def _adjust_with_diagnostics(
    returns: pd.DataFrame,
    *,
    ar_order: int = 1,
    mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
    warmup_period: int | None = 10,
    apply_ewma_mean_smoother: bool = True,
    non_negative: bool = False,
    insufficient_data: InsufficientData = InsufficientData.NAN,
) -> Diagnostics:
    """Call the public engine while retaining its three diagnostic panels."""
    result = adjust_returns_with_ar(
        returns=returns,
        ar_order=ar_order,
        span=20,
        mean_adj_type=mean_adj_type,
        warmup_period=warmup_period,
        apply_ewma_mean_smoother=apply_ewma_mean_smoother,
        non_negative=non_negative,
        return_diagnostics=True,
        insufficient_data=insufficient_data,
    )
    assert isinstance(result, tuple)
    return result


def _assert_prefix_equal(shorter: Sequence[pd.DataFrame], longer: Sequence[pd.DataFrame]) -> None:
    """Assert exact equality, including missing placement, over every diagnostic panel."""
    for shorter_panel, longer_panel in zip(shorter, longer, strict=True):
        pd.testing.assert_frame_equal(
            shorter_panel,
            longer_panel.head(len(shorter_panel)),
            check_exact=True,
        )


def test_adjust_returns_with_ar_does_not_retroactively_fill_warmup_prefix() -> None:
    """A newly identified beta must not be copied into an earlier unidentified prefix."""
    returns = _seeded_ar_returns(num_periods=6)
    shorter = _adjust_with_diagnostics(
        returns.head(5),
        mean_adj_type=MeanAdjType.NONE,
        warmup_period=2,
        apply_ewma_mean_smoother=False,
    )
    longer = _adjust_with_diagnostics(
        returns,
        mean_adj_type=MeanAdjType.NONE,
        warmup_period=2,
        apply_ewma_mean_smoother=False,
    )

    _assert_prefix_equal(shorter=shorter, longer=longer)
    assert bool(shorter[0].isna().all().all())
    assert bool(shorter[1].isna().all().all())


@pytest.mark.parametrize("warmup_period", [10, None])
@pytest.mark.parametrize("non_negative", [False, True])
def test_adjust_returns_with_ar_ewma_mean_is_prefix_invariant(
    warmup_period: int | None,
    non_negative: bool,
) -> None:
    """The default point-in-time mean seed must not depend on the eventual sample end."""
    returns = _seeded_ar_returns()
    shorter = _adjust_with_diagnostics(
        returns.head(40),
        ar_order=2,
        warmup_period=warmup_period,
        non_negative=non_negative,
    )
    longer = _adjust_with_diagnostics(
        returns,
        ar_order=2,
        warmup_period=warmup_period,
        non_negative=non_negative,
    )

    _assert_prefix_equal(shorter=shorter, longer=longer)


@pytest.mark.parametrize(
    ("ar_order", "warmup_period", "expected_floor"),
    [
        (1, 0, 3),
        (2, 0, 3),
        (1, 1, 5),
        (2, 1, 5),
        (1, 3, 9),
        (2, 3, 9),
        (1, None, 23),
        (2, None, 23),
    ],
)
def test_min_obs_for_ar_unsmoothing_matches_causal_availability(
    ar_order: int,
    warmup_period: int | None,
    expected_floor: int,
) -> None:
    """The documented row floor is the first length that yields one causal return."""
    returns = _seeded_ar_returns(num_periods=expected_floor)

    assert min_obs_for_ar_unsmoothing(ar_order, warmup_period) == expected_floor
    below_floor = _adjust_with_diagnostics(
        returns.head(-1),
        ar_order=ar_order,
        mean_adj_type=MeanAdjType.NONE,
        warmup_period=warmup_period,
        apply_ewma_mean_smoother=False,
    )[0]
    at_floor = _adjust_with_diagnostics(
        returns,
        ar_order=ar_order,
        mean_adj_type=MeanAdjType.NONE,
        warmup_period=warmup_period,
        apply_ewma_mean_smoother=False,
    )[0]

    assert below_floor.dropna().shape[0] == 0
    assert at_floor.dropna().shape[0] == 1


def test_adjust_returns_with_ar_insufficient_data_uses_causal_floor() -> None:
    """Raise and passthrough policies must use the same causal availability boundary."""
    returns = _seeded_ar_returns(num_periods=4)

    with pytest.raises(ValueError, match=r"needs 5 rows.*got 4"):
        _adjust_with_diagnostics(
            returns,
            mean_adj_type=MeanAdjType.NONE,
            warmup_period=1,
            apply_ewma_mean_smoother=False,
            insufficient_data=InsufficientData.RAISE,
        )

    unsmoothed, betas, r_squared = _adjust_with_diagnostics(
        returns,
        mean_adj_type=MeanAdjType.NONE,
        warmup_period=1,
        apply_ewma_mean_smoother=False,
        insufficient_data=InsufficientData.PASSTHROUGH,
    )
    pd.testing.assert_frame_equal(unsmoothed, returns)
    assert bool(betas.eq(0.0).all().all())
    assert bool(r_squared.isna().all().all())


def test_adjust_returns_with_ar_first_output_uses_prior_beta() -> None:
    """The first available return must use the coefficient estimated one period earlier."""
    returns = _seeded_ar_returns(num_periods=9)
    unsmoothed, betas, _ = _adjust_with_diagnostics(
        returns,
        mean_adj_type=MeanAdjType.NONE,
        warmup_period=3,
        apply_ewma_mean_smoother=False,
    )
    position = 8
    return_values = returns["asset"].to_numpy(dtype=float)
    beta_values = betas["asset"].to_numpy(dtype=float)
    expected = (
        return_values[position] - beta_values[position - 1] * return_values[position - 1]
    ) / (1.0 - beta_values[position - 1])

    assert bool(unsmoothed.head(position).isna().all().all())
    assert unsmoothed["asset"].to_numpy(dtype=float)[position] == pytest.approx(expected)


def test_adjust_returns_with_ar_mixed_panel_remains_column_local_and_causal() -> None:
    """Complete, ragged, and all-missing columns must coexist without cross-column leakage."""
    complete = _seeded_ar_returns()
    complete_values = complete.to_numpy(dtype=float).reshape(-1)
    mixed = pd.DataFrame(
        {
            "complete": complete_values,
            "ragged": np.where(np.arange(len(complete_values)) < 5, np.nan, complete_values),
            "all_missing": np.full(len(complete_values), np.nan),
        },
        index=complete.index,
    )
    shorter = _adjust_with_diagnostics(mixed.head(40), ar_order=2)
    longer = _adjust_with_diagnostics(mixed, ar_order=2)

    _assert_prefix_equal(shorter=shorter, longer=longer)
    assert list(shorter[0].columns) == ["complete", "ragged", "all_missing"]
    assert bool(np.isnan(shorter[0].to_numpy(dtype=float)[:, 2]).all())


def test_unsmooth_returns_ar1_ewma_preserves_prefix_through_public_shim() -> None:
    """The AR(1) compatibility entry point must retain the engine's causal history."""
    returns = _seeded_ar_returns()

    shorter = unsmooth_returns_ar1_ewma(returns=returns.head(40))
    longer = unsmooth_returns_ar1_ewma(returns=returns)

    _assert_prefix_equal(shorter=shorter, longer=longer)


@pytest.mark.parametrize("use_frequency_map", [False, True])
def test_compute_ar_unsmoothed_prices_preserves_prefix_after_return_conversion(
    use_frequency_map: bool,
) -> None:
    """Single- and per-asset frequency paths must preserve causal price reconstruction."""
    returns = _seeded_ar_returns()
    prices = 100.0 * (1.0 + returns).cumprod()
    frequency: str | pd.Series = (
        pd.Series({"asset": "ME"}, dtype="string") if use_frequency_map else "ME"
    )

    shorter = compute_ar_unsmoothed_prices(
        prices=prices.head(40),
        ar_order=1,
        freq=frequency,
        span=20,
        mean_adj_type=MeanAdjType.EWMA,
        warmup_period=3,
        is_log_returns=False,
    )
    longer = compute_ar_unsmoothed_prices(
        prices=prices,
        ar_order=1,
        freq=frequency,
        span=20,
        mean_adj_type=MeanAdjType.EWMA,
        warmup_period=3,
        is_log_returns=False,
    )

    _assert_prefix_equal(shorter=shorter, longer=longer)
