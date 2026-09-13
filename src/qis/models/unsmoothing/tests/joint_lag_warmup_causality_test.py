"""Verify that joint-lag unsmoothing publishes only point-in-time estimates.

The joint estimator must not revise an existing prefix when later returns are appended. These
tests isolate unavailable warmup coefficients from EWMA mean initialization, then combine the
default operations in a mixed panel under warnings-as-errors.
"""

from collections.abc import Sequence
from typing import cast
import warnings

import numpy as np
import pandas as pd

from qis.models.linear.ewm import MeanAdjType
from qis.models.unsmoothing.joint_lag import adjust_returns_with_joint_unsmoothing


Diagnostics = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def _joint_returns(
    num_periods: int = 60,
    seed: int = 20260913,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return deterministic monthly factor and joint own-lag return histories."""
    rng = np.random.default_rng(seed)
    factor_values = rng.normal(loc=0.002, scale=0.035, size=num_periods)
    innovations = rng.normal(loc=0.001, scale=0.018, size=(num_periods, 2))
    return_values = np.empty_like(innovations)
    return_values[0] = innovations[0]
    for position in range(1, num_periods):
        return_values[position, 0] = (
            0.45 * return_values[position - 1, 0]
            + 0.35 * factor_values[position - 1]
            + innovations[position, 0]
        )
        return_values[position, 1] = (
            0.25 * return_values[position - 1, 1]
            - 0.20 * factor_values[position - 1]
            + innovations[position, 1]
        )
    index = pd.date_range("2018-01-31", periods=num_periods, freq="ME")
    returns = pd.DataFrame(return_values, index=index, columns=["complete", "ragged"])
    factor = pd.Series(factor_values, index=index, name="factor")
    return returns, factor


def _adjust_with_diagnostics(
    returns: pd.DataFrame,
    factor_returns: pd.Series,
    *,
    mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
    warmup_period: int | None = 16,
    apply_ewma_mean_smoother: bool = True,
) -> Diagnostics:
    """Call the public joint engine while retaining both coefficient panels."""
    result = adjust_returns_with_joint_unsmoothing(
        returns=returns,
        factor_returns=factor_returns,
        span=20,
        mean_adj_type=mean_adj_type,
        warmup_period=warmup_period,
        apply_ewma_mean_smoother=apply_ewma_mean_smoother,
        return_diagnostics=True,
    )
    assert isinstance(result, tuple)
    return result


def _assert_prefix_equal(shorter: Sequence[pd.DataFrame], longer: Sequence[pd.DataFrame]) -> None:
    """Assert exact equality, including missing placement, over every returned panel."""
    for shorter_panel, longer_panel in zip(shorter, longer, strict=True):
        pd.testing.assert_frame_equal(
            shorter_panel,
            longer_panel.head(len(shorter_panel)),
            check_exact=True,
        )


def _first_joint_position(
    returns: pd.DataFrame,
    factor: pd.Series,
    column: str,
) -> int:
    """Return the first row where target, own lag, and factor lag are all observable."""
    jointly_observable = (
        returns[column].notna()
        & returns[column].shift(1).notna()
        & factor.shift(1).notna()
    )
    positions = np.flatnonzero(jointly_observable.to_numpy())
    assert positions.size
    return int(positions[0])


def test_adjust_returns_with_joint_unsmoothing_keeps_warmup_prefix_unavailable() -> None:
    """A newly identified coefficient pair must not fill an earlier unavailable prefix."""
    returns, factor = _joint_returns()
    shorter = _adjust_with_diagnostics(
        returns.head(10),
        factor.head(10),
        mean_adj_type=MeanAdjType.NONE,
        warmup_period=16,
        apply_ewma_mean_smoother=False,
    )
    longer = _adjust_with_diagnostics(
        returns,
        factor,
        mean_adj_type=MeanAdjType.NONE,
        warmup_period=16,
        apply_ewma_mean_smoother=False,
    )

    _assert_prefix_equal(shorter=shorter, longer=longer)
    assert all(bool(panel.isna().all().all()) for panel in shorter)

    corrected, phi1, beta1 = longer
    first_coefficient = phi1["complete"].first_valid_index()
    assert first_coefficient == beta1["complete"].first_valid_index()
    assert first_coefficient is not None
    coefficient_position = int(returns.index.get_indexer([first_coefficient])[0])
    assert coefficient_position == _first_joint_position(returns, factor, "complete") + 16
    first_position = coefficient_position + 1
    observed_return = cast(float, returns["complete"].iloc[first_position])
    prior_observed_return = cast(float, returns["complete"].iloc[first_position - 1])
    prior_phi = cast(float, phi1["complete"].iloc[first_position - 1])
    prior_beta = cast(float, beta1["complete"].iloc[first_position - 1])
    factor_change = cast(float, factor.iloc[first_position] - factor.iloc[first_position - 1])
    expected = (
        observed_return - prior_phi * prior_observed_return + prior_beta * factor_change
    ) / (1.0 - prior_phi)
    assert cast(bool, corrected["complete"].iloc[:first_position].isna().all())
    actual = cast(float, corrected["complete"].iloc[first_position])
    np.testing.assert_allclose(actual, expected)


def test_adjust_returns_with_joint_unsmoothing_ewma_mean_is_prefix_invariant() -> None:
    """The point-in-time EWMA mean seed must not depend on the eventual sample end."""
    returns, factor = _joint_returns()
    shorter = _adjust_with_diagnostics(
        returns.head(45),
        factor.head(45),
        warmup_period=None,
    )
    longer = _adjust_with_diagnostics(
        returns,
        factor,
        warmup_period=None,
    )

    _assert_prefix_equal(shorter=shorter, longer=longer)
    _, phi1, beta1 = longer
    first_joint = _first_joint_position(returns, factor, "complete")
    assert phi1["complete"].first_valid_index() == returns.index[first_joint]
    assert beta1["complete"].first_valid_index() == returns.index[first_joint]


def test_adjust_returns_with_joint_unsmoothing_mixed_panel_is_causal_without_warnings() -> None:
    """Complete, ragged, and all-missing columns must remain independent and causal."""
    returns, factor = _joint_returns()
    returns.loc[returns.index[:6], "ragged"] = np.nan
    returns["all_missing"] = np.nan
    returns_before = returns.copy(deep=True)
    factor_before = factor.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shorter = _adjust_with_diagnostics(returns.head(45), factor.head(45))
        longer = _adjust_with_diagnostics(returns, factor)

    _assert_prefix_equal(shorter=shorter, longer=longer)
    _, phi1, beta1 = longer
    for column in ("complete", "ragged"):
        expected_position = _first_joint_position(returns, factor, column) + 16
        assert phi1[column].first_valid_index() == returns.index[expected_position]
        assert beta1[column].first_valid_index() == returns.index[expected_position]
    for panel in shorter:
        assert panel.index.equals(returns.index[:45])
        assert panel.columns.equals(returns.columns)
        assert cast(bool, panel["complete"].notna().any())
        assert cast(bool, panel["ragged"].notna().any())
        assert cast(bool, panel["all_missing"].isna().all())
    pd.testing.assert_frame_equal(returns, returns_before)
    pd.testing.assert_series_equal(factor, factor_before)


def test_adjust_returns_with_joint_unsmoothing_holds_betas_across_joint_gaps() -> None:
    """A row excluded from the joint fit must age both regression moments equally."""
    returns, factor = _joint_returns()
    returns.loc[returns.index[30], "complete"] = np.nan
    _, phi1, beta1 = _adjust_with_diagnostics(
        returns,
        factor,
        mean_adj_type=MeanAdjType.NONE,
        warmup_period=None,
        apply_ewma_mean_smoother=False,
    )

    # Row 30 has no target and row 31 has no own lag, so neither adds information.
    np.testing.assert_allclose(phi1["complete"].iloc[31], phi1["complete"].iloc[29])
    np.testing.assert_allclose(beta1["complete"].iloc[31], beta1["complete"].iloc[29])
