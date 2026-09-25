"""Verify that factor-lag unsmoothing publishes only point-in-time estimates.

The factor-lag estimator must not revise an existing prefix when later returns are appended.
These tests isolate unavailable warm-up coefficients from EWMA mean initialization, then combine
the default operations with ragged and all-missing assets under warnings-as-errors.
"""

from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd
import pytest

from qis.models.linear.ewm import MeanAdjType
from qis.models.unsmoothing.factor_lag import adjust_returns_with_factor_lag


Diagnostics = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def _factor_lag_returns(
    num_periods: int = 80,
    seed: int = 20260923,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return deterministic monthly factor and lag-responsive asset histories."""
    rng = np.random.default_rng(seed)
    factor_values = rng.normal(loc=0.0, scale=0.04, size=num_periods)
    innovations = rng.normal(loc=0.0, scale=0.005, size=(num_periods, 2))
    factor_lag = np.concatenate(([np.nan], factor_values[:-1]))
    return_values = np.column_stack(
        [
            0.20 * factor_values + 0.80 * factor_lag + innovations[:, 0],
            -0.30 * factor_values - 0.50 * factor_lag + innovations[:, 1],
        ]
    )
    return_values[0] = innovations[0]
    index = pd.date_range("2015-01-31", periods=num_periods, freq="ME")
    returns = pd.DataFrame(return_values, index=index, columns=["positive", "negative"])
    factor = pd.Series(factor_values, index=index, name="factor")
    return returns, factor


def _adjust_with_diagnostics(
    returns: pd.DataFrame,
    factor_returns: pd.Series,
    *,
    factor_lag_order: int = 1,
    mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
    warmup_period: int | None = 4,
    sign_tie_to_contemporaneous: bool = True,
    apply_ewma_mean_smoother: bool = True,
) -> Diagnostics:
    """Call the public factor-lag engine while retaining both diagnostics."""
    result = adjust_returns_with_factor_lag(
        returns=returns,
        factor_returns=factor_returns,
        factor_lag_order=factor_lag_order,
        span=12,
        mean_adj_type=mean_adj_type,
        warmup_period=warmup_period,
        sign_tie_to_contemporaneous=sign_tie_to_contemporaneous,
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


def test_adjust_returns_with_factor_lag_keeps_warmup_prefix_unavailable() -> None:
    """A newly identified beta vector must not fill an earlier unavailable prefix."""
    returns, factor = _factor_lag_returns()
    shorter = _adjust_with_diagnostics(
        returns.head(6),
        factor.head(6),
        mean_adj_type=MeanAdjType.NONE,
        sign_tie_to_contemporaneous=False,
        apply_ewma_mean_smoother=False,
    )
    longer = _adjust_with_diagnostics(
        returns,
        factor,
        mean_adj_type=MeanAdjType.NONE,
        sign_tie_to_contemporaneous=False,
        apply_ewma_mean_smoother=False,
    )

    _assert_prefix_equal(shorter=shorter, longer=longer)
    corrected, beta_d, r_squared = shorter
    pd.testing.assert_frame_equal(corrected, returns.head(6))
    assert bool(beta_d.isna().all().all())
    assert bool(r_squared.isna().all().all())

    corrected, beta_d, r_squared = longer
    # The tensor mask ends at position 4; the outer mask skips four later finite estimates.
    first_coefficient = returns.index[9]
    assert beta_d.first_valid_index() == first_coefficient
    assert r_squared.first_valid_index() == first_coefficient
    pd.testing.assert_frame_equal(corrected.iloc[:10], returns.iloc[:10])
    assert not corrected.iloc[10].equals(returns.iloc[10])


@pytest.mark.parametrize("factor_lag_order", [1, 2])
def test_adjust_returns_with_factor_lag_ewma_mean_is_prefix_invariant(
    factor_lag_order: int,
) -> None:
    """The default point-in-time mean seed must not depend on the eventual sample end."""
    returns, factor = _factor_lag_returns()
    shorter = _adjust_with_diagnostics(
        returns.head(50),
        factor.head(50),
        factor_lag_order=factor_lag_order,
    )
    longer = _adjust_with_diagnostics(
        returns,
        factor,
        factor_lag_order=factor_lag_order,
    )

    _assert_prefix_equal(shorter=shorter, longer=longer)


def test_adjust_returns_with_factor_lag_mixed_panel_is_causal_without_warnings() -> None:
    """Complete, ragged, and all-missing assets must remain independent and causal."""
    returns, factor = _factor_lag_returns()
    returns.loc[returns.index[:7], "negative"] = np.nan
    returns["all_missing"] = np.nan
    returns_before = returns.copy(deep=True)
    factor_before = factor.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shorter = _adjust_with_diagnostics(returns.head(50), factor.head(50))
        longer = _adjust_with_diagnostics(returns, factor)

    _assert_prefix_equal(shorter=shorter, longer=longer)
    for panel in shorter:
        assert panel.index.equals(returns.index[:50])
        assert panel.columns.equals(returns.columns)
        assert bool(panel["all_missing"].isna().all())
    pd.testing.assert_frame_equal(returns, returns_before)
    pd.testing.assert_series_equal(factor, factor_before)
