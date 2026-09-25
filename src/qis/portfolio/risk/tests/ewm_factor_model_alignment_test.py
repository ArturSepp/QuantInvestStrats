"""Regression tests for EWM factor-model date alignment."""

import pandas as pd
import pytest

import qis


def _make_literal_factor_relation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned panels satisfying the independent relation ``y = 2x``."""
    index = pd.date_range(
        "2024-01-31",
        periods=6,
        freq="ME",
        tz="UTC",
        name="date",
    )
    factor_returns = pd.DataFrame(
        {"factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        index=index,
    )
    asset_returns = pd.DataFrame(
        {"asset": 2.0 * factor_returns["factor"].to_numpy()},
        index=index,
    )
    return factor_returns, asset_returns


def test_ewm_factor_model_preserves_aligned_literal_beta() -> None:
    """Keep the exact beta control for valid identically indexed panels."""
    factor_returns, asset_returns = _make_literal_factor_relation()
    model = qis.EwmLinearModel(x=factor_returns, y=asset_returns)

    model.fit(span=3, mean_adj_type=qis.MeanAdjType.NONE, warmup_period=0)

    # The first loading is unavailable by construction; every estimable beta stays exactly two.
    expected = pd.DataFrame(
        {"asset": [float("nan"), 2.0, 2.0, 2.0, 2.0, 2.0]},
        index=factor_returns.index,
    )
    pd.testing.assert_frame_equal(model.get_factor_loadings("factor"), expected)


@pytest.mark.parametrize(
    ("mismatch", "as_series"),
    (
        ("reordered", False),
        ("missing_extra", False),
        ("different_length", False),
        ("duplicate_multiplicity", False),
        ("timezone", False),
        ("reordered", True),
    ),
    ids=("reordered", "missing-extra", "different-length", "duplicates", "timezone", "series"),
)
def test_ewm_factor_model_rejects_misaligned_indexes_without_mutating_state(
    mismatch: str,
    as_series: bool,
) -> None:
    """Reject incompatible factor and asset dates before fitting or state assignment.

    Args:
        mismatch: Index incompatibility exercised by the regression cell.
        as_series: Use the constructor's Series-to-DataFrame normalization path.
    """
    factor_returns, asset_returns = _make_literal_factor_relation()
    if mismatch == "reordered":
        asset_returns = asset_returns.iloc[::-1]
    elif mismatch == "missing_extra":
        asset_returns.index = asset_returns.index[:-1].append(
            pd.DatetimeIndex([asset_returns.index[-1] + pd.offsets.MonthEnd()])
        )
    elif mismatch == "different_length":
        asset_returns = asset_returns.iloc[:-1]
    elif mismatch == "duplicate_multiplicity":
        factor_returns.index = pd.DatetimeIndex(
            [factor_returns.index[0], factor_returns.index[0], *factor_returns.index[1:-1]],
            name="date",
        )
        asset_returns.index = pd.DatetimeIndex(
            [
                asset_returns.index[0],
                asset_returns.index[1],
                asset_returns.index[1],
                *asset_returns.index[2:-1],
            ],
            name="date",
        )
    elif mismatch == "timezone":
        asset_returns.index = asset_returns.index.tz_convert("America/New_York")
    else:
        raise AssertionError(f"unsupported test mismatch: {mismatch}")

    x = factor_returns.iloc[:, 0] if as_series else factor_returns
    y = asset_returns.iloc[:, 0] if as_series else asset_returns
    model = qis.EwmLinearModel(x=x, y=y)
    # Snapshot normalized state to prove rejection happens before fitting mutates the model.
    original_x = model.x.copy(deep=True)
    original_y = model.y.copy(deep=True)

    with pytest.raises(
        ValueError, match="x and y pandas index labels and order must match exactly"
    ):
        model.fit(span=3, mean_adj_type=qis.MeanAdjType.NONE, warmup_period=0)

    assert model.loadings is None
    pd.testing.assert_frame_equal(model.x, original_x)
    pd.testing.assert_frame_equal(model.y, original_y)
