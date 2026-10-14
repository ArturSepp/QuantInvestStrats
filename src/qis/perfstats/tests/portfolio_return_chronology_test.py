"""Prior-close portfolio weights are dated decisions, not physical row neighbors.

Literal returns below are derived from each date's preceding decision. Permutations and future-only
changes must preserve that accounting rule, while output labels and caller-owned frames stay intact.
"""

import numpy as np
import pandas as pd
import pytest

import qis


_DATES = pd.date_range("2024-01-01", periods=6, freq="D", name="date")
_ORDERS = [list(range(6)), list(reversed(range(6))), [2, 0, 4, 1, 5, 3]]


def _histories(dtype: str = "float64") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alternating decisions expose both forward and backward physical-row lag errors."""
    weights = pd.DataFrame({"asset": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]}, index=_DATES, dtype=dtype)
    returns = pd.DataFrame({"asset": [0.0, 0.1, 0.1, 0.1, 0.1, 0.1]}, index=_DATES, dtype=dtype)
    return weights, returns


@pytest.mark.parametrize("weight_order", _ORDERS, ids=["ascending", "descending", "shuffled"])
@pytest.mark.parametrize("return_order", _ORDERS, ids=["ascending", "descending", "shuffled"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_to_portfolio_returns_uses_prior_close_dates_and_preserves_output_order(
    weight_order: list[int],
    return_order: list[int],
    dtype: str,
) -> None:
    """Lag on the shared date grid, even when the two frames arrive in different orders."""
    weights, returns = _histories(dtype)
    weights, returns = weights.iloc[weight_order], returns.iloc[return_order]
    original_weights, original_returns = weights.copy(deep=True), returns.copy(deep=True)
    # Jan 1/3/5 decisions are zero; Jan 2/4 decisions are one. The first date has no predecessor.
    expected = pd.Series(
        [np.nan, 0.0, 0.1, 0.0, 0.1, 0.0], index=_DATES, name="strategy", dtype=dtype
    )

    actual = qis.to_portfolio_returns(weights, returns, portfolio_name="strategy")

    pd.testing.assert_series_equal(actual, expected.reindex(returns.index), check_exact=True)
    pd.testing.assert_frame_equal(weights, original_weights, check_exact=True)
    pd.testing.assert_frame_equal(returns, original_returns, check_exact=True)


def test_to_portfolio_returns_does_not_revise_the_past_from_a_future_decision() -> None:
    """A Jan 5 decision changes only Jan 6; neither a full history nor its prefixes may leak it."""
    weights, returns = _histories()
    weights, returns = weights.iloc[_ORDERS[2]], returns.iloc[_ORDERS[2]]
    before = qis.to_portfolio_returns(weights, returns)
    future = weights.copy(deep=True)
    future.loc[_DATES[4], "asset"] = 1.0
    expected = pd.Series([np.nan, 0.0, 0.1, 0.0, 0.1, 0.1], index=_DATES, name="portfolios")

    actual = qis.to_portfolio_returns(future, returns)

    pd.testing.assert_series_equal(actual, expected.reindex(returns.index), check_exact=True)
    pd.testing.assert_series_equal(
        actual.loc[actual.index < _DATES[5]], before.loc[before.index < _DATES[5]], check_exact=True
    )
    for cutoff in _DATES[[2, 3]]:
        prefix_returns = returns.loc[returns.index <= cutoff]
        prefix = qis.to_portfolio_returns(future.loc[future.index <= cutoff], prefix_returns)
        pd.testing.assert_series_equal(
            prefix, expected.reindex(prefix_returns.index), check_exact=True
        )


def test_to_portfolio_returns_preserves_mixed_missing_terms_without_renormalizing() -> None:
    """Partial and zero-observation rows must keep the existing missing-term accounting."""
    dates = _DATES[:5].tz_localize("America/New_York")
    weights = pd.DataFrame({"a": [0.5] * 5, "b": [0.5, 0.5, np.nan, 0.5, 0.5]}, index=dates)
    returns = pd.DataFrame(
        {"a": [0.0, 0.02, np.nan, np.nan, 0.04], "b": [0.0, np.nan, 0.03, 0.06, 0.06]}, index=dates
    )
    order = [2, 0, 4, 1, 3]
    # Half of 2% is 1%, not 2%; on Jan 4 neither term is observed, including the missing weight.
    expected = pd.Series([np.nan, 0.01, 0.015, np.nan, 0.05], index=dates, name="portfolios")

    actual = qis.to_portfolio_returns(weights.iloc[order], returns.iloc[order])

    pd.testing.assert_series_equal(actual, expected.iloc[order], rtol=1e-14, atol=0.0)


def test_to_portfolio_returns_preserves_nullable_aggregation_behavior() -> None:
    """Date preparation must not widen the runtime's existing nullable conversion domain."""
    weights, returns = _histories("Float64")
    returns.loc[_DATES[2], "asset"] = pd.NA
    # NumPy conversion of nullable arrays differs across supported pandas versions. Characterize
    # the unchanged aggregation boundary, not a proposed new nullable-support policy.
    contributions = returns.multiply(weights.shift(1)).to_numpy()
    if contributions.dtype == object:
        with pytest.raises(TypeError):
            qis.to_portfolio_returns(weights.iloc[_ORDERS[2]], returns.iloc[_ORDERS[2]])
    else:
        expected = pd.Series([np.nan, 0.0, np.nan, 0.0, 0.1, 0.0], index=_DATES, name="portfolios")
        actual = qis.to_portfolio_returns(weights.iloc[_ORDERS[2]], returns.iloc[_ORDERS[2]])
        pd.testing.assert_series_equal(actual, expected.iloc[_ORDERS[2]], check_exact=True)


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(6),
        pd.DatetimeIndex([_DATES[2], _DATES[0], _DATES[2], _DATES[1], _DATES[5], _DATES[3]]),
        pd.DatetimeIndex([_DATES[2], pd.NaT, _DATES[4], _DATES[1], _DATES[5], _DATES[3]]),
    ],
    ids=["positional", "duplicate", "NaT"],
)
def test_to_portfolio_returns_retains_nonunique_or_nondated_grid_behavior(index: pd.Index) -> None:
    """The chronology correction must not invent ordering or rejection for ambiguous axes."""
    weights, returns = _histories()
    weights.index = returns.index = index
    expected = pd.Series([np.nan, 0.0, 0.1, 0.0, 0.1, 0.0], index=index, name="portfolios")

    actual = qis.to_portfolio_returns(weights, returns)

    pd.testing.assert_series_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("count", [0, 1])
def test_to_portfolio_returns_preserves_empty_and_single_date_results(count: int) -> None:
    """An empty grid stays empty; one dated observation still has no prior decision."""
    weights, returns = _histories()
    expected = pd.Series([np.nan] * count, index=_DATES[:count], name="portfolios", dtype="float64")

    actual = qis.to_portfolio_returns(weights.iloc[:count], returns.iloc[:count])

    pd.testing.assert_series_equal(actual, expected, check_exact=True)


def test_to_portfolio_returns_does_not_adopt_sparse_weight_schedule_semantics() -> None:
    """Unequal grids still require caller preparation rather than implicit holding persistence."""
    weights, returns = _histories()

    with pytest.raises(ValueError, match="Length of values"):
        qis.to_portfolio_returns(weights, returns.iloc[:-1])
