"""A requested portfolio total must not be replaced by an ordinary instrument group."""

import numpy as np
import pandas as pd
import pytest

import qis
from qis.utils.df_groups import agg_df_by_groups_ax1


# An empty string still requests a total; only None disables total creation.
@pytest.mark.parametrize("total_column", ["Total", "Portfolio", ""])
@pytest.mark.parametrize("index_data", [None, ["B"], []])
def test_get_group_dict_rejects_observed_total_label_even_without_selected_assets(
    total_column, index_data
):
    """An empty restricted subgroup still overwrote the dictionary's total key."""
    groups = pd.Series({"A": total_column, "B": "Cash"}, name="classification")
    before = groups.copy(deep=True)
    order = ["Cash", total_column]
    before_selection = None if index_data is None else list(index_data)

    with pytest.raises(ValueError, match="total_column.*group_data"):
        qis.get_group_dict(
            groups, index_data=index_data, group_order=order, total_column=total_column
        )

    pd.testing.assert_series_equal(groups, before, check_exact=True)
    assert order == ["Cash", total_column]
    assert index_data == before_selection


def _mixed_panel(storage):
    """Keep numeric storage differences at the same labeled grouping boundary."""
    dates = pd.date_range("2026-01-01", periods=2, tz="UTC", name="date")
    frame = pd.DataFrame({"A": [0.6, np.nan], "B": [0.4, 0.5]}, index=dates)
    if storage == "nullable":
        frame = frame.astype("Float64")
    elif storage == "mixed":
        frame["A"] = frame["A"].astype("Float64")
    frame.columns.name = "asset"
    return frame


@pytest.mark.parametrize("storage", ["ordinary", "nullable", "mixed"])
def test_split_df_by_groups_rejects_total_collision_without_mutating_inputs(storage):
    """Panel storage must not turn a conflicting total into a partial sub-panel."""
    frame = _mixed_panel(storage)
    groups = pd.Series({"A": "Total", "B": "Cash"}, name="classification")
    before_frame, before_groups = frame.copy(deep=True), groups.copy(deep=True)

    with pytest.raises(ValueError, match="total_column.*group_data"):
        qis.split_df_by_groups(frame, groups, total_column="Total")

    pd.testing.assert_frame_equal(frame, before_frame, check_exact=True)
    pd.testing.assert_series_equal(groups, before_groups, check_exact=True)


@pytest.mark.parametrize("storage", ["ordinary", "nullable", "mixed"])
def test_agg_df_by_groups_ax1_rejects_total_collision_before_custom_aggregation(storage):
    """Reject the ambiguous membership before a caller's aggregation can run."""
    frame = _mixed_panel(storage)
    groups = pd.Series({"A": "Total", "B": "Cash"})
    before_frame, before_groups = frame.copy(deep=True), groups.copy(deep=True)
    calls = []

    def observed_aggregation(row):
        calls.append(row.name)
        return 0.0

    with pytest.raises(ValueError, match="total_column.*group_data"):
        agg_df_by_groups_ax1(frame, groups, observed_aggregation, total_column="Total")

    assert calls == []
    pd.testing.assert_frame_equal(frame, before_frame, check_exact=True)
    pd.testing.assert_series_equal(groups, before_groups, check_exact=True)


@pytest.mark.parametrize("total_column", [None, "Portfolio"])
def test_get_group_dict_preserves_restricted_order_and_unobserved_total_placeholder(total_column):
    """A preferred-order placeholder is not an observed group and cannot overwrite a total."""
    groups = pd.Series({"A": "Bonds", "B": "Equity", "C": "Bonds"})
    selected = pd.Index(["C", "B", "absent", "A"], name="asset")
    before = selected.copy(deep=True)
    actual = qis.get_group_dict(
        groups, index_data=selected, group_order=["Portfolio", "Equity"], total_column=total_column
    )

    # Membership is literal: the unclassified ticker is excluded, and selected order wins.
    expected_keys = (
        ["Equity", "Bonds"] if total_column is None else ["Portfolio", "Equity", "Bonds"]
    )
    assert list(actual) == expected_keys
    assert actual["Equity"] == ["B"]
    assert actual["Bonds"] == ["C", "A"]
    if total_column is not None:
        pd.testing.assert_index_equal(actual["Portfolio"], pd.Index(["C", "B", "A"], name="asset"))
    pd.testing.assert_index_equal(selected, before)


@pytest.mark.parametrize("storage", ["ordinary", "nullable", "mixed"])
@pytest.mark.parametrize("total_column", [None, "All"])
def test_split_df_by_groups_preserves_distinct_total_and_no_total_panels(storage, total_column):
    """The guard must leave a normal group named Total usable when no conflicting total is added."""
    frame = _mixed_panel(storage)
    groups = pd.Series({"A": "Total", "B": "Cash"})
    before = frame.copy(deep=True)
    actual = qis.split_df_by_groups(frame, groups, total_column=total_column, group_order=["Total"])

    expected_keys = ["Total", "Cash"] if total_column is None else ["All", "Total", "Cash"]
    assert list(actual) == expected_keys
    pd.testing.assert_frame_equal(actual["Total"], frame[["A"]], check_exact=True)
    pd.testing.assert_frame_equal(actual["Cash"], frame[["B"]], check_exact=True)
    if total_column is not None:
        pd.testing.assert_frame_equal(actual["All"], frame, check_exact=True)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)


def test_get_grouped_long_short_weights_rejects_partial_total_and_preserves_full_total():
    """The public portfolio consumer must not report 60% as the total of 60%/40% holdings."""
    dates = pd.date_range("2026-01-01", periods=2, name="date")
    weights = pd.DataFrame({"A": [0.6, 0.6], "B": [0.4, 0.4]}, index=dates)
    groups = pd.Series({"A": "Total", "B": "Cash"})
    portfolio = qis.PortfolioData(
        nav=pd.Series(100.0, index=dates, name="Portfolio"),
        prices=pd.DataFrame(100.0, index=dates, columns=weights.columns),
        instrument_pnl=pd.DataFrame(0.0, index=dates, columns=weights.columns),
        weights=weights,
        group_data=groups,
    )
    before_weights, before_groups = weights.copy(deep=True), groups.copy(deep=True)

    with pytest.raises(ValueError, match="total_column.*group_data"):
        portfolio.get_grouped_long_short_weights(total_column="Total", weights_freq=None)

    actual, instruments = portfolio.get_grouped_long_short_weights(
        total_column="All", weights_freq=None
    )
    # Long-only .6 + .4 = 1.0; the subtotal remains .6, and there is no short exposure.
    expected = pd.DataFrame(
        {"Total": [1.0, 1.0], "Net Long": [1.0, 1.0], "Net Short": [0.0, 0.0]}, index=dates
    )
    pd.testing.assert_frame_equal(actual["All"], expected, check_exact=True)
    pd.testing.assert_series_equal(
        actual["Total"]["Total"], pd.Series(0.6, index=dates, name="Total")
    )
    pd.testing.assert_frame_equal(instruments["All"], weights, check_exact=True)
    pd.testing.assert_frame_equal(weights, before_weights, check_exact=True)
    pd.testing.assert_series_equal(groups, before_groups, check_exact=True)
