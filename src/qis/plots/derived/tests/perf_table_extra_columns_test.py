"""Keep asset metadata attached to its asset through table preparation and rendering."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis


@pytest.fixture
def prices():
    """Literal endpoint returns are 10% and 5%, independent of table preparation."""
    return pd.DataFrame(
        [
            [100.0, 100.0],
            [101.0, 99.0],
            [99.0, 102.0],
            [103.0, 101.0],
            [102.0, 104.0],
            [106.0, 103.0],
            [104.0, 106.0],
            [110.0, 105.0],
        ],
        index=pd.bdate_range("2024-01-02", periods=8),
        columns=pd.Index(["A", "B"], name="Ticker"),
    )


def _table(prices, extra=None, is_to_str=True):
    return qis.get_ra_perf_columns(
        prices,
        perf_params=qis.PerfParams(freq="B"),
        perf_columns=[qis.PerfStat.TOTAL_RETURN],
        df_to_add=extra,
        is_to_str=is_to_str,
    )


@pytest.mark.parametrize("labels", [["A", "B"], [101, 202]])
@pytest.mark.parametrize("is_to_str", [False, True])
def test_get_ra_perf_columns_matches_extra_rows_by_asset(prices, labels, is_to_str):
    """Row permutation must not swap classifications or weights beside valid returns."""
    prices.columns = pd.Index(labels, name="Ticker")
    extra = pd.DataFrame(
        {"Class": ["Equity", "Bond"], "Weight": [0.75, 0.25]},
        index=pd.Index(labels, name="Metadata"),
    )
    reversed_extra = extra.iloc[::-1]
    original_prices = prices.copy(deep=True)
    original_extra = reversed_extra.copy(deep=True)
    result = _table(prices, reversed_extra, is_to_str)
    aligned = _table(prices, extra, is_to_str)
    stats = _table(prices, is_to_str=is_to_str)

    pd.testing.assert_index_equal(result.index, pd.Index(labels, name="Asset"))
    assert result.columns.tolist() == ["Class", "Weight", "Total"]
    assert result["Class"].tolist() == ["Equity", "Bond"]
    np.testing.assert_allclose(result["Weight"], [0.75, 0.25])
    if is_to_str:
        assert result["Total"].tolist() == ["10%", "5%"]
    else:
        np.testing.assert_allclose(result["Total"], [0.10, 0.05])
    pd.testing.assert_frame_equal(result, aligned)
    pd.testing.assert_frame_equal(result[["Total"]], stats)
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(reversed_extra, original_extra)


@pytest.mark.parametrize("is_to_str", [False, True])
def test_get_ra_perf_columns_retains_native_nullable_value_conversion(prices, is_to_str):
    """Alignment must not introduce a second dtype or missing-value conversion policy."""
    extra = pd.DataFrame(
        {
            "Class": pd.array(["Equity", pd.NA], dtype="string"),
            "Weight": pd.array([0.75, pd.NA], dtype="Float64"),
        },
        index=["A", "B"],
    )
    result = _table(prices, extra.iloc[::-1], is_to_str)
    expected = _table(prices, is_to_str=is_to_str)
    # The existing insertion consumes NumPy values, not extension-array dtypes.
    for position, name in enumerate(extra.columns):
        expected.insert(position, name, extra[name].to_numpy())
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("supplied_axes", [False, True])
def test_plot_ra_perf_table_draws_metadata_beside_the_correct_asset(prices, supplied_axes):
    """Assert cell coordinates: a bag of correct text would miss an asset-row swap."""
    extra = pd.DataFrame({"Class": ["Bond", "Equity"], "Weight": [0.25, 0.75]}, index=["B", "A"])
    original = extra.copy(deep=True)
    sentinel = plt.figure()
    fig = None
    try:
        if supplied_axes:
            fig, ax = plt.subplots()
        else:
            ax = None
        result = qis.plot_ra_perf_table(
            prices,
            perf_params=qis.PerfParams(freq="B"),
            perf_columns=[qis.PerfStat.TOTAL_RETURN],
            df_to_add=extra,
            ax=ax,
        )
        if supplied_axes:
            assert result is None
        else:
            fig = result
            ax = fig.axes[0]
        fig.canvas.draw()
        table = ax.tables[0]
        for row, values in enumerate(
            [["A", "Equity", "0.75", "10%"], ["B", "Bond", "0.25", "5%"]], start=1
        ):
            assert [table[row, col].get_text().get_text() for col in range(4)] == values
        assert [table[0, col].get_text().get_text() for col in range(1, 4)] == [
            "Class",
            "Weight",
            "Total",
        ]
        pd.testing.assert_frame_equal(extra, original)
        assert plt.fignum_exists(sentinel.number)
    finally:
        if fig is not None:
            plt.close(fig)
        plt.close(sentinel)


@pytest.mark.parametrize("labels", [["B", "C"], [0, 1], ["B", "B"]])
def test_get_ra_perf_columns_preserves_legacy_nonmatching_metadata(prices, labels):
    """Incomplete or ambiguous labels must not acquire a new join or blank-fill policy."""
    extra = pd.DataFrame({"Class": ["first", "second"]}, index=labels)
    result = _table(prices, extra)
    assert result["Class"].tolist() == ["first", "second"]
    assert result.index.tolist() == ["A", "B"]


@pytest.mark.parametrize("labels", [[], ["B"], ["A", "B", "C"]])
def test_get_ra_perf_columns_preserves_extra_row_count_rejection(prices, labels):
    """Legacy length errors stay errors instead of accepting partial or surplus metadata."""
    extra = pd.DataFrame({"Class": ["metadata"] * len(labels)}, index=labels)
    original = extra.copy(deep=True)
    with pytest.raises(ValueError, match="Length of values"):
        _table(prices, extra)
    pd.testing.assert_frame_equal(extra, original)


def test_get_ra_perf_columns_empty_extra_columns_remain_a_noop(prices):
    """An unused duplicate index must not start failing merely because alignment was added."""
    empty = pd.DataFrame(index=["B", "B"])
    pd.testing.assert_frame_equal(_table(prices, empty), _table(prices))


def test_get_ra_perf_columns_preserves_one_asset_series(prices):
    """The Series-to-frame path retains its single asset and the existing display convention."""
    extra = pd.DataFrame({"Class": ["Equity"]}, index=["A"])
    result = _table(prices["A"], extra)
    assert result.index.tolist() == ["A"]
    assert result.loc["A", "Class"] == "Equity"
    assert result.loc["A", "Total"] == "10%"


@pytest.mark.parametrize("labels", [[0, 1], ["A", "B"]])
@pytest.mark.parametrize("empty_columns", [False, True])
@pytest.mark.parametrize("is_to_str", [False, True])
def test_get_ra_perf_columns_preserves_tuple_asset_metadata_fallback(
    prices, labels, empty_columns, is_to_str
):
    """Tuple assets do not turn legacy positional metadata or no-ops into errors."""
    prices.columns = pd.MultiIndex.from_tuples([("A", "USD"), ("B", "USD")])
    extra = pd.DataFrame(index=labels)
    if not empty_columns:
        extra["Class"] = ["first", "second"]
    original_prices = prices.copy(deep=True)
    original_extra = extra.copy(deep=True)
    expected = _table(prices, is_to_str=is_to_str)
    if not empty_columns:
        # This schema has no matching asset keys; the accepted contract is positional.
        expected.insert(0, "Class", extra["Class"].to_numpy())

    result = _table(prices, extra, is_to_str)

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(extra, original_extra)


@pytest.mark.parametrize("flat_metadata_index", [False, True])
@pytest.mark.parametrize("is_to_str", [False, True])
def test_get_ra_perf_columns_aligns_complete_tuple_asset_keys(
    prices, flat_metadata_index, is_to_str
):
    """Complete tuple keys identify assets in either native pandas index representation."""
    labels = [("A", "USD"), ("B", "USD")]
    prices.columns = pd.MultiIndex.from_tuples(labels)
    metadata_index = (
        pd.Index(labels, tupleize_cols=False)
        if flat_metadata_index
        else pd.MultiIndex.from_tuples(labels)
    )
    extra = pd.DataFrame({"Class": ["Equity", "Bond"]}, index=metadata_index)
    reversed_extra = extra.iloc[::-1]
    original = reversed_extra.copy(deep=True)
    expected = _table(prices, is_to_str=is_to_str)
    expected.insert(0, "Class", extra["Class"].to_numpy())

    result = _table(prices, reversed_extra, is_to_str)

    pd.testing.assert_frame_equal(result, expected)
    assert result["Class"].tolist() == ["Equity", "Bond"]
    pd.testing.assert_frame_equal(reversed_extra, original)


def test_plot_ra_perf_table_draws_tuple_assets_with_positional_metadata(prices):
    """The public plotter must retain the getter's legacy tuple-label fallback."""
    prices.columns = pd.MultiIndex.from_tuples([("A", "USD"), ("B", "USD")])
    extra = pd.DataFrame({"Class": ["first", "second"]}, index=[0, 1])
    fig = qis.plot_ra_perf_table(
        prices,
        perf_params=qis.PerfParams(freq="B"),
        perf_columns=[qis.PerfStat.TOTAL_RETURN],
        df_to_add=extra,
    )
    try:
        fig.canvas.draw()
        table = fig.axes[0].tables[0]
        assert [[table[row, col].get_text().get_text() for col in range(3)] for row in [1, 2]] == [
            ["('A', 'USD')", "first", "10%"],
            ["('B', 'USD')", "second", "5%"],
        ]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "asset_labels, metadata_labels",
    [
        ([2**53, 2**53 + 1], [float(2**53), float(2**53 + 2)]),
        ([2**53 + 1, 2**53 + 3], [float(2**53 + 4), float(2**53)]),
        ([float(2**53 + 4), float(2**53)], [2**53 + 1, 2**53 + 3]),
        ([False, True], [1, 0]),
        ([1, 0], [False, True]),
        ([1.5, 2.5], pd.IntervalIndex.from_tuples([(0, 2), (1, 3)])),
    ],
)
@pytest.mark.parametrize("is_to_str", [False, True])
def test_get_ra_perf_columns_preserves_nonmatching_numeric_metadata(
    prices, asset_labels, metadata_labels, is_to_str
):
    """Rounded membership or lookup positions do not establish equal asset labels."""
    prices.columns = pd.Index(asset_labels)
    extra = pd.DataFrame(
        {"Class": ["Equity", "Bond"], "Weight": [0.75, 0.25]}, index=metadata_labels
    )
    original_prices, original_extra = prices.copy(deep=True), extra.copy(deep=True)
    expected = _table(prices, is_to_str=is_to_str)
    # These labels have no native full-key mapping; retain accepted positional insertion.
    for position, name in enumerate(extra.columns):
        expected.insert(position, name, extra[name].to_numpy())

    result = _table(prices, extra, is_to_str)

    pd.testing.assert_frame_equal(result, expected)
    assert result["Class"].tolist() == ["Equity", "Bond"]
    np.testing.assert_allclose(result["Weight"], [0.75, 0.25])
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(extra, original_extra)


@pytest.mark.parametrize(
    "asset_labels, metadata_labels",
    [
        ([2**53, 2**53 + 2], [float(2**53 + 2), float(2**53)]),
        ([False, True], [True, False]),
    ],
)
@pytest.mark.parametrize("is_to_str", [False, True])
def test_get_ra_perf_columns_aligns_exact_mixed_numeric_labels(
    prices, asset_labels, metadata_labels, is_to_str
):
    """Exact mixed numeric keys and same-kind boolean keys must still align."""
    prices.columns = pd.Index(asset_labels)
    extra = pd.DataFrame(
        {"Class": ["Bond", "Equity"], "Weight": [0.25, 0.75]},
        index=metadata_labels,
    )
    result = _table(prices, extra, is_to_str)
    expected = _table(prices, is_to_str=is_to_str)
    expected.insert(0, "Class", np.array(["Equity", "Bond"], dtype=object))
    expected.insert(1, "Weight", [0.75, 0.25])
    pd.testing.assert_frame_equal(result, expected)


def test_plot_ra_perf_table_preserves_nonmatching_numeric_metadata(prices):
    """A rounded key must not duplicate the first asset's metadata on the drawn table."""
    prices.columns = pd.Index([2**53, 2**53 + 1])
    extra = pd.DataFrame(
        {"Class": ["Equity", "Bond"], "Weight": [0.75, 0.25]},
        index=[float(2**53), float(2**53 + 2)],
    )
    fig = qis.plot_ra_perf_table(
        prices,
        perf_params=qis.PerfParams(freq="B"),
        perf_columns=[qis.PerfStat.TOTAL_RETURN],
        df_to_add=extra,
    )
    try:
        fig.canvas.draw()
        table = fig.axes[0].tables[0]
        assert [[table[row, col].get_text().get_text() for col in range(4)] for row in [1, 2]] == [
            ["9007199254740992", "Equity", "0.75", "10%"],
            ["9007199254740993", "Bond", "0.25", "5%"],
        ]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("overlapping", [False, True])
@pytest.mark.parametrize("empty_columns", [False, True])
@pytest.mark.parametrize("is_to_str", [False, True])
def test_get_ra_perf_columns_preserves_unrelated_interval_metadata(
    prices, overlapping, empty_columns, is_to_str
):
    """Unrelated interval keys remain positional, not a point-lookup request to pandas."""
    intervals = [(0, 2), (1, 3)] if overlapping else [(0, 1), (1, 2)]
    extra = pd.DataFrame(index=pd.IntervalIndex.from_tuples(intervals))
    if not empty_columns:
        extra["Class"] = ["first", "second"]
    original_prices, original_extra = prices.copy(deep=True), extra.copy(deep=True)
    expected = _table(prices, is_to_str=is_to_str)
    if not empty_columns:
        expected.insert(0, "Class", np.array(["first", "second"], dtype=object))

    result = _table(prices, extra, is_to_str)

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(extra, original_extra)


@pytest.mark.parametrize("empty_columns", [False, True])
def test_plot_ra_perf_table_preserves_unrelated_interval_metadata(prices, empty_columns):
    """The public plotter must preserve positional rows and unused-metadata no-ops."""
    extra = pd.DataFrame(index=pd.IntervalIndex.from_tuples([(0, 2), (1, 3)]))
    if not empty_columns:
        extra["Class"] = ["first", "second"]
    original = extra.copy(deep=True)
    fig = qis.plot_ra_perf_table(
        prices,
        perf_params=qis.PerfParams(freq="B"),
        perf_columns=[qis.PerfStat.TOTAL_RETURN],
        df_to_add=extra,
    )
    try:
        fig.canvas.draw()
        table = fig.axes[0].tables[0]
        expected = (
            [["A", "10%"], ["B", "5%"]]
            if empty_columns
            else [["A", "first", "10%"], ["B", "second", "5%"]]
        )
        assert [
            [table[row, col].get_text().get_text() for col in range(len(values))]
            for row, values in enumerate(expected, start=1)
        ] == expected
        pd.testing.assert_frame_equal(extra, original)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("is_to_str", [False, True])
def test_get_ra_perf_columns_aligns_complete_interval_asset_keys(prices, is_to_str):
    """Overlapping intervals can still identify complete assets, rather than numeric points."""
    prices.columns = pd.IntervalIndex.from_tuples([(0, 2), (1, 3)])
    extra = pd.DataFrame({"Class": ["Bond", "Equity"]}, index=prices.columns[::-1])
    original = extra.copy(deep=True)
    expected = _table(prices, is_to_str=is_to_str)
    expected.insert(0, "Class", np.array(["Equity", "Bond"], dtype=object))

    result = _table(prices, extra, is_to_str)

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(extra, original)


@pytest.mark.parametrize(
    "labels",
    [
        pd.Index([0, pd.NA], dtype="Int64"),
        pd.Index([0, pd.NA], dtype="UInt64"),
        pd.Index([0.0, pd.NA], dtype="Float64"),
        pd.Index([False, pd.NA], dtype="boolean"),
        pd.Index(["A", pd.NA], dtype="string"),
        pd.IntervalIndex.from_tuples([(0, 1), (np.nan, np.nan)]),
    ],
)
@pytest.mark.parametrize("is_to_str", [False, True])
def test_get_ra_perf_columns_aligns_native_missing_asset_keys(prices, labels, is_to_str):
    """Native missing-key matches survive the statistics owner's index normalization."""
    prices.columns = labels
    extra = pd.DataFrame({"Class": ["Bond", "Equity"], "Weight": [0.25, 0.75]}, index=labels[::-1])
    original_prices, original_extra = prices.copy(deep=True), extra.copy(deep=True)
    expected = _table(prices, is_to_str=is_to_str)
    # Preserve the native result axis, not the original input index's representation.
    expected.insert(0, "Class", np.array(["Equity", "Bond"], dtype=object))
    expected.insert(1, "Weight", [0.75, 0.25])

    result = _table(prices, extra, is_to_str)

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(extra, original_extra)


@pytest.mark.parametrize("missing_label", [None, pd.NA])
@pytest.mark.parametrize("is_to_str", [False, True])
def test_get_ra_perf_columns_respects_native_missing_label_aliases(
    prices, missing_label, is_to_str
):
    """Comparison-only normalization must not invent a native missing-label match."""
    prices.columns = pd.Index(["A", missing_label], dtype=object)
    extra = pd.DataFrame(
        {"Class": ["Bond", "Equity"], "Weight": [0.25, 0.75]}, index=prices.columns[::-1]
    )
    original = extra.copy(deep=True)
    expected = _table(prices, is_to_str=is_to_str)
    # Native reindex independently determines whether this alias matches the normalized
    # result axis on the installed pandas version. Literal nonmissing classes reveal misses.
    native = extra.reindex(expected.index)
    aligned = native["Class"].notna().all()
    expected.insert(
        0, "Class", np.array(["Equity", "Bond"] if aligned else ["Bond", "Equity"], dtype=object)
    )
    expected.insert(1, "Weight", [0.75, 0.25] if aligned else [0.25, 0.75])

    pd.testing.assert_frame_equal(_table(prices, extra, is_to_str), expected)
    pd.testing.assert_frame_equal(extra, original)


def test_plot_ra_perf_table_draws_native_missing_asset_keys(prices):
    """A missing asset key must not silently swap displayed classifications or weights."""
    prices.columns = pd.Index([0, pd.NA], dtype="Int64")
    extra = pd.DataFrame(
        {"Class": ["Bond", "Equity"], "Weight": [0.25, 0.75]}, index=prices.columns[::-1]
    )
    original = extra.copy(deep=True)
    fig = qis.plot_ra_perf_table(
        prices,
        perf_params=qis.PerfParams(freq="B"),
        perf_columns=[qis.PerfStat.TOTAL_RETURN],
        df_to_add=extra,
    )
    try:
        fig.canvas.draw()
        table = fig.axes[0].tables[0]
        assert [table[1, col].get_text().get_text() for col in range(4)] == [
            "0",
            "Equity",
            "0.75",
            "10%",
        ]
        assert [table[2, col].get_text().get_text() for col in range(1, 4)] == [
            "Bond",
            "0.25",
            "5%",
        ]
        pd.testing.assert_frame_equal(extra, original)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "labels",
    [
        [("A", "USD"), ("B", None)],
        [(None, "USD"), ("B", "EUR")],
        [("A", np.nan), ("B", "EUR")],
        [("A", pd.NA), ("B", "EUR")],
        [(None, "USD"), ("B", None)],
        [("A", "USD", None), ("B", None, "spot")],
    ],
)
@pytest.mark.parametrize("flat_metadata_index", [False, True])
@pytest.mark.parametrize("is_to_str", [False, True])
def test_get_ra_perf_columns_aligns_missing_tuple_components(
    prices, labels, flat_metadata_index, is_to_str
):
    """A missing tuple component retains native MultiIndex asset identity."""
    prices.columns = pd.MultiIndex.from_tuples(labels)
    metadata_index = (
        pd.Index(labels, tupleize_cols=False)
        if flat_metadata_index
        else pd.MultiIndex.from_tuples(labels)
    )
    extra = pd.DataFrame(
        {"Class": ["Equity", "Bond"], "Weight": [0.75, 0.25]}, index=metadata_index
    ).iloc[::-1]
    original_prices, original_extra = prices.copy(deep=True), extra.copy(deep=True)
    expected = _table(prices, is_to_str=is_to_str)
    expected.insert(0, "Class", np.array(["Equity", "Bond"], dtype=object))
    expected.insert(1, "Weight", [0.75, 0.25])

    pd.testing.assert_frame_equal(_table(prices, extra, is_to_str), expected)
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(extra, original_extra)


@pytest.mark.parametrize("flat_metadata_index", [False, True])
def test_plot_ra_perf_table_draws_missing_tuple_components(prices, flat_metadata_index):
    """Drawn classifications and weights follow full asset keys, including missing levels."""
    labels = [("A", "USD"), ("B", None)]
    prices.columns = pd.MultiIndex.from_tuples(labels)
    metadata_index = (
        pd.Index(labels, tupleize_cols=False)
        if flat_metadata_index
        else prices.columns
    )
    extra = pd.DataFrame(
        {"Class": ["Equity", "Bond"], "Weight": [0.75, 0.25]}, index=metadata_index
    ).iloc[::-1]
    fig = qis.plot_ra_perf_table(
        prices,
        perf_params=qis.PerfParams(freq="B"),
        perf_columns=[qis.PerfStat.TOTAL_RETURN],
        df_to_add=extra,
    )
    try:
        fig.canvas.draw()
        table = fig.axes[0].tables[0]
        assert [table[1, col].get_text().get_text() for col in [1, 2]] == ["Equity", "0.75"]
        assert [table[2, col].get_text().get_text() for col in [1, 2]] == ["Bond", "0.25"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "asset_labels, metadata_labels",
    [
        (
            [(None, 2**53 + 1), ("B", 2**53 + 3)],
            [("B", float(2**53 + 4)), (None, float(2**53))],
        ),
        (
            [(None, float(2**53 + 4)), ("B", float(2**53))],
            [("B", 2**53 + 1), (None, 2**53 + 3)],
        ),
        (
            [("A", None), ("B", "EUR")],
            [("C", "EUR"), ("A", None)],
        ),
    ],
)
@pytest.mark.parametrize("flat_metadata_index", [False, True])
@pytest.mark.parametrize("is_to_str", [False, True])
def test_get_ra_perf_columns_keeps_unmatched_missing_tuples_positional(
    prices, asset_labels, metadata_labels, flat_metadata_index, is_to_str
):
    """Missing-level normalization must not invent numeric matches or partial joins."""
    prices.columns = pd.MultiIndex.from_tuples(asset_labels)
    metadata_index = (
        pd.Index(metadata_labels, tupleize_cols=False)
        if flat_metadata_index
        else pd.MultiIndex.from_tuples(metadata_labels)
    )
    extra = pd.DataFrame({"Class": ["first", "second"]}, index=metadata_index)
    expected = _table(prices, is_to_str=is_to_str)
    expected.insert(0, "Class", extra["Class"].to_numpy())
    pd.testing.assert_frame_equal(_table(prices, extra, is_to_str), expected)


@pytest.mark.parametrize("flat_metadata_index", [False, True])
@pytest.mark.parametrize("is_to_str", [False, True])
def test_get_ra_perf_columns_aligns_three_asset_missing_tuples(
    prices, flat_metadata_index, is_to_str
):
    """A cyclic metadata permutation cannot depend on NaN identity in flattened keys."""
    labels = [("A", 1), ("B", None), ("C", 3)]
    prices = pd.concat([prices, prices["A"].rename("C")], axis=1, sort=False)
    prices.columns = pd.MultiIndex.from_tuples(labels)
    metadata_index = (
        pd.Index(labels, tupleize_cols=False)
        if flat_metadata_index
        else prices.columns
    )
    extra = pd.DataFrame(
        {"Class": ["Equity", "Bond", "Cash"], "Weight": [0.6, 0.3, 0.1]},
        index=metadata_index,
    ).iloc[[2, 0, 1]]
    expected = _table(prices, is_to_str=is_to_str)
    expected.insert(0, "Class", np.array(["Equity", "Bond", "Cash"], dtype=object))
    expected.insert(1, "Weight", [0.6, 0.3, 0.1])
    pd.testing.assert_frame_equal(_table(prices, extra, is_to_str), expected)
