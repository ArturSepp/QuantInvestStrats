"""Transposed tables retain every metric, entity label and native display option."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.plots.derived import perf_table


pytestmark = pytest.mark.filterwarnings("error")


def _cell_text(ax):
    """Read the complete grid, so an attractive but incomplete table cannot pass."""
    return {key: cell.get_text().get_text() for key, cell in ax.tables[0].get_celld().items()}


@pytest.mark.parametrize("add_index_as_column", [True, False])
def test_plot_df_table_transpose_keeps_first_metric_and_asset_headers(add_index_as_column):
    """The first statistic is data, not a replacement for the asset names."""
    frame = pd.DataFrame({"Return": ["10%", "20%"], "Sharpe": ["1.1", "2.2"]}, index=["A", "B"])
    snapshot = frame.copy(deep=True)
    fig, ax = plt.subplots()
    try:
        assert (
            qis.plot_df_table(frame, transpose=True, add_index_as_column=add_index_as_column, ax=ax)
            is None
        )
        fig.canvas.draw()
        # These literal coordinates do not reuse the renderer's transpose or header logic.
        if add_index_as_column:
            expected = {
                (0, 0): " ",
                (0, 1): "A",
                (0, 2): "B",
                (1, 0): "Return",
                (1, 1): "10%",
                (1, 2): "20%",
                (2, 0): "Sharpe",
                (2, 1): "1.1",
                (2, 2): "2.2",
            }
        else:
            expected = {
                (0, 0): "A",
                (0, 1): "B",
                (1, 0): "10%",
                (1, 1): "20%",
                (2, 0): "1.1",
                (2, 1): "2.2",
            }
        assert _cell_text(ax) == expected
        pd.testing.assert_frame_equal(frame, snapshot)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("heatmap", [False, True])
@pytest.mark.parametrize(
    "caption,index,headers,hide_headers",
    [
        ("Asset", pd.Index(["Asset", "B"]), ["Asset", "B"], False),
        (" ", pd.Index([" ", "B"]), [" ", "B"], False),
        (
            "Asset",
            pd.MultiIndex.from_tuples([("Asset", ""), ("B", "y")]),
            ["('Asset', '')", "('B', 'y')"],
            False,
        ),
        ("Asset", pd.Index(["Asset", "B"]), ["Asset", "B"], True),
    ],
)
def test_plot_df_table_transpose_separates_caption_from_entity_labels(
    caption, index, headers, hide_headers, heatmap
):
    """A display caption is not a reserved data label, even during numeric formatting."""
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=index, columns=["Return", "Risk"])
    snapshot = frame.copy(deep=True)
    fig, ax = plt.subplots()
    try:
        qis.plot_df_table(
            frame,
            transpose=True,
            index_column_name=caption,
            var_format="{:.1f}",
            heatmap_columns=[1, 2] if heatmap else None,
            emply_column_names=hide_headers,
            ax=ax,
        )
        fig.canvas.draw()
        expected_headers = ["", ""] if hide_headers else headers
        assert _cell_text(ax) == {
            (0, 0): caption,
            (0, 1): expected_headers[0],
            (0, 2): expected_headers[1],
            (1, 0): "Return",
            (1, 1): "1.0",
            (1, 2): "3.0",
            (2, 0): "Risk",
            (2, 1): "2.0",
            (2, 2): "4.0",
        }
        if heatmap:
            # Each asset's Risk is its column maximum; the caption is not another observation.
            for column in (1, 2):
                for row, scale in ((1, 0.0), (2, 0.75)):
                    np.testing.assert_allclose(
                        ax.tables[0][row, column].get_facecolor(), plt.get_cmap("RdYlGn")(scale)
                    )
        pd.testing.assert_frame_equal(frame, snapshot)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("add_index_as_column", [False, True])
@pytest.mark.parametrize("axis,caption", [(None, " "), (0, " "), (1, " "), (1, "A")])
def test_plot_df_table_with_ci_transpose_keeps_estimates_and_scaled_colors_together(
    axis, caption, add_index_as_column, transpose
):
    """Move colours with estimates without recomputing their original row/column scale."""
    frame = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], index=["A", "B"], columns=["one", "two", "three"]
    )
    ci = frame / 10
    snapshot, ci_snapshot = frame.copy(deep=True), ci.copy(deep=True)
    # Independent normalizations: (estimate - minimum) / range * the native 0.75 colour scale.
    scales = {
        None: [[0.0, 0.15, 0.30], [0.45, 0.60, 0.75]],
        0: [[0.0, 0.0, 0.0], [0.75, 0.75, 0.75]],
        1: [[0.0, 0.375, 0.75], [0.0, 0.375, 0.75]],
    }[axis]
    if transpose:
        headings, names = ["A", "B"], ["one", "two", "three"]
        rows = [["1.00±0.10", "4.00±0.40"], ["2.00±0.20", "5.00±0.50"], ["3.00±0.30", "6.00±0.60"]]
    else:
        headings, names = ["one", "two", "three"], ["A", "B"]
        rows = [["1.00±0.10", "2.00±0.20", "3.00±0.30"], ["4.00±0.40", "5.00±0.50", "6.00±0.60"]]
    offset = int(add_index_as_column)
    grid = [([caption] if add_index_as_column else []) + headings]
    grid += [([name] if add_index_as_column else []) + row for name, row in zip(names, rows)]
    fig, ax = plt.subplots()
    try:
        qis.plot_df_table_with_ci(
            frame,
            ci,
            axis=axis,
            transpose=transpose,
            add_index_as_column=add_index_as_column,
            index_column_name=caption,
            ax=ax,
        )
        fig.canvas.draw()
        assert _cell_text(ax) == {
            (row, col): text for row, texts in enumerate(grid) for col, text in enumerate(texts)
        }
        for row, texts in enumerate(rows):
            for col in range(len(texts)):
                scale = scales[col][row] if transpose else scales[row][col]
                np.testing.assert_allclose(
                    ax.tables[0][row + 1, col + offset].get_facecolor(),
                    plt.get_cmap("RdYlGn")(scale),
                )
        pd.testing.assert_frame_equal(frame, snapshot)
        pd.testing.assert_frame_equal(ci, ci_snapshot)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("transpose", [False, True])
def test_plot_df_table_with_ci_transpose_keeps_heatmap_disabled(transpose):
    """Disabling estimate colours keeps ordinary row banding, rather than a second palette."""
    frame = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], index=["A", "B"], columns=["one", "two", "three"]
    )
    fig, ax = plt.subplots()
    try:
        qis.plot_df_table_with_ci(
            frame,
            frame / 10,
            transpose=transpose,
            is_add_heatmap=False,
            row_colors=["white"],
            ax=ax,
        )
        fig.canvas.draw()
        assert len(_cell_text(ax)) == 12
        for (row, _), cell in ax.tables[0].get_celld().items():
            if row > 0:
                np.testing.assert_allclose(cell.get_facecolor(), (1.0, 1.0, 1.0, 1.0))
    finally:
        plt.close(fig)


@pytest.mark.parametrize("add_index_as_column", [True, False])
@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(
            pd.DataFrame(
                [[1, 2], [3, 4], [5, 6]], index=["A", "B", "C"], columns=["Return", "Sharpe"]
            ),
            id="rectangular",
        ),
        pytest.param(pd.DataFrame({"Return": ["10%", "20%"]}, index=["A", "B"]), id="one-metric"),
        pytest.param(pd.Series(["10%", "20%"], index=["A", "B"], name="Return"), id="series"),
        pytest.param(
            pd.DataFrame([[1, 2], [3, 4]], index=["A", "A"], columns=["Return", "Return"]),
            id="duplicate-labels",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "complete": pd.array([1.0, 2.0], dtype="Float64"),
                    "ragged": pd.array([pd.NA, 3.0], dtype="Float64"),
                    "missing": pd.array([pd.NA, pd.NA], dtype="Float64"),
                },
                index=["A", "B"],
            ),
            id="nullable-mixed-panel",
        ),
        pytest.param(
            pd.DataFrame([[1.0, np.nan], [2.0, 3.0]], index=[10, 20], columns=[30, 40]),
            id="numeric-labels-and-nan",
        ),
        pytest.param(
            pd.DataFrame(
                [[1, 2], [3, 4]],
                index=pd.date_range("2020-01-01", periods=2, name="date"),
                columns=pd.MultiIndex.from_tuples([("X", 1), ("Y", 2)]),
            ),
            id="datetime-and-tuple-labels",
        ),
    ],
)
def test_plot_df_table_transpose_matches_native_frame_transpose(frame, add_index_as_column):
    """Native pandas transpose defines normalization; original dtypes need not survive it."""
    snapshot = frame.copy(deep=True)
    native_frame = frame.to_frame() if isinstance(frame, pd.Series) else frame
    expected_frame = native_frame.T
    figs = []
    try:
        # Also cover the figure-creating path, rather than only caller-supplied axes.
        actual = qis.plot_df_table(frame, transpose=True, add_index_as_column=add_index_as_column)
        figs.append(actual)
        expected = qis.plot_df_table(expected_frame, add_index_as_column=add_index_as_column)
        figs.append(expected)
        actual.canvas.draw()
        expected.canvas.draw()
        assert _cell_text(actual.axes[0]) == _cell_text(expected.axes[0])
        np.testing.assert_allclose(actual.get_size_inches(), expected.get_size_inches())
        if isinstance(frame, pd.Series):
            pd.testing.assert_series_equal(frame, snapshot)
        else:
            pd.testing.assert_frame_equal(frame, snapshot)
    finally:
        for fig in figs:
            plt.close(fig)


@pytest.mark.parametrize("add_index_as_column", [True, False])
def test_plot_df_table_transpose_preserves_rendered_cell_styling(add_index_as_column):
    """Display coordinates still refer to the transposed grid, including coloured cells."""
    frame = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"], columns=["Return", "Sharpe"])
    colors = [["red", "green"], ["blue", "yellow"]]
    options = dict(
        add_index_as_column=add_index_as_column,
        data_colors=colors,
        var_format="{:.2f}",
        left_aligned_first_col=True,
        first_column_width=4.0,
        first_row_height=0.2,
        special_rows_colors=[(1, "cyan")],
        rotation_for_columns_headers=15,
    )
    fig, axes = plt.subplots(1, 2)
    try:
        qis.plot_df_table(frame, transpose=True, ax=axes[0], **options)
        qis.plot_df_table(frame.T, ax=axes[1], **options)
        # Alignment inspects properties before drawing; compare geometry at that stage too.
        for key, cell in axes[0].tables[0].get_celld().items():
            expected = axes[1].tables[0].get_celld()[key]
            assert cell.get_height() == pytest.approx(expected.get_height())
            assert cell.get_width() == pytest.approx(expected.get_width())
        fig.canvas.draw()
        assert _cell_text(axes[0]) == _cell_text(axes[1])
        for key, cell in axes[0].tables[0].get_celld().items():
            expected = axes[1].tables[0].get_celld()[key]
            np.testing.assert_allclose(cell.get_facecolor(), expected.get_facecolor())
            assert cell.get_text().get_ha() == expected.get_text().get_ha()
            assert cell.get_text().get_rotation() == expected.get_text().get_rotation()
        assert colors == [["red", "green"], ["blue", "yellow"]]
    finally:
        plt.close(fig)


def test_plot_ra_perf_table_transpose_retains_prepared_statistics(monkeypatch):
    """The public wrapper must not lose its first prepared statistic or asset identities."""
    prepared = pd.DataFrame(
        {"Total": ["10%", "20%"], "Sharpe (rf=0)": ["1.1", "2.2"]}, index=["A", "B"]
    )
    snapshot = prepared.copy(deep=True)
    # Fix the analytics-to-renderer boundary: this test concerns display, not Sharpe estimation.
    monkeypatch.setattr(perf_table, "get_ra_perf_columns", lambda **kwargs: prepared)
    fig, ax = plt.subplots()
    try:
        assert (
            qis.plot_ra_perf_table(
                pd.DataFrame(),
                transpose=True,
                ax=ax,
                perf_columns=[qis.PerfStat.TOTAL_RETURN, qis.PerfStat.SHARPE_RF0],
            )
            is None
        )
        fig.canvas.draw()
        assert _cell_text(ax) == {
            (0, 0): " ",
            (0, 1): "A",
            (0, 2): "B",
            (1, 0): "Total",
            (1, 1): "10%",
            (1, 2): "20%",
            (2, 0): "Sharpe (rf=0)",
            (2, 1): "1.1",
            (2, 2): "2.2",
        }
        pd.testing.assert_frame_equal(prepared, snapshot)
    finally:
        plt.close(fig)
