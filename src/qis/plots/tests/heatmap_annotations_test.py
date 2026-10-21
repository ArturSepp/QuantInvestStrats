"""Keep cell text attached to its observation when a heatmap changes orientation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from qis import plot_heatmap
from qis.plots.derived import returns_heatmap


def _assert_cells(ax, values, text, columns, rows):
    """Check independent color values, visible text positions, and displayed labels together."""
    ax.figure.canvas.draw()
    np.testing.assert_allclose(ax.collections[0].get_array().filled(np.nan), values)
    expected = {
        (column + 0.5, row + 0.5): label
        for row, labels in enumerate(text)
        for column, label in enumerate(labels)
        if label is not None
    }
    assert {item.get_position(): item.get_text() for item in ax.texts} == expected
    assert [item.get_text() for item in ax.get_xticklabels()] == columns
    assert [item.get_text() for item in ax.get_yticklabels()] == rows


@pytest.mark.parametrize(
    "transpose,inverse,values,text,columns,rows",
    [
        (
            False,
            False,
            [[0.1, 0.2, 0.3], [0.4, np.nan, 0.6]],
            [["A-Jan", "A-Feb", "A-Mar"], ["B-Jan", None, "B-Mar"]],
            ["Jan", "Feb", "Mar"],
            ["A", "B"],
        ),
        (
            False,
            True,
            [[0.4, np.nan, 0.6], [0.1, 0.2, 0.3]],
            [["B-Jan", None, "B-Mar"], ["A-Jan", "A-Feb", "A-Mar"]],
            ["Jan", "Feb", "Mar"],
            ["B", "A"],
        ),
        (
            True,
            False,
            [[0.1, 0.4], [0.2, np.nan], [0.3, 0.6]],
            [["A-Jan", "B-Jan"], ["A-Feb", None], ["A-Mar", "B-Mar"]],
            ["A", "B"],
            ["Jan", "Feb", "Mar"],
        ),
        # Transpose has always taken precedence over inverse; do not compose a new reversal.
        (
            True,
            True,
            [[0.1, 0.4], [0.2, np.nan], [0.3, 0.6]],
            [["A-Jan", "B-Jan"], ["A-Feb", None], ["A-Mar", "B-Mar"]],
            ["A", "B"],
            ["Jan", "Feb", "Mar"],
        ),
    ],
)
@pytest.mark.parametrize("annotation_frame", [False, True])
def test_plot_heatmap_orients_rectangular_annotations_with_cells(
    transpose, inverse, values, text, columns, rows, annotation_frame
):
    data = pd.DataFrame(
        [[0.1, 0.2, 0.3], [0.4, np.nan, 0.6]], index=["A", "B"], columns=["Jan", "Feb", "Mar"]
    )
    labels = np.array([["A-Jan", "A-Feb", "A-Mar"], ["B-Jan", "masked", "B-Mar"]])
    labels.setflags(write=False)
    # Seaborn matches annotations by position, not by a DataFrame's index or column labels.
    annot = pd.DataFrame(labels, index=["other", "labels"]) if annotation_frame else labels
    before_data, before_labels = data.copy(deep=True), labels.copy()
    before_annot = annot.copy(deep=True) if annotation_frame else None
    fig, ax = plt.subplots()
    try:
        assert (
            plot_heatmap(
                data, transpose=transpose, inverse=inverse, annot=annot, var_format=None, ax=ax
            )
            is None
        )
        _assert_cells(ax, values, text, columns, rows)
        pd.testing.assert_frame_equal(data, before_data)
        np.testing.assert_array_equal(labels, before_labels)
        if annotation_frame:
            pd.testing.assert_frame_equal(annot, before_annot)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "index,rows",
    [
        (["A", "A"], ["A", "A"]),
        (["A", "B", "A"], ["A", "B", "A"]),
        (pd.to_datetime(["2020-01-31", "2020-02-29"]), ["2020", "2020"]),
    ],
)
@pytest.mark.parametrize("annotation_frame", [False, True])
def test_plot_heatmap_inverse_preserves_native_repeated_label_order(index, rows, annotation_frame):
    """Native reindex is a no-op for palindromic labels, including formatted-date collisions."""
    values = [[0.1, 0.2], [0.3, 0.4]]
    text = [["first-Jan", "first-Feb"], ["second-Jan", "second-Feb"]]
    if len(index) == 3:
        values.append([0.5, 0.6])
        text.append(["third-Jan", "third-Feb"])
    data = pd.DataFrame(values, index=index, columns=["Jan", "Feb"])
    labels = np.array(text)
    labels.setflags(write=False)
    # Annotation-frame labels remain irrelevant: only the data's native reindex selects rows.
    annot = pd.DataFrame(labels) if annotation_frame else labels
    before_data = data.copy(deep=True)
    before_annot = annot.copy(deep=True) if annotation_frame else labels.copy()
    fig, ax = plt.subplots()
    try:
        plot_heatmap(data, inverse=True, annot=annot, var_format=None, ax=ax)
        _assert_cells(ax, values, text, ["Jan", "Feb"], rows)
        pd.testing.assert_frame_equal(data, before_data)
        if annotation_frame:
            pd.testing.assert_frame_equal(annot, before_annot)
        else:
            np.testing.assert_array_equal(annot, before_annot)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "index,values,text,rows",
    [
        (
            ["A", "A"],
            [[0.1, 0.2], [0.3, 0.4]],
            [["2020-01-01", "2020-01-02"], ["2020-01-03", "2020-01-04"]],
            ["A", "A"],
        ),
        (
            ["A", "B"],
            [[0.3, 0.4], [0.1, 0.2]],
            [["2020-01-03", "2020-01-04"], ["2020-01-01", "2020-01-02"]],
            ["B", "A"],
        ),
    ],
)
def test_plot_heatmap_inverse_preserves_date_annotation_representation(index, values, text, rows):
    """Reordering labels must not add time-of-day text by normalizing their NumPy date unit."""
    data = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=index, columns=["Jan", "Feb"])
    labels = np.array(
        [["2020-01-01", "2020-01-02"], ["2020-01-03", "2020-01-04"]], dtype="datetime64[D]"
    )
    labels.setflags(write=False)
    before_data, before_labels = data.copy(deep=True), labels.copy()
    fig, ax = plt.subplots()
    try:
        plot_heatmap(data, inverse=True, annot=labels, var_format=None, ax=ax)
        _assert_cells(ax, values, text, ["Jan", "Feb"], rows)
        pd.testing.assert_frame_equal(data, before_data)
        np.testing.assert_array_equal(labels, before_labels)
        assert labels.dtype == np.dtype("datetime64[D]")
    finally:
        plt.close(fig)


def test_plot_heatmap_inverse_preserves_native_duplicate_reindex_rejection():
    """A non-palindromic duplicate index still reaches pandas' existing rejection."""
    data = pd.DataFrame([[0.1], [0.2], [0.3]], index=["A", "A", "B"])
    before = data.copy(deep=True)
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="cannot reindex on an axis with duplicate labels"):
            plot_heatmap(data, inverse=True, annot=np.array([["one"], ["two"], ["three"]]), ax=ax)
        pd.testing.assert_frame_equal(data, before)
    finally:
        plt.close(fig)


def test_plot_heatmap_transposes_square_numeric_annotations():
    """A square array can render successfully while silently swapping off-diagonal text."""
    data = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=["A", "B"], columns=["Jan", "Feb"])
    fig = plot_heatmap(
        data, transpose=True, annot=np.array([[10, 20], [30, 40]]), var_format="{:.0f}"
    )
    try:
        _assert_cells(
            fig.axes[0],
            [[0.1, 0.3], [0.2, 0.4]],
            [["10", "30"], ["20", "40"]],
            ["A", "B"],
            ["Jan", "Feb"],
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("annotation_frame", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_plot_heatmap_preserves_display_shaped_rectangular_annotations(annotation_frame, inverse):
    """Existing callers may already have arranged rectangular labels for the transposed grid."""
    data = pd.DataFrame(
        [[0.1, 0.2, 0.3], [0.4, np.nan, 0.6]], index=["A", "B"], columns=["Jan", "Feb", "Mar"]
    )
    # These labels are intentionally display-shaped, not paired with the input frame's positions.
    labels = np.array([["A-Jan", "B-Jan"], ["A-Feb", "masked"], ["A-Mar", "B-Mar"]])
    labels.setflags(write=False)
    annot = pd.DataFrame(labels) if annotation_frame else labels
    before_data, before_labels = data.copy(deep=True), labels.copy()
    before_annot = annot.copy(deep=True) if annotation_frame else None
    fig, ax = plt.subplots()
    try:
        plot_heatmap(data, transpose=True, inverse=inverse, annot=annot, var_format=None, ax=ax)
        _assert_cells(
            ax,
            [[0.1, 0.4], [0.2, np.nan], [0.3, 0.6]],
            [["A-Jan", "B-Jan"], ["A-Feb", None], ["A-Mar", "B-Mar"]],
            ["A", "B"],
            ["Jan", "Feb", "Mar"],
        )
        pd.testing.assert_frame_equal(data, before_data)
        np.testing.assert_array_equal(labels, before_labels)
        if annotation_frame:
            pd.testing.assert_frame_equal(annot, before_annot)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("annot", [True, False, None])
def test_plot_heatmap_preserves_automatic_and_disabled_annotations(annot):
    data = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=["A", "B"], columns=["Jan", "Feb"])
    fig, ax = plt.subplots()
    try:
        plot_heatmap(data, transpose=True, inverse=True, annot=annot, ax=ax)
        text = [["10.0%", "30.0%"], ["20.0%", "40.0%"]] if annot is True else [[None] * 2] * 2
        _assert_cells(ax, [[0.1, 0.3], [0.2, 0.4]], text, ["A", "B"], ["Jan", "Feb"])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("annot", [np.array(1), np.bool_(True), [1, 2], [[1]]])
@pytest.mark.parametrize("transpose,inverse", [(True, False), (False, True)])
def test_plot_heatmap_preserves_native_invalid_annotation_diagnostics(annot, transpose, inverse):
    """Non-matrix annotations must still reach seaborn's shape diagnostic, not slicing errors."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="`data` and `annot` must have same shape"):
            plot_heatmap(
                pd.DataFrame([[0.1, 0.2], [0.3, 0.4]]),
                annot=annot,
                transpose=transpose,
                inverse=inverse,
                ax=ax,
            )
    finally:
        plt.close(fig)


def test_plot_returns_heatmap_transposes_independently_scaled_annotations(monkeypatch):
    """The caller colors scaled returns but must label each cell with its unscaled return."""
    table = pd.DataFrame(
        [[0.01, 0.02, 0.20], [0.04, np.nan, 0.40]],
        index=["2024", "2025"],
        columns=["Jan", "Feb", "YTD"],
    )
    before = table.copy(deep=True)
    # Fix analytics independently: this regression protects the formatting-to-renderer boundary.
    monkeypatch.setattr(returns_heatmap, "compute_periodic_returns_table", lambda **kwargs: table)
    fig, ax = plt.subplots()
    try:
        returns_heatmap.plot_returns_heatmap(
            pd.Series(dtype=float), is_ytd_color_scale_independent=True, transpose=True, ax=ax
        )
        _assert_cells(
            ax,
            [[0.25, 1.0], [0.5, np.nan], [0.5, 1.0]],
            [["1.0%", "4.0%"], ["2.0%", None], ["20.0%", "40.0%"]],
            ["2024", "2025"],
            ["Jan", "Feb", "YTD"],
        )
        pd.testing.assert_frame_equal(table, before)
    finally:
        plt.close(fig)
