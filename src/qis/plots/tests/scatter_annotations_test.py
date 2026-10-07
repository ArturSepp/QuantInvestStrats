"""Keep scatter annotations attached to observations, not filtered row positions."""

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import qis  # noqa: E402
from qis.portfolio.smart_diversification.report import SmartDiversificationReport  # noqa: E402


def _assert_annotations(ax, expected):
    """Check point identity and styling after Matplotlib has rendered the artists."""
    ax.figure.canvas.draw()
    assert [(text.get_text(), text.xy) for text in ax.texts] == [
        (label, coordinates) for label, coordinates, _, _ in expected
    ]
    if not expected:
        return
    # The final collections are the annotation markers, not the underlying scatter cloud.
    markers = ax.collections[-len(expected) :]
    for text, collection, (_, coordinates, color, marker) in zip(ax.texts, markers, expected):
        assert to_rgba(text.get_color()) == to_rgba(color)
        np.testing.assert_array_equal(collection.get_offsets(), [coordinates])
        np.testing.assert_array_equal(collection.get_facecolors(), [to_rgba(color)])
        style = MarkerStyle(marker)
        reference = style.get_path().transformed(style.get_transform())
        np.testing.assert_array_equal(collection.get_paths()[0].vertices, reference.vertices)


@pytest.mark.parametrize("dtype", ("float64", "Float64"))
@pytest.mark.parametrize("missing_column", ("x", "y", "hue"))
@pytest.mark.parametrize("missing_position", (0, 2, 4))
def test_plot_scatter_preserves_annotation_identity_after_missing_row_filter(
    dtype, missing_column, missing_position
):
    """Missing first, interior or final points must not shift any annotation metadata."""
    frame = pd.DataFrame(
        {
            "x": pd.array([3, 1, 4, 2, 5], dtype=dtype),
            "y": pd.array([6, 2, 8, 4, 10], dtype=dtype),
            "hue": ["group"] * 5,
        },
        index=["duplicate", "duplicate", "third", "fourth", "fifth"],
    )
    frame.iloc[missing_position, frame.columns.get_loc(missing_column)] = (
        pd.NA if dtype == "Float64" else np.nan
    )
    before = frame.copy(deep=True)
    labels = ["A", "B", "C", "D", "E"]
    colors = ["red", "green", "blue", "purple", "orange"]
    markers = ["o", "s", "^", "D", "v"]
    # Literal original coordinates make the oracle independent of production filtering/sorting.
    expected = [
        (label, xy, color, marker)
        for position, (label, xy, color, marker) in enumerate(
            zip(labels, [(3, 6), (1, 2), (4, 8), (2, 4), (5, 10)], colors, markers)
        )
        if position != missing_position
    ]
    fig, ax = plt.subplots()
    try:
        result = qis.plot_scatter(
            frame,
            x="x",
            y="y",
            hue="hue" if missing_column == "hue" else None,
            order=1,
            full_sample_order=0,
            annotation_labels=labels,
            annotation_colors=colors,
            annotation_markers=markers,
            ax=ax,
        )
        assert result is None
        _assert_annotations(ax, expected)
        pd.testing.assert_frame_equal(frame, before)
        assert labels == ["A", "B", "C", "D", "E"]
        assert colors == ["red", "green", "blue", "purple", "orange"]
        assert markers == ["o", "s", "^", "D", "v"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("missing_column", ("x", "y"))
def test_plot_scatter_combines_hue_filtering_blank_labels_and_custom_styles(missing_column):
    """Group fitting and blank annotations must not change the surviving row's identity."""
    frame = pd.DataFrame({"x": [3, 1, 4, 2, 5], "y": [6, 2, 8, 4, 10], "hue": ["g"] * 5})
    frame.loc[1, missing_column] = np.nan
    fig, ax = plt.subplots()
    try:
        qis.plot_scatter(
            frame,
            x="x",
            y="y",
            hue="hue",
            order=1,
            full_sample_order=0,
            annotation_labels=["A", "B", "", "D", "E"],
            annotation_colors=["red", "green", "blue", "purple", "orange"],
            annotation_markers=["o", "s", "^", "D", "v"],
            ax=ax,
        )
        _assert_annotations(
            ax,
            [
                ("A", (3, 6), "red", "o"),
                ("D", (2, 4), "purple", "D"),
                ("E", (5, 10), "orange", "v"),
            ],
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("annotation_color", ("red", None))
def test_plot_scatter_preserves_default_annotation_styles(annotation_color):
    """Default styles are assigned to surviving points, including model-colour fallback."""
    frame = pd.DataFrame({"x": [1, np.nan, 3], "y": [2, 4, 6]})
    fig = qis.plot_scatter(
        frame,
        x="x",
        y="y",
        full_sample_order=0,
        annotation_labels=["A", "B", "C"],
        annotation_color=annotation_color,
    )
    try:
        color = "red" if annotation_color is not None else "blue"
        _assert_annotations(fig.axes[0], [("A", (1, 2), color, "o"), ("C", (3, 6), color, "o")])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("missing", (False, True))
def test_plot_scatter_preserves_short_metadata_truncation(missing):
    """Keep zip's established prefix behavior rather than introducing length validation."""
    frame = pd.DataFrame({"x": [1, np.nan if missing else 2, 3, 4], "y": [2, 4, 6, 8]})
    fig = qis.plot_scatter(
        frame,
        x="x",
        y="y",
        full_sample_order=0,
        annotation_labels=["A", "B", "C", "D"],
        annotation_colors=["red", "green", "blue"],
        annotation_markers=["o", "s"],
    )
    try:
        expected = [("A", (1, 2), "red", "o")]
        if not missing:
            expected.append(("B", (2, 4), "green", "s"))
        _assert_annotations(fig.axes[0], expected)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("annotations", (None, ["A", "B"]))
def test_plot_scatter_preserves_no_surviving_points(annotations):
    """Filtering every point still draws an empty scatter without extra annotation artists."""
    frame = pd.DataFrame({"x": [np.nan, np.nan], "y": [2, 4]})
    fig = qis.plot_scatter(frame, x="x", y="y", full_sample_order=0, annotation_labels=annotations)
    try:
        _assert_annotations(fig.axes[0], [])
    finally:
        plt.close(fig)


def test_plot_scatter_preserves_unfiltered_annotation_order():
    """Unsorted finite points retain their original labels and rendering order."""
    frame = pd.DataFrame({"x": [3, 1, 2], "y": [6, 2, 4]})
    fig = qis.plot_scatter(
        frame, x="x", y="y", full_sample_order=1, annotation_labels=["C", "A", "B"]
    )
    try:
        _assert_annotations(
            fig.axes[0],
            [("C", (3, 6), "red", "o"), ("A", (1, 2), "red", "o"), ("B", (2, 4), "red", "o")],
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("missing", (False, True))
def test_plot_scatter_preserves_annotation_order_after_wide_reshape(missing):
    """Apply filtering after native melting without inventing per-wide-row label expansion."""
    frame = pd.DataFrame({"x": [1, 2, 3], "a": [2, np.nan if missing else 4, 6], "b": [3, 6, 9]})
    before = frame.copy(deep=True)
    # Existing wide-form plotting consumes point rows column by column, not original wide rows.
    labels = ["a1", "a2", "a3", "b1", "b2", "b3"]
    expected = [("a1", (1, 2), "red", "o")]
    if not missing:
        expected.append(("a2", (2, 4), "red", "o"))
    expected.extend(
        [
            ("a3", (3, 6), "red", "o"),
            ("b1", (1, 3), "red", "o"),
            ("b2", (2, 6), "red", "o"),
            ("b3", (3, 9), "red", "o"),
        ]
    )
    fig = qis.plot_scatter(frame, x="x", order=1, full_sample_order=0, annotation_labels=labels)
    try:
        _assert_annotations(fig.axes[0], expected)
        pd.testing.assert_frame_equal(frame, before)
    finally:
        plt.close(fig)


def test_plot_smart_diversification_scatter_preserves_overlay_names(monkeypatch):
    """The real report delegator must retain fund names when one statistic is unavailable."""
    x_name, y_name = qis.PerfStat.VOL.to_str(), qis.PerfStat.SHARPE_RF0.to_str()
    points = pd.DataFrame(
        {x_name: [0.1, np.nan, 0.3, 0.4], y_name: [0.2, 0.4, 0.6, 0.8]},
        index=["Fund A", "Fund B", "Fund C", "Fund D"],
    )
    before = points.copy(deep=True)
    # Replace only analytics preparation: label creation, delegation and rendering remain real.
    report = object.__new__(SmartDiversificationReport)
    report.principal_nav = pd.Series([1.0, 1.1], name="Principal")
    monkeypatch.setattr(report, "get_overlay_points", lambda **kwargs: points)
    fig = report.plot_smart_diversification_scatter(
        x_var=qis.PerfStat.VOL,
        y_var=qis.PerfStat.SHARPE_RF0,
        full_sample_order=0,
        is_add_model_equation=False,
        ci=None,
    )
    try:
        _assert_annotations(
            fig.axes[0],
            [
                ("Fund A", (0.1, 0.2), "red", "o"),
                ("Fund C", (0.3, 0.6), "red", "o"),
                ("Fund D", (0.4, 0.8), "red", "o"),
            ],
        )
        pd.testing.assert_frame_equal(points, before)
    finally:
        plt.close(fig)
