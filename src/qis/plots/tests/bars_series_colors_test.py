"""Keep grouped-bar colours attached to their series when observations are missing."""

from colorsys import hls_to_rgb, rgb_to_hls

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb, to_rgba  # noqa: E402

import qis  # noqa: E402
from qis.plots.utils import get_n_colors  # noqa: E402


def _bar_rgba(color, *, seaborn=True):
    """Apply Seaborn's documented default saturation, independently of its hue selection."""
    rgb = to_rgb(color)
    if seaborn:
        hue, lightness, saturation = rgb_to_hls(*rgb)
        rgb = hls_to_rgb(hue, lightness, 0.75 * saturation)
    return (*rgb, 0.9 if seaborn else 1.0)


@pytest.mark.parametrize("missing_column", ["A", "B", "C"])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("default_colors", [False, True])
def test_plot_bars_keeps_original_series_colors_after_missing_values_are_removed(
    missing_column, dtype, default_colors
):
    """Missing first/interior/last series cannot steal a surviving neighbour's colour."""
    # Distinct heights identify the source series without consulting Seaborn's hue order.
    frame = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, np.nan], "C": [5.0, 6.0]}, dtype=dtype)
    frame[missing_column] = pd.Series([np.nan, np.nan], dtype=dtype)
    original = frame.copy(deep=True)
    colors = get_n_colors(3) if default_colors else ["red", "green", "blue"]
    original_colors = list(colors)
    expected = {
        float(value): _bar_rgba(colors[column])
        for column in range(3)
        for value in frame.iloc[:, column]
        if pd.notna(value)
    }
    fig, ax = plt.subplots()
    try:
        assert (
            qis.plot_bars(frame, stacked=False, colors=None if default_colors else colors, ax=ax)
            is None
        )
        fig.canvas.draw()
        bars = [patch for patch in ax.patches if patch.get_height() in expected]
        assert len(bars) == len(expected)
        for bar in bars:
            np.testing.assert_allclose(bar.get_facecolor(), expected[bar.get_height()])
            # Do not reserve empty hue slots: only the two available series share each group.
            np.testing.assert_allclose(bar.get_width(), 0.4)
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ["A", "B", "C"]
        for line, color in zip(ax.get_legend().get_lines(), colors):
            np.testing.assert_allclose(to_rgba(line.get_color()), to_rgba(color))
        pd.testing.assert_frame_equal(frame, original)
        assert colors == original_colors
    finally:
        plt.close(fig)


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_plot_bars_preserves_all_missing_series_legend_without_drawing_values(dtype):
    """An empty observed hue set still retains the original legend, not fabricated bars."""
    frame = pd.DataFrame(np.nan, index=[0, 1], columns=["A", "B", "C"], dtype=dtype)
    fig, ax = plt.subplots()
    try:
        qis.plot_bars(frame, stacked=False, colors=["red", "green", "blue"], ax=ax)
        fig.canvas.draw()
        assert not ax.patches
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ["A", "B", "C"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(("is_sns", "stacked"), [(True, False), (False, False), (True, True)])
def test_plot_bars_preserves_series_colors_in_healthy_renderer_controls(is_sns, stacked):
    """Healthy grouped bars and untouched pandas/stacked paths retain their colour policy."""
    frame = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]})
    colors = ["red", "green", "blue"]
    expected = {
        float(value): _bar_rgba(colors[column], seaborn=is_sns and not stacked)
        for column in range(3)
        for value in frame.iloc[:, column]
    }
    fig, ax = plt.subplots()
    try:
        qis.plot_bars(frame, stacked=stacked, is_sns=is_sns, colors=colors, ax=ax)
        fig.canvas.draw()
        bars = [patch for patch in ax.patches if patch.get_height() in expected]
        assert len(bars) == 6
        for bar in bars:
            np.testing.assert_allclose(bar.get_facecolor(), expected[bar.get_height()])
    finally:
        plt.close(fig)


def test_plot_bars_keeps_explicit_legend_overrides_separate_from_series_colors():
    """Custom legend swatches are intentional and must not recolour the plotted values."""
    frame = pd.DataFrame({"A": [np.nan, np.nan], "B": [2.0, 3.0], "C": [4.0, 5.0]})
    colors, legend_colors = ["red", "green", "blue"], ["black", "cyan", "magenta"]
    fig, ax = plt.subplots()
    try:
        qis.plot_bars(
            frame,
            stacked=False,
            colors=colors,
            legend_colors=legend_colors,
            legend_labels=["Absent", "Second", "Third"],
            reverse_columns=True,
            ax=ax,
        )
        fig.canvas.draw()
        for bar in ax.patches:
            if bar.get_height() in (2.0, 3.0, 4.0, 5.0):
                color = "green" if bar.get_height() in (2.0, 3.0) else "blue"
                np.testing.assert_allclose(bar.get_facecolor(), _bar_rgba(color))
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "Third",
            "Second",
            "Absent",
        ]
        assert [line.get_color() for line in ax.get_legend().get_lines()] == legend_colors[::-1]
        assert colors == ["red", "green", "blue"]
        assert legend_colors == ["black", "cyan", "magenta"]
    finally:
        plt.close(fig)


def test_plot_bars_preserves_short_palette_cycling():
    """An incomplete palette keeps the established Seaborn fallback, not a missing-key error."""
    frame = pd.DataFrame({"A": [np.nan, np.nan], "B": [2.0, 3.0], "C": [4.0, 5.0]})
    fig, ax = plt.subplots()
    try:
        with pytest.warns(UserWarning, match="palette list has fewer values"):
            qis.plot_bars(frame, stacked=False, colors=["red"], ax=ax)
        fig.canvas.draw()
        for bar in ax.patches:
            if bar.get_height() in (2.0, 3.0, 4.0, 5.0):
                np.testing.assert_allclose(bar.get_facecolor(), _bar_rgba("red"))
    finally:
        plt.close(fig)


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("all_missing", [False, True])
@pytest.mark.parametrize("unused_category", [None, "D", 0])
def test_plot_bars_preserves_categorical_metadata_and_original_series_colors(
    dtype, all_missing, unused_category
):
    """Unused categories retain native spacing without stealing a real series's colour."""
    categories = ["C", "B", "A"] if unused_category is None else [unused_category, "C", "B", "A"]
    columns = pd.CategoricalIndex(["A", "B", "C"], categories=categories)
    frame = pd.DataFrame([[np.nan, 2.0, 4.0], [np.nan, 3.0, 5.0]], columns=columns, dtype=dtype)
    if all_missing:
        frame.iloc[:, :] = np.nan
    original = frame.copy(deep=True)
    fig, ax = plt.subplots()
    try:
        qis.plot_bars(frame, stacked=False, colors=["red", "green", "blue"], ax=ax)
        fig.canvas.draw()
        if all_missing:
            assert not ax.patches
        else:
            expected = {2.0: "green", 3.0: "green", 4.0: "blue", 5.0: "blue"}
            bars = [patch for patch in ax.patches if patch.get_height() in expected]
            assert len(bars) == 4
            for bar in bars:
                np.testing.assert_allclose(
                    bar.get_facecolor(), _bar_rgba(expected[bar.get_height()])
                )
                # Native categorical metadata reserves even unobserved levels in the dodge layout.
                np.testing.assert_allclose(bar.get_width(), 0.8 / len(categories))
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ["A", "B", "C"]
        pd.testing.assert_frame_equal(frame, original)
    finally:
        plt.close(fig)


def test_plot_bars_preserves_duplicate_hue_behavior_with_explicit_legend():
    """An explicit legend can admit duplicate columns; do not impose a new last-colour policy."""
    frame = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], columns=["A", "A", "B"])
    fig, ax = plt.subplots()
    try:
        qis.plot_bars(
            frame,
            stacked=False,
            colors=["red", "green", "blue"],
            legend_labels=["First", "Second", "Third"],
            ax=ax,
        )
        fig.canvas.draw()
        # Duplicate A observations are averaged by the existing renderer, not this correction.
        expected = {1.5: "red", 4.5: "red", 3.0: "green", 6.0: "green"}
        bars = [patch for patch in ax.patches if patch.get_height() in expected]
        assert len(bars) == 4
        for bar in bars:
            np.testing.assert_allclose(bar.get_facecolor(), _bar_rgba(expected[bar.get_height()]))
    finally:
        plt.close(fig)


@pytest.mark.parametrize("levels", [2, 3])
@pytest.mark.parametrize("missing_first", [False, True])
@pytest.mark.parametrize("default_colors", [False, True])
def test_plot_bars_preserves_multiindex_positional_fallback(
    levels, missing_first, default_colors
):
    """Melted outer-level hues retain the established MultiIndex rendering policy."""
    tuples = [("Portfolio", "A"), ("Benchmark", "B")]
    if levels == 3:
        tuples = [(*column, "Return") for column in tuples]
    columns = pd.MultiIndex.from_tuples(tuples)
    frame = pd.DataFrame([[1.0, 3.0], [2.0, 4.0]], columns=columns)
    if missing_first:
        frame.iloc[:, 0] = np.nan
    original = frame.copy(deep=True)
    colors = get_n_colors(2) if default_colors else ["red", "blue"]
    original_colors = list(colors)
    # MultiIndex melt exposes outer labels, not full tuples. Preserve its positional
    # palette, including the first colour for the sole surviving outer-level series.
    expected = {3.0: colors[0 if missing_first else 1], 4.0: colors[0 if missing_first else 1]}
    if not missing_first:
        expected.update({1.0: colors[0], 2.0: colors[0]})
    fig, ax = plt.subplots()
    try:
        qis.plot_bars(frame, stacked=False, colors=None if default_colors else colors, ax=ax)
        fig.canvas.draw()
        bars = [patch for patch in ax.patches if patch.get_height() in expected]
        assert len(bars) == len(expected)
        for bar in bars:
            np.testing.assert_allclose(bar.get_facecolor(), _bar_rgba(expected[bar.get_height()]))
            np.testing.assert_allclose(bar.get_width(), 0.8 if missing_first else 0.4)
        pd.testing.assert_frame_equal(frame, original)
        assert colors == original_colors
    finally:
        plt.close(fig)
