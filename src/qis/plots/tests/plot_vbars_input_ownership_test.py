"""Keep total annotations on the chart without changing reusable analytical row IDs."""

from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import pytest

import qis


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize(
    ("index", "labels"),
    [
        (pd.Index(["P", "Q"], name="Portfolio"), ["P 50%", "Q 30%"]),
        (pd.Index([7, 3], name="Portfolio"), ["7 50%", "3 30%"]),
        (
            pd.date_range("2024-01-01", periods=2, tz="UTC", name="Date"),
            ["2024-01-01 00:00:00+00:00 50%", "2024-01-02 00:00:00+00:00 30%"],
        ),
        (
            pd.CategoricalIndex(["P", "Q"], categories=["Q", "P", "Unused"], name="Portfolio"),
            ["P 50%", "Q 30%"],
        ),
        (
            pd.MultiIndex.from_tuples([("P", "US"), ("Q", "UK")], names=["Portfolio", "Region"]),
            ["('P', 'US') 50%", "('Q', 'UK') 30%"],
        ),
    ],
    ids=["text", "integer", "dated", "categorical", "multiindex"],
)
def test_plot_vbars_keeps_row_identity_and_signed_geometry_across_repeated_draws(
    index, labels, dtype
):
    """Drawing a second report panel must not reuse labels modified by the first panel."""
    frame = pd.DataFrame({"Equity": [0.3, -0.1], "Bonds": [0.2, 0.4]}, index=index, dtype=dtype)
    frame.columns.name = "Asset class"
    frame.attrs = {"report": {"unit": "fraction"}}
    original = deepcopy(frame)
    original_index, original_columns = frame.index, frame.columns
    totals, colors = [0.5, 0.3], ["red", "blue"]

    for _ in range(2):
        fig, ax = plt.subplots()
        try:
            assert (
                qis.plot_vbars(
                    frame,
                    totals=totals,
                    add_total_to_index=True,
                    var_format="{:.0%}",
                    colors=colors,
                    is_category_names_colors=False,
                    ax=ax,
                )
                is None
            )
            fig.canvas.draw()
            assert [tick.get_text() for tick in ax.get_yticklabels()] == labels
            # Signed segments are specified from the input exposures, not another plotting path.
            np.testing.assert_allclose(
                [patch.get_width() for patch in ax.patches], [0, 0.1, 0, 0, 0.3, 0, 0.2, 0.4]
            )
            np.testing.assert_allclose(
                [patch.get_x() for patch in ax.patches], [0, -0.1, 0, 0, 0, 0, 0.3, 0]
            )
            for position, patch in enumerate(ax.patches):
                np.testing.assert_allclose(patch.get_facecolor(), to_rgba(colors[position % 2]))
            assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Equity", "Bonds"]
            assert ax.figure is fig
            pd.testing.assert_frame_equal(frame, original)
            assert frame.index is original_index and frame.columns is original_columns
            assert frame.attrs == original.attrs
            assert totals == [0.5, 0.3] and colors == ["red", "blue"]
        finally:
            plt.close(fig)


@pytest.mark.parametrize("as_series", [False, True])
@pytest.mark.parametrize(
    ("annotate", "totals"), [(False, [0.3, -0.1]), (False, None), (True, None)]
)
def test_plot_vbars_preserves_unannotated_container_controls(as_series, annotate, totals):
    """Neither a disabled option nor absent supplied totals should relabel a chart."""
    frame = pd.DataFrame({"Equity": [0.3, -0.1]}, index=pd.Index(["P", "Q"], name="Portfolio"))
    data = frame["Equity"] if as_series else frame
    data.attrs = {"report": {"unit": "fraction"}}
    original = deepcopy(data)
    fig, ax = plt.subplots()
    try:
        qis.plot_vbars(data, totals=totals, add_total_to_index=annotate, ax=ax)
        fig.canvas.draw()
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["P", "Q"]
        if as_series:
            pd.testing.assert_series_equal(data, original)
        else:
            pd.testing.assert_frame_equal(data, original)
        assert data.attrs == original.attrs
    finally:
        plt.close(fig)


def test_plot_vbars_keeps_input_when_renderer_rejects(monkeypatch):
    """The first artist failure occurs after label preparation, where caller mutation began."""
    frame = pd.DataFrame({"Equity": [0.3, -0.1]}, index=pd.Index(["P", "Q"], name="Portfolio"))
    original = deepcopy(frame)
    fig, ax = plt.subplots()

    def reject(*args, **kwargs):
        raise RuntimeError("renderer rejected")

    monkeypatch.setattr(ax, "barh", reject)
    try:
        with pytest.raises(RuntimeError, match="renderer rejected"):
            qis.plot_vbars(frame, totals=[0.3, -0.1], add_total_to_index=True, ax=ax)
        pd.testing.assert_frame_equal(frame, original)
    finally:
        plt.close(fig)


def test_plot_vbars_preserves_short_totals_rejection():
    """Owning the labels must not replace pandas' length check with a truncating zip."""
    frame = pd.DataFrame({"Equity": [0.3, -0.1]}, index=pd.Index(["P", "Q"], name="Portfolio"))
    original = deepcopy(frame)
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="Length mismatch"):
            qis.plot_vbars(frame, totals=[0.3], add_total_to_index=True, ax=ax)
        assert not ax.patches
        pd.testing.assert_frame_equal(frame, original)
    finally:
        plt.close(fig)


def test_plot_vbars_retains_native_metadata_handling_without_relabeling_input():
    """An owned working frame must not silently discard attrs that pandas would process."""
    frame = pd.DataFrame({"Equity": [0.3, -0.1]}, index=pd.Index(["P", "Q"], name="Portfolio"))
    original = frame.copy(deep=True)

    class ReportMetadata:
        def __deepcopy__(self, memo):
            raise RuntimeError("metadata cannot be copied")

    metadata = ReportMetadata()
    frame.attrs["report"] = metadata
    fig, ax = plt.subplots()
    try:
        with pytest.raises(RuntimeError, match="metadata cannot be copied"):
            qis.plot_vbars(frame, totals=[0.3, -0.1], add_total_to_index=True, ax=ax)
        # Frame equality boxes columns and copies attrs, which this fixture deliberately rejects.
        pd.testing.assert_index_equal(frame.index, original.index)
        pd.testing.assert_index_equal(frame.columns, original.columns)
        np.testing.assert_array_equal(frame.to_numpy(), original.to_numpy(), strict=True)
        assert frame.attrs["report"] is metadata
    finally:
        plt.close(fig)


def test_plot_vbars_returns_owned_figure_and_preserves_series_identity():
    """The no-axes Series path still returns a figure, not a replacement data container."""
    data = pd.Series([0.3, -0.1], index=pd.Index(["P", "Q"], name="Portfolio"), name="Equity")
    original = deepcopy(data)
    fig = qis.plot_vbars(data, totals=[0.3, -0.1], add_total_to_index=True, var_format="{:.0%}")
    try:
        fig.canvas.draw()
        assert [tick.get_text() for tick in fig.axes[0].get_yticklabels()] == ["P 30%", "Q -10%"]
        pd.testing.assert_series_equal(data, original)
    finally:
        plt.close(fig)
