"""Keep column alignment inexpensive without losing its pre-draw geometry update."""

import matplotlib.pyplot as plt
from matplotlib.table import Table
import pandas as pd
import pytest

from qis.plots.table import plot_df_table, set_align_for_column


@pytest.mark.parametrize("left_aligned", [False, True])
@pytest.mark.parametrize("add_index", [False, True])
@pytest.mark.parametrize("supplied_axes", [False, True])
def test_plot_df_table_inspects_properties_once(
    monkeypatch, left_aligned, add_index, supplied_axes
):
    """Both public alignment branches retain labels, geometry and caller ownership."""
    frame = pd.DataFrame(
        [["1.25%", "2.50%"], ["-3.00%", "nan"]],
        index=["Stock", "Bond"],
        columns=["Return", "Risk"],
    )
    original = frame.copy(deep=True)
    inspections = []
    properties = Table.properties

    def inspected(table):
        inspections.append(table)
        return properties(table)

    monkeypatch.setattr(Table, "properties", inspected)
    sentinel = plt.figure()
    fig = None
    try:
        if supplied_axes:
            fig, ax = plt.subplots()
            assert (
                plot_df_table(
                    frame,
                    ax=ax,
                    add_index_as_column=add_index,
                    left_aligned_first_col=left_aligned,
                )
                is None
            )
        else:
            fig = plot_df_table(
                frame,
                add_index_as_column=add_index,
                left_aligned_first_col=left_aligned,
            )
            ax = fig.axes[0]
        table = ax.tables[0]
        # Drawing could repair uninitialised geometry and hide a header-consumer regression.
        expected = (
            [[" ", "Return", "Risk"], ["Stock", "1.25%", "2.50%"], ["Bond", "-3.00%", "nan"]]
            if add_index
            else [["Return", "Risk"], ["1.25%", "2.50%"], ["-3.00%", "nan"]]
        )
        for row, texts in enumerate(expected):
            for col, text in enumerate(texts):
                cell = table[row, col]
                assert cell.get_text().get_text() == text
                assert cell.get_height() == pytest.approx(1 / 3)
                if col == 0:
                    assert cell._loc == ("left" if left_aligned else "right")
        # A mechanism assertion avoids a hardware-dependent elapsed-time test.
        assert inspections == [table]
        fig.canvas.draw()
        pd.testing.assert_frame_equal(frame, original)
        assert plt.fignum_exists(sentinel.number)
    finally:
        if fig is not None:
            plt.close(fig)
        plt.close(sentinel)


@pytest.mark.parametrize("populated", [False, True])
def test_set_align_for_column_missing_column_does_not_inspect(monkeypatch, populated):
    """A missing column stays a no-op even without an attached renderer-ready grid."""
    fig, ax = plt.subplots()
    try:
        table = ax.table(cellText=[["untouched"]]) if populated else Table(ax)

        def unexpected_properties():
            pytest.fail("an absent column must not trigger property getters")

        monkeypatch.setattr(table, "properties", unexpected_properties)
        set_align_for_column(table, col=99)
        if populated:
            assert table[0, 0]._loc == "right"
            assert table[0, 0].get_text().get_text() == "untouched"
    finally:
        plt.close(fig)


def test_set_align_for_column_refreshes_geometry_on_each_call():
    """Inspection remains per call: a later bbox change must not reuse stale geometry."""
    fig, ax = plt.subplots()
    try:
        table = ax.table(cellText=[["a", "b"], ["c", "d"]], bbox=(0, 0, 1, 1))
        set_align_for_column(table, col=1, align="left")
        assert table[0, 1].get_height() == pytest.approx(0.5)
        assert table[0, 0]._loc == "right"
        assert table[0, 1]._loc == "left"

        # A caller can resize an existing table before realigning another column.
        table._bbox = (0, 0, 1, 0.6)
        set_align_for_column(table, col=0, align="center")
        for row in (0, 1):
            assert table[row, 0].get_height() == pytest.approx(0.3)
            assert table[row, 0]._loc == "center"
            assert table[row, 1]._loc == "left"
        fig.canvas.draw()
    finally:
        plt.close(fig)
