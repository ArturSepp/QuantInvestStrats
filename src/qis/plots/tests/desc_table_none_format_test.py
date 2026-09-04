"""Regression tests for unformatted descriptive legends on time-series plots.

``plot_time_series`` accepts ``var_format=None`` so Matplotlib can choose the axis tick labels
and ordinary statistic legends can use native scalar text. A requested descriptive table shares
that public argument, so it must compose with the same native-text convention rather than call
``format`` on ``None`` after the plot has already been drawn.

The principal fixture combines complete, ragged, and all-missing columns in one ordered panel.
Every displayed ``DescTableType`` is exercised with ordinary ``float64`` and nullable
``Float64`` storage under warnings-as-errors. Exact short-table controls use independently
calculated means and sample standard deviations, and every successful plot is rendered through
the Matplotlib canvas before its caller-owned data and figure are released.
"""

from typing import Literal, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import qis as qis_module
from qis.perfstats.desc_table import DescTableType


pytestmark = pytest.mark.filterwarnings("error")


class _QisPublicProtocol(Protocol):
    """Typed test-side interface for the public time-series plot."""

    def plot_time_series(
        self,
        df: pd.DataFrame | pd.Series,
        *,
        desc_table_type: DescTableType,
        var_format: str | None,
        x_date_freq: str | None,
        legend_loc: str | None = "upper left",
        legend_labels: list[str] | None = None,
        ax: Axes | None = None,
    ) -> Figure | None:
        """Draw a time series with an optional descriptive legend."""
        raise NotImplementedError


_QIS = cast(_QisPublicProtocol, qis_module)


# =============================================================================
# Shared deterministic fixtures and rendering helpers
# =============================================================================

_ALL_TABLE_TYPES = tuple(
    table_type for table_type in DescTableType if table_type is not DescTableType.NONE
)
_DATES = pd.date_range("2024-01-31", periods=24, freq="ME", name="Date")
_COMPLETE_VALUES = tuple(np.tile(np.array([0.0, 1.0, 2.0, 3.0]), 6))


def _mixed_panel(*, nullable: bool) -> pd.DataFrame:
    """Create complete, ragged, and all-missing histories in one panel.

    Args:
        nullable: Store every column as pandas nullable ``Float64`` when true.

    Returns:
        Ordered three-column panel with materially different missing-data states.
    """
    missing = pd.NA if nullable else np.nan
    dtype = pd.Float64Dtype() if nullable else float
    return pd.DataFrame(
        {
            "Complete": _COMPLETE_VALUES,
            "Ragged": (missing,) * 12 + _COMPLETE_VALUES[:12],
            "Missing": (missing,) * 24,
        },
        index=_DATES,
        dtype=dtype,
    )


def _reference_series(*, nullable: bool) -> pd.Series:
    """Create a named sample whose mean and sample spread both equal two.

    Args:
        nullable: Store values as pandas nullable ``Float64`` when true.

    Returns:
        Named Series containing the exact sample ``[0.0, 2.0, 4.0]``.
    """
    dtype = pd.Float64Dtype() if nullable else float
    return pd.Series((0.0, 2.0, 4.0), name="Asset", dtype=dtype)


def _legend_snapshot(ax: Axes) -> tuple[str, tuple[str, ...]]:
    """Capture the rendered descriptive legend title and rows.

    Args:
        ax: Axis containing a descriptive-table legend.

    Returns:
        Legend title followed by its rows in display order.
    """
    legend = ax.get_legend()
    assert legend is not None
    return (
        legend.get_title().get_text(),
        tuple(text.get_text() for text in legend.get_texts()),
    )


def _render_table(
    data: pd.DataFrame | pd.Series,
    table_type: DescTableType,
    var_format: str | None,
) -> tuple[str, tuple[str, ...]]:
    """Render one caller-owned descriptive-table plot and capture its legend.

    Args:
        data: Series or ordered mixed-history panel to plot.
        table_type: Descriptive-table schema displayed in the legend.
        var_format: Explicit Python format string, or native scalar formatting.

    Returns:
        Rendered legend title and rows.
    """
    original = data.copy()
    fig, ax = plt.subplots()

    try:
        result = _QIS.plot_time_series(
            data,
            desc_table_type=table_type,
            var_format=var_format,
            x_date_freq=None,
            ax=ax,
        )
        fig.canvas.draw()

        assert result is None
        if isinstance(data, pd.DataFrame):
            assert isinstance(original, pd.DataFrame)
            pd.testing.assert_frame_equal(data, original)
        else:
            assert isinstance(original, pd.Series)
            pd.testing.assert_series_equal(data, original)
        return _legend_snapshot(ax)
    finally:
        plt.close(fig)


def _split_legend(
    snapshot: tuple[str, tuple[str, ...]],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Normalize table spacing while retaining every displayed token.

    Args:
        snapshot: Rendered descriptive legend title and rows.

    Returns:
        Whitespace-separated title and row tokens.
    """
    title, rows = snapshot
    return tuple(title.split()), tuple(tuple(row.split()) for row in rows)


# =============================================================================
# Native descriptive-table formatting
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    "table_type",
    _ALL_TABLE_TYPES,
    ids=tuple(table_type.name.lower() for table_type in _ALL_TABLE_TYPES),
)
def test_none_format_matches_native_text_for_every_descriptive_table(
    table_type: DescTableType,
    nullable: bool,
) -> None:
    """Make ``None`` equivalent to native scalar text for every table schema.

    Args:
        table_type: Public descriptive-table schema displayed in the legend.
        nullable: Whether the mixed panel uses nullable floating storage.
    """
    data = _mixed_panel(nullable=nullable)

    actual = _render_table(data, table_type, None)
    expected = _render_table(data, table_type, "{}")

    assert actual == expected


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    ("var_format", "expected_rows"),
    (
        (
            None,
            (
                ("Complete", "1.5", "1.1420804814403216"),
                ("Ragged", "1.5", "1.1677484162422844"),
                ("Missing", "nan", "nan"),
            ),
        ),
        (
            "{:.3f}",
            (
                ("Complete", "1.500", "1.142"),
                ("Ragged", "1.500", "1.168"),
                ("Missing", "nan", "nan"),
            ),
        ),
    ),
    ids=("native", "explicit"),
)
def test_short_table_returns_independently_calculated_text(
    var_format: str | None,
    expected_rows: tuple[tuple[str, ...], ...],
    nullable: bool,
) -> None:
    """Preserve exact native and explicit text for mixed-history short tables.

    The complete sample has variance ``30 / 23``; the twelve observed ragged values have
    variance ``15 / 11``. Both means are exactly ``1.5``.

    Args:
        var_format: Explicit three-decimal format or native scalar formatting.
        expected_rows: Independently calculated rows after table-spacing normalization.
        nullable: Whether the mixed panel uses nullable floating storage.
    """
    title, rows = _split_legend(
        _render_table(_mixed_panel(nullable=nullable), DescTableType.SHORT, var_format)
    )

    assert title == ("Avg", "Std")
    assert rows == expected_rows


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_none_format_preserves_series_dataframe_short_table_parity(nullable: bool) -> None:
    """Render identical native short-table values from Series and DataFrame inputs.

    Args:
        nullable: Whether the exact reference sample uses nullable floating storage.
    """
    series = _reference_series(nullable=nullable)
    series_snapshot = _render_table(series, DescTableType.SHORT, None)
    frame_snapshot = _render_table(series.to_frame(), DescTableType.SHORT, None)

    assert series_snapshot == frame_snapshot
    assert _split_legend(series_snapshot) == (
        ("Avg", "Std"),
        (("Asset", "2.0", "2.0"),),
    )


# =============================================================================
# Existing table bypasses and figure ownership
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize("bypass", ("labels", "hidden"))
def test_none_format_preserves_existing_descriptive_table_bypasses(
    bypass: Literal["labels", "hidden"],
    nullable: bool,
) -> None:
    """Keep caller labels and hidden legends independent of table formatting.

    Args:
        bypass: Existing option that prevents a descriptive table from being displayed.
        nullable: Whether the mixed panel uses nullable floating storage.
    """
    data = _mixed_panel(nullable=nullable)
    original = data.copy()
    fig, ax = plt.subplots()
    labels = ["First", "Second", "Third"] if bypass == "labels" else None
    legend_loc = "upper left" if bypass == "labels" else None

    try:
        result = _QIS.plot_time_series(
            data,
            desc_table_type=DescTableType.SHORT,
            var_format=None,
            x_date_freq=None,
            legend_loc=legend_loc,
            legend_labels=labels,
            ax=ax,
        )
        fig.canvas.draw()

        assert result is None
        if labels is not None:
            assert _legend_snapshot(ax)[1] == tuple(labels)
        else:
            legend = ax.get_legend()
            assert legend is not None
            assert not legend.get_visible()
        pd.testing.assert_frame_equal(data, original)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_none_format_returns_a_caller_closeable_descriptive_figure(nullable: bool) -> None:
    """Return the internally allocated descriptive figure after successful rendering.

    Args:
        nullable: Whether the mixed panel uses nullable floating storage.
    """
    data = _mixed_panel(nullable=nullable)
    original = data.copy()
    existing_figures = set(plt.get_fignums())
    figure: Figure | None = None

    try:
        figure = _QIS.plot_time_series(
            data,
            desc_table_type=DescTableType.SHORT,
            var_format=None,
            x_date_freq=None,
        )

        assert isinstance(figure, Figure)
        figure.canvas.draw()
        assert len(set(plt.get_fignums()) - existing_figures) == 1
        pd.testing.assert_frame_equal(data, original)
    finally:
        if figure is not None:
            plt.close(figure)
        for figure_number in set(plt.get_fignums()) - existing_figures:
            plt.close(figure_number)
