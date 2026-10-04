"""Stack totals distinguish unavailable observations from genuine observed zeros."""

import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis.plots.stackplot as stackplot
from qis.plots.utils import LegendStats


@pytest.mark.parametrize("use_bar_plot", [False, True], ids=["area", "bar"])
@pytest.mark.parametrize("storage", ["ordinary", "nullable", "mixed"])
@pytest.mark.parametrize("terminal_missing", [True, False], ids=["missing-end", "zero-end"])
@pytest.mark.parametrize(
    ("legend_stats", "add_total_line"),
    [(LegendStats.LAST, True), (LegendStats.FIRST_LAST_NON_ZERO, True), (LegendStats.LAST, False)],
    ids=["last-line", "nonzero-line", "last-title-only"],
)
def test_plot_stack_preserves_missing_totals_and_observed_zero_controls(
    monkeypatch,
    use_bar_plot: bool,
    storage: str,
    terminal_missing: bool,
    legend_stats: LegendStats,
    add_total_line: bool,
) -> None:
    """Keep missing, zero, signed-cancelling and partially observed rows distinct."""
    values = pd.DataFrame(
        {
            "Long": [0.4, np.nan, 0.0, 0.5, 0.3, 0.2, np.nan if terminal_missing else 0.0],
            "Hedge": [0.6, np.nan, 0.0, -0.5, np.nan, 0.4, np.nan if terminal_missing else 0.0],
            "Unavailable": [np.nan] * 7,
        }
    )
    if storage == "nullable":
        values = values.astype("Float64")
    elif storage == "mixed":
        values["Long"] = values["Long"].astype("Float64")
        values["Unavailable"] = values["Unavailable"].astype("Float64")
    original = values.copy(deep=True)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    captured = []
    native_lineplot = stackplot.sns.lineplot

    def capture_total(**kwargs):
        captured.append(kwargs["y"].copy())
        return native_lineplot(**kwargs)

    monkeypatch.setattr(stackplot.sns, "lineplot", capture_total)
    expected = np.array([1.0, np.nan, 0.0, 0.0, 0.3, 0.6, np.nan if terminal_missing else 0.0])
    fig, ax = plt.subplots()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = stackplot.plot_stack(
                values,
                use_bar_plot=use_bar_plot,
                add_total_line=add_total_line,
                legend_stats=legend_stats,
                colors=colors,
                ax=ax,
            )
            fig.canvas.draw()
        assert result is None
        assert len(captured) == int(add_total_line)
        assert len(ax.lines) == int(add_total_line)
        if add_total_line:
            pd.testing.assert_index_equal(captured[0].index, values.index)
            np.testing.assert_allclose(
                captured[0].to_numpy(dtype=float, na_value=np.nan),
                expected,
                rtol=0,
                atol=1e-15,
                equal_nan=True,
            )
            # Seaborn omits missing points; it must not create spurious zero observations.
            observed = ~np.isnan(expected)
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), values.index[observed])
            np.testing.assert_allclose(ax.lines[0].get_ydata(), expected[observed])
            assert mcolors.to_hex(ax.lines[0].get_color()) == "#000000"
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_title().get_text() == (
            "Total: last=nan%" if terminal_missing else "Total: last=0%"
        )
        if legend_stats == LegendStats.LAST:
            endpoint = "nan%" if terminal_missing else "0%"
            labels = [f"Long: last={endpoint}", f"Hedge: last={endpoint}", "Unavailable: last=nan%"]
        else:
            labels = [
                "Long: first=40%, last=20%",
                "Hedge: first=60%, last=40%",
                "Unavailable: first=nan%, last=nan%",
            ]
        assert [text.get_text() for text in legend.get_texts()] == labels
        pd.testing.assert_frame_equal(values, original)
        assert colors == ["tab:blue", "tab:orange", "tab:green"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("use_bar_plot", [False, True], ids=["area", "bar"])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_plot_stack_keeps_entirely_missing_totals_undefined(
    use_bar_plot: bool,
    dtype: str,
) -> None:
    """A nonempty panel with no observations must not acquire a zero total line."""
    values = pd.DataFrame({"Long": [np.nan] * 3, "Hedge": [np.nan] * 3}, dtype=dtype)
    original = values.copy(deep=True)
    fig, ax = plt.subplots()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            stackplot.plot_stack(
                values,
                use_bar_plot=use_bar_plot,
                add_total_line=True,
                legend_stats=LegendStats.LAST,
                ax=ax,
            )
            fig.canvas.draw()
        assert all(len(line.get_ydata()) == 0 for line in ax.lines)
        assert ax.get_legend().get_title().get_text() == "Total: last=nan%"
        pd.testing.assert_frame_equal(values, original)
    finally:
        plt.close(fig)
