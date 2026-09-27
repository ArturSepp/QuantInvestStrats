"""Regression tests for explicit x/y line legends."""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qis.plots.lineplot import plot_line  # noqa: E402


def test_plot_line_explicit_xy_legend_matches_rendered_y_artist() -> None:
    """Describe the rendered y series without treating x coordinates as another line."""
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [10.0, 20.0, 15.0]})
    data_before = data.copy(deep=True)
    fig, ax = plt.subplots()

    try:
        plot_line(df=data, x="x", y="y", ax=ax)
        fig.canvas.draw()
        legend = ax.get_legend()

        assert len(ax.lines) == 1
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["y"]
        assert len(legend.legend_handles) == 1
        pd.testing.assert_frame_equal(data, data_before)
    finally:
        plt.close(fig)
