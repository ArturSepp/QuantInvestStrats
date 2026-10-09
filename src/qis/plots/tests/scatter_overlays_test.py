"""Full-sample scatter overlays must use the fitted observations, not column names."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.plots.derived import perf_table


def _sample(order=1, fit_intercept=True):
    """Use known polynomial coefficients with residuals orthogonal to degrees zero to two."""
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    expected = 2 * x + (0.006 if fit_intercept else 0)
    if order == 2:
        expected += 0.5 * x**2
    # This finite-difference vector has zero inner product with 1, x and x², so an OLS
    # fit recovers the stated coefficients without using production regression as the oracle.
    residual = np.array([0.01, -0.04, 0.06, -0.04, 0.01])
    frame = pd.DataFrame({"horizontal": x, "vertical": expected + residual})
    frame = frame.iloc[[2, 0, 3, 1, 4]].copy()
    frame.index = ["duplicate", "duplicate", "third", "fourth", "fifth"]
    return frame, x, expected


def _assert_overlays(ax, x, expected, prediction, confidence, half_width=None):
    """Check coordinate/value association through actual rendering and preserve native CI math."""
    ax.figure.canvas.draw()
    overlays = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(overlays) == int(prediction) + 2 * int(confidence)
    if prediction:
        line = overlays.pop(0)
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_allclose(line.get_ydata(), expected, atol=1e-14)
        assert line.get_color() == "blue"
    if confidence:
        # For the five-point sample, preserve calc_ci's n-2 degrees of freedom and t(.95, 3),
        # even with a quadratic or no-intercept fit: this fixes routing, not CI semantics.
        if half_width is None:
            t_quantile = 2.3533634348018264
            residual_spread = np.sqrt(0.007 / 3)
            half_width = t_quantile * residual_spread * np.sqrt(1 / 5 + (x - 0.3) ** 2 / 0.1)
        for line, boundary in zip(overlays, (expected - half_width, expected + half_width)):
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_allclose(line.get_ydata(), boundary, atol=1e-14)
            assert line.get_color() == "0.5"


@pytest.mark.parametrize("prediction,confidence", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_plot_scatter_full_sample_overlays_use_fitted_numeric_coordinates(
    prediction, confidence, order, fit_intercept
):
    """Each flag and their interaction must render the independently known fitted sample."""
    frame, x, expected = _sample(order, fit_intercept)
    before = frame.copy(deep=True)
    fig, ax = plt.subplots()
    try:
        result = qis.plot_scatter(
            frame,
            x="horizontal",
            y="vertical",
            full_sample_order=order,
            fit_intercept=fit_intercept,
            add_universe_model_label=False,
            add_universe_model_prediction=prediction,
            add_universe_model_ci=confidence,
            ax=ax,
        )
        assert result is None
        _assert_overlays(ax, x, expected, prediction, confidence)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), x)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected, atol=1e-14)
        pd.testing.assert_frame_equal(frame, before)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_plot_scatter_default_fit_remains_unchanged(order, fit_intercept):
    """Enabling an optional path must not alter the existing default line or legend."""
    frame, x, expected = _sample(order, fit_intercept)
    fig, ax = plt.subplots()
    try:
        qis.plot_scatter(
            frame,
            x="horizontal",
            y="vertical",
            full_sample_order=order,
            fit_intercept=fit_intercept,
            ax=ax,
        )
        _assert_overlays(ax, x, expected, False, False)
        assert len(ax.lines) == 1
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), x)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected, atol=1e-14)
        assert ax.get_legend().get_texts()[0].get_text().startswith("Full sample:")
    finally:
        plt.close(fig)


@pytest.mark.parametrize("prediction,confidence", [(True, False), (False, True), (True, True)])
def test_plot_scatter_full_sample_overlays_remain_independent_of_hue_fits(prediction, confidence):
    """Group-specific fits must not replace the full-sample overlay coordinates."""
    frame, x, expected = _sample()
    lower = frame.assign(vertical=frame["vertical"] - 0.2, group="lower")
    upper = frame.assign(vertical=frame["vertical"] + 0.2, group="upper")
    frame = pd.concat([lower, upper])
    before = frame.copy(deep=True)
    # Opposite group intercept shifts cancel in the full sample. Its residual sum of
    # squares is 2*.007 + 10*.2², not the residual of whichever group was fitted last.
    x = np.repeat(x, 2)
    expected = np.repeat(expected, 2)
    half_width = 1.8595480375228424 * np.sqrt(0.414 / 8) * np.sqrt(1 / 10 + (x - 0.3) ** 2 / 0.2)
    fig, ax = plt.subplots()
    try:
        qis.plot_scatter(
            frame,
            x="horizontal",
            y="vertical",
            hue="group",
            order=1,
            full_sample_order=1,
            add_universe_model_prediction=prediction,
            add_universe_model_ci=confidence,
            ax=ax,
        )
        _assert_overlays(ax, x, expected, prediction, confidence, half_width)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected[::2] - 0.2, atol=1e-14)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), expected[::2] + 0.2, atol=1e-14)
        pd.testing.assert_frame_equal(frame, before)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("missing_column", ["horizontal", "vertical"])
def test_plot_scatter_full_sample_overlays_share_filtered_point_rows(missing_column):
    """A discarded row must not survive in the overlay while prediction and CI are paired."""
    frame, x, expected = _sample()
    # Add a rejected observation instead of removing an oracle row: the five known finite
    # points and residual spread remain fixed, independently of production missing-row filtering.
    rejected = pd.DataFrame({"horizontal": [9.0], "vertical": [99.0]}, index=["duplicate"])
    rejected[missing_column] = np.nan
    frame = pd.concat([frame.iloc[:2], rejected, frame.iloc[2:]])
    before = frame.copy(deep=True)
    fig, ax = plt.subplots()
    try:
        qis.plot_scatter(
            frame,
            x="horizontal",
            y="vertical",
            full_sample_order=1,
            add_universe_model_prediction=True,
            add_universe_model_ci=True,
            ax=ax,
        )
        _assert_overlays(ax, x, expected, True, True)
        pd.testing.assert_frame_equal(frame, before)
    finally:
        plt.close(fig)


def test_plot_scatter_full_sample_overlays_create_drawable_figure_with_inferred_columns():
    """Column inference and figure allocation must reach the same optional numeric overlays."""
    frame, x, expected = _sample()
    fig = None
    try:
        fig = qis.plot_scatter(
            frame.sort_values("horizontal"),
            full_sample_order=1,
            add_universe_model_prediction=True,
            add_universe_model_ci=True,
        )
        assert isinstance(fig, plt.Figure)
        _assert_overlays(fig.axes[0], x, expected, True, True)
    finally:
        if fig is not None:
            plt.close(fig)


def test_plot_ra_perf_scatter_forwards_full_sample_overlay_flags(monkeypatch):
    """The performance wrapper must forward opt-in flags to the same rendered coordinates."""
    frame, x, expected = _sample()
    # Isolate the wrapper's column/kwargs boundary; performance statistics have their own
    # regression owners. The injected table still reaches the real scatter and canvas renderer.
    frame.columns = [qis.PerfStat.MAX_DD.to_str(), qis.PerfStat.PA_RETURN.to_str()]
    monkeypatch.setattr(perf_table.rpt, "compute_ra_perf_table", lambda **kwargs: frame)
    fig, ax = plt.subplots()
    try:
        qis.plot_ra_perf_scatter(
            prices=pd.DataFrame(),
            ci=None,
            add_universe_model_prediction=True,
            add_universe_model_ci=True,
            ax=ax,
        )
        _assert_overlays(ax, x, expected, True, True)
    finally:
        plt.close(fig)
