"""Regression coverage for benchmark removal from regime plots.

Regime classifiers intentionally retain the benchmark in their component tables even when the
final presentation table drops it. The public ``plot_regime_data()`` option must apply that final
row selection at the renderer boundary without changing component data, regime columns, colors,
totals, or caller-owned state.

A deterministic classifier double isolates the accepted final-table/component-table distinction.
The matrix covers every ``RegimeData`` selection, both bar renderers, and enabled/disabled
benchmark removal. A nullable component table also reaches a warning-strict canvas draw so the
assertion protects the user-visible plot rather than only the intermediate artist construction.
"""

from __future__ import annotations

import warnings
from typing import Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import qis.plots.bars as bars_module
import qis.plots.derived.regime_data as regime_data_module
from qis.perfstats.config import RegimeData


class _RegimeClassifierProtocol(Protocol):
    """Test-side interface used by the regime-plot boundary."""

    def compute_regimes_pa_perf_table(
        self,
        *,
        drop_benchmark: bool,
        **kwargs: object,
    ) -> tuple[pd.DataFrame, dict[RegimeData, pd.DataFrame]]:
        """Return the final presentation table and its component tables."""
        raise NotImplementedError

    def get_regime_ids_colors(self) -> dict[str, str]:
        """Return the ordered regime-color mapping."""
        raise NotImplementedError


class _RegimeDataModuleProtocol(Protocol):
    """Typed test-side interface for the public plot helper."""

    def plot_regime_data(
        self,
        *,
        regime_classifier: _RegimeClassifierProtocol,
        regime_data_to_plot: RegimeData,
        drop_benchmark: bool,
        is_use_vbar: bool,
        add_bar_values: bool = True,
        ax: Axes | None = None,
    ) -> Figure:
        """Render one selected regime component table."""
        raise NotImplementedError


_REGIME_DATA_MODULE = cast(_RegimeDataModuleProtocol, regime_data_module)


# =============================================================================
# Deterministic classifier boundary and independent expectations
# =============================================================================

_ASSET_A = "Asset A"
_ASSET_B = "Asset B"
_BENCHMARK = "Benchmark"
_EXPECTED_COLORS = ("#a50026", "#006837")
_INDEX = pd.Index((_BENCHMARK, _ASSET_A, _ASSET_B), name="Instrument")

_COLUMNS = {
    RegimeData.REGIME_AVG: ("Bear Average", "Bull Average"),
    RegimeData.REGIME_PA: ("Bear P.a.", "Bull P.a."),
    RegimeData.REGIME_SHARPE: ("Bear Sharpe", "Bull Sharpe"),
}


class _RegimeClassifier:
    """Expose filtered final rows beside deliberately unfiltered components."""

    def __init__(self) -> None:
        self.calls: list[bool] = []
        self.components = {
            regime_data: pd.DataFrame(
                {
                    columns[0]: pd.array((0.01, 0.02, 0.03), dtype="Float64"),
                    columns[1]: pd.array((0.04, 0.05, 0.06), dtype="Float64"),
                },
                index=_INDEX,
            )
            for regime_data, columns in _COLUMNS.items()
        }

    def compute_regimes_pa_perf_table(
        self,
        *,
        drop_benchmark: bool,
        **kwargs: object,
    ) -> tuple[pd.DataFrame, dict[RegimeData, pd.DataFrame]]:
        """Apply benchmark removal only to the final presentation table."""
        del kwargs
        self.calls.append(drop_benchmark)
        final_labels = (_ASSET_A, _ASSET_B) if drop_benchmark else (_BENCHMARK, _ASSET_A, _ASSET_B)
        final_table = pd.DataFrame(
            {"Summary": pd.array((1.0,) * len(final_labels), dtype="Float64")},
            index=pd.Index(final_labels, name=_INDEX.name),
        )
        return final_table, self.components

    def get_regime_ids_colors(self) -> dict[str, str]:
        """Return colors in the same order as the component columns."""
        return {"Bear": _EXPECTED_COLORS[0], "Bull": _EXPECTED_COLORS[1]}


def _expected_component(
    classifier: _RegimeClassifier,
    regime_data: RegimeData,
    *,
    drop_benchmark: bool,
) -> pd.DataFrame:
    """Select expected rows independently from the plotting implementation."""
    rows = (_ASSET_A, _ASSET_B) if drop_benchmark else (_BENCHMARK, _ASSET_A, _ASSET_B)
    return classifier.components[regime_data].loc[list(rows)]


# =============================================================================
# Renderer boundary and user-visible canvas contract
# =============================================================================


@pytest.mark.parametrize("regime_data", tuple(RegimeData))
@pytest.mark.parametrize("is_use_vbar", (False, True))
@pytest.mark.parametrize("drop_benchmark", (False, True))
def test_plot_regime_data_applies_final_row_membership_at_renderer(
    monkeypatch: pytest.MonkeyPatch,
    regime_data: RegimeData,
    is_use_vbar: bool,
    drop_benchmark: bool,
) -> None:
    """Pass exact filtered rows and row-derived totals to either renderer."""
    classifier = _RegimeClassifier()
    original_components = {
        key: value.copy(deep=True) for key, value in classifier.components.items()
    }
    captured_data: list[pd.DataFrame] = []
    captured_colors: list[tuple[str, ...]] = []
    captured_totals: list[np.ndarray] = []
    figure = plt.figure()

    def capture_renderer(
        *,
        df: pd.DataFrame,
        colors: list[str],
        totals: np.ndarray,
        **kwargs: object,
    ) -> Figure:
        """Capture the final renderer boundary without invoking Matplotlib."""
        del kwargs
        captured_data.append(df.copy(deep=True))
        captured_colors.append(tuple(colors))
        captured_totals.append(np.asarray(totals, dtype=float))
        return figure

    renderer = "plot_vbars" if is_use_vbar else "plot_bars"
    monkeypatch.setattr(bars_module, renderer, capture_renderer)
    try:
        actual_figure = _REGIME_DATA_MODULE.plot_regime_data(
            regime_classifier=classifier,
            regime_data_to_plot=regime_data,
            drop_benchmark=drop_benchmark,
            is_use_vbar=is_use_vbar,
        )

        expected = _expected_component(
            classifier,
            regime_data,
            drop_benchmark=drop_benchmark,
        )
        assert actual_figure is figure
        assert classifier.calls == [drop_benchmark]
        assert len(captured_data) == 1
        pd.testing.assert_frame_equal(captured_data[0], expected)
        assert captured_colors == [_EXPECTED_COLORS]
        np.testing.assert_allclose(
            captured_totals[0],
            np.sum(expected.to_numpy(dtype=float), axis=1),
            rtol=0.0,
            atol=0.0,
        )
        for key, original in original_components.items():
            pd.testing.assert_frame_equal(classifier.components[key], original)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("regime_data", tuple(RegimeData))
@pytest.mark.parametrize("is_use_vbar", (False, True))
def test_plot_regime_data_drops_benchmark_on_rendered_canvas(
    regime_data: RegimeData,
    is_use_vbar: bool,
) -> None:
    """Exclude the benchmark from a warning-strict nullable-data canvas render."""
    classifier = _RegimeClassifier()
    original_components = {
        key: value.copy(deep=True) for key, value in classifier.components.items()
    }
    figure, axis = plt.subplots()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _REGIME_DATA_MODULE.plot_regime_data(
                regime_classifier=classifier,
                regime_data_to_plot=regime_data,
                drop_benchmark=True,
                is_use_vbar=is_use_vbar,
                add_bar_values=False,
                ax=axis,
            )
            figure.canvas.draw()

        label_axis = axis.get_yticklabels() if is_use_vbar else axis.get_xticklabels()
        assert [label.get_text() for label in label_axis] == [_ASSET_A, _ASSET_B]
        for key, original in original_components.items():
            pd.testing.assert_frame_equal(classifier.components[key], original)
    finally:
        plt.close(figure)
