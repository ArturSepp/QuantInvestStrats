"""
the factsheet panel-legend capacity guard: its arithmetic, its calibration, and its wiring.

A factsheet panel legend is a FIXED-height decoration carrying one row per series, and
matplotlib's ``constrained_layout`` counts the part of it that spills out of the axes as a layout
margin. Once the legend outgrows the panel cell the solver drives the axes height to zero and
disables itself for the whole figure, so every panel on the page reverts to the raw gridspec -
the page collapses, not only the panel that overflowed.

``test_legend_row_height_calibration`` is the test that matters: it re-measures the legend height
from matplotlib rather than trusting ``LEGEND_ROW_HEIGHT_PER_FONTSIZE``, so a matplotlib release
that invalidates the calibration fails here rather than silently in a report. The tests below it
pin the arithmetic and the wiring into ``generate_multi_asset_factsheet``.

Whether a rendered page looks right is `qis/tests/test_reporting_goldens.py`, not this module.
"""
# packages
import warnings
import matplotlib
matplotlib.use('Agg')  # noqa: E402  - a headless backend, set before pyplot is imported
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest
# qis
import qis as qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.plots.utils import set_legend
from qis.portfolio.reports.config import (LEGEND_ROW_HEIGHT_PER_FONTSIZE,
                                          PANEL_DECORATION_HEIGHT,
                                          estimate_legend_capacity,
                                          validate_legend_capacity)

A4_PORTRAIT = (8.3, 11.7)
CAPACITY_WARNING = 'exceed the panel legend capacity'


def measure_legend_height(n_entries: int,
                          fontsize: float
                          ) -> float:
    """
    height in inches of a qis panel legend carrying n_entries rows.

    Drawn on a figure tall enough that the legend is never clipped, so what is measured is the
    legend itself rather than the panel it would sit in.

    Args:
        n_entries: number of legend rows
        fontsize: font size in points

    Returns:
        legend height in inches
    """
    # Use a high-DPI canvas so integer-pixel font hinting does not dominate a
    # physical-height calibration at these deliberately small font sizes.
    fig, ax = plt.subplots(figsize=(8.0, 12.0), dpi=300)
    labels = [f"series {idx}" for idx in range(n_entries)]
    for idx, label in enumerate(labels):
        ax.plot([0.0, 1.0], [float(idx), float(idx)], label=label)
    set_legend(ax=ax, labels=labels, fontsize=fontsize)
    fig.canvas.draw()
    height = ax.get_legend().get_window_extent().height / fig.dpi
    plt.close(fig)
    return height


def assert_no_capacity_warning(func, **kwargs):
    """call func(**kwargs) and fail if it raises the capacity warning"""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        out = func(**kwargs)
    raised = [str(record.message) for record in records if CAPACITY_WARNING in str(record.message)]
    assert not raised, f"unexpected capacity warning: {raised}"
    return out


def test_legend_row_height_calibration() -> None:
    """the measured per-row legend height matches the constant the guard is built on"""
    n_entries = np.array([5, 10, 20, 40])
    for fontsize in (4.0, 5.0, 8.0):
        heights = np.array([measure_legend_height(n_entries=n, fontsize=fontsize)
                            for n in n_entries])
        slope = np.polyfit(n_entries, heights, deg=1)[0]
        expected = LEGEND_ROW_HEIGHT_PER_FONTSIZE * fontsize
        assert slope == pytest.approx(expected, rel=0.10), (
            f"legend row height at fontsize={fontsize} measured {slope:.5f} in/row against the "
            f"calibrated {expected:.5f} in/row: LEGEND_ROW_HEIGHT_PER_FONTSIZE needs re-measuring"
        )


def test_estimate_legend_capacity() -> None:
    """the capacity of the shipped page geometries, and its monotonicity"""
    # the two factsheet geometries: 2 of 14 rows and 1 of 7 rows are the same cell
    assert estimate_legend_capacity(figsize=A4_PORTRAIT, fontsize=5,
                                    panel_rows=2, gridspec_rows=14) == 15
    assert estimate_legend_capacity(figsize=A4_PORTRAIT, fontsize=5,
                                    panel_rows=1, gridspec_rows=7) == 15
    # smaller type and a taller page both buy rows
    assert estimate_legend_capacity(figsize=A4_PORTRAIT, fontsize=2.5,
                                    panel_rows=2, gridspec_rows=14) == 31
    assert estimate_legend_capacity(figsize=(8.3, 23.4), fontsize=5,
                                    panel_rows=2, gridspec_rows=14) == 36
    # a panel shorter than its own title and tick labels carries no legend at all
    assert estimate_legend_capacity(figsize=(8.3, 14.0 * PANEL_DECORATION_HEIGHT), fontsize=5,
                                    panel_rows=1, gridspec_rows=14) == 0


def test_estimate_legend_capacity_raises() -> None:
    """arguments that would make the capacity meaningless are rejected"""
    with pytest.raises(ValueError, match='fontsize'):
        estimate_legend_capacity(figsize=A4_PORTRAIT, fontsize=0.0, panel_rows=2, gridspec_rows=14)
    with pytest.raises(ValueError, match='panel_rows'):
        estimate_legend_capacity(figsize=A4_PORTRAIT, fontsize=5, panel_rows=0, gridspec_rows=14)


def test_validate_legend_capacity_warns_above_capacity() -> None:
    """the guard warns above capacity and stays silent at and below it"""
    kwargs = dict(figsize=A4_PORTRAIT, fontsize=5, panel_rows=2, gridspec_rows=14)
    with pytest.warns(UserWarning, match=CAPACITY_WARNING):
        validate_legend_capacity(n_legend_entries=22, **kwargs)
    assert_no_capacity_warning(validate_legend_capacity, n_legend_entries=15, **kwargs)


def test_validate_legend_capacity_suggestions_clear_the_guard() -> None:
    """the fontsize and figsize the warning suggests do not themselves trip the warning"""
    with pytest.warns(UserWarning, match=CAPACITY_WARNING) as record:
        validate_legend_capacity(n_legend_entries=22, figsize=A4_PORTRAIT, fontsize=5,
                                 panel_rows=2, gridspec_rows=14)
    message = str(record[0].message)
    fontsize_fit = float(message.split('fontsize=')[-1].split(',')[0])
    figsize_fit = float(message.split('figsize=(')[-1].split(')')[0].split(', ')[1].rstrip('.'))
    assert estimate_legend_capacity(figsize=A4_PORTRAIT, fontsize=fontsize_fit,
                                    panel_rows=2, gridspec_rows=14) >= 22
    assert estimate_legend_capacity(figsize=(A4_PORTRAIT[0], figsize_fit), fontsize=5,
                                    panel_rows=2, gridspec_rows=14) >= 22


def test_multi_asset_factsheet_warns() -> None:
    """the guard is wired into generate_multi_asset_factsheet and is quiet on an A4 page"""
    prices = qis.TimePeriod('31Dec2020', '31Dec2025').locate(generate_synthetic_prices())
    time_period = qis.get_time_period(prices)
    # 10 assets sit inside the A4 capacity of 15
    fig = assert_no_capacity_warning(qis.generate_multi_asset_factsheet,
                                     prices=prices, benchmark=prices.columns[0],
                                     time_period=time_period)
    plt.close(fig)
    # the same 10 assets do not fit a half-height page, whose capacity is 5
    with pytest.warns(UserWarning, match=CAPACITY_WARNING):
        fig = qis.generate_multi_asset_factsheet(prices=prices, benchmark=prices.columns[0],
                                                 time_period=time_period, figsize=(8.3, 5.85))
    plt.close(fig)


class LocalTest:
    """runnable checks, not part of the pytest suite"""
    LEGEND_HEIGHT_TABLE = 1


def run_local_test(local_test: LocalTest):
    if local_test == LocalTest.LEGEND_HEIGHT_TABLE:
        rows = []
        for fontsize in (2.5, 3.5, 5.0, 8.0):
            for n_entries in (5, 10, 15, 20, 30):
                rows.append(dict(fontsize=fontsize,
                                 n_entries=n_entries,
                                 measured=measure_legend_height(n_entries=n_entries,
                                                                fontsize=fontsize),
                                 model=fontsize * LEGEND_ROW_HEIGHT_PER_FONTSIZE * n_entries))
        print(pd.DataFrame(rows))


if __name__ == '__main__':
    run_local_test(local_test=LocalTest.LEGEND_HEIGHT_TABLE)
