"""Check benchmark event dates and the drawdown page through the public factsheet APIs."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.portfolio.reports._drawdown_analysis import plot_drawdown_analysis
from qis.portfolio.reports.tests.strategy_factsheet_test import _make_portfolio_data


@pytest.fixture
def chapter_paths():
    """Six observed month-ends used in the handbook."""
    dates = pd.date_range('2024-01-31', periods=6, freq='ME')
    return (pd.Series([100, 101, 103, 105, 103, 104], index=dates, name='Strategy'),
            pd.Series([100, 96, 89, 85, 94, 101], index=dates, name='Market'))


def test_chapter_bars_paths_and_episode_table(chapter_paths, tmp_path):
    """Reconcile every bar, both dated paths, strategy episode durations and open status."""
    strategy, market = chapter_paths
    dates = market.index
    event, = qis.compute_benchmark_drawdown_events(strategy, market).itertuples()
    assert (event.peak, event.trough, event.recovery) == (dates[0], dates[3], dates[5])
    fig = plot_drawdown_analysis(strategy, market, .10, None, (8.5, 11.7))
    try:
        fig.canvas.draw()
        table, paths, trough, recovery, selected = fig.axes
        expected = [(.05, -.15), (.04, .01)]
        for axis, pair in zip([trough, recovery], expected, strict=True):
            np.testing.assert_allclose([bars[0].get_width() for bars in axis.containers], pair)
        for line, nav in zip(selected.lines[:2], [strategy, market], strict=True):
            pd.testing.assert_index_equal(pd.DatetimeIndex(line.get_xdata()), dates)
            np.testing.assert_allclose(line.get_ydata(), 100 * nav / nav.iloc[0])
        assert pd.Timestamp(selected.lines[2].get_xdata()[0]) == dates[3]
        row = [table.tables[0][1, c].get_text().get_text() for c in range(6)]
        assert row == ['30 Apr 24', '31 May 24', '-1.9%', 'Open', '1', '—']
        path = next(line for line in paths.lines if len(line.get_xdata()))
        np.testing.assert_allclose(path.get_ydata(), np.array([105, 103, 104]) / 105 - 1)
        assert 'Maximum -1.9%' in table.get_title(loc='left')
        assert 'Current -1.0%' in table.get_title(loc='left')
        fig.savefig(tmp_path / 'chapter_13_drawdown.png', dpi=150)
    finally:
        plt.close(fig)


def test_missing_endpoints_and_open_recovery(chapter_paths):
    """Missing strategy observations affect only returns needing that endpoint."""
    strategy, market = chapter_paths
    for missing, expected in [(0, [np.nan, np.nan]), (3, [np.nan, .04]), (5, [.05, np.nan])]:
        nav = strategy.drop(strategy.index[missing])
        event = qis.compute_benchmark_drawdown_events(nav, market).iloc[0]
        np.testing.assert_allclose([event.strategy_trough_return, event.strategy_recovery_return],
                                   expected, equal_nan=True)
        np.testing.assert_allclose([event.benchmark_trough_return, event.benchmark_recovery_return],
                                   [-.15, .01])
    market = market.iloc[:-1]
    event = qis.compute_benchmark_drawdown_events(strategy, market).iloc[0]
    assert event.end == market.index[-1] and pd.isna(event.recovery)
    assert pd.isna(event.strategy_recovery_return) and pd.isna(event.benchmark_recovery_return)
    fig = plot_drawdown_analysis(strategy, market, .1, None, (8.5, 11.7))
    try:
        assert not fig.axes[3].patches
        assert [t.get_text() for t in fig.axes[3].texts] == ['n/a', 'n/a']
        assert pd.Timestamp(fig.axes[-1].lines[0].get_xdata()[-1]) == market.index[-1]
    finally:
        plt.close(fig)


def test_missing_peak_keeps_benchmark_and_missing_interior_breaks_path(chapter_paths):
    """Missing NAVs stay missing both in the plotted ratios and in the shared event data."""
    strategy, market = chapter_paths
    for missing in [0, 2]:
        fig = plot_drawdown_analysis(strategy.drop(strategy.index[missing]), market,
                                     .1, None, (8.5, 11.7))
        try:
            lines = {line.get_label(): line for line in fig.axes[-1].lines}
            np.testing.assert_allclose(lines['Market'].get_ydata(), market)
            if missing == 0:
                assert 'Strategy' not in lines
                assert all(len(ax.containers[0]) == 0 for ax in fig.axes[2:4])
            else:
                assert np.isnan(lines['Strategy'].get_ydata()[missing])
        finally:
            plt.close(fig)


def test_tied_peaks_troughs_and_named_windows():
    """The first plateau date and earliest trough win; a supplied local peak can split a fall."""
    dates = pd.date_range('2022-01-31', periods=8, freq='ME')
    market = pd.Series([100, 100, 85, 92, 72, 72, 94, 103], index=dates)
    strategy = pd.Series([100, 110, 103, 101, 90, 88, 99, 101], index=dates)
    automatic = qis.compute_benchmark_drawdown_events(strategy, market).iloc[0]
    assert (automatic.peak, automatic.trough, automatic.recovery) == (dates[0], dates[4], dates[7])
    windows = {'Second sell-off': qis.TimePeriod(dates[3], dates[4])}
    named = qis.compute_benchmark_drawdown_events(strategy, market, named_windows=windows).iloc[0]
    assert named['name'] == 'Second sell-off'
    assert (named.peak, named.trough, named.recovery) == (dates[3], dates[4], dates[6])
    assert named.strategy_trough_return == pytest.approx(90 / 101 - 1)
    assert named.benchmark_recovery_return == pytest.approx(94 / 92 - 1)
    with pytest.raises(ValueError, match='observed benchmark'):
        qis.compute_benchmark_drawdown_events(
            strategy, market.drop(dates[3]), named_windows=windows)


@pytest.mark.parametrize('levels', [[100, 90, 101], [100, 101, 102], [1.1, .99, 1.11]])
def test_strict_cutoff_and_empty_panels(levels):
    """An exact 10% drawdown and a rising benchmark both leave strategy panels in place."""
    dates = pd.date_range('2024-01-01', periods=3)
    market = pd.Series(levels, index=dates)
    strategy = pd.Series([100, 80, 102], index=dates)
    empty = qis.compute_benchmark_drawdown_events(strategy, market)
    assert empty.empty
    assert empty['recovery'].dtype == market.index.dtype
    fig = plot_drawdown_analysis(strategy, market, .1, None, (8.5, 11.7))
    try:
        assert len(fig.axes[0].tables) == 1
        assert any('No benchmark event' in t.get_text() for t in fig.axes[-1].texts)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('facade', [False, True])
def test_strategy_entry_points_and_opt_out(facade, tmp_path):
    """Render the full report and verify the selected original benchmark and page order."""
    portfolio, benchmarks = _make_portfolio_data()
    dates = portfolio.nav.index
    # A sparse, separately observed reference cannot be recovered from forward-filled prices.
    benchmarks = benchmarks.resample('ME').last().loc[:dates[-1]]
    monthly = pd.Series(np.linspace(60, 100, len(benchmarks)), index=benchmarks.index)
    monthly.iloc[-6:] = [100, 96, 89, 85, 94, 101]
    benchmarks = benchmarks.assign(Selected=monthly)
    options = dict(benchmark_prices=benchmarks, regime_benchmark='Selected',
                   add_performance_inference_sheet=False)
    make = qis.factsheet if facade else qis.generate_strategy_factsheet
    if facade:
        options['kind'] = 'strategy'
    opened = []
    try:
        default = make(portfolio, **options)
        opened.extend(default)
        off = make(portfolio, add_drawdown_analysis_sheet=False, **options)
        opened.extend(off)
        for fig in opened:
            fig.canvas.draw()
        assert len(default) == len(off) + 1
        assert [fig.get_suptitle() for fig in default[:-1]] == [fig.get_suptitle() for fig in off]
        assert sum('Drawdown analysis' in fig.get_suptitle() for fig in default) == 1
        page = default[-1]
        assert 'Benchmark: Selected' in page.get_suptitle()
        assert all('Drawdown analysis' not in fig.get_suptitle() for fig in off)
        line = next(line for line in page.axes[-1].lines if line.get_label() == 'Selected')
        pd.testing.assert_index_equal(pd.DatetimeIndex(line.get_xdata()), monthly.index[-6:],
                                      check_names=False)
        np.testing.assert_allclose(line.get_ydata(), monthly.iloc[-6:].to_numpy())
        np.testing.assert_allclose(np.asarray(default[0].canvas.buffer_rgba()),
                                   np.asarray(off[0].canvas.buffer_rgba()))
        page.savefig(tmp_path / 'strategy_drawdown.png', dpi=150)
    finally:
        for fig in opened:
            plt.close(fig)
