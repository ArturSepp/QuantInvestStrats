"""
the attribution panel's tail reduction: the geometry it engages on, and what it preserves.

An attribution panel draws one bar per instrument with the tick labels rotated 90 degrees, and a
rotated label occupies the font's line height horizontally whatever the name says. Past roughly
sixty instruments on a half-page panel the labels overlap and the panel names nothing, so
``plot_performance_attribution`` falls back to the tails and states the folded remainder in the
title rather than plotting it as a bar that would dominate the axis.

The two calibration tests are the ones that matter: they re-measure
``BAR_LABEL_WIDTH_PER_FONTSIZE`` and ``AXIS_SHARE_OF_CELL`` against matplotlib rather than
trusting them, so a matplotlib release that invalidates either fails here rather than silently in
a report. The rest pin the reduction's arithmetic and its wiring.
"""
# packages
import matplotlib
matplotlib.use('Agg')  # noqa: E402  - a headless backend, set before pyplot is imported
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest
# qis
import qis as qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.plots.utils import (AXIS_SHARE_OF_CELL,
                             BAR_LABEL_WIDTH_PER_FONTSIZE,
                             estimate_axis_width,
                             estimate_bar_label_capacity)
from qis.portfolio.portfolio_data import reduce_attribution_to_tails

TWO_SIDED = pd.Series([5.0, 3.0, 1.0, 0.4, 0.1, -0.2, -1.0, -4.0], index=list('abcdefgh'))
ONE_SIDED = pd.Series([5.0, 3.0, 1.0, 0.4, 0.1, 0.05], index=list('abcdef'))


def build_portfolio_data() -> qis.PortfolioData:
    """an equal-weight portfolio over the seeded synthetic panel"""
    prices = qis.TimePeriod('31Dec2020', '31Dec2025').locate(generate_synthetic_prices())
    weights = pd.DataFrame(1.0 / len(prices.columns), index=prices.index, columns=prices.columns)
    return qis.backtest_model_portfolio(prices=prices, weights=weights.iloc[::21, :],
                                        ticker='Test portfolio')


def test_performance_attribution_uses_instrument_display_names() -> None:
    """P&L attribution uses the same configured names as P&L-risk attribution."""
    portfolio_data = build_portfolio_data()
    names = {
        ticker: f'Asset name {idx}'
        for idx, ticker in enumerate(portfolio_data.prices.columns)
    }
    portfolio_data.tickers_to_names_map = names

    pnl = portfolio_data.get_performance_attribution_data(
        attribution_metric=qis.AttributionMetric.PNL
    )
    pnl_risk = portfolio_data.get_performance_attribution_data(
        attribution_metric=qis.AttributionMetric.PNL_RISK
    )

    assert pnl.index.tolist() == list(names.values())
    assert pnl_risk.index.tolist() == list(names.values())


def test_brinson_keeps_canonical_alignment_with_display_names() -> None:
    """Display names must not change Brinson groups or attribution values."""
    strategy = build_portfolio_data()
    benchmark_weights = {
        ticker: weight
        for ticker, weight in zip(
            strategy.prices.columns,
            np.linspace(1.0, 2.0, len(strategy.prices.columns)),
        )
    }
    total_weight = sum(benchmark_weights.values())
    benchmark_weights = {
        ticker: weight / total_weight
        for ticker, weight in benchmark_weights.items()
    }
    benchmark = qis.backtest_model_portfolio(
        prices=strategy.prices,
        weights=benchmark_weights,
        ticker='Benchmark',
    )
    group_order = ['EQ', 'FI', 'ALTS', 'Cash']
    group_data = pd.Series(
        [group_order[idx % len(group_order)]
         for idx in range(len(strategy.prices.columns))],
        index=strategy.prices.columns,
    )
    for portfolio in (strategy, benchmark):
        portfolio.set_group_data(group_data=group_data, group_order=group_order)
    multi_portfolio = qis.MultiPortfolioData(
        portfolio_datas=[strategy, benchmark],
        benchmark_prices=strategy.prices.iloc[:, [0]],
    )
    expected = multi_portfolio.compute_brinson_attribution(freq='ME')

    display_names = {
        ticker: f'Asset name {idx}'
        for idx, ticker in enumerate(strategy.prices.columns)
    }
    for portfolio in (strategy, benchmark):
        portfolio.tickers_to_names_map = display_names
    actual = multi_portfolio.compute_brinson_attribution(freq='ME')

    for expected_frame, actual_frame in zip(expected, actual):
        pd.testing.assert_frame_equal(actual_frame, expected_frame)


def test_bar_label_width_calibration() -> None:
    """the measured width of a rotated tick label matches the constant the capacity is built on"""
    labels = ['S&p_500', 'Wheat_minneapol', 'Ust_10y_ultra', 'Gasoil', 'Feeder_cattle']
    for fontsize in (4.0, 5.0, 8.0):
        # Use a high-DPI canvas so integer-pixel font hinting does not dominate a
        # physical-width calibration at these deliberately small font sizes.
        fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=300)
        qis.plot_bars(df=pd.Series(np.arange(len(labels), dtype=float), index=labels),
                      stacked=False, skip_y_axis=True, x_rotation=90, fontsize=fontsize,
                      legend_loc=None, ax=ax)
        fig.canvas.draw()
        width = max(t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
                    for t in ax.get_xticklabels())
        plt.close(fig)
        expected = BAR_LABEL_WIDTH_PER_FONTSIZE * fontsize
        assert width == pytest.approx(expected, rel=0.15), (
            f"rotated label width at fontsize={fontsize} measured {width:.5f} in against the "
            f"calibrated {expected:.5f} in: BAR_LABEL_WIDTH_PER_FONTSIZE needs re-measuring"
        )


def test_estimate_axis_width_calibration() -> None:
    """the pre-draw width estimate matches the width the axis actually gets"""
    for colspan, label in ((slice(2, 4), 'half width'), (slice(0, 4), 'full width')):
        fig = plt.figure(figsize=(8.5, 11.7), constrained_layout=True)
        gs = fig.add_gridspec(nrows=14, ncols=4, wspace=0.0, hspace=0.0)
        ax = fig.add_subplot(gs[8:10, colspan])
        qis.plot_bars(df=pd.Series([1.0, 2.0, 3.0], index=['a', 'b', 'c']), stacked=False,
                      skip_y_axis=True, x_rotation=90, fontsize=5, legend_loc=None, ax=ax)
        estimated = estimate_axis_width(ax=ax)
        fig.canvas.draw()
        drawn = ax.get_window_extent().width / fig.dpi
        plt.close(fig)
        assert estimated == pytest.approx(drawn, rel=0.05), (
            f"{label}: estimated {estimated:.3f} in against a drawn {drawn:.3f} in: "
            f"AXIS_SHARE_OF_CELL={AXIS_SHARE_OF_CELL} needs re-measuring"
        )


def test_estimate_bar_label_capacity() -> None:
    """the capacity of the panel geometries that ship, and its monotonicity"""
    assert estimate_bar_label_capacity(axis_width=4.09, fontsize=5) == 45   # half-page panel
    assert estimate_bar_label_capacity(axis_width=8.19, fontsize=5) == 92   # full-page panel
    # smaller type holds more labels, a narrower axis holds fewer
    assert (estimate_bar_label_capacity(axis_width=4.09, fontsize=4)
            > estimate_bar_label_capacity(axis_width=4.09, fontsize=5))
    assert (estimate_bar_label_capacity(axis_width=2.0, fontsize=5)
            < estimate_bar_label_capacity(axis_width=4.09, fontsize=5))
    # a panel too narrow for even one label still reports one, so nothing divides by zero
    assert estimate_bar_label_capacity(axis_width=0.01, fontsize=5) == 1


def test_estimate_bar_label_capacity_raises() -> None:
    """arguments that would make the capacity meaningless are rejected"""
    with pytest.raises(ValueError, match='axis_width'):
        estimate_bar_label_capacity(axis_width=0.0, fontsize=5)
    with pytest.raises(ValueError, match='fontsize'):
        estimate_bar_label_capacity(axis_width=4.0, fontsize=0.0)


def test_reduce_two_sided_keeps_both_tails() -> None:
    """signed data is cut from both ends and the total is preserved"""
    kept, folded = reduce_attribution_to_tails(data=TWO_SIDED, max_bars=4)
    assert list(kept.index) == ['a', 'b', 'g', 'h']
    assert kept.sum() + folded == pytest.approx(TWO_SIDED.sum())
    assert folded == pytest.approx(TWO_SIDED[['c', 'd', 'e', 'f']].sum())
    # an odd budget gives the extra bar to the bottom tail
    kept, _ = reduce_attribution_to_tails(data=TWO_SIDED, max_bars=5)
    assert list(kept.index) == ['a', 'b', 'f', 'g', 'h']


def test_reduce_one_sided_keeps_the_top_only() -> None:
    """a share-of-total metric has no bottom tail, so only the top is kept"""
    kept, folded = reduce_attribution_to_tails(data=ONE_SIDED, max_bars=3)
    assert list(kept.index) == ['a', 'b', 'c']
    assert kept.sum() + folded == pytest.approx(ONE_SIDED.sum())


def test_reduce_below_budget_only_sorts() -> None:
    """nothing is folded when everything fits, and the result is sorted"""
    kept, folded = reduce_attribution_to_tails(data=TWO_SIDED, max_bars=len(TWO_SIDED.index))
    assert folded == 0.0
    assert list(kept.index) == list(TWO_SIDED.sort_values(ascending=False).index)


def test_reduce_raises_on_empty_budget() -> None:
    with pytest.raises(ValueError, match='max_bars'):
        reduce_attribution_to_tails(data=TWO_SIDED, max_bars=0)


def test_plot_performance_attribution_reduces_only_when_crowded() -> None:
    """the panel reduces on an axis too narrow to label, and is untouched on a wide one"""
    portfolio_data = build_portfolio_data()

    def bar_count(figsize, max_bars, fontsize=5) -> int:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(nrows=14, ncols=4, wspace=0.0, hspace=0.0)
        ax = fig.add_subplot(gs[8:10, 2:])
        portfolio_data.plot_performance_attribution(attribution_metric=qis.AttributionMetric.PNL,
                                                    max_bars=max_bars, fontsize=fontsize,
                                                    legend_loc=None, ax=ax)
        count = len(ax.patches)
        plt.close(fig)
        return count

    # the panel draws one bar per instrument that traded, which is fewer than the universe:
    # remove_zero_data drops the ones with no attribution in the window
    n_bars = bar_count(figsize=(8.5, 11.7), max_bars=0)
    # a 1.34in panel at fontsize 12 holds 7 labels, fewer than the bars, so it is cut down
    assert bar_count(figsize=(2.8, 11.7), max_bars=None, fontsize=12) < n_bars
    # the A4 panel at fontsize 5 holds 45, so nothing is reduced
    assert bar_count(figsize=(8.5, 11.7), max_bars=None) == n_bars
    # max_bars=0 keeps every bar however narrow the panel
    assert bar_count(figsize=(2.8, 11.7), max_bars=0, fontsize=12) == n_bars


def test_plot_performance_attribution_states_the_fold() -> None:
    """the title carries what was folded away, so the stated sum cannot be misread"""
    portfolio_data = build_portfolio_data()
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    portfolio_data.plot_performance_attribution(attribution_metric=qis.AttributionMetric.PNL,
                                                max_bars=4, fontsize=5, legend_loc=None, ax=ax)
    title, n_bars = ax.get_title(), len(ax.patches)
    plt.close(fig)
    assert n_bars == 4
    assert 'top and bottom 4 of' in title
    assert 'folded away, summing to' in title


class LocalTest:
    """runnable checks, not part of the pytest suite"""
    CAPACITY_TABLE = 1


def run_local_test(local_test: LocalTest):
    if local_test == LocalTest.CAPACITY_TABLE:
        rows = []
        for axis_width in (2.0, 4.09, 6.0, 8.19):
            for fontsize in (3.0, 4.0, 5.0, 8.0):
                rows.append(dict(axis_width=axis_width, fontsize=fontsize,
                                 capacity=estimate_bar_label_capacity(axis_width=axis_width,
                                                                      fontsize=fontsize)))
        print(pd.DataFrame(rows).pivot(index='axis_width', columns='fontsize', values='capacity'))


if __name__ == '__main__':
    run_local_test(local_test=LocalTest.CAPACITY_TABLE)
