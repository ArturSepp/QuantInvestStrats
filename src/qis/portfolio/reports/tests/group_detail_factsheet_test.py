"""Keep ordinary asset classes distinct while bounding very large group universes."""
from dataclasses import replace
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets.synthetic import SYNTHETIC_UNIVERSE, generate_synthetic_prices
from qis.portfolio.signal_data import StrategySignalData
from qis.portfolio.reports.strategy_signal_factsheet import (
    generate_current_signal_report, generate_weight_change_report,
)


GROUPS = ('EQ', 'BD', 'STIR', 'FX', 'EN', 'MT', 'AG')


@pytest.fixture(scope='module')
def grouped_portfolio():
    """84 synthetic holdings with the same seven-group structure as the CTA report."""
    labels = np.repeat(GROUPS, [18, 14, 4, 9, 6, 6, 27])
    instruments = tuple(
        replace(SYNTHETIC_UNIVERSE[i % 10], ticker=f'S{i:02d}', group=group)
        for i, group in enumerate(labels)
    )
    prices = generate_synthetic_prices(
        instruments=instruments, start='2023-01-02', end='2025-12-31', apply_quirks=False,
    )
    # Predetermined long/short weights, not fitted to realised returns.
    target = np.where(np.arange(84) % 3 == 0, -1.0, 1.0) / 84
    portfolio = qis.backtest_model_portfolio(
        prices=prices, weights=target, rebalancing_freq='ME', ticker='Synthetic CTA',
    )
    portfolio.set_group_data(pd.Series(labels, index=prices.columns), group_order=list(GROUPS))
    returns = qis.to_returns(prices, is_log_returns=True).fillna(0.0)
    signal = np.tanh(20.0 * returns)
    portfolio.strategy_signal_data = StrategySignalData(
        log_returns=returns, signal=signal, momentum=signal,
        instrument_vols=0.2 + returns.abs(), instrument_target_vols=0.1 + returns.abs(),
        instrument_target_signal_vol_weights=0.1 * signal,
        instrument_portfolio_leverages=1.0 + returns, weights=portfolio.weights,
    )
    return portfolio


@pytest.fixture(autouse=True)
def close_figures():
    existing = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - existing:
        plt.close(number)


def _draw_and_check(figure):
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        figure.canvas.draw()
    assert not any('layout not applied' in str(record.message) for record in records)
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        legend = axis.get_legend()
        if legend is not None:
            assert legend.get_window_extent(renderer).height <= axis.bbox.height


def test_seven_group_strategy_appendices(grouped_portfolio, tmp_path):
    """All five appendices retain named groups; totals/series and inputs are unchanged."""
    portfolio = grouped_portfolio
    weights_before = portfolio.weights.copy(deep=True)
    nav_before = portfolio.nav.copy(deep=True)
    groups_before = portfolio.group_data.copy(deep=True)
    period = qis.TimePeriod('31Dec2024', '31Dec2025')
    figures = qis.generate_strategy_factsheet(
        portfolio, portfolio.prices.iloc[:, 0].rename('Market'),
        time_period=period, is_grouped=True, add_grouped_weights_sheet=True,
        add_current_signal_report=True, add_weight_change_report=True,
        add_grouped_exposures=True, add_grouped_cum_pnl=True, fontsize=4,
    )
    assert len(figures) == 7
    weights, signals, changes, exposures, pnl = figures[2:]
    for page in figures[2:]:
        assert not any('All instruments' in axis.get_title() for axis in page.axes)
        _draw_and_check(page)
    assert [axis.get_title().splitlines()[0] for axis in weights.axes[2:]] == [
        f'{group} by instrument' for group in GROUPS]
    for page in (signals, changes):
        assert [axis.get_title().splitlines()[0] for axis in page.axes[1:]] == list(GROUPS)
    for page in (exposures, pnl):
        assert [axis.get_title().splitlines()[0] for axis in page.axes] == [
            f'{group} {suffix}' for group in GROUPS for suffix in ('aggregated', 'by instrument')
        ]
    # Independent display reference: sum original holdings in each class, not compact_groups.
    sampled = portfolio.get_weights(time_period=period, freq='W-WED')
    for i, group in enumerate(GROUPS):
        columns = portfolio.group_data.index[portfolio.group_data == group]
        expected = sampled[columns].sum(axis=1)
        np.testing.assert_allclose(exposures.axes[2 * i].lines[0].get_ydata(), expected)
    pd.testing.assert_frame_equal(portfolio.weights, weights_before)
    pd.testing.assert_series_equal(portfolio.nav, nav_before)
    pd.testing.assert_series_equal(portfolio.group_data, groups_before)
    page_names = ('weights', 'signals', 'weight-changes', 'exposures', 'pnl')
    for name, page in zip(page_names, figures[2:]):
        page.savefig(tmp_path / f'{name}.png', dpi=140)


@pytest.mark.parametrize('num_groups', [6, 10, 11])
@pytest.mark.parametrize('report', ['signals', 'changes'])
def test_signal_report_group_limit(grouped_portfolio, num_groups, report):
    """Overview totals are not a class; 11 actual groups still use the compact fallback."""
    portfolio = replace(grouped_portfolio)
    names = [f'G{i}' for i in range(num_groups)]
    portfolio.set_group_data(pd.Series(
        [names[i % num_groups] for i in range(84)], index=portfolio.prices.columns,
    ), group_order=names)
    if report == 'signals':
        figure = generate_current_signal_report(portfolio, fontsize=4)
    else:
        figure = generate_weight_change_report(
            portfolio, qis.TimePeriod('31Dec2024', '31Dec2025'), fontsize=4,
        )
    titles = [axis.get_title().splitlines()[0] for axis in figure.axes if axis.get_title()]
    assert titles[1:] == (names if num_groups <= 10 else ['All instruments'])
    _draw_and_check(figure)
