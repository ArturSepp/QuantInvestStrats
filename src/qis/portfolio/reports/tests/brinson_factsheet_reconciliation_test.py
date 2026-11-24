"""Reconcile Chapter 11 attribution through the public strategy–benchmark factsheet."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.portfolio.reports.brinson_attribution import plot_brinson_totals_table


@pytest.fixture
def chapter_11_pair():
    """Fund separate instruments in each asset class and impose the two observed intervals."""
    history = generate_synthetic_prices(start='2020-01-02', end='2024-12-31', apply_quirks=False)
    groups = ['Equities', 'Bonds', 'Commodities']
    baseline, first_close, second_close = history.index[-3:]
    portfolios = []
    for name, tickers, weights, first_returns, second_returns in [
        ('Strategy', ['SEQ_US', 'SBD_TSY', 'SCM_GLD'], [.65, .25, .10],
         [.16, .06, .11], [-.04, .03, .02]),
        ('Benchmark', ['SEQ_EU', 'SBD_IG', 'SCM_BCOM'], [.50, .45, .05],
         [.15, .08, .10], [.01, -.02, .03]),
    ]:
        prices = history[tickers].copy()
        prices.loc[first_close] = prices.loc[baseline] * (1 + np.array(first_returns))
        prices.loc[second_close] = prices.loc[first_close] * (1 + np.array(second_returns))
        # Reset the opening weights at the book baseline, then hold units through both closes.
        schedule = pd.DataFrame([weights, weights], columns=tickers,
                                index=[prices.index[0], baseline])
        portfolio = qis.backtest_model_portfolio(
            prices, schedule, rebalancing_costs=0.0, ticker=name)
        portfolio.set_group_data(pd.Series(groups, index=tickers), group_order=groups)
        portfolios.append(portfolio)

    return qis.MultiPortfolioData(portfolios, benchmark_prices=history[['SEQ_EM']])


def opening_active_weights(page):
    """Read active weight bars against their displayed labels."""
    exposure = next(panel for panel in page.axes if panel.get_title() == 'Opening active weights')
    labels = [tick.get_text() for tick in exposure.get_yticklabels()]
    return dict(zip(labels, [bar.get_width() for bar in exposure.containers[0]], strict=True))


def test_chapter_11_brinson_factsheet(chapter_11_pair, tmp_path):
    """Compare the rendered table with the book, then check a second interval against NAVs."""
    pair = chapter_11_pair
    strategy, benchmark = pair.portfolio_datas
    groups = ['Equities', 'Bonds', 'Commodities']
    baseline, first_close, second_close = strategy.nav.index[-3:]
    period = qis.TimePeriod(baseline, first_close)
    target = 'qis.portfolio.reports.brinson_attribution.plot_brinson_totals_table'
    with patch(target, wraps=plot_brinson_totals_table) as renderer:
        figures = qis.factsheet(pair, kind='strategy_benchmark', brinson_time_period=period,
                                var_format='{:.2%}')
    try:
        renderer.assert_called_once()
        rendered = renderer.call_args.kwargs['totals_table']
        axis = renderer.call_args.kwargs['ax']
        page = axis.figure
        assert page in figures
        page.canvas.draw()
        assert f'{baseline:%d%b%Y} - {first_close:%d%b%Y}' in page.get_suptitle()
        assert 'Gross of realised trading costs' in page.get_suptitle()
        assert 'Interaction returns added 100% to instrument selection' in page.get_suptitle()

        rows = groups + ['Total Sum']
        effects = ['Asset\nAllocation', 'Instrument\nSelection', 'Total\nActive']
        split, *_ = pair.compute_brinson_attribution(
            time_period=period, is_linked=False, is_exclude_interaction_term=False)
        expected = split.loc[rows, effects].copy()
        expected['Instrument\nSelection'] += split.loc[rows, 'Interaction']
        np.testing.assert_allclose(rendered.loc[rows, effects], expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(rendered.loc[rows, effects] * 10_000,
                                   [[225, 65, 290], [-160, -50, -210], [50, 10, 60],
                                    [115, 25, 140]], rtol=0, atol=1e-8)
        cells = axis.tables[0].get_celld()
        total_row = next(i for (i, j), cell in cells.items()
                         if cell.get_text().get_text() == 'Total Sum')
        text = [cell.get_text().get_text() for (i, j), cell in sorted(cells.items())
                if i == total_row]
        assert text[-5:] == ['13.00%', '11.60%', '1.15%', '0.25%', '1.40%']
        for panel in page.axes[1:5]:
            observations = [line for line in panel.lines if len(line.get_ydata()) == 1]
            assert observations and all(line.get_marker() not in ('', 'None', None)
                                        for line in observations)
        bars = opening_active_weights(page)
        np.testing.assert_allclose([bars[group] for group in groups], [.15, -.20, .05], atol=1e-12)
        page.savefig(tmp_path / 'chapter_11_brinson.png', dpi=145)
    finally:
        for figure in figures:
            plt.close(figure)

    # Direct NAV growth includes drifted holdings in the second interval.
    _, increments, *_ = pair.compute_brinson_attribution(
        time_period=qis.TimePeriod(baseline, second_close))
    growth_difference = (strategy.nav / strategy.nav.loc[baseline]
                         - benchmark.nav / benchmark.nav.loc[baseline])
    assert increments.index.tolist() == [first_close, second_close]
    np.testing.assert_allclose(increments.sum(axis=1).cumsum(),
                               growth_difference.loc[increments.index], rtol=0, atol=1e-12)


def test_opening_active_weights_for_one_sided_classes(chapter_11_pair):
    """Keep the full signed position when a class is present in only one portfolio."""
    pair = chapter_11_pair
    strategy, benchmark = pair.portfolio_datas
    strategy.set_group_data(strategy.group_data.replace({'Commodities': 'Gold'}))
    benchmark.set_group_data(benchmark.group_data.replace({'Commodities': 'Broad commodity'}))
    baseline, close, _ = strategy.nav.index[-3:]
    figures = qis.factsheet(pair, kind='strategy_benchmark', add_performance_inference_sheet=False,
                            brinson_time_period=qis.TimePeriod(baseline, close))
    try:
        page = next(fig for fig in figures if 'Brinson' in fig.get_suptitle())
        bars = opening_active_weights(page)
        np.testing.assert_allclose([bars['Gold'], bars['Broad commodity']], [.10, -.05], atol=1e-12)
    finally:
        for figure in figures:
            plt.close(figure)
