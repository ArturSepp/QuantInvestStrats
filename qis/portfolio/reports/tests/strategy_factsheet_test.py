"""Tests for the strategy factsheet's long-history monthly-return appendix."""

import inspect

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.plots.derived.returns_heatmap import plot_returns_heatmap
from qis.portfolio.reports.strategy_benchmark_factsheet import (
    generate_strategy_benchmark_factsheet_plt,
)
from qis.portfolio.reports.strategy_factsheet import generate_strategy_factsheet


def _make_portfolio_data() -> tuple[qis.PortfolioData, pd.DataFrame]:
    rng = np.random.default_rng(17)
    index = pd.bdate_range(end='2025-12-31', periods=6 * 260)
    returns = 0.0002 + 0.008 * rng.standard_normal((len(index), 3))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)),
                          index=index,
                          columns=['A', 'B', 'C'])
    rebalancing_dates = prices.resample('ME').last().index
    weights = pd.DataFrame(1.0 / 3.0, index=rebalancing_dates, columns=prices.columns)
    portfolio = qis.backtest_model_portfolio(prices=prices, weights=weights, ticker='Strategy')
    return portfolio, prices[['A']]


def test_monthly_returns_summary_defaults() -> None:
    strategy_parameters = inspect.signature(generate_strategy_factsheet).parameters
    benchmark_parameters = inspect.signature(generate_strategy_benchmark_factsheet_plt).parameters
    heatmap_parameters = inspect.signature(plot_returns_heatmap).parameters
    assert strategy_parameters['monthly_returns_heatmap_max_years'].default == 10
    assert strategy_parameters['fontsize'].default == 5
    assert benchmark_parameters['fontsize'].default == 5
    assert heatmap_parameters['fontsize'].default == 5
    assert 'heatmap_fontsize' not in strategy_parameters
    assert 'heatmap_fontsize' not in benchmark_parameters


def test_long_history_warns_limits_summary_and_appends_full_heatmap() -> None:
    portfolio, benchmark_prices = _make_portfolio_data()
    full_nav = portfolio.get_portfolio_nav()
    expected_rows = len(qis.compute_periodic_returns_table(prices=full_nav,
                                                           is_inverse_order=True))

    with pytest.warns(UserWarning, match='latest 3 calendar years'):
        figs = generate_strategy_factsheet(
            portfolio_data=portfolio,
            benchmark_prices=benchmark_prices,
            monthly_returns_heatmap_max_years=3,
            fontsize=7,
        )
    try:
        assert len(figs) == 2
        summary_ax = next(ax for ax in figs[0].axes
                          if ax.get_title() == 'Monthly Returns - Last 3 Calendar Years')
        appendix_ax = next(ax for ax in figs[1].axes
                           if ax.get_title() == 'Monthly Returns - Full History')
        periodic_returns_ax = next(ax for ax in figs[0].axes if ax.get_title() == 'YE-returns')
        assert len(summary_ax.get_yticklabels()) == 3
        assert len(appendix_ax.get_yticklabels()) == expected_rows
        assert {text.get_fontsize() for text in summary_ax.texts} == {7.0}
        assert {text.get_fontsize() for text in appendix_ax.texts} == {7.0}
        assert {text.get_fontsize() for text in periodic_returns_ax.texts} == {7.0}
        width, height = figs[1].get_size_inches()
        assert width > height
    finally:
        plt.close('all')


def test_monthly_returns_heatmap_limit_must_be_positive() -> None:
    portfolio, benchmark_prices = _make_portfolio_data()
    with pytest.raises(ValueError, match='monthly_returns_heatmap_max_years must be positive'):
        generate_strategy_factsheet(
            portfolio_data=portfolio,
            benchmark_prices=benchmark_prices,
            monthly_returns_heatmap_max_years=0,
        )


def test_removed_heatmap_fontsize_keyword_is_rejected() -> None:
    portfolio, benchmark_prices = _make_portfolio_data()
    with pytest.raises(TypeError, match='heatmap_fontsize was removed; use fontsize'):
        generate_strategy_factsheet(
            portfolio_data=portfolio,
            benchmark_prices=benchmark_prices,
            heatmap_fontsize=4,
        )

    multi_portfolio = qis.MultiPortfolioData(
        portfolio_datas=[portfolio, portfolio],
        benchmark_prices=benchmark_prices,
    )
    with pytest.raises(TypeError, match='heatmap_fontsize was removed; use fontsize'):
        generate_strategy_benchmark_factsheet_plt(
            multi_portfolio_data=multi_portfolio,
            heatmap_fontsize=4,
        )
