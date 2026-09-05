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
from qis.portfolio.reports.config import _get_recent_ra_perf_table_time_period
from qis.portfolio.reports.strategy_benchmark_factsheet import (
    generate_strategy_benchmark_factsheet_plt,
)
from qis.portfolio.reports.multi_assets_factsheet import (
    MultiAssetsReport,
    generate_multi_asset_factsheet,
)
from qis.portfolio.reports.multi_strategy_factsheet import generate_multi_portfolio_factsheet
from qis.portfolio.reports import strategy_factsheet
from qis.portfolio.reports.strategy_factsheet import generate_strategy_factsheet


def _make_portfolio_data(
        n_assets: int = 3,
        n_years: int = 6,
) -> tuple[qis.PortfolioData, pd.DataFrame]:
    rng = np.random.default_rng(17)
    index = pd.bdate_range(end='2025-12-31', periods=n_years * 260)
    returns = 0.0002 + 0.008 * rng.standard_normal((len(index), n_assets))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)),
                          index=index,
                          columns=[f'Asset {idx + 1}' for idx in range(n_assets)])
    rebalancing_dates = prices.resample('ME').last().index
    weights = pd.DataFrame(1.0 / n_assets, index=rebalancing_dates, columns=prices.columns)
    portfolio = qis.backtest_model_portfolio(prices=prices, weights=weights, ticker='Strategy')
    return portfolio, prices.iloc[:, [0]].rename(columns={prices.columns[0]: 'Benchmark'})


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


def test_recent_ra_perf_table_start_date_public_defaults() -> None:
    expected = pd.Timestamp('2020-12-31')
    generators = (
        generate_strategy_factsheet,
        generate_strategy_benchmark_factsheet_plt,
        generate_multi_portfolio_factsheet,
        generate_multi_asset_factsheet,
    )

    for generator in generators:
        parameter = inspect.signature(generator).parameters['recent_ra_perf_table_start_date']
        assert parameter.default == expected


def test_recent_ra_perf_table_time_period_default_and_trailing_year() -> None:
    report_period = qis.TimePeriod(start='2005-01-01', end='2026-08-31')

    recent_period = _get_recent_ra_perf_table_time_period(time_period=report_period)
    trailing_period = _get_recent_ra_perf_table_time_period(
        time_period=report_period,
        recent_ra_perf_table_start_date=None,
    )

    assert recent_period.start == pd.Timestamp('2020-12-31')
    assert recent_period.end == pd.Timestamp('2026-08-31')
    assert trailing_period.start == pd.Timestamp('2025-08-31')
    assert trailing_period.end == pd.Timestamp('2026-08-31')


def test_strategy_and_multi_asset_factsheets_use_recent_ra_start_date(monkeypatch) -> None:
    portfolio, benchmark_prices = _make_portfolio_data()
    strategy_periods = []

    def capture_strategy_period(*args, **kwargs):
        strategy_periods.append(kwargs['time_period'])
        return kwargs['ax']

    monkeypatch.setattr(portfolio, 'plot_ra_perf_table', capture_strategy_period)
    generate_strategy_factsheet(
        portfolio_data=portfolio,
        benchmark_prices=benchmark_prices,
        recent_ra_perf_table_start_date=pd.Timestamp('2022-06-30'),
    )
    try:
        assert strategy_periods[1].start == pd.Timestamp('2022-06-30')
        assert strategy_periods[1].end == pd.Timestamp('2025-12-31')
    finally:
        plt.close('all')

    multi_asset_periods = []

    def capture_multi_asset_period(self, *args, **kwargs):
        multi_asset_periods.append(kwargs['time_period'])
        return kwargs['ax']

    monkeypatch.setattr(MultiAssetsReport, 'plot_ra_perf_table', capture_multi_asset_period)
    generate_multi_asset_factsheet(
        prices=portfolio.prices,
        benchmark=portfolio.prices.columns[0],
        drop_1y_ra_perf_table=False,
        recent_ra_perf_table_start_date=pd.Timestamp('2021-03-31'),
    )
    try:
        assert multi_asset_periods[1].start == pd.Timestamp('2021-03-31')
        assert multi_asset_periods[1].end == pd.Timestamp('2025-12-31')
    finally:
        plt.close('all')


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
        assert height > width
        appendix_position = appendix_ax.get_position()
        assert appendix_position.width < 0.9
        assert appendix_position.height < 0.8
    finally:
        plt.close('all')


def test_full_history_heatmap_layout_adapts_to_year_rows() -> None:
    short = strategy_factsheet._get_monthly_returns_appendix_bounds(
        num_years=10, num_columns=13, figsize=(8.5, 11.7))
    medium = strategy_factsheet._get_monthly_returns_appendix_bounds(
        num_years=22, num_columns=13, figsize=(8.5, 11.7))
    long = strategy_factsheet._get_monthly_returns_appendix_bounds(
        num_years=40, num_columns=13, figsize=(8.5, 11.7))

    assert short[3] < medium[3] < long[3]
    assert short[2] == pytest.approx(medium[2])
    assert long[2] < medium[2]
    for left, bottom, width, height in (short, medium, long):
        assert 0.0 < left < 1.0
        assert 0.0 < bottom < 1.0
        assert 0.0 < width < 1.0
        assert 0.0 < height < 1.0
        assert left + width < 1.0
        assert bottom + height < 1.0


def test_more_than_ten_groups_use_strategy_only_summary_tables() -> None:
    portfolio, benchmark_prices = _make_portfolio_data(n_assets=11, n_years=3)

    with pytest.warns(UserWarning, match='11 portfolio groups.*maximum of 10') as warnings_:
        figs = generate_strategy_factsheet(
            portfolio_data=portfolio,
            benchmark_prices=benchmark_prices,
            is_grouped=True,
        )
    try:
        assert sum('portfolio groups' in str(warning.message) for warning in warnings_) == 1
        titles = [ax.get_title() for ax in figs[0].axes]
        assert any(title.startswith('RA performance table  for') for title in titles)
        assert 'YE-returns' in titles
        assert any(title.startswith('Sharpe ratio attribution to Benchmark') for title in titles)
        assert not any('with portfolio groups' in title for title in titles)
        assert not any('returns by groups' in title for title in titles)
        assert not any('attribution by groups' in title for title in titles)
    finally:
        plt.close('all')


def test_strategy_benchmark_factsheet_limits_grouped_ra_tables(monkeypatch) -> None:
    strategy, benchmark_prices = _make_portfolio_data(n_assets=11, n_years=3)
    benchmark_portfolio, _ = _make_portfolio_data(n_assets=11, n_years=3)
    benchmark_portfolio.set_ticker('Benchmark Portfolio')
    multi_portfolio = qis.MultiPortfolioData(
        portfolio_datas=[strategy, benchmark_portfolio],
        benchmark_prices=benchmark_prices,
    )
    grouped_arguments = []
    recent_periods = []
    original_plot = multi_portfolio.plot_ac_ra_perf_table

    def capture_grouping(*args, **kwargs):
        grouped_arguments.append(kwargs['is_grouped'])
        recent_periods.append(kwargs.get('time_period'))
        return original_plot(*args, **kwargs)

    monkeypatch.setattr(multi_portfolio, 'plot_ac_ra_perf_table', capture_grouping)
    with pytest.warns(UserWarning, match='11 portfolio groups.*maximum of 10') as warnings_:
        generate_strategy_benchmark_factsheet_plt(
            multi_portfolio_data=multi_portfolio,
            add_brinson_attribution=False,
            is_grouped=True,
        )
    try:
        assert sum('portfolio groups' in str(warning.message) for warning in warnings_) == 1
        assert grouped_arguments == [False, False]
        assert recent_periods[1].start == pd.Timestamp('2020-12-31')
        assert recent_periods[1].end == pd.Timestamp('2025-12-31')
    finally:
        plt.close('all')


def test_multi_strategy_factsheet_limits_grouped_sharpe(monkeypatch) -> None:
    strategy, benchmark_prices = _make_portfolio_data(n_assets=11, n_years=3)
    second_strategy, _ = _make_portfolio_data(n_assets=11, n_years=3)
    second_strategy.set_ticker('Second Strategy')
    multi_portfolio = qis.MultiPortfolioData(
        portfolio_datas=[strategy, second_strategy],
        benchmark_prices=benchmark_prices,
    )
    grouped_arguments = []
    original_plot = multi_portfolio.plot_regime_data

    def capture_grouping(*args, **kwargs):
        grouped_arguments.append(kwargs['is_grouped'])
        return original_plot(*args, **kwargs)

    monkeypatch.setattr(multi_portfolio, 'plot_regime_data', capture_grouping)
    with pytest.warns(UserWarning, match='11 portfolio groups.*maximum of 10') as warnings_:
        generate_multi_portfolio_factsheet(
            multi_portfolio_data=multi_portfolio,
            group_data=strategy.group_data,
        )
    try:
        assert sum('portfolio groups' in str(warning.message) for warning in warnings_) == 1
        assert grouped_arguments == [False]
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


def test_brinson_assigns_interaction_to_selection_and_preserves_active_return() -> None:
    index = pd.DatetimeIndex(['2025-01-31'])
    columns = ['Equity', 'Bonds']
    group_data = pd.Series(columns, index=columns)
    strategy_pnl = pd.DataFrame([[0.06, 0.02]], index=index, columns=columns)
    benchmark_pnl = pd.DataFrame([[0.03, 0.01]], index=index, columns=columns)
    strategy_weights = pd.DataFrame([[0.60, 0.40]], index=index, columns=columns)
    benchmark_weights = pd.DataFrame([[0.50, 0.50]], index=index, columns=columns)

    raw = qis.compute_brinson_attribution_table(
        benchmark_pnl=benchmark_pnl,
        strategy_pnl=strategy_pnl,
        strategy_weights=strategy_weights,
        benchmark_weights=benchmark_weights,
        asset_class_data=group_data,
        is_exclude_interaction_term=False,
    )
    split = qis.compute_brinson_attribution_table(
        benchmark_pnl=benchmark_pnl,
        strategy_pnl=strategy_pnl,
        strategy_weights=strategy_weights,
        benchmark_weights=benchmark_weights,
        asset_class_data=group_data,
        is_exclude_interaction_term=True,
    )
    raw_allocation, raw_selection, raw_interaction = raw[2:]
    split_allocation, split_selection, split_interaction = split[2:]

    pd.testing.assert_frame_equal(split_allocation, raw_allocation)
    pd.testing.assert_frame_equal(split_selection, raw_selection + raw_interaction)
    assert np.allclose(split_interaction.to_numpy(), 0.0)
    pd.testing.assert_frame_equal(
        split_allocation + split_selection,
        raw_allocation + raw_selection + raw_interaction,
    )


def test_brinson_page_uses_requested_layout_titles_and_regime_backgrounds() -> None:
    strategy, benchmark_prices = _make_portfolio_data(n_assets=3, n_years=3)
    benchmark, _ = _make_portfolio_data(n_assets=3, n_years=3)
    benchmark.set_ticker('Benchmark Portfolio')
    group_order = ['Equity', 'Fixed Income', 'Alternatives']
    group_data = pd.Series(group_order, index=strategy.prices.columns)
    for portfolio in (strategy, benchmark):
        portfolio.set_group_data(group_data=group_data, group_order=group_order)
    multi_portfolio = qis.MultiPortfolioData(
        portfolio_datas=[strategy, benchmark],
        benchmark_prices=benchmark_prices,
    )

    figs = generate_strategy_benchmark_factsheet_plt(
        multi_portfolio_data=multi_portfolio,
        backtest_name='TAA vs Benchmark',
    )
    try:
        brinson_page = figs[1]
        brinson_page.canvas.draw()
        axes_by_title = {ax.get_title(): ax for ax in brinson_page.axes if ax.get_title()}
        expected_titles = {
            'Cumulative Active Attribution Effects',
            'Total Cumulative Active Effects by Groups',
            'Cumulative Asset Class Allocation Effects',
            'Cumulative Instrument Selection Effects',
            'Net exposure diff Strategy-Benchmark Portfolio',
        }
        assert set(axes_by_title) == expected_titles
        assert 'Interaction returns added 100% to instrument selection' in (
            brinson_page._suptitle.get_text()
        )

        active_total = axes_by_title['Cumulative Active Attribution Effects']
        active_by_group = axes_by_title['Total Cumulative Active Effects by Groups']
        allocation = axes_by_title['Cumulative Asset Class Allocation Effects']
        selection = axes_by_title['Cumulative Instrument Selection Effects']
        exposure = axes_by_title['Net exposure diff Strategy-Benchmark Portfolio']
        assert active_by_group.get_position().x0 > active_total.get_position().x0
        assert active_by_group.get_position().y0 > active_total.get_position().y0
        assert active_total.get_position().x0 < allocation.get_position().x0
        assert exposure.get_position().x0 < selection.get_position().x0
        for ax in axes_by_title.values():
            assert ax.patches
    finally:
        plt.close('all')
