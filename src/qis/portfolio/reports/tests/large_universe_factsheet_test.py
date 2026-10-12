"""Exercise real factsheet generation and rendering with 1,000 underlying assets.

The frozen synthetic generator supplies distinct paths, expanded through its existing
instrument-specification API. These are 1,000 holdings in each of three strategies, not
1,000 strategy NAVs. No data downloads or optional reporting dependencies are needed.
"""

from dataclasses import replace
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets.synthetic import SYNTHETIC_UNIVERSE, generate_synthetic_prices
from qis.portfolio.reports.multi_strategy_factsheet import generate_multi_portfolio_factsheet
from qis.portfolio.reports.strategy_benchmark_factsheet import (
    generate_strategy_benchmark_factsheet_plt,
)
from qis.portfolio.reports.strategy_factsheet import generate_strategy_factsheet
from qis.portfolio.reports import brinson_attribution
from qis.portfolio.signal_data import StrategySignalData


class FactsheetLayoutError(AssertionError):
    """A rendered factsheet exceeds the generous readability limits of this stress test."""


def _check_layout(condition, message):
    if not condition:
        raise FactsheetLayoutError(message)


@pytest.fixture(scope='module')
def large_portfolios():
    """Three monthly-rebalanced portfolios over three years of distinct daily prices."""
    instruments = tuple(
        replace(SYNTHETIC_UNIVERSE[index % 10], ticker=f'S{index:04d}')
        for index in range(1_000)
    )
    prices = generate_synthetic_prices(
        instruments=instruments, start='2023-01-02', end='2025-12-31', apply_quirks=False,
    )
    assert prices.shape[1] == 1_000
    assert prices.columns.is_unique
    groups = pd.Series([item.group for item in instruments], index=prices.columns)
    benchmarks = prices[['S0000']].rename(columns={'S0000': 'Market'})
    rebalance_dates = prices.resample('BME').last().index
    portfolios = []
    for index, name in enumerate(('Strategy', 'Benchmark', 'Alternative')):
        # Predetermined cross-sectional tilts: no estimation or future prices in weights.
        target = np.linspace(1.0, 1.0 + index, prices.shape[1])
        target /= target.sum()
        weights = pd.DataFrame(np.tile(target, (len(rebalance_dates), 1)),
                               index=rebalance_dates, columns=prices.columns)
        portfolio = qis.backtest_model_portfolio(
            prices=prices, weights=weights,
            rebalancing_costs=0.0005, ticker=name,
        )
        portfolio.set_group_data(groups)
        assert len(portfolio.group_order) == 4
        assert portfolio.weights.iloc[-1].ne(0.0).sum() == 1_000
        assert np.isfinite(portfolio.nav).all()
        portfolios.append(portfolio)
    return portfolios, benchmarks, groups


@pytest.fixture(autouse=True)
def close_report_figures():
    """Also close partially constructed pages when a renderer raises."""
    existing = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - existing:
        plt.close(number)


def _make_report(large_portfolios, kind, *, grouped=True, facade=False, **options):
    portfolios, benchmarks, groups = large_portfolios
    # Copies prevent report mutations and alternative grouping from leaking between tests.
    portfolios = [replace(portfolio) for portfolio in portfolios]
    if not grouped:
        for portfolio in portfolios:
            portfolio.set_group_data(pd.Series(portfolio.prices.columns,
                                               index=portfolio.prices.columns))
    if kind == 'strategy':
        if options.get('add_current_signal_report') or options.get('add_weight_change_report'):
            returns = qis.to_returns(portfolios[0].prices, is_log_returns=True).fillna(0.0)
            signal = np.tanh(20.0 * returns)
            portfolios[0].strategy_signal_data = StrategySignalData(
                log_returns=returns, momentum=signal, signal=signal,
                instrument_vols=0.2 + returns.abs(),
                instrument_target_vols=0.1 + returns.abs(),
                instrument_target_signal_vol_weights=0.1 * signal,
                instrument_portfolio_leverages=1.0 + returns,
                weights=portfolios[0].weights,
            )
        if facade:
            return qis.factsheet(portfolios[0], kind=kind, benchmark_prices=benchmarks,
                                 add_rates_data=False, **options)
        return generate_strategy_factsheet(portfolios[0], benchmarks, **options)
    count = 2 if kind == 'strategy_benchmark' else 3
    multi = qis.MultiPortfolioData(portfolios[:count], benchmark_prices=benchmarks)
    if facade:
        return qis.factsheet(multi, kind=kind, add_rates_data=False,
                             group_data=groups if grouped else None, **options)
    if kind == 'strategy_benchmark':
        return generate_strategy_benchmark_factsheet_plt(multi, **options)
    return generate_multi_portfolio_factsheet(
        multi, group_data=groups if grouped else None, **options,
    )


def _assert_readable_pages(figures, tail_count=10):
    """Force deferred layout and detect overflowing legends and unbounded asset labels."""
    assert figures
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        for figure in figures:
            _check_layout(max(figure.get_size_inches()) <= 24,
                          f'Oversized page: {figure.get_size_inches()} inches')
            # Reject grossly oversized panels before spending minutes laying out their text.
            for axis in figure.axes:
                # A4 panels cannot name hundreds of holdings, even if draw() succeeds.
                labels = axis.get_xticklabels() + axis.get_yticklabels()
                asset_labels = [label for label in labels
                                if label.get_text().startswith('S')
                                and label.get_text()[1:].isdigit()]
                _check_layout(len(asset_labels) <= 2 * tail_count,
                              f'Unbounded asset labels ({len(asset_labels)}): {axis.get_title()}')
                for table in axis.tables:
                    rows = {row for row, _ in table.get_celld()}
                    _check_layout(len(rows) <= 2 * tail_count + 4,
                                  f'Unbounded table ({len(rows)} rows): {axis.get_title()}')
                legend = axis.get_legend()
                if legend is not None:
                    _check_layout(len(legend.get_texts()) <= 2 * tail_count + 2,
                                  f'Unbounded legend: {axis.get_title()}')
            # Some report appendices deliberately detach their pyplot canvas on close.
            # Attach Agg so draw() really renders them rather than calling the base no-op.
            if not isinstance(figure.canvas, FigureCanvasAgg):
                FigureCanvasAgg(figure)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            for axis in figure.axes:
                assert axis.get_position().width > 0
                assert axis.get_position().height > 0
                legend = axis.get_legend()
                if legend is not None:
                    _check_layout(legend.get_window_extent(renderer).height <= axis.bbox.height,
                                  f'Legend overflows panel: {axis.get_title()}')
    _check_layout(not any('constrained_layout not applied' in str(item.message)
                          for item in records), 'Constrained layout collapsed')


def _guard_dense_panels(monkeypatch):
    """Fail before allocating unbounded stacks, bars, line legends, or tables.

    Computations and all manageable plots stay real. This guard only stops an already
    unbounded display at its plotting boundary, before Matplotlib allocates its artists.
    """
    original_stack = qis.plot_stack
    original_table = brinson_attribution.plot_brinson_totals_table
    original_perf_table = qis.plot_ra_perf_table
    original_df_table = qis.plot_df_table
    original_time_series = qis.plot_time_series
    original_bars = qis.plot_bars
    original_regime_boxplot = qis.plot_regime_boxplot

    def bounded_stack(df, *args, **kwargs):
        _check_layout(df.shape[1] <= 22, f'Unbounded stack: {df.shape[1]} series')
        return original_stack(df, *args, **kwargs)

    def bounded_table(totals_table, *args, **kwargs):
        _check_layout(len(totals_table) <= 22,
                      f'Unbounded Brinson table: {len(totals_table)} rows')
        return original_table(totals_table, *args, **kwargs)

    def bounded_perf_table(prices, *args, **kwargs):
        _check_layout(prices.shape[1] <= 22,
                      f'Unbounded performance table: {prices.shape[1]} assets')
        return original_perf_table(prices, *args, **kwargs)

    def bounded_df_table(df, *args, **kwargs):
        _check_layout(len(df) <= 22, f'Unbounded report table: {len(df)} rows')
        return original_df_table(df, *args, **kwargs)

    def bounded_time_series(df, *args, **kwargs):
        count = df.shape[1] if isinstance(df, pd.DataFrame) else 1
        _check_layout(count <= 22, f'Unbounded time-series panel: {count} series')
        return original_time_series(df, *args, **kwargs)

    def bounded_bars(df, *args, **kwargs):
        _check_layout(len(df) <= 22, f'Unbounded bar panel: {len(df)} categories')
        return original_bars(df, *args, **kwargs)

    def bounded_regime_boxplot(prices, *args, **kwargs):
        _check_layout(prices.shape[1] <= 22,
                      f'Unbounded regime boxplot: {prices.shape[1]} series')
        return original_regime_boxplot(*args, prices=prices, **kwargs)

    monkeypatch.setattr(qis, 'plot_stack', bounded_stack)
    monkeypatch.setattr(brinson_attribution, 'plot_brinson_totals_table', bounded_table)
    monkeypatch.setattr(qis, 'plot_ra_perf_table', bounded_perf_table)
    monkeypatch.setattr(qis, 'plot_df_table', bounded_df_table)
    monkeypatch.setattr(qis, 'plot_time_series', bounded_time_series)
    monkeypatch.setattr(qis, 'plot_bars', bounded_bars)
    monkeypatch.setattr(qis, 'plot_regime_boxplot', bounded_regime_boxplot)


@pytest.mark.parametrize(('grouped', 'kind'), [
    pytest.param(True, 'strategy', id='four-groups-strategy'),
    pytest.param(True, 'strategy_benchmark', id='four-groups-strategy_benchmark'),
    pytest.param(True, 'multi_strategy', id='four-groups-multi_strategy'),
    pytest.param(False, 'strategy', id='no-groups-strategy'),
    pytest.param(False, 'strategy_benchmark', id='no-groups-strategy_benchmark'),
    pytest.param(False, 'multi_strategy', id='no-groups-multi_strategy'),
])
def test_large_universe_summary(large_portfolios, kind, grouped, tmp_path, monkeypatch):
    """All report archetypes must actually draw and export their default pages."""
    if not grouped:
        _guard_dense_panels(monkeypatch)
    figures = _make_report(large_portfolios, kind, grouped=grouped)
    assert len(figures) == (2 if kind == 'strategy_benchmark' else 1)
    _assert_readable_pages(figures)
    qis.save_figs_to_pdf(figures, kind, local_path=str(tmp_path), add_current_date=False)
    assert (tmp_path / f'{kind}.pdf').read_bytes().startswith(b'%PDF-')
    figures[0].savefig(tmp_path / 'summary.png', dpi=140)


def test_large_universe_top_bottom_contributors(large_portfolios):
    """The contributor chart must select exactly the actual ten worst and ten best assets."""
    portfolio = large_portfolios[0][0]
    _, axis = plt.subplots(figsize=(8.5, 3))
    portfolio.plot_contributors(num_assets=10, ax=axis)
    axis.figure.canvas.draw()
    navs = portfolio.get_instruments_navs()
    ranked = (navs.iloc[-1] / navs.iloc[0] - 1.0).sort_values()
    expected = pd.concat([ranked.iloc[:10], ranked.iloc[-10:]])
    assert len(axis.patches) == 20
    assert [label.get_text() for label in axis.get_xticklabels()] == expected.index.tolist()
    np.testing.assert_allclose([bar.get_height() for bar in axis.patches], expected.to_numpy())


def test_large_universe_default_group_order_is_unique(large_portfolios):
    """The setter default must agree with construction using the same four asset groups.

    Repeated group_order entries duplicate columns in get_aligned_weights, making
    strategy_weights[group].rename(strategy_name) operate on a DataFrame and raise
    TypeError in the exposure-comparison appendix. Group-page loops also repeat pages.
    """
    portfolios, _, groups = large_portfolios
    portfolio = replace(portfolios[0])
    portfolio.set_group_data(groups)
    assert portfolio.group_order == groups.unique().tolist()


@pytest.mark.parametrize(('kind', 'options', 'page_count'), [
    pytest.param('strategy', {'add_current_position_var_risk_sheet': True}, 2, id='risk'),
    pytest.param('strategy', {'add_grouped_weights_sheet': True}, 2, id='weights'),
    pytest.param('strategy', {
        'add_performance_risk_attribution_sheet': True,
        'ytd_attribution_time_period': qis.TimePeriod('31Dec2024', '31Dec2025'),
    }, 2, id='attribution'),
    pytest.param('strategy', {'add_grouped_exposures': True}, 2, id='group-exposures'),
    pytest.param('strategy', {'add_grouped_cum_pnl': True}, 2, id='group-pnl'),
    pytest.param('strategy', {'add_instrument_history_report': True}, 2, id='history'),
    pytest.param('strategy', {'add_current_signal_report': True}, 2, id='signals'),
    pytest.param('strategy', {'add_weight_change_report': True}, 2, id='weight-changes'),
    pytest.param('strategy_benchmark', {'add_exposures_comp': True}, 6,
                 id='benchmark-exposures'),
    pytest.param('strategy_benchmark', {'add_exposures_pnl_attribution': True}, 3,
                 id='benchmark-pnl'),
    pytest.param('strategy_benchmark', {'add_joint_instrument_history_report': True}, 3,
                 id='benchmark-history'),
    pytest.param('strategy_benchmark', {'add_tracking_error_table': True}, 3,
                 id='tracking-error'),
    pytest.param('strategy_benchmark', {'add_strategy_factsheet': True}, 4,
                 id='benchmark-strategy-pages'),
    pytest.param('multi_strategy', {'add_group_exposures_and_pnl': True}, 5,
                 id='multi-group-pages'),
    pytest.param('multi_strategy', {'add_strategy_factsheets': True}, 7,
                 id='multi-strategy-pages'),
])
@pytest.mark.parametrize('grouped', [True, False], ids=['four-groups', 'no-groups'])
def test_large_universe_appendices(large_portfolios, kind, options, page_count, grouped,
                                   monkeypatch, tmp_path):
    """Optional pages must remain readable as the holdings universe grows."""
    _guard_dense_panels(monkeypatch)
    options = dict(options)
    if not grouped and kind == 'strategy_benchmark':
        # The no-group Brinson page is exercised by the summary test. Isolate appendices.
        options['add_brinson_attribution'] = False
        page_count -= 1
        if options.get('add_exposures_comp'):
            page_count = 21  # Summary plus twenty selected asset comparisons.
        elif options.get('add_exposures_pnl_attribution'):
            page_count = 6  # Summary plus five pages of four selected groups.
    elif not grouped and kind == 'multi_strategy':
        page_count = 21 if options.get('add_group_exposures_and_pnl') else 4
    figures = _make_report(large_portfolios, kind, grouped=grouped, **options)
    assert len(figures) == page_count
    _assert_readable_pages(figures)
    figures[-1].savefig(tmp_path / 'appendix.png', dpi=140)


@pytest.mark.parametrize('kind', ['strategy', 'strategy_benchmark', 'multi_strategy'])
def test_large_universe_facade(large_portfolios, kind):
    """The public facade must also draw with its frequency-calibrated presets."""
    _assert_readable_pages(_make_report(large_portfolios, kind, facade=True))


def test_large_universe_benchmark_without_brinson(large_portfolios):
    """The aggregate comparison page can work without instrument group metadata."""
    figures = _make_report(large_portfolios, 'strategy_benchmark', grouped=False,
                           add_brinson_attribution=False)
    assert len(figures) == 1
    _assert_readable_pages(figures)


@pytest.mark.parametrize('kind', ['strategy', 'strategy_benchmark', 'multi_strategy'])
def test_large_universe_twenty_per_tail(large_portfolios, kind, tmp_path):
    """The optional larger tail budget must render and propagate through the facade."""
    options = {'asset_tail_count': 20}
    if kind == 'strategy_benchmark':
        options.update(add_brinson_attribution=False, add_tracking_error_table=True)
    elif kind == 'multi_strategy':
        options['add_strategy_factsheets'] = True
    figures = _make_report(large_portfolios, kind, facade=True, **options)
    _assert_readable_pages(figures, tail_count=20)
    figures[-1].savefig(tmp_path / 'twenty-per-tail.png', dpi=140)
