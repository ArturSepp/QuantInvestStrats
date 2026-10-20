"""Benchmark intervals are optional, labelled and scaled by actual reporting equity."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from qis.portfolio.stress.analytics import run_portfolio_stress_test
from qis.portfolio.stress.reporting import StressReportConfig, generate_portfolio_stress_report
from qis.portfolio.stress._figures import _risk_page, report_pages
from qis.portfolio.stress._inference import portfolio_beta_interval_table
from qis.portfolio.stress.tests.accounting_test import account_portfolio
from qis.portfolio.stress.tests.analytics_test import request


def test_production_dot_is_bar_endpoint_even_outside_bootstrap_range(market):
    result = run_portfolio_stress_test(account_portfolio(market), request())
    actual = result.factor_betas
    supplied = pd.DataFrame({'estimate': actual, 'lower': actual+.1, 'upper': actual+.2})
    config = StressReportConfig(factor_beta_intervals=supplied,
        beta_interval_target='production', beta_interval_kind='stability',
        beta_interval_label='Full FCGL bootstrap')
    table = portfolio_beta_interval_table(result, config)
    assert 'benchmark_beta' not in table
    np.testing.assert_allclose(table.dollar_lower, (actual+.1)*90)
    figure = _risk_page(result, config)
    try:
        for axis, scale in zip(figure.axes[:2], [1., 90/1e6]):
            marker = next(line for line in axis.lines if line.get_marker() == 'o')
            np.testing.assert_allclose(marker.get_xdata(), actual*scale)
            np.testing.assert_allclose(marker.get_ydata(),
                [bar.get_y()+bar.get_height()/2 for bar in axis.containers[0]])
            assert all(text.get_position()[1] == 7 for text in axis.texts)
        assert 'stability range' in figure.axes[0].get_legend().get_texts()[1].get_text()
    finally:
        plt.close(figure)
    wrong = replace(config, factor_beta_intervals=supplied.assign(estimate=actual+.01))
    with pytest.raises(ValueError, match='must match'):
        portfolio_beta_interval_table(result, wrong)


def test_production_intervals_validate_order_but_not_containment():
    table = intervals().assign(lower=[.5, -.3], upper=[.7, .05])
    StressReportConfig(factor_beta_intervals=table, beta_interval_target='production')
    with pytest.raises(ValueError, match='lower bound'):
        StressReportConfig(factor_beta_intervals=table.assign(upper=[.4, .05]),
                           beta_interval_target='production')


def intervals():
    """Deliberately permuted factors with a benchmark unlike the production loadings."""
    return pd.DataFrame({"estimate": [.2, -.1], "lower": [.1, -.3], "upper": [.4, .05]},
                        index=["Credit", "Equity"])


def test_table_only_keeps_chart_scale_and_full_uncertainty(market):
    result = run_portfolio_stress_test(account_portfolio(market), request())
    actual = result.factor_betas
    bounds = pd.DataFrame({'estimate': actual, 'lower': -100., 'upper': 100.})
    bounds.loc['Equity', ['lower', 'upper']] = [-np.inf, np.inf]
    config = StressReportConfig(factor_beta_intervals=bounds,
        beta_interval_target='production', beta_interval_display='table')
    plain = _risk_page(result, StressReportConfig())
    diagnostic = _risk_page(result, config)
    try:
        for expected, got in zip(plain.axes[:2], diagnostic.axes[:2]):
            np.testing.assert_array_equal(got.get_xlim(), expected.get_xlim())
            np.testing.assert_array_equal([b.get_width() for b in got.containers[0]],
                                          [b.get_width() for b in expected.containers[0]])
            assert [t.get_text() for t in got.get_legend().get_texts()] == [
                t.get_text() for t in expected.get_legend().get_texts()]
            assert len(got.collections) == len(expected.collections)
        assert any('following diagnostic table' in t.get_text() for t in diagnostic.texts)
    finally:
        plt.close(plain)
        plt.close(diagnostic)
    table = portfolio_beta_interval_table(result, config)
    pd.testing.assert_frame_equal(table[['lower', 'upper']], bounds[['lower', 'upper']])
    assert np.isposinf(table.loc['Equity', 'dollar_upper'])
    assert table.loc['Credit', 'dollar_upper'] == 9000.
    with pytest.raises(ValueError, match='beta_interval_display'):
        replace(config, beta_interval_display='clip')


def test_bounds_use_net_equity_and_do_not_move_production_bars(market):
    result = run_portfolio_stress_test(account_portfolio(market), request())
    config = StressReportConfig(factor_beta_intervals=intervals())
    table = portfolio_beta_interval_table(result, config)
    # Independent account identity: assets 110 less borrowing 20 = equity 90.
    np.testing.assert_allclose(table.loc['Equity', ['dollar_benchmark_beta',
        'dollar_lower', 'dollar_upper']].astype(float), [-9., -27., 4.5])
    assert table.loc['Equity', 'production_beta'] == pytest.approx(100/90)
    assert table.loc['FX', 'status'] == 'Not supplied'
    figure = _risk_page(result, config)
    try:
        for axis, scale in zip(figure.axes[:2], [1., 90/1e6]):
            np.testing.assert_allclose([bar.get_width() for bar in axis.containers[0]],
                                       result.factor_betas*scale)
            error = axis.containers[1]
            np.testing.assert_allclose(error.lines[0].get_xdata(), [-.1*scale, .2*scale])
            segments = error.lines[2][0].get_segments()
            np.testing.assert_allclose([segment[:, 0] for segment in segments],
                                       np.array([[-.3, .05], [.1, .4]])*scale)
    finally:
        plt.close(figure)


@pytest.mark.parametrize('columns', [dict(lower=[.3, -.3]), dict(upper=[-np.inf, .05]),
                                    dict(lower=[np.inf, -.3]), dict(estimate=[np.inf, -.1]),
                                    dict(estimate=[np.nan, -.1])])
def test_invalid_bounds_fail_before_rendering(columns):
    table = intervals().assign(**columns)
    with pytest.raises(ValueError):
        StressReportConfig(factor_beta_intervals=table)


def test_unavailable_rows_and_copy_semantics():
    table = intervals()
    table.loc['Rates'] = np.nan
    config = StressReportConfig(factor_beta_intervals=table)
    table.loc['Equity', 'estimate'] = 99.
    assert config.factor_beta_intervals.loc['Equity', 'estimate'] == -.1
    assert config.factor_beta_intervals.loc['Rates'].isna().all()
    with pytest.raises(ValueError, match='unique'):
        StressReportConfig(factor_beta_intervals=pd.concat([table, table]))


def test_open_confidence_bounds_remain_visible_without_infinite_axes(market):
    result = run_portfolio_stress_test(account_portfolio(market), request())
    actual = result.factor_betas
    supplied = pd.DataFrame({'estimate': actual, 'lower': -np.inf, 'upper': np.inf})
    supplied.loc['Credit', 'lower'] = 0.
    config = StressReportConfig(factor_beta_intervals=supplied,
        beta_interval_target='production', beta_interval_label='Calibrated test inversion')
    table = portfolio_beta_interval_table(result, config)
    assert table.status.eq('Unbounded').all()
    assert table.loc['Credit', 'dollar_lower'] == 0.
    assert np.isposinf(table.dollar_upper).all()
    assert np.isneginf(table.loc['Equity', 'lower'])
    figure = _risk_page(result, config)
    try:
        for axis, scale in zip(figure.axes[:2], [1., 90/1e6]):
            assert np.isfinite(axis.get_xlim()).all()
            marker = next(line for line in axis.lines if line.get_marker() == 'o')
            np.testing.assert_allclose(marker.get_xdata(), actual*scale)
            assert 'Open bounds' in axis.get_legend().get_texts()[-1].get_text()
        from qis.portfolio.stress._inference import beta_interval_pages
        for _, page in beta_interval_pages(result, config):
            try:
                cell_text = [cell.get_text().get_text() for axis in page.axes
                             for table_artist in axis.tables
                             for cell in table_artist.get_celld().values()]
                assert '+inf' in cell_text and '-inf' in cell_text
            finally:
                plt.close(page)
    finally:
        plt.close(figure)


def test_optional_pages_and_export_without_repricing(market, tmp_path):
    result = run_portfolio_stress_test(account_portfolio(market), request())
    before = result.valuations['requested'].pnl.copy()
    base = StressReportConfig(write_workbook=False)
    titles = []
    for title, figure in report_pages(result, base):
        titles.append(title)
        plt.close(figure)
    assert not any('uncertainty' in title for title in titles)
    config = replace(base, factor_beta_intervals=intervals(), beta_interval_display='table')
    artifacts = generate_portfolio_stress_report(result, tmp_path/'with-ci', config)
    manifest = json.loads(artifacts.manifest_path.read_text())
    assert manifest['page_count'] == len(titles)+1
    insertion = titles.index('Portfolio Factor model exposures and risk')+1
    assert manifest['page_titles'][insertion] == 'Portfolio factor exposure uncertainty (95% CI)'
    assert manifest['beta_inference']['confidence'] == .95
    assert manifest['beta_inference']['display'] == 'table'
    emitted = pd.read_csv(artifacts.table_paths['Portfolio beta confidence intervals'], index_col=0)
    assert emitted.loc['Equity', 'dollar_lower'] == pytest.approx(-27.)
    pd.testing.assert_frame_equal(result.valuations['requested'].pnl, before)
    with pytest.raises(ValueError, match='unknown fitted factor'):
        generate_portfolio_stress_report(result, tmp_path/'invalid',
            replace(base, factor_beta_intervals=intervals().rename(index={'Equity': 'Unknown'})))
    assert not (tmp_path/'invalid').exists()
