"""Independent fractional-exposure and selected-period reporting contracts."""
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import qis
from qis.portfolio.tests.brinson_wrapper_test import make_portfolios


def _fixture():
    """Return stored, drifting portfolios and a fully partitioned mixed instrument."""
    strategy, benchmark = make_portfolios()
    multi = qis.MultiPortfolioData([strategy, benchmark], benchmark_prices=benchmark.nav.to_frame())
    matrix = pd.DataFrame({'Growth': [.52, 0., 0.], 'Credit': [.16, 1., 1.],
                           'Hedge': [.22, 0., 0.], 'ILS': [.10, 0., 0.]},
                          index=['SEQ_US', 'SBD_TSY', 'SBD_IG'])
    return multi, matrix


def test_fractional_brinson_matches_independently_split_instruments():
    """Virtual instruments provide an independent reference without matrix grouping."""
    multi, matrix = _fixture()
    period = qis.TimePeriod('2021-01-29', '2021-06-30')
    frames = []
    labels = {}
    for portfolio in multi.portfolio_datas:
        pnl, weights = portfolio.get_brinson_inputs(freq=None)
        keep = (pnl.index > period.start) & (pnl.index <= period.end)
        for data in (pnl.loc[keep], weights.loc[keep]):
            split = {}
            for asset in data:
                for group, share in matrix.loc[asset].items():
                    if share:
                        key = f'{asset}/{group}'
                        split[key] = data[asset] * share
                        labels[key] = group
            frames.append(pd.DataFrame(split))
    sp, sw, bp, bw = frames
    reference = qis.compute_brinson_attribution_table(
        bp, sp, sw, bw, pd.Series(labels), group_order=list(matrix))
    actual = multi.compute_brinson_attribution(time_period=period, lookthrough_matrix=matrix)
    for got, expected in zip(actual, reference):
        pd.testing.assert_frame_equal(got, expected, check_exact=False, atol=1e-12, rtol=1e-12)
    original = multi.compute_brinson_attribution(time_period=period)
    assert actual[0].loc['Total Sum', 'Total\nActive'] == pytest.approx(
        original[0].loc['Total Sum', 'Total\nActive'], abs=1e-12)
    assert actual[0].loc['Growth', 'Strategy\nWeight Ave'] == pytest.approx(
        matrix.at['SEQ_US', 'Growth'] * sw.filter(like='SEQ_US').sum(axis=1).mean())


@pytest.mark.parametrize('defect', ['missing', 'overlap', 'nan', 'negative', 'duplicate'])
def test_lookthrough_rejects_non_partition_mappings(defect):
    """Missing or overlapping policy memberships must never be silently normalised."""
    multi, matrix = _fixture()
    if defect == 'missing':
        matrix = matrix.drop(index='SEQ_US')
    elif defect == 'overlap':
        matrix.at['SEQ_US', 'Growth'] = 1.
    elif defect == 'nan':
        matrix.at['SEQ_US', 'Growth'] = np.nan
    elif defect == 'negative':
        matrix.loc['SEQ_US', ['Growth', 'Credit']] = [-.1, .78]
    else:
        matrix = pd.concat([matrix, matrix.iloc[:1]])
    with pytest.raises(ValueError):
        multi.compute_brinson_attribution(
            lookthrough_matrix=matrix,
            group_data=(pd.Series({"SBD_TSY": "Bonds", "SBD_IG": "Bonds"})
                        if defect == "missing" else None))


def test_lookthrough_exposures_preserve_drift_and_totals():
    """Exposure accounting is a split of realised weights, never target rebalancing."""
    multi, matrix = _fixture()
    weights = multi.portfolio_datas[0].weights
    result = qis.compute_lookthrough_exposures(weights, matrix)
    np.testing.assert_allclose(result.sum(axis=1), weights.sum(axis=1), atol=1e-14)
    np.testing.assert_allclose(result.Growth, .52 * weights.SEQ_US, atol=1e-14)
    np.testing.assert_allclose(result.Credit, .16 * weights.SEQ_US + weights.SBD_TSY)


def test_exposure_factsheet_uses_native_prior_weights_and_period_end():
    """Displayed averages match attribution, and current bars use realised end holdings."""
    from qis.portfolio.reports.lookthrough_exposures import _exposure_tables
    multi, matrix = _fixture()
    period = qis.TimePeriod('2021-01-29', '2021-06-30')
    average, current, actual_period = _exposure_tables(multi, matrix, period)
    attribution = multi.compute_brinson_attribution(
        time_period=period, lookthrough_matrix=matrix)[0]
    np.testing.assert_allclose(average.Strategy, attribution['Strategy\nWeight Ave'].iloc[:-1])
    expected = qis.compute_lookthrough_exposures(multi.portfolio_datas[0].weights, matrix)
    np.testing.assert_allclose(current.Strategy, expected.loc[actual_period.end])
    figures = qis.generate_lookthrough_exposures_factsheet(multi, matrix, time_period=period)
    try:
        assert len(figures) == 1
        figures[0].canvas.draw()
        assert len(figures[0].axes) == 2
        assert '29Jan2021' in figures[0]._suptitle.get_text()
    finally:
        for fig in figures:
            plt.close(fig)


def test_brinson_report_period_is_independent_of_performance_window(monkeypatch):
    """Every Brinson panel receives its selected baseline, and other pages keep theirs."""
    multi, matrix = _fixture()
    full = qis.TimePeriod('2020-12-31', '2021-06-30')
    recent = qis.TimePeriod('2021-03-31', '2021-06-30')
    captured = []
    original = multi.compute_brinson_attribution
    def capture(**kwargs):
        captured.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(multi, 'compute_brinson_attribution', capture)
    options = qis.fetch_default_report_kwargs(
        time_period=full, reporting_frequency=qis.ReportingFrequency.MONTHLY, add_rates_data=False)
    figures = qis.generate_strategy_benchmark_factsheet_plt(
        multi, time_period=full, brinson_time_period=recent, lookthrough_matrix=matrix, **options)
    try:
        assert captured[0]['time_period'].start == recent.start
        assert captured[0]['time_period'].end == recent.end
        pd.testing.assert_frame_equal(captured[0]['lookthrough_matrix'], matrix)
        assert '31Mar2021' in figures[1]._suptitle.get_text()
        assert '30Jun2021' in figures[1]._suptitle.get_text()
    finally:
        for fig in figures:
            plt.close(fig)



def test_large_partition_keeps_complete_table_and_readable_names():
    """A larger exposure grid gets a tall table instead of overflowing a small panel."""
    multi, _ = _fixture()
    names = [f'Long sleeve name {i}' for i in range(10)]
    matrix = pd.DataFrame(.1, index=['SEQ_US', 'SBD_TSY', 'SBD_IG'], columns=names)
    period = qis.TimePeriod('2020-12-31', '2021-06-30')
    options = qis.fetch_default_report_kwargs(
        time_period=period, reporting_frequency=qis.ReportingFrequency.MONTHLY,
        add_rates_data=False)
    figures = qis.generate_strategy_benchmark_factsheet_plt(
        multi, time_period=period, lookthrough_matrix=matrix, **options)
    try:
        page = figures[1]
        page.canvas.draw()
        assert len(page.axes) == 5
        table = next(iter(page.axes[0].tables))
        assert len([key for key in table.get_celld() if key[1] == 0]) == 12
        renderer = page.canvas.get_renderer()
        heading = page._suptitle.get_window_extent(renderer)
        assert page.bbox.contains(heading.x0, heading.y0)
        assert page.bbox.contains(heading.x1, heading.y1)
        for cell in table.get_celld().values():
            bounds = cell.get_text().get_window_extent(renderer)
            assert page.bbox.contains(bounds.x0, bounds.y0)
            assert page.bbox.contains(bounds.x1, bounds.y1)
    finally:
        for fig in figures:
            plt.close(fig)
