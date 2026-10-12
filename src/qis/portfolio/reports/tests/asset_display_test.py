"""Independent selection and conservation checks for factsheet display reduction."""
import numpy as np
import pandas as pd
import pytest

from qis.portfolio._asset_display import (
    compact_groups, factsheet_kwargs, price_tails, reduce_columns, reduce_rows, tail_index,
)


@pytest.mark.parametrize('count', [10, 20])
@pytest.mark.parametrize('offset', [-500, 1])
def test_signed_and_one_sided_tails(count, offset):
    scores = pd.Series(np.arange(1_000) + offset, index=[f'A{i}' for i in range(1_000)])
    original = scores.copy()
    selected = tail_index(scores, count)
    assert selected.tolist() == scores.index[:count].tolist() + scores.index[-count:].tolist()
    pd.testing.assert_series_equal(scores, original)


def test_ties_missing_and_small_universe():
    scores = pd.Series([np.nan, np.inf] + [1.0] * 25, index=range(27))
    assert tail_index(scores, 10).tolist() == list(range(2, 12)) + list(range(17, 27))
    small = pd.Series([3.0, -1.0, 2.0])
    pd.testing.assert_index_equal(tail_index(small, 10), small.index)


def test_additive_series_conserve_each_date_and_keep_true_total():
    frame = pd.DataFrame(np.arange(3_000).reshape(3, 1_000), dtype=float)
    frame['Total'] = frame.sum(axis=1)
    original = frame.copy()
    actual, title = reduce_columns(frame, 10, 'Exposure', additive=True, keep=['Total'])
    assert actual.shape[1] == 22
    pd.testing.assert_series_equal(actual['Total'], original['Total'])
    np.testing.assert_allclose(actual.drop(columns='Total').sum(axis=1), original['Total'])
    pd.testing.assert_frame_equal(actual.iloc[:, :20], original.loc[:, actual.columns[:20]])
    assert '980 in Other' in title
    pd.testing.assert_frame_equal(frame, original)


def test_nonadditive_table_does_not_sum_ratios():
    table = pd.DataFrame({'IR': np.linspace(-2, 2, 1_000), 'Vol': 0.2})
    actual, title = reduce_rows(table, table['IR'], 10, 'IR', metric='IR')
    assert len(actual) == 20
    assert '980 omitted' in title
    pd.testing.assert_frame_equal(actual, table.iloc[list(range(10)) + list(range(990, 1_000))])


def test_duplicate_display_names_cannot_expand_selected_rows():
    data = pd.Series(np.arange(1_000, dtype=float), index=['Same display name'] * 1_000)
    actual, _ = reduce_rows(data, data, 10, additive=True)
    assert len(actual) == 21
    np.testing.assert_array_equal(actual.iloc[:20], np.r_[np.arange(10), np.arange(990, 1_000)])
    assert actual.sum() == data.sum()


def test_additive_bar_caption_preserves_remainder_without_flattening_tails(monkeypatch):
    import qis
    from qis.portfolio._asset_display import plot_asset_bars
    data = pd.Series(np.linspace(-0.01, 0.05, 1_000))
    captured = {}

    def capture(df, **kwargs):
        captured.update(data=df, **kwargs)

    monkeypatch.setattr(qis, 'plot_bars', capture)
    plot_asset_bars(data, asset_tail_count=10, asset_additive=True, yvar_format='{:.4f}')
    assert len(captured['data']) == 20
    expected = data.iloc[list(range(10)) + list(range(990, 1_000))]
    pd.testing.assert_series_equal(captured['data'], expected)
    remainder = data.sum() - expected.sum()
    assert f'omitted sum {remainder:.4f}' in captured['title']


def test_price_history_ranks_observed_total_returns():
    prices = pd.DataFrame([np.ones(30), np.linspace(0.5, 2.0, 30), np.full(30, np.nan)])
    actual, title = price_tails(prices, 10, 'History')
    assert actual.columns.tolist() == list(range(10)) + list(range(20, 30))
    assert 'by total return' in title


def test_factsheet_attribution_keeps_zero_contributors_in_bottom_tail(monkeypatch):
    import qis
    portfolio = qis.PortfolioData(nav=pd.Series(
        [1.0, 1.1], index=pd.bdate_range('2024-01-01', periods=2), name='Portfolio'))
    values = pd.Series(np.r_[np.zeros(500), np.linspace(0.001, 0.01, 500)],
                       index=[f'A{i}' for i in range(1_000)])
    monkeypatch.setattr(portfolio, 'get_performance_attribution_data', lambda **kw: values)
    captured = []
    monkeypatch.setattr(qis, 'plot_bars', lambda df, **kw: captured.append(df))
    portfolio.plot_performance_attribution(asset_tail_count=10)
    assert captured[0].index.tolist() == [f'A{i}' for i in range(10)] + [
        f'A{i}' for i in range(990, 1_000)]
    np.testing.assert_array_equal(captured[0].iloc[:10], np.zeros(10))


def test_compact_groups_never_double_counts_total():
    instruments = {f'G{i}': pd.DataFrame({f'A{i}': [i, -i]}) for i in range(10)}
    instruments['Total'] = pd.concat(instruments.values(), axis=1)
    aggregates = {key: value.sum(axis=1).to_frame('Total')
                  for key, value in instruments.items()}
    reduced, assets = compact_groups(aggregates, instruments)
    pd.testing.assert_frame_equal(next(iter(reduced.values())), aggregates['Total'])
    pd.testing.assert_frame_equal(next(iter(assets.values())), instruments['Total'])


def test_brinson_display_keeps_full_totals_and_additive_remainder(monkeypatch):
    """Table/curve reduction must preserve linked effects at every observation."""
    import matplotlib.pyplot as plt
    import qis
    from qis.portfolio.reports import brinson_attribution as report

    dates = pd.bdate_range('2024-01-01', periods=5)
    effects = pd.DataFrame(np.tile(np.arange(-500, 500) / 1e6, (5, 1)), index=dates)
    effects['Total Sum'] = effects.sum(axis=1)
    totals = pd.concat([effects.sum()] * 7, axis=1)
    totals.columns = [f'Metric {i}' for i in range(7)]
    table_calls, curve_calls = [], []
    monkeypatch.setattr(report, 'plot_brinson_totals_table',
                        lambda totals_table, **kw: table_calls.append(totals_table))
    monkeypatch.setattr(qis, 'plot_time_series',
                        lambda df, **kw: curve_calls.append(df))
    figures = report.plot_brinson_attribution_table(
        totals, effects[['Total Sum']], effects, effects * 2, effects * 0,
        asset_tail_count=10)
    actual = table_calls[0]
    assert len(actual) == 22
    assert actual.index[-1] == 'Total Sum'
    pd.testing.assert_series_equal(actual.loc['Total Sum'], totals.loc['Total Sum'])
    np.testing.assert_allclose(actual.drop(index='Total Sum').sum(), totals.loc['Total Sum'])
    # Curves include active effects (allocation + selection), allocation and selection.
    for curve, multiplier in zip(curve_calls, [3, 1, 2]):
        assert curve.shape[1] <= 22
        np.testing.assert_allclose(curve.drop(columns='Total Sum', errors='ignore').sum(axis=1),
                                   effects['Total Sum'].cumsum() * multiplier)
    for figure in figures:
        if figure is not None:
            plt.close(figure)


@pytest.mark.parametrize('count', [0, -1, None, True, 15, 10.0, '10'])
def test_reject_invalid_display_budget(count):
    with pytest.raises(ValueError, match='asset_tail_count'):
        factsheet_kwargs({'asset_tail_count': count})
