"""Brinson inputs retain the opening report month and realised trading costs."""
import numpy as np
import pandas as pd
import pytest
import qis


def _cost_only_portfolios(
        dtype: str, missing: object,
) -> tuple[qis.PortfolioData, qis.PortfolioData]:
    """Return a cost-only instrument beside a finite-contribution control."""
    dates = pd.date_range('2026-01-31', periods=3, freq='ME')
    columns = pd.Index(['Cost only', 'Live'])
    groups = pd.Series(['Cost only', 'Live'], index=columns)
    common = dict(
        prices=pd.DataFrame(100.0, index=dates, columns=columns, dtype=dtype),
        weights=pd.DataFrame([[0.0, 1.0]] * 3, index=dates, columns=columns, dtype=dtype),
        units=pd.DataFrame([[0.0, 1.0]] * 3, index=dates, columns=columns, dtype=dtype),
        group_data=groups,
        group_order=groups.tolist(),
    )
    strategy = qis.PortfolioData(
        nav=pd.Series([100.0, 101.0, 101.0], index=dates, name='Strategy', dtype=dtype),
        instrument_pnl=pd.DataFrame(
            [[0.0, 0.0], [missing, 0.02], [0.0, 0.0]],
            index=dates,
            columns=columns,
            dtype=dtype,
        ),
        realized_costs=pd.DataFrame(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            index=dates,
            columns=columns,
            dtype=dtype,
        ),
        ticker='Strategy',
        **common,
    )
    benchmark = qis.PortfolioData(
        nav=pd.Series(100.0, index=dates, name='Benchmark', dtype=dtype),
        instrument_pnl=pd.DataFrame(0.0, index=dates, columns=columns, dtype=dtype),
        realized_costs=pd.DataFrame(0.0, index=dates, columns=columns, dtype=dtype),
        ticker='Benchmark',
        **common,
    )
    return strategy, benchmark


@pytest.mark.parametrize('cost', [0.0, .002])
def test_brinson_inputs_reconcile_to_monthly_nav_returns(cost) -> None:
    """Instrument contributions equal fee-free NAV returns, including rebalance costs."""
    dates = pd.date_range('2020-12-31', periods=8, freq='ME')
    prices = pd.DataFrame({
        'A': [100, 105, 103, 110, 114, 111, 120, 118],
        'B': [100, 101, 100, 102, 103, 104, 105, 106]}, index=dates, dtype=float)
    weights = pd.DataFrame({
        'A': [.6, .8, .3, .7, .5, .9, .4, .6],
        'B': [.4, .2, .7, .3, .5, .1, .6, .4]}, index=dates)
    portfolio = qis.backtest_model_portfolio(
        prices, weights, management_fee=0.0, rebalancing_costs=cost)
    period = qis.TimePeriod('2021-01-01', '2021-07-31')
    pnl, applied_weights = portfolio.get_brinson_inputs(period, freq='ME', is_net=True)
    expected = portfolio.nav.pct_change().loc['2021-01-31':]
    assert pnl.index[0] == pd.Timestamp('2021-01-31')
    assert np.isfinite(pnl.to_numpy()).all()
    np.testing.assert_allclose(pnl.sum(axis=1), expected, atol=1e-14)
    np.testing.assert_allclose(applied_weights.iloc[0], portfolio.weights.iloc[0], atol=1e-14)
    gross, _ = portfolio.get_brinson_inputs(period, freq='ME', is_net=False)
    costs = portfolio.realized_costs.div(portfolio.nav.shift(1), axis=0).loc[pnl.index]
    np.testing.assert_allclose(gross - pnl, costs, atol=1e-14)


@pytest.mark.parametrize(
    ('dtype', 'missing'), [('float64', np.nan), ('Float64', pd.NA)],
    ids=['float64', 'nullable-float64'],
)
def test_brinson_inputs_retain_cost_when_instrument_pnl_is_missing(
        dtype: str, missing: object,
) -> None:
    """A missing inactive contribution cannot erase its authoritative realized cost."""
    strategy, benchmark = _cost_only_portfolios(dtype, missing)
    strategy.weights = strategy.weights.copy(deep=True)
    strategy.units = strategy.units.copy(deep=True)
    # Entry at the report date does not change the exposure during the preceding period.
    strategy.weights.loc[strategy.weights.index[1], 'Cost only'] = 1.0
    strategy.units.loc[strategy.units.index[1], 'Cost only'] = 1.0
    original_pnl = strategy.instrument_pnl.copy(deep=True)
    original_costs = strategy.realized_costs.copy(deep=True)

    direct = strategy.get_instruments_pnl(is_net=True)
    gross, _ = strategy.get_brinson_inputs(freq=None, is_net=False)
    net, _ = strategy.get_brinson_inputs(freq=None, is_net=True)
    result = qis.MultiPortfolioData([strategy, benchmark]).compute_brinson_attribution(
        freq=None, is_net=True,
    )

    np.testing.assert_allclose(direct.iloc[1].astype(float), [-0.01, 0.02])
    assert pd.isna(gross.loc[gross.index[0], 'Cost only'])
    np.testing.assert_allclose(net.iloc[0].astype(float), [-0.01, 0.02])
    assert all(str(column_dtype) == dtype for column_dtype in net.dtypes)
    assert result[0].loc['Total Sum', 'Total\nActive'] == pytest.approx(0.01)
    pd.testing.assert_frame_equal(strategy.instrument_pnl, original_pnl)
    pd.testing.assert_frame_equal(strategy.realized_costs, original_costs)


@pytest.mark.parametrize(
    ('dtype', 'missing'), [('float64', np.nan), ('Float64', pd.NA)],
    ids=['float64', 'nullable-float64'],
)
@pytest.mark.parametrize('cost', [0.0, 1.0], ids=['zero-cost', 'finite-cost'])
@pytest.mark.parametrize(
    'exposure', [0.0, 1.0, -1.0, None], ids=['inactive', 'long', 'short', 'unknown'],
)
def test_net_instruments_and_brinson_inputs_fill_only_confirmed_inactive_pnl(
        dtype: str, missing: object, cost: float, exposure: float | None,
) -> None:
    """Prior exposure, not a closing position or a fixture label, permits a cost-only fill."""
    strategy, _ = _cost_only_portfolios(dtype, missing)
    strategy.weights = strategy.weights.copy(deep=True)
    strategy.units = strategy.units.copy(deep=True)
    beginning_exposure = missing if exposure is None else exposure
    strategy.weights.loc[strategy.weights.index[0], 'Cost only'] = beginning_exposure
    strategy.units.loc[strategy.units.index[0], 'Cost only'] = beginning_exposure
    strategy.realized_costs.loc[strategy.realized_costs.index[1], 'Cost only'] = cost
    originals = [frame.copy(deep=True) for frame in (
        strategy.instrument_pnl, strategy.realized_costs, strategy.weights, strategy.units,
    )]
    expected = strategy.instrument_pnl.copy(deep=True)
    expected_raw = expected.copy(deep=True)
    if exposure == 0.0:
        expected.loc[expected.index[1], 'Cost only'] = -cost / 100.0
        expected_raw.loc[expected_raw.index[1], 'Cost only'] = -cost

    direct = strategy.get_instruments_pnl(is_net=True)
    raw = strategy.get_instruments_pnl(is_net=True, is_unit_based_traded_volume=False)
    net, applied_weights = strategy.get_brinson_inputs(freq=None, is_net=True)

    pd.testing.assert_frame_equal(direct, expected)
    pd.testing.assert_frame_equal(raw, expected_raw)
    pd.testing.assert_frame_equal(net, expected.iloc[1:])
    pd.testing.assert_frame_equal(applied_weights, strategy.weights.shift(1).iloc[1:])
    pd.testing.assert_frame_equal(strategy.get_instruments_pnl(), originals[0])
    for current, original in zip((
        strategy.instrument_pnl, strategy.realized_costs, strategy.weights, strategy.units,
    ), originals):
        pd.testing.assert_frame_equal(current, original)


@pytest.mark.parametrize(
    ('dtype', 'missing'), [('float64', np.nan), ('Float64', pd.NA)],
    ids=['float64', 'nullable-float64'],
)
@pytest.mark.parametrize('exposure', [0.0, 1.0, None], ids=['inactive', 'held', 'unknown'])
def test_net_instruments_and_brinson_inputs_align_exposure_to_cost_only_columns(
        dtype: str, missing: object, exposure: float | None,
) -> None:
    """Cost-only columns require the same inactivity evidence as missing P&L cells."""
    strategy, _ = _cost_only_portfolios(dtype, missing)
    strategy.weights = strategy.weights.copy(deep=True)
    strategy.units = strategy.units.copy(deep=True)
    beginning_exposure = missing if exposure is None else exposure
    strategy.weights.loc[strategy.weights.index[0], 'Cost only'] = beginning_exposure
    strategy.units.loc[strategy.units.index[0], 'Cost only'] = beginning_exposure
    strategy.instrument_pnl = strategy.instrument_pnl.drop(columns='Cost only')
    expected = strategy.instrument_pnl.copy(deep=True)
    expected.insert(0, 'Cost only', pd.Series(
        [missing, -0.01 if exposure == 0.0 else missing, 0.0],
        index=expected.index, dtype=dtype,
    ))

    pd.testing.assert_frame_equal(strategy.get_instruments_pnl(is_net=True), expected)
    net, _ = strategy.get_brinson_inputs(freq=None, is_net=True)
    pd.testing.assert_frame_equal(net, expected.iloc[1:])


@pytest.mark.parametrize(
    ('dtype', 'missing'), [('float64', np.nan), ('Float64', pd.NA)],
    ids=['float64', 'nullable-float64'],
)
@pytest.mark.parametrize('absent_axis', ['row', 'column'])
def test_net_instruments_and_brinson_inputs_preserve_pnl_without_exposure_labels(
        dtype: str, missing: object, absent_axis: str,
) -> None:
    """An absent exposure label is unknown, even beside a confirmed inactive instrument."""
    strategy, _ = _cost_only_portfolios(dtype, missing)
    if absent_axis == 'row':
        strategy.weights = strategy.weights.iloc[1:].copy()
    else:
        strategy.weights = strategy.weights.drop(columns='Cost only')
    strategy.realized_costs = strategy.realized_costs.iloc[:, ::-1]
    expected = strategy.instrument_pnl.copy(deep=True)

    pd.testing.assert_frame_equal(strategy.get_instruments_pnl(is_net=True), expected)
    net, _ = strategy.get_brinson_inputs(freq=None, is_net=True)
    pd.testing.assert_frame_equal(net, expected.iloc[1:])


@pytest.mark.parametrize(
    ('dtype', 'missing'), [('float64', np.nan), ('Float64', pd.NA)],
    ids=['float64', 'nullable-float64'],
)
def test_net_instruments_and_brinson_inputs_preserve_missing_costs_and_opening_baseline(
        dtype: str, missing: object,
) -> None:
    """Missing costs leave finite gross P&L intact; opening costs stay in baseline NAV."""
    strategy, _ = _cost_only_portfolios(dtype, missing)
    strategy.realized_costs.iloc[0] = 25.0
    strategy.realized_costs.iloc[1] = missing
    expected = strategy.instrument_pnl.copy(deep=True)

    pd.testing.assert_frame_equal(strategy.get_instruments_pnl(is_net=True), expected)
    net, _ = strategy.get_brinson_inputs(freq=None, is_net=True)
    pd.testing.assert_frame_equal(net, expected.iloc[1:])


def test_brinson_inputs_reject_sparse_cost_dates_before_subtraction() -> None:
    """A partial cost axis cannot silently erase a later valid contribution."""
    strategy, benchmark = _cost_only_portfolios('float64', np.nan)
    strategy.instrument_pnl.loc[strategy.instrument_pnl.index[-1], 'Live'] = 0.02
    strategy.nav.iloc[-1] = 103.0
    strategy.realized_costs = strategy.realized_costs.iloc[:2]
    gross, _ = strategy.get_brinson_inputs(freq=None, is_net=False)
    assert gross.loc[gross.index[-1], 'Live'] == pytest.approx(0.02)

    with pytest.raises(ValueError, match='NAV index and cost report index'):
        strategy.get_instruments_pnl(is_net=True)
    with pytest.raises(ValueError, match='NAV index and cost report index'):
        strategy.get_brinson_inputs(freq=None, is_net=True)
    with pytest.raises(ValueError, match='NAV index and cost report index'):
        qis.MultiPortfolioData([strategy, benchmark]).compute_brinson_attribution(
            freq=None, is_net=True,
        )
