"""The factsheet facade compounds return input from a base level before the first return.

``qis.returns_to_nav`` keeps the index of its input, so a return series without a leading
missing row becomes a NAV whose first level is ``1 + r_1``. Fed to a report, that NAV drops the
first return from every statistic. ``qis.factsheet(..., data_is_returns=True)`` therefore adds a
base observation one native period before a non-zero first return. These tests capture the panel
the facade hands to the multi-asset generator, so no figure is rendered, and compare it with
products of the returns computed directly in numpy.
"""

import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets import generate_synthetic_universe
from qis.portfolio.reports import multi_assets_factsheet


@pytest.fixture(scope='module')
def daily_returns() -> pd.Series:
    """daily simple returns with no leading missing row, as pyfolio and QuantStats take them."""
    universe = generate_synthetic_universe(start='2018-01-02', end='2025-12-31', seed=20260725,
                                           apply_quirks=False)
    return universe.prices['SEQ_US'].pct_change().dropna().rename('strategy')


@pytest.fixture
def captured(monkeypatch) -> dict:
    """replace the multi-asset generator with a recorder of the arguments the facade passes."""
    calls = {}

    def capture(**kwargs):
        calls.update(kwargs)
        return []

    monkeypatch.setattr(multi_assets_factsheet, 'generate_multi_asset_factsheet', capture)
    return calls


def _wealth_with_base(returns: pd.Series) -> np.ndarray:
    """reference levels 1.0, 1 + r_1, (1 + r_1)(1 + r_2), ... built without qis."""
    return np.concatenate([[1.0], np.cumprod(1.0 + returns.to_numpy())])


def test_daily_returns_compound_from_a_base_observation(daily_returns, captured):
    qis.factsheet(daily_returns, data_is_returns=True)
    navs = captured['prices']['strategy']
    base_date = daily_returns.index[0] - pd.offsets.BDay(1)

    assert navs.index[0] == base_date
    assert navs.iloc[0] == 1.0
    assert navs.index[1:].equals(daily_returns.index)
    wealth = _wealth_with_base(daily_returns)
    np.testing.assert_allclose(navs.to_numpy(), wealth, rtol=1e-12)
    assert captured['time_period'].start == base_date

    # the statistics now include the first return: total return 0.559374 and max drawdown
    # -0.286942 on this sample, as pyfolio and QuantStats report from the same returns
    table = qis.compute_ra_perf_table(prices=captured['prices'])
    total_return = wealth[-1] - 1.0
    max_dd = np.min(wealth / np.maximum.accumulate(wealth) - 1.0)
    assert table.loc['strategy', qis.PerfStat.TOTAL_RETURN.to_str()] == pytest.approx(total_return,
                                                                                      rel=1e-10)
    assert table.loc['strategy', qis.PerfStat.MAX_DD.to_str()] == pytest.approx(max_dd, rel=1e-10)
    assert table.loc['strategy', qis.PerfStat.NUM_OBS.to_str()] == len(daily_returns)


def test_benchmark_returns_compound_from_a_base_observation(daily_returns, captured):
    benchmark_returns = 0.5 * daily_returns.rename('benchmark')
    qis.factsheet(daily_returns, benchmark_prices=benchmark_returns, data_is_returns=True)
    benchmark_navs = captured['benchmark_prices']

    assert benchmark_navs.index[0] == daily_returns.index[0] - pd.offsets.BDay(1)
    np.testing.assert_allclose(benchmark_navs.to_numpy(), _wealth_with_base(benchmark_returns),
                               rtol=1e-12)


def test_monthly_base_is_the_previous_month_end(daily_returns, captured):
    monthly_returns = (1.0 + daily_returns).resample('ME').prod().sub(1.0)
    qis.factsheet(monthly_returns, data_is_returns=True)
    navs = captured['prices']['strategy']

    assert navs.index[0] == monthly_returns.index[0] - pd.offsets.MonthEnd(1)
    np.testing.assert_allclose(navs.to_numpy(), _wealth_with_base(monthly_returns), rtol=1e-12)


@pytest.mark.parametrize('first_row', [np.nan, 0.0], ids=['missing', 'zero'])
def test_existing_base_row_is_not_duplicated(daily_returns, captured, first_row):
    """pct_change() output and pct_change().fillna(0) already start from a base."""
    returns = daily_returns.copy()
    returns.iloc[0] = first_row
    qis.factsheet(returns, data_is_returns=True)

    pd.testing.assert_frame_equal(captured['prices'], qis.returns_to_nav(returns).to_frame())


def test_ragged_panel_adds_a_base_only_where_the_first_return_is_observed(daily_returns, captured):
    dates = daily_returns.index
    returns = pd.DataFrame({'early': daily_returns,
                            'late': daily_returns.where(dates >= dates[9]),
                            'zero_first': daily_returns.where(dates > dates[0], 0.0)})
    qis.factsheet(returns, data_is_returns=True)
    navs = captured['prices']
    base_date = daily_returns.index[0] - pd.offsets.BDay(1)

    assert navs.index[0] == base_date
    assert navs.loc[base_date, 'early'] == 1.0
    assert navs.loc[base_date, ['late', 'zero_first']].isna().all()
    np.testing.assert_allclose(navs['early'].to_numpy(), _wealth_with_base(daily_returns),
                               rtol=1e-12)
    # columns that already start from a base keep the levels returns_to_nav gives them
    unchanged = qis.returns_to_nav(returns[['late', 'zero_first']])
    pd.testing.assert_frame_equal(navs[['late', 'zero_first']].iloc[1:], unchanged,
                                  check_freq=False)
