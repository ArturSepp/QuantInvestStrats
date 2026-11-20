"""Independent one-year tail checks for dates, shared draws, compounding and coverage."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from qis.portfolio.risk.one_year_tail_risk import compute_one_year_tail_risk


def _panel():
    """Two assets with opposite monthly patterns and a known annual covariance."""
    dates = pd.date_range('2021-01-31', periods=24, freq='ME')
    returns = pd.DataFrame({'A': np.linspace(-.13, .16, 24),
                            'B': np.tile([.04, .09, -.07], 8)}, index=dates)
    weights = pd.Series({'B': .4, 'A': .6})
    bench = pd.Series({'B': 1.0})
    cov = pd.DataFrame([[.04, .01], [.01, .0225]], index=['A', 'B'], columns=['A', 'B'])
    return returns, weights, bench, cov


def test_circular_draws_compounding_and_fractional_tail():
    """Reconstruct four wrapping blocks per year and exactly 10.02 observations of ES."""
    returns, weights, bench, cov = _panel()
    seed, count = 17, 1002
    table = compute_one_year_tail_risk(cov, weights, bench, returns.index[-1], 'USD', returns,
                                       simulations=count, seed=seed)
    rng = np.random.RandomState(seed)
    starts = rng.randint(0, len(returns), size=(count, 4))
    assert (starts >= len(returns) - 2).any()
    strategy = .6 * returns.A.to_numpy() + .4 * returns.B.to_numpy()
    for label, monthly in [('strategy', strategy), ('benchmark', returns.B.to_numpy())]:
        outcomes = []
        for path in starts:
            wealth = 1.
            for start in path:
                for offset in range(3):
                    wealth *= 1. + monthly[(start + offset) % len(monthly)]
            outcomes.append(wealth - 1.)
        losses = np.mean(outcomes) - np.asarray(outcomes)
        ranked = np.sort(losses)[::-1]
        var = .01 * ranked[11] + .99 * ranked[10]  # linear 99th percentile of 1002
        es = (sum(ranked[:10]) + .02 * ranked[10]) / 10.02
        row = table.loc['Historical bootstrap']
        assert row[f'{label}_var'] == pytest.approx(var)
        assert row[f'{label}_es'] == pytest.approx(es)


def test_normal_reference_and_coverage():
    """Check risk in fractional units, the MC tolerance and the unavailable MC ES."""
    returns, weights, bench, cov = _panel()
    table = compute_one_year_tail_risk(cov, weights, bench, returns.index[-1], 'USD')
    analytic, mc, historical = (r for _, r in table.iterrows())
    for label, vol in [('strategy', np.sqrt(.6**2*.04 + .4**2*.0225 + 2*.6*.4*.01)),
                       ('benchmark', .15)]:
        assert analytic[f'{label}_var'] == pytest.approx(norm.ppf(.99) * vol)
        assert analytic[f'{label}_es'] == pytest.approx(norm.pdf(norm.ppf(.99)) * vol / .01)
        assert abs(mc[f'{label}_var'] / analytic[f'{label}_var'] - 1) < .015
        assert pd.isna(mc[f'{label}_es']) and pd.isna(historical[f'{label}_var'])
    assert 'not supplied' in historical.coverage


def test_matching_dates_gaps_and_unheld_assets():
    """Future rows, absent holdings and a gap must never form artificial return blocks."""
    returns, weights, bench, cov = _panel()
    date = returns.index[-3]
    before = returns.copy()
    before.loc[returns.index[3], 'B'] = np.nan
    before['unused'] = np.nan
    table = compute_one_year_tail_risk(cov, weights, bench, date, 'USD', before, simulations=1000)
    row = table.iloc[-1]
    assert (row.sample_start, row.sample_end, row.matched_months) == (returns.index[4], date, 18)
    after = before.copy()
    after.loc[after.index > date, 'A'] = 7000
    after = after.drop(returns.index[3])
    pd.testing.assert_frame_equal(table, compute_one_year_tail_risk(
        cov, weights, bench, date, 'USD', after, simulations=1000))
    for panel, reason in [(before.drop(columns='A'), 'Missing held assets'),
                          (before.tail(5), 'Fewer than 12')]:
        out = compute_one_year_tail_risk(cov, weights, bench, date, 'USD', panel)
        assert reason in out.iloc[-1].coverage and pd.isna(out.iloc[-1].strategy_es)
        assert out.iloc[0].strategy_es > 0


def test_missing_covariance_and_input_contract():
    """Historical estimation can stand alone; malformed inputs fail explicitly."""
    returns, weights, bench, cov = _panel()
    date = returns.index[-1]
    out = compute_one_year_tail_risk(None, weights, bench, date, 'USD', returns)
    assert pd.isna(out.iloc[0].strategy_var) and out.iloc[-1].strategy_es > 0
    assert out.iloc[1].paths == 0 and pd.isna(out.iloc[1].seed)
    assert out.iloc[-1].paths == 100_000
    future = date + pd.Timedelta(days=1)
    for options in [dict(block_months=5), dict(seed=-1), dict(covariance_date=future)]:
        with pytest.raises(ValueError):
            compute_one_year_tail_risk(cov, weights, bench, date, 'USD', returns, **options)
    with pytest.raises(ValueError, match='one row per month'):
        compute_one_year_tail_risk(cov, weights, bench, date, 'USD', pd.concat([returns, returns]))
    with pytest.raises(ValueError, match='reference currency'):
        compute_one_year_tail_risk(cov, weights, bench, date, '')
