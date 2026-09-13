"""Reference checks for Brinson sector returns and compound-return linking."""
import numpy as np
import pandas as pd
import pytest

from qis import compute_brinson_attribution_table


def _inputs():
    """Return weighted contributions with known unweighted sector returns."""
    index = pd.date_range('2025-01-31', periods=3, freq='ME')
    columns = ['Equity', 'Bonds', 'Alternatives']
    sw = pd.DataFrame([[.6, .4, 0], [.7, .2, .1], [.5, .4, .1]], index=index, columns=columns)
    bw = pd.DataFrame([[.5, .5, 0]] * 3, index=index, columns=columns)
    sr = pd.DataFrame(
        [[.1, .05, 0], [-.2, .02, .04], [.15, .03, -.02]], index=index, columns=columns)
    br = pd.DataFrame([[.06, .02, 0], [-.1, .01, 0], [.08, .02, 0]], index=index, columns=columns)
    return dict(strategy_pnl=sw * sr, benchmark_pnl=bw * br,
                strategy_weights=sw, benchmark_weights=bw,
                asset_class_data=pd.Series(columns, index=columns)), sr, br




def test_identical_sector_returns_have_no_selection():
    """A pure overweight is allocation, including with interaction merged."""
    index = pd.DatetimeIndex(['2025-01-31'])
    sw = pd.DataFrame([[.6, .4]], index=index, columns=['Equity', 'Cash'])
    bw = pd.DataFrame([[.4, .6]], index=index, columns=sw.columns)
    returns = pd.DataFrame([[.1, 0.0]], index=index, columns=sw.columns)
    totals, _, allocation, selection, _ = compute_brinson_attribution_table(
        benchmark_pnl=bw * returns, strategy_pnl=sw * returns,
        strategy_weights=sw, benchmark_weights=bw,
        asset_class_data=pd.Series(sw.columns, index=sw.columns))
    assert allocation.loc[index[0], 'Equity'] == pytest.approx(.02)
    np.testing.assert_allclose(selection, 0.0, atol=1e-14)
    assert totals.loc['Total Sum', 'Total\nActive'] == pytest.approx(.02)


def test_brinson_uses_sector_returns_not_weighted_contributions():
    """Single-period allocation matches the textbook BHB equation."""
    inputs, sr, br = _inputs()
    _, _, allocation, selection, interaction = compute_brinson_attribution_table(
        **inputs, is_exclude_interaction_term=False, is_linked=False)
    sw, bw = inputs['strategy_weights'], inputs['benchmark_weights']
    expected_allocation = (sw - bw) * br
    expected_selection = bw * (sr - br)
    expected_interaction = (sw - bw) * (sr - br)
    for actual, expected in ((allocation, expected_allocation),
                             (selection, expected_selection),
                             (interaction, expected_interaction)):
        np.testing.assert_allclose(actual.reindex(columns=expected.columns), expected, atol=1e-14)


@pytest.mark.parametrize('merged', [False, True])
def test_linked_brinson_matches_recursive_reference_and_nav_difference(merged):
    """Linked effects reconcile at every prefix, including off-benchmark groups."""
    inputs, _, _ = _inputs()
    raw = compute_brinson_attribution_table(
        **inputs, is_exclude_interaction_term=merged, is_linked=False)
    linked = compute_brinson_attribution_table(
        **inputs, is_exclude_interaction_term=merged, is_linked=True)
    rp, rb = inputs['strategy_pnl'].sum(axis=1), inputs['benchmark_pnl'].sum(axis=1)
    for monthly, adjusted in zip(raw[2:], linked[2:]):
        prior = np.zeros(len(monthly.columns))
        wealth = 1.0
        reference = []
        for position in range(len(rp)):
            prior = prior * (1.0 + rb.iloc[position]) + monthly.iloc[position].to_numpy() * wealth
            wealth *= 1.0 + rp.iloc[position]
            reference.append(prior.copy())
        np.testing.assert_allclose(adjusted.cumsum(), reference, atol=1e-14)
    active = linked[2] + linked[3] + linked[4]
    expected = (1.0 + rp).cumprod() - (1.0 + rb).cumprod()
    np.testing.assert_allclose(active['Total Sum'].cumsum(), expected, atol=1e-14)
    total = linked[0].loc['Total Sum']
    assert total['Strategy\nReturn Total'] == pytest.approx((1.0 + rp).prod() - 1)
    assert total['Benchmark\nReturn Total'] == pytest.approx((1.0 + rb).prod() - 1)
    assert total['Total\nActive'] == pytest.approx(expected.iloc[-1])
    prefix_inputs = {key: value.iloc[:2] if isinstance(value, pd.DataFrame) else value
                     for key, value in inputs.items()}
    prefix = compute_brinson_attribution_table(
        **prefix_inputs, is_exclude_interaction_term=merged, is_linked=True)
    for full, shorter in zip(linked[2:], prefix[2:]):
        pd.testing.assert_frame_equal(full.iloc[:2], shorter)


@pytest.mark.parametrize('problem', ['dates', 'duplicates', 'classification', 'infinity', 'total'])
def test_invalid_inputs_fail_instead_of_silently_changing_attribution(problem):
    """Malformed periods and classification metadata cannot produce plausible totals."""
    inputs, _, _ = _inputs()
    if problem == 'dates':
        inputs['benchmark_weights'] = inputs['benchmark_weights'].iloc[1:]
    elif problem == 'duplicates':
        inputs['strategy_pnl'] = pd.concat([inputs['strategy_pnl']] * 2, axis=1)
    elif problem == 'classification':
        inputs['asset_class_data'] = inputs['asset_class_data'].drop('Bonds')
    elif problem == 'infinity':
        inputs['strategy_pnl'].iloc[0, 0] = np.inf
    else:
        inputs['asset_class_data'].iloc[0] = 'Total Sum'
    with pytest.raises(ValueError):
        compute_brinson_attribution_table(**inputs)


def test_linking_rejects_zero_or_negative_wealth():
    """Frongello's positive-wealth domain is validated explicitly."""
    inputs, _, _ = _inputs()
    inputs['strategy_pnl'].iloc[0] = [-1., 0., 0.]
    with pytest.raises(ValueError, match='greater than -100%'):
        compute_brinson_attribution_table(**inputs)
