
"""Numerical contracts for the optional accounting partition throughout QIS."""
import numpy as np
import pandas as pd
import pytest
import qis

from qis.portfolio.tests.brinson_wrapper_test import make_portfolios
from qis.portfolio.attribution.group_exposures import (
    resolve_group_partition, resolve_reporting_inputs,
)
from qis.portfolio.risk.contributions import compute_group_portfolio_risk_contribution_ratios


@pytest.fixture
def mixed():
    """Return a partially overridden partition and portfolios with different assets."""
    a, b = make_portfolios(.002)
    matrix = pd.DataFrame({"PE": [.6], "Credit": [.4]}, index=["SEQ_US"])
    return a, b, matrix


@pytest.mark.parametrize("bad", [
    pd.DataFrame({"PE": [.9]}, index=["a"]),
    pd.DataFrame({"PE": [1.1], "Credit": [-.1]}, index=["a"]),
    pd.DataFrame({"PE": [np.nan]}, index=["a"]),
    pd.DataFrame({"PE": [np.inf]}, index=["a"]),
    pd.DataFrame([[.5, .5]], index=["a"], columns=["PE", "PE"]),
    pd.DataFrame({"PE": [1., 1.]}, index=["a", "a"]),
])
def test_invalid_fractional_rows_are_never_repaired(bad):
    """Even unused supplied rows must satisfy the explicit partition contract."""
    with pytest.raises(ValueError):
        resolve_group_partition(pd.Series({"b": "Bonds"}), pd.Index(["b"]), bad)


def test_series_label_contract_and_missing_fallback(mixed):
    """A matrix is a separate input; unresolved memberships fail explicitly."""
    a, b, matrix = mixed
    with pytest.raises(TypeError, match="Series"):
        resolve_group_partition(matrix, a.weights.columns, matrix)
    with pytest.raises(ValueError, match="Missing"):
        resolve_group_partition(pd.Series(dtype=object), a.weights.columns, matrix)
    with pytest.raises(TypeError, match="lookthrough_matrix"):
        qis.MultiPortfolioData([a, b]).compute_brinson_attribution(group_data=matrix)


def test_conflicting_labels_require_explicit_override(mixed):
    """Comparisons cannot silently use different classifications for the same asset."""
    a, b, matrix = mixed
    b.group_data.loc["SEQ_US"] = "Different"
    # The explicit fractional row removes this ambiguity.
    labels, full = resolve_reporting_inputs([a, b], lookthrough_matrix=matrix)
    assert full.loc["SEQ_US", "PE"] == .6
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_reporting_inputs(
            [a, b], lookthrough_matrix=matrix.rename(index={"SEQ_US": "other"}))


@pytest.mark.parametrize("method,kwargs", [
    ("get_weights", dict(is_grouped=True, freq=None, add_total=True)),
    ("get_group_navs", dict(is_add_group_total=True)),
    ("get_turnover", dict(is_grouped=True, roll_period=None)),
    ("get_costs", dict(is_grouped=True, roll_period=None)),
    ("compute_portfolio_vars", dict(freq=None, vol_span=3)),
])
def test_one_hot_partition_matches_categorical_results(mixed, method, kwargs):
    """A one-hot partition reproduces the old grouping, including total placement."""
    a, _, _ = mixed
    labels = a.group_data
    matrix = pd.get_dummies(labels, dtype=float).reindex(columns=a.group_order)
    expected = getattr(a, method)(**kwargs)
    actual = getattr(a, method)(**kwargs, lookthrough_matrix=matrix)
    if isinstance(expected, tuple):
        for got, wanted in zip(actual, expected):
            if wanted is None:
                assert got is None
            else:
                pd.testing.assert_frame_equal(got, wanted, atol=1e-12, rtol=1e-12)
    else:
        pd.testing.assert_frame_equal(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("method,kwargs", [
    ("get_turnover", dict(roll_period=None)),
    ("get_costs", dict(roll_period=None)),
])
def test_trading_panels_split_actual_instrument_trading(mixed, method, kwargs):
    """Accounting fractions attribute fund trading without inventing sleeve trades."""
    a, _, matrix = mixed
    raw = getattr(a, method)(is_grouped=False, add_total=False, **kwargs)
    got = getattr(a, method)(is_grouped=True, add_total=False,
                            lookthrough_matrix=matrix, **kwargs)
    np.testing.assert_allclose(got.PE, .6 * raw.SEQ_US, equal_nan=True)
    np.testing.assert_allclose(got.Credit, .4 * raw.SEQ_US, equal_nan=True)
    np.testing.assert_allclose(got.Bonds, raw.SBD_TSY, equal_nan=True)


def test_long_short_and_cumulative_pnl_reconcile(mixed):
    """Split signed holdings and their contributions before aggregating each sleeve."""
    a, _, matrix = mixed
    a.weights.loc[a.weights.index[::2], "SEQ_US"] *= -1
    totals, pieces = a.get_grouped_long_short_weights(weights_freq=None,
                                                     lookthrough_matrix=matrix)
    np.testing.assert_allclose(pieces["PE"].SEQ_US, a.weights.SEQ_US * .6)
    np.testing.assert_allclose(totals["PE"]["Net Long"], (a.weights.SEQ_US * .6).clip(lower=0))
    np.testing.assert_allclose(totals["PE"]["Net Short"], (a.weights.SEQ_US * .6).clip(upper=0))
    totals, pieces = a.get_grouped_cum_pnls(lookthrough_matrix=matrix)
    expected = a.get_instruments_pnl(is_compounded=False).fillna(0).SEQ_US.cumsum() * .6
    np.testing.assert_allclose(pieces["PE"].SEQ_US, expected)
    np.testing.assert_allclose(totals["PE"]["Total"], expected)


def test_group_nav_is_compounded_split_contributions(mixed):
    """Sleeve NAVs compound fractional contributions; final performance is not split."""
    a, _, matrix = mixed
    got = a.get_group_navs(lookthrough_matrix=matrix)
    pnl = a.get_instruments_pnl()
    expected = qis.returns_to_nav(pnl.SEQ_US.mul(.6).to_frame("PE"))
    pd.testing.assert_series_equal(got.PE, expected.PE)


def test_standalone_and_euler_risk_match_independent_covariance_reference():
    """Standalone sleeve TRE and additive Euler risk have distinct covariance contracts."""
    assets = pd.Index(["mixed", "bond"])
    covar = pd.DataFrame([[.04, .012], [.012, .01]], index=assets, columns=assets)
    labels = pd.Series({"mixed": "Alts", "bond": "Debt"})
    matrix = pd.DataFrame({"Growth": [.6], "Debt": [.4]}, index=["mixed"])
    w = pd.Series([.7, .3], index=assets)
    b = pd.Series([.4, .6], index=assets)
    date = pd.Timestamp("2025-12-31")
    model = qis.RiskModel(covar={date: covar})
    active = (w-b).to_numpy()
    partition = np.array([[.6, .4], [0., 1.]])
    total = np.sqrt(active @ covar.to_numpy() @ active)
    got = model.compute_tre_at_date(b, w, date, labels, lookthrough_matrix=matrix)
    expected = [total]
    for g in range(2):
        part = active * partition[:, g]
        expected.append(np.sqrt(part @ covar.to_numpy() @ part))
    np.testing.assert_allclose(got, expected, atol=1e-14)
    mcte = model.compute_marginal_tre_at_date(b, w, date, labels,
                                             lookthrough_matrix=matrix)
    asset_euler = active * (covar.to_numpy() @ active) / total
    np.testing.assert_allclose(mcte.loc[["Growth", "Debt"], "mcte"],
                               partition.T @ asset_euler, atol=1e-14)
    assert mcte.loc[["Growth", "Debt"], "mcte"].sum() == pytest.approx(total)
    history = model.compute_tre_history(b, w, labels, lookthrough_matrix=matrix)
    np.testing.assert_allclose(history.iloc[0], got)
    rc = compute_group_portfolio_risk_contribution_ratios(
        w, covar, group_data=labels, lookthrough_matrix=matrix)
    variance = w.to_numpy() @ covar.to_numpy() @ w.to_numpy()
    expected_rc = partition.T @ (w.to_numpy() * (covar.to_numpy() @ w.to_numpy()) / variance)
    np.testing.assert_allclose(rc, expected_rc, atol=1e-14)


def test_generic_risk_loadings_stay_signed_and_overlapping():
    """The pre-existing general group-loadings API is not made into a partition API."""
    assets = pd.Index(["a", "b"])
    covar = pd.DataFrame(np.eye(2), index=assets, columns=assets)
    date = pd.Timestamp("2025-12-31")
    model = qis.RiskModel(covar={date: covar})
    w, b = pd.Series([.7, .3], index=assets), pd.Series(0., index=assets)
    loadings = pd.DataFrame({"overlap": [2., -1.], "other": [1., 1.]}, index=assets)
    result = model.compute_tre_by_group_loadings_at_date(b, w, date, loadings)
    assert result["overlap"] == pytest.approx(np.sqrt(1.4**2+.3**2))
    with pytest.raises(ValueError):
        model.compute_tre_at_date(b, w, date, lookthrough_matrix=loadings)


def test_fractional_var_uses_positions_and_original_returns(mixed):
    """Compare each sleeve to an independent unsplit one-portfolio VaR calculation."""
    a, _, matrix = mixed
    got = qis.compute_portfolio_correlated_var_by_groups(
        a.prices, a.weights, a.group_data, freq=None, vol_span=4, lookthrough_matrix=matrix)
    weights = a.weights.copy()
    weights["SEQ_US"] *= .6
    weights["SBD_TSY"] = 0
    reference = qis.compute_portfolio_correlated_var_by_groups(
        a.prices, weights, freq=None, vol_span=4)
    np.testing.assert_allclose(got.PE, reference["Total VAR"], atol=1e-13, equal_nan=True)


def test_multi_portfolio_grouped_comparisons_share_union(mixed):
    """Both sides retain all requested sleeves despite different native instruments."""
    a, b, matrix = mixed
    multi = qis.MultiPortfolioData([a, b])
    aw, bw = multi.get_aligned_weights(is_grouped=True, lookthrough_matrix=matrix)
    assert aw.columns.equals(bw.columns)
    expected_order = multi.compute_brinson_attribution(
        lookthrough_matrix=matrix)[0].index[:-1].tolist()
    assert aw.columns.tolist() == expected_order
    assert set(aw) == {"PE", "Credit", "Bonds"}
    groups = multi.get_grouped_weights(freq=None, lookthrough_matrix=matrix)
    assert set(groups) == {"PE", "Credit", "Bonds"}
    assert set(groups["PE"]) == {"Strategy", "Benchmark"}


def test_compact_panels_recombine_duplicate_fractional_instruments():
    """Many sleeves collapse to one original instrument, without double counting."""
    from qis.portfolio._asset_display import compact_groups
    index = pd.date_range("2025-01-01", periods=2)
    pieces = {f"g{i}": pd.DataFrame({"fund": [.1, .2]}, index=index) for i in range(12)}
    aggregates = {g: p.rename(columns={"fund": "Total"}) for g, p in pieces.items()}
    totals, actual = compact_groups(aggregates, pieces)
    panel = next(iter(actual.values()))
    assert panel.columns.tolist() == ["fund"]
    np.testing.assert_allclose(panel.fund, [1.2, 2.4])


def test_facade_does_not_silently_drop_matrix_on_raw_prices(mixed):
    """Raw prices lack portfolio accounting weights and cannot support this option."""
    a, _, matrix = mixed
    with pytest.raises(ValueError, match="PortfolioData"):
        qis.factsheet(a.prices, lookthrough_matrix=matrix)
