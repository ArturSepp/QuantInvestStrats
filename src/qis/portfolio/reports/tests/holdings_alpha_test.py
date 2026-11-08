"""Independent holdings-weighted EWMA reference, timing and strict coverage contracts."""

import numpy as np
import pandas as pd
import pytest

from qis.datasets.synthetic import generate_synthetic_prices
from qis.portfolio.reports._holdings_alpha import compute_holdings_alpha


@pytest.fixture(scope="module")
def inputs():
    prices = generate_synthetic_prices(end="2025-12-31", apply_quirks=False)
    prices = prices[["SEQ_US", "SBD_TSY"]].resample("ME").last()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.SEQ_US = 0.6 + 0.15 * np.sin(np.arange(len(prices)) / 7)
    weights.SBD_TSY = 1.2 - weights.SEQ_US  # actual 120% exposure, not renormalised
    benchmark = prices[["SEQ_US"]].rename(columns={"SEQ_US": "Reference"})
    returns = prices.divide(prices.shift()).subtract(1)
    portfolio = returns.mul(weights.shift()).sum(axis=1, skipna=False)
    return prices, weights, benchmark, portfolio, benchmark.divide(benchmark.shift()).subtract(1)


def run(inputs, start=40, span=12):
    prices, weights, benchmark, portfolio, reference = inputs
    return compute_holdings_alpha(
        prices, weights, benchmark, portfolio.iloc[start:], reference.iloc[start:], "ME", span
    )


def independent_beta(prices, weights, benchmark, span=12):
    """Pandas means + explicit scalar cross-moment recursion, not QIS beta machinery."""
    x = np.log(benchmark.iloc[:, 0]).diff()
    y = np.log(prices).diff()
    x = x - x.ewm(span=span, adjust=False).mean()
    y = y - y.ewm(span=span, adjust=False).mean()
    decay = 1 - 2 / (span + 1)
    variance, covariance = 0.0, np.zeros(prices.shape[1])
    loadings = np.full(prices.shape, np.nan)
    for i in range(1, len(prices)):
        variance = decay * variance + (1 - decay) * x.iloc[i] ** 2
        covariance = decay * covariance + (1 - decay) * x.iloc[i] * y.iloc[i].to_numpy()
        if i > span:
            loadings[i] = covariance / variance
    return pd.DataFrame(loadings, index=prices.index, columns=prices.columns).mul(weights).sum(
        axis=1, skipna=False
    )


def test_dynamic_weights_and_pre_report_beta_match_independent_reference(inputs):
    prices, weights, benchmark, portfolio, reference = inputs
    alpha, diagnostics, meta, issues = run(inputs)
    expected_beta = independent_beta(prices, weights, benchmark).shift().iloc[40:]
    np.testing.assert_allclose(diagnostics.lagged_beta, expected_beta, rtol=1e-12, atol=1e-13)
    expected_alpha = portfolio.iloc[40:] - expected_beta * reference.iloc[40:, 0]
    np.testing.assert_allclose(alpha, expected_alpha, rtol=1e-12, atol=1e-14)
    assert pd.notna(alpha.iloc[0])  # opening beta retained despite later performance start
    assert diagnostics.complete_coverage.all()
    assert meta["estimation_start"] == prices.index[1]
    assert meta["factor_beta_span"] == 12 and not issues


def test_cash_adjustment_identity_and_no_future_leak(inputs):
    prices, weights, benchmark, portfolio, reference = inputs
    cash = pd.Series(0.002, index=portfolio.index)
    excess_inputs = (prices, weights, benchmark, portfolio - cash, reference.sub(cash, axis=0))
    alpha, diagnostics, _, _ = run(excess_inputs)
    total_alpha, _, _, _ = run(inputs)
    np.testing.assert_allclose(alpha, total_alpha + (diagnostics.lagged_beta - 1) * 0.002)
    cut = prices.index[150]
    changed = [value.copy() for value in excess_inputs]
    changed[0].loc[cut:] *= 1.4
    changed[1].loc[cut:] *= 2.0
    changed[2].loc[cut:] *= 0.8
    updated, updated_diagnostics, _, _ = run(tuple(changed))
    pd.testing.assert_series_equal(alpha.loc[:cut], updated.loc[:cut])
    pd.testing.assert_frame_equal(diagnostics.loc[:cut], updated_diagnostics.loc[:cut])


def test_per_instrument_warmup_partial_coverage_is_not_alpha(inputs):
    changed = [value.copy() for value in inputs]
    changed[0].loc[:changed[0].index[50], "SBD_TSY"] = np.nan
    alpha, diagnostic, _, issues = run(tuple(changed), start=1)
    # First young-asset return is at 52; 13 paired returns end at 64; lag makes 65 eligible.
    assert alpha.loc[:changed[0].index[64]].isna().all()
    assert pd.notna(alpha.loc[changed[0].index[65]])
    assert not diagnostic.complete_coverage.iloc[0]
    assert "fully covered" in issues[0]


@pytest.mark.parametrize(
    "kind", ["missing_price", "missing_bin", "missing_reference", "unknown_weight", "no_price"]
)
def test_missing_held_data_is_never_filled_or_renormalised(inputs, kind):
    changed = [value.copy() for value in inputs]
    date = changed[0].index[80]
    next_date = changed[0].index[81]
    if kind == "missing_price":
        changed[0].loc[date, "SBD_TSY"] = np.nan
    elif kind == "missing_bin":
        changed[0] = changed[0].drop(index=date)
    elif kind == "missing_reference":
        changed[2].loc[date] = np.nan
    elif kind == "unknown_weight":
        changed[1].loc[date, "SBD_TSY"] = np.nan
    else:
        changed[0] = changed[0].drop(columns="SBD_TSY")
    alpha, diagnostics, _, issues = run(tuple(changed))
    assert pd.isna(alpha.loc[next_date])
    assert not diagnostics.complete_coverage.loc[next_date]
    assert issues


def test_all_cash_is_zero_beta_not_unknown_holdings(inputs):
    changed = [value.copy() for value in inputs]
    date = changed[0].index[80]
    next_date = changed[0].index[81]
    changed[1].loc[date] = 0.0
    changed[0].loc[date, "SBD_TSY"] = np.nan
    alpha, diagnostics, _, _ = run(tuple(changed))
    assert diagnostics.lagged_beta.loc[next_date] == 0.0
    assert diagnostics.complete_coverage.loc[next_date]
    assert alpha.loc[next_date] == inputs[3].loc[next_date]


def test_grid_and_single_benchmark_are_explicit(inputs):
    prices, weights, benchmark, portfolio, reference = inputs
    with pytest.raises(ValueError, match="exactly one"):
        compute_holdings_alpha(prices, weights, prices, portfolio, reference, "ME", 12)
    with pytest.raises(ValueError, match="match exactly"):
        compute_holdings_alpha(prices, weights, benchmark, portfolio, reference.iloc[1:], "ME", 12)
