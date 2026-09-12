"""Independent functional contracts for reusable factor stress testing."""
import math

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.risk.stress_testing import (
    conditional_factor_covariance, conditional_factor_shock,
    compute_conditional_scenario_band, compute_factor_sensitivity,
    duration_log_shock, price_target_log_shock, project_factor_scenarios, return_log_shock,
)


def inputs():
    """Return a hand-worked two-factor, two-asset annual model."""
    factors = pd.Index(["Equity", "Rates"])
    assets = pd.Index(["Stock", "Fund"])
    covariance = pd.DataFrame([[.04, .01], [.01, .01]], index=factors, columns=factors)
    betas = pd.DataFrame([[1., 0.], [2., 1.]], index=assets, columns=factors)
    residual = pd.Series([.0025, .01], index=assets)
    amounts = pd.Series([60., 40.], index=assets)
    return covariance, betas, residual, amounts


def test_anchor_conversions_have_explicit_decimal_units():
    """Price and duration anchors preserve the reviewed signs and no-carry convention."""
    assert price_target_log_shock(100., 120.) == pytest.approx(math.log(1.2))
    assert duration_log_shock(-.005, 8.) == pytest.approx(math.log(1.04))
    assert duration_log_shock(.005, 8.) == pytest.approx(math.log(.96))
    assert return_log_shock(-.2, .5) == pytest.approx(math.log(.9))
    assert return_log_shock(.2) == pytest.approx(math.log(1.2))


@pytest.mark.parametrize("function,args", [
    (price_target_log_shock, (0., 1.)), (price_target_log_shock, (1., -1.)),
    (price_target_log_shock, (np.nan, 1.)), (price_target_log_shock, (1., np.inf)),
    (return_log_shock, (-1.,)), (return_log_shock, (-.6, 2.)),
    (return_log_shock, (np.inf,)), (duration_log_shock, (.2, 8.)),
    (duration_log_shock, (.01, -1.)), (duration_log_shock, (np.nan, 1.)),
])
def test_invalid_anchor_endpoints_fail(function, args):
    """Invalid economic endpoints and nonfinite values cannot become valid shocks."""
    with pytest.raises(ValueError):
        function(*args)


def test_single_anchor_conditioning_keeps_explicit_zeros_and_order():
    """The covariance ratio is .25, not the correlation .5; zero is a real anchor."""
    cov, _, _, _ = inputs()
    anchor = math.log(.8)
    shock = conditional_factor_shock(cov, {"Equity": anchor})
    np.testing.assert_allclose(shock, [anchor, .25 * anchor], atol=1e-14)
    fixed = conditional_factor_shock(cov, {"Rates": 0., "Equity": anchor})
    np.testing.assert_allclose(fixed, [anchor, 0.], atol=1e-14)
    assert fixed.index.equals(cov.index)


def test_joint_credit_covariance_and_shocks_match_closed_form_inverse():
    """Two correlated anchors use the same joint solve, not a sum of marginal solves."""
    labels = ["Credit", "Credit EM", "Equity"]
    cov = pd.DataFrame([[.01, .004, .006], [.004, .01, .002], [.006, .002, .04]],
                       index=labels, columns=labels)
    original = cov.copy(deep=True)
    z = math.log(.8)
    shock = conditional_factor_shock(cov, {"Credit": z, "Credit EM": z})
    assert shock.Equity == pytest.approx((.006 + .002) / .014 * z)
    reduction = (.01 * .006**2 - 2 * .004 * .006 * .002 + .01 * .002**2) / (
        .01**2 - .004**2)
    cond = conditional_factor_covariance(cov, ["Credit", "Credit EM"])
    np.testing.assert_array_equal(cond.iloc[:2], 0.)
    assert cond.loc["Equity", "Equity"] == pytest.approx(.04 - reduction)
    assert np.linalg.eigvalsh(cond).min() >= 0.
    np.testing.assert_array_equal(conditional_factor_covariance(cov, labels), 0.)
    pd.testing.assert_frame_equal(cov, original)


@pytest.mark.parametrize("anchors", [[], ["Missing"], ["Equity", "Equity"]])
def test_invalid_anchor_identities_rejected(anchors):
    """Unknown, repeated and absent anchors are explicit errors."""
    cov, _, _, _ = inputs()
    with pytest.raises(ValueError, match="anchors"):
        conditional_factor_covariance(cov, anchors)


@pytest.mark.parametrize("kind", ["asymmetric", "negative", "nonfinite", "misaligned", "singular"])
def test_invalid_covariance_rejected(kind):
    """No silent symmetrisation, relabelling or pseudo-inverse of ambiguous anchors."""
    cov, _, _, _ = inputs()
    if kind == "asymmetric":
        cov.iloc[0, 1] = .02
    elif kind == "negative":
        cov.iloc[0, 0] = -.01
    elif kind == "nonfinite":
        cov.iloc[0, 0] = np.nan
    elif kind == "misaligned":
        cov.columns = cov.columns[::-1]
    else:
        cov[:] = .01
    with pytest.raises(ValueError):
        conditional_factor_shock(cov, {"Equity": 0., "Rates": 0.})


def test_projection_exact_nonlinearity_short_positions_and_adjustment_reconcile():
    """Asset-first expm1 preserves a beta-two payoff and signed attribution cancellation."""
    _, betas, _, amounts = inputs()
    shocks = pd.DataFrame([[math.log(.8), 0.], [0., 0.]],
                          columns=betas.columns, index=["down", "flat"])
    first = project_factor_scenarios(betas, amounts, shocks)
    np.testing.assert_allclose(first.asset_pnl.loc["down"], [-12., -14.4], atol=1e-12)
    amounts.iloc[1] = -40.
    extra = pd.DataFrame(0., index=shocks.index, columns=betas.index)
    extra.loc["flat", "Stock"] = math.log(1.1)
    result = project_factor_scenarios(betas, amounts, shocks, extra)
    np.testing.assert_allclose(result.asset_pnl.loc["down"], [-12., 14.4], atol=1e-12)
    assert result.asset_pnl.loc["flat", "Stock"] == pytest.approx(6.)
    np.testing.assert_allclose(result.factor_attribution.sum(axis=1),
                               result.asset_pnl.sum(axis=1), atol=1e-12)
    adjustment = result.factor_attribution.loc["flat", "Anchor / residual adjustment"]
    assert adjustment == pytest.approx(6.)


def test_zero_log_sum_keeps_offsetting_factor_contributions():
    """The continuous expm1(g)/g limit retains two cancelling contributions."""
    _, betas, _, amounts = inputs()
    betas.loc["Stock"] = [1., 1.]
    shocks = pd.DataFrame([[.1, -.1]], columns=betas.columns)
    result = project_factor_scenarios(betas.iloc[:1], amounts.iloc[:1], shocks)
    np.testing.assert_allclose(result.factor_attribution.iloc[0], [6., -6., 0.])
    assert result.asset_pnl.iloc[0, 0] == 0.


@pytest.mark.parametrize("kind", ["amount_order", "factor_order", "adjustment", "nan", "overflow"])
def test_projection_rejects_bad_alignment_and_values(kind):
    """Mismatched labels and overflow fail before a plausible-looking report is returned."""
    _, betas, _, amounts = inputs()
    shocks = pd.DataFrame([[.1, -.1]], columns=betas.columns)
    extra = None
    if kind == "amount_order":
        amounts = amounts.iloc[::-1]
    elif kind == "factor_order":
        shocks = shocks.iloc[:, ::-1]
    elif kind == "adjustment":
        extra = pd.DataFrame([[0., 0.]], columns=betas.index[::-1])
    elif kind == "nan":
        betas.iloc[0, 0] = np.nan
    else:
        shocks.iloc[0, 0] = 1000.
    with pytest.raises(ValueError):
        project_factor_scenarios(betas, amounts, shocks, extra)


def test_conditional_band_known_variance_and_horizon_scaling():
    """Remaining Rates variance plus independent residuals gives annual variance .0037."""
    cov, betas, residual, amounts = inputs()
    centres = pd.Series([-.1, 0., .1], index=["down", "flat", "up"])
    monthly = compute_conditional_scenario_band(
        cov, betas, residual, amounts / 100, ["Equity"], centres, 1 / 12)
    expected = .4**2 * (.01 - .01**2 / .04) + .6**2 * .0025 + .4**2 * .01
    assert expected == pytest.approx(.0037)
    assert monthly.annual_total_vol**2 == pytest.approx(expected)
    width = 1.959963984540054 * math.sqrt(expected / 12)
    np.testing.assert_allclose(monthly.summary.band_half_width, width)
    np.testing.assert_allclose(monthly.summary.lower_bound, centres - width)
    quarterly = compute_conditional_scenario_band(
        cov, betas, residual, amounts / 100, ["Equity"], centres, .25)
    np.testing.assert_allclose(quarterly.summary.band_half_width / monthly.summary.band_half_width,
                               math.sqrt(3.))
    fixed = compute_conditional_scenario_band(
        cov, betas, residual, amounts / 100, list(cov.index), centres, 1.)
    assert fixed.annual_factor_vol == 0.
    assert fixed.annual_total_vol == pytest.approx(.05)


@pytest.mark.parametrize("horizon,confidence", [(0., .95), (np.nan, .95), (1., 1.), (1., np.nan)])
def test_invalid_band_parameters(horizon, confidence):
    """Horizon and probability are explicit, finite and physically meaningful."""
    cov, betas, residual, amounts = inputs()
    with pytest.raises(ValueError, match="Band"):
        compute_conditional_scenario_band(cov, betas, residual, amounts / 100,
                                          ["Equity"], pd.Series([0.]), horizon, confidence)


def test_sensitivity_preserves_grid_nav_and_signed_positions_without_mutation():
    """Independent scalar valuation agrees at every grid point for a long/short book."""
    cov, betas, residual, amounts = inputs()
    amounts.iloc[1] = -40.
    original = amounts.copy()
    grid = pd.Index([-.2, 0., .2], name="equity_return")
    result = compute_factor_sensitivity(cov, betas, residual, amounts, 100.,
                                        ["Equity"], grid, 1 / 12)
    expected = .6 * grid.to_numpy() - .4 * ((1 + grid.to_numpy())**2.25 - 1)
    np.testing.assert_allclose(result.band.summary.portfolio_return, expected, atol=1e-14)
    assert result.band.summary.index.equals(grid)
    pd.testing.assert_series_equal(amounts, original)
    assert result.anchors == ("Equity",)


@pytest.mark.parametrize("nav,grid", [(0., [0.]), (np.inf, [0.]), (100., [-1.]),
                                     (100., [0., 0.]), (100., []), (100., [np.nan])])
def test_sensitivity_invalid_nav_or_grid(nav, grid):
    """A silent NAV inference or invalid simple-return endpoint is forbidden."""
    cov, betas, residual, amounts = inputs()
    with pytest.raises(ValueError):
        compute_factor_sensitivity(cov, betas, residual, amounts, nav, ["Equity"],
                                    pd.Index(grid), 1 / 12)


def test_residual_variance_alignment_and_nonnegativity():
    """Residual risk cannot be silently omitted or aligned positionally."""
    cov, betas, residual, amounts = inputs()
    for bad in [residual.iloc[::-1], residual * -1, residual * np.nan]:
        with pytest.raises(ValueError, match="Residual"):
            compute_conditional_scenario_band(cov, betas, bad, amounts / 100, ["Equity"],
                                              pd.Series([0.]), 1 / 12)
