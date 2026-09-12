# Factor stress testing

The public APIs in qis.portfolio.risk.stress_testing work with labelled pandas inputs
and have no dependency on a particular factor model, provider or optimisation package.

## Conventions

- Betas and scenario factor shocks use **log returns**. Simple returns are decimals.
- Positions are signed amounts in one reference currency. NAV is explicit and positive;
  neither positions nor NAV weights are renormalised.
- Factor covariance and independent asset residual variances are **annualised**.
  A band horizon T is explicitly measured in years; covariance scales by T.
- Identical ordered labels are required. Missing exposures, nonfinite inputs, ambiguous
  anchor sets and ill-conditioned anchor blocks are rejected, not filled or pseudo-inverted.
- Inputs are point-in-time snapshots selected by the caller. There is no data acquisition,
  refitting, historical date selection or interpolation in these functions.

## Shock construction

price_target_log_shock(P0, P1) returns log(P1/P0).
return_log_shock(r, effective_weight) returns log(1 + effective_weight*r), with
zero instantaneous carry on the uninvested portion.
duration_log_shock(delta_y, duration) returns log(1 - duration*delta_y).
The duration approximation is first order; delta_y is decimal yield, so 50bp is 0.005.

For anchored factors A and free factors F, conditional_factor_shock computes
z_F = Sigma_FA solve(Sigma_AA, z_A). All explicit anchors, including zero, remain fixed.
Multiple anchors use one joint solve; marginal scenarios are not added together.

conditional_factor_covariance computes the Schur complement:
C = Sigma_FF - Sigma_FA solve(Sigma_AA, Sigma_AF).
The result retains full factor order with zero anchor rows and columns.
Gaussian conditional covariance does not depend on the realised anchor values.

## Valuation and attribution

project_factor_scenarios computes g_i = beta_i.T z + adjustment_i, then
PnL_i = amount_i * expm1(g_i). Portfolio P&L is the sum of asset P&Ls.
Each additive log component receives its proportional share using expm1(g_i)/g_i
and the continuous limit 1 at zero. This reconciles exactly under the model,
including short positions and explicit adjustments; it is an allocation convention,
not a claim of causal attribution. The result contains asset P&L, asset log returns
and factor attribution, including a separate adjustment column.

## Analytical prediction bands

compute_conditional_scenario_band delegates portfolio risk to RiskModel using a
consistent conditioned factor/asset covariance snapshot and independent residual variances.
With baseline NAV weights w and exposures e = beta.T w:

v = e_F.T C e_F + sum_i w_i^2 * residual_variance_i

The pointwise band is R_p(z) +/- normal_quantile((1+c)/2) * sqrt(T*v).
Scenario centres may be exact expm1 valuations, while the risk overlay remains a
**linearised, additive NAV-return approximation using fixed baseline exposures**.
Width is therefore constant within a fixed-anchor grid. It includes remaining factor
dispersion and idiosyncratic risk, but excludes parameter uncertainty, covariance-regime
changes and non-normal tails. It is not a regression confidence interval and assigns
no probability to the anchor value. No Monte Carlo is used.

compute_factor_sensitivity combines conditioning, valuation and bands for a supplied
simple-return grid. Every named anchor receives the same grid return simultaneously.
Its structured result retains the full covariance, shocks, attribution and band components.
Grids and factor-family selection belong to the consumer.

## Minimal model-independent example

    import pandas as pd
    import qis

    factors = ["Equity", "Rates"]
    assets = ["Asset A", "Asset B"]
    covariance = pd.DataFrame([[0.04, 0.01], [0.01, 0.01]],
                              index=factors, columns=factors)
    betas = pd.DataFrame([[1., 0.], [2., 1.]], index=assets, columns=factors)
    result = qis.compute_factor_sensitivity(
        covariance=covariance, betas=betas,
        residual_variances=pd.Series([0.0025, 0.01], index=assets),
        amounts=pd.Series([60., 40.], index=assets), nav=100.,
        anchors=["Equity"], grid=pd.Index([-.2, 0., .2], name="equity_return"),
        horizon_years=1/12, confidence=.95,
    )
    print(result.band.summary)
