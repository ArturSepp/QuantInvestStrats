"""Scenario-local conditional volatility bands and their additive Euler audit."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from qis.portfolio.risk.contributions import compute_portfolio_risk_contributions
from qis.portfolio.risk.stress_testing import _conditional_risk_model
from qis.portfolio.stress._diagnostics import _family_euler


def conditional_grid_bands(portfolio, shocks, expanded, centres, horizon, confidence):
    """Condition risk once per anchor set and update all local sensitivities at each point."""
    model, date = portfolio.risk_model, portfolio.risk_date
    betas = model.factor_loadings[date]
    local_date = pd.Timestamp(0)
    models, rows, eulers, families, exposures = {}, {}, {}, {}, {}
    scale = np.sqrt(horizon)
    for label, shock in shocks.iterrows():
        anchors = tuple(expanded.loc[label].dropna().index)
        if anchors not in models:
            models[anchors] = _conditional_risk_model(
                model.factor_covar[date], betas, model.residual_vars[date], anchors
            )
        conditional = models[anchors]
        dollars = portfolio.response_jacobian(shock).sum()
        weights = dollars / portfolio.reporting_denominator
        parts = conditional.compute_tre_decomposition_at_date(weights * 0, weights, local_date)
        total, systematic, residual = parts.tracking_error, parts.factor_te, parts.residual_te
        sigma = float(total * scale)
        centre = centres.loc[label]
        width = norm.ppf((1 + confidence) / 2) * sigma
        rows[label] = {
            "portfolio_return": centre,
            "conditional_vol_horizon": sigma,
            "conditional_factor_vol_horizon": systematic * scale,
            "residual_vol_horizon": residual * scale,
            "lower_1sigma": centre - sigma, "upper_1sigma": centre + sigma,
            "lower_2sigma": centre - 2 * sigma, "upper_2sigma": centre + 2 * sigma,
            # Retain the configurable Gaussian-quantile columns for existing table consumers.
            "lower_bound": centre - width, "upper_bound": centre + width,
            "band_half_width": width,
        }
        beta = conditional.compute_exposures_at_date(weights, local_date)
        factor = compute_portfolio_risk_contributions(beta, conditional.factor_covar[local_date])
        factor *= systematic / total * scale if total else 0.0
        idio = residual**2 / total * scale if total else 0.0
        family = _family_euler(factor, model.factor_groups).euler_vol
        eulers[label] = pd.concat([factor, pd.Series({"Residual": idio, "Total": sigma})])
        families[label] = pd.concat([family, pd.Series({"Residual": idio, "Total": sigma})])
        exposures[label] = dollars
    def frame(values):
        """Preserve supplied scenario labels and index naming in each detached audit."""
        return pd.DataFrame.from_dict(values, orient="index").set_axis(shocks.index)
    return frame(rows), {
        "Grid conditional factor Euler": frame(eulers),
        "Grid conditional family Euler": frame(families),
        "Grid scenario response exposures": frame(exposures),
    }
