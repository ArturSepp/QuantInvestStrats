"""Computed risk and regression exhibits for detached stress-report results."""

import numpy as np
import pandas as pd

from qis.portfolio.risk.contributions import compute_portfolio_risk_contributions
from qis.utils.regression import fit_ols


def report_diagnostics(model, date, jacobian, denominator, risk, grids, all_funded, confidence):
    """Compute additive Euler exhibits and unit-response risk using canonical QIS analytics.

    The risk table uses the factor-model total so its systematic and independent
    residual terms reconcile. The authoritative supplied-covariance risk remains
    separately available in the result. Family Euler sums never use shock weights.
    """
    dollars = jacobian.sum()
    weights = dollars / denominator
    beta = model.compute_exposures_at_date(weights, date)
    total = risk.annual_factor_model_vol
    systematic = risk.annual_systematic_vol
    residual = risk.annual_residual_vol
    vols = pd.Series([total, systematic, residual], index=["Total", "Systematic", "Idiosyncratic"])
    table = pd.DataFrame(
        {
            "annual_vol": vols,
            "dollar_vol": vols * denominator,
            "variance_share": vols.pow(2) / total**2 if total else vols * 0,
            "euler_vol": vols.pow(2) / total if total else vols * 0,
        }
    )
    factor = compute_portfolio_risk_contributions(beta, model.factor_covar[date])
    factor *= systematic / total if total else 0.0
    families = _family_euler(factor, model.factor_groups)
    # compute_marginal_tre_at_date aggregates over factors; this exhibit retains
    # each holding/factor cell, including offsetting holdings at zero net factor beta.
    holding = (
        (jacobian @ model.factor_loadings[date] / denominator).mul(
            model.factor_covar[date] @ beta, axis=1
        )
        / total
        if total
        else jacobian @ (model.factor_loadings[date] * 0.0)
    )
    unit_risk = {}
    for response in weights.index:
        unit = weights * 0.0
        unit.loc[response] = 1.0
        parts = model.compute_tre_decomposition_at_date(unit * 0, unit, date)
        unit_risk[response] = {
            "Model total vol": parts.tracking_error,
            "Systematic vol": parts.factor_te,
            "Idio vol": parts.residual_te,
        }
    order = dollars.abs().sort_values(ascending=False, kind="stable").index
    aggregates = {}
    for key, ids in (("Rest of assets", order[20:]), ("Portfolio", order)):
        if len(ids) == 0:
            continue
        sleeve = weights.where(weights.index.isin(ids), 0.0)
        parts = model.compute_tre_decomposition_at_date(sleeve * 0, sleeve, date)
        row = model.compute_exposures_at_date(sleeve, date).to_dict()
        row.update(
            {
                "Model total vol": parts.tracking_error,
                "Systematic vol": parts.factor_te,
                "Idio vol": parts.residual_te,
                "response_exposure": dollars.loc[ids].sum(),
            }
        )
        aggregates[key] = row
    reported, holding_reported = {}, {}
    for key, row in families.iterrows():
        group = (model.factor_groups or {}).get(key)
        members = list(group.members) if group is not None else [key]
        display_key = key if len(members) > 1 else members[0]
        label = row["label"] if len(members) > 1 else members[0]
        reported[display_key] = {
            "label": label, "members": ", ".join(members), "member_count": len(members),
            "factor_beta": beta.loc[members].sum(), "euler_vol": row.euler_vol}
        holding_reported[display_key] = holding[members].sum(axis=1)
    regressions, confidence_bands = _grid_regressions(grids, confidence)
    return {
        "Annualised portfolio risk": table,
        "Factor Euler volatility": factor.rename("euler_vol").to_frame(),
        "Family Euler volatility": families,
        "Reported factor groups": pd.DataFrame.from_dict(reported, orient="index"),
        "Holding reported factor Euler volatility": pd.DataFrame(holding_reported),
        "Holding factor Euler volatility": holding,
        "Unit response risk": pd.DataFrame.from_dict(unit_risk, orient="index"),
        "Loading aggregates": pd.DataFrame.from_dict(aggregates, orient="index"),
        "Grid polynomial regressions": regressions,
        "Grid regression confidence bands": confidence_bands,
    }


def _grid_regressions(grids, confidence):
    """Fit through-zero quadratics and pointwise Student-t intervals for their fitted mean."""
    regressions, bands = {}, {}
    band_columns = ["mean", "mean_se", "mean_ci_lower", "mean_ci_upper"]
    for key, summary in grids.items():
        try:
            x = np.asarray(summary.index, dtype=float)
            y = summary.portfolio_return.astype(float)
        except (TypeError, ValueError):
            continue
        if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
            continue
        design = pd.DataFrame({"linear": x, "quadratic": x * x}, index=summary.index)
        if np.linalg.matrix_rank(design) < 2:
            continue
        model = fit_ols(x=x, y=y.to_numpy(), order=2, fit_intercept=False)
        prediction = pd.Series(model.predict(design), index=summary.index)
        params = pd.Series(model.params, index=design.columns)
        # Match no-intercept statsmodels R-squared without warning on a zero response.
        r_squared = (1.0 - (y - prediction).pow(2).sum() / y.pow(2).sum()
                     if np.any(y != 0.0) else np.nan)
        regressions[key] = {
            "linear": params["linear"], "quadratic": params["quadratic"],
            "cubic": 0.0, "order": 2, "r_squared": r_squared,
        }
        if model.df_resid > 0:
            band = model.get_prediction(design).summary_frame(alpha=1.0 - confidence)
            band = band[band_columns].set_axis(summary.index)
        else:
            # Coefficients are identified, but residual variance cannot be estimated.
            band = pd.DataFrame(np.nan, index=summary.index, columns=band_columns)
            band["mean"] = prediction
        band["confidence"] = confidence
        band["df_resid"] = model.df_resid
        bands[key] = band
    coefficients = pd.DataFrame.from_dict(
        regressions, orient="index", columns=["linear", "quadratic", "cubic", "order", "r_squared"]
    )
    confidence_bands = (
        pd.concat(bands, names=["grid", "factor_return"])
        if bands else pd.DataFrame(
            columns=band_columns + ["confidence", "df_resid"],
            index=pd.MultiIndex.from_tuples([], names=["grid", "factor_return"]),
        )
    )
    return coefficients, confidence_bands


def _family_euler(factor, groups):
    """Aggregate signed Euler contributions only over a nonoverlapping partition."""
    families, assigned = {}, set()
    groups = groups or {}
    members = [member for group in groups.values() for member in group.members]
    # Overlapping scenario groups have no unique additive partition: display atomic
    # factors instead, preserving their valid scenario semantics and total risk.
    if len(members) != len(set(members)):
        groups = {}
    for key, group in groups.items():
        assigned.update(group.members)
        families[key] = {
            "label": group.label or key,
            "members": ", ".join(group.members),
            "euler_vol": factor.loc[list(group.members)].sum(),
        }
    for name in factor.index:
        if name not in assigned:
            families[name] = {"label": name, "members": name, "euler_vol": factor[name]}
    return pd.DataFrame.from_dict(families, orient="index")
