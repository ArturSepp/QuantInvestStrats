"""Instrument-aware stress orchestration using the canonical QIS RiskModel."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from qis.portfolio.risk.stress_testing import (
    compute_conditional_scenario_band,
    project_factor_scenarios,
)
from qis.portfolio.stress.portfolio import InstrumentPortfolio, PortfolioValuationResult
from qis.portfolio.stress.scenarios import ScenarioMode, StressScenarios


@dataclass(frozen=True)
class StressTestConfig:
    """Numerical analysis settings with explicit covariance horizon.

    Attributes:
        historical_count: Number of worst complete historical months to present.
        horizon_years: Positive risk-band horizon in years.
        confidence: Central Gaussian probability for supported funded-asset bands.
        include_conditional_comparison: Also evaluate joint conditional completion.
        ordinary_asset_bands: Enable existing Gaussian grid bands for funded portfolios.
    """

    historical_count: int = 10
    horizon_years: float = 1.0 / 12.0
    confidence: float = 0.95
    include_conditional_comparison: bool = True
    ordinary_asset_bands: bool = True

    def __post_init__(self):
        """Reject implicit or invalid horizon and ranking assumptions."""
        if (
            not isinstance(self.historical_count, int)
            or isinstance(self.historical_count, bool)
            or self.historical_count < 1
        ):
            raise ValueError("historical_count must be a positive integer")
        if (
            not np.isfinite(self.horizon_years)
            or self.horizon_years <= 0
            or not np.isfinite(self.confidence)
            or not 0 < self.confidence < 1
        ):
            raise ValueError("positive horizon_years and confidence in (0, 1) required")


@dataclass(frozen=True)
class PortfolioStressResult:
    """Computed report inputs, detached from live portfolio/model objects.

    Every value, quote and covariance needed for the report is copied at analysis
    time. The result contains no executable payoff or estimator object.

    Attributes:
        valuations: Requested, independent and optional conditional scenario evaluations.
        summaries: Value/P&L and denominator returns for each valuation batch.
        attribution: Reconciled factor contributions and nonlinear adjustment by batch.
        grids: Caller-named evaluated sensitivity grids.
        grid_summaries: Curve returns and supported ordinary-asset bands.
        grid_metadata: Bump units, group/member allocations and band coverage.
        historical: All eligible complete monthly vectors revalued on current holdings.
        historical_ranking: Worst months ranked by exact current-portfolio P&L.
        historical_coverage: Eligibility status for every supplied historical row.
        response_jacobian: Original holding by shared response dollar sensitivities.
        response_exposures: Aggregated shared response dollar sensitivities.
        factor_exposures: Current portfolio factor dollar sensitivities.
        factor_betas: Factor exposures divided by the explicit reporting denominator.
        holding_factor_exposures: Current holding-by-factor dollar sensitivities.
        factor_group_exposures: Family exposure sums and allocated-bump sensitivities.
        risk: Current annual total, systematic and residual local volatility.
        response_risk_contributions: Canonical Euler contributions by shared response.
        holding_risk: Standalone holding local risk, expressed against the denominator.
        factor_loadings: Copied underlying-response estimation betas.
        factor_covariance: Copied annual factor covariance.
        residual_variances: Copied annual shared response residual variances.
        positions: Observed marks, model anchors and declared payoff coverage.
        leg_terms: Auditable vanilla decomposition under original holding identities.
        metadata: Snapshot dates, currencies, denominator and analysis policies.
        report_diagnostics: Detached Euler, unit-response risk and quadratic-fit exhibits.
    """

    valuations: Mapping[str, PortfolioValuationResult]
    summaries: Mapping[str, pd.DataFrame]
    attribution: Mapping[str, pd.DataFrame]
    grids: Mapping[str, PortfolioValuationResult]
    grid_summaries: Mapping[str, pd.DataFrame]
    grid_metadata: pd.DataFrame
    historical: PortfolioValuationResult | None
    historical_ranking: pd.DataFrame
    historical_coverage: pd.DataFrame
    response_jacobian: pd.DataFrame
    response_exposures: pd.Series
    factor_exposures: pd.Series
    factor_betas: pd.Series
    holding_factor_exposures: pd.DataFrame
    factor_group_exposures: pd.DataFrame
    risk: pd.Series
    response_risk_contributions: pd.DataFrame
    holding_risk: pd.DataFrame
    factor_loadings: pd.DataFrame
    factor_covariance: pd.DataFrame
    residual_variances: pd.Series
    positions: pd.DataFrame
    leg_terms: pd.DataFrame
    metadata: Mapping[str, object] = field(default_factory=dict)
    report_diagnostics: Mapping[str, pd.DataFrame] = field(default_factory=dict)


def _summary(valuation: PortfolioValuationResult, denominator: float) -> pd.DataFrame:
    """Build denominator-labelled value changes without interpreting them as NAV history."""
    return pd.DataFrame(
        {
            "portfolio_mtm": valuation.portfolio_mtm,
            "portfolio_pnl": valuation.portfolio_pnl,
            "portfolio_return": valuation.portfolio_pnl / denominator,
        }
    )


def _attribution(
    portfolio: InstrumentPortfolio, valuation: PortfolioValuationResult, jacobian: pd.DataFrame
) -> pd.DataFrame:
    """Preserve funded attribution and expose nonlinear payoff change separately."""
    betas = portfolio.risk_model.factor_loadings[portfolio.risk_date]
    shocks = valuation.factor_log_shocks
    reserved = "Nonlinear payoff adjustment"
    if reserved in betas.columns or "Anchor / residual adjustment" in betas.columns:
        raise ValueError("factor name conflicts with reserved attribution column")
    contribution = pd.DataFrame(0.0, index=shocks.index, columns=betas.columns)
    for holding in portfolio.holdings:
        response = jacobian.loc[holding.holding_id]
        if holding.is_delta_one and holding.observed_mtm != 0:
            holding_betas = pd.DataFrame(
                [
                    portfolio.risk_model.compute_exposures_at_date(
                        response / holding.observed_mtm, portfolio.risk_date
                    )
                ],
                index=[holding.holding_id],
            )
            projection = project_factor_scenarios(
                holding_betas, pd.Series([holding.observed_mtm], index=holding_betas.index), shocks
            )
            contribution += projection.factor_attribution.loc[:, betas.columns]
        else:
            factor_dollars = portfolio.risk_model.compute_exposures_at_date(
                response, portfolio.risk_date
            )
            contribution += shocks.mul(factor_dollars, axis=1)
    contribution[reserved] = valuation.portfolio_pnl - contribution.sum(axis=1)
    return contribution


def _history(portfolio: InstrumentPortfolio, history: pd.DataFrame | None, count: int):
    """Validate monthly vectors, record omissions and rank every eligible revaluation."""
    if history is None:
        return None, pd.DataFrame(), pd.DataFrame(columns=["status"])
    factors = portfolio.risk_model.factor_loadings[portfolio.risk_date].columns
    if (
        not isinstance(history.index, pd.DatetimeIndex)
        or history.index.has_duplicates
        or history.index.hasnans
        or history.columns.has_duplicates
        or set(history.columns) != set(factors)
    ):
        raise ValueError("history needs unique monthly dates and exactly the fitted factors")
    if history.index.tz is not None:
        raise ValueError("monthly scenario labels must use timezone-naive dates")
    if history.index.to_period("M").has_duplicates:
        raise ValueError("historical input must contain at most one realization per month")
    history = history.reindex(columns=factors).sort_index().astype(float).copy()
    if np.isinf(history.to_numpy()).any():
        raise ValueError("historical returns must not contain infinite values")
    coverage = pd.DataFrame("eligible", index=history.index, columns=["status"])
    coverage.loc[history.isna().any(axis=1), "status"] = "incomplete factor vector"
    coverage.loc[history.index > portfolio.valuation_date, "status"] = "after valuation date"
    eligible = history.loc[coverage.status == "eligible"]
    if eligible.empty:
        return None, pd.DataFrame(), coverage
    valuation = portfolio.evaluate(eligible)
    ranking = (
        _summary(valuation, portfolio.reporting_denominator)
        .sort_values("portfolio_pnl", kind="stable")
        .head(count)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return valuation, ranking, coverage


def _grid_metadata(request, expanded, key, band_status):
    """Describe the displayed grid units independently from expanded member shocks."""
    return {
        "grid": key,
        "bump_convention": request.convention.value,
        "completion": "row-specific" if request.scenario_modes else request.mode.value,
        "requested_keys": ", ".join(map(str, request.anchors.columns)),
        "expanded_factors": ", ".join(map(str, expanded.columns[expanded.notna().any()])),
        "band_status": band_status,
        "axis": "supplied scenario index; family simple bumps split before log1p",
    }


def run_portfolio_stress_test(
    portfolio: InstrumentPortfolio,
    scenarios: StressScenarios,
    historical_factor_log_returns: pd.DataFrame | None = None,
    factor_grids: Mapping[str, StressScenarios] | None = None,
    config: StressTestConfig | None = None,
) -> PortfolioStressResult:
    """Revalue requested, conditional, historical and grid shocks through one portfolio.

    No estimation, provider access, plotting or file output occurs here. Current
    risk is a local derivative calculation; intrinsic stress P&L uses full payoffs.

    Args:
        portfolio: Validated absolute-position snapshot referencing a QIS RiskModel.
        scenarios: Requested factor/family anchors with explicit units and completion.
        historical_factor_log_returns: Optional complete-factor monthly log-return panel.
            Incomplete rows and rows after the valuation date are reported and excluded.
        factor_grids: Caller-named grid requests; row labels are retained as the displayed axis.
        config: Optional explicit risk horizon, ranking and band policies.

    Returns:
        Detached numerical result usable by reporting, workbooks or a consumer dashboard.
    """
    config = config or StressTestConfig()
    model, date = portfolio.risk_model, portfolio.risk_date
    betas = model.factor_loadings[date]
    denominator = portfolio.reporting_denominator
    jacobian = portfolio.response_jacobian()
    exposures = jacobian.sum(axis=0).rename("response_exposure")
    weights = exposures / denominator
    zero = weights * 0.0
    factors = model.compute_exposures_at_date(exposures, date).rename("factor_exposure")
    decomp = model.compute_tre_decomposition_at_date(zero, weights, date)
    risk = pd.Series(
        {
            "annual_total_vol": model.compute_tre_at_date(zero, weights, date),
            "annual_systematic_vol": decomp.factor_te,
            "annual_residual_vol": decomp.residual_te,
            "annual_factor_model_vol": decomp.tracking_error,
            "local_vol_horizon": model.compute_tre_at_date(zero, weights, date)
            * np.sqrt(config.horizon_years),
        }
    )
    holding_risk = {}
    for name, response in jacobian.iterrows():
        w = response / denominator
        parts = model.compute_tre_decomposition_at_date(zero, w, date)
        holding_risk[name] = {
            "annual_total_vol": model.compute_tre_at_date(zero, w, date),
            "annual_systematic_vol": parts.factor_te,
            "annual_residual_vol": parts.residual_te,
        }
    requested_shocks = scenarios.resolve(model, date)
    valuations = {"requested": portfolio.evaluate(requested_shocks)}
    independent = scenarios.resolve(model, date, ScenarioMode.INDEPENDENT)
    valuations["independent"] = portfolio.evaluate(independent)
    if config.include_conditional_comparison:
        valuations["conditional"] = portfolio.evaluate(
            scenarios.resolve(model, date, ScenarioMode.CONDITIONAL)
        )
    summaries = {key: _summary(value, denominator) for key, value in valuations.items()}
    attribution = {
        key: _attribution(portfolio, value, jacobian) for key, value in valuations.items()
    }
    grids, grid_summaries, grid_metadata = {}, {}, []
    all_funded = all(h.is_delta_one for h in portfolio.holdings)
    for key, request in (factor_grids or {}).items():
        if not isinstance(key, str) or not key:
            raise ValueError("grid names must be nonempty strings")
        expanded = request.expanded_anchors(model, date)
        value = portfolio.evaluate(request.resolve(model, date))
        grids[key] = value
        summary = _summary(value, denominator)
        band_status = "disabled"
        if not all_funded:
            band_status = "unavailable: nonlinear/derivative payoff; deterministic curve only"
        elif config.ordinary_asset_bands and all(
            request.scenario_modes.get(label, request.mode) is ScenarioMode.CONDITIONAL
            for label in expanded.index
        ):
            rows = []
            for label, row in expanded.iterrows():
                band = compute_conditional_scenario_band(
                    model.factor_covar[date],
                    betas,
                    model.residual_vars[date],
                    weights,
                    row.dropna().index.tolist(),
                    summary.loc[[label], "portfolio_return"],
                    config.horizon_years,
                    config.confidence,
                )
                rows.append(band.summary)
            summary = summary.drop(columns="portfolio_return").join(pd.concat(rows))
            band_status = "baseline Gaussian conditional factor + shared residual"
        elif config.ordinary_asset_bands:
            band_status = "unavailable: grid does not request conditional completion"
        grid_summaries[key] = summary
        grid_metadata.append(_grid_metadata(request, expanded, key, band_status))
    historical, ranking, historical_coverage = _history(
        portfolio, historical_factor_log_returns, config.historical_count
    )
    terms = []
    for holding in portfolio.holdings:
        for i, leg in enumerate(holding.legs):
            quote = portfolio.underlyings[leg.underlying_id]
            terms.append(
                {
                    "holding_id": holding.holding_id,
                    "leg": i + 1,
                    "type": leg.instrument_type.value,
                    "underlying_id": leg.underlying_id,
                    "response_id": quote.response_id,
                    "response_basis": quote.response_basis.value,
                    "spot0": quote.spot0,
                    "currency": quote.currency,
                    "quantity": leg.quantity,
                    "multiplier": leg.multiplier,
                    "strike": leg.strike,
                }
            )
    metadata = {
        "all_funded": all_funded,
        "valuation_date": str(portfolio.valuation_date),
        "risk_date": str(date),
        "reference_currency": portfolio.reference_currency,
        "reporting_denominator": denominator,
        "denominator_label": portfolio.denominator_label,
        "horizon_years": config.horizon_years,
        "confidence": config.confidence,
        "historical_count": config.historical_count,
        "risk_convention": "annualized log-return covariance; current local payoff Jacobian",
        "attribution": (
            "exact funded attribution; local derivative plus nonlinear payoff adjustment"
        ),
        "scenario_descriptions": {str(k): v for k, v in scenarios.descriptions.items()},
        "scenario_completion_overrides": {
            str(k): v.value for k, v in scenarios.scenario_modes.items()
        },
        "requested_convention": scenarios.convention.value,
        "requested_completion": scenarios.mode.value,
        "factor_groups": {
            key: {
                "members": list(group.members),
                "weights": list(group.weights),
                "label": group.label,
            }
            for key, group in (model.factor_groups or {}).items()
        },
        "fx_quotes": {
            currency: {"spot0": q.spot0, "response_id": q.response_id}
            for currency, q in portfolio.fx_rates.items()
        },
    }
    from qis.portfolio.stress._diagnostics import report_diagnostics

    diagnostics = report_diagnostics(
        model, date, jacobian, denominator, risk, grid_summaries, all_funded
    )
    return PortfolioStressResult(
        MappingProxyType(valuations),
        MappingProxyType(summaries),
        MappingProxyType(attribution),
        MappingProxyType(grids),
        MappingProxyType(grid_summaries),
        pd.DataFrame(grid_metadata).set_index("grid") if grid_metadata else pd.DataFrame(),
        historical,
        ranking,
        historical_coverage,
        jacobian,
        exposures,
        factors,
        (factors / denominator).rename("factor_beta"),
        jacobian @ betas,
        model.compute_factor_group_exposures_at_date(exposures, date),
        risk,
        model.compute_marginal_tre_at_date(zero, weights, date),
        pd.DataFrame.from_dict(holding_risk, orient="index"),
        betas.copy(deep=True),
        model.factor_covar[date].copy(deep=True),
        model.residual_vars[date].copy(deep=True),
        valuations["requested"].audit.copy(deep=True),
        pd.DataFrame(terms),
        MappingProxyType(metadata),
        MappingProxyType(diagnostics),
    )
