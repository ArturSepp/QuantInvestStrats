"""Provider-neutral factor stress valuation and analytical conditional prediction bands.

Factor shocks and loadings use log returns; positions use a single reference currency.
Covariance and residual variances are annualised. No estimation, interpolation, carry,
Monte Carlo or plotting occurs here. See qis/docs/stress_testing.md.
"""
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from qis.portfolio.risk.risk_model import RiskModel


@dataclass(frozen=True)
class FactorScenarioProjection:
    """Exact asset valuation and additive attribution of a log-factor scenario.

    Attributes:
        asset_pnl: Scenarios by assets, in position currency.
        factor_attribution: Scenarios by factors plus the explicit adjustment component.
        asset_log_returns: Scenarios by assets before expm1 valuation.
    """
    asset_pnl: pd.DataFrame
    factor_attribution: pd.DataFrame
    asset_log_returns: pd.DataFrame


@dataclass(frozen=True)
class ConditionalScenarioBand:
    """Pointwise Gaussian prediction band with fixed baseline portfolio exposures.

    Attributes:
        summary: Centre, bounds, horizon volatilities and half-width in decimal NAV returns.
        conditional_factor_covariance: Annual full-order covariance, zero at fixed anchors.
        annual_idio_vol: Portfolio residual volatility under independent asset residuals.
        annual_factor_vol: Remaining conditional factor volatility.
        annual_total_vol: Combined conditional annual portfolio volatility.
        horizon_years: Positive covariance scaling horizon.
        confidence: Central probability in (0, 1).
    """
    summary: pd.DataFrame
    conditional_factor_covariance: pd.DataFrame
    annual_idio_vol: float
    annual_factor_vol: float
    annual_total_vol: float
    horizon_years: float
    confidence: float


@dataclass(frozen=True)
class FactorSensitivityResult:
    """Joint-anchor grid, exact valuation and analytical band.

    Attributes:
        anchors: Factors fixed to the same simple return at each grid point.
        factor_log_shocks: Complete correlated log-shock vectors.
        projection: Exact per-asset P&L and reconciled factor attribution.
        band: Conditional factor plus residual prediction band.
    """
    anchors: tuple
    factor_log_shocks: pd.DataFrame
    projection: FactorScenarioProjection
    band: ConditionalScenarioBand


def _finite(values, role):
    """Reject NaN and infinite numerical inputs."""
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError(f"{role} must be finite")


def _covariance(covariance):
    """Validate labelled positive-semidefinite covariance without altering it."""
    if (not isinstance(covariance, pd.DataFrame) or covariance.empty
            or not covariance.index.is_unique or covariance.index.hasnans
            or not covariance.columns.equals(covariance.index)):
        raise ValueError("Covariance requires identical unique row/column factor labels")
    _finite(covariance, "Covariance")
    values = covariance.to_numpy()
    if (not np.allclose(values, values.T, atol=1e-12, rtol=0)
            or np.linalg.eigvalsh(values).min() < -1e-10):
        raise ValueError("Covariance must be symmetric positive semidefinite")


def _anchors(covariance, anchors):
    """Validate explicit, nonempty anchor identities and a stable anchor block."""
    labels = list(anchors)
    if (not labels or len(set(labels)) != len(labels)
            or not set(labels).issubset(covariance.index)):
        raise ValueError("Conditional covariance requires unique known anchors")
    block = covariance.loc[labels, labels]
    if np.linalg.cond(block) > 1e8:
        raise ValueError("Unstable anchor solve")
    return labels, block


def price_target_log_shock(current: float, target: float) -> float:
    """Convert positive price endpoints to a log-return shock.

    Args:
        current: Positive current price.
        target: Positive target price in the same units.
    """
    _finite([current, target], "Price anchors")
    if current <= 0 or target <= 0:
        raise ValueError("Price anchors must be positive")
    result = float(np.log(target / current))
    _finite(result, "Price log shock")
    return result


def return_log_shock(market_return: float, effective_weight: float = 1.0) -> float:
    """Convert a simple market return with fixed exposure to a factor log return.

    Args:
        market_return: Decimal simple return strictly greater than -1.
        effective_weight: Fixed exposure; the uninvested portion has zero instantaneous carry.
    """
    _finite([market_return, effective_weight], "Return anchor")
    value = effective_weight * market_return
    if market_return <= -1 or value <= -1:
        raise ValueError("Nonpositive underlying or factor endpoint")
    result = float(np.log1p(value))
    _finite(result, "Return log shock")
    return result


def duration_log_shock(delta_y: float, duration: float) -> float:
    """Convert a yield change using the first-order duration approximation.

    Args:
        delta_y: Decimal yield change, e.g. +0.005 for +50 basis points.
        duration: Nonnegative effective duration; no asset-specific default.
    """
    _finite([delta_y, duration], "Duration anchor")
    if duration < 0:
        raise ValueError("Duration must be nonnegative")
    value = -duration * delta_y
    if value <= -1:
        raise ValueError("Duration approximation implies nonpositive factor value")
    result = float(np.log1p(value))
    _finite(result, "Duration log shock")
    return result


def conditional_factor_shock(
        covariance: pd.DataFrame, anchors: Mapping[str, float]) -> pd.Series:
    """Condition free log-factor shocks on explicit anchors under a zero-mean model.

    Args:
        covariance: Labelled factor covariance; any consistent covariance frequency.
        anchors: Named finite log-return anchors, including explicitly fixed zero values.

    Returns:
        Full factor vector in covariance order. All anchors are preserved.
    """
    _covariance(covariance)
    labels, block = _anchors(covariance, anchors)
    values = np.asarray([anchors[name] for name in labels], dtype=float)
    _finite(values, "Anchors")
    solved = np.linalg.solve(block.to_numpy(), values)
    result = covariance.loc[:, labels] @ solved
    if not np.allclose(result.loc[labels], values, atol=1e-12, rtol=1e-10):
        raise ValueError("Conditional solve failed to preserve anchors")
    return result


def conditional_factor_covariance(
        covariance: pd.DataFrame, anchors: Sequence[str]) -> pd.DataFrame:
    """Compute a Schur complement, retaining full factor order and zero anchor rows.

    Args:
        covariance: Symmetric positive-semidefinite factor covariance.
        anchors: Unique factors being fixed, including anchors whose shock is zero.

    Returns:
        Free block Sigma_FF - Sigma_FA solve(Sigma_AA, Sigma_AF), with fixed-factor
        rows and columns zero. Units match the input. Gaussian conditional covariance
        is independent of anchor values. Ill-conditioned anchor blocks are rejected.
    """
    _covariance(covariance)
    labels, block = _anchors(covariance, anchors)
    free = [name for name in covariance.index if name not in labels]
    result = pd.DataFrame(0., index=covariance.index, columns=covariance.columns)
    if free:
        values = (covariance.loc[free, free].to_numpy()
                  - covariance.loc[free, labels].to_numpy()
                  @ np.linalg.solve(block.to_numpy(), covariance.loc[labels, free].to_numpy()))
        values = .5 * (values + values.T)
        if np.linalg.eigvalsh(values).min() < -1e-10:
            raise ValueError("Conditional covariance is not positive semidefinite")
        result.loc[free, free] = values
    return result


def _positions(betas, amounts):
    """Validate strict asset alignment and finite model exposures."""
    if (betas.empty or not betas.index.is_unique or not betas.columns.is_unique
            or betas.index.hasnans or betas.columns.hasnans):
        raise ValueError("Loadings require unique asset and factor labels")
    if not betas.index.equals(amounts.index):
        raise ValueError("Amount labels/order differ")
    _finite(betas, "Factor loadings")
    _finite(amounts, "Amounts")


def project_factor_scenarios(
        betas: pd.DataFrame, amounts: pd.Series, factor_log_shocks: pd.DataFrame,
        adjustments: Optional[pd.DataFrame] = None) -> FactorScenarioProjection:
    """Value each asset through expm1 of its frozen log-factor exposure, then aggregate.

    Args:
        betas: Assets by factors, frozen log-return loadings.
        amounts: Signed positions in a common reference currency, exactly aligned to assets.
        factor_log_shocks: Scenarios by factors, exactly aligned to the loading columns.
        adjustments: Optional additional log returns, scenarios by assets in exact order.

    Returns:
        Currency P&L, reconciled factor attribution and asset log returns. Attribution
        allocates exp(g)-1 proportionally to additive log components, with the continuous
        limit at zero. It is a convention for allocating model P&L, not causal attribution.
    """
    _positions(betas, amounts)
    if not betas.columns.equals(factor_log_shocks.columns):
        raise ValueError("Factor labels/order differ")
    if (factor_log_shocks.empty or not factor_log_shocks.index.is_unique
            or factor_log_shocks.index.hasnans):
        raise ValueError("Scenarios require unique nonempty labels")
    if "Anchor / residual adjustment" in betas.columns:
        raise ValueError("Factor name is reserved for the adjustment attribution component")
    _finite(factor_log_shocks, "Log shocks")
    components = factor_log_shocks.to_numpy()[:, None, :] * betas.to_numpy()[None, :, :]
    extra = pd.DataFrame(0., index=factor_log_shocks.index, columns=betas.index)
    if adjustments is not None:
        if (not adjustments.index.equals(extra.index)
                or not adjustments.columns.equals(extra.columns)):
            raise ValueError("Adjustment labels/order differ")
        _finite(adjustments, "Adjustments")
        extra = adjustments
    logs = components.sum(axis=2) + extra.to_numpy()
    _finite(logs, "Log shock")
    with np.errstate(over="ignore"):
        simple = np.expm1(logs)
    _finite(simple, "Projected return")
    pnl = pd.DataFrame(simple * amounts.to_numpy(), index=extra.index, columns=extra.columns)
    scale = np.divide(simple, logs, out=np.ones_like(logs), where=np.abs(logs) > 1e-14)
    contribution = components * scale[:, :, None] * amounts.to_numpy()[None, :, None]
    attribution = pd.DataFrame(contribution.sum(axis=1), index=extra.index,
                               columns=betas.columns)
    attribution["Anchor / residual adjustment"] = (
        extra.to_numpy() * scale * amounts.to_numpy()).sum(axis=1)
    _finite(pnl, "Projected P&L")
    if not np.allclose(attribution.sum(axis=1), pnl.sum(axis=1), atol=1e-7, rtol=1e-12):
        raise ValueError("Scenario attribution does not reconcile")
    return FactorScenarioProjection(
        pnl, attribution, pd.DataFrame(logs, index=extra.index, columns=extra.columns))


def compute_conditional_scenario_band(
        covariance: pd.DataFrame, betas: pd.DataFrame, residual_variances: pd.Series,
        weights: pd.Series, anchors: Sequence[str], centres: pd.Series,
        horizon_years: float, confidence: float = .95) -> ConditionalScenarioBand:
    """Compute analytical conditional-factor plus independent-residual prediction bounds.

    Args:
        covariance: Annualised factor log-return covariance.
        betas: Assets by factors, exactly matching the covariance order.
        residual_variances: Nonnegative annual asset residual variances in asset order.
        weights: Signed baseline MTM/NAV weights in asset order, without renormalisation.
        anchors: Fixed factors; all remaining factor dispersion enters the band.
        centres: Finite scenario NAV returns indexed by scenario.
        horizon_years: Positive horizon in years for covariance scaling.
        confidence: Central Gaussian probability strictly between zero and one.

    Returns:
        Pointwise additive bounds centred on supplied scenarios. Width uses baseline
        exposures and the same covariance at all grid points. It excludes parameter,
        covariance-regime and tail uncertainty and is not a regression confidence interval.
    """
    _positions(betas, weights)
    if not betas.columns.equals(covariance.index):
        raise ValueError("Factor labels/order differ")
    if not residual_variances.index.equals(betas.index):
        raise ValueError("Residual variance labels/order differ")
    _finite(residual_variances, "Residual variances")
    if (residual_variances < 0).any():
        raise ValueError("Residual variances must be nonnegative")
    if (not np.isfinite(horizon_years) or horizon_years <= 0
            or not np.isfinite(confidence) or not 0 < confidence < 1):
        raise ValueError("Band requires a positive horizon and confidence between zero and one")
    if centres.empty or not centres.index.is_unique or centres.index.hasnans:
        raise ValueError("Scenario centres require unique nonempty labels")
    _finite(centres, "Scenario centres")
    conditional = conditional_factor_covariance(covariance, anchors)
    # Build a consistent static asset covariance; portfolio risk arithmetic stays in RiskModel.
    asset_covariance = betas @ conditional @ betas.T
    asset_covariance += pd.DataFrame(np.diag(residual_variances),
                                    index=betas.index, columns=betas.index)
    date = pd.Timestamp(0)  # Internal static snapshot key; no date selection or annualisation.
    model = RiskModel(covar={date: asset_covariance}, factor_loadings={date: betas},
                      factor_covar={date: conditional}, residual_vars={date: residual_variances})
    risk = model.compute_tre_decomposition_at_date(weights * 0, weights, date)
    residual_vol = float(risk.residual_te * np.sqrt(horizon_years))
    conditional_vol = float(risk.tracking_error * np.sqrt(horizon_years))
    half_width = float(norm.ppf((1 + confidence) / 2) * conditional_vol)
    summary = pd.DataFrame({
        "portfolio_return": centres, "lower_bound": centres - half_width,
        "upper_bound": centres + half_width, "residual_vol_horizon": residual_vol,
        "band_half_width": half_width, "conditional_vol_horizon": conditional_vol,
        "conditional_factor_vol_horizon": float(risk.factor_te * np.sqrt(horizon_years)),
    }, index=centres.index)
    return ConditionalScenarioBand(summary, conditional, float(risk.residual_te),
                                   float(risk.factor_te), float(risk.tracking_error),
                                   horizon_years, confidence)


def compute_factor_sensitivity(
        covariance: pd.DataFrame, betas: pd.DataFrame, residual_variances: pd.Series,
        amounts: pd.Series, nav: float, anchors: Sequence[str], grid: pd.Index,
        horizon_years: float, confidence: float = .95) -> FactorSensitivityResult:
    """Run an explicit simple-return grid through joint conditioning, valuation and bands.

    Args:
        covariance: Annual factor log-return covariance.
        betas: Asset log-return loadings in covariance factor order.
        residual_variances: Annual independent residual variances in asset order.
        amounts: Signed reference-currency MTM in asset order.
        nav: Explicit positive finite NAV; never inferred from net or gross positions.
        anchors: Factors receiving the same simple-return grid value simultaneously.
        grid: Nonempty unique simple returns greater than -1; index name is retained.
        horizon_years: Positive prediction-band horizon in years.
        confidence: Central Gaussian probability in (0, 1).

    Returns:
        Labelled shock vectors, exact asset valuation and the analytical prediction band.
        This function does not assign probabilities to anchor values.
    """
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("NAV must be positive and finite")
    if not isinstance(grid, pd.Index) or grid.empty or not grid.is_unique:
        raise ValueError("Grid must be a nonempty unique pandas Index")
    _finite(grid, "Grid")
    if (grid.to_numpy() <= -1).any():
        raise ValueError("Grid returns must be greater than -1")
    _covariance(covariance)
    labels, _ = _anchors(covariance, anchors)
    shocks = pd.DataFrame([
        conditional_factor_shock(covariance, {a: float(np.log1p(x)) for a in labels})
        for x in grid], index=grid, columns=covariance.index)
    projection = project_factor_scenarios(betas, amounts, shocks)
    centres = projection.asset_pnl.sum(axis=1) / nav
    band = compute_conditional_scenario_band(covariance, betas, residual_variances,
                                            amounts / nav, labels, centres,
                                            horizon_years, confidence)
    return FactorSensitivityResult(tuple(labels), shocks, projection, band)
