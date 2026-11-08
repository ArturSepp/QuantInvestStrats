"""Paired report samples; reuse canonical Sharpe, holdings-beta and HAC estimators."""

import warnings

import numpy as np
import pandas as pd

from qis.perfstats.significance import InterceptInference
from qis.portfolio.reports._holdings_alpha import compute_holdings_alpha
from qis.portfolio.reports._performance_inference import (
    PerformanceInferenceData, _inference_views, _sample_returns, validate_alpha_inference,
)
from qis.portfolio.reports.config import validate_reporting_frequency


def compute_paired_performance_inference(
    strategy, comparison, benchmark_prices, time_period, freq, rates_data=None,
    *, freq_beta="B", factor_beta_span=260, alpha_inference=InterceptInference.BARTLETT_HAC,
):
    """Compare two actual portfolios on paired dates, against the first market reference.

    Args:
        strategy: PortfolioData with NAV, instrument prices and actual holdings.
        comparison: Comparison PortfolioData; not the market factor used for alpha.
        benchmark_prices: Market reference prices; select only the first column before fitting.
        time_period: Realised performance window; pre-window prices can warm instrument betas.
        freq: Paired Sharpe sampling grid, normally perf_params.freq_vol.
        rates_data: Same annual cash-rate series for both portfolios and the market reference.
        freq_beta: Holdings-beta and raw alpha grid, shared by both portfolios.
        factor_beta_span: Shared EWM span and per-instrument warm-up in beta-grid observations.
        alpha_inference: Primary marginal-alpha method; alternate is same-sample sensitivity.

    Returns:
        Paired inference data with marginal unsmoothed alpha and both holdings-beta diagnostics.
        Unknown alpha coverage never becomes zero or a NAV-regression estimate. Positive
        differences favour strategy. No selection or repeated-window adjustment is applied.
    """
    validate_alpha_inference(alpha_inference)
    issues = []
    portfolios = {"Strategy": strategy, "Comparison": comparison}
    navs = pd.concat({key: value.get_portfolio_nav() for key, value in portfolios.items()}, axis=1)
    try:
        for portfolio in portfolios.values():
            validate_reporting_frequency(time_period.locate(portfolio.get_portfolio_nav()), freq)
        sampled = _sample_returns(navs, time_period, freq, rates_data)
    except ValueError as error:
        sampled = pd.DataFrame(columns=portfolios, index=pd.DatetimeIndex([]), dtype=float)
        issues.append(f"Paired Sharpe: sampling grid unavailable ({error}); no substitute fit.")
    end = sampled.index[-1] if len(sampled) else pd.Timestamp(time_period.end or navs.index[-1])
    windows, expanding, rolling, metadata = _inference_views(
        sampled, freq, end, issues, paired=True)

    alpha_grid_valid = True
    try:
        for portfolio in portfolios.values():
            validate_reporting_frequency(
                time_period.locate(portfolio.get_portfolio_nav()), freq_beta)
        alpha_pair = _sample_returns(navs, time_period, freq_beta, rates_data)
    except ValueError as error:
        alpha_grid_valid = False
        alpha_pair = pd.DataFrame(columns=portfolios, index=pd.DatetimeIndex([]), dtype=float)
        issues.append(f"Marginal alpha: sampling grid unavailable ({error}); no substitute fit.")
    alpha_end = alpha_pair.index[-1] if len(alpha_pair) else end
    components = pd.DataFrame(np.nan, index=alpha_pair.index, columns=portfolios)
    diagnostics = pd.DataFrame(index=alpha_pair.index)
    references = benchmark_prices.to_frame() if isinstance(benchmark_prices, pd.Series) else (
        benchmark_prices)
    alpha_meta = dict(benchmark="Unavailable", freq_beta=freq_beta,
                      factor_beta_span=factor_beta_span, portfolios={})
    if references is None or references.empty or not references.shape[1]:
        issues.append("Marginal alpha: no market benchmark; no substitute benchmark selected.")
    else:
        first = references.iloc[:, :1]
        alpha_meta["benchmark"] = str(first.columns[0])
        if alpha_grid_valid:
            market = _sample_returns(first, time_period, freq_beta, rates_data).reindex(
                alpha_pair.index)
            for key, portfolio in portfolios.items():
                if portfolio.prices is None or portfolio.weights is None:
                    issues.append(
                        f"Marginal alpha: {key} holdings unavailable; no NAV-beta fallback.")
                    continue
                try:
                    alpha, diagnostic, beta_meta, beta_issues = compute_holdings_alpha(
                        instrument_prices=portfolio.prices, weights=portfolio.weights,
                        benchmark_prices=first, portfolio_returns=alpha_pair[key],
                        benchmark_returns=market, freq_beta=freq_beta,
                        factor_beta_span=factor_beta_span,
                    )
                    components[key] = alpha
                    diagnostics = diagnostics.join(diagnostic.add_prefix(key + "_"))
                    alpha_meta["portfolios"][key] = beta_meta
                    issues.extend(f"{key}: {message}" for message in beta_issues)
                except ValueError as error:
                    issues.append(f"Marginal alpha: {key} holdings inputs unavailable ({error}); "
                                  "no substitute fit.")
    # Subtraction retains NaN whenever either leg is unknown. Never drop dates before HAC.
    marginal = (components.Strategy - components.Comparison).rename("Marginal alpha")
    diagnostics = diagnostics.join(components.add_suffix("_alpha"))
    aw, ae, ar, am = _inference_views(marginal, freq_beta, alpha_end, issues,
                                     alpha=True, paired=True, alpha_inference=alpha_inference)
    alpha_meta.update(am)
    metadata.update(
        paired=True, strategy=str(strategy.ticker), comparison=str(comparison.ticker),
        alpha=alpha_meta, confidence_level=0.95,
        return_basis="excess of supplied cash" if rates_data is not None else
        "total returns; zero cash assumed", cost_basis="as carried in NAV; no extra deductions",
        variance="Sharpe: unprewhitened Bartlett HAC; alpha: " + alpha_inference.name +
        "; automatic Newey-West lags; paired covariance",
    )
    issues = list(dict.fromkeys(issues))
    if issues:
        warnings.warn("Paired performance inference appendix: " + " ".join(issues),
                      UserWarning, stacklevel=2)
    return PerformanceInferenceData(windows, expanding, rolling, aw, ae, ar,
                                    sampled, marginal, diagnostics, metadata, issues)
