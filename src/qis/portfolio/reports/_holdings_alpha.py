"""Report-only holdings alpha; strict by default, with explicit coverage assistance."""

import numpy as np
import pandas as pd

from qis.models.linear.ewm import InitType, MeanAdjType
from qis.perfstats.returns import prices_at_freq, to_returns
from qis.portfolio.reports.alpha_coverage import validate_alpha_coverage
from qis.portfolio.risk.ewm_factor_model import EwmLinearModel
from qis.portfolio.risk.factor_model import compute_benchmarks_beta_attribution_from_returns


def _strict_log_returns(prices, freq):
    """Sample completed boundaries without turning an empty native bin into a stale price."""
    levels = prices_at_freq(prices, freq=freq, ffill_nans=False)
    observed = prices.resample(freq).count().reindex(levels.index)
    levels = levels.where(observed.gt(0))
    return to_returns(levels, is_log_returns=True, ffill_nans=False)


def _eligible_loadings(asset_returns, factor_returns, name, span):
    """Keep the canonical estimator and per-instrument eligibility identical for proxies."""
    paired = np.isfinite(asset_returns).mul(np.isfinite(factor_returns).all(axis=1), axis=0)
    model = EwmLinearModel(x=factor_returns, y=asset_returns)
    model.fit(span=span, mean_adj_type=MeanAdjType.EWMA, init_type=InitType.X0,
              warmup_period=span)
    return model.loadings[name].where(paired.cumsum().gt(span) & paired)


def compute_holdings_alpha(
    instrument_prices,
    weights,
    benchmark_prices,
    portfolio_returns,
    benchmark_returns,
    freq_beta,
    factor_beta_span,
    *, alpha_coverage=None,
):
    """Attribute unsmoothed simple/excess returns to fully covered, lagged holdings betas.

    Args:
        instrument_prices: Historical instrument levels, including pre-measurement history.
        weights: Actual NAV exposures in force on each date, not target portfolio allocations.
            Off-grid rows are selected as of the beta date; explicit missing weights stay unknown.
        benchmark_prices: Exactly one reference-level column, selected before fitting betas.
        portfolio_returns: Report-period simple returns, cash adjusted if desired by the caller.
        benchmark_returns: Same-index one-column reference returns on the identical cash basis.
        freq_beta: Common estimation and attribution grid. No frequency conversion of betas.
        factor_beta_span: EWM span and per-instrument paired-return warm-up count.
        alpha_coverage: Optional HoldingsAlphaCoverage; None retains strict coverage.

    Returns:
        Alpha Series, same-index diagnostics, metadata and human-readable issue strings. No
        unresolved material coverage is published. Strict alpha is retained in diagnostics.
        The canonical model estimates log-return betas;
        using these betas on simple returns is the existing second-order approximation.
        This is a realised exposure-adjusted payoff, not a fitted NAV-regression intercept.
    """
    validate_alpha_coverage(alpha_coverage, weights.columns)
    if benchmark_prices.shape[1] != 1 or benchmark_returns.shape[1] != 1:
        raise ValueError("holdings alpha requires exactly one benchmark")
    if not portfolio_returns.index.equals(benchmark_returns.index):
        raise ValueError("portfolio and benchmark return dates must match exactly")
    if not isinstance(factor_beta_span, (int, np.integer)) or factor_beta_span < 1:
        raise ValueError("factor_beta_span must be a positive integer")
    for name, frame in (
        ("instrument_prices", instrument_prices), ("weights", weights),
        ("benchmark_prices", benchmark_prices), ("portfolio_returns", portfolio_returns),
    ):
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"{name} requires a DatetimeIndex")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{name} dates must be unique and increasing")
    if not instrument_prices.columns.is_unique or not weights.columns.is_unique:
        raise ValueError("instrument and weight columns must be unique")
    if instrument_prices.empty or weights.empty or benchmark_prices.empty:
        raise ValueError("holdings alpha requires instrument prices, weights and benchmark history")

    name = benchmark_prices.columns[0]
    metadata = dict(
        benchmark=str(name), freq_beta=freq_beta, factor_beta_span=int(factor_beta_span),
        estimation_start=pd.NaT, mean_adjustment="EWMA / X0", beta_warmup=factor_beta_span,
        beta_return_basis="log returns; simple-return application is an approximation",
        beta_coverage="all held instruments; actual exposures; no renormalisation",
        timing="opening-date holdings beta, applied one beta-grid period later",
        coverage_enabled=alpha_coverage is not None, proxy_assisted=False,
        proxy_map=dict(alpha_coverage.proxy_map) if alpha_coverage is not None else {},
        weight_tolerance=alpha_coverage.weight_tolerance if alpha_coverage is not None else 0.0,
        total_weight_tolerance=(alpha_coverage.total_weight_tolerance
                                if alpha_coverage is not None else 0.0),
    )
    if portfolio_returns.empty:
        diagnostics = pd.DataFrame(
            index=portfolio_returns.index, columns=["lagged_beta", "complete_coverage"]
        )
        return portfolio_returns.rename("Alpha"), diagnostics, metadata, ["Alpha: no returns."]

    # Clip future inputs, not pre-report history. The opening beta must precede the first
    # reported realised return; shifting after cropping would incorrectly discard that return.
    end = portfolio_returns.index[-1]
    asset_returns = _strict_log_returns(instrument_prices.loc[:end], freq_beta)
    factor_returns = _strict_log_returns(benchmark_prices.loc[:end], freq_beta)
    start = min(instrument_prices.index[0], benchmark_prices.index[0])
    grid = pd.date_range(start, end, freq=freq_beta)
    if not portfolio_returns.index.isin(grid).all():
        raise ValueError("report returns must lie on the beta-estimation grid")
    asset_returns = asset_returns.reindex(grid)
    factor_returns = factor_returns.reindex(grid)
    paired = np.isfinite(asset_returns).mul(np.isfinite(factor_returns).all(axis=1), axis=0)
    paired_dates = paired.any(axis=1)
    if paired_dates.any():
        metadata["estimation_start"] = grid[paired_dates][0]
    # The estimator carries its state across missing rows. Such rows are not observations
    # for warm-up and cannot publish a fresh fully covered beta in this inference appendix.
    loading = _eligible_loadings(asset_returns, factor_returns, name, factor_beta_span)
    aligned_weights = weights.reindex(grid, method="ffill")
    # Price columns without a weight column are not in the portfolio. Conversely a nonzero
    # weight without instrument history is unknown beta, never an implicit zero contribution.
    loading = loading.reindex(columns=weights.columns)
    known_weights = np.isfinite(aligned_weights).all(axis=1)
    held = aligned_weights.ne(0.0)
    missing_loading = held & ~np.isfinite(loading)
    strict_complete = known_weights & ~missing_loading.any(axis=1)
    strict_beta = loading.where(held, 0.0).mul(aligned_weights).sum(axis=1, skipna=False)
    strict_beta = strict_beta.where(strict_complete).rename(name)
    proxy_used = pd.DataFrame(False, index=grid, columns=weights.columns)
    if alpha_coverage is not None and alpha_coverage.proxy_map:
        # Fit separately: extra proxy prehistory must not change own-instrument model inputs.
        proxy_prices = alpha_coverage.proxy_prices.loc[:end, list(dict.fromkeys(
            alpha_coverage.proxy_map.values()))]
        if not proxy_prices.empty:
            proxy_grid = pd.date_range(min(proxy_prices.index[0], benchmark_prices.index[0]),
                                       end, freq=freq_beta)
            proxy_returns = _strict_log_returns(proxy_prices, freq_beta).reindex(proxy_grid)
            proxy_factors = _strict_log_returns(benchmark_prices.loc[:end], freq_beta).reindex(
                proxy_grid)
            proxies = _eligible_loadings(proxy_returns, proxy_factors, name,
                                         factor_beta_span).reindex(grid)
            for instrument, proxy in alpha_coverage.proxy_map.items():
                use = missing_loading[instrument] & np.isfinite(proxies[proxy])
                proxy_used[instrument] = use
                loading[instrument] = loading[instrument].where(~use, proxies[proxy])
    remaining = missing_loading & ~proxy_used
    gross = aligned_weights.abs()
    omitted = pd.DataFrame(False, index=grid, columns=weights.columns)
    if alpha_coverage is not None:
        tiny = remaining & gross.le(alpha_coverage.weight_tolerance)
        within_cap = gross.where(tiny, 0.0).sum(axis=1).le(
            alpha_coverage.total_weight_tolerance) & known_weights
        omitted = tiny.mul(within_cap, axis=0)
    uncovered = remaining & ~omitted
    complete = known_weights & ~uncovered.any(axis=1)
    beta = loading.where(held & ~omitted, 0.0).mul(aligned_weights).sum(axis=1, skipna=False)
    beta = beta.where(complete).rename(name)
    inputs = benchmark_returns.copy()
    inputs.columns = [name]
    attribution = compute_benchmarks_beta_attribution_from_returns(
        portfolio_returns=portfolio_returns.reindex(grid),
        benchmark_returns=inputs.reindex(grid),
        portfolio_benchmark_betas=beta.to_frame(),
    )
    alpha = attribution.Alpha.reindex(portfolio_returns.index)
    strict_attribution = compute_benchmarks_beta_attribution_from_returns(
        portfolio_returns=portfolio_returns.reindex(grid),
        benchmark_returns=inputs.reindex(grid), portfolio_benchmark_betas=strict_beta.to_frame(),
    )
    diagnostics = pd.DataFrame(
        {"lagged_beta": beta.shift(1), "complete_coverage": complete.shift(1, fill_value=False),
         "strict_beta": strict_beta.shift(1),
         "strict_complete_coverage": strict_complete.shift(1, fill_value=False),
         "strict_alpha": strict_attribution.Alpha,
         "unknown_weights": (~known_weights).shift(1, fill_value=True),
         "total_gross": gross.sum(axis=1, skipna=False).shift(1)},
        index=grid,
    ).reindex(portfolio_returns.index)
    for label, mask in (("direct", held & ~missing_loading), ("proxy", proxy_used),
                        ("omitted", omitted), ("uncovered", uncovered)):
        diagnostics[label + "_gross"] = gross.where(mask, 0.0).sum(axis=1).where(
            known_weights).shift(1).reindex(portfolio_returns.index)
    metadata["proxy_assisted"] = bool(diagnostics.proxy_gross.gt(0).any())
    metadata["omission_used"] = bool(diagnostics.omitted_gross.gt(0).any())
    if alpha_coverage is not None:
        metadata["beta_coverage"] = (
            "explicit proxy/dust policy; actual exposures; no renormalisation")
    issues = []
    unavailable = ~diagnostics.complete_coverage
    if unavailable.any():
        issues.append(
            f"Alpha: {int(unavailable.sum())} return dates lack a fully covered opening beta "
            "(warm-up, missing held-instrument returns/betas or unknown weights); left unavailable."
        )
    if metadata["proxy_assisted"] or metadata["omission_used"]:
        issues.append(
            "Alpha coverage assistance: "
            f"{int(diagnostics.proxy_gross.gt(0).sum())} proxy-assisted dates; "
            f"{int(diagnostics.omitted_gross.gt(0).sum())} dust-omission dates. "
            "See coverage audit; intervals exclude proxy and omission model uncertainty."
        )
    return alpha, diagnostics, metadata, issues
