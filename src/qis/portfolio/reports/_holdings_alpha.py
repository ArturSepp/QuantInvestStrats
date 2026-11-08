"""Strict, report-only holdings alpha; the EWM and attribution estimators are reused."""

import numpy as np
import pandas as pd

from qis.models.linear.ewm import InitType, MeanAdjType
from qis.perfstats.returns import prices_at_freq, to_returns
from qis.portfolio.risk.ewm_factor_model import EwmLinearModel
from qis.portfolio.risk.factor_model import compute_benchmarks_beta_attribution_from_returns


def _strict_log_returns(prices, freq):
    """Sample completed boundaries without turning an empty native bin into a stale price."""
    levels = prices_at_freq(prices, freq=freq, ffill_nans=False)
    observed = prices.resample(freq).count().reindex(levels.index)
    levels = levels.where(observed.gt(0))
    return to_returns(levels, is_log_returns=True, ffill_nans=False)


def compute_holdings_alpha(
    instrument_prices,
    weights,
    benchmark_prices,
    portfolio_returns,
    benchmark_returns,
    freq_beta,
    factor_beta_span,
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

    Returns:
        Alpha Series, same-index diagnostics, metadata and human-readable issue strings. No
        partial-coverage alpha is published. The canonical model estimates log-return betas;
        using these betas on simple returns is the existing second-order approximation.
        This is a realised exposure-adjusted payoff, not a fitted NAV-regression intercept.
    """
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
    model = EwmLinearModel(x=factor_returns, y=asset_returns)
    model.fit(
        span=factor_beta_span, mean_adj_type=MeanAdjType.EWMA,
        init_type=InitType.X0, warmup_period=factor_beta_span,
    )
    # The estimator carries its state across missing rows. Such rows are not observations
    # for warm-up and cannot publish a fresh fully covered beta in this inference appendix.
    loading = model.loadings[name].where(paired.cumsum().gt(factor_beta_span) & paired)
    aligned_weights = weights.reindex(grid, method="ffill")
    # Price columns without a weight column are not in the portfolio. Conversely a nonzero
    # weight without instrument history is unknown beta, never an implicit zero contribution.
    loading = loading.reindex(columns=weights.columns)
    known_weights = np.isfinite(aligned_weights).all(axis=1)
    held = aligned_weights.ne(0.0)
    missing_loading = held & ~np.isfinite(loading)
    complete = known_weights & ~missing_loading.any(axis=1)
    beta = loading.where(held, 0.0).mul(aligned_weights).sum(axis=1, skipna=False)
    beta = beta.where(complete).rename(name)
    inputs = benchmark_returns.copy()
    inputs.columns = [name]
    attribution = compute_benchmarks_beta_attribution_from_returns(
        portfolio_returns=portfolio_returns.reindex(grid),
        benchmark_returns=inputs.reindex(grid),
        portfolio_benchmark_betas=beta.to_frame(),
    )
    alpha = attribution.Alpha.reindex(portfolio_returns.index)
    diagnostics = pd.DataFrame(
        {"lagged_beta": beta.shift(1), "complete_coverage": complete.shift(1, fill_value=False)},
        index=grid,
    ).reindex(portfolio_returns.index)
    issues = []
    unavailable = ~diagnostics.complete_coverage
    if unavailable.any():
        issues.append(
            f"Alpha: {int(unavailable.sum())} return dates lack a fully covered opening beta "
            "(warm-up, missing held-instrument returns/betas or unknown weights); left unavailable."
        )
    return alpha, diagnostics, metadata, issues
