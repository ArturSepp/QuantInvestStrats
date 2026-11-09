"""Multi-strategy inference overview; paired estimators and Holm remain in their owners."""

from dataclasses import dataclass
from numbers import Integral
import textwrap
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qis.perfstats.multiple_testing import MultipleTestingMethod, adjust_pvalues
from qis.perfstats.significance import InterceptInference
from qis.portfolio.reports.alpha_coverage import validate_alpha_coverage
from qis.portfolio.reports._paired_performance_inference import compute_paired_performance_inference
from qis.portfolio.reports._performance_inference import (
    alpha_method_label, validate_alpha_inference,
)
from qis.utils.dates import TimePeriod

_WINDOWS = ("Full history", "Recent 5 years", "Recent 3 years")
_METRICS = ("sharpe", "alpha")
_PER_PAGE = 6


def _pvalue_label(value):
    """Avoid displaying tiny positive tail probabilities as exact zero."""
    if not np.isfinite(value):
        return "N/A"
    return "<0.001" if value < .001 else f"{value:.3f}"


@dataclass
class MultiPerformanceInferenceData:
    """Report-owned paired results and an explicitly declared summary-test family.

    Attributes:
        summary: MultiIndex rows (portfolio position, metric, window), including unavailable tests.
        pairs: Paired appendix results by non-base portfolio position, in original order.
        metadata: Base, market, sampling settings and complete Holm-family membership/status.
        issues: Collected eligibility reasons and whole-family adjustment warnings.
    """

    summary: pd.DataFrame
    pairs: dict
    metadata: dict
    issues: list


def validate_inference_base(portfolios, base_portfolio_id):
    """Validate zero-based identity before building report figures, not by observed results."""
    if (isinstance(base_portfolio_id, (bool, np.bool_)) or
            not isinstance(base_portfolio_id, Integral) or
            not 0 <= base_portfolio_id < len(portfolios)):
        raise ValueError("base_portfolio_id must be a valid zero-based portfolio position")


def compute_multi_performance_inference(
    multi_portfolio_data, base_portfolio_id=0, time_period=None, freq="ME", rates_data=None,
    *, freq_beta="B", factor_beta_span=260, alpha_inference=InterceptInference.BARTLETT_HAC,
    alpha_coverage=None,
):
    """Prepare portfolio-minus-base estimates without changing any legacy estimator defaults.

    Args:
        multi_portfolio_data: Actual portfolios and shared market benchmark prices.
        base_portfolio_id: Zero-based portfolio position; excluded from comparison rows.
        time_period: Realised reporting window; None uses the union NAV boundaries.
        freq: Paired Sharpe return grid; report passes perf_params.freq_vol.
        rates_data: Same annual cash quotes for all portfolios and the market reference.
        freq_beta: Shared holdings-beta and raw-alpha grid, independent of Sharpe sampling.
        factor_beta_span: Report's EWM span and instrument warm-up on the beta grid.
        alpha_inference: Primary alpha method in the unchanged 6K-test Holm family; the alternate
            method is same-sample sensitivity only, with raw unadjusted p-values.
        alpha_coverage: Optional universe-wide coverage policy, restricted to each paired report.

    Returns:
        All six tests per non-base portfolio, pair-level diagnostics and predeclared family.
        Holm is applied once across both metrics, all windows and all pages, only when every
        declared raw p-value is finite. Otherwise all adjusted values stay NaN with a warning.
        Valid raw results remain visible; no fabricated p-values or reduced family is used.
    """
    validate_alpha_inference(alpha_inference)
    portfolios = multi_portfolio_data.portfolio_datas
    validate_alpha_coverage(alpha_coverage, {column for portfolio in portfolios
                            if portfolio.weights is not None for column in portfolio.weights})
    validate_inference_base(portfolios, base_portfolio_id)
    base = portfolios[base_portfolio_id]
    positions = [i for i in range(len(portfolios)) if i != base_portfolio_id]
    family = pd.MultiIndex.from_tuples(
        [(i, metric, window) for i in positions for metric in _METRICS for window in _WINDOWS],
        names=["portfolio_id", "metric", "window"],
    )
    # Freeze membership before any estimator is called; pagination never changes this family.
    metadata = dict(base_portfolio_id=int(base_portfolio_id), base=str(base.ticker),
                    portfolio_ids=positions, family_members=tuple(family),
                    family_id="multi-strategy-inference-summary", family_size=len(family),
                    adjustment="Holm", adjustment_status="unavailable", freq=freq,
                    freq_beta=freq_beta, factor_beta_span=factor_beta_span,
                    alpha_inference=alpha_inference.name,
                    coverage_enabled=alpha_coverage is not None)
    if time_period is None:
        navs = multi_portfolio_data.get_navs()
        time_period = TimePeriod(navs.index[0], navs.index[-1])
    pairs, rows, issues = {}, [], []
    for position in positions:
        # Aggregate warnings once, retaining portfolio identity in every reason.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            pair = compute_paired_performance_inference(
                strategy=portfolios[position], comparison=base,
                benchmark_prices=multi_portfolio_data.benchmark_prices,
                time_period=time_period, freq=freq, rates_data=rates_data,
                freq_beta=freq_beta, factor_beta_span=factor_beta_span,
                alpha_inference=alpha_inference,
                alpha_coverage=(alpha_coverage.for_instruments({
                    column for portfolio in (portfolios[position], base)
                    if portfolio.weights is not None for column in portfolio.weights})
                    if alpha_coverage is not None else None),
            )
        pairs[position] = pair
        issues.extend(f"Portfolio {position} ({portfolios[position].ticker}): {message}"
                      for message in pair.issues)
        # Do not hide unrelated runtime warnings emitted by canonical code.
        for warning in caught:
            if not str(warning.message).startswith("Paired performance inference appendix:"):
                warnings.warn(warning.message, warning.category, stacklevel=2)
        for metric, table in (("sharpe", pair.windows), ("alpha", pair.alpha_windows)):
            for window in _WINDOWS:
                result = table.loc[window]
                rows.append(dict(
                    portfolio=str(portfolios[position].ticker), estimate=result[metric],
                    ci_lower=result.hac_lower, ci_upper=result.hac_upper, se=result.hac_se,
                    pvalue=result.pvalue, start=result.start, end=result.end,
                    n_obs=result.n_obs, hac_lags=result.hac_lags,
                    comparison_lower=result.get("comparison_lower", np.nan),
                    comparison_upper=result.get("comparison_upper", np.nan),
                    comparison_se=result.get("comparison_se", np.nan),
                    comparison_pvalue=result.get("comparison_pvalue", np.nan),
                ))
    summary = pd.DataFrame(rows, index=family, columns=[
        "portfolio", "estimate", "ci_lower", "ci_upper", "se", "pvalue", "start", "end",
        "n_obs", "hac_lags", "comparison_lower", "comparison_upper", "comparison_se",
        "comparison_pvalue"])
    summary["pvalue_holm"] = np.nan
    if not positions:
        issues.append("No non-base portfolios; inference appendix omitted.")
        metadata["adjustment_status"] = "no comparisons"
    elif np.isfinite(summary.pvalue.to_numpy(dtype=float)).all():
        adjusted = adjust_pvalues(summary.pvalue, family_id=metadata["family_id"],
                                  method=MultipleTestingMethod.HOLM)
        summary["pvalue_holm"] = adjusted.pvalue_adjusted
        metadata["adjustment_status"] = "complete"
    else:
        missing = summary.index[~np.isfinite(summary.pvalue.to_numpy(dtype=float))]
        metadata["unavailable_family_members"] = tuple(missing)
        issues.append(f"Holm unavailable: {len(missing)} of {len(family)} "
                      "declared tests unavailable; "
                      "family not reduced. Valid raw p-values and pointwise CIs are retained.")
    issues = list(dict.fromkeys(issues))
    if issues:
        warnings.warn("Multi-strategy inference appendix: " + " ".join(issues),
                      UserWarning, stacklevel=2)
    return MultiPerformanceInferenceData(summary, pairs, metadata, issues)


def plot_multi_performance_inference(data, figsize=(8.3, 11.7)):
    """Six comparison panels per page; no result sorting, dropped portfolios or CI clipping."""
    positions = data.metadata["portfolio_ids"]
    if not positions:
        return []
    meta = data.metadata
    sample_meta = data.pairs[positions[0]].metadata
    primary = alpha_method_label(meta["alpha_inference"])
    alternate = "PW" if primary == "B" else "B"
    limits = {}
    for metric in _METRICS:
        columns = ["ci_lower", "ci_upper"]
        if metric == "alpha":
            columns += ["comparison_lower", "comparison_upper"]
        values = data.summary.xs(metric, level="metric")[columns].to_numpy()
        finite = np.abs(values[np.isfinite(values)])
        maximum = float(finite.max()) if len(finite) else 0.0
        limits[metric] = 1.1 * maximum if maximum > 0 else (1.0 if metric == "sharpe" else .01)
    figures = []
    pages = (len(positions) + _PER_PAGE - 1) // _PER_PAGE
    with plt.rc_context({"font.size": 8}):
        for page, offset in enumerate(range(0, len(positions), _PER_PAGE), 1):
            selected = positions[offset:offset + _PER_PAGE]
            figure = plt.figure(figsize=figsize)
            figure.subplots_adjust(left=.065, right=.97, top=.825, bottom=.235,
                                   hspace=.48, wspace=.16)
            grid = figure.add_gridspec(3, 2)
            base_label = textwrap.fill(meta["base"], width=58)
            figure.suptitle(f"Incremental performance vs base\n{base_label}", y=.98,
                           fontsize=13, weight="bold", color="#164773")
            figure.text(.065, .905, "Paired Sharpe difference", fontsize=11,
                        weight="bold", color="#164773")
            figure.text(.55, .905, "Coverage-adjusted marginal alpha" if
                        meta.get("coverage_enabled") else "Marginal holdings alpha", fontsize=10,
                        weight="bold", color="#164773")
            figure.text(.065, .87, f"Arithmetic | {meta['freq']} | af={sample_meta['af']:g}",
                        fontsize=7)
            figure.text(.55, .87,
                        f"Market: {sample_meta['alpha']['benchmark']} (first reference)\n"
                        f"{meta['freq_beta']} | af={sample_meta['alpha']['af']:g} | "
                        f"beta span/warm-up: {meta['factor_beta_span']}\n"
                        f"Primary: {primary} | dashed sensitivity: {alternate}", fontsize=7)
            for row, window in enumerate(_WINDOWS):
                for col, metric in enumerate(_METRICS):
                    axis = figure.add_subplot(grid[row, col])
                    axis.set_title("Full eligible history" if row == 0 else window,
                                   fontsize=9, weight="bold", pad=12)
                    axis.set_xlim(0, 1)
                    axis.set_ylim(-.75, len(selected) - .05)
                    axis.axis("off")
                    bound = limits[metric]

                    def location(value):
                        return .48 + .12 * value / bound

                    def number(value):
                        return f"{value:.1%}" if metric == "alpha" else f"{value:.2f}"

                    axis.vlines(location(0), -.38, len(selected) - .6,
                                color=".55", linestyle="--", linewidth=.7)
                    axis.text(.64, len(selected) - .06, "Estimate [95% CI]; p raw / Holm",
                              fontsize=5.9, va="bottom")
                    for index, position in enumerate(selected):
                        result = data.summary.loc[(position, metric, window)]
                        y = len(selected) - 1 - index
                        axis.axhspan(y - .45, y + .45, color="#edf3f8" if index % 2 == 0 else
                                     "#f7f9fb", zorder=0)
                        # Explicit position prevents collisions between equal/truncated tickers.
                        label = textwrap.shorten(str(result.portfolio), width=22, placeholder="…")
                        axis.text(0, y + .18, f"{position}: {label}", fontsize=6.7,
                                  weight="bold", va="center")
                        dates = (f"{result.start:%b%Y}-{result.end:%b%Y}" if
                                 pd.notna(result.start) else "No sample")
                        lag = f"{result.hac_lags:g}" if np.isfinite(result.hac_lags) else "N/A"
                        axis.text(0, y -.18, f"{dates}; N={result.n_obs:.0f}; L={lag}",
                                  fontsize=5.4, color=".35", va="center")
                        if np.isfinite([result.estimate, result.ci_lower, result.ci_upper]).all():
                            axis.hlines(y, location(result.ci_lower), location(result.ci_upper),
                                        color="#1764a0", linewidth=1.5)
                            axis.plot(location(result.estimate), y, "o", markersize=3,
                                      color="#1764a0")
                            estimate = (f"{number(result.estimate)} "
                                        f"[{number(result.ci_lower)}, {number(result.ci_upper)}]")
                        else:
                            estimate = "N/A"
                        p = _pvalue_label(result.pvalue)
                        holm = _pvalue_label(result.pvalue_holm)
                        if metric == "alpha":
                            axis.text(.64, y + .32, f"{primary}: {estimate}", fontsize=5.4,
                                      va="center")
                            axis.text(.64, y, f"p: {p} / {holm}", fontsize=5.3,
                                      color=".35", va="center")
                            other = "N/A"
                            if np.isfinite([result.comparison_lower,
                                            result.comparison_upper]).all():
                                axis.hlines(y - .18, location(result.comparison_lower),
                                            location(result.comparison_upper), color="#ba7539",
                                            linestyle="--", linewidth=1)
                                other = (f"[{number(result.comparison_lower)}, "
                                         f"{number(result.comparison_upper)}]")
                            axis.text(.64, y - .32,
                                      f"{alternate}: {other}; "
                                      f"p {_pvalue_label(result.comparison_pvalue)}",
                                      fontsize=5.1, color="#8b5427", va="center")
                        else:
                            axis.text(.64, y + .18, estimate, fontsize=6.0, va="center")
                            axis.text(.64, y -.18, f"p: {p} / {holm}", fontsize=6.0,
                                      color=".35", va="center")
                    for value in (-bound, 0, bound):
                        axis.text(location(value), -.65, number(value), fontsize=5.6,
                                  ha="center", color=".35")
            footer = (
                f"Base: portfolio {meta['base_portfolio_id']}. Differences = portfolio minus base; "
                "positive favours portfolio.\n"
                f"Basis: simple {sample_meta['return_basis']}; costs already in NAV retained. "
                "Shared axes within each metric.\n"
                "Marginal alpha = return difference − prior holdings beta difference × market "
                "excess return. No NAV regression.\n"
                "EWMA-demeaned instrument log betas; simple-return application is approximate. "
                "Both portfolios must meet the declared coverage policy.\n"
                "Minimum: 24 returns and 2 observation-equivalent years; HAC needs N ≥5(L+1). "
                "Automatic lags; actual paired dates shown.\n"
                "Unavailable tests stay N/A; no padded zeros, beta=1 prior or dropped "
                "interior gaps. "
                "Different pairs may use different dates.\n"
                f"Holm: {meta['family_size']} tests across ALL portfolios, metrics and windows; "
                f"status: {meta['adjustment_status']}.\n"
                "If any declared test is missing, all Holm values are N/A; the family is never "
                "reduced or split by page.\n"
                "CIs are pointwise/asymptotic, not simultaneous or Holm-adjusted. "
                "No guarantee of finite-sample coverage or pure manager skill."
                "\nAlpha: B = Bartlett; PW = AR(1)-prewhitened Bartlett. Same N/L; alternate raw p "
                "is sensitivity only, outside Holm."
            )
            if meta.get("coverage_enabled"):
                footer += ("\nSee coverage audit pages for assisted exposures and excluded dates. "
                           "CIs exclude proxy/omission uncertainty.")
            figure.text(.065, .065, footer, fontsize=6.1, color=".35", linespacing=1.55,
                        va="bottom")
            for x, label, target in (
                (.065, "Paired diagnostics", "factsheets_and_reporting.html#"
                 "multi-strategy-inference-against-a-base"),
                (.39, "Holdings alpha", "model_layer_attribution.html#"
                 "holdings-exposure-versus-model-layer-nav-regression"),
                (.70, "HAC / Holm limits", "factor_and_backtest_significance.html"),
            ):
                figure.text(x, .035, f"qis docs: {label}", fontsize=6.4, color="#1764a0",
                            url="https://quantinveststrats.readthedocs.io/en/latest/" + target)
            figure.text(.97, .015, f"Inference {page}/{pages}", fontsize=6, ha="right")
            figures.append(figure)
    return figures
