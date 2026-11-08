"""Report-only Sharpe/holdings-alpha views; estimators remain in canonical modules."""

from dataclasses import dataclass
import math
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import norm

from qis.perfstats.returns import compute_excess_returns, prices_at_freq, to_returns
from qis.perfstats.sharpe_inference import (
    SharpeInference, sharpe_difference_test, sharpe_ratio_test,
)
from qis.perfstats.significance import InterceptInference
from qis.portfolio.reports._holdings_alpha import compute_holdings_alpha
from qis.portfolio.reports.config import validate_reporting_frequency
from qis.utils.annualisation import get_annualization_factor
from qis.utils.hac_intercept_prewhitening import estimate_prewhitened_intercepts
from qis.utils.regression import estimate_hac_mean, estimate_hac_regression, newey_west_lag_rule


@dataclass
class PerformanceInferenceData:
    """Owned appendix tables and provenance, not another estimator.

    Attributes:
        windows: Full, five-year and three-year arithmetic-Sharpe (or paired-difference) intervals.
        expanding: Month-end evaluated expanding Sharpe HAC intervals.
        rolling: Month-end evaluated calendar-window Sharpe HAC intervals.
        alpha_windows: Annualised mean holdings-alpha HAC intervals and two-sided p-values.
        alpha_expanding: Expanding-history mean-alpha HAC intervals.
        alpha_rolling: Calendar-window mean-alpha HAC intervals.
        returns: Strategy simple returns, cash adjusted if supplied, on the Sharpe grid.
        alpha_returns: Unsmoothed adjusted returns or paired difference; warm-up stays NaN.
        alpha_diagnostics: Lagged holdings beta and coverage, for each portfolio in paired mode.
        metadata: Frequencies, annualisation, eligibility rules and actual sample settings.
        issues: Deduplicated missing/short/undefined-panel reasons, also emitted as a warning.
    """

    windows: pd.DataFrame
    expanding: pd.DataFrame
    rolling: pd.DataFrame
    alpha_windows: pd.DataFrame
    alpha_expanding: pd.DataFrame
    alpha_rolling: pd.DataFrame
    returns: pd.DataFrame
    alpha_returns: pd.Series
    alpha_diagnostics: pd.DataFrame
    metadata: dict
    issues: list


def _sample_returns(prices, time_period, freq, rates_data):
    """Keep the opening price for cash accrual, and preserve missing native sampling bins."""
    original = time_period.locate(prices)
    levels = prices_at_freq(original, freq=freq, ffill_nans=False)
    observed = original.resample(freq).count()
    levels = levels.where(observed.reindex(levels.index).gt(0))
    result = to_returns(levels, is_log_returns=False, ffill_nans=False)
    if rates_data is not None and len(result):
        result = compute_excess_returns(result, rates_data)
    return result.iloc[1:]


def validate_alpha_inference(method):
    """Reject misspelled/unsupported methods rather than silently selecting the default."""
    if not isinstance(method, InterceptInference):
        raise ValueError("alpha_inference must be an InterceptInference member")


def alpha_method_label(method):
    """Compact labels used consistently in appendix tables and legends."""
    return "B" if method == InterceptInference.BARTLETT_HAC.name else "PW"


def _alpha_estimates(data, lags, af, method, paired, issue, context):
    """Delegate both fits on one validated sample; alternate failure never replaces primary."""
    fits = {}
    for candidate in InterceptInference:
        try:
            if candidate == InterceptInference.PREWHITENED_AR1:
                row = estimate_prewhitened_intercepts(data, hac_lags=lags).table.iloc[0]
                fit = dict(estimate=af * row.estimate, se=af * row.standard_error,
                           lower=af * row.ci_lower_pointwise, upper=af * row.ci_upper_pointwise,
                           pvalue=row.pvalue_normal, rho=row.rho)
            elif paired:
                # Retain the legacy neutral zero-difference convention (SE=0, p=1).
                result = estimate_hac_regression(data, x=None, span=None, hac_lags=lags)
                fit = dict(estimate=af * result.coefficients.iloc[0, 0],
                           se=af * result.hac_se.iloc[0, 0],
                           lower=af * result.confidence_lower.iloc[0, 0],
                           upper=af * result.confidence_upper.iloc[0, 0],
                           pvalue=result.pvalues.iloc[0, 0], rho=np.nan)
            else:
                result = estimate_hac_mean(data, hac_lags=lags)
                fit = dict(estimate=af * result.mean, se=af * result.hac_se,
                           lower=af * result.confidence_interval[0],
                           upper=af * result.confidence_interval[1],
                           pvalue=result.pvalue, rho=np.nan)
        except ValueError as error:
            issue(f"{context}: {candidate.name} unavailable ({error}); no method substitution.")
            fit = dict.fromkeys(("estimate", "se", "lower", "upper", "pvalue", "rho"), np.nan)
        fits[candidate] = fit
    alternate = next(candidate for candidate in InterceptInference if candidate != method)
    return dict(fits[method], lags=lags,
                **{f"comparison_{key}": value for key, value in fits[alternate].items()})


def _inference_views(sampled, freq, end, issues, *, alpha=False, paired=False,
                     alpha_inference=InterceptInference.BARTLETT_HAC):
    """Select complete report samples and delegate fits; never condense interior holes."""
    validate_alpha_inference(alpha_inference)
    prefix = "Alpha" if alpha else "Sharpe"
    if paired:
        prefix = "Marginal alpha" if alpha else "Paired Sharpe"
    af = float(get_annualization_factor(freq))
    minimum = max(24, math.ceil(2 * af))

    def issue(message):
        message = f"{prefix}: {message}"
        if message not in issues:
            issues.append(message)

    finite = np.isfinite(sampled.to_numpy(dtype=float))
    positions = np.flatnonzero(finite.all(axis=1) if finite.ndim == 2 else finite)
    trimmed = sampled.iloc[positions[0]:positions[-1] + 1] if len(positions) else sampled.iloc[:0]
    if len(trimmed) != len(sampled):
        issue("trims unavailable leading/trailing returns; see actual dates (no zero padding).")
    sampled = trimmed

    def estimate(data, context, iid=False):
        if len(data) < minimum:
            issue(f"{context}: requires {minimum} returns (24 and >=2 years at {freq}).")
            return None
        if not np.isfinite(data.to_numpy(dtype=float)).all():
            issue(f"{context}: interior missing/nonfinite returns; "
                  "unavailable, no filling/dropping.")
            return None
        lags = 0 if iid else newey_west_lag_rule(len(data))
        if not iid and len(data) < 5 * (lags + 1):
            issue(f"{context}: HAC needs N >=5*(L+1); no IID substitution.")
            return None
        try:
            if alpha:
                return _alpha_estimates(data, lags, af, alpha_inference, paired, issue, context)
            if paired:
                result = sharpe_difference_test(data.iloc[:, 0], data.iloc[:, 1],
                                                af=af, n_lags=lags)
                margin = norm.isf(0.025) * result.se_an
                return dict(estimate=result.diff_an, se=result.se_an,
                            lower=result.diff_an - margin, upper=result.diff_an + margin,
                            pvalue=result.p_value, lags=result.n_lags)
            method = SharpeInference.IID_NORMAL if iid else SharpeInference.HAC
            result = sharpe_ratio_test(data, method=method, af=af, n_lags=lags)
            return dict(estimate=result.sr_an, se=result.se_an, lower=result.ci_lower,
                        upper=result.ci_upper, lags=result.n_lags)
        except ValueError:
            issue(f"{context}: undefined {'alpha mean/variance' if alpha else 'Sharpe/variance'}.")
            return None

    def value(result, key):
        return result[key] if result else np.nan

    windows = []
    for label, years in (("Full history", None), ("Recent 5 years", 5), ("Recent 3 years", 3)):
        data = sampled if years is None else sampled.loc[
            sampled.index > end - pd.DateOffset(years=years)]
        hac = estimate(data, label)
        iid = None if alpha or paired else estimate(data, label, iid=True)
        row = dict(
            window=label, start=data.index[0] if len(data) else pd.NaT,
            end=data.index[-1] if len(data) else pd.NaT, n_obs=len(data), years=len(data) / af,
            hac_lower=value(hac, "lower"), hac_upper=value(hac, "upper"),
            hac_se=value(hac, "se"), hac_lags=value(hac, "lags"),
        )
        if alpha:
            row.update(alpha=value(hac, "estimate"), pvalue=value(hac, "pvalue"))
            row.update({key: value(hac, key) for key in (
                "rho", "comparison_estimate", "comparison_se", "comparison_lower",
                "comparison_upper", "comparison_pvalue", "comparison_rho")})
        elif paired:
            row.update(sharpe=value(hac, "estimate"), pvalue=value(hac, "pvalue"))
        else:
            row.update(sharpe=value(iid, "estimate"), iid_lower=value(iid, "lower"),
                       iid_upper=value(iid, "upper"))
        windows.append(row)

    normal_warmup = max(minimum, math.ceil(3 * af))
    warmup = normal_warmup if len(sampled) >= normal_warmup else minimum
    if minimum <= len(sampled) < normal_warmup:
        issue("Expanding chart: 3-year warm-up unavailable; using eligible 2-year minimum.")

    def calendar_window(history, date, years):
        cutoff = date - pd.DateOffset(years=years)
        window = history.loc[history.index > cutoff]
        expected = pd.date_range(cutoff, date, freq=freq)
        expected = expected[expected > cutoff]
        complete = len(window) >= minimum and window.index.equals(expected)
        return window, complete

    rolling_years = next((y for y in (5, 3, 2) if calendar_window(sampled, end, y)[1]), None)
    if rolling_years != 5:
        issue("Rolling chart: 5-year history unavailable; "
              f"{str(rolling_years) + '-year fallback' if rolling_years else 'panel unavailable'}.")
    expanding, rolling = [], []
    endpoints = sampled.groupby(sampled.index.to_period("M")).tail(1).index
    for date in endpoints:
        history = sampled.loc[:date]
        candidates = [(expanding, history, len(history) >= warmup, "Expanding chart")]
        if rolling_years is not None:
            window, complete = calendar_window(history, date, rolling_years)
            candidates.append((rolling, window, complete, "Rolling chart"))
        for target, sample, eligible, context in candidates:
            if eligible:
                result = estimate(sample, context)
                target.append(dict(date=date, estimate=value(result, "estimate"),
                                   ci_lower=value(result, "lower"),
                                   ci_upper=value(result, "upper")))
                if alpha:
                    target[-1].update({key: value(result, key) for key in (
                        "comparison_estimate", "comparison_lower", "comparison_upper")})
    metadata = dict(freq=freq, af=af, minimum_observations=minimum, rolling_years=rolling_years,
                    expanding_warmup=warmup)
    if alpha:
        metadata.update(inference=alpha_inference.name, comparison_inference=next(
            method.name for method in InterceptInference if method != alpha_inference),
            comparison_role="same-sample sensitivity only; unadjusted, not a second primary test")
    return (pd.DataFrame(windows).set_index("window"), pd.DataFrame(expanding),
            pd.DataFrame(rolling), metadata)


def compute_performance_inference(nav, benchmark_prices, time_period, freq, rates_data=None,
                                  *, instrument_prices=None, weights=None, freq_beta=None,
                                  factor_beta_span=52,
                                  alpha_inference=InterceptInference.BARTLETT_HAC):
    """Prepare independent Sharpe and holdings-alpha samples for the two-column appendix.

    Args:
        nav: Strategy NAV; existing fees/costs are not deducted again.
        benchmark_prices: Raw reference prices; only the first column is used for alpha.
        time_period: Performance boundaries, applied before realised return sampling.
        freq: Sharpe sampling grid, normally perf_params.freq_vol.
        rates_data: Optional annual cash quotes; applied to both portfolio and benchmark.
        instrument_prices: Full-history instrument prices for point-in-time beta estimation.
        weights: Actual instrument exposures as of each beta date; never target weights.
        freq_beta: Alpha return and instrument-beta grid; defaults to freq for direct callers.
        factor_beta_span: Report's instrument beta span and warm-up on freq_beta.
        alpha_inference: Primary alpha method; the other method is same-sample sensitivity only.

    Returns:
        Tables, unsmoothed returns and provenance. Ineligible panels stay unavailable with a
        warning, never substituted with a NAV regression or an invented beta.
    """
    validate_alpha_inference(alpha_inference)
    issues = []
    validate_reporting_frequency(time_period.locate(nav), freq)
    sampled = _sample_returns(nav, time_period, freq, rates_data).rename("Strategy")
    end = sampled.index[-1] if len(sampled) else pd.Timestamp(time_period.end)
    windows, expanding, rolling, metadata = _inference_views(sampled, freq, end, issues)
    alpha_freq = freq_beta or freq
    alpha_grid_valid = True
    try:
        validate_reporting_frequency(time_period.locate(nav), alpha_freq)
        alpha_portfolio = _sample_returns(nav, time_period, alpha_freq, rates_data)
    except ValueError as error:
        alpha_grid_valid = False
        alpha_portfolio = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        issues.append(f"Alpha: sampling grid unavailable ({error}); no upsampling/substitute fit.")
    alpha_end = alpha_portfolio.index[-1] if len(alpha_portfolio) else end
    alpha = pd.Series(np.nan, index=alpha_portfolio.index, name="Alpha")
    diagnostics = pd.DataFrame(index=alpha.index)
    references = benchmark_prices.to_frame() if isinstance(benchmark_prices, pd.Series) else (
        benchmark_prices)
    alpha_meta = dict(benchmark="Unavailable", freq_beta=alpha_freq,
                      factor_beta_span=factor_beta_span)
    if not alpha_grid_valid:
        if references is not None and references.shape[1]:
            alpha_meta["benchmark"] = str(references.columns[0])
    elif references is None or references.empty or references.shape[1] == 0:
        issues.append("Alpha: no benchmark supplied; no substitute benchmark selected.")
    elif instrument_prices is None or weights is None:
        alpha_meta["benchmark"] = str(references.columns[0])
        issues.append("Alpha: instrument prices/actual holdings unavailable; no NAV-beta fallback.")
    else:
        first = references.iloc[:, :1]
        alpha_meta["benchmark"] = str(first.columns[0])
        reference_returns = _sample_returns(first, time_period, alpha_freq, rates_data)
        reference_returns = reference_returns.reindex(alpha_portfolio.index)
        try:
            alpha, diagnostics, beta_meta, beta_issues = compute_holdings_alpha(
                instrument_prices=instrument_prices, weights=weights, benchmark_prices=first,
                portfolio_returns=alpha_portfolio, benchmark_returns=reference_returns,
                freq_beta=alpha_freq, factor_beta_span=factor_beta_span,
            )
            alpha_meta.update(beta_meta)
            issues.extend(beta_issues)
        except ValueError as error:
            issues.append(f"Alpha: holdings-beta inputs unavailable ({error}); no substitute fit.")
    aw, ae, ar, am = _inference_views(alpha, alpha_freq, alpha_end, issues, alpha=True,
                                     alpha_inference=alpha_inference)
    alpha_meta.update(am)
    metadata.update(
        alpha=alpha_meta, confidence_level=0.95,
        return_basis="excess of supplied cash" if rates_data is not None else
        "total returns; zero cash assumed", cost_basis="as carried in NAV; no extra deductions",
        variance="Sharpe: unprewhitened Bartlett HAC; alpha: " + alpha_inference.name +
        "; automatic Newey-West lags",
    )
    issues = list(dict.fromkeys(issues))
    if issues:
        warnings.warn("Performance inference appendix: " + " ".join(issues),
                      UserWarning, stacklevel=2)
    return PerformanceInferenceData(windows, expanding, rolling, aw, ae, ar,
                                    sampled.to_frame(), alpha, diagnostics, metadata, issues)


def plot_performance_inference(data, strategy_name, figsize=(8.5, 11.7)):
    """Draw two diagnostic columns without changing any summary-page artist or calculation."""
    with plt.rc_context({"font.size": 8}):
        figure = plt.figure(figsize=figsize)
        figure.subplots_adjust(left=0.09, right=0.96, top=0.84, bottom=0.23,
                               hspace=0.55, wspace=0.38)
        grid = figure.add_gridspec(3, 2, height_ratios=[1.05, 1, 1])
        paired = data.metadata.get("paired", False)
        title = "Incremental performance" if paired else "Performance confidence and stability"
        figure.suptitle(f"{strategy_name}\n{title}", y=0.98,
                       fontsize=14, weight="bold", color="#164773")
        meta = data.metadata
        alpha_meta = meta["alpha"]
        primary = alpha_method_label(alpha_meta.get("inference", "BARTLETT_HAC"))
        alternate = "PW" if primary == "B" else "B"
        columns = (
            ("Paired Sharpe difference" if paired else "Sharpe diagnosis",
             f"{'Difference in arithmetic Sharpe' if paired else 'Arithmetic Sharpe'} | "
             f"{meta['freq']} | af={meta['af']:g}",
             data.windows, data.expanding, data.rolling, meta, False),
            ("Marginal holdings alpha" if paired else "Alpha diagnosis",
             f"Holdings-weighted vs {alpha_meta['benchmark']} (first benchmark)\n"
             f"{alpha_meta['freq']} | af={alpha_meta['af']:g} | "
             f"beta span / warm-up: {alpha_meta['factor_beta_span']}\n"
             f"Primary: {primary} | same-sample sensitivity: {alternate}",
             data.alpha_windows, data.alpha_expanding, data.alpha_rolling, alpha_meta, True),
        )
        for col, (heading, subtitle, windows, expanding, rolling, settings, is_alpha) in enumerate(
                columns):
            x = 0.09 if col == 0 else 0.595
            figure.text(x, 0.916, heading, fontsize=12, weight="bold", color="#164773")
            figure.text(x, 0.88, subtitle, fontsize=7.5, va="center", linespacing=1.5)
            axis = figure.add_subplot(grid[0, col])
            axis.axis("off")

            def interval(lower, upper):
                if not np.isfinite(lower + upper):
                    return "—"
                return (f"[{lower:.1%}, {upper:.1%}]" if is_alpha else
                        f"[{lower:.2f}, {upper:.2f}]")

            rows = []
            for name, row in windows.iterrows():
                dates = (f"{row.start:%b%Y}–{row.end:%b%Y}" if pd.notna(row.start) else "No sample")
                lag = f"{row.hac_lags:g}" if np.isfinite(row.hac_lags) else "—"
                shown_name = (
                    "Full eligible history" if is_alpha and name == "Full history" else name)
                label = f"{shown_name}\n{dates}\nN={row.n_obs:.0f}; L={lag}"
                estimate = row.alpha if is_alpha else row.sharpe
                estimate = (f"{estimate:.1%}" if is_alpha else f"{estimate:.2f}") if (
                    np.isfinite(estimate)) else "—"
                def p_label(value):
                    if not np.isfinite(value):
                        return "N/A"
                    return "<.001" if value < .001 else f"{value:.3f}"

                if is_alpha:
                    other_interval = interval(row.comparison_lower, row.comparison_upper)
                    rows.append([label, estimate,
                                 f"{primary}: {interval(row.hac_lower, row.hac_upper)}\n"
                                 f"{alternate}: {other_interval}",
                                 f"{primary}: {p_label(row.pvalue)}\n"
                                 f"{alternate}: {p_label(row.comparison_pvalue)}"])
                else:
                    rows.append([label, estimate, interval(row.hac_lower, row.hac_upper),
                                 (f"{row.pvalue:.3f}" if np.isfinite(row.pvalue) else "—") if
                                 paired else interval(row.iid_lower, row.iid_upper)])
            table = axis.table(cellText=rows,
                               colLabels=["Sample", ("ΔAlpha¹" if is_alpha else "ΔSharpe") if
                                          paired else ("Alpha¹" if is_alpha else "Sharpe"),
                                          "HAC 95% CI",
                                          "p (2-sided)" if is_alpha or paired else "IID 95% CI"],
                               colWidths=[0.33, 0.14, 0.34, 0.19] if is_alpha else
                               [0.39, 0.15, 0.25, 0.21], cellLoc="center",
                               bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(6.4)
            for (row, _), cell in table.get_celld().items():
                cell.set_edgecolor("white")
                cell.set_facecolor("#164773" if row == 0 else
                                   "#edf3f8" if row % 2 else "#f7f9fb")
                if row == 0:
                    cell.set_text_props(color="white", weight="bold")
                    cell.set_height(cell.get_height() * 0.6)
            for row, frame, title in (
                (1, expanding, "Expanding history"),
                (2, rolling, f"{settings['rolling_years']}-year rolling" if
                 settings['rolling_years'] else "Rolling history — unavailable"),
            ):
                axis = figure.add_subplot(grid[row, col])
                axis.set_title(title, fontsize=9, weight="bold", pad=10)
                axis.axhline(0, color=".4", linestyle="--", linewidth=0.8)
                comparison_available = (is_alpha and not frame.empty and
                                        frame.comparison_lower.notna().any())
                if not frame.empty and (frame.estimate.notna().any() or comparison_available):
                    axis.fill_between(frame.date, frame.ci_lower, frame.ci_upper,
                                      color="#c8dff1", label=f"{primary} 95% CI (primary)" if
                                      is_alpha else "HAC 95% CI")
                    if is_alpha:
                        axis.plot(frame.date, frame.comparison_lower, "--", color="#ba7539",
                                  linewidth=.9, label=f"{alternate} 95% CI (sensitivity)")
                        axis.plot(frame.date, frame.comparison_upper, "--", color="#ba7539",
                                  linewidth=.9)
                    curve_label = "Mean alpha / year" if is_alpha else "Arithmetic Sharpe"
                    if paired:
                        curve_label = "Marginal alpha / year" if is_alpha else "Sharpe difference"
                    axis.plot(frame.date, frame.estimate, color="#1764a0", linewidth=1.2,
                              label=curve_label,
                              marker="o" if len(frame) == 1 else None)
                    axis.legend(fontsize=6, loc="upper left")
                    axis.tick_params(axis="x", rotation=35, labelsize=7)
                    if len(frame) > 1:
                        axis.set_xlim(frame.date.iloc[0], frame.date.iloc[-1])
                    else:
                        point = frame.iloc[0]
                        axis.errorbar(point.date, point.estimate,
                                      yerr=[[point.estimate - point.ci_lower],
                                            [point.ci_upper - point.estimate]],
                                      color="#1764a0", capsize=4)
                else:
                    axis.text(0.5, 0.5, "Insufficient eligible history\nSee safeguards below",
                              transform=axis.transAxes, ha="center", va="center", fontsize=7)
                    axis.set_xticks([])
                    axis.set_yticks([])
                if is_alpha:
                    axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
                ylabel = "Annualised mean alpha" if is_alpha else "Annualised Sharpe"
                if paired:
                    ylabel = "Annualised marginal alpha" if is_alpha else "Annualised ΔSharpe"
                axis.set_ylabel(ylabel, fontsize=7)
                axis.grid(alpha=0.2)
                axis.spines[["top", "right"]].set_visible(False)
        first_alpha = data.alpha_returns.first_valid_index()
        alpha_start = f"{first_alpha:%d%b%Y}" if first_alpha is not None else "unavailable"
        footer = (
            f"Basis: simple {meta['return_basis']}; existing NAV costs retained. "
            "All intervals: 95%.\n"
            "Alpha = portfolio excess return − prior holdings beta × benchmark excess return. "
            "No NAV regression.\n"
            "Betas: EWMA-demeaned instrument log returns; "
            "applied to simple returns (approximation). "
            "No beta=1 prior.\n"
            f"Minimum N: Sharpe {meta['minimum_observations']}, alpha "
            f"{alpha_meta['minimum_observations']}; each ≥24 and ≥2 years; HAC needs N ≥5(L+1). "
            "Automatic lags.\n"
            f"First fully covered alpha return: {alpha_start}. "
            "Warm-up excluded, no padded zeros; interior gaps stay gaps.\n"
            "Expanding warm-up: normally 3 years. Rolling: 5 → 3 → 2 years with warning. "
            "Unavailable fits stay unavailable.\n"
            "Pointwise asymptotic diagnostics, not simultaneous or selection-adjusted. "
            "Overlapping windows are dependent.\n"
            "¹ Alpha is annualised mean adjusted return, not certainty about true beta. "
            "Finite-sample coverage is not guaranteed."
        )
        if paired:
            footer = (
                f"Strategy minus comparison portfolio: positive favours strategy. "
                f"Simple {meta['return_basis']}.\n"
                "ΔAlpha = strategy return − comparison return − prior holdings beta difference "
                "× market excess return.\n"
                "Paired HAC retains covariance. Betas: EWMA-demeaned instrument log returns; "
                "simple-return application is approximate.\n"
                f"Minimum N: Sharpe {meta['minimum_observations']}, alpha "
                f"{alpha_meta['minimum_observations']}; each ≥24 and ≥2 years; "
                "HAC needs N ≥5(L+1). Automatic lags.\n"
                f"First jointly covered alpha return: {alpha_start}. "
                "No beta=1 prior, NAV regression, padded zeros or dropped interior gaps.\n"
                "Expanding warm-up: normally 3 years. Rolling: 5 → 3 → 2 years with warning. "
                "Existing NAV costs retained.\n"
                "Pointwise asymptotic intervals, not simultaneous or selection-adjusted. "
                "Overlapping windows are dependent.\n"
                "¹ Annualised mean residual difference; not pure manager skill or "
                "certainty about true beta. Coverage is not guaranteed."
            )
        footer += ("\nAlpha only: B = Bartlett HAC; PW = AR(1)-prewhitened Bartlett. Same N/L; "
                   "sensitivity p-values unadjusted; no method selection by p.")
        figure.text(0.09, 0.067, footer, fontsize=6.3, color=".35", linespacing=1.6, va="bottom")
        base = "https://quantinveststrats.readthedocs.io/en/latest/"
        for x, label, target in (
            (0.09, "Paired Sharpe test" if paired else "Sharpe CI (IID/HAC)",
             "factsheets_and_reporting.html#paired-strategy-benchmark-inference" if paired else
             "performance_analytics_and_sharpe.html#sampling-uncertainty"),
            (0.39, "Holdings-weighted alpha", "model_layer_attribution.html#"
             "holdings-exposure-versus-model-layer-nav-regression"),
            (0.74, "HAC limitations", "factor_and_backtest_significance.html"),
        ):
            figure.text(x, 0.035, f"qis docs: {label}", fontsize=6.5, color="#1764a0",
                        va="bottom", url=base + target)
    return figure
