"""Validate and display supplied beta inference; estimation stays with the caller."""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Annotation


def validate_beta_intervals(table, confidence, label, target="benchmark", kind="confidence"):
    """Copy caller-owned bounds without inferring missing estimates or coverage."""
    if isinstance(confidence, (bool, np.bool_)) or not 0 < confidence < 1:
        raise ValueError("beta_interval_confidence must be between zero and one")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("beta_interval_label must be nonempty")
    if target not in ("production", "benchmark"):
        raise ValueError("beta_interval_target must be production or benchmark")
    if kind not in ("confidence", "stability"):
        raise ValueError("beta_interval_kind must be confidence or stability")
    if table is None:
        return None
    if (not isinstance(table, pd.DataFrame) or table.empty or table.index.has_duplicates
            or table.columns.has_duplicates or not {"estimate", "lower", "upper"} <= set(table)):
        raise ValueError("factor_beta_intervals requires unique labels and estimate/lower/upper")
    table = table.copy(deep=True)
    values = table[["estimate", "lower", "upper"]].astype(float)
    valid = (np.isfinite(values.estimate) & values.lower.notna() & values.upper.notna()
             & (values.lower != np.inf) & (values.upper != -np.inf))
    if not (valid | values.isna().all(axis=1)).all():
        raise ValueError("interval rows require a finite estimate and valid bounds, "
                         "or are entirely missing")
    if (values.loc[valid, "lower"] > values.loc[valid, "upper"]).any():
        raise ValueError("interval lower bound must not exceed upper bound")
    if target == "benchmark" and ((values.loc[valid, "lower"] > values.loc[valid, "estimate"])
            | (values.loc[valid, "estimate"] > values.loc[valid, "upper"])).any():
        raise ValueError("interval bounds must contain the benchmark estimate")
    table[["estimate", "lower", "upper"]] = values
    return table


def validate_production_estimates(result, config):
    """Reject a different estimator masquerading as a production interval."""
    supplied = config.factor_beta_intervals.estimate.dropna()
    actual = result.factor_betas.reindex(supplied.index)
    if not np.allclose(supplied, actual, rtol=1e-9, atol=1e-12):
        raise ValueError("production interval estimates must match reported production betas")


def portfolio_beta_interval_table(result, config):
    """Align by factor identity and convert to dollars using the reporting denominator."""
    table = config.factor_beta_intervals.reindex(result.factor_betas.index).copy()
    production = config.beta_interval_target == "production"
    if production:
        validate_production_estimates(result, config)
    table.insert(0, "production_beta", result.factor_betas)
    estimate_column = "interval_estimate" if production else "benchmark_beta"
    table = table.rename(columns={"estimate": estimate_column})
    table["confidence"] = config.beta_interval_confidence
    table["method" if production else "benchmark"] = config.beta_interval_label
    table["target"] = config.beta_interval_target
    table["interval_kind"] = config.beta_interval_kind
    table["status"] = table.get("status", pd.Series(index=table.index, dtype=object))
    default_status = pd.Series(np.where(table[estimate_column].notna(),
        "approximate_conditional", "Not supplied"), index=table.index)
    open_bounds = table[estimate_column].notna() & (
        ~np.isfinite(table.lower) | ~np.isfinite(table.upper))
    default_status.loc[open_bounds] = "Unbounded"
    table["status"] = table.status.fillna(default_status)
    for column in ("production_beta", estimate_column, "lower", "upper"):
        table[f"dollar_{column}"] = table[column] * result.metadata["reporting_denominator"]
    types = result.metadata.get("factor_return_types", {})
    table["unit"] = ["return / index point" if types.get(f) == "DIFFERENCE"
                     else "return / log change" for f in table.index]
    return table


def overlay_beta_intervals(ax, intervals, confidence, scale=1., label="WLS-HAC benchmark",
                           target="benchmark", kind="confidence"):
    """Display bounds and correctly placed points without changing production bars."""
    bars = ax.containers[0]
    y = np.array([bar.get_y() + bar.get_height()/2 for bar in bars])
    valid = np.isfinite(intervals[["estimate", "lower", "upper"]]).all(axis=1).to_numpy()
    points = np.isfinite(intervals.estimate).to_numpy()
    open_bounds = points & ~valid
    selected = intervals.loc[valid]
    center = selected.estimate.to_numpy() * scale
    color = "#B56910"
    if target == "production":
        # Bootstrap percentiles need not contain the original point estimate.
        # Draw bounds independently; never relocate the production dot to their midpoint.
        lower, upper = selected.lower.to_numpy()*scale, selected.upper.to_numpy()*scale
        ax.hlines(y[valid], lower, upper, color=color, linewidth=1.3, zorder=5)
        ax.vlines(lower, y[valid]-.08, y[valid]+.08, color=color, linewidth=1.1, zorder=5)
        ax.vlines(upper, y[valid]-.08, y[valid]+.08, color=color, linewidth=1.1, zorder=5)
        ax.plot(intervals.estimate.to_numpy()[points]*scale, y[points],
                'o', color='#19364A', markersize=4, zorder=6)
        for annotation in ax.texts:
            if isinstance(annotation, Annotation) and annotation.anncoords == 'offset points':
                text_x, text_y = annotation.get_position()
                annotation.set_position((text_x, text_y+7))
            elif annotation.get_transform() == ax.transData:
                text_x, text_y = annotation.get_position()
                annotation.set_position((text_x, text_y-.26))
    else:
        errors = np.vstack([selected.estimate-selected.lower,
                            selected.upper-selected.estimate])*scale
        ax.errorbar(center, y[valid] + .18, xerr=errors, fmt="D", color=color,
                    markersize=3.5, capsize=2.5, linewidth=1.2, zorder=5)
    ax.margins(x=.2)
    handles = [Patch(facecolor="#315B7A", label="Production exposure"),
                       Line2D([], [], color=color, marker="o" if target == "production" else "D",
                              markerfacecolor='#19364A' if target == "production" else color,
                              markersize=4,
                              label=f"{label}: {confidence:.0%} "
                                    + ("stability range" if kind == "stability" else "CI"))]
    if open_bounds.any():
        handles.append(Line2D([], [], linestyle='none', label='Open bounds: see interval table'))
    ax.legend(handles=handles,
              loc="lower left", bbox_to_anchor=(0., 1.03 if open_bounds.any() else 1.07),
              fontsize=8, frameon=False)


def beta_interval_pages(result, config):
    """Show the full portfolio interval table immediately after the exposure page."""
    from qis.portfolio.stress._figures import _page, _table, _footnotes

    table = portfolio_beta_interval_table(result, config)
    currency = result.metadata["reference_currency"]
    confidence = f"{config.beta_interval_confidence:.0%}"
    production = config.beta_interval_target == "production"
    interval_name = "bootstrap stability" if config.beta_interval_kind == "stability" else "CI"
    for offset in range(0, len(table), 16):
        part = table.iloc[offset:offset+16]
        title = f"Portfolio factor exposure uncertainty ({confidence} {interval_name})"
        if len(table) > 16:
            title += f" - {offset//16+1}"
        fig = _page(result, config, 5, title,
                    f"{config.beta_interval_label}; pointwise intervals. "
                    "Production betas and stress P&L remain unchanged.")
        columns = ["production_beta", "lower", "upper"] if production else [
            "production_beta", "benchmark_beta", "lower", "upper"]
        display = part[columns].apply(
            lambda column: column.map(lambda x: f"{x:+.4f}" if pd.notna(x) else "Unavailable"))
        point_columns = (["Production beta"] if production
                         else ["Production beta", "Benchmark beta"])
        display.columns = point_columns + [f"{confidence} lower", f"{confidence} upper"]
        display[f"{confidence} dollar interval\n({currency} millions)"] = [
            f"[{lo/1e6:+.2f}, {hi/1e6:+.2f}]" if pd.notna(lo) else "Unavailable"
            for lo, hi in zip(part.dollar_lower, part.dollar_upper)]
        display["Status"] = part.status.replace({"approximate_conditional": "Approximate",
            "weak_effective_support": "Weak support", "unavailable": "Unavailable"})
        display.index = [str(config.factor_labels.get(f, f)).replace("\n", " ")
                         for f in display.index]
        _table(fig.add_axes([.04, .35, .92, .50]), display, first=.20, fontsize=10)
        _footnotes(fig, [
            "LOG beta: portfolio return per unit log factor change; DIFFERENCE beta: return "
            "per index point. Dollar bounds use the same reporting denominator as the stress "
            "report; for a financed account this includes borrowing through net equity.",
            ("These are production-estimator stability ranges; nominal confidence coverage is "
             "not established. They are not simultaneous intervals or future-P&L prediction bands."
             if config.beta_interval_kind == "stability" else
             "These are pointwise estimation intervals, not simultaneous intervals or "
             "future-P&L prediction bands. The caller owns confidence-coverage validation."),
            "Conditional market/residual risk bands on sensitivity pages describe a different "
            "source of uncertainty. Missing or excluded estimates are marked unavailable. "
            "Inf denotes an open bound. " + (
                "Exposure charts show estimates only; all supplied intervals are retained here."
                if config.beta_interval_display == "table" else
                "Chart whiskers show only finite intervals."),
            *config.beta_interval_notes], y=.29, width=185)
        yield title, fig
