"""QIS plotting pages for already-computed portfolio stress results."""

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator, FuncFormatter

from qis.plots.bars import plot_bars
from qis.plots.table import plot_df_table
from qis.plots.derived.clustering import plot_clusters
from qis.utils.np_ops import covar_to_corr


INK = "#18354B"
BLUE = "#315B7A"
RED = "#A64045"


def _page(result, config, number, title, subtitle):
    """Create a consistent landscape canvas with dated currency/denominator footers."""
    fig = plt.figure(figsize=(16.54, 11.69), facecolor="white")
    fig.text(0.04, 0.955, title, fontsize=21, weight="bold", color=INK)
    fig.text(0.04, 0.917, textwrap.fill(subtitle, 160), fontsize=10, color=BLUE)
    meta = result.metadata
    footer = (
        f"{config.title} | {config.model_label} | Positions {meta['valuation_date'][:10]} | "
        f"Risk {meta['risk_date'][:10]} | {meta['reference_currency']} | "
        f"{meta['denominator_label']} {meta['reporting_denominator']:,.2f}"
    )
    fig.text(0.04, 0.025, textwrap.shorten(footer, 220), fontsize=8, color=BLUE)
    fig.text(0.96, 0.025, str(number), ha="right", fontsize=9, color=INK)
    return fig


def _note(fig, text):
    """Place wrapped variable definitions in a reserved footer band."""
    fig.text(0.04, 0.07, textwrap.fill(text, 170), fontsize=8.5, color=BLUE, va="center")


def _empty(ax, message):
    """Label unavailable diagnostics without inventing fitted information."""
    ax.axis("off")
    ax.text(0.5, 0.5, textwrap.fill(message, 75), ha="center", va="center", fontsize=12, color=BLUE)


def _table(ax, data, title="", first=0.25, fontsize=9):
    """Render an already-formatted table through the existing QIS table implementation."""
    if data.empty:
        _empty(ax, "No applicable observations supplied.")
        return
    plot_df_table(
        data,
        ax=ax,
        title=title,
        fontsize=fontsize,
        header_color=INK,
        header_text_color="white",
        row_colors=["#F0F4F7", "white"],
        edge_color="white",
        linewidth=0.5,
        left_aligned_first_col=True,
        col_widths=[first] + [(1 - first) / len(data.columns)] * len(data.columns),
    )
    for table in ax.tables:
        for (row, col), cell in table.get_celld().items():
            cell.PAD = 0.035
            if row > 0 and col > 0:
                cell.get_text().set_ha("center")


def _bars(ax, values, title, percent=False):
    """Use canonical QIS bars with signed labels in explicit currency or ratio units."""
    if values.empty:
        _empty(ax, "No applicable observations supplied.")
        return
    values = values.copy()
    values.index = [
        "\n".join(
            textwrap.wrap(str(item.date()) if isinstance(item, pd.Timestamp) else str(item), 30)
        )
        for item in values.index
    ]
    colors = [RED if value < 0 else BLUE for value in values]
    plot_bars(
        values,
        ax=ax,
        is_horizontal=True,
        stacked=False,
        colors=colors,
        series_color=BLUE,
        title=title,
        fontsize=9,
        add_bar_values=False,
        legend_loc=None,
        x_rotation=0,
        xvar_format="{:+.1%}" if percent else "{:+,.1f}",
        yvar_format="{:+.1%}" if percent else "{:+,.1f}",
    )
    fmt = "{:+.2%}" if percent else "{:+,.2f}"
    for container in ax.containers:
        ax.bar_label(
            container, labels=[fmt.format(p.get_width()) for p in container], padding=4, fontsize=8
        )
    ax.axvline(0, color="#7B8D9B", lw=0.6)
    ax.grid(axis="x", color="#E4EAF0", linewidth=0.5)
    ax.margins(x=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis_format = "{:+.1%}" if percent else "{:+,.1f}"
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: axis_format.format(value)))


def _scenario_page(result, config, number, title, valuation, subtitle):
    """Show scenario totals and original-holding contributors without repricing."""
    fig = _page(result, config, number, title, subtitle)
    if valuation is None:
        _empty(
            fig.add_axes([0.08, 0.2, 0.84, 0.62]), "This scenario set was not supplied or computed."
        )
        return fig
    denominator = result.metadata["reporting_denominator"]
    currency = result.metadata["reference_currency"]
    scale = 1e6 if denominator >= 1e6 else 1.0
    unit = f"{currency} millions" if scale == 1e6 else currency
    rows = valuation.pnl.index[:12]
    if number == 3:
        rows = result.historical_ranking.index[:12]
    pnl = valuation.portfolio_pnl.loc[rows]
    labels = result.metadata.get("scenario_descriptions", {}) if number != 3 else {}
    overrides = result.metadata.get("scenario_completion_overrides", {}) if number != 3 else {}
    display_labels = {
        key: (str(labels.get(str(key), key)) + (" *" if str(key) in overrides else ""))
        if number != 3 else key
        for key in rows
    }
    pnl = pnl.rename(index=display_labels)
    _bars(fig.add_axes([0.16, 0.57, 0.32, 0.28]), pnl / scale, f"Portfolio P&L ({unit})")
    _bars(
        fig.add_axes([0.63, 0.57, 0.30, 0.28]),
        pnl / denominator,
        f"P&L / {result.metadata['denominator_label']}",
        percent=True,
    )
    cells = []
    for label in rows:
        values = valuation.pnl.loc[label]
        selected = values.abs().sort_values(ascending=False, kind="stable").head(10).index
        row = {}
        for rank, holding_id in enumerate(selected, 1):
            name = result.positions.loc[
                holding_id,
                "metadata:short_name" if "metadata:short_name" in result.positions else "name",
            ]
            display = "\n".join(textwrap.wrap(str(name), width=18, max_lines=2, placeholder="..."))
            row[str(rank)] = f"{display}\n{values.loc[holding_id] / denominator:+.2%}"
        cells.append(row)
    frame = pd.DataFrame(cells, index=rows).fillna("").rename(index=display_labels)
    frame.index = [
        str(item.date())
        if isinstance(item, pd.Timestamp)
        else "\n".join(textwrap.wrap(str(item), 25))
        for item in frame.index
    ]
    _table(
        fig.add_axes([0.04, 0.14, 0.92, 0.37]),
        frame,
        "Ten largest absolute holding contributions; signed percentage points of denominator",
        first=0.18,
        fontsize=8,
    )
    _note(
        fig,
        "P&L = stressed value minus observed value. Percentages use the stated reporting "
        "denominator. Current holdings and fitted betas are held fixed in every scenario. "
        "Contributor rank uses absolute holding P&L; gains remain positive. "
        "At most 12 scenarios are displayed; complete values and original IDs are exported. "
        + ("* Caller-pinned completion is retained on both pages; modes are exported."
           if overrides else ""),
    )
    return fig


def _currency_scale(result):
    """Keep monetary chart axes readable while preserving explicit value units."""
    scale = 1e6 if result.metadata["reporting_denominator"] >= 1e6 else 1.0
    currency = result.metadata["reference_currency"]
    return scale, f"{currency} millions" if scale == 1e6 else currency


def _risk_page(result, config):
    """Present betas, currency exposures and canonical current local risk."""
    fig = _page(
        result,
        config,
        4,
        "Current exposures and local risk",
        "Dollar sensitivities come from the current payoff Jacobian; "
        "zero or negative derivative marks do not remove their risk.",
    )
    order = result.factor_exposures.abs().sort_values(ascending=False).index
    labels = {key: config.factor_labels.get(key, key) for key in order}
    scale, unit = _currency_scale(result)
    _bars(
        fig.add_axes([0.13, 0.45, 0.33, 0.39]),
        result.factor_betas.loc[order].rename(index=labels),
        "Portfolio factor betas",
    )
    _bars(
        fig.add_axes([0.63, 0.45, 0.30, 0.39]),
        result.factor_exposures.loc[order].rename(index=labels) / scale,
        f"Factor sensitivities ({unit})",
    )
    risk = (
        result.risk.rename(
            index={
                "annual_total_vol": "Annual total vol (supplied covariance)",
                "annual_systematic_vol": "Annual systematic vol",
                "annual_residual_vol": "Annual residual vol",
                "annual_factor_model_vol": "Annual factor-model total vol",
                "local_vol_horizon": "Total local vol at stated horizon",
            }
        )
        .map(lambda x: f"{x:.2%}")
        .to_frame("Volatility")
    )
    _table(fig.add_axes([0.05, 0.16, 0.48, 0.22]), risk, first=0.78)
    groups = result.factor_group_exposures.map(lambda x: f"{x:,.2f}")
    groups = groups.rename(
        columns={"exposure_sum": "Exposure sum", "split_bump_exposure": "Split-bump sensitivity"}
    )
    _table(fig.add_axes([0.61, 0.20, 0.34, 0.13]), groups, first=0.32)
    _note(
        fig,
        "Factor beta = current factor currency sensitivity / reporting denominator. "
        "Exposure sum adds member-factor sensitivities; split-bump sensitivity weights them "
        "by the declared family allocation. These are local derivatives. Volatility uses "
        "annualized log covariance and aggregated shared residual exposures, without "
        "renormalizing signed holdings. Covariance and factor-model risk remain separate views.",
    )
    return fig


def _contributor_page(result, config):
    """Show the most influential original holdings for the six largest factor exposures."""
    fig = _page(
        result,
        config,
        5,
        "Holding contributions to factor exposures",
        "Six largest absolute net portfolio factor sensitivities; "
        "ten largest absolute holding sensitivities per factor.",
    )
    factors = result.factor_exposures.abs().sort_values(ascending=False).head(6).index
    scale, unit = _currency_scale(result)
    grid = fig.add_gridspec(
        2, 3, left=0.13, right=0.95, top=0.84, bottom=0.15, hspace=0.5, wspace=0.75
    )
    for i, factor in enumerate(factors):
        values = result.holding_factor_exposures[factor]
        selected = values.abs().sort_values(ascending=False).head(10).index
        values = values.loc[selected]
        values.index = result.positions.loc[selected, "name"]
        _bars(
            fig.add_subplot(grid[i // 3, i % 3]),
            values / scale,
            f"{config.factor_labels.get(factor, factor)} ({unit})",
        )
    _note(
        fig,
        "Each bar is the original holding's current currency sensitivity to one unit "
        "of factor log return. Synthetic option legs remain grouped under the source holding. "
        "Shared underlying response sensitivities are aggregated before portfolio risk.",
    )
    return fig


def _grid_page(result, config):
    """Draw deterministic payoff curves and only analytically supported existing bands."""
    fig = _page(
        result,
        config,
        6,
        "Factor sensitivity curves",
        "Each grid point revalues the full holding payoff through the same scenario engine.",
    )
    keys = config.selected_grids or tuple(result.grids)[:4]
    grid = fig.add_gridspec(
        2, 2, left=0.09, right=0.95, top=0.85, bottom=0.16, hspace=0.55, wspace=0.24
    )
    for i in range(4):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        if i >= len(keys):
            _empty(ax, "No additional sensitivity grid supplied.")
            continue
        key = keys[i]
        summary = result.grid_summaries[key]
        try:
            x = np.asarray(summary.index, dtype=float)
        except (TypeError, ValueError):
            x = np.arange(len(summary))
            ax.set_xticks(x, [str(item) for item in summary.index], rotation=30)
        ax.plot(x, summary.portfolio_return, color=BLUE, lw=2)
        if "lower_bound" in summary:
            ax.fill_between(x, summary.lower_bound, summary.upper_bound, alpha=0.14, color=BLUE)
        ax.axhline(0, color="#888888", lw=0.6)
        groups = result.metadata.get("factor_groups", {})
        requested_key = result.grid_metadata.loc[key, "requested_keys"]
        group = groups.get(requested_key)
        title = key
        axis_label = summary.index.name or "Requested bump (supplied grid index)"
        if group is not None:
            title = " + ".join(group["members"]) + " (split total bump)"
            axis_label = f"Total {group['label'] or requested_key} family bump"
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_xlabel(axis_label, fontsize=9)
        ax.set_ylabel("P&L / reporting denominator", fontsize=9)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
        status = result.grid_metadata.loc[key, "band_status"]
        ax.text(
            0.02,
            0.98,
            "\n".join(textwrap.wrap(status, 66)),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color=BLUE,
        )
    _note(
        fig,
        "Factor shocks are log returns. For a total simple family bump x with weights w, "
        "each member receives log(1 + w*x); an equal n-member split is log(1 + x/n). "
        "Conditional completion fixes all members jointly. Shading, when present, is a "
        f"{result.metadata['confidence']:.0%} baseline Gaussian conditional-factor plus residual "
        f"band over {result.metadata['horizon_years']:.6g} years. Derivative curves have no "
        "Gaussian band or quadratic approximation.",
    )
    return fig


def _beta_page(result, config):
    """Render supplied estimation diagnostics beside the fitted response betas."""
    fig = _page(
        result,
        config,
        7,
        "Underlying response betas and fit diagnostics",
        "Fit diagnostics belong to the caller's estimator; absent R-squared "
        "is explicitly unavailable.",
    )
    responses = result.response_exposures.abs().sort_values(ascending=False).head(20).index
    factors = result.factor_exposures.abs().sort_values(ascending=False).head(12).index
    frame = result.factor_loadings.loc[responses, factors].map(lambda x: f"{x:.2f}")
    frame = frame.rename(columns=config.factor_labels)
    if config.response_diagnostics is not None and "r2" in config.response_diagnostics:
        frame["R-squared"] = config.response_diagnostics.r2.reindex(responses).map(
            lambda x: "Unavailable" if pd.isna(x) else f"{x:.1%}"
        )
    else:
        frame["R-squared"] = "Unavailable"
    diagnostics = config.response_diagnostics
    if diagnostics is not None:
        for column, label in (
            ("annual_systematic_vol", "Systematic vol"),
            ("annual_residual_vol", "Residual vol"),
            ("annual_factor_model_vol", "Model total vol"),
        ):
            if column in diagnostics:
                frame[label] = diagnostics[column].reindex(responses).map(
                    lambda x: "Unavailable" if pd.isna(x) else f"{x:.1%}"
                )
        if "name" in diagnostics:
            frame.index = [
                textwrap.shorten(str(diagnostics.loc[key, "name"]), 30)
                if key in diagnostics.index and pd.notna(diagnostics.loc[key, "name"]) else key
                for key in responses
            ]
    frame.columns = ["\n".join(textwrap.wrap(str(key), 11)) for key in frame.columns]
    _table(fig.add_axes([0.04, 0.2, 0.92, 0.62]), frame, first=0.22, fontsize=8)
    _note(
        fig,
        "Rows are fitted response/proxy identities, not option marks. Columns are "
        "underlying log-return factor betas. R-squared is supplied regression explanatory "
        "power; it is never inferred from volatility. Supplied annual systematic/residual "
        "vols describe unit response exposure, not portfolio weights. Up to 20 responses "
        "and twelve factors "
        "are displayed, ranked by current exposure; complete matrices and diagnostics "
        "are exported.",
    )
    return fig


def _clusters_page(result, config):
    """Render fitted tree topology using the existing QIS composite cluster plot."""
    fig = _page(
        result,
        config,
        8,
        "Fitted cluster structure",
        "Only caller-supplied fitted linkage and membership are displayed; no clustering "
        "or covariance estimation occurs in this report.",
    )
    if not config.cluster_memberships:
        _empty(
            fig.add_axes([0.08, 0.22, 0.84, 0.55]), "Fitted cluster diagnostics were not supplied."
        )
    else:
        keys = list(config.cluster_memberships)
        grid = fig.add_gridspec(
            len(keys), 3, left=0.08, right=0.96, top=0.83, bottom=0.16, hspace=0.35, wspace=0.4
        )
        axes = {key: fig.add_subplot(grid[i, :2]) for i, key in enumerate(keys)}
        table_ax = fig.add_subplot(grid[:, 2])
        plot_clusters(
            config.cluster_memberships,
            config.cluster_linkages,
            config.cluster_cutoffs,
            axes=axes,
            table_ax=table_ax,
            fontsize=8,
            show_distance=True,
        )
    _note(
        fig,
        "Leaves preserve actual fitted response IDs and supplied linkage order. "
        "Merge distances and cluster cutoffs are estimator diagnostics, not probabilities "
        "of joint losses. Missing trees are reported as unavailable.",
    )
    return fig


def _methodology_page(result, config):
    """Display fitted correlation and the valuation/conditional decision conventions."""
    fig = _page(
        result,
        config,
        9,
        "Correlation and scenario methodology",
        "A scenario vector is an explicit hypothetical realization, not a probability forecast.",
    )
    cov = result.factor_covariance
    positive = np.diag(cov) > 0
    correlation = pd.DataFrame(np.nan, index=cov.index, columns=cov.columns)
    if positive.any():
        correlation.loc[positive, positive] = covar_to_corr(cov.loc[positive, positive])
    ax = fig.add_axes([0.13, 0.3, 0.4, 0.48])
    artist = ax.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [config.factor_labels.get(key, key) for key in cov.index]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    fig.colorbar(artist, ax=ax, fraction=0.05)
    notes = [
        "Funded assets: P&L = observed value * expm1(beta dot factor log shock).",
        "Calls/puts: signed intrinsic payoff at the shocked local quote; no time value.",
        "Futures: signed units * multiplier * quote change; source MTM may be zero.",
        "Derivatives: stressed value = observed MTM + model payoff change from zero shock.",
        "FX: remove FX from a reference-currency fitted response before applying local strikes; "
        "convert payoff once at stressed FX.",
        "Conditional scenarios: fix all supplied factors jointly; use the fitted covariance "
        "to complete the remaining factors. Explicit zero anchors remain zero.",
        "Historical replay: apply each complete monthly factor vector to current holdings, "
        "then rank exact portfolio P&L.",
        "Nonlinear attribution: local factor components plus a separate payoff adjustment.",
    ]
    y = 0.82
    for note in notes:
        wrapped = textwrap.fill(note, 69)
        fig.text(0.6, y, wrapped, va="top", fontsize=10, color=INK)
        y -= 0.028 * (wrapped.count("\n") + 1) + 0.028
    _note(
        fig,
        "Correlation is annual covariance divided by the corresponding factor standard "
        "deviations. Zero-variance pairs are unavailable. Deterministic scenarios have no "
        "assigned probability; current local volatility does not capture payoff jumps or "
        "unobserved contract path states.",
    )
    return fig


def _coverage_page(result, config):
    """Summarize explicit coverage and original-position reconciliation."""
    fig = _page(
        result,
        config,
        10,
        "Coverage and valuation reconciliation",
        "Every original holding remains in valuation and risk mapping. "
        "Application-specific delivery, margin and collateral rules remain caller-owned.",
    )
    positions = result.positions
    historical_count = 0 if result.historical is None else len(result.historical.pnl)
    excluded_count = (
        (result.historical_coverage.status != "eligible").sum()
        if not result.historical_coverage.empty
        else 0
    )
    values = pd.Series(
        {
            "Original holdings": f"{len(positions):,}",
            "Vanilla legs": f"{len(result.leg_terms):,}",
            "Fitted response identities": f"{len(result.factor_loadings):,}",
            "Observed portfolio value": f"{positions.observed_mtm.sum():,.2f}",
            "Model baseline value": f"{positions.model_baseline.sum():,.2f}",
            "Constant reference-currency basis offset": f"{positions.basis_offset.sum():,.2f}",
            "Eligible historical months": f"{historical_count:,}",
            "Excluded historical rows": f"{excluded_count:,}",
        }
    ).to_frame("Value")
    _table(fig.add_axes([0.05, 0.47, 0.54, 0.35]), values, first=0.76)
    coverage = positions.groupby("coverage", sort=False).size().to_frame("Holdings")
    coverage.index = ["\n".join(textwrap.wrap(item, 48)) for item in coverage.index]
    coverage_height = min(0.35, 0.045 * (len(coverage) + 1))
    _table(fig.add_axes([0.64, 0.82 - coverage_height, 0.31, coverage_height]),
           coverage, first=0.82, fontsize=8)
    notes = list(config.notes) or ["No additional application-specific notes supplied."]
    notes += [
        "Systematic/residual risk shares the underlying response across stock and derivative legs.",
        "Zero-shock P&L is zero. The observed-minus-model basis offset is constant in reference "
        "currency and is not a stressed option premium or available liquidation cash.",
        "Option time value, volatility surfaces, barrier paths and delivery obligations are "
        "outside primitive intrinsic valuation. Composite approximations declare their coverage.",
        "Historical replay describes today's holdings under old factor returns, not realized "
        "client performance. No lending value, credit limit or liquidation trigger is inferred.",
    ]
    note_text = "\n\n".join(textwrap.fill(note, 155) for note in notes)
    # Fit caller notes inside the reserved region on the fixed ten-page template.
    line_count = note_text.count("\n") + 1
    note_fontsize = min(10.0, 0.265 * 11.69 * 72 / (1.2 * line_count))
    fig.text(
        0.05,
        0.38,
        note_text,
        va="top",
        fontsize=note_fontsize,
        linespacing=1.2,
        color=INK,
    )
    _note(
        fig,
        "The complete position audit, response Jacobian, synthetic leg terms, resolved "
        "shocks, scenario values and historical exclusions are retained in the table exports. "
        "The manifest records conventions, source snapshot dates and artifact hashes.",
    )
    return fig


def report_pages(result, config):
    """Yield ten report subjects from completed numerical results, without model access."""
    title = "Requested factor stress scenarios"
    yield (
        title,
        _scenario_page(
            result,
            config,
            1,
            title,
            result.valuations["requested"],
            f"Requested completion policy: "
            f"{result.metadata['requested_completion']}. "
            "Values include the full supplied holding payoff.",
        ),
    )
    title = "Conditional factor stress scenarios"
    yield (
        title,
        _scenario_page(
            result,
            config,
            2,
            title,
            result.valuations.get("conditional"),
            "Supplied anchors are fixed jointly; other factor moves follow "
            "the fitted covariance conditional mean.",
        ),
    )
    title = (
        f"Worst {len(result.historical_ranking)} historical months on current holdings"
        if result.historical is not None
        else "Historical stress on current holdings"
    )
    yield (
        title,
        _scenario_page(
            result,
            config,
            3,
            title,
            result.historical,
            "All eligible complete monthly factor realizations are evaluated "
            "first, then ranked by exact portfolio P&L.",
        ),
    )
    for title, builder in [
        ("Current exposures and local risk", _risk_page),
        ("Holding contributions to factor exposures", _contributor_page),
        ("Factor sensitivity curves", _grid_page),
        ("Underlying response betas and fit diagnostics", _beta_page),
        ("Fitted cluster structure", _clusters_page),
        ("Correlation and scenario methodology", _methodology_page),
        ("Coverage and valuation reconciliation", _coverage_page),
    ]:
        yield title, builder(result, config)
