"""QIS plotting pages for already-computed portfolio stress results."""

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

from qis.plots.bars import plot_bars
from qis.plots.heatmap import plot_heatmap
from qis.portfolio.stress._clusters import (
    compute_cluster_contributions, display_cluster_table, cluster_top_contributors)
from qis.plots.table import plot_df_table
from qis.plots.derived.clustering import plot_clusters
from qis.plots.scatter import plot_scatter
from qis.models.linear.plot_correlations import plot_corr_matrix_from_covar
from qis.portfolio.stress.reporting import _loading_table, _report_heading


INK = "#18354B"
BLUE = "#315B7A"
RED = "#A64045"
FOOTNOTE_FONTSIZE = 9


def _page(result, config, number, title, subtitle):
    """Create a consistent landscape canvas with dated currency/denominator footers."""
    fig = plt.figure(figsize=(16.54, 11.69), facecolor="white")
    title_y, subtitle_y = 0.955, 0.923
    if number == 1:
        heading = fig.text(0.04, 0.978, _report_heading(result, config),
                           fontsize=12, weight="bold", color=BLUE)
        width = heading.get_window_extent(fig.canvas.get_renderer()).width
        heading.set_fontsize(min(12., 12. * fig.bbox.width * .92 / max(width, 1.)))
        title_y, subtitle_y = 0.94, 0.907
    fig.text(0.04, title_y, title, fontsize=21, weight="bold", color=INK)
    fig.text(0.04, subtitle_y, textwrap.fill(subtitle, 160), fontsize=10, color=BLUE)
    meta = result.metadata
    footer = (
        f"{config.title} | {config.model_label} | Positions {meta['valuation_date'][:10]} | "
        f"Risk {meta['risk_date'][:10]} | {meta['reference_currency']} | "
        f"{meta['denominator_label']} {meta['reporting_denominator']:,.2f}"
    )
    fig.text(0.04, 0.025, textwrap.shorten(footer, 220), fontsize=FOOTNOTE_FONTSIZE, color=BLUE)
    fig.text(0.96, 0.025, str(number), ha="right", fontsize=9, color=INK)
    return fig


def _note(fig, text):
    """Place wrapped variable definitions in a reserved footer band."""
    fig.text(0.04, 0.07, textwrap.fill(text, 170), fontsize=FOOTNOTE_FONTSIZE,
             color=BLUE, va="center")


def _empty(ax, message):
    """Label unavailable diagnostics without inventing fitted information."""
    ax.axis("off")
    ax.text(0.5, 0.5, textwrap.fill(message, 75), ha="center", va="center", fontsize=12, color=BLUE)


def _table(ax, data, title="", first=0.25, fontsize=9, widths=None):
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
        col_widths=widths or [first] + [(1 - first) / len(data.columns)] * len(data.columns),
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
    axis_format = "{:+.2%}" if percent else "{:+,.2f}"
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
    denominator_name = "NAV" if result.metadata["all_funded"] else "reporting denominator"
    currency = result.metadata["reference_currency"]
    scale = 1e6 if denominator >= 1e6 else 1.0
    unit = f"{currency} millions" if scale == 1e6 else currency
    rows = valuation.pnl.index[:12]
    if number == 3:
        rows = result.historical_ranking.index[:12]
    pnl = valuation.portfolio_pnl.loc[rows]
    labels = result.metadata.get("scenario_descriptions", {}) if number != 3 else {}
    display_labels = {key: (str(labels.get(str(key), key))) if number != 3 else key for key in rows}
    pnl = pnl.rename(index=display_labels)
    _bars(fig.add_axes([0.16, 0.57, 0.32, 0.28]), pnl / scale, f"Total P&L ({unit})")
    _bars(
        fig.add_axes([0.63, 0.57, 0.30, 0.28]),
        pnl / denominator,
        f"Portfolio P&L (% of {denominator_name})",
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
            display = "\n".join(textwrap.wrap(str(name), width=20, max_lines=2, placeholder="..."))
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
        fig.add_axes([0.04, 0.125, 0.92, 0.32]),
        frame,
        "Top 10 asset contributions by absolute size; signed percentage points of "
        + ("portfolio NAV" if result.metadata["all_funded"] else "reporting denominator"),
        first=0.14,
        fontsize=7.5,
    )
    for ax in fig.axes:
        for table in ax.tables:
            for (r, c), cell in table.get_celld().items():
                if r > 0 and c > 0:
                    value = cell.get_text().get_text().split("\n")[-1]
                    cell.set_facecolor("#F9EDEF" if value.startswith("-") else "#EAF4F1")
    notes = [
        "Each row is ranked independently. Cells show asset name and contribution / full "
        f"portfolio {denominator_name}. The omitted remainder is retained in the "
        "CSV reconciliation."
    ]
    if number != 3:
        for scenario, note in result.metadata.get("scenario_notes", {}).items():
            if scenario not in valuation.factor_log_shocks.index:
                continue
            note = str(note).replace(
                "{shock_kind}",
                "correlated shock"
                if number == 2
                else "direct shock (correlated version on page 2)",
            )
            for factor, shock in valuation.factor_log_shocks.loc[scenario].items():
                note = note.replace("{" + str(factor) + "}", f"{np.expm1(shock):+.2%}")
            if note not in notes:
                notes.append(note)
        notes.append("Correlated-shock methodology and formulas: see Appendix, page 10.")
    _footnotes(fig, notes, y=0.115)
    return fig


def _currency_scale(result):
    """Keep monetary chart axes readable while preserving explicit value units."""
    scale = 1e6 if result.metadata["reporting_denominator"] >= 1e6 else 1.0
    currency = result.metadata["reference_currency"]
    return scale, f"{currency} millions" if scale == 1e6 else currency


def _footnotes(fig, notes, y=0.115, width=220):
    """Wrap complete explanations at the fixed, report-wide readable footnote size."""
    lines = [line for note in notes for line in textwrap.wrap(str(note), width)]
    fig.text(0.04, y, "\n".join(lines), fontsize=FOOTNOTE_FONTSIZE,
             color=INK, va="top", linespacing=1.3)


def _panel_note(fig, x, y, note):
    """Wrap one panel's variable definitions without crossing the adjacent column."""
    fig.text(x, y, textwrap.fill(note, 88), fontsize=FOOTNOTE_FONTSIZE, color=INK, va="top")


def _risk_page(result, config):
    """Restore the annualised risk table and additive family Euler chart."""
    fig = _page(
        result,
        config,
        4,
        f"Portfolio {config.model_name} exposures and risk",
        "Exposure ratios refer to each factor definition. Annualised risk uses the "
        "assigned factor covariance and residual variances.",
    )
    labels = config.factor_labels
    currency = result.metadata["reference_currency"]
    _bars(
        fig.add_axes([0.16, 0.535, 0.32, 0.325]),
        result.factor_betas.rename(index=labels),
        "Weighted factor exposure (ratio)",
    )
    _bars(
        fig.add_axes([0.65, 0.535, 0.29, 0.325]),
        result.factor_exposures.rename(index=labels) / 1e6,
        f"Dollar factor exposure ({currency} m)",
    )
    _panel_note(
        fig,
        0.055,
        0.478,
        "e_f = sum_i w_i beta_if, where w_i = MTM_i / NAV. This is the portfolio sensitivity "
        "to factor f: a small +1% factor move contributes approximately e_f% of NAV."
        if result.metadata["all_funded"]
        else "e_f = factor dollar sensitivity / reporting denominator. Sensitivities use the "
        "current payoff Jacobian, including zero-mark futures and shared underlying responses.",
    )
    _panel_note(
        fig,
        0.565,
        0.478,
        ("Dollar exposure = NAV x e_f. " if result.metadata["all_funded"]
         else "Dollar exposure = reporting denominator x e_f. ")
        + "A small +1% factor move contributes approximately "
        "1% of this amount to portfolio P&L; it is not invested cash.",
    )
    risk = result.report_diagnostics["Annualised portfolio risk"]
    formatted = pd.DataFrame(
        {
            "Annual vol": risk.annual_vol.map("{:.2%}".format),
            "Dollar vol (m)": (risk.dollar_vol / 1e6).map("{:.2f}".format),
            "Variance share": risk.variance_share.map("{:.1%}".format),
            "Euler vol (%)": risk.euler_vol.map("{:+.2%}".format),
        }
    )
    _table(
        fig.add_axes([0.045, 0.205, 0.50, 0.17]),
        formatted,
        "Annualised portfolio risk",
        first=0.24,
        fontsize=9.5,
    )
    family = result.report_diagnostics["Family Euler volatility"]
    values = family.euler_vol.copy()
    values.index = family.label
    _bars(
        fig.add_axes([0.65, 0.205, 0.29, 0.19]),
        values,
        "Factor Euler volatility contributions (by family)",
        percent=True,
    )
    _panel_note(
        fig,
        0.055,
        0.146,
        "Systematic variance = e.T Sigma e; idio variance = sum_i w_i^2 residual_var_i. "
        "Total vol is the square root of their sum. Dollar vol = NAV x vol. Variance shares "
        "and Euler vol contributions allocate total risk; standalone vols do not add."
        if result.metadata["all_funded"]
        else "w_j = shared response dollar sensitivity / N; N is the reporting denominator. "
        "Systematic variance = e.T Sigma e; residual variance = sum_j w_j^2 residual_var_j. "
        "Total vol = sqrt(their sum); dollar vol = N x vol. Euler contributions add; "
        "standalone vols do not.",
    )
    _panel_note(
        fig,
        0.565,
        0.146,
        "Factor RC_f = e_f (Sigma e)_f / total portfolio vol. A family sums its members' "
        "signed RC_f, without shock-split weights. Negative values reduce risk. Families "
        "sum to the systematic Euler contribution in the table.",
    )
    if not np.isclose(result.risk.annual_total_vol, result.risk.annual_factor_model_vol):
        _footnotes(
            fig,
            [
                "The table uses the factor-model total. Supplied asset-covariance "
                "volatility differs and is retained separately in Current risk."
            ],
            y=0.075,
        )
    return fig


def _contributor_factors(result):
    """Return the shared six-family display order, including stable ties."""
    rc = result.report_diagnostics["Reported factor groups"].euler_vol
    return tuple(rc.loc[rc.ne(0)].abs().sort_values(
        ascending=False, kind="stable").head(6).index)


def _contributor_page(result, config):
    """Allocate the largest factor Euler contributions to original holdings."""
    fig = _page(
        result,
        config,
        5,
        "Largest factor exposures: asset risk contributors",
        "Factor families ranked by absolute summed Euler contribution. Assets ranked by absolute "
        "Euler contribution to total annual portfolio volatility.",
    )
    groups = result.report_diagnostics["Reported factor groups"]
    rc = groups.euler_vol
    factors = _contributor_factors(result)
    grid = fig.add_gridspec(
        2, 3, left=0.11, right=0.96, top=0.86, bottom=0.20, hspace=0.48, wspace=0.68
    )
    for i, factor in enumerate(factors):
        values = result.report_diagnostics["Holding reported factor Euler volatility"][factor]
        selected = values.abs().sort_values(ascending=False, kind="stable").head(10).index
        values = values.loc[selected].copy()
        name_col = "metadata:short_name" if "metadata:short_name" in result.positions else "name"
        values.index = result.positions.loc[selected, name_col]
        _bars(
            fig.add_subplot(grid[i // 3, i % 3]),
            values,
            f"{config.factor_labels.get(factor, groups.loc[factor, 'label'])}\n"
            f"Exposure {groups.loc[factor, 'factor_beta']:+.2f}; factor Euler {rc[factor]:+.2%}",
            percent=True,
        )
    _footnotes(
        fig,
        [
            "Factor Euler RC_f = e_f x (Sigma e)_f / sigma_p: portfolio factor beta times "
            "marginal volatility. Sigma is annual factor covariance; sigma_p is total model "
            "volatility. Values are percentage points of annual volatility; negative terms "
            "reduce risk.",
            "Additivity: all factor Euler terms sum to systematic variance / sigma_p. Adding "
            "residual variance / sigma_p gives total volatility sigma_p. Family exposures and "
            "holding Euler terms sum all member factors without "
            "shock-allocation weights. Holding contributions sum to the family Euler term.",
            "Six largest absolute factor/family Euler sums and ten largest absolute holding "
            "contributions per factor are shown; displayed subsets need not sum to full totals. "
            "Complete tables are exported. Funded factor beta e_f = "
            "sum_i (MTM_i / NAV) x beta_if."
            if result.metadata["all_funded"]
            else "Six largest absolute factor/family Euler sums and ten largest absolute holding "
            "contributions per factor are shown; displayed subsets need not sum to full totals. "
            "Derivative betas use current payoff Jacobians and shared responses; these are local "
            "risk contributions. Complete tables are exported."
        ],
        y=0.13,
        width=185,
    )
    return fig


def _grid_page(result, config):
    """Show exact grids and quadratic fits with conditional one/two-sigma local-risk bands."""
    funded = result.metadata["all_funded"]
    months = result.metadata["horizon_years"] * 12
    keys = config.selected_grids
    ranked = _contributor_factors(result)
    if not keys:
        known = set(result.factor_exposures.index) | set(
            result.report_diagnostics["Reported factor groups"].index)
        by_request = {row.requested_keys: key for key, row in result.grid_metadata.iterrows()}
        keys = tuple(factor if factor in result.grids else by_request[factor]
                     for factor in ranked if factor in result.grids or factor in by_request)
        if not keys and ranked and not set(result.grids).issubset(known):
            keys = tuple(result.grids)[:6]
    subtitle = "Correlated shocks; factor-return ranges shown on each axis."
    if keys and keys == ranked:
        subtitle += " Same factor order as the contributor page."
    if any("lower_1sigma" in result.grid_summaries[key] for key in keys):
        subtitle += f" Shading: conditional +/-1sigma and +/-2sigma ({months:g} month)."
    else:
        subtitle += " Deterministic payoff curves; local-risk bands disabled or unavailable."
    fig = _page(result, config, 6, "Sensitivity to largest factor exposures", subtitle)
    grid = fig.add_gridspec(
        2, 3, left=0.07, right=0.97, top=0.83, bottom=0.275, hspace=0.58, wspace=0.38
    )
    for i in range(6):
        ax = fig.add_subplot(grid[i // 3, i % 3])
        if i >= len(keys):
            ax.set_axis_off()
            continue
        key = keys[i]
        summary = result.grid_summaries[key]
        numeric = True
        try:
            x = np.asarray(summary.index, dtype=float)
        except (TypeError, ValueError):
            numeric = False
            x = np.arange(len(summary), dtype=float)
        requested = result.grid_metadata.loc[key, "requested_keys"]
        group = result.metadata.get("factor_groups", {}).get(requested)
        title = config.factor_labels.get(requested, requested)
        xlabel = title + " factor return"
        if group is not None:
            title = group["label"] or requested
            equal = np.allclose(group["weights"], 1.0/len(group["members"]))
            split = "equal split" if equal else "/".join(
                f"{weight:.0%}" for weight in group["weights"])
            xlabel = f"Total {group['label'] or requested} family bump ({split})"
        points = pd.DataFrame({"factor_return": x, "portfolio_return": summary.portfolio_return})
        plot_scatter(
            points,
            x="factor_return",
            y="portfolio_return",
            xlabel=xlabel,
            ylabel=("Portfolio return (% of NAV)" if funded
                    else "Portfolio P&L (% of reporting denominator)"),
            full_sample_order=0,
            add_universe_model_label=False,
            add_universe_model_prediction=False,
            add_universe_model_ci=False,
            ci=None,
            legend_loc=None,
            full_sample_color=BLUE,
            xvar_format="{:+.0%}" if numeric else "{:.0f}",
            yvar_format="{:+.0%}",
            markersize=13,
            fontsize=8,
            ax=ax,
        )
        band_handles = []
        if "lower_1sigma" in summary:
            outer = ax.fill_between(
                x, summary.lower_2sigma, summary.upper_2sigma,
                color=BLUE, alpha=0.12, zorder=0, label="Conditional +/-2sigma",
            )
            inner = ax.fill_between(
                x, summary.lower_1sigma, summary.upper_1sigma,
                color=BLUE, alpha=0.25, zorder=1, label="Conditional +/-1sigma",
            )
            band_handles = [inner, outer]
            zero = np.flatnonzero(np.isclose(x, 0.0, rtol=0.0, atol=1e-12)) if numeric else []
            sigma = summary.conditional_vol_horizon.iloc[zero[0]] if len(zero) == 1 else np.nan
            widths = [f"{multiple * sigma:.2%}" if np.isfinite(sigma) else "n/a"
                      for multiple in (1, 2)]
            ax.text(0.98, 0.03, f"1sigma band +/- {widths[0]}\n2sigma band +/- {widths[1]}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
        coefficients = result.report_diagnostics["Grid polynomial regressions"]
        if key in coefficients.index:
            row = coefficients.loc[key]
            b1, b2 = row[["linear", "quadratic"]]
            equation = rf"$R_p(x)={b1:.2f}x{b2:+.2f}x^2"
            fit = rf"$R^2$={row.r_squared:.1%}" if np.isfinite(row.r_squared) else "$R^2$: n/a"
            label = equation + "$\n" + fit
            curve, = ax.plot(x, b1 * x + b2 * x * x,
                             color="#C46B27", ls="--", lw=1.5, label=label)
            handles = [curve, *band_handles]
            ax.legend(
                handles=handles,
                loc="upper right" if str(key).lower() == "fx" else "upper left",
                fontsize=7.5,
                frameon=False,
            )
        elif band_handles:
            ax.legend(handles=band_handles, loc="upper left", fontsize=7.5, frameon=False)
        ax.set_title(title, fontsize=12, color=INK, fontweight="bold", pad=12)
        ax.grid(True, color="#DFE7F0", linewidth=0.6)
        ax.axhline(0, color="#7B8D9B", lw=0.6)
        ax.axvline(0, color="#7B8D9B", lw=0.6)
        if numeric and len(x):
            ax.set_xlim(x.min() - 0.015, x.max() + 0.015)
            ax.set_xticks(np.arange(np.ceil(x.min() * 10), np.floor(x.max() * 10) + 1) / 10)
        elif len(x):
            ax.set_xticks(x, [str(item) for item in summary.index], rotation=30)
    notes = [
        "Each single-factor panel anchors its named factor at log(1+x); other factors use "
        "the joint conditional covariance solve. Family panels divide x equally across members "
        "before log1p; all member anchors are fixed jointly (Credit + Credit EM: x/2 each). "
        "See scenario construction in the appendix, page 10.",
        f"Shading: conditional +/-1sigma (dark) and +/-2sigma (light) over {months:g} month. "
        "Bands are centred on exact payoff valuations; local sensitivities are recalculated "
        "at each grid point using the fixed reporting denominator. Panel labels show half-widths "
        "at zero shock (n/a if unavailable); shading varies along the grid.",
        "Sigma includes remaining conditional factor risk plus shared-underlying residual "
        "risk. Signed factor Euler contributions plus residual Euler sum to sigma; the "
        "selected factors are fixed, not removed by subtracting their original Euler values.",
        "Local delta/Gaussian covariance approximation: about 68%/95% coverage only under "
        "that approximation. Curvature, strike/knockout jumps and covariance/tail uncertainty "
        "are omitted; zero local delta does not establish absence of nonlinear risk.",
        "All x-axes are factor returns, not yield/spread changes. Dashed: quadratic OLS "
        "through zero; uncentered R-squared = 1 - SSE/sum(return squared). Regression CIs "
        "are exported as diagnostics only and do not determine the shading.",
    ]
    for key in keys:
        group = result.metadata.get("factor_groups", {}).get(
            result.grid_metadata.loc[key, "requested_keys"])
        if group and not np.allclose(group["weights"], 1.0/len(group["members"])):
            notes[0] = ("Family grids allocate total bumps using the exported member weights "
                        "before log1p; the panel axis identifies the split. Other factors use "
                        "the joint conditional covariance solve; see appendix, page 10.")
            break
    if any(result.grid_metadata.loc[key, "bump_convention"] != "simple"
           or result.grid_metadata.loc[key, "completion"] != "conditional" for key in keys):
        notes[0] = ("Each panel uses the supplied grid index and exported Grid conventions. "
                    "Simple family bumps split before log1p; log family bumps split in log units. "
                    "Only conditional grids complete free factors by a joint covariance solve.")
    _footnotes(fig, notes, y=0.185)
    return fig


def _beta_page(result, config):
    """Show the v0 heatmap table with unit-risk and full-denominator summary rows."""
    funded = result.metadata["all_funded"]
    table = _loading_table(result, config)
    count = min(20, len(result.response_exposures))
    fig = _page(
        result,
        config,
        7,
        f"Estimated {config.model_name} loadings and explanatory power",
        f"{count} assets shown by largest absolute MTM; portfolio row uses all modelled "
        "holdings, including assets outside this display."
        if funded
        else f"{count} fitted responses shown by absolute local dollar sensitivity; "
        "portfolio row uses all shared responses and the full reporting denominator.",
    )
    names = (
        config.response_diagnostics["name"].to_dict()
        if config.response_diagnostics is not None and "name" in config.response_diagnostics
        else {}
    )
    labels = []
    for key, row in table.iterrows():
        if key == "Portfolio":
            label = (
                "Modelled subtotal" if result.metadata.get("scope") == "modelled subtotal" else key
            ) + f" | {row.response_exposure / 1e6:.1f}m"
        else:
            label = f"{names.get(key, key)} | {row.response_exposure / 1e6:.1f}m"
        labels.append(label)
    table = table.drop(columns="response_exposure")
    table.index = labels
    factors = result.factor_loadings.columns
    formatted = table.map(lambda x: "n/a" if pd.isna(x) else f"{x:+.2f}")
    formatted["R-squared"] = table["R-squared"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1%}")
    for column in ["Model total vol", "Systematic vol", "Idio vol"]:
        formatted[column] = table[column].map(lambda x: "n/a" if pd.isna(x) else f"{x:.2%}")
    formatted = formatted.rename(
        columns={**config.factor_labels, "Model total vol": f"{config.model_name} total vol"}
    )
    formatted.columns = ["\n".join(textwrap.wrap(str(c), 10)) for c in formatted.columns]
    widths = [0.225] + [0.5 / len(factors)] * len(factors) + [0.065, 0.07, 0.07, 0.07]
    ax = fig.add_axes([0.035, 0.205, 0.93, 0.63])
    _table(
        ax,
        formatted,
        f"Factor loadings and R-squared; annualised {config.model_name}-implied volatility",
        fontsize=8.5,
        first=0.225,
        widths=widths,
    )
    for t in ax.tables:
        for (r, c), cell in t.get_celld().items():
            if r > 0 and 1 <= c <= len(factors):
                value = table.iloc[r - 1, c - 1]
                if np.isfinite(value):
                    color = plt.get_cmap("PiYG")((np.clip(value, -1.5, 1.5) + 1.5) / 3)
                    cell.set_facecolor(color)
                    luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
                    cell.get_text().set_color("white" if luminance < 0.5 else INK)
            elif r > 0 and c == len(factors) + 1:
                value = table.iloc[r - 1, c - 1]
                if np.isfinite(value) and value < 0.3:
                    cell.set_facecolor("#FFF1D8")
            if r == len(table):
                cell.get_text().set_fontweight("bold")
                cell.set_edgecolor(INK)
                cell.set_linewidth(0.7)
    notes = [
        f"Asset systematic vol = sqrt(beta_i.T Sigma beta_i); idio vol = sqrt(residual variance). "
        f"{config.model_name} total vol = sqrt(systematic vol squared + idio vol squared).",
        "Portfolio betas = sum_i (MTM_i / NAV) beta_i. Portfolio volatilities use the full "
        "covariance and signed weights; they are not averages of asset volatilities.",
        "Portfolio R-squared is the absolute-MTM-weighted average of available fitted asset "
        "R-squared, not a portfolio regression R-squared. Asset R-squared measures fit quality.",
        "All volatilities are annualised. Amber R-squared is below 30%. Beta colours: appendix "
        "PiYG palette, capped at +/-1.5; printed values are uncapped. "
        "Missing estimates remain n/a.",
    ]
    if "Rest of assets" in result.report_diagnostics["Loading aggregates"].index:
        notes.append(
            "Rest of assets: omitted modelled holdings aggregated with full-NAV weights; "
            "not a normalised sleeve. R-squared uses their absolute MTM; sleeve vols "
            "do not add to portfolio vol."
        )
    if not funded:
        notes = [
            note.replace("MTM", "local response exposure").replace("holdings", "responses")
            .replace("full-NAV weights", "full-denominator weights").replace("NAV", "N")
            for note in notes
        ]
        notes.append(
            "Response rows describe unit underlying risk, not derivative marks. "
            "Portfolio rows use aggregated payoff sensitivities, including shared "
            "residual risk. N is the explicit reporting denominator."
        )
    _footnotes(fig, notes, y=0.15)
    return fig


def _clusters_page(result, config):
    """Render original fitted cadence trees and the membership table through QIS."""
    fig = _page(
        result,
        config,
        8,
        f"{config.model_name} asset cluster dendrograms",
        "Clusters and merge distances from the assigned production fit; each observation "
        "cadence is clustered separately.",
    )
    if not config.cluster_memberships:
        _empty(
            fig.add_axes([0.08, 0.22, 0.84, 0.55]),
            "Clustering topology is unavailable in the supplied model snapshot. "
            "Supply the fitted clusters, linkages and cutoffs to display the original "
            "estimator trees.",
        )
        return fig
    clusters, linkages, cutoffs = (
        config.cluster_memberships,
        config.cluster_linkages,
        config.cluster_cutoffs,
    )
    names = (
        config.response_diagnostics["name"].copy()
        if config.response_diagnostics is not None and "name" in config.response_diagnostics
        else pd.Series(result.factor_loadings.index, index=result.factor_loadings.index)
    )
    names = names.fillna(pd.Series(names.index, index=names.index))
    for i, asset in enumerate(names.index[names.duplicated(keep=False)], 1):
        names.loc[asset] = f"{names.loc[asset][:15]} [{i}]"
    order = list(reversed(linkages))
    gap = 0.075
    sizes = np.array([max(len(clusters[freq]), 3) for freq in order], dtype=float)
    heights = (0.68 - gap * (len(order) - 1)) * sizes / sizes.sum()
    top = 0.85
    axes, titles = {}, {}
    for freq, height in zip(order, heights):
        axes[freq] = fig.add_axes([0.19, top - height, 0.35, height])
        cadence = {"ME": "Monthly", "QE": "Quarterly"}.get(freq, str(freq))
        titles[freq] = (
            f"{cadence}: {len(clusters[freq])} assets; "
            f"{clusters[freq].nunique()} clusters; cutoff {cutoffs[freq]:.2f}"
        )
        top -= height + gap
    table_ax = fig.add_axes([0.575, 0.17, 0.39, 0.68])
    plot_clusters(
        clusters,
        linkages,
        cutoffs,
        axes=axes,
        table_ax=table_ax,
        titles=titles,
        display_names=names.to_dict(),
        fontsize=9,
        show_distance=True,
        table_title="Fitted cluster membership",
        table_kwargs={"fontsize": 8, "col_widths": [.44, .16, .40]}
        if config.cluster_labels else {"fontsize": 10},
        cluster_labels=config.cluster_labels,
    )
    for ax in axes.values():
        ax.set_title(ax.get_title(), fontsize=11, color=INK, pad=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("Merge distance", fontsize=8, color=INK)
    table_ax.set_title("Fitted cluster membership", fontsize=11, color=INK, pad=10)
    _footnotes(
        fig,
        [
            "Leaves are assets in the fitted clustering universe. Branches show "
            "hierarchical merges; the vertical black line is the fitted distance cutoff and "
            "branch colours identify groups below that cut. Cluster IDs are local to each "
            "cadence. These are return-dependence clusters used in factor estimation, not a "
            "new clustering of the displayed betas. Original asset IDs, linkages and cutoffs "
            "are exported. Descriptive labels are supplied by the caller; ROSAA uses "
            "FactorLasso factor/volatility labels from this fitted snapshot (equal member "
            "weights), without changing memberships or estimating a new tree."
        ],
        y=0.115,
    )
    return fig


def _cluster_contribution_page(result, config):
    """Compare cluster stress, exposures and signed Euler risk on aligned QIS panels."""
    clusters = compute_cluster_contributions(result, config.cluster_memberships)
    summary = display_cluster_table(clusters.summary, clusters)
    scope = ("Modelled subtotal" if result.metadata.get("scope") == "modelled subtotal"
             else "Portfolio")
    denominator_label = "NAV" if result.metadata["all_funded"] else "reporting denominator"
    fig = _page(result, config, 9, "Cluster contributions to stress, factor exposures and risk",
                f"Fitted clusters ordered by gross MTM. Signed contributions use full "
                f"{denominator_label}; {scope.lower()} includes {len(clusters.holdings)} holdings.")

    def displayed(frame):
        """Keep the common gross-MTM order and append an additive portfolio total."""
        table = display_cluster_table(frame, clusters)
        table.loc[scope] = frame.sum()
        return table

    def heatmap(ax, table, percent, labels=None, fontsize=8):
        """Place column names above signed cells in the report's diverging palette."""
        table = table.copy()
        if labels is not None:
            table.index = [labels.get(key, key) for key in table.index]
        limit = max(float(table.abs().max().max()), 1e-12)
        plot_heatmap(table, ax=ax, cmap="PiYG", var_format="{:+.2%}" if percent else "{:.2f}",
                     fontsize=fontsize, top_x_label=True, vmin=-limit, vmax=limit,
                     date_format=None, hline_rows=[len(table)-1], x_rotation=0)
        ax.set_ylabel("")
        ax.tick_params(axis="both", length=0)
        ax.get_yticklabels()[-1].set_weight("bold")

    def stacked(ax, values, colors, legend_columns):
        """Draw shared-order signed Euler bars and annotate their displayed net totals."""
        plot_bars(values, ax=ax, is_horizontal=True, stacked=True, colors=colors,
                  fontsize=7.5, add_bar_values=False, x_rotation=0, legend_loc=None,
                  xvar_format="{:.1%}", yvar_format="{:.1%}")
        ax.set_ylabel("")
        ax.axvline(0., color=INK, lw=.7)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.1%}"))
        ax.grid(axis="x", color="#E4EAF0", linewidth=.5)
        ax.margins(x=.25)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=legend_columns,
                  fontsize=6.7, frameon=False, columnspacing=.8, handlelength=1.3)
        for y, value in enumerate(values.sum(axis=1)):
            ax.annotate(f"{value:+.2%}", (value, y), xytext=(4 if value >= 0 else -4, 0),
                        textcoords="offset points", ha="right" if value < 0 else "left",
                        va="center", fontsize=7.4, color=INK)

    labels = {name: f"{name} | {row.nav_weight:+.1%} | n={int(row.holding_count)}"
              for name, row in summary.iterrows()}
    labels[scope] = (f"{scope} | {summary.nav_weight.sum():+.1%} | "
                     f"n={int(summary.holding_count.sum())}")
    fig.text(.21, .89, f"Correlated requested scenarios: cluster P&L (% of {denominator_label})",
             fontsize=12, weight="bold", color=INK)
    ax = fig.add_axes([.21, .575, .575, .255])
    contributor_ax = fig.add_axes([.80, .575, .17, .255])
    if "conditional" in clusters.scenario_nav:
        stress = displayed(clusters.scenario_nav["conditional"].iloc[:, :12])
        descriptions = result.metadata.get("scenario_descriptions", {})
        stress.columns = ["\n".join(textwrap.wrap(str(descriptions.get(str(key), key)), 13,
                                                break_long_words=False))
                          for key in stress.columns]
        heatmap(ax, stress, True, labels, fontsize=7.7)
        top = cluster_top_contributors(result, clusters, displayed=True).reindex(stress.index)
        cells = []
        for _, row in top.iterrows():
            name = textwrap.shorten(str(row["name"]), width=25, placeholder="...")
            scenario = textwrap.shorten(
                str(descriptions.get(str(row.scenario), row.scenario)), width=31,
                placeholder="...")
            cells.append(f"{name} {row.nav_contribution:+.2%}\n{scenario}")
        from matplotlib.colors import ListedColormap
        plot_heatmap(pd.DataFrame(0., index=top.index, columns=["Top contributor"]),
                     ax=contributor_ax, annot=np.asarray(cells).reshape(-1, 1),
                     var_format=None, cmap=ListedColormap(["#F0F4F7"]),
                     fontsize=7.4, top_x_label=True, date_format=None,
                     hline_rows=[len(top)-1], x_rotation=0)
        contributor_ax.set_yticklabels([])
        contributor_ax.set_ylabel("")
        contributor_ax.tick_params(axis="both", length=0)
    else:
        _empty(ax, "Conditional requested scenarios were not computed.")
        contributor_ax.set_axis_off()

    fig.text(.105, .49, "Portfolio-weighted factor exposures", fontsize=11,
             weight="bold", color=INK)
    exposure = displayed(clusters.factor_exposures)
    exposure.columns = ["\n".join(textwrap.wrap(str(config.factor_labels.get(key, key)), 8,
                                               break_long_words=False))
                        for key in exposure.columns]
    heatmap(fig.add_axes([.105, .27, .40, .18]), exposure, False, fontsize=7.1)

    fig.text(.605, .49, "Top-five factor risk contributions", fontsize=10,
             weight="bold", color=INK)
    selected = result.report_diagnostics["Factor Euler volatility"].euler_vol.abs().sort_values(
        ascending=False, kind="stable").head(5).index
    factor_risk = displayed(clusters.factor_risk.loc[:, selected])
    factor_risk.columns = [config.factor_labels.get(key, key) for key in selected]
    stacked(fig.add_axes([.605, .27, .145, .18]), factor_risk,
            ["#315B7A", "#C49A3A", "#648E6C", "#A15B77", "#8A7CB4"][:len(selected)], 2)

    fig.text(.835, .49, "Annual model volatility", fontsize=10,
             weight="bold", color=INK)
    stacked(fig.add_axes([.835, .27, .13, .18]),
            displayed(clusters.risk[["Systematic", "Idiosyncratic"]]),
            [BLUE, "#C49A3A"], 1)
    notes = [
        "Stress (first 12 requested correlated scenarios): exact holding P&L summed by cluster / "
        "full notional. Top contributor: largest absolute holding P&L in that row's worst "
        f"correlated scenario, with its signed % of {denominator_label} and scenario name; "
        "the portfolio row uses "
        "the portfolio's worst scenario. Methodology: appendix, page 10.",
        "Exposures: sum current holding response dollar sensitivities times factor betas / "
        "notional. Factor risk bars: portfolio's five largest absolute atomic-factor Euler "
        "contributions, using the same factors and colours for every cluster. Labels show their "
        "signed subtotal; omitted factors mean this is not total systematic risk. All factors "
        "are retained in the exported table.",
        "Annual model volatility: systematic plus shared-response idiosyncratic Euler terms; "
        "components and clusters sum to total portfolio model volatility. Negative contributions "
        "reduce risk. These are additive contributions, not standalone cluster volatilities.",
        "Rows show cadence-local ID, net weight and holding count. At most eight display groups "
        "preserve smaller clusters in Other clusters. Unassigned and multi-cluster holdings remain "
        "explicit. Full memberships, labels and contribution tables are exported."
    ]
    if not config.cluster_memberships:
        notes.append("No fitted memberships supplied: modelled holdings are shown as unassigned.")
    if result.metadata.get("scope") == "modelled subtotal":
        excluded = result.metadata.get("excluded_position_ids", [])
        gross = result.metadata.get("excluded_gross_mtm", 0.)
        notes.append(f"Excluded: {len(excluded)} unmodelled holdings, gross MTM "
                     f"{result.metadata['reference_currency']} {gross:,.0f}. Their stress and risk "
                     "are unknown; displayed ratios retain the full portfolio notional.")
    _footnotes(fig, notes, y=.18)
    return fig


def _methodology_page(result, config):
    """Append the owning qis covariance display and the conditional-shock formula."""
    fig = _page(
        result,
        config,
        10,
        f"{config.model_name} correlation and scenario construction",
        f"Fitted covariance at {pd.Timestamp(result.metadata['risk_date']):%d %b %Y}; "
        "lower triangle: correlations; diagonal: annualised factor volatilities.",
    )
    ax = fig.add_axes([0.105, 0.235, 0.48, 0.60])
    plot_corr_matrix_from_covar(
        result.factor_covariance.rename(index=config.factor_labels, columns=config.factor_labels),
        ax=ax,
        title=None,
        cmap="PiYG",
        corr_format="{:.2f}",
        vol_format="{:.1%}",
        fontsize=9,
        x_rotation=90,
    )
    fig.text(
        0.105,
        0.895,
        str(
            result.metadata.get("source_provenance", {}).get(
                "covariance_method", "Covariance supplied by the assigned portfolio model."
            )
        ),
        fontsize=11,
        color=INK,
    )
    from fractions import Fraction

    left = 0.62
    fig.text(left, 0.835, "Conditional factor shocks", fontsize=15, fontweight="bold", color=INK)
    fig.text(left, 0.791, r"$z_A=\log(P_A^{\mathrm{target}}/P_A^0)$", fontsize=16, color=INK)
    fig.text(
        left,
        0.746,
        r"$z_F=\Sigma_{FA}\Sigma_{AA}^{-1}z_A;\quad r_i=e^{z_i}-1$",
        fontsize=15,
        color=INK,
    )
    fig.text(
        left,
        0.704,
        "A: anchored factors; F: free factors.\nSigma: annual factor log-return covariance.\n"
        "Anchors stay fixed; no mean return is added.",
        fontsize=10,
        color=INK,
        va="top",
        linespacing=1.3,
    )
    fig.text(left, 0.625, "Conditional covariance", fontsize=15, fontweight="bold", color=INK)
    fig.text(
        left,
        0.580,
        r"$\Sigma_{F|A}=\Sigma_{FF}-\Sigma_{FA}\Sigma_{AA}^{-1}\Sigma_{AF}$",
        fontsize=15,
        color=INK,
    )
    fig.text(left, 0.530, "Conditional local volatility bands",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(
        left,
        0.486,
        r"$v_{p|A}(x)=e_F(x)^\top\Sigma_{F|A}e_F(x)+\sum_j w_j(x)^2\sigma_{\epsilon,j}^2$",
        fontsize=14,
        color=INK,
    )
    fig.text(
        left, 0.442, r"$R_p(x)\;\pm\;k\sqrt{T\,v_{p|A}(x)},\quad k=1,2$", fontsize=14, color=INK
    )
    fig.text(
        left,
        0.403,
        "w(x) = scenario response dollar sensitivity / N.\n"
        "e(x) = beta.T w(x); N: fixed reporting denominator.\n"
        f"T = {Fraction(result.metadata['horizon_years']).limit_denominator(365)} year. "
        "Shared response residuals.\n"
        "Signed factor Euler + residual Euler = local sigma.\n"
        "Fixed conditional covariance; scenario-local deltas.",
        fontsize=9.5,
        color=INK,
        va="top",
        linespacing=1.35,
    )
    fig.text(left, 0.299, "Portfolio scenario valuation", fontsize=13, fontweight="bold", color=INK)
    fig.text(left, 0.256, r"$R_p(x)=\sum_jw_j[\exp(\beta_j^\top z(x))-1]$", fontsize=15, color=INK)
    fig.text(
        0.105,
        0.186,
        r"Single anchor: $z_i=\rho_{ia}(\sigma_i/\sigma_a)z_a$",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.105,
        0.151,
        "Yield mapping: z = log(1 - D x change in yield), or the declared external proxy.",
        fontsize=9,
        color=INK,
    )
    notes = [
        "Level/price targets use correlated shocks on both requested pages. Explicit return shocks "
        "use isolated factors on page 1 and conditional co-moves on page 2.",
        "This is a conditional scenario under the fitted covariance, not a shock to the "
        "correlation matrix or a forecast probability. The EWMA span is not a hard rolling window.",
    ]
    special = result.metadata.get("source_provenance", {}).get("external_target_method")
    if special:
        notes.append(str(special))
    _footnotes(fig, notes)
    if not result.metadata["all_funded"]:
        # The ordinary funded-asset formula does not value calls, puts or futures.
        for text in fig.texts:
            if text.get_text().startswith("$R_p(x)="):
                text.set_text(r"$R_p(x)=\sum_h[V_h(z(x))-V_h(0)]/N$")
    return fig


def _coverage_page(result, config):
    """Render only the optional parser-supplied table and footnote explanations."""
    fig = _page(result, config, 11, config.appendix_title, config.appendix_subtitle)
    _table(fig.add_axes([0.04, 0.265, 0.92, 0.55]), config.appendix_table, first=0.23, fontsize=8.5)
    _footnotes(fig, config.appendix_notes, y=0.215)
    return fig


def report_pages(result, config):
    """Yield ten core exhibits and an eleventh page only when supplied by the parser."""
    model = config.model_name
    meta = result.metadata
    currency = meta["reference_currency"]
    denominator = meta["reporting_denominator"]
    date = pd.Timestamp(meta["risk_date"])
    qualifier = " | Modelled subtotal only" if meta.get("scope") == "modelled subtotal" else ""
    overrides = meta.get("scenario_completion_overrides", {})
    subtitle = (
        "Level/price targets: correlated shocks. Explicit return shocks: other factors "
        "held at zero."
        if overrides
        else "Specified factor shocks; other factors held at zero. Instantaneous valuation."
    )
    if meta["requested_completion"] != "independent":
        subtitle = f"Requested completion policy: {meta['requested_completion']}."
    title = (
        "Requested independent stress scenarios"
        if meta["requested_completion"] == "independent"
        else "Requested stress scenarios"
    )
    yield (
        title,
        _scenario_page(
            result,
            config,
            1,
            title,
            result.valuations["requested"],
            subtitle + f" Using notional of {currency} {denominator:,.0f}." + qualifier,
        ),
    )
    title = f"Requested scenarios with latest {model} co-moves"
    yield (
        title,
        _scenario_page(
            result,
            config,
            2,
            title,
            result.valuations.get("conditional"),
            f"Conditional shocks using the fitted covariance at {date:%d %b %Y}; explicit anchors "
            "are preserved." + qualifier,
        ),
    )
    count = meta["historical_count"]
    title = f"{'Ten' if count == 10 else count} worst {model} historical scenario months"
    yield (
        title,
        _scenario_page(
            result,
            config,
            3,
            title,
            result.historical,
            "Complete historical monthly factor vectors ranked by loss on today's holdings "
            "and loadings; descriptive replay, not realised performance." + qualifier,
        ),
    )
    for title, builder in [
        (f"Portfolio {model} exposures and risk", _risk_page),
        ("Largest factor exposures: asset risk contributors", _contributor_page),
        ("Sensitivity to largest factor exposures", _grid_page),
        (f"Estimated {model} loadings and explanatory power", _beta_page),
        (f"{model} asset cluster dendrograms", _clusters_page),
        ("Cluster contributions to stress, factor exposures and risk", _cluster_contribution_page),
        (f"{model} correlation and scenario construction", _methodology_page),
    ]:
        yield title, builder(result, config)
    if config.appendix_table is not None:
        yield config.appendix_title, _coverage_page(result, config)
