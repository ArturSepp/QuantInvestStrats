"""QIS plotting pages for already-computed portfolio stress results."""

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

from qis.plots.bars import plot_bars
from qis.plots.table import plot_df_table
from qis.plots.derived.clustering import plot_clusters
from qis.plots.scatter import plot_scatter
from qis.models.linear.plot_correlations import plot_corr_matrix_from_covar
from qis.portfolio.stress.reporting import _loading_table


INK = "#18354B"
BLUE = "#315B7A"
RED = "#A64045"


def _page(result, config, number, title, subtitle):
    """Create a consistent landscape canvas with dated currency/denominator footers."""
    fig = plt.figure(figsize=(16.54, 11.69), facecolor="white")
    fig.text(0.04, 0.955, title, fontsize=21, weight="bold", color=INK)
    fig.text(0.04, 0.923, textwrap.fill(subtitle, 160), fontsize=10, color=BLUE)
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
        notes.append("Correlated-shock methodology and formulas: see Appendix, page 9.")
    _footnotes(fig, notes, y=0.10, width=190, fontsize=7.6)
    return fig


def _currency_scale(result):
    """Keep monetary chart axes readable while preserving explicit value units."""
    scale = 1e6 if result.metadata["reporting_denominator"] >= 1e6 else 1.0
    currency = result.metadata["reference_currency"]
    return scale, f"{currency} millions" if scale == 1e6 else currency


def _footnotes(fig, notes, y=0.115, width=190, fontsize=8):
    """Place complete explanations within the reserved note band."""
    lines = [line for note in notes for line in textwrap.wrap(str(note), width)]
    available = max(y - 0.044, 0.02) * 11.69 * 72
    fontsize = min(fontsize, available / max(1, len(lines)) / 1.3)
    fig.text(0.04, y, "\n".join(lines), fontsize=fontsize, color=INK, va="top", linespacing=1.3)


def _panel_note(fig, x, y, note):
    """Wrap one panel's variable definitions without crossing the adjacent column."""
    fig.text(x, y, textwrap.fill(note, 88), fontsize=8, color=INK, va="top")


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


def _contributor_page(result, config):
    """Allocate the largest factor Euler contributions to original holdings."""
    fig = _page(
        result,
        config,
        5,
        "Largest factor exposures: asset risk contributors",
        "Factors ranked by absolute factor Euler contribution. Assets ranked by absolute Euler "
        "contribution to total annual portfolio volatility.",
    )
    rc = result.report_diagnostics["Factor Euler volatility"].euler_vol
    factors = (
        rc.loc[rc.ne(0)]
        .abs()
        .sort_values(ascending=False, kind="stable")
        .head(6)
        .index
    )
    grid = fig.add_gridspec(
        2, 3, left=0.11, right=0.96, top=0.86, bottom=0.20, hspace=0.48, wspace=0.68
    )
    for i, factor in enumerate(factors):
        values = result.report_diagnostics["Holding factor Euler volatility"][factor]
        selected = values.abs().sort_values(ascending=False, kind="stable").head(10).index
        values = values.loc[selected].copy()
        name_col = "metadata:short_name" if "metadata:short_name" in result.positions else "name"
        values.index = result.positions.loc[selected, name_col]
        _bars(
            fig.add_subplot(grid[i // 3, i % 3]),
            values,
            f"{config.factor_labels.get(factor, factor)}\n"
            f"Exposure {result.factor_betas[factor]:+.2f}; factor Euler {rc[factor]:+.2%}",
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
            "residual variance / sigma_p gives total volatility sigma_p. All holding contributions "
            "to a factor sum to that factor's Euler term.",
            "Six largest absolute factor Euler terms and ten largest absolute holding "
            "contributions per factor are shown; displayed subsets need not sum to full totals. "
            "Complete tables are exported. Funded factor beta e_f = "
            "sum_i (MTM_i / NAV) x beta_if."
            if result.metadata["all_funded"]
            else "Six largest absolute factor Euler terms and ten largest absolute holding "
            "contributions per factor are shown; displayed subsets need not sum to full totals. "
            "Derivative betas use current payoff Jacobians and shared responses; these are local "
            "risk contributions. Complete tables are exported."
        ],
        y=0.13,
        width=185,
        fontsize=9,
    )
    return fig


def _grid_page(result, config):
    """Show exact grids, quadratic mean-fit confidence intervals and conditional risk bands."""
    funded = result.metadata["all_funded"]
    confidence = result.metadata["confidence"]
    months = result.metadata["horizon_years"] * 12
    regression_bands = result.report_diagnostics["Grid regression confidence bands"]
    keys = config.selected_grids or tuple(result.grids)[:4]
    ranges = []
    for key in keys:
        try:
            x = np.asarray(result.grid_summaries[key].index, dtype=float)
            ranges.append(f"{key} {x.min():+.0%} to {x.max():+.0%}")
        except (TypeError, ValueError):
            ranges.append(key)
    subtitle = "Correlated shocks: " + "; ".join(ranges) + "."
    standard = tuple(str(key).lower() for key in keys) == ("equity", "rates", "credit", "fx")
    if standard:
        expected = [np.arange(-30, 31) / 100] + [np.arange(-20, 21) / 100] * 3
        standard = all(
            len(result.grid_summaries[key]) == len(axis)
            and np.allclose(np.asarray(result.grid_summaries[key].index, dtype=float), axis)
            for key, axis in zip(keys, expected)
        )
    if standard:
        subtitle = "Correlated shocks: equity +/-30%; rates, credit, FX +/-20%; step 1%."
    if any("lower_bound" in result.grid_summaries[key] for key in keys):
        subtitle += f" Blue: {confidence:.0%} conditional prediction band ({months:g} month)."
    else:
        subtitle += " Deterministic payoff curves."
    if any(key in regression_bands.index.get_level_values("grid")
           and regression_bands.loc[key, "mean_ci_lower"].notna().all() for key in keys):
        subtitle += f" Orange: {confidence:.0%} quadratic-fit confidence band."
    fig = _page(result, config, 6, "Sensitivity to equity, rates, credit and FX factors", subtitle)
    boxes = [
        [0.085, 0.575, 0.38, 0.265],
        [0.575, 0.575, 0.38, 0.265],
        [0.085, 0.215, 0.38, 0.265],
        [0.575, 0.215, 0.38, 0.265],
    ]
    for i, box in enumerate(boxes):
        ax = fig.add_axes(box)
        if i >= len(keys):
            _empty(ax, "No additional sensitivity grid supplied.")
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
        title = requested
        xlabel = requested + " factor return"
        if group is not None:
            title = " + ".join(group["members"]) + " (split total family bump)"
            xlabel = f"Total {group['label'] or requested} family bump"
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
            markersize=17,
            fontsize=9,
            ax=ax,
        )
        if "lower_bound" in summary:
            ax.fill_between(
                x, summary.lower_bound, summary.upper_bound, color=BLUE, alpha=0.18, zorder=0
            )
            ax.text(
                0.98,
                0.03,
                f"{confidence:.0%} band: +/-{summary.band_half_width.iloc[0]:.2%} of NAV",
                transform=ax.transAxes,
                ha="right",
                fontsize=8,
                color=INK,
            )
        coefficients = result.report_diagnostics["Grid polynomial regressions"]
        if key in coefficients.index:
            row = coefficients.loc[key]
            b1, b2 = row[["linear", "quadratic"]]
            equation = rf"$R_p(x)={b1:.2f}x{b2:+.2f}x^2"
            fit = rf"$R^2$={row.r_squared:.1%}" if np.isfinite(row.r_squared) else "$R^2$: n/a"
            label = equation + "$\n" + fit
            curve, = ax.plot(x, b1 * x + b2 * x * x,
                             color="#C46B27", ls="--", lw=1.5, label=label)
            handles = [curve]
            if key in regression_bands.index.get_level_values("grid"):
                band = regression_bands.loc[key].reindex(summary.index)
                if band[["mean_ci_lower", "mean_ci_upper"]].notna().all().all():
                    shading = ax.fill_between(
                        x, band.mean_ci_lower, band.mean_ci_upper,
                        color="#C46B27", alpha=0.23, zorder=1,
                        label=f"{confidence:.0%} quadratic-fit CI",
                    )
                    handles.append(shading)
            ax.legend(
                handles=handles,
                loc="upper right" if str(key).lower() == "fx" else "upper left",
                fontsize=9,
                frameon=False,
            )
        ax.set_title(title, fontsize=12, color=INK, fontweight="bold", pad=12)
        ax.grid(True, color="#DFE7F0", linewidth=0.6)
        ax.axhline(0, color="#7B8D9B", lw=0.6)
        ax.axvline(0, color="#7B8D9B", lw=0.6)
        if numeric and len(x):
            ax.set_xlim(x.min() - 0.015, x.max() + 0.015)
            ax.set_xticks(np.arange(np.ceil(x.min() * 10), np.floor(x.max() * 10) + 1) / 10)
        elif len(x):
            ax.set_xticks(x, [str(item) for item in summary.index], rotation=30)
    if any("lower_bound" in result.grid_summaries[key] for key in keys):
        fig.text(
            0.085,
            0.15,
            f"{confidence:.0%} bands include conditional factor risk + "
            f"idiosyncratic risk over {months:g} month. Baseline portfolio exposures; "
            "analytical covariance method.",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )
    notes = [
        "Credit: x is the total family bump; each of n Credit factors receives "
        "log(1+x/n) for an equal split. Other panels anchor at log(1+x). Free factors "
        "use one joint conditional solve. Appendix, page 9.",
        "All x-axes are factor returns, not yield or spread changes.",
    ]
    if funded and any("lower_bound" in result.grid_summaries[key] for key in keys):
        notes += [
            "Blue: conditional factor dispersion plus asset residual risk, centred on exact "
            "scenario valuations. Baseline exposures, Gaussian covariance and horizon scaling; "
            "constant width within each panel.",
        ]
    elif not funded:
        notes += ["Derivative points use exact intrinsic payoffs, including strike kinks and "
                  "knockout jumps. A smooth quadratic can miss these features."]
    notes += [
        f"Orange: {confidence:.0%} pointwise Student-t confidence interval for the quadratic "
        "OLS fitted mean, with zero intercept and n-2 error degrees of freedom.",
        "Grid points are deterministic. OLS intervals assume independent, constant-variance "
        "regression errors; they describe the fitted approximation, not future portfolio "
        "loss risk.",
        "Dashed: quadratic fit in decimal returns. Uncentered R-squared = 1 - SSE/sum(return "
        "squared); n/a for a zero curve. No CI without error degrees of freedom; no fit on a "
        "rank-deficient grid.",
    ]
    if any(result.grid_metadata.loc[key, "bump_convention"] != "simple"
           or result.grid_metadata.loc[key, "completion"] != "conditional" for key in keys):
        notes[0] = ("Each panel uses the supplied grid index and exported Grid conventions. "
                    "Simple family bumps split before log1p; log family bumps split in log units. "
                    "Only conditional grids complete free factors by a joint covariance solve.")
    _footnotes(fig, notes, y=0.135, width=190, fontsize=7.6)
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
            )
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
    _footnotes(fig, notes, y=0.15, width=190, fontsize=8.5)
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
        axes[freq] = fig.add_axes([0.20, top - height, 0.42, height])
        cadence = {"ME": "Monthly", "QE": "Quarterly"}.get(freq, str(freq))
        titles[freq] = (
            f"{cadence}: {len(clusters[freq])} assets; "
            f"{clusters[freq].nunique()} clusters; cutoff {cutoffs[freq]:.2f}"
        )
        top -= height + gap
    table_ax = fig.add_axes([0.69, 0.17, 0.27, 0.68])
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
        table_kwargs={"fontsize": 10},
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
            "are exported."
        ],
        y=0.105,
        width=185,
        fontsize=8,
    )
    return fig


def _methodology_page(result, config):
    """Append the owning qis covariance display and the conditional-shock formula."""
    fig = _page(
        result,
        config,
        9,
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
    fig.text(left, 0.530, "Analytical prediction band", fontsize=15, fontweight="bold", color=INK)
    fig.text(
        left,
        0.486,
        r"$v_{p|A}=e_F^\top\Sigma_{F|A}e_F+\sum_j w_j^2\sigma_{\epsilon,j}^2$",
        fontsize=14,
        color=INK,
    )
    fig.text(
        left, 0.442, r"$R_p(x)\;\pm\;\Phi^{-1}((1+c)/2)\sqrt{T\,v_{p|A}}$", fontsize=14, color=INK
    )
    fig.text(
        left,
        0.403,
        "w = MTM / NAV; e = beta.T w (baseline exposures).\n"
        f"T = {Fraction(result.metadata['horizon_years']).limit_denominator(365)} year; "
        f"c = {result.metadata['confidence']:.0%}. Independent residuals.\n"
        "Gaussian conditional covariance; linearised NAV risk.\n"
        "Same covariance and exposures at every grid point.",
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
    lines = []
    for note in notes:
        lines.extend(textwrap.wrap(note, 175))
    fig.text(0.04, 0.115, "\n".join(lines), fontsize=9, color=INK, va="top", linespacing=1.4)
    if not result.metadata["all_funded"]:
        # The ordinary funded-asset formula does not value calls, puts or futures.
        for text in fig.texts:
            if text.get_text().startswith("$R_p(x)="):
                text.set_text(r"$R_p(x)=\sum_h[V_h(z(x))-V_h(0)]/N$")
            if text.get_text() == "Analytical prediction band":
                text.set_text("Local risk (no derivative bands)")
            if text.get_text().startswith("w = MTM"):
                text.set_text(
                    "w = shared response dollar sensitivity / N.\n"
                    "N: explicit reporting denominator.\n"
                    "Intrinsic payoff changes retain observed MTM anchors.\n"
                    "No derivative prediction bands are displayed."
                )
            if text.get_text().startswith("$R_p(x)\\;"):
                text.set_text("Band formula applies to funded assets only.")
    return fig


def _coverage_page(result, config):
    """Render only the optional parser-supplied table and footnote explanations."""
    fig = _page(result, config, 10, config.appendix_title, config.appendix_subtitle)
    _table(fig.add_axes([0.04, 0.265, 0.92, 0.55]), config.appendix_table, first=0.23, fontsize=8.5)
    _footnotes(fig, config.appendix_notes, y=0.215, width=175, fontsize=9)
    return fig


def report_pages(result, config):
    """Yield nine core exhibits and a tenth page only when supplied by the parser."""
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
        ("Sensitivity to equity, rates, credit and FX factors", _grid_page),
        (f"Estimated {model} loadings and explanatory power", _beta_page),
        (f"{model} asset cluster dendrograms", _clusters_page),
        (f"{model} correlation and scenario construction", _methodology_page),
    ]:
        yield title, builder(result, config)
    if config.appendix_table is not None:
        yield config.appendix_title, _coverage_page(result, config)
