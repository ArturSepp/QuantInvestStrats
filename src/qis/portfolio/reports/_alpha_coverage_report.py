"""Paginated exposure/mapping/exclusion audit for opt-in holdings-alpha inference."""

import textwrap

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from qis.portfolio.reports._performance_inference import _date_ranges


def plot_alpha_coverage(data, strategy_name, figsize=(8.5, 11.7), portfolio_keys=None):
    """Draw every mapping and excluded interval, without truncating or changing samples."""
    meta = data.metadata["alpha"]
    if not meta.get("coverage_enabled"):
        return []
    paired = data.metadata.get("paired", False)
    entries = meta.get("portfolios", {}) if paired else {strategy_name: meta}
    figures = []
    for key, settings in entries.items():
        if portfolio_keys is not None and key not in portfolio_keys:
            continue
        label = f"{key}: {data.metadata[key.lower()]}" if paired else str(key)
        prefix = key + "_" if paired else ""
        columns = [name for name in data.alpha_diagnostics if name.startswith(prefix)]
        diagnostic = data.alpha_diagnostics[columns].rename(
            columns={name: name[len(prefix):] for name in columns})
        returns = diagnostic["alpha"] if paired else data.alpha_returns
        required = [f"{kind}_gross" for kind in ("direct", "proxy", "omitted", "uncovered")]
        lines = [f"Benchmark: {settings.get('benchmark', meta['benchmark'])}; "
                 f"grid {meta['freq']}; beta span/warm-up {meta['factor_beta_span']}.",
                 "Actual signed holdings weight betas; exposure chart uses absolute NAV weights.",
                 "Own eligible beta first; mapped proxy second; bounded missing-beta dust last.",
                 f"Dust caps: {settings.get('weight_tolerance', 0) * 10000:g} bp per instrument; "
                 f"{settings.get('total_weight_tolerance', 0) * 10000:g} bp gross total."]
        if all(column in diagnostic for column in required):
            valid = np.isfinite(returns)
            strict = np.isfinite(diagnostic.strict_alpha)
            lines += [f"Eligible alpha returns: {int(valid.sum())}/{len(valid)}; "
                      f"strict comparator: {int(strict.sum())}/{len(strict)}."]
            for kind in ("direct", "proxy", "omitted", "uncovered"):
                values = diagnostic[kind + "_gross"]
                lines.append(f"{kind.capitalize()} exposure: mean {values.mean() * 10000:.4g} bp; "
                             f"max {values.max() * 10000:.4g} bp; "
                             f"{int(values.gt(0).sum())} dates >0.")
            lines.append(f"Unknown opening weights: {int(diagnostic.unknown_weights.sum())} dates; "
                         "their exposure is unknown, not zero.")
        else:
            valid = np.isfinite(returns)
            lines.append("Holdings coverage diagnostics unavailable; see inference warnings.")
        lines.append("Explicit instrument -> proxy mappings (no automatic selection):")
        mapping = settings.get("proxy_map", {})
        lines.extend([f"  {asset} -> {proxy}" for asset, proxy in mapping.items()] or ["  None."])
        lines.append("Excluded alpha return dates (coverage or nonfinite realised returns):")
        ranges = _date_ranges(returns.index, ~valid)
        lines.extend([f"  {item['start']:%d%b%Y} - {item['end']:%d%b%Y} "
                      f"({item['n_periods']} dates)" for item in ranges] or ["  None."])
        if paired:
            lines.append("Joint marginal-alpha exclusions (either portfolio or realised input):")
            ranges = _date_ranges(data.alpha_returns.index, ~np.isfinite(data.alpha_returns))
            lines.extend([f"  {item['start']:%d%b%Y} - {item['end']:%d%b%Y} "
                          f"({item['n_periods']} dates)" for item in ranges] or ["  None."])
        elif meta.get("discarded_valid_n_obs", 0):
            lines.append("Full/expanding latest-block selection also excludes earlier valid dates;")
            lines.append("all dates outside that selected block:")
            lines.extend(f"  {item['start']:%d%b%Y} - {item['end']:%d%b%Y} "
                         f"({item['n_periods']} dates)" for item in meta["excluded_ranges"])
        lines = [part for line in lines for part in textwrap.wrap(line, width=98)]
        for offset in range(0, len(lines), 27):
            figure = plt.figure(figsize=figsize)
            figure.suptitle("Holdings alpha coverage audit\n" + textwrap.fill(label, width=65),
                           fontsize=12, weight="bold", y=.98, color="#164773")
            axis = figure.add_axes((.10, .65, .83, .22))
            if all(column in diagnostic for column in required):
                for column, colour in zip(required, ("#1764a0", "#ba7539", "#6b4c9a", "#b03030")):
                    axis.plot(diagnostic.index, diagnostic[column], color=colour,
                              label=column.replace("_gross", ""), linewidth=1)
                axis.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(.5, 1.2))
            axis.set_ylabel("Gross exposure / NAV", fontsize=8)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.tick_params(labelsize=7)
            axis.grid(alpha=.2)
            figure.text(.08, .57, "\n".join(lines[offset:offset + 27]), va="top",
                        fontsize=8, linespacing=1.4)
            figure.text(.08, .065,
                        "Proxy-assisted/dust-adjusted alpha is approximate. CIs do not include "
                        "proxy or omission model uncertainty.\n"
                        "No shorter beta warm-up, future-data fill, weight renormalisation or "
                        "NAV changes. Strict alpha is retained in diagnostics.", fontsize=6.5,
                        color=".35")
            figure.text(.08, .035, "qis docs: alpha coverage policy", fontsize=7, color="#1764a0",
                        url="https://quantinveststrats.readthedocs.io/en/latest/"
                            "factsheets_and_reporting.html#optional-holdings-alpha-coverage")
            figure.text(.93, .035, f"Coverage {offset // 27 + 1}/{(len(lines) + 26) // 27}",
                        ha="right", fontsize=7)
            figures.append(figure)
    return figures
