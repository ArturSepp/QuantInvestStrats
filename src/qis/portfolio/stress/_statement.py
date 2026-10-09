"""SOA-style account opening page from detached accounting and allocation tables."""

import pandas as pd
from matplotlib.lines import Line2D

from qis.plots.bars import plot_bars
from qis.plots.pie import plot_pie


def displayed_allocation(allocation, maximum_currencies=5):
    """Keep all amounts while limiting the displayed currencies to top four plus Other."""
    totals = allocation.clip(lower=0).sum(axis=0).sort_values(ascending=False, kind="stable")
    if len(totals) <= maximum_currencies:
        return allocation.reindex(columns=totals.index)
    kept = totals.index[:maximum_currencies - 1]
    displayed = allocation.loc[:, kept].copy()
    displayed["Other"] = allocation.drop(columns=kept).sum(axis=1)
    return displayed


def statement_page(result, config):
    """Render balance-sheet reconciliation, gross allocation and two small charts."""
    from qis.portfolio.stress._figures import _page, _table, _empty, _note, INK, BLUE
    meta = result.metadata
    currency = meta["reference_currency"]
    title = f"Statement of assets as of {pd.Timestamp(meta['valuation_date']):%d.%m.%Y}"
    fig = _page(result, config, 0, title,
                f"Account overview | Values in {currency}, including accrued amounts")
    accounting = meta["accounting"]
    labels = {"gross_assets": "Gross assets", "borrowing": "Borrowing",
              "other_liabilities": "Other liabilities", "net_equity": "Net equity / NAV",
              "assets_to_equity": "Assets / equity", "debt_to_equity": "Debt / equity",
              "reporting_denominator": "Percentage denominator"}
    rows = {labels[key]: (f"{value:.2f}x" if key == "assets_to_equity" else
                         f"{value:.2%}" if key == "debt_to_equity" else
                         f"{currency} {value:,.2f}") for key, value in accounting.items()}
    _table(fig.add_axes([.04, .59, .42, .27]), pd.DataFrame.from_dict(
        rows, orient="index", columns=["Amount / ratio"]), "Reconciled balance sheet",
        first=.55, fontsize=11)
    allocation = result.report_diagnostics.get("Asset allocation with borrowing", pd.DataFrame())
    if allocation.empty:
        _empty(fig.add_axes([.51, .59, .45, .27]),
               "Supply asset_class and currency mappings to display gross allocation.")
        _note(fig, "Net equity after stress is shown on the scenario pages. "
              "The complete balance sheet and equity-after-stress tables are exported.")
        return fig
    displayed = displayed_allocation(allocation)
    scale = 1e6 if accounting["gross_assets"] >= 1e6 else 1.
    unit = currency + (" millions" if scale == 1e6 else "")
    gross = accounting["gross_assets"]
    amounts = displayed.copy()
    amounts.insert(0, "Total", amounts.sum(axis=1))
    amounts.loc["Gross assets"] = amounts.drop(index="Borrowing").sum(axis=0)
    formatted = amounts.map(lambda value: f"{value / scale:,.3f}")
    formatted["% gross"] = amounts.Total.map(lambda value: f"{value / gross:.2%}")
    _table(fig.add_axes([.50, .59, .46, .27]), formatted,
           f"Asset allocation and borrowing ({unit})", first=.25, fontsize=8.5)
    fig.add_artist(Line2D(
        [.04, .96], [.53, .53], transform=fig.transFigure, color=INK, lw=.6))
    class_table = result.report_diagnostics["Asset class allocation with borrowing"]
    classes = class_table.gross_asset_share.copy()
    classes.index = [str(name).replace("Fixed Income", "Fixed\nIncome") for name in classes.index]
    colors = ["#315B7A", "#A64045", "#A18C68", "#5F8D83", "#766C91"]
    bar_ax = fig.add_axes([.09, .19, .36, .27])
    plot_bars(classes, ax=bar_ax, is_horizontal=False, stacked=False, colors=colors,
              title="Asset-class allocation", fontsize=10, x_rotation=0,
              add_bar_values=True, legend_loc=None, yvar_format="{:.1%}")
    bar_ax.set_ylabel("Share of gross assets", color=BLUE, fontsize=10)
    bar_ax.set_ylim(min(0., float(classes.min()) * 1.25), max(1., float(classes.max())) * 1.15)
    bar_ax.axhline(0, color=INK, lw=.7)
    bar_ax.grid(axis="y", color="#E4EAF0", linewidth=.5)
    currencies = displayed.drop(index="Borrowing").sum(axis=0)
    currencies = currencies.loc[currencies.gt(0)]
    pie_ax = fig.add_axes([.59, .16, .24, .32])
    plot_pie(currencies, ax=pie_ax, title="Currency allocation", colors=colors[:len(currencies)],
             autopct=None, fontsize=10)
    for label in pie_ax.texts:
        label.set_visible(False)
    wedges = pie_ax.patches
    pie_ax.legend(wedges, [f"{key}: {value / gross:.2%}" for key, value in currencies.items()],
                  loc="center left", bbox_to_anchor=(1., .5), frameon=False, fontsize=10)
    _note(fig, "Invested assets sum to 100% of gross assets. Borrowing is shown as a negative "
          "percentage of that same gross amount. Currency allocation uses invested assets; "
          f"all amounts are valued in {currency}. Up to five currency groups are displayed.")
    return fig
