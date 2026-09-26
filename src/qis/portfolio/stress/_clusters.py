
"""Group detached holding valuations and canonical Euler risk by fitted asset clusters."""
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd


UNASSIGNED = "Unassigned modelled"
MULTIPLE = "Multi-cluster holdings"
OTHER = "Other clusters"


@dataclass(frozen=True)
class ClusterContributions:
    """Complete cluster audit tables and a bounded, additive presentation grouping."""

    holdings: pd.DataFrame
    summary: pd.DataFrame
    factor_exposures: pd.DataFrame
    factor_dollars: pd.DataFrame
    risk: pd.DataFrame
    scenario_pnl: Mapping[str, pd.DataFrame]
    scenario_nav: Mapping[str, pd.DataFrame]
    display_groups: pd.Series
    factor_risk: pd.DataFrame


def compute_cluster_contributions(result, memberships, max_groups=8):
    """Sum exact holding P&L, local factor exposures and canonical Euler allocations.

    Fitted memberships label shared responses. A holding is assigned to their unique
    cluster; holdings spanning clusters and holdings without a complete assignment
    stay in explicit buckets. No averaging of betas or repricing occurs here.
    """
    if max_groups < 3:
        raise ValueError("max_groups must accommodate special and remainder buckets")
    response_groups = {}
    for cadence, members in memberships.items():
        if not members.index.is_unique:
            raise ValueError("Cluster response IDs must be unique within each cadence")
        if set(members.index) - set(result.factor_loadings.index):
            raise ValueError("Cluster membership contains unknown response IDs")
        for response, label in members.items():
            if response in response_groups:
                raise ValueError("Cluster response IDs must belong to only one cadence")
            response_groups[response] = None if pd.isna(label) else f"{cadence}-{label}"

    assignments = {}
    for holding in result.positions.index:
        # Leg identities retain cluster membership when an option's current delta is zero.
        declared = set(result.leg_terms.loc[
            result.leg_terms.holding_id.eq(holding), "response_id"].dropna()) \
            if not result.leg_terms.empty else set()
        row = result.response_jacobian.loc[holding]
        declared.update(row.index[row.ne(0)])
        groups = {response_groups.get(response) for response in declared}
        assignments[holding] = (UNASSIGNED if not groups or None in groups else
                                next(iter(groups)) if len(groups) == 1 else MULTIPLE)
    groups = pd.Series(assignments, name="cluster")
    marks = result.positions.observed_mtm.reindex(groups.index)
    denominator = float(result.metadata["reporting_denominator"])
    holdings = pd.DataFrame({"cluster": groups, "net_mtm": marks,
                             "gross_mtm": marks.abs(), "holding_count": 1})
    summary = holdings.drop(columns="cluster").groupby(groups, sort=False).sum()
    summary["nav_weight"] = summary.net_mtm / denominator
    order = summary.gross_mtm.sort_values(ascending=False, kind="stable").index
    summary = summary.reindex(order)

    dollars = result.holding_factor_exposures.groupby(groups, sort=False).sum().reindex(order)
    factor_exposures = dollars / denominator
    systematic = result.report_diagnostics["Holding factor Euler volatility"].sum(axis=1)
    # RiskModel's response Euler terms use the supplied covariance denominator. Align
    # to the report's model-total volatility, then allocate by signed response shares.
    residual = result.response_risk_contributions.mcte_residual.copy()
    total_vol = float(result.risk.annual_factor_model_vol)
    target = float(result.risk.annual_residual_vol)**2 / total_vol if total_vol else 0.
    source_total = float(residual.sum())
    if source_total:
        residual *= target / source_total
    elif not np.isclose(target, 0., atol=1e-14):
        raise ValueError("Missing canonical residual Euler contributions")
    shares = result.response_jacobian.div(
        result.response_exposures.replace(0., np.nan), axis=1).fillna(0.)
    holding_residual = shares @ residual
    factor_risk = result.report_diagnostics["Holding factor Euler volatility"].groupby(
        groups, sort=False).sum().reindex(order)
    risk = pd.DataFrame({"Systematic": systematic, "Idiosyncratic": holding_residual})
    risk = risk.groupby(groups, sort=False).sum().reindex(order)
    risk["Total"] = risk.sum(axis=1)
    risk["variance_share"] = risk.Total / total_vol if total_vol else 0.

    pnl = {}
    valuations = dict(result.valuations)
    if result.historical is not None:
        valuations["historical"] = result.historical
    for name, valuation in valuations.items():
        pnl[name] = valuation.pnl.T.groupby(groups, sort=False).sum().reindex(order)
        np.testing.assert_allclose(pnl[name].sum(), valuation.portfolio_pnl, atol=1e-7)
    nav = {name: values / denominator for name, values in pnl.items()}
    np.testing.assert_allclose(dollars.sum(), result.factor_exposures, atol=1e-7)
    np.testing.assert_allclose(risk.Total.sum(), total_vol, atol=1e-12)

    display = pd.Series(order, index=order, name="display_cluster", dtype=object)
    if len(order) > max_groups:
        special = [name for name in order if name in (UNASSIGNED, MULTIPLE)]
        regular = [name for name in order if name not in special]
        keep = set(regular[:max_groups - len(special) - 1] + special)
        display.loc[~display.index.isin(keep)] = OTHER
    return ClusterContributions(holdings, summary, factor_exposures, dollars, risk,
                                pnl, nav, display, factor_risk)


def display_cluster_table(frame, contributions):
    """Collapse only the presentation remainder while preserving exact signed sums."""
    result = frame.groupby(contributions.display_groups, sort=False).sum(min_count=1)
    gross = contributions.summary.gross_mtm.groupby(
        contributions.display_groups, sort=False).sum()
    return result.reindex(gross.sort_values(ascending=False, kind="stable").index)



def cluster_top_contributors(result, contributions, displayed=False):
    """Rank three absolute holding P&Ls in each cluster's worst conditional scenario.

    The worst scenario is chosen over the whole conditional-comparison batch, not only the
    scenarios displayed on the cluster page, so it can lie beyond the displayed columns.
    """
    fields = ["holding_id", "name", "pnl", "nav_contribution"]
    columns = ["scenario", *fields, *[f"{field}_{rank}"
               for rank in (2, 3) for field in fields]]
    valuation = result.valuations.get("conditional")
    if valuation is None:
        return pd.DataFrame(columns=columns)
    assignments = contributions.holdings.cluster
    if displayed:
        assignments = assignments.map(contributions.display_groups)
        order = display_cluster_table(contributions.summary, contributions).index
    else:
        order = contributions.summary.index
    selections = {group: assignments.index[assignments.eq(group)] for group in order}
    scope = ("Modelled subtotal" if result.metadata.get("scope") == "modelled subtotal"
             else "Portfolio")
    selections[scope] = assignments.index
    name_col = "metadata:short_name" if "metadata:short_name" in result.positions else "name"
    rows = {}
    for group, ids in selections.items():
        scenario = valuation.pnl[ids].sum(axis=1).idxmin()
        ranked = valuation.pnl.loc[scenario, ids].abs().sort_values(
            ascending=False, kind="stable").head(3).index
        rows[group] = {"scenario": scenario}
        for rank, holding in enumerate(ranked, 1):
            suffix = "" if rank == 1 else f"_{rank}"
            pnl = float(valuation.pnl.loc[scenario, holding])
            rows[group].update({
                "holding_id" + suffix: holding,
                "name" + suffix: result.positions.loc[holding, name_col],
                "pnl" + suffix: pnl,
                "nav_contribution" + suffix: pnl/result.metadata["reporting_denominator"],
            })
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)
