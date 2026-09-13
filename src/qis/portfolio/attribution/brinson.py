"""Canonical BHB sector attribution and Frongello linking for all QIS reports."""
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from qis.utils.df_groups import agg_df_by_groups_ax1


def _link_effects(
        effects: pd.DataFrame, strategy_returns: pd.Series, benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """Return Frongello-adjusted increments using only data through each date.

    Cumulative F[t] = (1 + rb[t]) * F[t-1] + P[t-1] * A[t].
    Reference: Frongello (2002), Linking single period attribution results;
    https://github.com/R-Finance/PortfolioAttribution/blob/master/R/Frongello.R.
    """
    if strategy_returns.le(-1.0).any() or benchmark_returns.le(-1.0).any():
        raise ValueError('Linked attribution requires returns greater than -100%')
    prior_strategy = (1.0 + strategy_returns).cumprod().shift(1, fill_value=1.0)
    benchmark_wealth = (1.0 + benchmark_returns).cumprod()
    cumulative = effects.mul(prior_strategy / benchmark_wealth, axis=0).cumsum().mul(
        benchmark_wealth, axis=0)
    adjusted = cumulative.diff()
    adjusted.iloc[0] = cumulative.iloc[0]
    return adjusted


def compute_brinson_attribution_table(
        benchmark_pnl: pd.DataFrame, strategy_pnl: pd.DataFrame,
        strategy_weights: pd.DataFrame, benchmark_weights: pd.DataFrame,
        asset_class_data: pd.Series, group_order: Optional[Sequence[str]] = None,
        total_column: str = 'Total Sum', is_exclude_interaction_term: bool = True,
        strategy_name: str = 'Strategy', benchmark_name: str = 'Benchmark',
        *, is_linked: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute BHB attribution from weighted arithmetic return contributions.

    Divide each grouped contribution by its corresponding beginning-of-period
    weight before applying allocation and selection formulas. Interaction is the
    residual; by default it is included in selection. A zero-weight group has
    zero defined sector return, with any residual P&L retained in interaction.

    Frongello linking is on by default: summing returned adjusted increments
    reproduces compounded strategy return minus compounded benchmark return at
    every date. Set is_linked=False for ordinary arithmetic period effects.
    The authoritative methodology is docs/brinson_attribution.md in the repository:
    https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/brinson_attribution.md.

    Args:
        benchmark_pnl: Benchmark instrument arithmetic return contributions, not
            unweighted instrument returns or currency P&L; rows sum to benchmark returns.
        strategy_pnl: Strategy contributions on the same basis and return dates.
        strategy_weights: Beginning-of-period strategy weights corresponding to the P&L.
        benchmark_weights: Beginning-of-period benchmark weights corresponding to the P&L.
        asset_class_data: Group label indexed by canonical instrument identifier.
        group_order: Preferred group order; other observed groups are appended.
        total_column: Name reserved for the total row and effect-series total column.
        is_exclude_interaction_term: Include interaction in selection when True.
        strategy_name: Strategy label in summary column headers.
        benchmark_name: Benchmark label in summary column headers.
        is_linked: Link period effects to compounded active return when True.

    Returns:
        Five DataFrames: totals table, aggregate effect increments, grouped allocation
        increments, grouped selection increments and grouped interaction increments.
        Use cumsum() on increments for curves. Return Total columns are contributions
        to each portfolio's compounded return; arithmetic mode uses Return Sum.

    Raises:
        ValueError: For empty/mismatched/duplicate dates, duplicate instruments,
            missing classifications, infinite inputs, a conflicting total label,
            or a linked portfolio return at or below -100%.
    """
    frames = (strategy_pnl, benchmark_pnl, strategy_weights, benchmark_weights)
    index = strategy_pnl.index
    if (len(index) == 0 or not index.is_unique or not index.is_monotonic_increasing
            or any(not frame.index.equals(index) for frame in frames)):
        raise ValueError('Brinson inputs require identical, ordered, unique return dates')
    if any(not frame.columns.is_unique for frame in frames) or not asset_class_data.index.is_unique:
        raise ValueError('Brinson instrument identifiers must be unique')
    assets = strategy_pnl.columns
    for frame in frames[1:]:
        assets = assets.union(frame.columns, sort=False)
    if len(assets) == 0:
        raise ValueError('Brinson requires at least one instrument')
    groups = asset_class_data.reindex(assets)
    if groups.isna().any():
        raise ValueError('Every attribution instrument requires an asset-class label')
    if groups.eq(total_column).any():
        raise ValueError('The Brinson total label cannot also be an asset-class label')
    grouped = []
    for frame in frames:
        aligned = frame.reindex(columns=assets).fillna(0.0)
        if not np.isfinite(aligned.to_numpy(dtype=float)).all():
            raise ValueError('Brinson inputs must be finite')
        grouped.append(agg_df_by_groups_ax1(
            aligned, group_data=groups, group_order=group_order))
    sp, bp, sw, bw = grouped
    sr = sp.div(sw.where(sw.ne(0.0))).fillna(0.0)
    br = bp.div(bw.where(bw.ne(0.0))).fillna(0.0)
    active = sp - bp
    allocation = (sw - bw) * br
    selection = bw * (sr - br)
    interaction = active - allocation - selection
    if is_exclude_interaction_term:
        selection = selection + interaction
        interaction = interaction * 0.0
    if is_linked:
        rp, rb = sp.sum(axis=1), bp.sum(axis=1)
        allocation = _link_effects(allocation, rp, rb)
        selection = _link_effects(selection, rp, rb)
        interaction = _link_effects(interaction, rp, rb)
        active = allocation + selection + interaction
        sp = sp.mul((1.0 + rp).cumprod().shift(1, fill_value=1.0), axis=0)
        bp = bp.mul((1.0 + rb).cumprod().shift(1, fill_value=1.0), axis=0)
    return_label = 'Return Total' if is_linked else 'Return Sum'
    totals = pd.DataFrame({
        f'{strategy_name}\nWeight Ave': sw.mean(),
        f'{benchmark_name}\nWeight Ave': bw.mean(),
        f'{strategy_name}\n{return_label}': sp.sum(),
        f'{benchmark_name}\n{return_label}': bp.sum(),
        'Asset\nAllocation': allocation.sum(),
        'Instrument\nSelection': selection.sum(),
    })
    if not is_exclude_interaction_term:
        totals['Interaction'] = interaction.sum()
    totals['Total\nActive'] = active.sum()
    totals.loc[total_column] = totals.sum()
    for effects in (allocation, selection, interaction):
        effects[total_column] = effects.sum(axis=1)
    active_total = pd.DataFrame({
        'Allocation Total': allocation[total_column], 'Selection Total': selection[total_column]})
    if not is_exclude_interaction_term:
        active_total['Interaction Total'] = interaction[total_column]
    return totals, active_total, allocation, selection, interaction
