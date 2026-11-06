"""Look-through exposure factsheets using the same partition as Brinson attribution."""
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from qis.portfolio.attribution.group_exposures import resolve_reporting_inputs

from qis.plots.bars import plot_bars
from qis.portfolio.attribution.group_exposures import compute_lookthrough_exposures
from qis.portfolio.multi_portfolio_data import MultiPortfolioData
from qis.utils.dates import TimePeriod


def _exposure_tables(
    multi_portfolio_data: MultiPortfolioData,
    group_loadings: Optional[pd.DataFrame] = None,
    time_period: Optional[TimePeriod] = None,
    *,
    group_data: Optional[pd.Series] = None,
    lookthrough_matrix: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, TimePeriod]:
    """Compare mean prior realised weights and common end-date realised holdings.

    Args:
        lookthrough_matrix: Optional static instrument-by-sleeve fractions. Supplied rows
            are finite, nonnegative and sum to one; omitted rows use group_data.
            None preserves categorical reporting. This is an accounting split.
        group_data: Ordinary instrument labels as a Series; defaults to portfolio metadata
            where available. Fractional memberships belong in lookthrough_matrix.
    """
    if group_loadings is not None:
        if lookthrough_matrix is not None:
            raise ValueError("Pass lookthrough_matrix or legacy group_loadings, not both")
        lookthrough_matrix = group_loadings
    group_data, lookthrough_matrix = resolve_reporting_inputs(
        multi_portfolio_data.portfolio_datas, group_data, lookthrough_matrix)
    portfolios = multi_portfolio_data.portfolio_datas
    if not portfolios:
        raise ValueError('Look-through factsheet requires at least one portfolio')
    names = [p.ticker or f'Portfolio {i + 1}' for i, p in enumerate(portfolios)]
    if len(set(names)) != len(names):
        raise ValueError('Look-through factsheet requires distinct portfolio names')
    navs = pd.concat([p.nav for p in portfolios], axis=1, join='inner', sort=True).dropna()
    if time_period is not None:
        navs = time_period.locate(navs)
    if len(navs) < 2:
        raise ValueError('Look-through factsheet requires at least two shared NAV dates')
    baseline, end = navs.index[0], navs.index[-1]
    averages, current = {}, {}
    native_index = None
    for name, portfolio in zip(names, portfolios):
        _, prior_weights = portfolio.get_brinson_inputs(freq=None)
        prior_weights = prior_weights.loc[
            (prior_weights.index > baseline) & (prior_weights.index <= end)]
        if native_index is not None and not prior_weights.index.equals(native_index):
            raise ValueError('Look-through comparisons require matching native return dates')
        native_index = prior_weights.index
        if prior_weights.empty:
            raise ValueError('Look-through factsheet has no shared return intervals')
        averages[name] = compute_lookthrough_exposures(
            prior_weights,
            lookthrough_matrix=lookthrough_matrix,
            group_data=portfolio.group_data if group_data is None else group_data,
        ).mean()
        current[name] = compute_lookthrough_exposures(
            portfolio.weights.loc[[end]],
            lookthrough_matrix=lookthrough_matrix,
            group_data=portfolio.group_data if group_data is None else group_data,
        ).iloc[0]
    return pd.DataFrame(averages), pd.DataFrame(current), TimePeriod(baseline, end)


def generate_lookthrough_exposures_factsheet(
    multi_portfolio_data: MultiPortfolioData,
    group_loadings: Optional[pd.DataFrame] = None,
    time_period: Optional[TimePeriod] = None,
    figsize: Tuple[float, float] = (11.7, 8.3),
    fontsize: int = 9,
    title: str = "Look-through portfolio exposures",
    notes: Optional[str] = None,
    *,
    group_data: Optional[pd.Series] = None,
    lookthrough_matrix: Optional[pd.DataFrame] = None,
) -> List[plt.Figure]:
    """Compare average and final realised sleeve exposures with grouped horizontal bars.

    Args:
        lookthrough_matrix: Optional static instrument-by-sleeve fractions. Supplied rows
            are finite, nonnegative and sum to one; omitted rows use group_data.
            None preserves categorical reporting. This is an accounting split.
        group_data: Ordinary instrument labels as a Series; defaults to portfolio metadata
            where available. Fractional memberships belong in lookthrough_matrix.
        multi_portfolio_data: Named portfolio histories on matching native return dates.
        group_loadings: Legacy alias for a complete lookthrough_matrix. Each instrument
            must have nonnegative fractions summing to one. The same matrix can be
            passed as lookthrough_matrix to MultiPortfolioData.compute_brinson_attribution.
        time_period: Requested NAV window. The first shared NAV is the baseline.
        figsize: Page dimensions in inches. Up to ten sleeves are shown per page.
        fontsize: Base label size in points.
        title: Page heading; the actual shared baseline and final NAV date are appended.
        notes: Optional mapping/provenance explanation printed on every page.

    Returns:
        Figures with native-observation mean beginning-of-period weights and final
        realised holdings. Exposures are decimal weights, not returns or risk shares.
        This is an accounting split using supplied static metadata, not a claim to
        observe underlying sleeve returns or historical changes in a fund's holdings.
    """
    average, current, period = _exposure_tables(
        multi_portfolio_data,
        group_loadings,
        time_period,
        lookthrough_matrix=lookthrough_matrix,
        group_data=group_data,
    )
    figures = []
    groups = average.index
    page_count = (len(groups) + 9) // 10
    per_page = (len(groups) + page_count - 1) // page_count
    for start in range(0, len(groups), per_page):
        selected = groups[start:start + per_page]
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.subplots_adjust(left=.21, right=.97, bottom=.25, top=.84, wspace=.70)
        fig.suptitle(f'{title}\n{period.start:%d%b%Y} - {period.end:%d%b%Y}',
                     fontsize=fontsize + 3, fontweight='bold', color='navy')
        for ax, table, heading in zip(axes, (average, current), (
                'Average beginning-of-period exposure',
                f'Realised exposure at {period.end:%d%b%Y}')):
            plot_bars(table.loc[selected], stacked=False, is_horizontal=True,
                      is_sns=False, annotate_totals=False, x_rotation=0,
                      xvar_format='{:.0%}', yvar_format='{:.0%}',
                      legend_loc='upper center', bbox_to_anchor=(.5, -.13),
                      title=heading, fontsize=fontsize,
                      xlabel='Portfolio weight', ax=ax)
        footer = ('Averages use native return intervals after the baseline; final bars use '
                  'realised holdings, including drift.\n'
                  'Static supplied fractions partition weights and P&L; they do not measure '
                  'separate underlying sleeve returns.')
        if notes:
            footer += '\n' + notes
        fig.text(.05, .045, footer, fontsize=fontsize - 1, va='bottom')
        figures.append(fig)
    return figures
