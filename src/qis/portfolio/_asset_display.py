"""Bound factsheet displays without changing portfolio calculations.

Factsheets opt in with ``asset_tail_count`` (10 by default, or 20). Selection is by
signed value, including for one-sided data: the smallest observations are retained
as well as the largest. Ties keep input order. Non-finite scores are not ranked.
Totals are kept separately; only additive quantities may have an Other remainder.
"""
import numpy as np
import pandas as pd
import textwrap


def wrap_title(title, ax, fontsize):
    """Keep long metric/date labels inside their factsheet panel."""
    from qis.plots.utils import estimate_axis_width
    width = estimate_axis_width(ax) if ax is not None else None
    if title and width:
        chars = max(24, int(width * 72 / (max(fontsize, 1) * 0.55)))
        return '\n'.join(textwrap.fill(line, width=chars) for line in title.splitlines())
    return title


def factsheet_kwargs(kwargs):
    """Validate the display budget passed through existing report keyword arguments."""
    result = dict(kwargs)
    count = result.setdefault('asset_tail_count', 10)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count not in (10, 20):
        raise ValueError('asset_tail_count must be 10 or 20 (assets per tail)')
    return result


def tail_index(scores, count, keep=()):
    """Select disjoint bottom/top tails, retaining explicitly named totals separately."""
    protected = scores.index.intersection(keep, sort=False)
    candidates = scores.drop(index=protected)
    if count is None or len(candidates) <= 2 * count:
        return scores.index
    ranked = candidates.replace([np.inf, -np.inf], np.nan).dropna().sort_values(
        kind='stable')
    selected = ranked.iloc[:count].index.union(ranked.iloc[-count:].index, sort=False)
    return selected.append(protected)


def selection_title(title, total, shown, count, metric='value'):
    """Disclose the ranking and omitted population on the displayed panel."""
    if total <= shown:
        return title
    return (f'{title or ""}\nBottom/top {count} by {metric}; '
            f'{total - shown} omitted').strip()


def reduce_rows(data, scores, count, title=None, *, additive=False, keep=(), metric='value'):
    """Select rows, optionally aggregating omitted additive values without renormalising."""
    if count is None:
        return data, title
    scores = scores.reindex(data.index)
    # Display names can coincide (including shortened names). Select positions so
    # retaining one label cannot accidentally re-expand to hundreds of instruments.
    positions = pd.Series(scores.to_numpy(), index=np.arange(len(data)))
    protected = positions.index[data.index.isin(keep)]
    selected = tail_index(positions, count, keep=protected)
    if len(selected) == len(data):
        return data, title
    omitted = data.iloc[positions.index.difference(selected, sort=False)]
    result = data.iloc[selected].copy()
    title = selection_title(title, len(data), len(result), count, metric)
    if additive:
        label = f'Other ({len(omitted)})'
        while label in data.index:
            label = '_' + label
        # Preserve missing-only observations, rather than manufacturing zeros.
        result.loc[label] = omitted.sum(axis=0, min_count=1)
        is_total = result.index.isin(keep)
        result = pd.concat([result.loc[~is_total], result.loc[is_total]])
        title = title.replace('omitted', 'in Other')
    return result, title


def reduce_columns(data, count, title=None, *, additive=False, keep=(), scores=None):
    """Select complete time series by last available value, with a fixed membership."""
    if count is None or not isinstance(data, pd.DataFrame) or data.empty:
        return data, title
    if scores is None:
        scores = data.ffill().iloc[-1]
    reduced, title = reduce_rows(data.T, scores, count, title, additive=additive,
                                 keep=keep, metric='last value')
    return reduced.T, title


def plot_asset_bars(df, *args, asset_tail_count=None, asset_additive=False,
                    asset_scores=None, asset_totals=(), title=None, **kwargs):
    """Render bounded asset bars; direct non-factsheet callers retain their full data."""
    import qis
    scores = asset_scores
    if scores is None:
        scores = df if isinstance(df, pd.Series) else df.iloc[:, -1]
    original = df
    df, title = reduce_rows(df, scores, asset_tail_count, title,
                           additive=False, keep=asset_totals)
    if len(df) != len(original) and asset_additive:
        remainder = original.sum(axis=0, min_count=1) - df.sum(axis=0, min_count=1)
        fmt = kwargs.get('yvar_format', '{:.2%}')
        if isinstance(remainder, pd.Series):
            remainder = (remainder.sum(min_count=1) if kwargs.get('stacked', False)
                         else remainder.iloc[-1])
        title += f'; omitted sum {fmt.format(remainder)}'
    if len(df) != len(original) and 'totals' in kwargs:
        # Stacked attribution totals are additive across the columns in each row.
        kwargs['totals'] = df.sum(axis=1).tolist()
    if asset_tail_count is not None:
        title = wrap_title(title, kwargs.get('ax'), kwargs.get('fontsize', 10))
        if '%' in kwargs.get('yvar_format', '') and len(df):
            values = np.asarray(df, dtype=float)
            if np.nanmax(np.abs(values)) < 0.001:
                kwargs['yvar_format'] = '{:.4%}'
    return qis.plot_bars(df, *args, title=title, **kwargs)


def _series_kwargs(data, kwargs, count):
    """Use compact multi-column legends once a factsheet carries many assets."""
    import qis
    if count is not None and isinstance(data, pd.DataFrame) and data.shape[1] > 10:
        kwargs = dict(kwargs)
        kwargs['ncols'] = 5 if count == 20 else 3
        kwargs['legend_stats'] = qis.LegendStats.LAST_NONNAN
        kwargs['bbox_to_anchor'] = None
        kwargs['legend_loc'] = 'upper left'
        if '%' in kwargs.get('var_format', ''):
            kwargs['var_format'] = '{:.2%}'
    return kwargs


def _readable_legend(ax, count):
    """Keep names readable when the large palette includes pale line colours."""
    if count is not None and ax is not None:
        legend = ax.get_legend()
        if legend is not None and len(legend.get_texts()) > 10:
            for label in legend.get_texts():
                label.set_color('black')


def plot_asset_time_series(df, *args, asset_tail_count=None, asset_additive=False,
                           asset_totals=(), title=None, **kwargs):
    """Render selected series, adding a remainder only for explicitly additive inputs."""
    import qis
    df, title = reduce_columns(df, asset_tail_count, title, additive=asset_additive,
                              keep=asset_totals)
    kwargs = _series_kwargs(df, kwargs, asset_tail_count)
    if asset_tail_count is not None:
        title = wrap_title(title, kwargs.get('ax'), kwargs.get('fontsize', 10))
    result = qis.plot_time_series(df, *args, title=title, **kwargs)
    _readable_legend(kwargs.get('ax'), asset_tail_count)
    return result


def plot_asset_stack(df, *args, asset_tail_count=None, title=None, **kwargs):
    """Retain full stacked totals by adding the omitted assets as Other."""
    import qis
    df, title = reduce_columns(df, asset_tail_count, title, additive=True)
    kwargs = _series_kwargs(df, kwargs, asset_tail_count)
    if asset_tail_count is not None:
        title = wrap_title(title, kwargs.get('ax'), kwargs.get('fontsize', 10))
    result = qis.plot_stack(df, *args, title=title, **kwargs)
    _readable_legend(kwargs.get('ax'), asset_tail_count)
    return result


def price_tails(prices, count, title):
    """Select history-table assets by full-window simple return using observed endpoints."""
    scores = prices.ffill().iloc[-1].divide(prices.bfill().iloc[0]).subtract(1.0)
    selected = tail_index(scores, count)
    return prices.loc[:, selected], selection_title(
        title, prices.shape[1], len(selected), count, 'total return')


def compact_groups(aggregates, instruments, max_groups=6):
    """Use an all-instrument panel when individual groups would crowd a single page.

    Aggregate dictionaries contain additive Total/Net Long/Net Short series. The
    existing Total group, when present, already includes every other group.
    """
    if len(aggregates) <= max_groups:
        return aggregates, instruments
    if 'Total' in aggregates:
        aggregate = aggregates['Total']
        assets = instruments['Total']
    else:
        aggregate = pd.concat(aggregates.values()).groupby(level=0).sum(min_count=1)
        assets = pd.concat(instruments.values(), axis=1, sort=True)
    label = f'All instruments ({len(aggregates)} groups)'
    return {label: aggregate}, {label: assets}
