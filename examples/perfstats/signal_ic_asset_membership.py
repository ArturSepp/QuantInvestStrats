"""Trace native and decision-calendar rank ICs for the handbook's monthly M and quarterly Q.

Run offline from the repository root with python -m examples.perfstats.signal_ic_asset_membership.
All returns are end-labelled log increments. The two-asset minimum illustrates membership.
"""
import numpy as np
import pandas as pd

from qis.perfstats.forward_return_targets import compute_forward_return_targets
from qis.perfstats.signal_diagnostics import compute_ic_timeseries, estimate_signal_diagnostics
from qis.perfstats.signal_ic import compute_matched_signal_ic


HORIZONS = (1, 3)
MIN_ASSETS = 2


def worked_inputs():
    """Return the handbook's end-labelled increments, dated scores and monthly decisions."""
    months = pd.date_range('2024-07-31', '2025-12-31', freq='ME')
    quarters = months[months.is_quarter_end]
    decisions = pd.date_range('2025-03-31', periods=4, freq='ME')
    mar, apr, may, jun = decisions

    monthly = pd.DataFrame(np.nan, index=months, columns=['M'])
    quarterly = pd.DataFrame(np.nan, index=quarters, columns=['Q'])
    monthly.loc[apr:jun, 'M'] = [0.02, -0.01, 0.03]
    quarterly.loc[jun:, 'Q'] = [0.06, -0.02, 0.01]
    score = pd.DataFrame(np.nan, index=months, columns=['M', 'Q'])
    score.loc[mar] = [0.8, -0.4]
    score.loc[may, 'M'] = 0.1
    return {'ME': monthly, 'QE': quarterly}, score, decisions


def show_members(title, matched, ic):
    """Print the finite score/return pairs and the QIS rank IC for this cross-section."""
    monthly = ', '.join(matched.loc[matched.cadence.eq('ME'), 'asset']) or '-'
    quarterly = ', '.join(matched.loc[matched.cadence.eq('QE'), 'asset']) or '-'
    value = f'{ic:+.2f}' if pd.notna(ic) else 'undefined'
    print(f'{title} | monthly: {monthly} | quarterly: {quarterly} | '
          f'n={len(matched)} | rank IC: {value}')
    for row in matched.itertuples(index=False):
        print(f'  {row.asset}  score {row.z:+.1f} on {row.formation_date:%d %b}  '
              f'log return {row.r:+.0%} through {row.realisation_date:%d %b}')


def show_native(returns, score, decisions):
    """Group pairs by the first return label, even when their holding periods differ."""
    # Retain singletons to display coverage. The IC itself still requires two assets.
    result = estimate_signal_diagnostics(returns, score, horizons=HORIZONS, is_log_returns=True,
                                         min_obs_per_date=1, is_vol_normalised=False)
    dated = compute_ic_timeseries(result, method='spearman', return_col='r',
                                 min_obs_per_date=MIN_ASSETS)
    print('\nNative periods | cross-sections by first return date')
    for h in HORIZONS:
        label = str(h)
        pairs = result.pairs[label].rename(columns={'asset_freq': 'cadence'}).copy()
        # Both score observations in this example lie exactly on their native period ends.
        for freq, group in pairs.groupby('cadence'):
            step = pd.tseries.frequencies.to_offset(freq)
            pairs.loc[group.index, 'formation_date'] = group.date - step
            pairs.loc[group.index, 'realisation_date'] = group.date + (h - 1) * step
        # All input grids start with the first common return/signal label in this example.
        # Include scheduled labels with no pairs, such as 31 May at h=1.
        sampled = pd.DatetimeIndex([])
        for frame in returns.values():
            sampled = sampled.union(frame.index[::h])
        for date in decisions[1:].intersection(sampled):
            matched = pairs.loc[pairs.date.eq(date)]
            ic = dated[label]['IC'].get(date, np.nan)
            show_members(f'h={h}  {date:%d %b}', matched, ic)


def show_decision_calendar(returns, score, decisions):
    """Join unlagged opening scores to common windows and list off-grid or missing names."""
    targets = compute_forward_return_targets(decisions, returns, horizons=HORIZONS)
    opening_scores = score.rename_axis(index='formation_date', columns='asset').stack().rename('z')
    panel = targets.join(opening_scores, on=['formation_date', 'asset']).assign(eligible=True)
    dated = compute_matched_signal_ic(panel, signals=['z'], targets=['raw'], reference_target='raw',
                                     method='spearman', min_obs_per_date=MIN_ASSETS)
    keys = ['horizon', 'formation_date', 'realisation_date']
    summary = dated.loc[dated['sample'].eq('available_raw')].set_index(keys)
    print('\nMonthly decisions | cross-sections by common holding window')
    for key, window in panel.groupby(keys, sort=True):
        h, opening, closing = key
        valid = window.eligible & np.isfinite(window.z) & np.isfinite(window.raw)
        matched = window.loc[valid].rename(columns={'raw': 'r'})
        assert len(matched) == summary.at[key, 'n_assets']
        show_members(f'h={h}  {opening:%d %b} to {closing:%d %b}', matched, summary.at[key, 'ic'])
        for row in window.loc[~valid].itertuples(index=False):
            reason = row.raw_status if pd.isna(row.raw) else 'missing formation score'
            print(f'  {row.asset} excluded: {reason}')


def main():
    """Print each cross-section for the 2025 March-to-June worked example."""
    returns, score, decisions = worked_inputs()
    print('2025 | M reports monthly, Q quarterly | minimum 2 matched assets for a rank IC')
    show_native(returns, score, decisions)
    show_decision_calendar(returns, score, decisions)


if __name__ == '__main__':
    main()
