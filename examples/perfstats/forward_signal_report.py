"""Do low beta and trailing momentum rank raw and factor-adjusted returns?

Run `python -m examples.perfstats.forward_signal_report` for the offline tables.
The synthetic market has negative drift and the previous quarter's return rank
enters the next quarter's residual. All horizons count quarterly investment decisions.
"""
from enum import Enum

import numpy as np
import pandas as pd

from qis.datasets.synthetic import generate_synthetic_prices
from qis.perfstats.forward_return_targets import (
    ForwardReturnAdjustment, compute_forward_return_targets,
)
from qis.perfstats.signal_ic import compute_matched_signal_ic, compute_signal_ic_summary


class Locals(Enum):
    FORWARD_REPORT = 1


def synthetic_inputs():
    """Build two publication cadences, formation loadings and observed quarter returns."""
    prices = generate_synthetic_prices(start='2004-12-01', end='2024-12-31', apply_quirks=False)
    monthly = np.log(prices[['SEQ_US', 'SBD_TSY']].resample('ME').last()).diff()
    factors = pd.DataFrame({'market': 0.35 * monthly.SEQ_US - 0.008, 'bond': monthly.SBD_TSY})
    decisions = factors.resample('QE-DEC').last().index
    names = pd.Index([f'S{i:02}' for i in range(10)], name='asset')
    loadings = pd.DataFrame({'market': [1.25, -0.45, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.5],
                             'bond': [0.2, 0.5, 0.8, -0.4, 0, 1, 0.6, -0.6, 0.2, 0.4]},
                            index=names)
    increments = pd.DataFrame(np.nan, index=factors.index, columns=names)
    rng = np.random.default_rng(314159)
    previous = pd.Series(0.0, index=names)
    for start, end in zip(decisions[:-1], decisions[1:]):
        months = factors.index[(factors.index > start) & (factors.index <= end)]
        if start == decisions[0]:
            factors.loc[months, 'market'] = 0.020 / 3
        # Rank only returns already published at the preceding quarter-end.
        ranks = previous.rank()
        score = ranks - ranks.mean()
        noise = rng.normal(0.0, 0.018, size=(len(months), len(names)))
        asset_returns = factors.loc[months] @ loadings.T + noise + 0.0015 * score
        if start == decisions[0]:
            asset_returns.loc[:, names[0]] = 0.030 / 3
        increments.loc[months, :] = asset_returns
        previous = asset_returns.sum()

    quarterly = increments.resample('QE-DEC').sum(min_count=3)
    observed = {'ME': increments[names[:8]], 'QE-DEC': quarterly[names[8:]]}
    factor_grids = {'ME': factors, 'QE-DEC': factors.resample('QE-DEC').sum(min_count=3)}
    # A subsequent public snapshot omits one bond loading. The asset still has market coverage.
    partial = loadings.copy()
    partial.loc[names[-1], 'bond'] = np.nan
    adjustments = {
        'market_adjusted': ForwardReturnAdjustment(
            {freq: frame[['market']] for freq, frame in factor_grids.items()},
            {decisions[0]: loadings[['market']]}),
        'factor_adjusted': ForwardReturnAdjustment(
            factor_grids, {decisions[0]: loadings, decisions[-17]: partial}),
    }
    return decisions, observed, adjustments, quarterly, -loadings.market


def dated_ic_for_lookback(targets, quarterly, low_beta, lookback):
    """Join formation scores to completed windows and return the matched rank ICs by date."""
    momentum = quarterly.rolling(lookback).sum()
    scores = momentum.rename_axis(index='formation_date', columns='asset').stack().rename(
        'momentum').reset_index()
    scores['low_beta'] = scores.asset.map(low_beta)
    # All lookbacks start on the same date, after four observed quarters.
    scores['eligible'] = scores.formation_date >= quarterly.index[4]
    panel = targets.merge(scores, on=['formation_date', 'asset'], how='left',
                          validate='many_to_one')
    return compute_matched_signal_ic(
        panel, signals=['low_beta', 'momentum'], targets=['raw', 'market_adjusted',
                                                         'factor_adjusted'],
        reference_target='raw')


def report_for_lookback(targets, quarterly, low_beta, lookback):
    """Report the EWMA-weighted rank ICs for one momentum lookback."""
    dated = dated_ic_for_lookback(targets, quarterly, low_beta, lookback)
    return compute_signal_ic_summary(dated, as_of=quarterly.index[-1],
                                     ewma_span=24, warmup_periods=12).summary


def run_local(local: Locals) -> None:
    """Print common-sample results, a coverage comparison and a momentum lookback sweep."""
    if local != Locals.FORWARD_REPORT:
        raise ValueError(local)
    dates, observed, adjustments, quarterly, low_beta = synthetic_inputs()
    targets = compute_forward_return_targets(dates, observed, horizons=(1, 2, 3, 6),
                                             adjustments=adjustments)
    first = targets.loc[(targets.horizon == 1) & (targets.asset == 'S00') &
                        (targets.formation_date == dates[0])].iloc[0]
    np.testing.assert_allclose([first.raw, first.market_adjusted_explained, first.market_adjusted],
                               [0.030, 1.25 * 0.020, 0.005])

    summaries = pd.concat([report_for_lookback(targets, quarterly, low_beta, k).assign(lookback=k)
                           for k in (1, 2, 4)], ignore_index=True)
    common = summaries.query("lookback == 1 and sample == 'common'")
    columns = ['signal', 'horizon', 'target', 'mean_ic', 'ic_lower', 'ic_upper',
               'mean_ic_minus_reference', 'difference_lower', 'difference_upper',
               'mean_n_assets', 'effective_n', 'inference']
    print('Common assets, EWMA span 24, horizons in quarters\n',
          common[columns].sort_values(['signal', 'horizon', 'target']).round(3).to_string(
              index=False))
    coverage = summaries.query(
        "lookback == 1 and horizon == 1 and signal == 'low_beta' and target == 'raw'")
    print('\nRaw IC and coverage by sample\n',
          coverage[['sample', 'mean_n_assets', 'mean_ic']].round(3).to_string(index=False))
    sweep = summaries.query("sample == 'common' and target == 'factor_adjusted' and "
                            "signal == 'momentum' and weighting == 'ewma' and ewma_span == 24")
    print('\nFactor-adjusted momentum IC by lookback and forward horizon (quarters)\n',
          sweep.pivot(index='lookback', columns='horizon', values='mean_ic').round(3))


if __name__ == '__main__':
    run_local(local=Locals.FORWARD_REPORT)
