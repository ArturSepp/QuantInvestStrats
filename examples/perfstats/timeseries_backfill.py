"""Extend a newer price series backwards with an older provider history.

With ``is_prices=True``, ``qis.bfill_timeseries`` joins in return space and reconstructs the
combined NAV at the newer provider's terminal level. The newer history is therefore unchanged,
while the older provider contributes only the missing early returns.

Run with ``python -m examples.perfstats.timeseries_backfill``.
"""

import matplotlib.pyplot as plt
import pandas as pd

import qis
from qis.datasets.synthetic import generate_synthetic_prices


def build_backfilled_history() -> tuple[pd.DataFrame, pd.Series]:
    """Return the two provider histories and their spliced price series."""
    underlying = generate_synthetic_prices(
        start='2005-01-03',
        end='2025-12-31',
        apply_quirks=False,
    )['SBD_TSY']
    older = (0.8 * underlying.loc[:'2018-12-31']).rename('Bond index')
    newer = (1.2 * underlying.loc['2014-01-01':]).rename('Bond index')
    backfilled = qis.bfill_timeseries(
        df_newer=newer,
        df_older=older,
        freq='B',
        is_prices=True,
    ).rename('Backfilled')
    providers = pd.concat([older.rename('Older provider'), newer.rename('Newer provider')], axis=1)
    return providers, backfilled


def run_example(show: bool = True) -> pd.Series:
    """Run the offline backfill example and optionally display its chart."""
    providers, backfilled = build_backfilled_history()
    newer = providers['Newer provider'].dropna()
    preserved_error = backfilled.reindex(newer.index).subtract(newer).abs().max()
    print(f'Maximum absolute error over newer history: {preserved_error:.3e}')
    print(f'History extended from {newer.index[0]:%d%b%Y} to {backfilled.index[0]:%d%b%Y}')

    normalized = pd.concat([providers, backfilled], axis=1)
    normalized = normalized.divide(normalized.apply(lambda x: x.dropna().iloc[0])).multiply(100.0)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    qis.plot_time_series(
        df=normalized,
        title='Price-history backfill across two providers',
        ylabel='Each series starts at 100',
        ax=ax,
    )
    if show:
        plt.show()
    else:
        plt.close(fig)
    return backfilled


if __name__ == '__main__':
    run_example()
