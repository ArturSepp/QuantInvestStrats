"""Summarize saved ICs under three date-weighting policies, offline.

Run `python -m examples.perfstats.signal_ic_summary` from the repository root.
"""
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from qis.perfstats.signal_ic import compute_signal_ic_summary


class Locals(Enum):
    SAVED_ICS = 1


def run_local(local: Locals) -> None:
    """Print selected weights, effective dates and pointwise inference."""
    if local != Locals.SAVED_ICS:
        raise ValueError(local)
    history = pd.DataFrame({
        'realisation_date': pd.date_range('2022-01-31', periods=24, freq='ME'),
        'horizon': 1, 'sample': 'common', 'signal': 'momentum', 'target': 'raw',
        'ic': np.tile([0.25, -0.50, 0.75], 8),
    })
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'dated_ic.csv'
        history.to_csv(path, index=False)
        saved = pd.read_csv(path, parse_dates=['realisation_date'])

    cutoff = saved.realisation_date.iloc[2]
    reports = {
        'full': compute_signal_ic_summary(saved, as_of=cutoff),
        'rolling 2': compute_signal_ic_summary(saved, as_of=cutoff, rolling_span=2),
        'EWMA 3': compute_signal_ic_summary(saved, as_of=cutoff, ewma_span=3),
    }
    weights = pd.DataFrame({name: report.dated.set_index('realisation_date').weight
                            for name, report in reports.items()})
    stats = pd.concat({name: report.summary.iloc[0] for name, report in reports.items()}, axis=1).T
    np.testing.assert_allclose(stats.mean_ic.astype(float), [1 / 6, 1 / 8, 9 / 28])
    np.testing.assert_allclose(weights['EWMA 3'], np.array([1, 2, 4]) / 7)
    np.testing.assert_allclose(stats.loc['EWMA 3', 'effective_n'], 7 / 3)
    assert stats.inference.eq('insufficient_dates').all()

    short = compute_signal_ic_summary(saved, as_of=cutoff, ewma_span=3, warmup_periods=0)
    long = compute_signal_ic_summary(saved, ewma_span=24, warmup_periods=12)
    intervals = pd.concat([short.summary, long.summary], ignore_index=True)
    assert intervals.inference.eq('ok').all()
    columns = ['mean_ic', 'effective_n', 'ic_lower', 'ic_upper', 'inference']
    print('Normalized weights through the third realisation\n', weights.round(4))
    print('\nDefault warm-up of 12 effective dates\n', stats[columns].to_string())
    print('\nPointwise HAC intervals\n',
          intervals[['n_valid', 'ewma_span', 'warmup_periods', *columns]].round(4).to_string(
              index=False))


if __name__ == '__main__':
    run_local(local=Locals.SAVED_ICS)
