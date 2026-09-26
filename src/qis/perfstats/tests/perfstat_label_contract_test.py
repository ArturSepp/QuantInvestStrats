"""``PerfStat`` labels select the columns the tables actually produce.

Two label defects are covered:

- The regime-average members ``BEAR_AVG``, ``NORMAL_AVG`` and ``BULL_AVG`` were labelled
  'Bear Avg', 'Normal Avg' and 'Bull Avg', while the regime table names its average columns
  'Bear Average', 'Normal Average' and 'Bull Average'. Selecting a regime-average column by its
  member raised ``KeyError``. The member labels now match the table.
- ``ALPHA`` and ``ALPHA_AN`` shared the wrapped label 'Alpha', so a wide table containing both
  could not tell the periodic from the annualised intercept.
"""

from collections import Counter

import numpy as np
import pandas as pd

# qis
from qis.perfstats.config import PerfParams, PerfStat, REGIME_CONDITIONAL_PERFS
from qis.perfstats.regime_classifier import compute_bnb_regimes_pa_perf_table


def _prices() -> pd.DataFrame:
    """Ten years of month-end levels for a benchmark and one asset, from a fixed seed."""
    rng = np.random.default_rng(20260725)
    returns = 0.006 + 0.04 * rng.standard_normal((120, 2))
    dates = pd.date_range('2014-12-31', periods=121, freq='ME')
    levels = 100.0 * np.vstack([np.ones((1, 2)), np.cumprod(1.0 + returns, axis=0)])
    return pd.DataFrame(levels, index=dates, columns=['Benchmark', 'Asset'])


def test_every_regime_member_selects_a_regime_table_column() -> None:
    """Each regime-conditional PerfStat label is a column of the regime table."""
    table = compute_bnb_regimes_pa_perf_table(prices=_prices(), benchmark='Benchmark',
                                              perf_params=PerfParams(freq='ME'))
    for stat in REGIME_CONDITIONAL_PERFS:
        assert stat.to_str() in table.columns, stat
    averages = table[[PerfStat.BEAR_AVG.to_str(), PerfStat.NORMAL_AVG.to_str(),
                      PerfStat.BULL_AVG.to_str()]]
    assert list(averages.columns) == ['Bear Average', 'Normal Average', 'Bull Average']


def test_wrapped_labels_are_unique() -> None:
    """No two members share a wrapped header, in particular ALPHA and ALPHA_AN."""
    labels = [member.value.short_n for member in PerfStat if member.value.short_n is not None]
    assert [label for label, count in Counter(labels).items() if count > 1] == []
    assert PerfStat.ALPHA.to_str(short_n=True) != PerfStat.ALPHA_AN.to_str(short_n=True)
