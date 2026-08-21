import warnings

import numpy as np
import pandas as pd

from qis.perfstats.config import PerfStat
from qis.perfstats.perf_stats import (compute_desc_freq_table)


def test_compute_desc_freq_table_sum_matches_explicit_period_sums():
    """The historical ``np.sum`` default stays warning-free and keeps resampler semantics."""
    index = pd.to_datetime([
        '2020-01-02', '2020-12-31',
        '2021-01-04', '2021-12-31',
        '2022-06-30', '2022-12-30',
    ])
    data = pd.DataFrame({
        'asset_a': [1.0, np.nan, 2.0, 4.0, 10.0, 20.0],
        'asset_b': [-1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
    }, index=index)

    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        actual = compute_desc_freq_table(df=data, freq='YE', agg_func=np.sum)

    period_totals = data.resample('YE').sum().dropna(axis=0, how='any')
    values = period_totals.to_numpy()
    expected = pd.DataFrame(index=period_totals.columns)
    expected[PerfStat.AVG.to_str()] = np.nanmean(values, axis=0)
    expected[PerfStat.STD.to_str()] = np.nanstd(values, ddof=1, axis=0)
    expected[PerfStat.QUANT_M_1STD.to_str()] = np.nanquantile(values, q=0.16, axis=0)
    expected[PerfStat.MEDIAN.to_str()] = np.nanmedian(values, axis=0)
    expected[PerfStat.QUANT_P1_STD.to_str()] = np.nanquantile(values, q=0.84, axis=0)
    pd.testing.assert_frame_equal(actual, expected)


    # new tests for refactored functionality
    # Additional visual checks live in perfstats/run_local/perf_stats_run.py.
