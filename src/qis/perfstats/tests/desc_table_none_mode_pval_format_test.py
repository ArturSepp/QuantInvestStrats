"""``compute_desc_table`` handles ``DescTableType.NONE`` and formats the p-value by its member.

- ``DescTableType.NONE`` is the "no table" setting that the plotting functions use. Passed to
  ``compute_desc_table`` it raised ``TypeError``; it now returns a table indexed by ticker with no
  statistic columns.
- The normality p-value was printed with ``'{:.2f}'`` although ``PerfStat.NORMTEST`` carries
  ``ValueType.FLOAT4``. It is now formatted with the member's own format, four decimals, so a
  p-value of 0.004 no longer prints as 0.00.
"""

import numpy as np
import pandas as pd
from scipy import stats

# qis
from qis.perfstats.config import PerfStat
from qis.perfstats.desc_table import DescTableType, compute_desc_table


def _returns() -> pd.DataFrame:
    """Sixty deterministic returns for two columns, one skewed."""
    rng = np.random.default_rng(7)
    normal = 0.01 * rng.standard_normal(60)
    skewed = 0.01 * (rng.exponential(size=60) - 1.0)
    return pd.DataFrame({'normal': normal, 'skewed': skewed},
                        index=pd.date_range('2020-01-31', periods=60, freq='ME'))


def test_none_mode_returns_an_empty_table_indexed_by_ticker() -> None:
    """NONE reports no statistics but keeps the ticker index."""
    table = compute_desc_table(df=_returns(), desc_table_type=DescTableType.NONE)
    assert list(table.index) == ['normal', 'skewed']
    assert table.shape == (2, 0)
    series_table = compute_desc_table(df=_returns()['normal'], desc_table_type=DescTableType.NONE)
    assert list(series_table.index) == ['normal'] and series_table.shape == (1, 0)


def test_normality_p_value_uses_the_member_format() -> None:
    """The P-val column has four decimals, the FLOAT4 format of PerfStat.NORMTEST."""
    data = _returns()
    table = compute_desc_table(df=data, desc_table_type=DescTableType.WITH_NORMAL_PVAL)
    expected = [PerfStat.NORMTEST.to_format().format(p) for p in stats.normaltest(data)[1]]
    assert PerfStat.NORMTEST.to_format() == '{:.4f}'
    assert list(table[PerfStat.NORMTEST.to_str()]) == expected
    assert all(len(value.split('.')[1]) == 4 for value in table[PerfStat.NORMTEST.to_str()])
