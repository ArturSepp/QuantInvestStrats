"""``compute_performance_table`` accepts the Series its signature promises.

The function was annotated ``Union[pd.DataFrame, pd.Series]`` but raised ``TypeError`` for a
Series. A Series is now treated as a one-column frame named after it, as
``compute_ra_perf_table`` already does.
"""

import numpy as np
import pandas as pd
import pytest

# qis
from qis.perfstats.config import PerfParams, PerfStat
from qis.perfstats.perf_stats import compute_performance_table


def _prices() -> pd.Series:
    """A short deterministic month-end NAV."""
    dates = pd.date_range('2020-01-31', periods=30, freq='ME')
    return pd.Series(100.0 * np.exp(0.01 * np.arange(30) + 0.02 * np.sin(np.arange(30))),
                     index=dates, name='nav')


def test_series_equals_one_column_frame() -> None:
    """The Series result equals the result on the equivalent one-column frame."""
    prices = _prices()
    from_series = compute_performance_table(prices=prices, perf_params=PerfParams())
    from_frame = compute_performance_table(prices=prices.to_frame(), perf_params=PerfParams())
    pd.testing.assert_frame_equal(from_series, from_frame)
    total = prices.iloc[-1] / prices.iloc[0] - 1.0
    np.testing.assert_allclose(from_series.loc['nav', PerfStat.TOTAL_RETURN.to_str()], total,
                               rtol=1e-12)


def test_other_types_still_raise() -> None:
    """Anything that is neither a Series nor a DataFrame is rejected."""
    with pytest.raises(TypeError):
        compute_performance_table(prices=_prices().to_numpy(), perf_params=PerfParams())
