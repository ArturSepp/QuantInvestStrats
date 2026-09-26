"""Regime exhibits are labelled truthfully and accept every regime ``PerfStat`` member.

- ``plot_regime_data`` defaulted to the title 'Conditional Excess Sharpe ratio', although no cash
  return is deducted in any regime convention. The default is now 'Conditional Sharpe ratio'.
- ``plot_ra_perf_scatter(x_var=PerfStat.BEAR_AVG)`` raised ``KeyError`` because the member was
  labelled 'Bear Avg' while the regime table names the column 'Bear Average'.
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# qis
from qis.perfstats.config import PerfParams, PerfStat  # noqa: E402
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantilesRegime  # noqa: E402
from qis.plots.derived.perf_table import plot_ra_perf_scatter  # noqa: E402
from qis.plots.derived.regime_data import plot_regime_data  # noqa: E402


def _prices() -> pd.DataFrame:
    """Ten years of month-end levels for a benchmark and two assets, from a fixed seed."""
    rng = np.random.default_rng(20260725)
    returns = 0.006 + 0.04 * rng.standard_normal((120, 3))
    dates = pd.date_range('2014-12-31', periods=121, freq='ME')
    levels = 100.0 * np.vstack([np.ones((1, 3)), np.cumprod(1.0 + returns, axis=0)])
    return pd.DataFrame(levels, index=dates, columns=['Benchmark', 'A', 'B'])


def test_default_regime_title_does_not_claim_excess_returns() -> None:
    """The default title names a conditional Sharpe ratio, not an excess one."""
    fig = plot_regime_data(regime_classifier=BenchmarkReturnsQuantilesRegime(),
                           prices=_prices(), benchmark='Benchmark', perf_params=PerfParams())
    titles = [ax.get_title() for ax in fig.axes]
    assert 'Conditional Sharpe ratio' in titles
    assert not any('Excess' in title for title in titles)
    plt.close(fig)


def test_scatter_accepts_regime_average_members() -> None:
    """BEAR_AVG selects the regime table's 'Bear Average' column."""
    fig = plot_ra_perf_scatter(prices=_prices(), benchmark='Benchmark',
                               perf_params=PerfParams(), x_var=PerfStat.BEAR_AVG,
                               y_var=PerfStat.PA_RETURN)
    assert len(fig.axes) >= 1
    plt.close(fig)
