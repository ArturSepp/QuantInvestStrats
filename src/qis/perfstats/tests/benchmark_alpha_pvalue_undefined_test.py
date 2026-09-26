"""The benchmark row's alpha p-value is 1.0 only when its self-regression is defined.

``compute_ra_perf_table_with_benchmark`` regresses the benchmark on itself, which gives alpha 0,
beta 1 and an undefined t-statistic, so the table reports a p-value of 1.0 for it. When that
regression is itself undefined (a benchmark whose returns do not vary), the alpha, beta and R2
of the row are NaN and the p-value must be NaN too, not a confident 1.0.
"""

# packages
import numpy as np
import pandas as pd

# qis
import qis


def _prices(benchmark_returns: np.ndarray) -> pd.DataFrame:
    """Month-end prices of a benchmark with the given returns and of a noisy asset."""
    dates = pd.date_range('2020-01-31', periods=len(benchmark_returns) + 1, freq='ME')
    rng = np.random.default_rng(7)
    asset = np.concatenate([[1.0], np.cumprod(1.0 + rng.normal(0.005, 0.03, len(dates) - 1))])
    benchmark = np.concatenate([[1.0], np.cumprod(1.0 + benchmark_returns)])
    return pd.DataFrame({'bench': benchmark, 'asset': asset}, index=dates)


def test_benchmark_pvalue_is_one_for_a_defined_self_regression() -> None:
    """A varying benchmark keeps the documented p-value of 1.0 on its own row."""
    rng = np.random.default_rng(3)
    prices = _prices(rng.normal(0.004, 0.04, 36))
    table = qis.compute_ra_perf_table_with_benchmark(
        prices=prices, benchmark='bench', perf_params=qis.PerfParams(freq='ME', freq_reg='ME'))
    assert table.loc['bench', qis.PerfStat.ALPHA_PVALUE.to_str()] == 1.0


def test_benchmark_pvalue_is_missing_when_its_regression_is_undefined() -> None:
    """A constant benchmark return leaves every regression statistic of its row undefined."""
    prices = _prices(np.full(36, 0.01))
    table = qis.compute_ra_perf_table_with_benchmark(
        prices=prices, benchmark='bench', perf_params=qis.PerfParams(freq='ME', freq_reg='ME'))
    assert np.isnan(table.loc['bench', qis.PerfStat.BETA.to_str()])
    assert np.isnan(table.loc['bench', qis.PerfStat.ALPHA_PVALUE.to_str()])
