"""
guards for the returns-level regime Sharpe decomposition proposed for qis 5.0.7:
table-branch parity, exact additivity under missing values, index-agnosticism,
the LOG convention, and the PA rejection
"""
# packages
import numpy as np
import pandas as pd
import pytest
# qis
import qis
from qis.perfstats.config import PerfParams, SharpeConvention

AF = 4.0  # quarterly


def make_reference_prices(n: int = 140, seed: int = 7) -> pd.DataFrame:
    """fixed reference series: benchmark plus two assets, quarterly"""
    rng = np.random.RandomState(seed)
    idx = pd.date_range('1990-12-31', periods=n, freq='QE')
    bench = pd.Series(0.015 + 0.05 * rng.randn(n), index=idx).add(1.0).cumprod().rename('Balanced')
    a1 = pd.Series(0.010 + 0.06 * rng.randn(n), index=idx).add(1.0).cumprod().rename('A1')
    a2 = pd.Series(0.008 + 0.04 * rng.randn(n), index=idx).add(1.0).cumprod().rename('A2')
    return pd.concat([bench, a1, a2], axis=1)

from qis.perfstats.regime_classifier import compute_regime_sharpe_decomposition


def test_standalone_equals_table_branch():
    """on an aligned panel without missing values, the standalone reproduces the
    ARITHMETIC regime branch of the table to machine precision"""
    prices = make_reference_prices()
    perf_params = PerfParams(freq='QE', sharpe_convention=SharpeConvention.ARITHMETIC)
    table = qis.compute_bnb_regimes_pa_perf_table(prices=prices, benchmark='Balanced', freq='QE',
                                                  q=np.array([0.0, 0.16, 0.84, 1.0]),
                                                  perf_params=perf_params)
    regime_columns = [c for c in table.columns
                      if 'Sharpe' in c and any(r in c for r in ('Bear', 'Normal', 'Bull'))]
    returns = prices.pct_change().dropna()
    standalone = compute_regime_sharpe_decomposition(returns=returns,
                                                     benchmark_returns=returns['Balanced'],
                                                     af=AF, q=np.array([0.0, 0.16, 0.84, 1.0]))
    diff = np.abs(table[regime_columns].to_numpy() - standalone[regime_columns].to_numpy())
    assert diff.max() < 1e-14


def test_standalone_additivity_with_missing_values():
    """per-asset moments make the decomposition exactly additive for any nan pattern"""
    prices = make_reference_prices()
    returns = prices.pct_change().dropna()
    ragged = returns.copy()
    ragged.loc[ragged.index[10:40], 'A1'] = np.nan  # a missing block in one asset
    standalone = compute_regime_sharpe_decomposition(returns=ragged[['A1', 'A2']],
                                                     benchmark_returns=returns['Balanced'], af=AF)
    regime_columns = [c for c in standalone.columns if 'Total' not in c]
    gap = np.abs(standalone[regime_columns].sum(axis=1) - standalone['Total-Sharpe'])
    assert gap.max() < 1e-12


def test_standalone_is_index_agnostic():
    """a RangeIndex must work: the function never touches the index type"""
    rng = np.random.RandomState(9)
    r_b = pd.Series(0.01 + 0.05 * rng.randn(120))
    r_a = pd.Series(0.008 + 0.06 * rng.randn(120))
    out = compute_regime_sharpe_decomposition(returns=r_a, benchmark_returns=r_b, af=4.0)
    regime_values = out[[c for c in out.index if 'Total' not in c]]
    assert abs(regime_values.sum() - out['Total-Sharpe']) < 1e-12


def test_standalone_log_convention():
    """LOG: the same construction on log1p(r), exactly additive to the log Sharpe"""
    prices = make_reference_prices()
    returns = prices.pct_change().dropna()
    out = compute_regime_sharpe_decomposition(returns=returns['A1'],
                                              benchmark_returns=returns['Balanced'],
                                              af=AF, sharpe_convention=SharpeConvention.LOG)
    log_r = np.log1p(returns['A1'])
    sr_log = float(np.sqrt(AF) * log_r.mean() / log_r.std(ddof=1))
    assert abs(out['Total-Sharpe'] - sr_log) < 1e-12


def test_standalone_rejects_pa():
    """PA does not decompose additively without the c-adjustment: point to the table path"""
    prices = make_reference_prices()
    returns = prices.pct_change().dropna()
    with pytest.raises(ValueError, match="c-adjusted table path"):
        compute_regime_sharpe_decomposition(returns=returns['A1'],
                                            benchmark_returns=returns['Balanced'],
                                            af=AF, sharpe_convention=SharpeConvention.PA)


def test_one_quantile_default_across_the_library():
    """the regime classifier and the returns-level standalone share the one-sigma default"""
    from qis.perfstats.regime_classifier import BenchmarkReturnsQuantilesRegime
    classifier = BenchmarkReturnsQuantilesRegime(freq='QE')
    np.testing.assert_array_equal(classifier.q, np.array([0.0, 0.16, 0.84, 1.0]))
    # the standalone default classifies identically to the classifier default
    prices = make_reference_prices()
    returns = prices.pct_change().dropna()
    standalone_default = compute_regime_sharpe_decomposition(returns=returns['A1'],
                                                             benchmark_returns=returns['Balanced'], af=AF)
    standalone_explicit = compute_regime_sharpe_decomposition(returns=returns['A1'],
                                                              benchmark_returns=returns['Balanced'], af=AF,
                                                              q=np.array([0.0, 0.16, 0.84, 1.0]))
    pd.testing.assert_series_equal(standalone_default, standalone_explicit)
