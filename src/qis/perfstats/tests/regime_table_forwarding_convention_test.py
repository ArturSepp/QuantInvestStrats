"""Regime tables honour their options and conventions on every call path.

Four defects are covered, each against an independent NumPy calculation on quarterly returns
compounded from literal monthly returns:

1. ``RegimeClassifier.compute_regimes_pa_perf_table`` documented
   ``additive_pa_returns_to_pa_total`` but did not forward it, and swallowed its other keywords
   (``is_report_pa_returns``), so the per-annum patch was always applied and the regime returns
   were always compounded.
2. With ``is_use_benchmark_means=True`` the benchmark's per-annum regime values are replaced by
   its periodic means for display. Under ``SharpeConvention.PA`` its regime Sharpe ratios were
   then computed from those periodic means over an annualised volatility.
3. The ``LOG`` convention applied ``log1p`` to the classifier's returns even when the classifier
   already produced log returns (``ReturnTypes.LOG``), and the ``ARITHMETIC`` convention then
   decomposed a log-return Sharpe ratio.
4. An empty regime was a missing value in the table but 0.0 in
   ``compute_regime_sharpe_decomposition``; both now report a missing value.
"""

import numpy as np
import pandas as pd

# qis
from qis.perfstats.config import (PerfParams, PerfStat, RegimeData, ReturnTypes,
                                  SharpeConvention)
from qis.perfstats.regime_classifier import (BenchmarkReturnsQuantilesRegime, RegimeClassifier,
                                             compute_regime_sharpe_decomposition,
                                             compute_regimes_pa_perf_table_from_sampled_returns)


_Q = np.array([0.0, 0.16, 0.84, 1.0])
_LABELS = ['Bear', 'Normal', 'Bull']


def _monthly_returns() -> np.ndarray:
    """120 months of benchmark and hedge returns from a fixed seed."""
    rng = np.random.default_rng(20260725)
    bench = 0.007 + 0.045 * rng.standard_normal(120)
    hedge = 0.003 - 0.30 * bench + 0.015 * rng.standard_normal(120)
    return np.column_stack([bench, hedge])


def _prices() -> pd.DataFrame:
    """Month-end levels from 2014-12-31, so the quarterly grid has no stub periods."""
    dates = pd.date_range('2014-12-31', periods=121, freq='ME')
    growth = np.vstack([np.ones((1, 2)), np.cumprod(1.0 + _monthly_returns(), axis=0)])
    return pd.DataFrame(100.0 * growth, index=dates, columns=['Benchmark', 'Hedge'])


def _quarterly() -> pd.DataFrame:
    """Quarterly simple returns compounded from the monthly returns with NumPy."""
    quarterly = (1.0 + _monthly_returns()).reshape(40, 3, 2).prod(axis=1) - 1.0
    index = pd.date_range('2015-03-31', periods=40, freq='QE')
    return pd.DataFrame(quarterly, index=index, columns=['Benchmark', 'Hedge'])


def _frequencies_and_means(returns: pd.DataFrame) -> tuple:
    """Regime frequencies and conditional means from a direct qcut of the benchmark."""
    labels = pd.qcut(_quarterly()['Benchmark'], q=_Q, labels=_LABELS)
    freqs = labels.value_counts(normalize=True).reindex(_LABELS).to_numpy()
    means = returns.groupby(labels, observed=False).mean().T.to_numpy()  # assets x regimes
    return freqs, means


def test_classifier_forwards_the_patch_switch() -> None:
    """additive_pa_returns_to_pa_total=False leaves the compounded contributions unpatched."""
    classifier = BenchmarkReturnsQuantilesRegime()
    _, datas = classifier.compute_regimes_pa_perf_table(
        prices=_prices(), benchmark='Benchmark', perf_params=PerfParams(),
        additive_pa_returns_to_pa_total=False)
    freqs, means = _frequencies_and_means(_quarterly())
    np.testing.assert_allclose(datas[RegimeData.REGIME_PA].to_numpy(),
                               np.expm1(4.0 * freqs * means), rtol=0.0, atol=1e-14)


def test_base_method_forwards_its_keywords() -> None:
    """The base method passes is_report_pa_returns through: linear contributions AN p_g m_g."""
    classifier = BenchmarkReturnsQuantilesRegime()
    prices = _prices()
    _, datas = RegimeClassifier.compute_regimes_pa_perf_table(
        classifier,
        regime_id_func_kwargs=dict(prices=prices, benchmark='Benchmark'),
        prices=prices, benchmark='Benchmark', freq='QE', perf_params=PerfParams(),
        additive_pa_returns_to_pa_total=False, is_report_pa_returns=False)
    freqs, means = _frequencies_and_means(_quarterly())
    np.testing.assert_allclose(datas[RegimeData.REGIME_PA].to_numpy(), 4.0 * freqs * means,
                               rtol=0.0, atol=1e-14)


def test_benchmark_means_do_not_enter_the_pa_regime_sharpe() -> None:
    """The display substitution leaves the benchmark's per-annum regime Sharpe ratios intact."""
    classifier = BenchmarkReturnsQuantilesRegime()
    prices = _prices()
    sampled = classifier.compute_sampled_returns_with_regime_id(prices=prices,
                                                                benchmark='Benchmark')
    results = {}
    for flag in (False, True):
        results[flag] = compute_regimes_pa_perf_table_from_sampled_returns(
            sampled_returns_with_regime_id=sampled, prices=prices, benchmark='Benchmark',
            perf_params=PerfParams(), freq='QE', is_use_benchmark_means=flag,
            regime_ids=_LABELS)
    plain, substituted = results[False][1], results[True][1]
    pd.testing.assert_frame_equal(substituted[RegimeData.REGIME_SHARPE],
                                  plain[RegimeData.REGIME_SHARPE])
    freqs, means = _frequencies_and_means(_quarterly())
    np.testing.assert_allclose(substituted[RegimeData.REGIME_PA].loc['Benchmark'], means[0],
                               rtol=0.0, atol=1e-14)
    # the patched per-annum values over the table volatility, computed independently
    table = results[False][0]
    vol = table[PerfStat.VOL.to_str()].to_numpy()
    pa_return = table[PerfStat.PA_RETURN.to_str()].to_numpy()
    compounded = np.expm1(4.0 * freqs * means)
    patched = compounded + (pa_return - compounded.sum(axis=1))[:, None] * freqs
    np.testing.assert_allclose(substituted[RegimeData.REGIME_SHARPE].to_numpy(),
                               patched / vol[:, None], rtol=0.0, atol=1e-13)


def test_log_classifier_is_not_logged_twice() -> None:
    """With a log-return classifier, LOG and ARITHMETIC decompose their own Sharpe ratios."""
    classifier = BenchmarkReturnsQuantilesRegime(return_type=ReturnTypes.LOG)
    simple = _quarterly()
    log = np.log1p(simple)
    for convention, returns in ((SharpeConvention.LOG, log),
                                (SharpeConvention.ARITHMETIC, simple)):
        _, datas = classifier.compute_regimes_pa_perf_table(
            prices=_prices(), benchmark='Benchmark',
            perf_params=PerfParams(sharpe_convention=convention))
        freqs, means = _frequencies_and_means(returns)
        std = returns.std(ddof=1).to_numpy()
        expected = 2.0 * freqs * means / std[:, None]
        np.testing.assert_allclose(datas[RegimeData.REGIME_SHARPE].to_numpy(), expected,
                                   rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(datas[RegimeData.REGIME_SHARPE].sum(axis=1),
                                   2.0 * returns.mean() / returns.std(ddof=1), atol=1e-13)


def test_empty_regime_is_missing_in_the_table_and_the_decomposition() -> None:
    """An asset unobserved in the Bear quarters has a missing Bear contribution on both paths."""
    quarterly = _quarterly()
    labels = pd.qcut(quarterly['Benchmark'], q=_Q, labels=_LABELS)
    gappy = quarterly.copy()
    gappy.loc[labels == 'Bear', 'Hedge'] = np.nan

    standalone = compute_regime_sharpe_decomposition(returns=gappy[['Hedge']],
                                                     benchmark_returns=gappy['Benchmark'],
                                                     af=4.0)
    assert np.isnan(standalone.loc['Hedge', 'Bear-Sharpe'])
    observed = gappy['Hedge'].dropna()
    total = 2.0 * observed.mean() / observed.std(ddof=1)
    np.testing.assert_allclose(standalone.loc['Hedge', 'Total-Sharpe'], total, atol=1e-14)
    regime_columns = [f'{label}-Sharpe' for label in _LABELS]
    np.testing.assert_allclose(np.nansum(standalone.loc['Hedge', regime_columns]), total,
                               atol=1e-14)

    sampled = gappy.copy()
    sampled['regime'] = labels
    prices = _prices()
    _, datas = compute_regimes_pa_perf_table_from_sampled_returns(
        sampled_returns_with_regime_id=sampled, prices=prices, benchmark='Benchmark',
        perf_params=PerfParams(sharpe_convention=SharpeConvention.ARITHMETIC), freq='QE',
        is_add_ra_perf_table=False, regime_ids=_LABELS)
    assert np.isnan(datas[RegimeData.REGIME_SHARPE].loc['Hedge', 'Bear-Sharpe'])
