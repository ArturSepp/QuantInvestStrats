"""Active-risk contributions of ``compute_benchmark_portfolio_risk_contributions``.

The function decomposes tracking error, TE = sqrt(d' S d) with d = w_p - w_b, into Euler
contributions d_i (S d)_i / TE that add up to TE, as ``RiskModel.compute_marginal_tre_at_date``
does. Both weight vectors are aligned to the covariance labels, and a zero benchmark or a zero
tracking error does not produce infinities.
"""

import numpy as np
import pandas as pd
import pytest

import qis


ASSETS = ['Equity', 'Credit', 'Hedge']
VOLS = np.array([0.20, 0.10, 0.15])
CORR = np.array([[1.0, 0.5, -0.6],
                 [0.5, 1.0, -0.2],
                 [-0.6, -0.2, 1.0]])
COVAR = pd.DataFrame(np.outer(VOLS, VOLS) * CORR, index=ASSETS, columns=ASSETS)
W_PORTFOLIO = pd.Series([0.5, 0.3, 0.2], index=ASSETS)
W_BENCHMARK = pd.Series([0.6, 0.4, 0.0], index=ASSETS)


def _euler_te_contributions(w_portfolio: pd.Series, w_benchmark: pd.Series) -> pd.Series:
    """Independent reference: d_i (S d)_i / sqrt(d' S d) by hand arithmetic on arrays.

    Args:
        w_portfolio: portfolio weights in covariance order.
        w_benchmark: benchmark weights in covariance order.

    Returns:
        Euler contributions to tracking error indexed by asset.
    """
    d = w_portfolio.to_numpy() - w_benchmark.to_numpy()
    sigma_d = COVAR.to_numpy() @ d
    return pd.Series(d * sigma_d / np.sqrt(d @ sigma_d), index=ASSETS)


def test_contributions_sum_to_tracking_error_and_match_risk_model() -> None:
    """The contributions are the Euler split of TE, not TE^2 scaled by benchmark volatility."""
    actual = qis.compute_benchmark_portfolio_risk_contributions(
        w_portfolio=W_PORTFOLIO, w_benchmark=W_BENCHMARK, covar=COVAR)

    # hand arithmetic: S d = (-0.0086, -0.0026, 0.0066) and d' S d = 0.00244
    te = np.sqrt(0.00244)
    np.testing.assert_allclose(actual.to_numpy(),
                               np.array([-0.1, -0.1, 0.2]) * np.array([-0.0086, -0.0026, 0.0066])
                               / te, atol=1e-15)
    assert actual.sum() == pytest.approx(te, abs=1e-15)
    pd.testing.assert_series_equal(actual, _euler_te_contributions(W_PORTFOLIO, W_BENCHMARK),
                                   check_names=False, atol=1e-15)

    date = pd.Timestamp('2024-12-31')
    mcte = qis.RiskModel(covar={date: COVAR}).compute_marginal_tre_at_date(
        benchmark_weights=W_BENCHMARK, portfolio_weights=W_PORTFOLIO, date=date)['mcte']
    np.testing.assert_allclose(actual.to_numpy(), mcte.to_numpy(), atol=1e-15)


def test_benchmark_is_aligned_to_the_covariance_labels() -> None:
    """A benchmark with a missing, an extra or a reordered label is aligned by label."""
    expected = _euler_te_contributions(W_PORTFOLIO, W_BENCHMARK)

    missing_hedge = W_BENCHMARK.drop('Hedge')  # a zero weight left out
    extra_label = pd.concat([W_BENCHMARK, pd.Series({'Outside': 0.0})])
    reordered = W_BENCHMARK[['Hedge', 'Credit', 'Equity']]
    for benchmark in (missing_hedge, extra_label, reordered):
        actual = qis.compute_benchmark_portfolio_risk_contributions(
            w_portfolio=W_PORTFOLIO, w_benchmark=benchmark, covar=COVAR)
        assert actual.index.tolist() == ASSETS
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-15)

    reordered_portfolio = W_PORTFOLIO[['Credit', 'Hedge', 'Equity']]
    actual = qis.compute_benchmark_portfolio_risk_contributions(
        w_portfolio=reordered_portfolio, w_benchmark=reordered, covar=COVAR)
    assert actual.index.tolist() == ASSETS
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-15)


def test_zero_benchmark_and_zero_tracking_error_are_finite() -> None:
    """A zero benchmark gives the portfolio's own Euler split; zero TE gives zeros."""
    zero_benchmark = pd.Series(0.0, index=ASSETS)
    actual = qis.compute_benchmark_portfolio_risk_contributions(
        w_portfolio=W_PORTFOLIO, w_benchmark=zero_benchmark, covar=COVAR)
    assert np.all(np.isfinite(actual.to_numpy()))
    np.testing.assert_allclose(
        actual.to_numpy(),
        qis.compute_portfolio_risk_contributions(w=W_PORTFOLIO, covar=COVAR).to_numpy(),
        atol=1e-15)

    identical = qis.compute_benchmark_portfolio_risk_contributions(
        w_portfolio=W_PORTFOLIO, w_benchmark=W_PORTFOLIO, covar=COVAR)
    pd.testing.assert_series_equal(identical, pd.Series(0.0, index=ASSETS))


def test_numpy_inputs_give_the_same_contributions() -> None:
    """Arrays are taken in covariance order and return an array."""
    actual = qis.compute_benchmark_portfolio_risk_contributions(
        w_portfolio=W_PORTFOLIO.to_numpy(), w_benchmark=W_BENCHMARK.to_numpy(),
        covar=COVAR.to_numpy())
    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, _euler_te_contributions(W_PORTFOLIO, W_BENCHMARK)
                               .to_numpy(), atol=1e-15)


def test_standalone_option_is_the_undiversified_bound() -> None:
    """``is_independent_risk=True`` returns |d_i| sigma_i, whose sum bounds TE from above."""
    standalone = qis.compute_benchmark_portfolio_risk_contributions(
        w_portfolio=W_PORTFOLIO, w_benchmark=W_BENCHMARK[['Hedge', 'Credit', 'Equity']],
        covar=COVAR, is_independent_risk=True)
    assert standalone.index.tolist() == ASSETS
    np.testing.assert_allclose(standalone.to_numpy(), [0.02, 0.01, 0.03], atol=1e-15)
    assert standalone.sum() >= np.sqrt(0.00244)


@pytest.mark.parametrize('carrier,dtype', [
    ('numpy', 'uint8'), ('labelled-array', 'uint16'), ('series', 'uint32'),
    ('series', 'UInt8'), ('series', 'UInt64'), ('mixed-boolean', 'UInt8'),
])
@pytest.mark.parametrize('standalone', [False, True])
def test_compute_benchmark_portfolio_risk_contributions_with_unsigned_weights(
        carrier: str, dtype: str, standalone: bool) -> None:
    """An active short remains negative after alignment, including mixed Boolean storage."""
    assets = pd.Index(['A', 'B'], name='asset')
    covar = pd.DataFrame([[.01, .005], [.005, .04]], index=assets, columns=assets)
    portfolio = pd.Series([1, 0], index=assets, dtype=dtype, name='portfolio')
    benchmark = pd.Series([1, 0], index=assets[::-1], dtype=dtype, name='benchmark')
    if carrier == 'mixed-boolean':
        benchmark = benchmark.astype(bool)
    if carrier in ('numpy', 'labelled-array'):
        portfolio = portfolio.to_numpy()
        benchmark = benchmark.iloc[::-1].to_numpy()
    if carrier == 'numpy':
        covar = covar.to_numpy()
    before = (portfolio.tolist(), benchmark.tolist(), portfolio.dtype, benchmark.dtype)

    actual = qis.compute_benchmark_portfolio_risk_contributions(
        portfolio, benchmark, covar, is_independent_risk=standalone)

    # d=(1,-1): S*d=(.005,-.035), variance=.04. Correlation exposes a lost sign.
    expected = [.1, .2] if standalone else [.025, .175]
    if carrier == 'numpy':
        assert isinstance(actual, np.ndarray)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    else:
        result_dtype = 'Float64' if dtype.startswith('UInt') else 'float64'
        pd.testing.assert_series_equal(
            actual, pd.Series(expected, index=assets, dtype=result_dtype),
            rtol=1e-13, atol=1e-15)
    assert (portfolio.tolist(), benchmark.tolist(), portfolio.dtype, benchmark.dtype) == before
    if isinstance(portfolio, pd.Series):
        assert tuple(portfolio.index) == ('A', 'B') and portfolio.index.name == 'asset'
        assert tuple(benchmark.index) == ('B', 'A') and benchmark.index.name == 'asset'
        assert portfolio.name == 'portfolio' and benchmark.name == 'benchmark'


@pytest.mark.parametrize('dtype', ['uint64', 'UInt64'])
@pytest.mark.parametrize('portfolio_level,benchmark_level,other_active', [
    (2**53 + 1, 2**53, 0), (2**63 + 1, 2**63, 0),
    (2**63, 2**63 + 1, 1), (2**64 - 1, 0, -1), (0, 2**64 - 1, 1),
])
@pytest.mark.parametrize('standalone', [False, True])
def test_compute_benchmark_portfolio_risk_contributions_keeps_wide_integer_differences(
        dtype: str, portfolio_level: int, benchmark_level: int,
        other_active: int, standalone: bool) -> None:
    """Subtract exactly before floating risk; compare wide operands before rounding them."""
    assets = ['A', 'B']
    portfolio = pd.Series([portfolio_level, max(other_active, 0)], index=assets, dtype=dtype)
    benchmark = pd.Series([benchmark_level, max(-other_active, 0)], index=assets, dtype=dtype)
    covar = pd.DataFrame([[.01, .005], [.005, .04]], index=assets, columns=assets)
    if dtype == 'uint64':
        portfolio, benchmark, covar = portfolio.to_numpy(), benchmark.to_numpy(), covar.to_numpy()

    actual = qis.compute_benchmark_portfolio_risk_contributions(
        portfolio, benchmark, covar, is_independent_risk=standalone)

    # Python integers retain the one-unit difference independently of NumPy/pandas promotion.
    d, other = portfolio_level - benchmark_level, other_active
    if standalone:
        expected = [abs(d) * .1, abs(other) * .2]
    else:
        te = np.sqrt(.01 * d * d + .01 * d * other + .04 * other * other)
        expected = [d * (.01 * d + .005 * other) / te,
                    other * (.005 * d + .04 * other) / te]
    np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=1e-13, atol=1e-15)
    if dtype == 'UInt64':
        assert isinstance(actual, pd.Series) and actual.dtype == 'Float64'
        assert actual.index.tolist() == assets


def test_compute_benchmark_portfolio_risk_contributions_keeps_standalone_precision() -> None:
    """Unsigned magnitude needs no signed promotion in the standalone branch."""
    actual = qis.compute_benchmark_portfolio_risk_contributions(
        np.array([1, 0], dtype=np.uint16), np.array([0, 1], dtype=np.uint16),
        np.diag(np.array([.01, .04], dtype=np.float32)), is_independent_risk=True)

    assert actual.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(actual, np.array([.1, .2], dtype=np.float32))


def test_compute_benchmark_portfolio_risk_contributions_retains_boolean_rejection() -> None:
    """Mixed unsigned/Boolean promotion must not admit a Boolean-only subtraction."""
    with pytest.raises(TypeError, match='boolean subtract'):
        qis.compute_benchmark_portfolio_risk_contributions(
            np.array([True, False]), np.array([False, True]), np.diag([.01, .04]))


@pytest.mark.parametrize('standalone', [False, True])
def test_compute_benchmark_portfolio_risk_contributions_preserves_sparse_integer_differences(
        standalone: bool) -> None:
    """Sparse maximum/minimum would round away a difference native subtraction retains."""
    portfolio = pd.Series([2**53 + 1, 0], index=['A', 'B'], dtype=pd.SparseDtype('uint64', 0))
    benchmark = pd.Series([2**53, 0], index=['A', 'B'], dtype=pd.SparseDtype('uint64', 0))
    covar = pd.DataFrame([[.01, .005], [.005, .04]], index=['A', 'B'], columns=['A', 'B'])

    actual = qis.compute_benchmark_portfolio_risk_contributions(
        portfolio, benchmark, covar, is_independent_risk=standalone)

    # The exact active position is (1, 0), so either risk mode gives (sqrt(.01), 0).
    pd.testing.assert_series_equal(
        actual, pd.Series([.1, 0.], index=['A', 'B'], dtype=pd.SparseDtype('float64', 0)),
        rtol=1e-13, atol=1e-15)
