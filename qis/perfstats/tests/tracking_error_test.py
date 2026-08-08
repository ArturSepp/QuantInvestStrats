"""Tests for canonical EWMA realised tracking error."""
import numpy as np
import pandas as pd

import qis


def _nav_from_returns(returns: np.ndarray, scale: float = 1.0) -> pd.Series:
    index = pd.date_range('2000-01-31', periods=len(returns) + 1, freq='ME')
    nav = np.concatenate(([scale], scale * np.cumprod(1.0 + returns)))
    return pd.Series(nav, index=index)


def _tracking_error(return_diff: np.ndarray, span: int = 6) -> pd.Series:
    portfolio_nav = _nav_from_returns(return_diff)
    benchmark_nav = _nav_from_returns(np.zeros_like(return_diff))
    return qis.compute_ewma_realised_tracking_error(
        portfolio_nav=portfolio_nav,
        benchmark_nav=benchmark_nav,
        ewma_span=span,
    )


def test_constant_magnitude_difference_has_exact_monthly_annualisation() -> None:
    magnitude = 0.01
    return_diff = magnitude * np.tile([1.0, -1.0], 30)

    result = _tracking_error(return_diff=return_diff, span=6).dropna()

    np.testing.assert_allclose(result, magnitude * np.sqrt(12.0), rtol=1e-12, atol=0.0)


def test_tracking_error_is_homogeneous_in_return_difference() -> None:
    return_diff = np.tile([0.004, -0.007, 0.011], 20)
    base = _tracking_error(return_diff=return_diff, span=8)
    doubled = _tracking_error(return_diff=2.0 * return_diff, span=8)

    np.testing.assert_allclose(doubled, 2.0 * base, rtol=1e-10, atol=0.0)


def test_tracking_error_is_invariant_to_nav_levels() -> None:
    return_diff = np.tile([0.006, -0.009], 25)
    portfolio_nav = _nav_from_returns(return_diff)
    benchmark_nav = _nav_from_returns(np.zeros_like(return_diff))

    base = qis.compute_ewma_realised_tracking_error(portfolio_nav, benchmark_nav, ewma_span=5)
    rescaled = qis.compute_ewma_realised_tracking_error(
        17.0 * portfolio_nav,
        0.03 * benchmark_nav,
        ewma_span=5,
    )

    pd.testing.assert_series_equal(base, rescaled, rtol=1e-12, atol=0.0)


def test_every_span_converges_to_constant_magnitude_steady_state() -> None:
    magnitude = 0.012
    return_diff = magnitude * np.tile([1.0, -1.0], 50)

    final_values = [_tracking_error(return_diff=return_diff, span=span).iloc[-1]
                    for span in (2, 7, 24)]

    np.testing.assert_allclose(final_values, magnitude * np.sqrt(12.0),
                               rtol=1e-12, atol=0.0)


def test_identical_portfolio_and_benchmark_have_zero_tracking_error() -> None:
    returns = np.tile([0.01, -0.005, 0.002], 20)
    nav = _nav_from_returns(returns)

    result = qis.compute_ewma_realised_tracking_error(nav, nav, ewma_span=6).dropna()

    np.testing.assert_array_equal(result.to_numpy(), np.zeros(len(result)))


def test_warmup_length_responds_to_span() -> None:
    return_diff = np.tile([0.003, -0.004], 15)

    short = _tracking_error(return_diff=return_diff, span=3)
    long = _tracking_error(return_diff=return_diff, span=9)

    assert short.isna().sum() == 3
    assert long.isna().sum() == 9


def test_final_value_matches_explicit_ewma_variance_recursion() -> None:
    span = 5
    return_diff = np.array([0.003, -0.006, 0.011, -0.004, 0.008, -0.002,
                            0.007, -0.009, 0.005, -0.001, 0.004, -0.008])
    result = _tracking_error(return_diff=return_diff, span=span)

    alpha = 2.0 / (span + 1.0)
    # compute_ewm_vol's InitType.X0 seeds the variance with the first squared difference.
    variance = return_diff[0] ** 2
    for difference in return_diff[1:]:
        variance = (1.0 - alpha) * variance + alpha * difference ** 2
    expected = np.sqrt(12.0 * variance)

    np.testing.assert_allclose(result.iloc[-1], expected, rtol=1e-10, atol=0.0)
    assert result.name == 'Tracking error'
