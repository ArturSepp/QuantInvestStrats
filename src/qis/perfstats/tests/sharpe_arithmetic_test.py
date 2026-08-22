"""Regression coverage for arithmetic Sharpe annualization inference.

The arithmetic convention is ``sqrt(periods_per_year) * mean(r) / std(r)`` on periodic simple
returns. Month-end observations imply exactly twelve periods per year, so the expected Series and
DataFrame results below can be calculated directly without calling a QIS annualization helper.
Both inferred and explicit annualization must preserve the same values, labels, and input data.
"""

import numpy as np
import pandas as pd

# qis
from qis.perfstats.perf_stats import compute_sharpe_arithmetic


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_DATES = pd.date_range('2024-01-31', periods=6, freq='ME')

_ASSET_A_RETURNS = (0.01, 0.02, -0.01, 0.03, 0.00, 0.01)
_ASSET_B_RETURNS = (-0.02, 0.01, 0.04, -0.01, 0.02, 0.03)

_ANNUALIZATION_FACTOR = 12.0
_TOLERANCE = 1e-12


def _monthly_returns() -> pd.DataFrame:
    """Create two labeled monthly simple-return histories.

    Returns:
        New DataFrame with a regular month-end index and two distinct return columns.
    """
    return pd.DataFrame(
        {
            'Asset A': _ASSET_A_RETURNS,
            'Asset B': _ASSET_B_RETURNS,
        },
        index=_DATES,
    )


# =============================================================================
# Automatic monthly annualization
# =============================================================================

def test_compute_sharpe_arithmetic_infers_monthly_series_factor() -> None:
    """Infer twelve periods per year for a monthly Series using sample volatility.

    The independently calculated reference is ``sqrt(12) * mean(r) / std(r, ddof=1)``. Automatic
    inference and explicit ``af=12`` must agree with that scalar while leaving the named input
    unchanged.
    """
    returns = _monthly_returns()['Asset A']
    original_returns = returns.copy(deep=True)
    values = np.asarray(_ASSET_A_RETURNS, dtype=float)
    expected = float(
        np.sqrt(_ANNUALIZATION_FACTOR)
        * np.mean(values)
        / np.std(values, ddof=1)
    )

    inferred = compute_sharpe_arithmetic(returns=returns)
    explicit = compute_sharpe_arithmetic(returns=returns, af=_ANNUALIZATION_FACTOR)

    assert isinstance(inferred, float)
    assert isinstance(explicit, float)
    np.testing.assert_allclose(inferred, expected, rtol=0.0, atol=_TOLERANCE)
    np.testing.assert_allclose(explicit, expected, rtol=0.0, atol=_TOLERANCE)
    pd.testing.assert_series_equal(returns, original_returns)


def test_compute_sharpe_arithmetic_infers_monthly_dataframe_factor() -> None:
    """Infer monthly scaling per DataFrame column while honoring population volatility.

    With ``ddof=0``, each independent reference is ``sqrt(12) * mean(r) / std(r, ddof=0)``.
    Automatic and explicit annualization must return the same Series in original column order and
    must not mutate the caller's DataFrame.
    """
    returns = _monthly_returns()
    original_returns = returns.copy(deep=True)
    values = np.asarray((_ASSET_A_RETURNS, _ASSET_B_RETURNS), dtype=float).T
    expected = pd.Series(
        np.sqrt(_ANNUALIZATION_FACTOR)
        * np.mean(values, axis=0)
        / np.std(values, axis=0, ddof=0),
        index=returns.columns,
    )

    inferred = compute_sharpe_arithmetic(returns=returns, ddof=0)
    explicit = compute_sharpe_arithmetic(
        returns=returns,
        af=_ANNUALIZATION_FACTOR,
        ddof=0,
    )

    assert isinstance(inferred, pd.Series)
    assert isinstance(explicit, pd.Series)
    pd.testing.assert_series_equal(inferred, expected, rtol=0.0, atol=_TOLERANCE)
    pd.testing.assert_series_equal(explicit, expected, rtol=0.0, atol=_TOLERANCE)
    pd.testing.assert_frame_equal(returns, original_returns)
