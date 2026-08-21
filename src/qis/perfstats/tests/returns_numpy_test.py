"""NumPy coverage for converting returns to constant-trade-level NAVs.

``returns_to_nav`` accepts both pandas objects and NumPy arrays. A constant trade level changes
the economic calculation from geometric compounding to additive accumulation against a fixed
notional. These tests use small arrays whose expected paths can be calculated directly, keeping
the NumPy implementation independent from the already-tested pandas branch.
"""
# packages
import numpy as np
from numpy.typing import NDArray

# qis
from qis.perfstats.returns import returns_to_nav


# =============================================================================
# Shared deterministic references
# =============================================================================
_TOLERANCE = 1e-12


def _make_two_asset_returns() -> NDArray[np.float64]:
    """Create a two-asset return matrix with offsetting final observations.

    Returns:
        A new array with time on axis 0 and assets on axis 1.
    """
    return np.array([
        [0.00, 0.00],
        [0.10, 0.20],
        [-0.05, -0.10],
    ])


def _assert_array_close(actual: object, expected: NDArray[np.float64]) -> None:
    """Assert that a public conversion result is the expected NumPy array.

    Args:
        actual: Result returned under the current pandas-only output annotation.
        expected: Independently calculated array reference.
    """
    # The runtime already returns an ndarray for ordinary NumPy input, although the production
    # return annotation has not yet been widened to express that established behavior.
    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=_TOLERANCE,
        atol=_TOLERANCE,
    )


# =============================================================================
# Additive accumulation and shape
# =============================================================================
def test_numpy_constant_trade_level_accumulates_one_dimensional_returns() -> None:
    """Accumulate a one-dimensional return path against a fixed unit notional.

    A constant trade level adds each return to the same starting notional instead of multiplying
    gross-return factors. The direct cumulative returns are ``0.00, 0.10, 0.05``, so adding the
    unit starting NAV produces ``1.00, 1.10, 1.05``.
    """
    returns = np.array([0.00, 0.10, -0.05])
    expected_nav = np.array([1.00, 1.10, 1.05])

    actual_nav = returns_to_nav(
        returns=returns,
        constant_trade_level=True,
    )

    _assert_array_close(actual_nav, expected_nav)


def test_numpy_constant_trade_level_accumulates_each_column() -> None:
    """Accumulate a two-dimensional array independently along its time axis.

    Each column represents one asset. Axis-0 cumulative sums must finish at ``1.05`` and ``1.10``
    respectively, without flattening or combining the cross-section. The caller-owned matrix is
    also retained exactly because NAV conversion must not consume or rewrite source returns.
    """
    returns = _make_two_asset_returns()
    original_returns = returns.copy()
    expected_nav = np.array([
        [1.00, 1.00],
        [1.10, 1.20],
        [1.05, 1.10],
    ])

    actual_nav = returns_to_nav(
        returns=returns,
        constant_trade_level=True,
    )

    _assert_array_close(actual_nav, expected_nav)
    np.testing.assert_array_equal(returns, original_returns)


def test_numpy_constant_trade_level_preserves_and_resumes_missing_histories() -> None:
    """Preserve missing observations without contaminating later additive NAVs.

    The columns cover a return missing between observations, a ragged start plus a later gap,
    and a history that never starts. Each observed return accumulates against its column's fixed
    unit notional, while the caller-owned missing observations remain missing in the output.
    """
    returns = np.array([
        [0.00, np.nan, np.nan],
        [0.10, 0.20, np.nan],
        [np.nan, -0.10, np.nan],
        [0.05, np.nan, np.nan],
        [-0.02, 0.03, np.nan],
    ])
    original_returns = returns.copy()
    expected_nav = np.array([
        [1.00, np.nan, np.nan],
        [1.10, 1.20, np.nan],
        [np.nan, 1.10, np.nan],
        [1.15, np.nan, np.nan],
        [1.13, 1.13, np.nan],
    ])

    actual_nav = returns_to_nav(
        returns=returns,
        constant_trade_level=True,
    )

    _assert_array_close(actual_nav, expected_nav)
    np.testing.assert_array_equal(returns, original_returns)


# =============================================================================
# Option composition
# =============================================================================
def test_numpy_constant_trade_level_scales_each_column_to_initial_value() -> None:
    """Scale each additive array column from one to its requested initial value.

    Before scaling, the two columns follow unit-based paths ending at ``1.05`` and ``1.10``.
    Multipliers of 100 and 200 therefore produce terminal values of 105 and 220. This verifies
    that the new accumulation branch continues through the existing per-column scaling path.
    """
    returns = _make_two_asset_returns()
    expected_nav = np.array([
        [100.0, 200.0],
        [110.0, 240.0],
        [105.0, 220.0],
    ])

    actual_nav = returns_to_nav(
        returns=returns,
        init_value=np.array([100.0, 200.0]),
        constant_trade_level=True,
    )

    _assert_array_close(actual_nav, expected_nav)


def test_numpy_constant_trade_level_converts_log_returns_before_accumulating() -> None:
    """Convert log returns to simple returns before fixed-notional accumulation.

    Log gross factors ``1.00, 1.10, 0.95`` represent simple returns ``0.00, 0.10, -0.05``.
    Additive accumulation must therefore reproduce the direct ``1.00, 1.10, 1.05`` reference,
    rather than geometrically compounding the factors to ``1.045``.
    """
    log_returns = np.log(np.array([1.00, 1.10, 0.95]))
    expected_nav = np.array([1.00, 1.10, 1.05])

    actual_nav = returns_to_nav(
        returns=log_returns,
        constant_trade_level=True,
        is_log_returns=True,
    )

    _assert_array_close(actual_nav, expected_nav)
