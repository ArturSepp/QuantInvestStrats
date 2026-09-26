"""EWM Sharpe ratio first row and the unit standard deviation of compute_ewm_std1_norm."""
# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.models.linear.ewm import compute_ewm_sharpe, compute_ewm_std1_norm, ewm_recursion


def test_ewm_sharpe_uses_the_first_return() -> None:
    """Row 0 enters the EWM mean: norm 0 is AN times (1 - lambda) r_0 at row 0."""
    returns = pd.DataFrame({'a': [0.1, 0.0, 0.0]},
                           index=pd.date_range('2020-01-31', periods=3, freq='ME'))
    sharpe = compute_ewm_sharpe(returns, span=3, norm_type=0)
    np.testing.assert_allclose(sharpe['a'], 12.0 * np.array([0.05, 0.025, 0.0125]))
    ratio = compute_ewm_sharpe(returns, span=3, norm_type=1)
    np.testing.assert_allclose(ratio['a'].iloc[0], np.sqrt(12.0) * np.sqrt(0.5))


@pytest.mark.parametrize('is_demean', [True, False])
def test_std1_norm_has_unit_standard_deviation_on_iid_input(is_demean: bool) -> None:
    """The name's promise holds with and without the same-span EWMA demeaning."""
    rng = np.random.default_rng(31)
    data = pd.DataFrame(rng.standard_normal((20_000, 24)))
    normalised = compute_ewm_std1_norm(data, span=260, is_demean=is_demean)
    pooled_std = np.sqrt(np.mean(np.square(normalised.iloc[2_000:].to_numpy())))
    assert abs(pooled_std - 1.0) < 0.04


def test_ewm_recursion_is_cached_on_disk() -> None:
    """The numba kernel is compiled with cache=True, so a new process skips recompilation."""
    assert type(ewm_recursion._cache).__name__ == 'FunctionCache'
