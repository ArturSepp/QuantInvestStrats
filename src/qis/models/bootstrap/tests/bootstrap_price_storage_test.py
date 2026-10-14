"""Price-bootstrap storage reuse, reconstruction, and ownership regressions.

Capture the native gather result to prove that float64 reconstruction replaces its draw slots
rather than retaining a second collection. Fixed indexes and an independent scalar compounding
oracle keep allocation savings from masking changed prices. Float32 gathers need a separate
float64 output list because Numba lists cannot change their element dtype.
"""

import math

import numpy as np
import pandas as pd
import pytest
from numba.typed import List

import qis.models.bootstrap.bootstrap_numba as bootstrap


INDEXES = np.array([[0, 2], [1, 0], [2, 1]], dtype=np.int64)


def _prices(dtype: str, as_series: bool = False) -> pd.Series | pd.DataFrame:
    """Build nonconstant prices whose supplied draw order differs across paths.

    Args:
        dtype: Native floating-point storage to exercise.
        as_series: Whether to use the first asset as a named Series.

    Returns:
        Dated prices with duplicate labels in the panel case.
    """
    prices = pd.DataFrame(
        [[100, 80], [105, 88], [99, 84], [108, 92]],
        index=pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
        columns=["asset", "asset"],
        dtype=dtype,
    )
    return prices.iloc[:, 0] if as_series else prices


def _expected_paths(prices, *, is_log_returns: bool, init_to_end: bool) -> list[np.ndarray]:
    """Derive the anchor and later growth without QIS return or NAV helpers.

    Args:
        prices: Finite source prices, including their native arithmetic precision.
        is_log_returns: Whether source returns are logarithmic.
        init_to_end: Whether to use the last rather than first price as anchor.

    Returns:
        One independently compounded array per supplied draw column.
    """
    values = prices.to_numpy()
    if values.ndim == 1:
        values = values[:, None]
    # Preserve float32 source arithmetic before the public function's float64 reconstruction.
    returns = np.diff(np.log(values), axis=0) if is_log_returns else (values[1:] / values[:-1] - 1)
    expected = []
    for draw in INDEXES.T:
        path = np.empty((len(draw), values.shape[1]), dtype=np.float64)
        for column in range(values.shape[1]):
            anchor = float(values[-1 if init_to_end else 0, column])
            path[0, column] = anchor
            for row in range(1, len(draw)):
                sampled = float(returns[draw[row], column])
                growth = math.exp(sampled) if is_log_returns else 1 + sampled
                path[row, column] = path[row - 1, column] * growth
        expected.append(path)
    return expected


@pytest.mark.parametrize("as_series", [False, True])
@pytest.mark.parametrize("is_log_returns", [False, True])
@pytest.mark.parametrize("init_to_end", [False, True])
def test_bootstrap_price_data_reuses_float64_draw_storage(
    monkeypatch, as_series: bool, is_log_returns: bool, init_to_end: bool
) -> None:
    """Reuse only private gather storage, preserving draw values and caller-owned inputs."""
    prices = _prices("float64", as_series)
    original = prices.copy(deep=True)
    indexes = INDEXES.copy()
    gathered = []
    native_bootstrap_data = bootstrap.bootstrap_data

    def capture_draws(**kwargs):
        """Observe the collection at its owning boundary without replacing the native gather."""
        draws = native_bootstrap_data(**kwargs)
        gathered.append(draws)
        return draws

    monkeypatch.setattr(bootstrap, "bootstrap_data", capture_draws)
    actual = bootstrap.bootstrap_price_data(
        prices,
        bootstrapped_indices=indexes,
        is_log_returns=is_log_returns,
        init_to_end=init_to_end,
        num_samples=99,
    )
    # Identity is the allocation regression: equal prices alone would allow both full lists.
    assert actual is gathered[0]
    assert isinstance(actual, List)
    assert len(actual) == INDEXES.shape[1]
    for path, expected in zip(
        actual, _expected_paths(prices, is_log_returns=is_log_returns, init_to_end=init_to_end)
    ):
        assert path.dtype == np.float64
        assert path.flags.c_contiguous
        np.testing.assert_allclose(path, expected, rtol=1e-13, atol=1e-13)

    # An active write detects shared buffers that equality and list identity cannot rule out.
    neighbor = actual[1].copy()
    actual[0][-1, 0] = -1
    np.testing.assert_array_equal(actual[1], neighbor)
    np.testing.assert_array_equal(indexes, INDEXES)
    if as_series:
        pd.testing.assert_series_equal(prices, original)
    else:
        pd.testing.assert_frame_equal(prices, original)


@pytest.mark.parametrize("is_log_returns", [False, True])
@pytest.mark.parametrize("init_to_end", [False, True])
def test_bootstrap_price_data_preserves_float32_gather_fallback(
    monkeypatch, is_log_returns: bool, init_to_end: bool
) -> None:
    """A float32 typed list must not receive reconstructed float64 arrays."""
    prices = _prices("float32")
    original = prices.copy(deep=True)
    gathered = []
    native_bootstrap_data = bootstrap.bootstrap_data

    def capture_draws(**kwargs):
        """Retain the original float32 draws to assert that the fallback leaves them intact."""
        draws = native_bootstrap_data(**kwargs)
        gathered.append((draws, [path.copy() for path in draws]))
        return draws

    monkeypatch.setattr(bootstrap, "bootstrap_data", capture_draws)
    actual = bootstrap.bootstrap_price_data(
        prices,
        bootstrapped_indices=INDEXES,
        is_log_returns=is_log_returns,
        init_to_end=init_to_end,
    )
    draws, sampled_copies = gathered[0]
    assert actual is not draws
    assert isinstance(actual, List)
    for sampled, original_sample in zip(draws, sampled_copies):
        assert sampled.dtype == np.float32
        np.testing.assert_array_equal(sampled, original_sample)
    for path, expected in zip(
        actual, _expected_paths(prices, is_log_returns=is_log_returns, init_to_end=init_to_end)
    ):
        assert path.dtype == np.float64
        np.testing.assert_allclose(path, expected, rtol=1e-13, atol=1e-13)
    pd.testing.assert_frame_equal(prices, original)


def test_bootstrap_price_data_preserves_empty_draw_list() -> None:
    """No draw can establish a Numba element dtype; preserve the native empty result."""
    actual = bootstrap.bootstrap_price_data(_prices("float64"), num_samples=0, index_length=3)
    assert isinstance(actual, List)
    assert len(actual) == 0
