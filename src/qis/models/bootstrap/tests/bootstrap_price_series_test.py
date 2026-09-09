"""Price-bootstrap container and reconstruction boundary regressions.

``bootstrap_price_data`` accepts a Series or DataFrame and exposes two output containers. Fixed
sample indexes isolate container normalization and price reconstruction from the random bootstrap
kernels, while independent arithmetic and logarithmic formulas verify both return conventions.
The matrix also covers both documented price anchors and ordinary versus nullable finite storage.
"""

from __future__ import annotations

from collections.abc import Iterable

# packages
import numpy as np
import pandas as pd
import pytest

# qis / project
from qis.models.bootstrap.bootstrap_numba import BootstrapOutput, bootstrap_price_data

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

PRICE_VALUES: FloatArray = np.array([100.0, 102.0, 101.0, 104.0, 106.0], dtype=np.float64)
SAMPLE_INDEXES: IntArray = np.array([[0, 3], [1, 2], [2, 1], [3, 0]], dtype=np.int64)
PATH_COLUMNS = ["path_1", "path_2"]


def _price_series(*, nullable: bool) -> pd.Series:
    """Construct the same named dated prices with ordinary or nullable finite storage.

    Args:
        nullable: Whether to use pandas' nullable floating-point dtype.

    Returns:
        Finite price Series used by every container and reconstruction case.
    """
    prices = pd.Series(
        PRICE_VALUES,
        index=pd.date_range("2024-01-01", periods=len(PRICE_VALUES), freq="D"),
        name="Asset",
    )
    if nullable:
        prices = prices.astype(pd.Float64Dtype())
    return prices


def _expected_price_paths(
    prices: pd.Series, *, is_log_returns: bool, init_to_end: bool
) -> np.ndarray:
    """Reconstruct fixed sampled paths without using QIS return or NAV helpers.

    Args:
        prices: Source price history.
        is_log_returns: Whether to reconstruct from logarithmic rather than arithmetic returns.
        init_to_end: Whether to anchor the generated paths to the final source price.

    Returns:
        Expected rows-by-paths price matrix.
    """
    values = prices.to_numpy(dtype=float, na_value=np.nan)
    if is_log_returns:
        source_returns = np.diff(np.log(values))
        sampled_growth = np.exp(np.cumsum(source_returns[SAMPLE_INDEXES], axis=0))
    else:
        source_returns = values[1:] / values[:-1] - 1.0
        sampled_growth = np.cumprod(1.0 + source_returns[SAMPLE_INDEXES], axis=0)

    # The first sampled return establishes scale; subsequent returns evolve from the chosen anchor.
    anchor = values[-1] if init_to_end else values[0]
    return sampled_growth * (anchor / sampled_growth[0, :])


def _bootstrap_list(
    prices: pd.Series | pd.DataFrame, *, is_log_returns: bool, init_to_end: bool
) -> object:
    """Call the public list-output path with the fixed cross-container fixture.

    Args:
        prices: Series or one-column DataFrame representation.
        is_log_returns: Whether to reconstruct logarithmic returns.
        init_to_end: Whether to anchor paths to the final source price.

    Returns:
        Public list-output result retained as an object for explicit runtime narrowing.
    """
    return bootstrap_price_data(
        prices=prices,
        bootstrap_output=BootstrapOutput.DF_TO_LIST_ARRAYS,
        num_samples=len(PATH_COLUMNS),
        index_length=len(SAMPLE_INDEXES),
        is_log_returns=is_log_returns,
        bootstrapped_indices=SAMPLE_INDEXES,
        init_to_end=init_to_end,
    )


def _list_result_to_matrix(result: object) -> FloatArray:
    """Stack a list-output result into the rows-by-paths reference shape.

    Args:
        result: Public bootstrap result expected to contain one-column NumPy paths.

    Returns:
        Matrix with one generated path per column.
    """
    assert isinstance(result, Iterable)
    assert not isinstance(result, (pd.DataFrame, pd.Series))
    paths: list[FloatArray] = []
    for path in result:
        assert isinstance(path, np.ndarray)
        path_float: FloatArray = path.astype(np.float64, copy=False)
        paths.append(path_float)
    assert [path.shape for path in paths] == [(len(SAMPLE_INDEXES), 1)] * len(PATH_COLUMNS)
    return np.column_stack([path[:, 0] for path in paths])


@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("is_log_returns", [False, True])
@pytest.mark.parametrize("init_to_end", [False, True])
def test_bootstrap_price_data_supports_series_to_dataframe(
    nullable: bool, is_log_returns: bool, init_to_end: bool
) -> None:
    """Use positional anchors and preserve Series input across reconstruction conventions."""
    prices = _price_series(nullable=nullable)
    original = prices.copy(deep=True)
    expected = _expected_price_paths(prices, is_log_returns=is_log_returns, init_to_end=init_to_end)

    actual = bootstrap_price_data(
        prices=prices,
        bootstrap_output=BootstrapOutput.SERIES_TO_DF,
        num_samples=len(PATH_COLUMNS),
        index_length=len(SAMPLE_INDEXES),
        is_log_returns=is_log_returns,
        bootstrapped_indices=SAMPLE_INDEXES,
        init_to_end=init_to_end,
    )

    assert isinstance(actual, pd.DataFrame)
    assert actual.columns.tolist() == PATH_COLUMNS
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-13, atol=1e-13)
    pd.testing.assert_series_equal(prices, original)


@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("is_log_returns", [False, True])
@pytest.mark.parametrize("init_to_end", [False, True])
def test_bootstrap_price_data_preserves_one_column_list_parity(
    nullable: bool, is_log_returns: bool, init_to_end: bool
) -> None:
    """Return equivalent one-column list paths for Series and DataFrame representations."""
    prices = _price_series(nullable=nullable)
    frame = prices.to_frame()
    original_series = prices.copy(deep=True)
    original_frame = frame.copy(deep=True)
    expected = _expected_price_paths(prices, is_log_returns=is_log_returns, init_to_end=init_to_end)
    frame_result = _bootstrap_list(frame, is_log_returns=is_log_returns, init_to_end=init_to_end)
    series_result = _bootstrap_list(prices, is_log_returns=is_log_returns, init_to_end=init_to_end)
    frame_matrix = _list_result_to_matrix(frame_result)
    series_matrix = _list_result_to_matrix(series_result)

    np.testing.assert_allclose(frame_matrix, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(series_matrix, expected, rtol=1e-13, atol=1e-13)
    pd.testing.assert_series_equal(prices, original_series)
    pd.testing.assert_frame_equal(frame, original_frame)


def test_bootstrap_price_data_reconstructs_multi_asset_log_list() -> None:
    """Reconstruct distinct panel columns without pandas-only NumPy accumulation arguments."""
    prices = pd.DataFrame(
        {
            "Asset": PRICE_VALUES,
            "Second": np.array([50.0, 49.0, 51.0, 50.0, 52.0]),
        },
        index=pd.date_range("2024-01-01", periods=len(PRICE_VALUES), freq="D"),
    )
    original = prices.copy(deep=True)
    expected = np.stack(
        [
            _expected_price_paths(column, is_log_returns=True, init_to_end=True)
            for _, column in prices.items()
        ],
        axis=2,
    )

    result = _bootstrap_list(prices, is_log_returns=True, init_to_end=True)

    assert isinstance(result, Iterable)
    assert not isinstance(result, (pd.DataFrame, pd.Series))
    paths: list[FloatArray] = []
    for path in result:
        assert isinstance(path, np.ndarray)
        path_float: FloatArray = path.astype(np.float64, copy=False)
        paths.append(path_float)
    assert [path.shape for path in paths] == [(len(SAMPLE_INDEXES), len(prices.columns))] * len(
        PATH_COLUMNS
    )
    for path_index, path in enumerate(paths):
        np.testing.assert_allclose(path, expected[:, path_index, :], rtol=1e-13, atol=1e-13)
    pd.testing.assert_frame_equal(prices, original)


def test_bootstrap_price_data_rejects_dataframe_for_series_output_without_mutation() -> None:
    """Retain the established one-Series-only contract of ``SERIES_TO_DF``."""
    prices = _price_series(nullable=False).to_frame()
    original = prices.copy(deep=True)

    with pytest.raises(ValueError, match="data must be series"):
        bootstrap_price_data(
            prices=prices,
            bootstrap_output=BootstrapOutput.SERIES_TO_DF,
            num_samples=len(PATH_COLUMNS),
            index_length=len(SAMPLE_INDEXES),
            bootstrapped_indices=SAMPLE_INDEXES,
        )

    pd.testing.assert_frame_equal(prices, original)
