"""Regression tests for per-series continuation anchors in price bootstraps.

Fixed sampled-return indexes separate price reconstruction from random sampling. The tests cover
trailing-ragged Series and DataFrame inputs, both public output constructions, arithmetic and log
returns, duplicate labels, and the public price-plus-fundamentals delegate.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

# packages
import numpy as np
import pandas as pd
import pytest

# qis / project
from qis.models.bootstrap.bootstrap_numba import (
    BootstrapOutput,
    BootstrapType,
    bootstrap_price_data,
    bootstrap_price_fundamental_data,
)

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

DATES = pd.date_range("2024-01-01", periods=5, freq="D")
SAMPLE_INDEXES: IntArray = np.array([[0], [1], [0]], dtype=np.int64)
COMPLETE_PRICES: FloatArray = np.array([100.0, 110.0, 121.0, 133.1, 146.41], dtype=np.float64)
RAGGED_PRICES: FloatArray = np.array([50.0, 55.0, 60.5, np.nan, np.nan], dtype=np.float64)


def _mixed_prices(*, duplicate_labels: bool = False) -> pd.DataFrame:
    """Construct complete and trailing-ragged price histories in one panel.

    Args:
        duplicate_labels: Whether both positional columns use the same label.

    Returns:
        Mixed price panel with complete and trailing-ragged columns.
    """
    labels = ["Asset", "Asset"] if duplicate_labels else ["Complete", "Ragged"]
    return pd.DataFrame(
        np.column_stack([COMPLETE_PRICES, RAGGED_PRICES]), index=DATES, columns=labels
    )


def _expected_path(values: FloatArray, *, is_log_returns: bool) -> FloatArray:
    """Recompound fixed sampled returns from the last positive finite source level.

    Args:
        values: One source price history.
        is_log_returns: Whether to derive logarithmic rather than arithmetic returns.

    Returns:
        Independently reconstructed continuation path.
    """
    valid_values = values[np.isfinite(values) & (values > 0.0)]
    if is_log_returns:
        source_returns = np.diff(np.log(valid_values))
        growth = np.exp(np.cumsum(source_returns[SAMPLE_INDEXES[:, 0]]))
    else:
        source_returns = valid_values[1:] / valid_values[:-1] - 1.0
        growth = np.cumprod(1.0 + source_returns[SAMPLE_INDEXES[:, 0]])
    return growth * (valid_values[-1] / growth[0])


def _first_list_path(result: object) -> FloatArray:
    """Narrow the public list result and return its first path as float64.

    Args:
        result: Result from the list-output bootstrap path.

    Returns:
        First rows-by-assets bootstrap path.
    """
    assert isinstance(result, Iterable)
    assert not isinstance(result, (pd.DataFrame, pd.Series))
    result_iterable = cast(Iterable[object], result)
    path = next(iter(result_iterable))
    assert isinstance(path, np.ndarray)
    return path.astype(np.float64, copy=False)


@pytest.mark.parametrize("is_log_returns", [False, True])
def test_bootstrap_price_data_anchors_mixed_ragged_columns_independently(
    is_log_returns: bool,
) -> None:
    """A longer neighbor cannot replace a ragged asset's own continuation anchor."""
    prices = _mixed_prices()
    original = prices.copy(deep=True)

    result = cast(
        object,
        bootstrap_price_data(
            prices=prices,
            bootstrap_output=BootstrapOutput.DF_TO_LIST_ARRAYS,
            num_samples=1,
            index_length=len(SAMPLE_INDEXES),
            is_log_returns=is_log_returns,
            bootstrapped_indices=SAMPLE_INDEXES,
            init_to_end=True,
        ),
    )
    actual = _first_list_path(result)
    expected = np.column_stack(
        [
            _expected_path(COMPLETE_PRICES, is_log_returns=is_log_returns),
            _expected_path(RAGGED_PRICES, is_log_returns=is_log_returns),
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    pd.testing.assert_frame_equal(prices, original)


@pytest.mark.parametrize(
    "bootstrap_output", [BootstrapOutput.DF_TO_LIST_ARRAYS, BootstrapOutput.SERIES_TO_DF]
)
@pytest.mark.parametrize("is_log_returns", [False, True])
def test_bootstrap_price_data_anchors_ragged_series_in_both_output_modes(
    bootstrap_output: BootstrapOutput, is_log_returns: bool
) -> None:
    """Both Series reconstruction paths continue from the last valid observed level."""
    prices = pd.Series(RAGGED_PRICES, index=DATES, name="Ragged")
    original = prices.copy(deep=True)

    result = cast(
        object,
        bootstrap_price_data(
            prices=prices,
            bootstrap_output=bootstrap_output,
            num_samples=1,
            index_length=len(SAMPLE_INDEXES),
            is_log_returns=is_log_returns,
            bootstrapped_indices=SAMPLE_INDEXES,
            init_to_end=True,
        ),
    )
    if bootstrap_output == BootstrapOutput.DF_TO_LIST_ARRAYS:
        actual = _first_list_path(result)[:, 0]
    else:
        assert isinstance(result, pd.DataFrame)
        actual = cast(FloatArray, result.iloc[:, 0].to_numpy(dtype=np.float64))

    np.testing.assert_allclose(
        actual,
        _expected_path(RAGGED_PRICES, is_log_returns=is_log_returns),
        rtol=1e-13,
        atol=1e-13,
    )
    pd.testing.assert_series_equal(prices, original)


def test_bootstrap_price_data_selects_ragged_anchors_by_column_position() -> None:
    """Duplicate labels do not merge or make per-column anchor selection ambiguous."""
    prices = _mixed_prices(duplicate_labels=True)

    result = cast(
        object,
        bootstrap_price_data(
            prices=prices,
            bootstrap_output=BootstrapOutput.DF_TO_LIST_ARRAYS,
            num_samples=1,
            index_length=len(SAMPLE_INDEXES),
            bootstrapped_indices=SAMPLE_INDEXES,
            init_to_end=True,
        ),
    )
    actual = _first_list_path(result)

    np.testing.assert_allclose(actual[0], [COMPLETE_PRICES[-1], 60.5])


@pytest.mark.parametrize("invalid_terminal", [np.nan, 0.0, -5.0, np.inf])
def test_bootstrap_price_data_skips_invalid_terminal_price_anchors(
    invalid_terminal: float,
) -> None:
    """Continuation anchors obey the positive-finite price endpoint domain."""
    prices = pd.Series([50.0, 55.0, 60.5, invalid_terminal], name="Ragged")
    sampled_indexes: IntArray = np.array([[0], [1], [0]], dtype=np.int64)

    result = cast(
        object,
        bootstrap_price_data(
            prices=prices,
            bootstrap_output=BootstrapOutput.DF_TO_LIST_ARRAYS,
            num_samples=1,
            index_length=len(sampled_indexes),
            bootstrapped_indices=sampled_indexes,
            init_to_end=True,
        ),
    )
    actual = _first_list_path(result)[:, 0]

    np.testing.assert_allclose(actual, [60.5, 66.55, 73.205])


def test_bootstrap_price_fundamental_data_preserves_ragged_price_anchors() -> None:
    """The public paired wrapper inherits per-series continuation anchoring."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {
            "Complete": [100.0, 105.0, 110.25, 115.7625, 121.550625, 127.62815625],
            "Ragged": [50.0, 52.5, 55.125, 57.88125, np.nan, np.nan],
        },
        index=dates,
    )
    fundamentals = pd.DataFrame(
        {
            "Complete": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            "Ragged": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        },
        index=dates,
    )

    bootstrapped_prices, _ = bootstrap_price_fundamental_data(
        price_datas={"prices": prices},
        fundamental_datas={"fundamentals": fundamentals},
        bootstrap_type=BootstrapType.FIXED_BLOCK,
        bootstrap_output=BootstrapOutput.DF_TO_LIST_ARRAYS,
        num_samples=1,
        index_length=3,
        block_size=1,
        seed=7,
    )
    actual = np.asarray(bootstrapped_prices["prices"][0], dtype=np.float64)

    np.testing.assert_allclose(actual[0], [127.62815625, 57.88125])
    assert np.isfinite(actual).all()
