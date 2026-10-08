"""Last-state heatmaps omit history without changing analytics or rendered output."""

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numba.core.errors import TypingError

import qis
from qis.models.linear import ewm, plot_correlations as plots
from qis.perfstats.config import ReturnTypes
from qis.perfstats.returns import to_returns
from qis.utils.np_ops import tensor_mean


def _prices():
    """Small positive levels with changing correlations, suitable for a weighted-sum oracle."""
    return pd.DataFrame(
        [
            [100.0, 80.0, 40.0],
            [101.0, 79.0, 42.0],
            [99.0, 81.0, 41.0],
            [103.0, 80.0, 44.0],
            [102.0, 83.0, 43.0],
        ],
        index=pd.date_range("2020-01-01", periods=5, tz="UTC", name="observed"),
        columns=["first", "second", "third"],
    )


def _accepted_frame(prices, **parameters):
    """The complete tensor path is the compatibility reference, not the numerical oracle."""
    values = to_returns(
        prices,
        return_type=parameters.get("return_type", ReturnTypes.LOG),
        freq=parameters.get("freq"),
    ).to_numpy()
    seed = ewm.set_init_dim2(values, parameters.get("init_type", ewm.InitType.ZERO))
    history = ewm.compute_ewm_covar_tensor(
        values,
        span=parameters.get("span"),
        ewm_lambda=parameters.get("ewm_lambda", 0.94),
        covar0=seed,
        is_corr=True,
    )
    matrix = history[-1] if parameters.get("is_last", True) else tensor_mean(history)
    return pd.DataFrame(matrix, index=prices.columns, columns=prices.columns)


def _weighted_correlation(prices, decay):
    """Explicit weighted sums independently establish the zero-seeded final correlation."""
    values = to_returns(prices, return_type=ReturnTypes.LOG).to_numpy()
    covariance = np.zeros((values.shape[1], values.shape[1]))
    for i in range(values.shape[1]):
        for j in range(values.shape[1]):
            covariance[i, j] = math.fsum(
                (1.0 - decay) * decay ** (len(values) - 1 - t) * row[i] * row[j]
                for t, row in enumerate(values)
                if np.isfinite(row[i]) and np.isfinite(row[j])
            )
    vols = np.sqrt(np.diag(covariance))
    return pd.DataFrame(
        covariance / np.outer(vols, vols), index=prices.columns, columns=prices.columns
    )


def _capture_heatmap(monkeypatch):
    """Inspect exact pandas data at the renderer boundary, before display formatting."""
    monkeypatch.setattr(plots.phe, "plot_heatmap", lambda df, **kwargs: df)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_plot_returns_ewm_corr_table_last_avoids_history(monkeypatch, dtype):
    """A last-state request must not allocate the unused date-by-asset-by-asset tensor."""
    prices = _prices().astype(dtype)
    assert to_returns(prices, return_type=ReturnTypes.LOG).to_numpy().dtype == np.dtype(dtype)
    before = prices.copy(deep=True)
    expected = _accepted_frame(prices, ewm_lambda=0.5)

    def unexpected_history(*args, **kwargs):
        pytest.fail("last-state heatmaps must not allocate the full correlation history")

    monkeypatch.setattr(ewm, "compute_ewm_covar_tensor", unexpected_history)
    _capture_heatmap(monkeypatch)
    actual = qis.plot_returns_ewm_corr_table(prices, ewm_lambda=0.5)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    if dtype == "float64":
        # The independent sum checks the calculation; exact tensor parity also protects
        # float32's native intermediate rounding without relaxing the float64 oracle.
        pd.testing.assert_frame_equal(
            actual, _weighted_correlation(prices, 0.5), rtol=1e-14, atol=1e-14
        )
    pd.testing.assert_frame_equal(prices, before, check_exact=True)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"span": 4},
        {"ewm_lambda": np.float32(0.5)},
        {"ewm_lambda": 0.0},
        {"span": 1, "ewm_lambda": np.inf},
        {"init_type": ewm.InitType.X0},
        {"return_type": ReturnTypes.RELATIVE},
        {"freq": "2D"},
    ],
)
def test_plot_returns_ewm_corr_table_last_preserves_mixed_panel(monkeypatch, dtype, parameters):
    """Ragged, gappy, unavailable and constant neighbors keep exact data and duplicate labels."""
    prices = _prices().astype(dtype)
    prices.iloc[:2, 1] = np.nan
    prices.iloc[3, 2] = np.nan
    prices["unavailable"] = np.nan
    prices["constant"] = 100.0
    prices.columns = ["first", "same", "same", 3, "constant"]
    # New scalar columns default to float64; recast the complete panel so the float32
    # case exercises that dispatcher rather than silently converging to float64.
    prices = prices.astype(dtype)
    values = to_returns(
        prices,
        return_type=parameters.get("return_type", ReturnTypes.LOG),
        freq=parameters.get("freq"),
    ).to_numpy()
    assert values.dtype == np.dtype(dtype)
    before = prices.copy(deep=True)
    _capture_heatmap(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        expected = _accepted_frame(prices, **parameters)
        actual = qis.plot_returns_ewm_corr_table(prices, **parameters)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(prices, before, check_exact=True)


@pytest.mark.parametrize(
    "parameters",
    [
        {"is_last": False},
        {"span": np.array([3.0, 4.0, 5.0])},
        {"ewm_lambda": np.array([0.5, 0.6, 0.7])},
    ],
)
def test_plot_returns_ewm_corr_table_retains_history_dispatch(monkeypatch, parameters):
    """Averages retain history; vector smoothing retains native results or span rejection."""
    prices = _prices()
    before = prices.copy(deep=True)

    def unexpected_final(*args, **kwargs):
        pytest.fail("average or vector smoothing must retain native tensor dispatch")

    monkeypatch.setattr(ewm, "compute_ewm_covar", unexpected_final)
    _capture_heatmap(monkeypatch)
    if isinstance(parameters.get("span"), np.ndarray):
        # The tensor dispatcher already rejects array spans during type inference.
        # Preserve that boundary rather than making a storage optimization accept them.
        with pytest.raises(TypingError, match="Cannot unify") as accepted:
            _accepted_frame(prices, **parameters)
        with pytest.raises(TypingError, match="Cannot unify") as actual:
            qis.plot_returns_ewm_corr_table(prices, **parameters)
        assert str(actual.value) == str(accepted.value)
        pd.testing.assert_frame_equal(prices, before, check_exact=True)
        return
    expected = _accepted_frame(prices, **parameters)
    actual = qis.plot_returns_ewm_corr_table(prices, **parameters)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    if not parameters.get("is_last", True):
        assert not np.allclose(actual, _accepted_frame(prices), equal_nan=True)


@pytest.mark.parametrize(
    "parameters",
    [
        {"span": 0},
        {"span": np.nan},
        {"span": True},
        {"ewm_lambda": -0.1},
        {"ewm_lambda": 1.0},
        {"ewm_lambda": np.inf},
        {"init_type": ewm.InitType.MEAN},
    ],
)
def test_plot_returns_ewm_corr_table_preserves_parameter_errors(parameters):
    """Storage dispatch must not bypass or reinterpret native seed and smoothing validation."""
    prices = _prices()
    before = prices.copy(deep=True)
    with pytest.raises((ValueError, TypeError)) as accepted:
        _accepted_frame(prices, **parameters)
    with pytest.raises(type(accepted.value)) as actual:
        qis.plot_returns_ewm_corr_table(prices, **parameters)
    assert str(actual.value) == str(accepted.value)
    pd.testing.assert_frame_equal(prices, before, check_exact=True)


def test_plot_returns_ewm_corr_table_preserves_nullable_rejection():
    """Nullable missing returns keep the accepted object-array rejection, not a new conversion."""
    prices = _prices().astype("Float64")
    prices.iloc[2, 1] = pd.NA
    before = prices.copy(deep=True)
    for evaluate in (_accepted_frame, qis.plot_returns_ewm_corr_table):
        with pytest.raises(TypingError, match="non-precise type array\\(pyobject"):
            evaluate(prices)
        pd.testing.assert_frame_equal(prices, before, check_exact=True)


def test_plot_returns_ewm_corr_table_preserves_empty_error():
    """An empty history has no last matrix; the optimization must not invent one."""
    prices = _prices().iloc[:0]
    with pytest.raises(IndexError) as accepted:
        _accepted_frame(prices)
    with pytest.raises(IndexError) as actual:
        qis.plot_returns_ewm_corr_table(prices)
    assert str(actual.value) == str(accepted.value)


def test_plot_returns_ewm_corr_table_last_accepts_single_price_row(monkeypatch):
    """One price row is valid but has no observed return, so every correlation is undefined."""
    prices = _prices().iloc[:1]
    _capture_heatmap(monkeypatch)
    pd.testing.assert_frame_equal(
        qis.plot_returns_ewm_corr_table(prices), _accepted_frame(prices), check_exact=True
    )
    assert qis.plot_returns_ewm_corr_table(prices).isna().all().all()


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"annot": False, "transpose": True},
        {"var_format": "{:.3f}", "cmap": "coolwarm", "square": True},
    ],
)
def test_plot_returns_ewm_corr_table_last_preserves_canvas_and_supplied_axis(options):
    """Exact analytics must also survive label formatting, annotations and deferred rendering."""
    prices = _prices()
    expected = _accepted_frame(prices)
    reference_figure, reference_axis = plt.subplots()
    actual_figure, actual_axis = plt.subplots()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            plots.phe.plot_heatmap(
                expected,
                ax=reference_axis,
                var_format=options.get("var_format", "{:.0%}"),
                cmap=options.get("cmap", "PiYG"),
                **{
                    key: value
                    for key, value in options.items()
                    if key not in ("var_format", "cmap")
                },
            )
            result = qis.plot_returns_ewm_corr_table(prices, ax=actual_axis, **options)
            reference_figure.canvas.draw()
            actual_figure.canvas.draw()
        assert result is None
        assert plt.fignum_exists(actual_figure.number)
        np.testing.assert_array_equal(
            np.asarray(actual_figure.canvas.buffer_rgba()),
            np.asarray(reference_figure.canvas.buffer_rgba()),
        )
    finally:
        plt.close(reference_figure)
        plt.close(actual_figure)
