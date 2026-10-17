"""Series bootstrap frame construction, native fallbacks and ownership contracts.

The allocation regression forbids the list-of-draws gather only for ordinary numeric frames.
Literal row selections independently verify the output, while fallback cases compare with the
original per-column construction so batching cannot silently widen Numba's accepted input domain.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import qis
import qis.models.bootstrap.bootstrap_numba as bootstrap


INDEXES = np.array([[3, 0], [1, 3], [0, 1]], dtype=np.int64)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "uint64", "bool"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_bootstrap_data_batches_numeric_series_frames(monkeypatch, dtype, index_dtype):
    """One gathered matrix preserves exact values/dtypes without per-draw frame assembly."""
    values = [0, 1, 2, 3] if dtype != "bool" else [False, True, True, False]
    source = pd.Series(values, dtype=dtype, name="asset")
    original = source.copy(deep=True)
    indexes = INDEXES.astype(index_dtype)
    original_indexes = indexes.copy()

    def unexpected_list_gather(**kwargs):
        pytest.fail("numeric Series frames must not allocate a list of separate draws")

    monkeypatch.setattr(bootstrap, "get_bootstrap_data_list", unexpected_list_gather)
    actual = qis.bootstrap_data(
        source,
        bootstrap_output=qis.BootstrapOutput.SERIES_TO_DF,
        bootstrapped_indices=indexes,
        num_samples=99,
    )
    expected = pd.DataFrame(source.to_numpy()[indexes], columns=["path_1", "path_2"])
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_series_equal(source, original)
    np.testing.assert_array_equal(indexes, original_indexes)


@pytest.mark.parametrize("layout", ["fortran", "reversed"])
def test_bootstrap_data_series_frames_preserve_strides_missingness_and_ownership(layout):
    """Selection is positional; input dates, missing values and output edits cannot alias inputs."""
    source = pd.Series(
        [np.nan, -0.0, np.inf, -np.inf, 5.0, 8.0],
        index=pd.date_range("2024-01-01", periods=6, tz="UTC", name="date")[::-1],
        name="asset",
    ).iloc[::2]
    source.attrs["units"] = {"basis": "return"}
    indexes = np.array([[0, 2], [1, 0]], dtype=np.int64)
    indexes = np.asfortranarray(indexes) if layout == "fortran" else indexes[:, ::-1]
    original = source.copy(deep=True)
    original_indexes = indexes.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = bootstrap.bootstrap_data(
            source,
            bootstrap_output=bootstrap.BootstrapOutput.SERIES_TO_DF,
            bootstrapped_indices=indexes,
        )
    expected = pd.DataFrame(source.to_numpy()[indexes], columns=["path_1", "path_2"])
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.attrs == {}
    actual.iloc[0, 0] = 100.0
    actual.index = ["first", "second"]
    actual.columns = ["left", "right"]
    actual.attrs["units"] = {"basis": "changed"}
    pd.testing.assert_series_equal(source, original)
    assert source.attrs == {"units": {"basis": "return"}}
    np.testing.assert_array_equal(indexes, original_indexes)


@pytest.mark.parametrize("bootstrap_type", list(bootstrap.BootstrapType))
def test_bootstrap_data_series_frames_keep_seeded_paired_draws(bootstrap_type):
    """Both Series use the same sampled rows, without consuming NumPy's global random stream."""
    first = pd.Series([0.1, -0.2, 0.3, -0.4])
    second = 2.0 * first + 1.0
    kwargs = dict(
        bootstrap_type=bootstrap_type, num_samples=3, index_length=7, block_size=2, seed=19
    )
    indexes = bootstrap.generate_bootstrapped_indices(num_data_index=len(first), **kwargs)
    np.random.seed(31)
    state = np.random.get_state()
    actual = bootstrap.bootstrap_data(
        first, bootstrap_output=bootstrap.BootstrapOutput.SERIES_TO_DF, **kwargs
    )
    paired = bootstrap.bootstrap_data(
        second,
        bootstrap_output=bootstrap.BootstrapOutput.SERIES_TO_DF,
        bootstrapped_indices=indexes,
    )
    columns = ["path_1", "path_2", "path_3"]
    pd.testing.assert_frame_equal(actual, pd.DataFrame(first.to_numpy()[indexes], columns=columns))
    pd.testing.assert_frame_equal(paired, pd.DataFrame(second.to_numpy()[indexes], columns=columns))
    # The relation is independently specified by source rows, not by either bootstrap result.
    np.testing.assert_allclose(paired.to_numpy(), 2.0 * actual.to_numpy() + 1.0, rtol=0, atol=0)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[0] == after[0] and state[2:] == after[2:]


@pytest.mark.parametrize(
    "case",
    [
        "object",
        "float16",
        "complex",
        "nullable",
        "foreign-endian",
        "float-index",
        "bool-index",
        "uint-index",
        "narrow-index",
        "one-dimensional",
        "three-dimensional",
    ],
)
def test_bootstrap_data_series_frames_retain_native_fallback(monkeypatch, case):
    """Inputs outside the batching domain retain the native gather and its outcome."""
    source = pd.Series([0.0, 1.0, 2.0, 3.0])
    indexes = INDEXES.copy()
    if case in {"object", "float16", "complex", "foreign-endian"}:
        dtype = {
            "object": object,
            "float16": "float16",
            "complex": "complex128",
            "foreign-endian": ">f8",
        }[case]
        source = source.astype(dtype)
    elif case == "nullable":
        # Explicit object storage carries pd.NA on both older and newer pandas versions.
        source = pd.Series([0.0, pd.NA, 2.0, 3.0], dtype=object)
    elif case in {"float-index", "bool-index", "uint-index", "narrow-index"}:
        dtype = {
            "float-index": "float64",
            "bool-index": "bool",
            "uint-index": "uint64",
            "narrow-index": "int16",
        }[case]
        indexes = indexes.astype(dtype)
    elif case == "one-dimensional":
        indexes = indexes[0]
    else:
        indexes = indexes[None]

    native_gather = bootstrap.get_bootstrap_data_list
    matrix = np.broadcast_to(source.to_numpy(), (1, len(source))).T
    try:
        draws = native_gather(data_np=matrix, bootstrapped_indices=indexes)
        expected = pd.concat(
            [pd.DataFrame(draw, columns=[f"path_{i + 1}"]) for i, draw in enumerate(draws)],
            axis=1,
            sort=False,
        )
        error = None
    except Exception as exc:
        error = type(exc)

    calls = []

    def observed_native_gather(**kwargs):
        calls.append(True)
        return native_gather(**kwargs)

    monkeypatch.setattr(bootstrap, "get_bootstrap_data_list", observed_native_gather)
    kwargs = dict(
        bootstrap_output=bootstrap.BootstrapOutput.SERIES_TO_DF, bootstrapped_indices=indexes
    )
    if error is not None:
        with pytest.raises(error):
            bootstrap.bootstrap_data(source, **kwargs)
    else:
        actual = bootstrap.bootstrap_data(source, **kwargs)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert calls == [True]


@pytest.mark.parametrize("generated", [False, True])
def test_bootstrap_data_series_frames_preserve_empty_draw_rejection(generated):
    """A zero-path request must not become a successful empty DataFrame."""
    kwargs = {"num_samples": 0} if generated else {"bootstrapped_indices": INDEXES[:, :0]}
    # Supplied emptiness is rejected by bounds reduction; generated emptiness reaches the
    # existing empty-concatenation rejection. Batching must preserve both ValueError paths.
    with pytest.raises(ValueError):
        bootstrap.bootstrap_data(
            pd.Series([0.0, 1.0, 2.0, 3.0]),
            bootstrap_output=bootstrap.BootstrapOutput.SERIES_TO_DF,
            **kwargs,
        )
