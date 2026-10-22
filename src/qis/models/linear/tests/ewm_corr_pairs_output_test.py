"""compute_ewm_corr_df pair selection and compute_ewm_corr_single, as documented."""

# packages
import math
import warnings
import numpy as np
import pandas as pd
import pytest
from numba.core.errors import TypingError

# qis
from qis.models.linear import ewm
from qis.models.linear.corr_cov_matrix import (
    CorrMatrixOutput,
    compute_ewm_corr_df,
    compute_ewm_corr_single,
)


def _returns(num_columns: int) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(rng.standard_normal((60, num_columns)),
                        columns=[f"x{i}" for i in range(num_columns)])


def test_full_returns_pairs_below_the_diagonal() -> None:
    """FULL returns (i, j) with j < i, named '<column i> - <column j>'."""
    corr = compute_ewm_corr_df(df=_returns(3), corr_matrix_output=CorrMatrixOutput.FULL)
    assert corr.columns.tolist() == ['x1 - x0', 'x2 - x0', 'x2 - x1']


def test_sub_top_coincides_with_full() -> None:
    """SUB_TOP is documented as returning the same pairs as FULL."""
    df = _returns(4)
    full = compute_ewm_corr_df(df=df, corr_matrix_output=CorrMatrixOutput.FULL)
    sub_top = compute_ewm_corr_df(df=df, corr_matrix_output=CorrMatrixOutput.SUB_TOP)
    pd.testing.assert_frame_equal(full, sub_top)


def test_top_row_returns_pairs_of_the_first_column() -> None:
    """TOP_ROW returns the first column against every later one."""
    corr = compute_ewm_corr_df(df=_returns(3), corr_matrix_output=CorrMatrixOutput.TOP_ROW)
    assert corr.columns.tolist() == ['x0 - x1', 'x0 - x2']


def test_single_returns_the_one_pair() -> None:
    """The two-column case returns one Series equal to the FULL column."""
    df = _returns(2)
    single = compute_ewm_corr_single(returns=df, span=20)
    full = compute_ewm_corr_df(df=df, ewm_lambda=1.0 - 2.0 / 21.0)
    pd.testing.assert_series_equal(single, full.iloc[:, 0])
    assert single.name == 'x1 - x0'


def test_single_rejects_other_widths_with_column_names() -> None:
    """The error message names the columns it received."""
    with pytest.raises(ValueError, match='x2'):
        compute_ewm_corr_single(returns=_returns(3))


def _tensor_top_row(df: pd.DataFrame, **parameters) -> pd.DataFrame:
    """Complete accepted output, independently of the wrapper's selected storage path."""
    seed = parameters.get('init_value')
    if seed is None:
        seed = ewm.set_init_dim2(df.to_numpy(), parameters.get('init_type', ewm.InitType.ZERO))
    tensor = ewm.compute_ewm_covar_tensor(
        df.to_numpy(), span=parameters.get('span'), ewm_lambda=parameters.get('ewm_lambda', 0.94),
        covar0=seed, is_corr=True)
    return pd.concat([
        pd.Series(tensor[:, 0, j], name=f'{df.columns[0]} - {df.columns[j]}')
        for j in range(1, df.shape[1])
    ], axis=1, sort=False).set_index(df.index)


def _weighted_top_row(df: pd.DataFrame, decay: float) -> pd.DataFrame:
    """Closed-form weighted sums, not the production recursion, for modest-scale fixtures."""
    values = df.to_numpy()
    output = np.full((len(df), df.shape[1] - 1), np.nan)
    for t in range(len(df)):
        weights = [(1.0 - decay) * decay ** (t - s) for s in range(t + 1)]
        for j in range(1, df.shape[1]):
            x = [float(value) if np.isfinite(value) else 0.0 for value in values[:t + 1, 0]]
            y = [float(value) if np.isfinite(value) else 0.0 for value in values[:t + 1, j]]
            vx = math.fsum(w * a * a for w, a in zip(weights, x))
            vy = math.fsum(w * b * b for w, b in zip(weights, y))
            if vx > 0.0 and vy > 0.0:
                cross = math.fsum(w * a * b for w, a, b in zip(weights, x, y))
                output[t, j - 1] = cross / (math.sqrt(vx) * math.sqrt(vy))
    return pd.DataFrame(output, index=df.index,
                        columns=[f'{df.columns[0]} - {column}' for column in df.columns[1:]])


@pytest.mark.parametrize('width', [2, 3])
def test_compute_ewm_corr_df_top_row_avoids_full_tensor(monkeypatch, width) -> None:
    """The two-column threshold and a ragged third asset avoid unrequested history."""
    df = pd.DataFrame([[1.0, 2.0, np.nan], [2.0, -1.0, 1.0],
                       [-1.0, 3.0, -2.0], [4.0, 0.0, 2.0]],
                      index=pd.date_range('2020-01-01', periods=4, tz='UTC', name='observed'),
                      columns=['a', 'b', 'c']).iloc[:, :width]
    before = df.copy(deep=True)
    expected = _weighted_top_row(df, 0.5)

    def unexpected_tensor(*args, **kwargs):
        pytest.fail('TOP_ROW must not allocate the full correlation tensor')

    monkeypatch.setattr(ewm, 'compute_ewm_covar_tensor', unexpected_tensor)
    actual = compute_ewm_corr_df(df, CorrMatrixOutput.TOP_ROW, ewm_lambda=0.5)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-14, atol=1e-14)
    actual.iloc[-1, 0] = -99.0
    pd.testing.assert_frame_equal(df, before, check_exact=True)


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
@pytest.mark.parametrize('parameters', [
    {}, {'init_type': ewm.InitType.X0}, {'span': 7}, {'ewm_lambda': 0.0},
    {'span': 1, 'ewm_lambda': np.inf}, {'ewm_lambda': np.float32(0.5)},
])
def test_compute_ewm_corr_df_top_row_matches_tensor_on_mixed_panel(dtype, parameters) -> None:
    """Missing states and duplicate labels interact in one panel, not isolated fixtures."""
    df = _returns(8).astype(dtype)
    df.iloc[:3, 0] = np.nan
    df.iloc[:7, 1] = np.nan
    df.iloc[10:12, 2] = np.nan
    df.iloc[30:, 3] = np.nan
    df.iloc[:, 4] = np.nan
    df.iloc[:, 5] = 0.0
    df.iloc[:, 6] *= 1e-9
    df.iloc[15, 7] = np.inf
    df.columns = ['pivot', 'same', 'same', 3, np.nan, 'zero', 'tiny', 'nonfinite']
    df.index = pd.date_range('2020-01-01', periods=len(df), tz='UTC', name='observed')
    before = df.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        expected = _tensor_top_row(df, **parameters)
        actual = compute_ewm_corr_df(df, CorrMatrixOutput.TOP_ROW, **parameters)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(df, before, check_exact=True)


@pytest.mark.parametrize('parameters', [
    {}, {'init_type': ewm.InitType.X0}, {'span': 7}, {'ewm_lambda': 0.0},
    {'span': 1, 'ewm_lambda': np.inf},
])
def test_compute_ewm_corr_df_top_row_preserves_nullable_rejection(parameters) -> None:
    """Nullable pandas data reaches the accepted object-array rejection before dispatch."""
    df = _returns(3).astype('Float64')
    df.iloc[0, 1] = pd.NA
    before = df.copy(deep=True)
    for evaluate in (_tensor_top_row,
                     lambda data, **kwargs: compute_ewm_corr_df(
                         data, CorrMatrixOutput.TOP_ROW, **kwargs)):
        with pytest.raises(TypingError, match='non-precise type array\\(pyobject'):
            evaluate(df, **parameters)
        pd.testing.assert_frame_equal(df, before, check_exact=True)


def test_compute_ewm_corr_df_top_row_retains_integer_fallback(monkeypatch) -> None:
    """Non-floating arrays keep native tensor ownership even when their values are valid."""
    df = pd.DataFrame([[1, 2, 0], [2, -1, 1], [-1, 3, -2]], columns=['a', 'b', 'c'])
    expected = _tensor_top_row(df)

    def unexpected_specialization(*args, **kwargs):
        pytest.fail('integer data must retain native tensor dispatch')

    monkeypatch.setattr(ewm, '_compute_ewm_corr_top_row', unexpected_specialization)
    pd.testing.assert_frame_equal(compute_ewm_corr_df(df, CorrMatrixOutput.TOP_ROW),
                                  expected, check_exact=True)


@pytest.mark.parametrize('mode', [CorrMatrixOutput.TOP_ROW, CorrMatrixOutput.FULL])
def test_compute_ewm_corr_df_preserves_global_scale_after_overflow(mode) -> None:
    """An unrequested covariance can mask a finite pair after a diagonal update overflows."""
    df = pd.DataFrame([[1e135, 1e135, 0.0, 0.0], [0.0, 0.0, 1e160, 1e140]],
                      columns=['a', 'b', 'c', 'd'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        actual = compute_ewm_corr_df(df, mode, ewm_lambda=0.5)
        reference = _tensor_top_row if mode == CorrMatrixOutput.TOP_ROW else _tensor_pairs
        expected = reference(df, ewm_lambda=0.5)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.iloc[0, 0] == pytest.approx(1.0)
    assert np.isnan(actual.iloc[1, 0])


@pytest.mark.parametrize('seed', [np.eye(3), np.array([[1.0, 0.2, 0.0],
                                                     [0.1, 1.0, 1e20], [0.0, 1e20, 1.0]])])
def test_compute_ewm_corr_df_top_row_preserves_explicit_seed_and_ownership(seed) -> None:
    """Explicit seeds keep their full-matrix policy, and override unsupported init types."""
    df = _returns(3)
    before = seed.copy()
    parameters = dict(init_value=seed, init_type=ewm.InitType.VAR)
    expected = _tensor_top_row(df, **parameters)
    actual = compute_ewm_corr_df(df, CorrMatrixOutput.TOP_ROW, **parameters)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    np.testing.assert_array_equal(seed, before)


@pytest.mark.parametrize('parameters', [
    {'span': 0}, {'span': np.nan}, {'span': True},
    {'ewm_lambda': -0.1}, {'ewm_lambda': 1.0}, {'ewm_lambda': np.inf},
])
def test_compute_ewm_corr_df_top_row_preserves_parameter_errors(parameters) -> None:
    """Selected smoothing is rejected before recursion, including the Boolean boundary."""
    df = _returns(3)
    with pytest.raises(ValueError) as accepted:
        _tensor_top_row(df, **parameters)
    with pytest.raises(ValueError) as actual:
        compute_ewm_corr_df(df, CorrMatrixOutput.TOP_ROW, **parameters)
    assert str(actual.value) == str(accepted.value)


@pytest.mark.parametrize('init_type', [ewm.InitType.MEAN, ewm.InitType.VAR])
def test_compute_ewm_corr_df_top_row_preserves_unsupported_seed_error(init_type) -> None:
    """Storage specialization must not bypass the accepted initialization boundary."""
    with pytest.raises(TypeError, match='unsupported init_type'):
        compute_ewm_corr_df(_returns(3), CorrMatrixOutput.TOP_ROW, init_type=init_type)


@pytest.mark.parametrize('parameters', [
    {'init_value': np.eye(3)}, {'span': np.array([3.0, 4.0, 5.0])},
    {'ewm_lambda': np.array([0.5, 0.6, 0.7])},
])
def test_compute_ewm_corr_df_top_row_retains_native_fallback(monkeypatch, parameters) -> None:
    """Native tensor dispatch retains ownership of non-specialized initialization/smoothing."""
    df = _returns(3)
    calls = []

    def native_tensor(a, **kwargs):
        calls.append(kwargs)
        return np.tile(np.eye(a.shape[1]), (a.shape[0], 1, 1))

    monkeypatch.setattr(ewm, 'compute_ewm_covar_tensor', native_tensor)
    actual = compute_ewm_corr_df(df, CorrMatrixOutput.TOP_ROW, **parameters)
    assert len(calls) == 1
    assert actual.shape == (len(df), 2)
    assert (actual == 0.0).all().all()


@pytest.mark.parametrize('rows', [0, 1])
@pytest.mark.parametrize('mode', [CorrMatrixOutput.TOP_ROW, CorrMatrixOutput.FULL])
def test_compute_ewm_corr_df_preserves_short_sample(rows, mode) -> None:
    """Empty history and first-observation normalization retain exact frames."""
    df = _returns(3).iloc[:rows]
    reference = _tensor_top_row if mode == CorrMatrixOutput.TOP_ROW else _tensor_pairs
    pd.testing.assert_frame_equal(compute_ewm_corr_df(df, mode),
                                  reference(df), check_exact=True)


@pytest.mark.parametrize('width', [0, 1])
@pytest.mark.parametrize('mode', list(CorrMatrixOutput))
def test_compute_ewm_corr_df_preserves_no_pair_error(width, mode) -> None:
    """A panel without any pair keeps the accepted concat error, not an empty result."""
    with pytest.raises(ValueError, match='No objects to concatenate'):
        compute_ewm_corr_df(_returns(width), mode)


def _tensor_pairs(df: pd.DataFrame, **parameters) -> pd.DataFrame:
    """Native full-history construction remains the compatibility reference, not the oracle."""
    seed = parameters.get('init_value')
    if seed is None:
        seed = ewm.set_init_dim2(df.to_numpy(), parameters.get('init_type', ewm.InitType.ZERO))
    tensor = ewm.compute_ewm_covar_tensor(
        df.to_numpy(), span=parameters.get('span'), ewm_lambda=parameters.get('ewm_lambda', 0.94),
        covar0=seed, is_corr=True)
    return pd.concat([
        pd.Series(tensor[:, i, j], name=f'{column_i} - {column_j}')
        for i, column_i in enumerate(df.columns) for j, column_j in enumerate(df.columns) if j < i
    ], axis=1, sort=False).set_index(df.index)


@pytest.mark.parametrize('mode', [CorrMatrixOutput.FULL, CorrMatrixOutput.SUB_TOP])
@pytest.mark.parametrize('width', [2, 3])
def test_compute_ewm_corr_df_pairs_avoids_full_tensor(monkeypatch, mode, width) -> None:
    """Retain only returned history, without replacing the independently weighted result."""
    df = pd.DataFrame([[1., 2., np.nan], [2., -1., 1.], [-1., 3., -2.], [4., 0., 2.]],
                      columns=['a', 'b', 'c']).iloc[:, :width]
    before = df.copy(deep=True)
    expected = pd.concat([
        _weighted_top_row(df.iloc[:, [i, j]], 0.5)
        for i in range(width) for j in range(i)
    ], axis=1)

    def unexpected_tensor(*args, **kwargs):
        pytest.fail('all-pairs output must not allocate unrequested full tensor history')

    monkeypatch.setattr(ewm, 'compute_ewm_covar_tensor', unexpected_tensor)
    actual = compute_ewm_corr_df(df, mode, ewm_lambda=0.5)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-14, atol=1e-14)
    actual.iloc[-1, 0] = -99.
    pd.testing.assert_frame_equal(actual.iloc[:, 1:], expected.iloc[:, 1:], check_exact=True)
    pd.testing.assert_frame_equal(df, before, check_exact=True)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('index', [
    pd.RangeIndex(60, name='rows'),
    pd.date_range('2020-01-01', periods=60, freq='B', tz='Europe/London', name='dates'),
    pd.period_range('2020-01-01', periods=60, freq='D', name='period'),
    pd.MultiIndex.from_arrays([np.arange(60) // 2, np.arange(60) % 2], names=['day', 'part']),
])
@pytest.mark.parametrize('parameters', [
    {}, {'init_type': ewm.InitType.X0}, {'span': 7, 'ewm_lambda': np.inf},
    {'ewm_lambda': np.float32(0.5)},
])
def test_compute_ewm_corr_df_pairs_preserves_mixed_panel_and_axes(dtype, index, parameters) -> None:
    """One ragged panel crosses the new storage and pandas-construction boundaries."""
    df = _returns(8).astype(dtype).set_axis(index)
    df.iloc[:3, 0] = np.nan
    df.iloc[:7, 1] = np.nan
    df.iloc[10:12, 2] = np.nan
    df.iloc[30:, 3] = np.nan
    df.iloc[:, 4] = np.nan
    df.iloc[:, 5] = 0.
    df.iloc[:, 6] *= 1e-9
    df.iloc[15, 7] = np.inf
    df.columns = pd.Index(['pivot', 'same', 'same', 3, np.nan, ('a', 'b'), 'tiny', 'inf'],
                          name='assets')
    before = df.copy(deep=True)
    # Separate axis snapshots keep shared pandas metadata from masking a caller mutation.
    before.index = before.index.copy(deep=True)
    before.columns = before.columns.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        expected = _tensor_pairs(df, **parameters)
        actual = compute_ewm_corr_df(df, **parameters)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(df, before, check_exact=True)


@pytest.mark.parametrize('dtype, parameters', [
    ('int64', {}), ('Float64', {}), ('float64', {'init_value': np.eye(3),
                                              'init_type': ewm.InitType.VAR}),
    ('float64', {'span': np.array([3., 4., 5.])}),
    ('float64', {'ewm_lambda': np.array([.5, .6, .7])}),
    ('float64', {'ewm_lambda': np.float16(.5)}),
    ('float64', {'ewm_lambda': np.longdouble(.5)}),
    ('float64', {'init_type': ewm.InitType.MEAN}),
])
def test_compute_ewm_corr_df_pairs_retains_native_fallback(monkeypatch, dtype, parameters) -> None:
    """Do not broaden Numba admission or bypass explicit-seed and initialization policy."""
    df = _returns(3).astype(dtype)
    expected_error = None
    try:
        expected = _tensor_pairs(df, **parameters)
    except Exception as error:
        expected_error = error

    def unexpected_specialization(*args, **kwargs):
        pytest.fail('exceptional inputs must retain native tensor dispatch')

    monkeypatch.setattr(ewm, '_compute_ewm_corr_pairs', unexpected_specialization)
    if expected_error is not None:
        with pytest.raises(type(expected_error)) as actual:
            compute_ewm_corr_df(df, **parameters)
        assert str(actual.value) == str(expected_error)
    else:
        pd.testing.assert_frame_equal(compute_ewm_corr_df(df, **parameters), expected,
                                      check_exact=True)


@pytest.mark.parametrize('parameters', [
    {'span': 0}, {'span': np.nan}, {'span': True},
    {'ewm_lambda': -.1}, {'ewm_lambda': 1.}, {'ewm_lambda': np.inf},
])
def test_compute_ewm_corr_df_pairs_preserves_parameter_errors(parameters) -> None:
    """The selected smoothing parameter keeps native validation and error wording."""
    with pytest.raises(ValueError) as expected:
        _tensor_pairs(_returns(3), **parameters)
    with pytest.raises(ValueError) as actual:
        compute_ewm_corr_df(_returns(3), **parameters)
    assert str(actual.value) == str(expected.value)


def test_compute_ewm_corr_df_pairs_preserves_series_rejection() -> None:
    """A malformed container still reaches the native dimensionality diagnostic."""
    data = _returns(1).iloc[:, 0]
    with pytest.raises(ValueError, match='only 2-d arrays are supported'):
        compute_ewm_corr_df(data)


@pytest.mark.parametrize('fill_value', [0.0, np.nan])
def test_compute_ewm_corr_df_top_row_preserves_undefined_panel(fill_value) -> None:
    """A zero-scale normalizer keeps the whole output undefined without warnings."""
    df = pd.DataFrame(fill_value, index=range(3), columns=['a', 'b', 'c'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        actual = compute_ewm_corr_df(df, CorrMatrixOutput.TOP_ROW)
    pd.testing.assert_frame_equal(actual, _tensor_top_row(df), check_exact=True)
    assert actual.isna().all().all()


def test_compute_ewm_corr_df_top_row_preserves_negative_seed_error() -> None:
    """The full-matrix fallback must reject a negative variance, including unrequested assets."""
    seed = np.diag([1.0, 1.0, -1000.0])
    before = seed.copy()
    with pytest.raises(ValueError, match='materially negative values'):
        compute_ewm_corr_df(_returns(3), CorrMatrixOutput.TOP_ROW, init_value=seed)
    np.testing.assert_array_equal(seed, before)
