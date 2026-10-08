"""Tests for canonical ex-post tracking-error estimators."""
from fractions import Fraction
import warnings

import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets.synthetic import generate_synthetic_universe


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


def _constant_magnitude_navs(index, is_log_returns=False, dtype='float64'):
    """Build known active returns without using the production return converter."""
    differences = 0.01 * np.resize([1.0, -1.0], len(index) - 1)
    levels = (np.exp(np.cumsum(differences)) if is_log_returns
              else np.cumprod(1.0 + differences))
    portfolio = pd.Series(np.concatenate(([100.0], 100.0 * levels)), index=index,
                          name='Portfolio', dtype=dtype)
    benchmark = pd.Series(100.0, index=index, name='Benchmark', dtype=dtype)
    return portfolio, benchmark


@pytest.mark.parametrize('freq,periods,span,factor', [
    ('B', 4, 3, 252), ('ME', 2, 1, 12), ('QE', 2, 1, 4), ('D', 4, 3, 365),
    ('2ME', 2, 1, 6), ('BQE', 2, 1, 4), ('3QE', 2, 1, 4 / 3),
    ('WOM-2WED', 2, 1, 12), ('SME-15', 2, 1, 24),
])
@pytest.mark.parametrize('is_log_returns', [False, True])
@pytest.mark.parametrize('dtype', ['float64', 'Float64'])
def test_compute_ewma_realised_tracking_error_honors_explicit_frequency(
        freq, periods, span, factor, is_log_returns, dtype):
    """Short samples cannot overrule the caller's grid or its periods-per-year units."""
    index = pd.date_range('2026-06-01', periods=periods + 1, freq=freq, tz='UTC')
    portfolio, benchmark = _constant_magnitude_navs(index, is_log_returns, dtype)
    before_portfolio, before_benchmark = portfolio.copy(), benchmark.copy()

    result = qis.compute_ewma_realised_tracking_error(
        portfolio, benchmark, ewma_span=span, freq=freq, is_log_returns=is_log_returns)

    # Each squared active return is .0001. X0 seeds that same value, so every EWMA update
    # leaves it unchanged. Factors are independently counted, not taken from the helper.
    values = np.full(periods, 0.01 * np.sqrt(factor))
    values[:span] = np.nan
    expected = pd.Series(values, index=index[1:], name='Tracking error')
    pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=0.0)
    pd.testing.assert_series_equal(portfolio, before_portfolio)
    pd.testing.assert_series_equal(benchmark, before_benchmark)


@pytest.mark.parametrize('is_log_returns', [False, True])
def test_compute_ewma_realised_tracking_error_preserves_business_day_prefix(is_log_returns):
    """Monday's arrival must not rescale Friday's already-observable risk estimate."""
    index = pd.date_range('2026-06-01', periods=6, freq='B')
    portfolio, benchmark = _constant_magnitude_navs(index, is_log_returns)
    short = qis.compute_ewma_realised_tracking_error(
        portfolio.iloc[:5], benchmark.iloc[:5], ewma_span=3, freq='B',
        is_log_returns=is_log_returns)
    extended = qis.compute_ewma_realised_tracking_error(
        portfolio, benchmark, ewma_span=3, freq='B', is_log_returns=is_log_returns)

    pd.testing.assert_series_equal(short, extended.iloc[:len(short)])
    np.testing.assert_allclose(short.iloc[-1], 0.01 * np.sqrt(252), rtol=1e-12)


@pytest.mark.parametrize('irregular', [False, True])
def test_compute_ewma_realised_tracking_error_retains_inference_without_frequency(irregular):
    """An omitted grid keeps the existing inference and irregular-index warning/fallback."""
    index = (pd.DatetimeIndex(['2026-06-01', '2026-06-02', '2026-06-04',
                               '2026-06-05', '2026-06-08']) if irregular
             else pd.date_range('2026-06-01', periods=5, freq='B'))
    portfolio, benchmark = _constant_magnitude_navs(index)
    # Tuesday through Friday infer as D even though the original NAV index carries B.
    # This compatibility path deliberately differs from explicitly requesting freq='B'.
    if irregular:
        with pytest.warns(UserWarning, match='cannot infer None - using 252'):
            result = qis.compute_ewma_realised_tracking_error(
                portfolio, benchmark, ewma_span=3, freq=None)
    else:
        result = qis.compute_ewma_realised_tracking_error(
            portfolio, benchmark, ewma_span=3, freq=None)
    factor = 252 if irregular else 365
    expected = pd.Series([np.nan, np.nan, np.nan, 0.01 * np.sqrt(factor)],
                         index=index[1:], name='Tracking error')
    pd.testing.assert_series_equal(result, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize('freq', ['D_8H', 'B_8H', 'M-FRI', 'Q-FRI', 'Q-3FRI', 'SE'])
@pytest.mark.parametrize('is_log_returns', [False, True])
@pytest.mark.parametrize('dtype', ['float64', 'Float64'])
def test_compute_ewma_realised_tracking_error_preserves_bespoke_grids(
        freq, is_log_returns, dtype):
    """QIS-only schedules retain the existing inferred scaling and warning contract."""
    index = pd.date_range('2020-01-01', periods=810, freq='D', tz='UTC')
    portfolio, benchmark = _constant_magnitude_navs(index, is_log_returns, dtype)
    before_portfolio, before_benchmark = portfolio.copy(), benchmark.copy()
    prices = pd.concat([portfolio.rename('p'), benchmark.rename('b')], axis=1)
    returns = qis.to_returns(prices, freq=freq, is_log_returns=is_log_returns, drop_first=True)
    differences = (returns['p'] - returns['b']).dropna()

    # Replay the established delegation, not a new annualisation convention for bespoke
    # dates. SE also protects the all-warm-up result and short-index fallback warning.
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter('always', UserWarning)
        expected = qis.compute_ewm_vol(
            differences, span=3, annualize=True, warmup_period=3).rename('Tracking error')
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter('always', UserWarning)
        result = qis.compute_ewma_realised_tracking_error(
            portfolio, benchmark, ewma_span=3, freq=freq, is_log_returns=is_log_returns)

    pd.testing.assert_series_equal(result, expected, check_exact=True)
    assert [(w.category, str(w.message)) for w in actual_warnings] == [
        (w.category, str(w.message)) for w in expected_warnings]
    if freq == 'D_8H':
        # Daily constant-magnitude active returns give an independent numerical check.
        np.testing.assert_allclose(result.dropna(), 0.01 * np.sqrt(365), rtol=1e-12)
    pd.testing.assert_series_equal(portfolio, before_portfolio)
    pd.testing.assert_series_equal(benchmark, before_benchmark)


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


def _synthetic_return_diffs() -> pd.DataFrame:
    universe = generate_synthetic_universe(
        start='2010-01-01',
        end='2015-12-31',
        apply_quirks=False,
    )
    strategy_returns = qis.to_returns(
        universe.prices[['SEQ_US', 'SBD_TSY']],
        freq='ME',
        is_log_returns=False,
        drop_first=True,
    )
    benchmark_returns = qis.to_returns(
        universe.benchmark_prices.iloc[:, 0],
        freq='ME',
        is_log_returns=False,
        drop_first=True,
    )
    return strategy_returns.sub(benchmark_returns, axis=0)


def test_in_sample_te_ir_match_pre_move_characterisation() -> None:
    return_diffs = _synthetic_return_diffs()

    te, ir = qis.compute_te_ir_errors(return_diffs)

    expected_te = pd.Series(
        {'SEQ_US': 0.07887977676419106, 'SBD_TSY': 0.11823501914914415},
        name='TE',
    )
    expected_ir = pd.Series(
        {'SEQ_US': 0.032279891161704155, 'SBD_TSY': -0.027047443953465655},
        name='IR',
    )
    pd.testing.assert_series_equal(te, expected_te, rtol=1e-12, atol=0.0)
    pd.testing.assert_series_equal(ir, expected_ir, rtol=1e-12, atol=0.0)


def test_in_sample_te_ir_scaling_and_constant_difference() -> None:
    return_diffs = _synthetic_return_diffs()
    return_diffs['constant'] = 0.0

    te, ir = qis.compute_te_ir_errors(return_diffs)
    scaled_te, scaled_ir = qis.compute_te_ir_errors(3.0 * return_diffs)

    assert te.name == 'TE'
    assert ir.name == 'IR'
    assert np.isnan(ir['constant'])
    pd.testing.assert_series_equal(scaled_te, 3.0 * te, rtol=1e-10, atol=0.0)
    pd.testing.assert_series_equal(scaled_ir, ir, rtol=1e-10, atol=0.0)


def test_compute_info_ratio_table_uses_whole_sample_estimator() -> None:
    return_diffs = _synthetic_return_diffs()
    expected_te, expected_ir = qis.compute_te_ir_errors(return_diffs)

    te_table, ir_table = qis.compute_info_ratio_table(
        {'Base': return_diffs, 'Double': 2.0 * return_diffs}
    )

    pd.testing.assert_series_equal(te_table['Base'], expected_te.rename('Base'))
    pd.testing.assert_series_equal(te_table['Double'], (2.0 * expected_te).rename('Double'))
    pd.testing.assert_series_equal(ir_table['Base'], expected_ir.rename('Base'))
    pd.testing.assert_series_equal(ir_table['Double'], expected_ir.rename('Double'))


def _pairwise_te_ir(samples, columns, annualisation=12):
    """Derive sample spread from pairwise distances, independently of NumPy reductions."""
    tracking_errors, information_ratios = [], []
    for sample in samples:
        values = [Fraction(value) for value in sample if value is not None]
        count = len(values)
        if count < 2:
            tracking_errors.append(np.nan)
            information_ratios.append(np.nan)
            continue
        variance = sum((left - right) ** 2 for i, left in enumerate(values)
                       for right in values[i + 1:]) / (count * (count - 1))
        spread = float(variance) ** 0.5
        tracking_errors.append(annualisation ** 0.5 * spread)
        information_ratios.append(
            annualisation ** 0.5 * float(sum(values) / count) / spread if spread else np.nan
        )
    return (pd.Series(tracking_errors, index=columns, name='TE'),
            pd.Series(information_ratios, index=columns, name='IR'))


@pytest.mark.filterwarnings('ignore:Mean of empty slice:RuntimeWarning')
@pytest.mark.filterwarnings('ignore:Degrees of freedom <= 0 for slice:RuntimeWarning')
@pytest.mark.parametrize('dtype', ['float64', 'Float64', 'Float32', 'Int64', 'mixed'])
@pytest.mark.parametrize('entry_point', ['direct', 'table'])
def test_compute_te_ir_errors_omits_nullable_values_per_column(dtype, entry_point):
    """Ragged and undefined samples must not contaminate healthy neighboring strategies."""
    samples = [
        [1, 2, -1, 0, 3],
        [None, 2, -1, 0, 3],
        [1, None, -1, 0, 3],
        [1, 2, -1, None, None],
        [0, 0, 0, 0, 0],
        [None, None, None, None, None],
        [None, None, 1, None, None],
    ]
    # Binary-exact fractions isolate nullable conversion from float32 rounding. Int64 uses
    # the same sample in integer units; TE changes with units, but IR does not.
    divisor = 1 if dtype == 'Int64' else 128
    scaled_samples = [[None if value is None else value / divisor for value in sample]
                      for sample in samples]
    columns = pd.Index(['complete', 7, 7, 'tail', 'constant', 'empty', 'single'], name='strategy')
    index = pd.date_range('2020-01-31', periods=5, freq='ME', name='observation', tz='UTC')
    series = [pd.Series(sample, index=index,
                        dtype=('Float64' if i % 2 else 'float64') if dtype == 'mixed' else dtype)
              for i, sample in enumerate(scaled_samples)]
    panel = pd.concat(series, axis=1)
    panel.columns = columns
    before = panel.copy(deep=True)
    expected_te, expected_ir = _pairwise_te_ir(scaled_samples, columns)

    # Undefined columns stay NaN; they are neither filled with zero nor rescued by
    # observations belonging to another column. NumPy's empty-slice warnings are incidental.
    if entry_point == 'direct':
        te, ir = qis.compute_te_ir_errors(panel)
    else:
        te_table, ir_table = qis.compute_info_ratio_table({'Base': panel, 'Copy': panel})
        pd.testing.assert_frame_equal(te_table, pd.concat(
            [expected_te.rename('Base'), expected_te.rename('Copy')], axis=1))
        pd.testing.assert_frame_equal(ir_table, pd.concat(
            [expected_ir.rename('Base'), expected_ir.rename('Copy')], axis=1))
        te, ir = te_table['Base'].rename('TE'), ir_table['Base'].rename('IR')
    pd.testing.assert_series_equal(te, expected_te, rtol=1e-14, atol=0.0)
    pd.testing.assert_series_equal(ir, expected_ir, rtol=1e-14, atol=0.0)
    pd.testing.assert_frame_equal(panel, before)


@pytest.mark.filterwarnings('ignore:invalid value encountered:RuntimeWarning')
@pytest.mark.parametrize('dtype', ['float64', 'Float64'])
def test_compute_te_ir_errors_does_not_treat_infinity_as_missing(dtype):
    """Nullable normalization must not inherit the finite-value helper's filtering policy."""
    panel = pd.DataFrame({'infinite': [1., np.inf, 3.], 'healthy': [1., 2., 3.]},
                         index=pd.date_range('2020-01-31', periods=3, freq='ME'), dtype=dtype)
    before = panel.copy(deep=True)
    te, ir = qis.compute_te_ir_errors(panel)
    expected_te, expected_ir = _pairwise_te_ir([[1, 2, 3]], pd.Index(['healthy']))
    assert np.isnan(te['infinite']) and np.isnan(ir['infinite'])
    pd.testing.assert_series_equal(te.iloc[1:], expected_te)
    pd.testing.assert_series_equal(ir.iloc[1:], expected_ir)
    pd.testing.assert_frame_equal(panel, before)


def test_compute_te_ir_errors_handles_ordinary_float32():
    """Ordinary float32 panels give the exact sample TE and IR to float32 precision."""
    samples = [[1, 3, -2, 6, 4]]
    panel = pd.DataFrame({'strategy': samples[0]}, dtype='float32',
                         index=pd.date_range('2020-01-31', periods=5, freq='ME'))
    expected_te, expected_ir = _pairwise_te_ir(samples, pd.Index(['strategy']))
    te, ir = qis.compute_te_ir_errors(panel)
    # The tolerance admits float32 or float64 accumulation, so the reduction dtype stays free.
    pd.testing.assert_series_equal(te, expected_te, check_dtype=False, rtol=1e-6)
    pd.testing.assert_series_equal(ir, expected_ir, check_dtype=False, rtol=1e-6)


def test_compute_te_ir_errors_does_not_coerce_numeric_strings():
    """The nullable fix is not permission to parse a nonnumeric panel as financial returns."""
    panel = pd.DataFrame({'strings': ['1', '2', '3']},
                         index=pd.date_range('2020-01-31', periods=3, freq='ME'))
    before = panel.copy(deep=True)
    with pytest.raises(TypeError):
        qis.compute_te_ir_errors(panel)
    pd.testing.assert_frame_equal(panel, before)
