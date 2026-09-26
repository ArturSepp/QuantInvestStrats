"""First-observation, seed and missing-observation contracts of the EWM recursion.

The seed is the state before a column's first finite observation, and every finite observation,
the first included, updates it. ``InitType.X0`` seeds with that first observation, so a column that
starts late is treated exactly like one that starts on row 0. ``NanBackfill`` fixes what the state
does at a later gap; before the first observation the output is missing under every policy.
"""
# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.models.linear.ewm import (InitType, MeanAdjType, NanBackfill, compute_ewm,
                                   compute_ewm_covar, compute_ewm_covar_tensor,
                                   compute_ewm_long_short_filter, compute_ewm_vol, ewm_recursion)


def test_explicit_seed_is_a_prior_and_row_zero_enters() -> None:
    """An explicit seed is the state before row 0; the first observation still updates it."""
    actual = compute_ewm(pd.Series([2.0, 4.0]), ewm_lambda=0.5, init_value=0.0)
    np.testing.assert_allclose(actual, [1.0, 2.5])


def test_zero_seed_keeps_the_first_observation() -> None:
    """ZERO seeds the prior at zero: the first row gets weight 1 - lambda, not zero."""
    actual = compute_ewm(pd.Series([2.0, 4.0, 0.0]), ewm_lambda=0.5, init_type=InitType.ZERO)
    np.testing.assert_allclose(actual, [1.0, 2.5, 1.25])


def test_explicit_vol_seed_keeps_the_first_return() -> None:
    """compute_ewm_vol with init_value no longer drops r_0."""
    variance = compute_ewm_vol(pd.Series([0.1, 0.0]), ewm_lambda=0.5, init_value=0.0,
                               apply_sqrt=False)
    np.testing.assert_allclose(variance, [0.005, 0.0025])


@pytest.mark.parametrize('init_type', [InitType.X0, InitType.ZERO])
def test_leading_missing_rows_do_not_change_the_estimate(init_type: InitType) -> None:
    """A column that starts after missing rows equals the same column starting on row 0."""
    values = np.array([0.01, -0.02, 0.015, 0.0, 0.03, -0.01])
    padded = np.r_[np.nan, np.nan, values]
    for fn in (compute_ewm, compute_ewm_vol):
        plain = fn(pd.Series(values), ewm_lambda=0.8, init_type=init_type)
        shifted = fn(pd.Series(padded), ewm_lambda=0.8, init_type=init_type)
        assert shifted.iloc[:2].isna().all()
        np.testing.assert_allclose(shifted.iloc[2:].to_numpy(), plain.to_numpy(), rtol=1e-14)


def test_first_variance_after_a_leading_missing_row_is_the_first_squared_return() -> None:
    """With the qis.to_returns leading NaN row, v_1 = r_1^2, not (1 - lambda) r_1^2."""
    variance = compute_ewm_vol(pd.Series([np.nan, 0.01, 0.02]), ewm_lambda=0.94,
                               apply_sqrt=False)
    np.testing.assert_allclose(variance.iloc[1:], [1e-4, 0.94e-4 + 0.06 * 4e-4], rtol=1e-12)


def test_x0_seed_matches_pandas_adjust_false_on_ragged_panel() -> None:
    """X0 is pandas adjust=False per column, whether the column starts on row 0 or later."""
    rng = np.random.default_rng(3)
    values = rng.standard_normal((40, 3))
    values[:5, 1] = np.nan
    values[:17, 2] = np.nan
    frame = pd.DataFrame(values, columns=['a', 'b', 'c'])
    np.testing.assert_allclose(compute_ewm(frame, span=9),
                               frame.ewm(span=9, adjust=False).mean(), rtol=1e-12)


def test_long_short_filter_uses_a_finite_first_row() -> None:
    """The two-span filter no longer drops row 0 when it is finite."""
    impulse = pd.Series([1.0, 0.0, 0.0, 0.0])
    filtered = compute_ewm_long_short_filter(impulse, long_span=63, short_span=5,
                                             warmup_period=None)
    lam_l, lam_s = 1.0 - 2.0 / 64.0, 1.0 - 2.0 / 6.0
    kappa = np.sqrt(1 / (1 - lam_l ** 2) + 1 / (1 - lam_s ** 2) - 2 / (1 - lam_l * lam_s))
    lags = np.arange(4)
    np.testing.assert_allclose(filtered, (lam_l ** lags - lam_s ** lags) / kappa, atol=1e-15)


GAPPY = np.array([1.0, 2.0, np.nan, np.nan, 4.0])


@pytest.mark.parametrize(('nan_backfill', 'expected'), [
    (NanBackfill.FFILL, [1.0, 1.5, 1.5, 1.5, 2.75]),
    (NanBackfill.DEFLATED_FFILL, [1.0, 1.5, 0.75, 0.375, 2.1875]),
    (NanBackfill.ZERO_FILL, [1.0, 1.5, 0.0, 0.0, 2.0]),
    (NanBackfill.NAN_FILL, [1.0, 1.5, np.nan, np.nan, 2.0]),
])
def test_nan_backfill_policies_at_interior_gaps(nan_backfill: NanBackfill,
                                                expected: list) -> None:
    """FFILL holds, DEFLATED_FFILL decays, ZERO_FILL resets, NAN_FILL resets and reports NaN."""
    actual = compute_ewm(GAPPY, ewm_lambda=0.5, nan_backfill=nan_backfill)
    np.testing.assert_allclose(actual, expected)
    frame = compute_ewm(pd.DataFrame({'a': GAPPY, 'b': GAPPY}), ewm_lambda=0.5,
                        nan_backfill=nan_backfill)
    np.testing.assert_allclose(frame['b'], expected)


def test_deflated_ffill_is_a_zero_observation() -> None:
    """DEFLATED_FFILL equals the recursion run on the series with the gaps set to zero."""
    np.testing.assert_allclose(
        compute_ewm(GAPPY, ewm_lambda=0.5, nan_backfill=NanBackfill.DEFLATED_FFILL),
        compute_ewm(np.nan_to_num(GAPPY), ewm_lambda=0.5))


@pytest.mark.parametrize('nan_backfill', list(NanBackfill))
def test_leading_missing_rows_stay_missing_under_every_policy(nan_backfill: NanBackfill) -> None:
    """Before the first finite observation no policy produces a number."""
    lead = np.array([np.nan, np.nan, 1.0, 2.0])
    np.testing.assert_allclose(compute_ewm(lead, ewm_lambda=0.5, nan_backfill=nan_backfill),
                               [np.nan, np.nan, 1.0, 1.5])
    both = np.column_stack([lead, lead])
    np.testing.assert_allclose(
        ewm_recursion(both, init_value=np.zeros(2), ewm_lambda=0.5, nan_backfill=nan_backfill),
        np.column_stack([[np.nan, np.nan, 0.5, 1.25]] * 2))


def test_nan_fill_covariance_reports_gaps_and_keeps_genuine_zeros() -> None:
    """NAN_FILL flags the entries of a missing asset and leaves an exact zero covariance alone."""
    a = np.array([[1.0, 0.0], [np.nan, 1.0], [1.0, 1.0]])
    tensor = compute_ewm_covar_tensor(a, ewm_lambda=0.5, nan_backfill=NanBackfill.NAN_FILL)
    np.testing.assert_allclose(tensor[0], [[0.5, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(tensor[1], [[np.nan, np.nan], [np.nan, 0.5]])
    np.testing.assert_allclose(tensor[2], [[0.5, 0.5], [0.5, 0.75]])
    last = compute_ewm_covar(a[:2], ewm_lambda=0.5, nan_backfill=NanBackfill.NAN_FILL)
    np.testing.assert_allclose(last, tensor[1])


def test_var_seed_is_the_variance_of_the_observations() -> None:
    """InitType.VAR seeds a variance recursion with Var(x), not with Var(x^2)."""
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(500) * 0.0095
    variance = compute_ewm_vol(returns, ewm_lambda=0.94, init_type=InitType.VAR,
                               apply_sqrt=False)
    expected = 0.94 * np.var(returns) + 0.06 * returns[0] ** 2
    np.testing.assert_allclose(variance[0], expected, rtol=1e-12)
    assert variance[0] > 1e-5


def test_var_seed_is_rejected_for_a_mean() -> None:
    """A mean recursion cannot be seeded with a variance."""
    with pytest.raises(ValueError, match='InitType.VAR'):
        compute_ewm(np.ones(5), init_type=InitType.VAR)


def test_var_seed_with_mean_adjustment_seeds_the_mean_with_the_sample_mean() -> None:
    """Under VAR the EWMA mean adjustment of compute_ewm_vol takes the full-sample mean seed."""
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.standard_normal(200) * 0.01 + 0.002)
    actual = compute_ewm_vol(returns, ewm_lambda=0.9, mean_adj_type=MeanAdjType.EWMA,
                             init_type=InitType.VAR, apply_sqrt=False)
    centred = returns - compute_ewm(returns, ewm_lambda=0.9, init_type=InitType.MEAN)
    expected = compute_ewm_vol(centred, ewm_lambda=0.9, init_value=float(np.var(centred)),
                               apply_sqrt=False)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
