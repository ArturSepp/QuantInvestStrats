"""Outlier filtering: local numpy error state, NaN-aware quantiles, per-column clipping, the
non-anticipating 2-d recursion and presets whose EWM cut can act."""
# packages
import numpy as np
import pytest

# qis
from qis.models.linear.ewm_winsor_outliers import (OutlierPolicy, OutlierPolicyTypes,
                                                   ReplacementType, compute_ewm_score,
                                                   ewm_insample_winsorising,
                                                   ewm_winsdor_markovian_score, filter_outliers,
                                                   score_of_move)


def test_filter_outliers_leaves_numpy_error_state_unchanged() -> None:
    """The invalid-value suppression is local to the call."""
    before = np.geterr()
    filter_outliers(np.array([1.0, np.nan, 2.0]), OutlierPolicy(std_abs_ceil=3.0))
    assert np.geterr() == before


def test_insample_winsorising_handles_a_column_with_missing_values() -> None:
    """A single NaN no longer disables winsorising of its column."""
    rng = np.random.default_rng(41)
    data = rng.standard_normal((300, 2))
    data[5, 0] = np.nan
    data[100, :] = 50.0
    cleaned = ewm_insample_winsorising(data, nan_replacement_type=ReplacementType.NAN)
    assert np.isnan(cleaned[100, 0]) and np.isnan(cleaned[100, 1])
    assert np.isnan(cleaned[:, 0]).sum() == np.isnan(cleaned[:, 1]).sum() + 1
    quantiles = ewm_insample_winsorising(data, nan_replacement_type=ReplacementType.QUANTILES)
    assert np.isfinite(quantiles[100, 0]) and quantiles[100, 0] < 50.0


def test_score_clip_quantile_is_per_column() -> None:
    """A column's score does not depend on the scale of the other columns."""
    rng = np.random.default_rng(42)
    data = np.column_stack([1e-3 * rng.standard_normal(400), rng.standard_normal(400)])
    _, joint = compute_ewm_score(data)
    _, alone = compute_ewm_score(data[:, [0]])
    np.testing.assert_allclose(joint[:, 0], alone[:, 0], equal_nan=True)


def test_markovian_score_two_dimensional_branch_matches_one_dimensional() -> None:
    """The 2-d recursion updates at regular points and holds at outliers, as the 1-d one does."""
    rng = np.random.default_rng(43)
    data = rng.standard_normal((80, 2)) * 0.1
    data[30, 0] = 100.0
    data[50, 1] = np.nan
    clean, ewm, ewm2, score = ewm_winsdor_markovian_score(data, init_value=np.zeros(2),
                                                          init_var=np.full(2, 0.01))
    for column in range(2):
        clean_1d, ewm_1d, ewm2_1d, score_1d = ewm_winsdor_markovian_score(
            data[:, column], init_value=0.0, init_var=0.01)
        np.testing.assert_allclose(ewm[:, column], ewm_1d)
        np.testing.assert_allclose(ewm2[:, column], ewm2_1d)
        np.testing.assert_allclose(clean[:, column], clean_1d, equal_nan=True)
    assert ewm[30, 0] == ewm[29, 0] and clean[30, 0] == clean[29, 0]
    assert ewm[50, 1] == ewm[49, 1]  # a missing observation holds the state


@pytest.mark.parametrize('policy', [OutlierPolicyTypes.SOFT_RANGE_CEIL_POLICY,
                                    OutlierPolicyTypes.SOFT_POSITIVE_LOG_POLICY])
def test_preset_ewm_ceiling_can_fire(policy: OutlierPolicyTypes) -> None:
    """The presets' EWM ceiling sits below the score bound sqrt(lambda / (1 - lambda))."""
    config = policy.value
    bound = np.sqrt(config.ewm_lambda / (1.0 - config.ewm_lambda))
    assert config.std_ewm_ceil < bound
    np.testing.assert_allclose(config.std_ewm_ceil, score_of_move(10.0, config.ewm_lambda))


def test_large_jump_is_removed_and_five_sigma_move_is_kept() -> None:
    """The preset EWM cut alone removes a 25-sigma jump and keeps a 5-sigma move."""
    rng = np.random.default_rng(44)
    returns = rng.standard_normal((600, 2)) * 0.01
    returns[400, 0] = 0.25   # 25 standard deviations
    returns[400, 1] = 0.05   # 5 standard deviations
    ceiling = OutlierPolicyTypes.SOFT_RANGE_CEIL_POLICY.value.std_ewm_ceil
    ewm_cut = OutlierPolicy(std_ewm_ceil=ceiling)
    cleaned = filter_outliers(returns, ewm_cut)
    assert np.isnan(cleaned[400, 0]) and np.isfinite(cleaned[400, 1])


def test_score_of_move_matches_the_contemporaneous_score() -> None:
    """A k-sigma move against a zero-mean, unit-variance state scores lambda k / sqrt(...)."""
    lam = 0.94
    k = 7.0
    m_prev, v_prev = 0.0, 1.0
    m_t = lam * m_prev + (1.0 - lam) * k
    v_t = lam * v_prev + (1.0 - lam) * k ** 2
    np.testing.assert_allclose(score_of_move(k, lam), (k - m_t) / np.sqrt(v_t))
