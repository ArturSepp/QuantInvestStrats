"""Check attribution identities and paired fold timing on deterministic synthetic data."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from qis.perfstats.signal_ic_attribution import compute_beta_reuse_ic, decompose_rank_ic


def test_rank_attribution_with_ties_and_excluded_date():
    dates = pd.date_range('2020-01-31', periods=3, freq='ME')
    scores = pd.DataFrame([[0, 0, 1], [1, 1, 0], [np.nan]*3], index=dates, columns=list('abc'))
    outcome = scores[['c', 'a', 'b']].iloc[::-1]
    weights = pd.Series([3., 1., np.nan], index=dates)
    result = decompose_rank_ic(scores, outcome, weights)
    # Opposite score directions, identical score/outcome rankings on each date.
    # The weighted mean unit rank is half the first unit vector.
    np.testing.assert_allclose(result[['persistent', 'within']].sum(), [.25, .75])
    np.testing.assert_allclose(result.mean_score_rank, [1.75, 1.75, 2.5])
    with pytest.raises(ValueError, match='complete fixed basket'):
        decompose_rank_ic(scores, outcome, weights.fillna(1))


def test_alternating_month_hedges_against_direct_weighted_least_squares():
    rng = np.random.default_rng(427196)
    dates = pd.date_range('2010-01-31', periods=64, freq='ME')
    x = pd.DataFrame(rng.normal(size=(64, 2)), index=dates, columns=['market', 'bond'])
    y = pd.DataFrame(rng.normal(size=(64, 7)), index=dates, columns=list('abcdefg'))
    decisions = dates[25:]
    report, folds = compute_beta_reuse_ic(
        y, x, decisions, score_factor='market', beta_ewma_span=8, rolling_span=4, horizons=[3])
    for row in folds.itertuples():
        t = dates.get_loc(row.formation_date)
        n = np.arange(t + 1)
        w = np.sqrt((7 / 9) ** (t - n))
        fits = []
        for mask in (n % 2 == 0, n % 2 == 1, n >= 0):
            fits.append(np.linalg.lstsq(x.iloc[n[mask]].values * w[mask, None],
                                        y.iloc[n[mask]].values * w[mask, None], rcond=None)[0])
        k = 0 if row.score_fold == 'A' else 1
        score = -fits[k][0]
        f = x.iloc[t+1:t+4].sum().values
        raw = y.iloc[t+1:t+4].sum().values
        expected = []
        for loadings in (fits[k], fits[1-k]):
            bond = fits[2][1] if row.hedge == 'score_factor' else loadings[1]
            expected.append(spearmanr(score, raw - f[0]*loadings[0] - f[1]*bond).statistic)
        np.testing.assert_allclose([row.same, row.disjoint], expected, atol=1e-12)
    for row in report.summary.query("target == 'same'").itertuples():
        recent = folds.loc[folds.hedge.eq(row.hedge)].groupby('realisation_date')[
            ['same', 'disjoint']].mean().tail(4)
        assert row.n_selected == 4
        delta = recent.same - recent.disjoint
        np.testing.assert_allclose(row.mean_ic_minus_reference, delta.mean())
        assert pd.isna(row.difference_lower)  # Four selected dates are below the default warmup.


def test_varying_membership_conditions_asset_means_on_matched_dates():
    scores = pd.DataFrame([[1., 2., np.nan], [9., 1., 2.]], columns=list('abc'))
    outcomes = pd.DataFrame([[1., 2., 5.], [np.nan, 1., 2.]], columns=list('abc'))
    weights = pd.Series([3., 1.])
    parts = decompose_rank_ic(scores, outcomes, weights, varying_membership=True)
    np.testing.assert_allclose(parts.persistent, [.375, .125, .125])
    np.testing.assert_allclose(parts.within, [0., .375, 0.], atol=1e-15)
    np.testing.assert_allclose(parts.mean_score_rank, [1., 1.75, 2.])
    with pytest.raises(ValueError, match='nonconstant'):
        decompose_rank_ic(scores, outcomes * 0, weights, varying_membership=True)
