"""EWM covariance on a rebalancing schedule around each asset's inception.

``estimate_rolling_ewma_covar`` returns one full matrix per rebalancing date. For an asset without
data its row and column are missing (NaN) in both the direct and the vol-normalised estimator;
they used to be zero in the direct path, which reads as a riskless asset. On an asset's first
return date the demeaned residual is its first return (the EWM mean is seeded at zero, so the
one-step error is taken against a zero prior), and both estimators give a finite, positive
semi-definite matrix; the vol-normalised path used to give NaN there. The direct recursion starts
from zero, so its first variance carries the warm-up factor ``1 - lambda``; the vol-normalised
estimator rebuilds the covariance from EWM volatilities seeded with the first squared residual,
so its first variance does not. ``warmup_period=k`` additionally masks each asset until it has
more than ``k`` returns.
"""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
import qis

_SPAN = 12
_LAMBDA = 1.0 - 2.0 / (_SPAN + 1.0)
_DATES = pd.date_range('2020-01-31', periods=24, freq='ME')
_LATE_START = 6  # position of the late asset's first price


def _prices() -> pd.DataFrame:
    """Two month-end price paths; ``late`` starts at position ``_LATE_START``."""
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0, 0.04, size=(len(_DATES), 2))
    prices = pd.DataFrame(100.0 * np.cumprod(1.0 + returns, axis=0), index=_DATES,
                          columns=['early', 'late'])
    prices.iloc[:_LATE_START, 1] = np.nan
    return prices


def _covars(normalised: bool, **kwargs) -> dict:
    return qis.estimate_rolling_ewma_covar(prices=_prices(), returns_freq='ME',
                                           rebalancing_freq='ME', span=_SPAN,
                                           is_apply_vol_normalised_returns=normalised,
                                           apply_an_factor=False, **kwargs)


@pytest.mark.parametrize('normalised', [False, True])
def test_entries_are_missing_before_inception(normalised: bool) -> None:
    """Before the late asset's first return its row and column are NaN, not zero."""
    covars = _covars(normalised)
    first_return_date = _DATES[_LATE_START + 1]
    for date, matrix in covars.items():
        if date < first_return_date:
            assert matrix.loc['late'].isna().all() and matrix['late'].isna().all()
            if date > _DATES[1]:
                assert np.isfinite(matrix.loc['early', 'early'])


@pytest.mark.parametrize('normalised', [False, True])
def test_first_return_date_variance_starts_from_the_first_return(normalised: bool) -> None:
    """The first variance is (1 + lambda)/2 x^2, times 1 - lambda in the direct recursion."""
    covars = _covars(normalised)
    date = _DATES[_LATE_START + 1]
    x_late = np.log1p(_prices()['late'].pct_change().iloc[_LATE_START + 1])
    warm_up = 1.0 if normalised else 1.0 - _LAMBDA
    expected = warm_up * 0.5 * (1.0 + _LAMBDA) * x_late ** 2
    assert abs(covars[date].loc['late', 'late'] - expected) < 1e-15


@pytest.mark.parametrize('normalised', [False, True])
def test_matrices_after_inception_are_finite_and_psd(normalised: bool) -> None:
    """From the late asset's first return on, every matrix is finite and PSD."""
    for date, matrix in _covars(normalised).items():
        if date >= _DATES[_LATE_START + 1]:
            values = matrix.to_numpy()
            assert np.all(np.isfinite(values))
            assert np.linalg.eigvalsh(values).min() > -1e-15


@pytest.mark.parametrize('normalised', [False, True])
def test_warmup_period_masks_each_asset_until_it_has_enough_returns(normalised: bool) -> None:
    """With warmup_period=3 an asset is missing until its fourth return, counted per asset."""
    covars = _covars(normalised, warmup_period=3)
    early_first_warm = _DATES[1 + 3]           # early's first return is at position 1
    late_first_warm = _DATES[_LATE_START + 1 + 3]
    for date, matrix in covars.items():
        assert np.isnan(matrix.loc['early', 'early']) == (date < early_first_warm)
        assert np.isnan(matrix.loc['late', 'late']) == (date < late_first_warm)
        if date < late_first_warm:
            assert np.isnan(matrix.loc['early', 'late'])
