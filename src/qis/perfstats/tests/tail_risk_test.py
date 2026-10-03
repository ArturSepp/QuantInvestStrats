"""Verify historical CVaR against exact examples and an independent variational formula."""

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.tail_risk import compute_cvar


def test_fractional_tail_mass():
    """Include half of the second worst observation, rather than rounding the tail."""
    sample = pd.Series([-0.10, -0.04, 0.01, 0.02, 0.03])
    assert compute_cvar(sample, 0.7) == pytest.approx(0.08)
    naive = -sample[sample <= sample.quantile(0.3)].mean()
    assert naive != pytest.approx(0.08)


@pytest.mark.parametrize('confidence', [0.01, 0.5, 0.7, 0.95, 0.999])
def test_variational_reference(confidence):
    """Match the minimum of z + E[(loss-z)+]/(1-confidence) at every kink."""
    sample = np.random.default_rng(731).normal(0.003, 0.08, 47)
    sample[:5] = -0.04
    losses = -sample
    candidates = [z + np.maximum(losses - z, 0).mean() / (1 - confidence)
                  for z in np.unique(losses)]
    assert compute_cvar(pd.Series(sample), confidence) == pytest.approx(min(candidates))


@pytest.mark.parametrize('sample, confidence, expected', [
    ([-0.1, -0.1, -0.1, 0.02], 0.5, 0.1),
    ([-0.04] * 10, 0.95, 0.04),
    ([0.02] * 10, 0.95, -0.02),
    ([-0.2, 0.1, 0.2], 0.99, 0.2),
    ([-0.2, -0.1, 0.1, 0.2], 0.5, 0.15),
])
def test_tail_examples(sample, confidence, expected):
    """Keep ties, profitable tails and integer tail sizes consistent."""
    assert compute_cvar(pd.Series(sample), confidence) == pytest.approx(expected)


def test_columnwise_missing_values_and_labels():
    """Drop missing values per column and preserve even duplicate column names."""
    frame = pd.DataFrame({'a': pd.Series([-0.1, None, 0.02], dtype='Float64'),
                          'b': [-0.2, -0.1, 0.02], 'empty': [np.nan] * 3})
    frame.columns = pd.Index(['same', 'same', 'empty'], name='Fund')
    actual = compute_cvar(frame, 0.5)
    expected = pd.Series([0.1, 1 / 6, np.nan], index=frame.columns)
    pd.testing.assert_series_equal(actual, expected)


def test_empty_samples():
    """Return NaN for empty samples and a labelled empty result for no columns."""
    assert np.isnan(compute_cvar(pd.Series(dtype=float)))
    assert np.isnan(compute_cvar(pd.Series([np.nan])))
    assert compute_cvar(pd.DataFrame()).empty
    assert np.isnan(compute_cvar(pd.DataFrame(columns=['a']))['a'])


@pytest.mark.parametrize('confidence', [0, 1, -0.1, 1.1, np.nan, np.inf, True, '0.95', [0.95]])
def test_invalid_confidence(confidence):
    """Reject invalid confidence probabilities instead of returning misleading risk."""
    with pytest.raises(ValueError, match='confidence_level'):
        compute_cvar(pd.Series([0.1]), confidence)


@pytest.mark.parametrize('sample', [[np.inf], [-np.inf], ['bad']])
def test_invalid_returns(sample):
    """Reject nonnumeric and infinite observations."""
    with pytest.raises(ValueError, match='returns'):
        compute_cvar(pd.Series(sample))


def test_invalid_container():
    """Require the documented pandas container contract."""
    with pytest.raises(TypeError, match='pandas'):
        compute_cvar([-0.1, 0.1])
