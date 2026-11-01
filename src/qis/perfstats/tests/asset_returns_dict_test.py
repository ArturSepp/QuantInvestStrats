"""Frequency-group return options preserve missing prices and legacy positional calls.

Month-end and quarter-end references are calculated directly from fixed price levels.
No provider data or random sample is needed to check the fill and return conventions.
"""

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.returns import compute_asset_returns_dict, to_returns


@pytest.fixture
def prices() -> pd.DataFrame:
    """Daily prices with a delisted tail, a reporting gap, and a late start."""
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    live = pd.Series(100.0 * 1.1 ** (dates.month - 1), index=dates)
    return pd.DataFrame({
        'live': live,
        'delisted': live.where(dates.month <= 2),
        'gap': live.where(dates.month != 2),
        'late': live.where(dates.month >= 3),
    })


@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('ffill_nans', [False, True])
@pytest.mark.parametrize('is_log_returns', [False, True])
@pytest.mark.parametrize('drop_first', [False, True])
def test_fill_and_return_conventions_against_fixed_levels(
    prices: pd.DataFrame, grouped: bool, ffill_nans: bool,
    is_log_returns: bool, drop_first: bool,
) -> None:
    """Every frequency group matches independent arithmetic, including missing endpoints."""
    frequencies = (pd.Series({'live': 'ME', 'delisted': 'QE', 'gap': 'ME', 'late': 'ME'})
                   if grouped else 'ME')
    actual = compute_asset_returns_dict(
        prices, frequencies, drop_first=drop_first, is_first_zero=False,
        is_log_returns=is_log_returns, ffill_nans=ffill_nans,
    )
    expected = pd.DataFrame({
        'live': [np.nan, .1, .1, .1, .1, .1],
        'delisted': ([np.nan, .1, 0, 0, 0, 0] if ffill_nans else
                     [np.nan, .1, np.nan, np.nan, np.nan, np.nan]),
        'gap': ([np.nan, 0, .21, .1, .1, .1] if ffill_nans else
                [np.nan, np.nan, np.nan, .1, .1, .1]),
        'late': [np.nan, np.nan, np.nan, .1, .1, .1],
    }, index=pd.date_range('2024-01-31', periods=6, freq='ME'))
    expected_groups = {'ME': expected}
    if grouped:
        expected_groups = {
            'ME': expected[['live', 'gap', 'late']],
            'QE': pd.DataFrame(
                {'delisted': [np.nan, 0.0 if ffill_nans else np.nan]},
                index=pd.date_range('2024-03-31', periods=2, freq='QE'),
            ),
        }
    assert set(actual) == set(expected_groups)
    for frequency, reference in expected_groups.items():
        if is_log_returns:
            reference = np.log1p(reference)
        if drop_first:
            reference = reference.iloc[1:]
        pd.testing.assert_frame_equal(actual[frequency], reference, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('is_first_zero', [False, True])
@pytest.mark.parametrize('drop_first', [False, True])
@pytest.mark.parametrize('is_log_returns', [False, True])
def test_default_fill_preserves_legacy_positional_calls(
    prices: pd.DataFrame, is_first_zero: bool, drop_first: bool, is_log_returns: bool,
) -> None:
    """The original fifth argument still selects log returns and omitted fill stays True."""
    actual = compute_asset_returns_dict(
        prices, 'ME', drop_first, is_first_zero, is_log_returns,
    )['ME']
    expected = to_returns(
        prices, freq='ME', drop_first=drop_first, is_first_zero=is_first_zero,
        is_log_returns=is_log_returns, ffill_nans=True,
    )
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize('is_first_zero', [False, True])
@pytest.mark.parametrize('drop_first', [False, True])
def test_no_fill_preserves_initialization_and_drop_options(
    prices: pd.DataFrame, is_first_zero: bool, drop_first: bool,
) -> None:
    """Disabling fill retains the return kernel's ragged-start and first-row policy."""
    actual = compute_asset_returns_dict(
        prices, 'ME', drop_first=drop_first, is_first_zero=is_first_zero,
        is_log_returns=True, ffill_nans=False,
    )['ME']
    expected = to_returns(
        prices, freq='ME', drop_first=drop_first, is_first_zero=is_first_zero,
        is_log_returns=True, ffill_nans=False,
    )
    pd.testing.assert_frame_equal(actual, expected)
