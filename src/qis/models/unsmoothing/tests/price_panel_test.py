"""Native-mark support, cadence isolation and causal price-reconstruction contracts."""
import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.models.unsmoothing import price_panel as panel


@pytest.fixture(scope='module')
def prices():
    return generate_synthetic_prices(start='2005-01-03', end='2024-12-31')[
        ['SAL_HF', 'SAL_PE', 'SEQ_US']]


def arguments(prices, method, is_log_returns, frequency):
    return dict(prices=prices[['SAL_HF', 'SAL_PE']], frequency=frequency, method=method,
                span={'ME': 120, 'QE': 40}, warmup_period={'ME': 12, 'QE': 4},
                is_log_returns=is_log_returns, factor_prices=prices['SEQ_US'])


@pytest.mark.parametrize('method', list(qis.UnsmoothingMethod))
@pytest.mark.parametrize('is_log_returns', [True, False])
@pytest.mark.parametrize('frequency', ['ME', 'QE'])
def test_appending_future_marks_does_not_change_completed_prefix(
        prices, method, is_log_returns, frequency):
    cut = pd.Timestamp('2018-12-31')
    args = arguments(prices, method, is_log_returns, frequency)
    full, diagnostics = qis.compute_unsmoothed_price_panel(**args, return_diagnostics=True)
    args['prices'] = args['prices'].loc[:cut]
    args['factor_prices'] = args['factor_prices'].loc[:cut]
    prefix, prefix_diagnostics = qis.compute_unsmoothed_price_panel(
        **args, return_diagnostics=True)
    pd.testing.assert_frame_equal(prefix, full.loc[:cut], check_exact=True)
    pd.testing.assert_frame_equal(
        prefix_diagnostics, diagnostics.loc[diagnostics.date <= cut].reset_index(drop=True),
        check_exact=True)


@pytest.mark.parametrize('method', list(qis.UnsmoothingMethod))
@pytest.mark.parametrize('is_log_returns', [True, False])
def test_mixed_cadence_matches_independent_sleeves(prices, method, is_log_returns):
    frequencies = pd.Series({'SAL_HF': 'ME', 'SAL_PE': 'QE'})
    args = arguments(prices, method, is_log_returns, frequencies)
    original = args['prices'].copy(deep=True)
    combined, diagnostics = qis.compute_unsmoothed_price_panel(**args, return_diagnostics=True)
    for ticker, frequency in frequencies.items():
        separate = qis.compute_unsmoothed_price_panel(
            **dict(args, prices=original[[ticker]], frequency=frequency))
        pd.testing.assert_frame_equal(combined[[ticker]], separate, check_exact=True)
        detail = diagnostics.loc[diagnostics.ticker == ticker]
        assert set(detail.frequency) == {frequency}
        np.testing.assert_array_equal(
            detail.applied_ar.iloc[1:], detail.fitted_ar.iloc[:-1])
    pd.testing.assert_frame_equal(args['prices'], original, check_exact=True)


@pytest.mark.parametrize('method', list(qis.UnsmoothingMethod))
def test_missing_quarter_is_not_filled_before_regression(prices, method, monkeypatch):
    captured = []

    def inspect_returns(returns, **kwargs):
        captured.append(returns.copy())
        return returns

    monkeypatch.setattr(panel, 'adjust_returns_with_ar', inspect_returns)
    monkeypatch.setattr(panel, 'adjust_returns_with_joint_unsmoothing', inspect_returns)
    args = arguments(prices, method, True, 'QE')
    args['prices'] = args['prices'].copy()
    args['prices'].loc['2017-07-01':'2017-09-30'] = np.nan
    qis.compute_unsmoothed_price_panel(**args)
    assert captured[0].loc['2017-09-30'].isna().all()
    assert captured[0].loc['2017-12-31'].isna().all()
    assert captured[0].loc['2018-03-31'].notna().all()


def test_stale_ending_month_is_not_an_observation(prices, monkeypatch):
    captured = []

    def sampled_returns(native, **kwargs):
        # Deliberately supply a value: the independent native-mark mask must reject it.
        return pd.DataFrame(0.01, index=pd.date_range('2016-03-31', '2020-12-31', freq='QE'),
                            columns=native.columns)

    def inspect_returns(returns, **kwargs):
        captured.append(returns.copy())
        return returns

    monkeypatch.setattr(panel, 'to_returns', sampled_returns)
    monkeypatch.setattr(panel, 'adjust_returns_with_ar', inspect_returns)
    args = arguments(prices, qis.UnsmoothingMethod.AR, False, 'QE')
    args['prices'] = args['prices'].copy()
    args['prices'].loc['2017-09-01':'2017-09-30'] = np.nan
    qis.compute_unsmoothed_price_panel(**args)
    assert captured[0].loc['2017-09-30'].isna().all()
    assert captured[0].loc['2017-06-30'].notna().all()


def test_weekend_period_end_nav_is_available_on_next_business_day(prices, monkeypatch):
    sampled_nav = pd.DataFrame({'SAL_HF': [1., 2., 3.], 'SAL_PE': [4., 5., 6.]},
                              index=pd.to_datetime(['2017-06-30', '2017-09-30', '2017-12-31']))
    monkeypatch.setattr(panel, 'adjust_returns_with_ar', lambda returns, **kwargs: returns)
    monkeypatch.setattr(panel, 'returns_to_nav', lambda returns: sampled_nav)
    result = qis.compute_unsmoothed_price_panel(
        **arguments(prices, qis.UnsmoothingMethod.AR, False, 'QE'))
    np.testing.assert_array_equal(result.loc['2017-09-29'], [1., 4.])
    np.testing.assert_array_equal(result.loc['2017-10-02'], [2., 5.])


@pytest.mark.parametrize('method', list(qis.UnsmoothingMethod))
def test_insufficient_warmup_fails_with_asset_identity(prices, method):
    args = arguments(prices.loc['2024-01-01':], method, True, 'QE')
    with pytest.raises(ValueError, match='SAL_HF'):
        qis.compute_unsmoothed_price_panel(**args)


@pytest.mark.parametrize('changes,match', [
    ({'span': {'ME': 120}}, 'span has no horizon'),
    ({'warmup_period': {'ME': 12}}, 'warmup_period has no horizon'),
    ({'span': {'QE': True}}, 'positive integer'),
    ({'frequency': pd.Series({'SAL_HF': 'QE'})}, 'every asset'),
    ({'factor_prices': None}, 'requires factor_prices'),
])
def test_invalid_configuration_fails_explicitly(prices, changes, match):
    args = arguments(prices, qis.UnsmoothingMethod.JOINT, True, 'QE')
    with pytest.raises(ValueError, match=match):
        qis.compute_unsmoothed_price_panel(**dict(args, **changes))


def test_empty_warmup_is_explicit_and_leading_missing_marks_are_preserved(prices):
    args = arguments(prices, qis.UnsmoothingMethod.JOINT, True, 'QE')
    args['prices'] = args['prices'].copy()
    args['prices'].loc[:'2007-12-31', 'SAL_PE'] = np.nan
    args['warmup_period'] = {}
    result = qis.compute_unsmoothed_price_panel(**args)
    assert result.loc[:'2007-12-31', 'SAL_PE'].isna().all()
    assert result.loc['2015-01-01':, 'SAL_PE'].notna().all()
