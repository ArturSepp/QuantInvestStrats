"""
a static allocation over a universe whose instruments start and stop at different dates.

``generate_static_weights_schedule`` is the answer to a specific defect: a fixed weight vector
against a ragged panel leaves the missing instrument's weight in cash, silently, for as long as
the instrument has no price. These tests pin the reallocation, the exposure it preserves, and the
two shapes of the schedule that make the resulting frame safe to hand to the backtester - its
index is a subset of the price dates, and it never rebalances more often than the panel does.

Data is the seeded synthetic panel, which already carries a late start, a delisted tail and
scattered gaps. No network, no data files.
"""
# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
import qis as qis
from qis.datasets.synthetic import generate_synthetic_prices

TICKERS = ['SEQ_US', 'SBD_TSY', 'SEQ_EM']  # SEQ_EM carries the late start
STATIC_WEIGHTS = {'SEQ_US': 0.5, 'SBD_TSY': 0.25, 'SEQ_EM': 0.25}


@pytest.fixture(scope='module')
def prices() -> pd.DataFrame:
    """seeded panel restricted to two clean instruments and one that starts late."""
    return generate_synthetic_prices(start='2010-01-04', end='2016-12-30')[TICKERS]


def test_late_start_is_reallocated_over_the_priced_instruments(prices: pd.DataFrame) -> None:
    """before the late instrument prices, its weight goes to the other two in proportion."""
    weights = qis.generate_static_weights_schedule(prices=prices, weights=STATIC_WEIGHTS,
                                                   rebalancing_freq='QE')
    inception = prices['SEQ_EM'].first_valid_index()
    before = weights.loc[weights.index < inception, :]
    assert len(before.index) > 0, "the fixture must contain rebalancings before the late start"
    assert np.allclose(before['SEQ_EM'].to_numpy(), 0.0)
    assert np.allclose(before['SEQ_US'].to_numpy(), 0.5 / 0.75)
    assert np.allclose(before['SBD_TSY'].to_numpy(), 0.25 / 0.75)
    assert np.allclose(before.sum(axis=1).to_numpy(), 1.0)

    after = weights.loc[weights.index > inception, :]
    assert np.allclose(after.to_numpy(), np.array([0.5, 0.25, 0.25]))


def test_total_exposure_is_preserved_not_forced_to_one(prices: pd.DataFrame) -> None:
    """
    a book that is 90% invested by design stays 90% invested.

    Forcing the row to one would lever it by 11% on exactly the dates an instrument is missing,
    which is the failure that makes a plain row normalisation the wrong default here.
    """
    static = {'SEQ_US': 0.5, 'SBD_TSY': 0.25, 'SEQ_EM': 0.15}  # 0.90 invested, 10% cash by design
    weights = qis.generate_static_weights_schedule(prices=prices, weights=static,
                                                   rebalancing_freq='QE')
    inception = prices['SEQ_EM'].first_valid_index()
    before = weights.loc[weights.index < inception, :]
    assert np.allclose(before.sum(axis=1).to_numpy(), 0.90)
    assert np.allclose(before['SEQ_US'].to_numpy(), 0.90 * 0.5 / 0.75)

    forced = qis.generate_static_weights_schedule(prices=prices, weights=static,
                                                  rebalancing_freq='QE',
                                                  is_preserve_total_exposure=False)
    assert np.allclose(forced.loc[forced.index < inception, :].sum(axis=1).to_numpy(), 1.0)


def test_no_rescaling_leaves_the_weight_in_cash(prices: pd.DataFrame) -> None:
    """``is_rescale_to_live_universe=False`` is the cash residual, with 0.0 rather than nan."""
    weights = qis.generate_static_weights_schedule(prices=prices, weights=STATIC_WEIGHTS,
                                                   rebalancing_freq='QE',
                                                   is_rescale_to_live_universe=False)
    inception = prices['SEQ_EM'].first_valid_index()
    before = weights.loc[weights.index < inception, :]
    assert np.allclose(before.sum(axis=1).to_numpy(), 0.75)
    assert not before.isna().to_numpy().any()


def test_a_full_universe_returns_the_specification_unchanged(prices: pd.DataFrame) -> None:
    """with nothing missing the schedule is the static vector repeated, rescaled by 1.0."""
    clean = prices[['SEQ_US', 'SBD_TSY']].dropna()
    weights = qis.generate_static_weights_schedule(prices=clean,
                                                   weights={'SEQ_US': 0.6, 'SBD_TSY': 0.4},
                                                   rebalancing_freq='QE')
    assert np.allclose(weights.to_numpy(), np.array([0.6, 0.4]))


def test_a_date_with_no_priced_instrument_is_all_zeros(prices: pd.DataFrame) -> None:
    """no live instrument means no allocation, not a division by zero."""
    blanked = prices.copy()
    blanked.loc[:, :] = np.nan
    weights = qis.generate_static_weights_schedule(prices=blanked, weights=STATIC_WEIGHTS,
                                                   rebalancing_freq='QE')
    assert np.allclose(weights.to_numpy(), 0.0)
    assert not weights.isna().to_numpy().any()


def test_schedule_dates_are_price_dates(prices: pd.DataFrame) -> None:
    """
    the returned index is a subset of the price dates.

    This is the property that makes the frame safe to pass straight back to the backtester: a
    weights frame denser than the price panel resolves two weight dates onto one traded date,
    and the backtester now refuses it.
    """
    weights = qis.generate_static_weights_schedule(prices=prices, weights=STATIC_WEIGHTS,
                                                   rebalancing_freq='QE')
    assert weights.index.isin(prices.index).all()
    assert weights.index.is_monotonic_increasing
    assert not weights.index.has_duplicates
    assert weights.index[0] == prices.index[0], "include_start_date defaults to True"


def test_round_trip_through_the_backtester_is_fully_invested(prices: pd.DataFrame) -> None:
    """realised weights sum to one at every rebalancing, including before the late start."""
    weights = qis.generate_static_weights_schedule(prices=prices, weights=STATIC_WEIGHTS,
                                                   rebalancing_freq='QE')
    portfolio_data = qis.backtest_model_portfolio(prices=prices, weights=weights, ticker='rescaled')
    is_rebalancing = portfolio_data.is_rebalancing
    realised = portfolio_data.weights.loc[is_rebalancing[is_rebalancing == True].index, :]
    assert np.allclose(realised.sum(axis=1).to_numpy(), 1.0)


def test_a_book_with_no_net_exposure_raises(prices: pd.DataFrame) -> None:
    """a market-neutral specification has no total exposure to preserve."""
    with pytest.raises(ValueError, match='no total exposure to preserve'):
        qis.generate_static_weights_schedule(prices=prices,
                                             weights={'SEQ_US': 1.0, 'SBD_TSY': -1.0, 'SEQ_EM': 0.0},
                                             rebalancing_freq='QE')


def test_live_weights_that_cancel_raise(prices: pd.DataFrame) -> None:
    """
    a rescale onto a live set whose weights cancel is undefined, and says so.

    The specification has net exposure, so it passes the check above; it is the *live* subset on
    the dates before the late start that sums to zero.
    """
    with pytest.raises(ValueError, match='live weights cancel'):
        qis.generate_static_weights_schedule(prices=prices,
                                             weights={'SEQ_US': 1.0, 'SBD_TSY': -1.0, 'SEQ_EM': 0.5},
                                             rebalancing_freq='QE')


def test_align_weights_to_columns_agrees_across_the_accepted_forms(prices: pd.DataFrame) -> None:
    """a Dict, a pd.Series, a List and an np.ndarray naming the same allocation agree."""
    columns = prices.columns
    from_dict = qis.align_weights_to_columns(weights=STATIC_WEIGHTS, columns=columns)
    from_series = qis.align_weights_to_columns(weights=pd.Series(STATIC_WEIGHTS), columns=columns)
    from_list = qis.align_weights_to_columns(weights=[0.5, 0.25, 0.25], columns=columns)
    from_array = qis.align_weights_to_columns(weights=np.array([0.5, 0.25, 0.25]), columns=columns)
    for aligned in [from_series, from_list, from_array]:
        assert np.allclose(aligned.to_numpy(), from_dict.to_numpy())
    assert from_dict.index.equals(columns)

    # a Dict is aligned by name, so its order does not matter; a List is positional
    shuffled = qis.align_weights_to_columns(weights={'SEQ_EM': 0.25, 'SEQ_US': 0.5, 'SBD_TSY': 0.25},
                                            columns=columns)
    assert np.allclose(shuffled.to_numpy(), from_dict.to_numpy())


def test_align_weights_to_columns_rejects_bad_input(prices: pd.DataFrame) -> None:
    """the shared normaliser keeps the error contract the backtester documented."""
    with pytest.raises(ValueError):
        qis.align_weights_to_columns(weights=[0.5, 0.5], columns=prices.columns)
    with pytest.raises(ValueError):
        qis.align_weights_to_columns(weights={'NOT_A_TICKER': 1.0}, columns=prices.columns)
    with pytest.raises(NotImplementedError):
        qis.align_weights_to_columns(weights='0.5', columns=prices.columns)
