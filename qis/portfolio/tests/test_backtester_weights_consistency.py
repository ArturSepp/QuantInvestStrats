"""
what the backtester says when the weights and the prices do not fit each other.

The backtester takes weights as given and never modifies them, so every mismatch between a
weight and the price panel it meets has to be reported rather than absorbed. Three of them are
pinned here: a weighted instrument with no price on its traded date, a nan inside an
instrument's own reported history, and two weight dates resolving onto one traded date.

The third is the one that produced wrong numbers silently. Weights are consumed one row per
rebalancing flag, so a collision does not skip a row - it shifts every later row, and the
staleness grows without bound. ``weight_implementation_lag`` is counted in observations of the
price index rather than calendar days precisely so that it cannot cause one.

Data is the seeded synthetic panel. No network, no data files.
"""
# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
import qis as qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.utils.dates import set_rebalancing_timeindex_on_given_timeindex

CLEAN_TICKERS = ['SEQ_US', 'SBD_TSY']


@pytest.fixture(scope='module')
def clean_prices() -> pd.DataFrame:
    """two fully reported instruments: no warning is expected from the panel itself."""
    return generate_synthetic_prices(start='2010-01-04', end='2014-12-31')[CLEAN_TICKERS]


@pytest.fixture(scope='module')
def ragged_prices() -> pd.DataFrame:
    """a panel whose third instrument starts after the other two."""
    return generate_synthetic_prices(start='2010-01-04', end='2016-12-30')[['SEQ_US', 'SBD_TSY', 'SEQ_EM']]


def test_the_fixture_is_clean(clean_prices: pd.DataFrame) -> None:
    """
    the clean fixture carries no missing price at all.

    Without this the negative assertions below pass trivially: a test that cannot fail reads as a
    guarantee it does not give.
    """
    assert not clean_prices.isna().to_numpy().any()


def test_weighted_instrument_without_a_price_warns(ragged_prices: pd.DataFrame) -> None:
    """the leg is not traded and its weight stays in cash, which the caller is told."""
    with pytest.warns(UserWarning, match='stays in the cash balance'):
        qis.backtest_model_portfolio(prices=ragged_prices,
                                     weights={'SEQ_US': 0.5, 'SBD_TSY': 0.25, 'SEQ_EM': 0.25},
                                     rebalancing_freq='QE',
                                     is_rebalanced_at_first_date=True)


def test_a_zero_weight_without_a_price_is_silent(ragged_prices: pd.DataFrame) -> None:
    """
    a zero weight on an instrument that is not priced yet is not a defect.

    This is the shape of ``examples/portfolios/balanced_60_40_with_btc.py``, which carries the
    late-starting instrument at 0.0 to hold the column: warning on it would train the reader to
    ignore the warning that matters.
    """
    with warnings_are_errors():
        qis.backtest_model_portfolio(prices=ragged_prices,
                                     weights={'SEQ_US': 0.6, 'SBD_TSY': 0.4, 'SEQ_EM': 0.0},
                                     rebalancing_freq='QE',
                                     is_rebalanced_at_first_date=True)


def test_rescaled_weights_do_not_warn(ragged_prices: pd.DataFrame) -> None:
    """the schedule allocates over the priced instruments, so nothing is unfunded."""
    weights = qis.generate_static_weights_schedule(prices=ragged_prices,
                                                   weights={'SEQ_US': 0.5, 'SBD_TSY': 0.25, 'SEQ_EM': 0.25},
                                                   rebalancing_freq='QE')
    with warnings_are_errors():
        qis.backtest_model_portfolio(prices=ragged_prices, weights=weights, ticker='rescaled')


def test_an_interior_nan_warns_and_suggests_ffill(clean_prices: pd.DataFrame) -> None:
    """a hole inside an instrument's own history removes the leg from the nav on those dates."""
    holed = clean_prices.copy()
    holed.loc['2012-03-05':'2012-03-09', 'SBD_TSY'] = np.nan
    with pytest.warns(UserWarning, match='prices.ffill'):
        qis.backtest_model_portfolio(prices=holed, weights={'SEQ_US': 0.6, 'SBD_TSY': 0.4},
                                     rebalancing_freq='QE')


def test_leading_and_trailing_nans_do_not_warn_as_interior(clean_prices: pd.DataFrame) -> None:
    """
    an instrument that starts late or stops reporting is a universe change, not a hole.

    Warning on those would fire on every ragged panel and so on every real one.
    """
    ragged = clean_prices.copy()
    ragged.loc[:'2010-06-30', 'SBD_TSY'] = np.nan   # not trading yet
    ragged.loc['2014-06-30':, 'SBD_TSY'] = np.nan   # no longer reporting
    with warnings_are_errors(allowed='stays in the cash balance'):
        qis.backtest_model_portfolio(prices=ragged, weights={'SEQ_US': 0.6, 'SBD_TSY': 0.4},
                                     rebalancing_freq='QE')


def test_ffill_clears_the_interior_warning(clean_prices: pd.DataFrame) -> None:
    """the remedy the message names actually removes the condition it warns about."""
    holed = clean_prices.copy()
    holed.loc['2012-03-05':'2012-03-09', 'SBD_TSY'] = np.nan
    with warnings_are_errors():
        qis.backtest_model_portfolio(prices=holed.ffill(), weights={'SEQ_US': 0.6, 'SBD_TSY': 0.4},
                                     rebalancing_freq='QE')


def test_colliding_weight_dates_raise(clean_prices: pd.DataFrame) -> None:
    """
    a weights index denser than the price index would shift every row after the collision.

    A calendar-daily weights frame against a business-day panel puts Saturday and Sunday on the
    same traded date. Before this check the two rows became one rebalancing and the remainder of
    the schedule was applied one rebalancing late, for the rest of the backtest.
    """
    calendar_index = pd.date_range(clean_prices.index[0], clean_prices.index[-1], freq='D')
    weights = pd.DataFrame(np.array([[0.6, 0.4]] * len(calendar_index)),
                           index=calendar_index, columns=clean_prices.columns)
    with pytest.raises(ValueError, match='resolve to'):
        qis.backtest_model_portfolio(prices=clean_prices, weights=weights)


def test_lag_is_counted_in_observations_not_calendar_days(clean_prices: pd.DataFrame) -> None:
    """
    every weight row is traded, exactly ``weight_implementation_lag`` observations on.

    Under the calendar-day reading a lag of two collapsed every Thursday/Friday pair onto the
    following Monday - 106 of 523 rows lost on a two-year daily schedule - because a lagged date
    was resolved onto the price grid by taking the next available date.
    """
    weights = pd.DataFrame(np.array([[0.6, 0.4]] * len(clean_prices.index)),
                           index=clean_prices.index, columns=clean_prices.columns)
    for lag in [1, 2, 3]:
        portfolio_data = qis.backtest_model_portfolio(prices=clean_prices, weights=weights,
                                                     weight_implementation_lag=lag)
        is_rebalancing = portfolio_data.is_rebalancing
        traded_dates = is_rebalancing[is_rebalancing == True].index
        assert len(traded_dates) == len(weights.index) - lag, f"rows lost at lag={lag}"
        assert traded_dates.equals(clean_prices.index[lag:]), f"wrong entry dates at lag={lag}"


def test_lag_does_not_touch_instrument_returns(clean_prices: pd.DataFrame) -> None:
    """
    the lag selects the entry price for the units and nothing else.

    ``PortfolioData`` carries the price panel it was given, so a lag that shifted prices rather
    than the trade date would show up here.
    """
    weights = pd.DataFrame(np.array([[0.6, 0.4]] * len(clean_prices.index)),
                           index=clean_prices.index, columns=clean_prices.columns)
    unlagged = qis.backtest_model_portfolio(prices=clean_prices, weights=weights)
    lagged = qis.backtest_model_portfolio(prices=clean_prices, weights=weights,
                                          weight_implementation_lag=2)
    assert unlagged.prices.equals(lagged.prices)
    assert np.allclose(qis.to_returns(unlagged.prices, drop_first=True).to_numpy(),
                       qis.to_returns(lagged.prices, drop_first=True).to_numpy())


def test_weights_trading_past_the_history_are_dropped_with_a_warning(clean_prices: pd.DataFrame) -> None:
    """
    a weight observed too close to the end of the panel is never implemented.

    Dropping rather than clamping is what keeps a lag of one identical to the calendar-day
    behaviour it replaces; the warning is what stops the drop being silent.
    """
    weights = pd.DataFrame(np.array([[0.6, 0.4]] * len(clean_prices.index)),
                           index=clean_prices.index, columns=clean_prices.columns)
    with pytest.warns(UserWarning, match='trade past the end of the price history'):
        portfolio_data = qis.backtest_model_portfolio(prices=clean_prices, weights=weights,
                                                      weight_implementation_lag=3)
    is_rebalancing = portfolio_data.is_rebalancing
    assert int(is_rebalancing.sum()) == len(weights.index) - 3


def test_lag_one_reproduces_the_calendar_day_schedule(clean_prices: pd.DataFrame) -> None:
    """
    the regression pin for the change of unit.

    A lag of one is what every ``optimalportfolios`` call site and every qis example passes, and
    a calendar-day shift of one resolved onto the price grid lands on the same observation as a
    one-observation shift, for any weights index that sits on the price grid. Results at lag=1
    therefore do not move; results at lag>=2 do, and were wrong before.
    """
    for freq in ['ME', 'QE', 'W-FRI', 'B']:
        weight_dates = qis.generate_rebalancing_indicators(df=clean_prices, freq=freq,
                                                           include_start_date=True,
                                                           return_true_only=True).index
        weights = pd.DataFrame(np.array([[0.6, 0.4]] * len(weight_dates)),
                               index=weight_dates, columns=clean_prices.columns)
        calendar_day = set_rebalancing_timeindex_on_given_timeindex(
            given_index=clean_prices.index,
            rebalancing_index=weight_dates + pd.Timedelta(days=1))
        with warnings_are_errors(allowed='trade past the end of the price history'):
            portfolio_data = qis.backtest_model_portfolio(prices=clean_prices, weights=weights,
                                                          weight_implementation_lag=1)
        assert portfolio_data.is_rebalancing.equals(calendar_day), f"schedule moved at freq={freq}"


def warnings_are_errors(allowed: str = None):
    """
    context manager turning UserWarning into an error, except one whose message contains ``allowed``.

    ``pytest.warns`` asserts a warning was raised; there is no built-in inverse, and
    ``filterwarnings('error')`` alone cannot make an exception for the warning a test is not
    about.

    Args:
        allowed: substring of the one warning message that stays a warning. None forbids all

    Returns:
        a context manager for use in a ``with`` block
    """
    import warnings
    from contextlib import contextmanager

    @contextmanager
    def manager():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            yield
        unexpected = [str(w.message) for w in caught
                      if issubclass(w.category, UserWarning)
                      and (allowed is None or allowed not in str(w.message))]
        assert not unexpected, f"unexpected warnings: {unexpected}"

    return manager()
