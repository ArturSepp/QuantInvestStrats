"""
pytest unit tests for qis.tests.synthetic_data

These pin the properties that other tests rely on: reproducibility, the shape of the panel, and
the presence of each declared data defect. They deliberately do not pin exact prices — the
generator is frozen by convention, not by a stored baseline.
"""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.tests.synthetic_data import (BENCHMARK_TICKER, GROUP_ORDER, SMOOTHING_AR1,
                                      SYNTHETIC_UNIVERSE, DataQuirk, SyntheticInstrument,
                                      generate_synthetic_prices, generate_synthetic_universe)


def _ticker_with_quirk(quirk: DataQuirk) -> str:
    return next(x.ticker for x in SYNTHETIC_UNIVERSE if x.quirk == quirk)


def test_same_seed_reproduces_panel() -> None:
    pd.testing.assert_frame_equal(generate_synthetic_prices(), generate_synthetic_prices())


def test_different_seed_changes_panel() -> None:
    first = generate_synthetic_prices(seed=1)
    second = generate_synthetic_prices(seed=2)
    assert not np.allclose(first.to_numpy(), second.to_numpy(), equal_nan=True)


def test_clean_panel_has_no_nans() -> None:
    prices = generate_synthetic_prices(apply_quirks=False)
    assert prices.isna().to_numpy().sum() == 0
    assert (prices.to_numpy() > 0.0).all()


def test_clean_vol_matches_target() -> None:
    prices = generate_synthetic_prices(apply_quirks=False)
    realised = np.log(prices).diff().std() * np.sqrt(260)
    for instrument in SYNTHETIC_UNIVERSE:
        # 15% tolerance: this is a finite sample, not an identity
        assert realised[instrument.ticker] == pytest.approx(instrument.vol, rel=0.15)


def test_each_quirk_leaves_its_signature() -> None:
    prices = generate_synthetic_prices()
    assert prices[_ticker_with_quirk(DataQuirk.GAPS)].isna().sum() > 0
    assert prices[_ticker_with_quirk(DataQuirk.LATE_START)].first_valid_index() > prices.index[0]
    assert prices[_ticker_with_quirk(DataQuirk.DELISTED)].last_valid_index() < prices.index[-1]
    # a monthly reporter carries one observation per calendar month of the panel
    monthly = prices[_ticker_with_quirk(DataQuirk.MONTHLY_REPORTED)].dropna()
    assert len(monthly) == len(prices.index.to_period('M').unique())


def test_stale_instrument_repeats_prices() -> None:
    prices = generate_synthetic_prices()
    stale = prices[_ticker_with_quirk(DataQuirk.STALE)]
    clean = generate_synthetic_prices(apply_quirks=False)[_ticker_with_quirk(DataQuirk.STALE)]
    assert (stale.diff() == 0.0).sum() > 0
    assert (clean.diff().dropna() == 0.0).sum() == 0


def test_smoothed_returns_are_autocorrelated() -> None:
    prices = generate_synthetic_prices()
    smoothed = np.log(prices[_ticker_with_quirk(DataQuirk.SMOOTHED)]).diff().dropna()
    clean_ticker = _ticker_with_quirk(DataQuirk.CLEAN)
    clean = np.log(generate_synthetic_prices()[clean_ticker]).diff().dropna()
    # the ar(1) smoother induces first-order autocorrelation of about phi / (1 + phi^2)
    assert smoothed.autocorr(lag=1) > 0.5 * SMOOTHING_AR1 / (1.0 + SMOOTHING_AR1 ** 2)
    assert abs(clean.autocorr(lag=1)) < 0.05


def test_universe_metadata_is_aligned() -> None:
    universe = generate_synthetic_universe()
    assert universe.prices.columns.equals(universe.group_data.index)
    assert universe.prices.index.equals(universe.benchmark_prices.index)
    assert list(universe.benchmark_prices.columns) == [BENCHMARK_TICKER]
    assert universe.benchmark_prices.isna().to_numpy().sum() == 0
    assert sorted(universe.group_data.unique()) == sorted(GROUP_ORDER)


def test_invalid_instrument_raises() -> None:
    with pytest.raises(ValueError, match='systematic variance'):
        SyntheticInstrument(ticker='BAD', group='Equities', vol=0.1, drift=0.0,
                            beta_market=0.9, beta_group=0.9)
    with pytest.raises(ValueError, match='vol must be positive'):
        SyntheticInstrument(ticker='BAD', group='Equities', vol=0.0, drift=0.0,
                            beta_market=0.1, beta_group=0.1)


def test_empty_range_raises() -> None:
    with pytest.raises(ValueError, match='no business day'):
        generate_synthetic_prices(start='2025-01-04', end='2025-01-05')  # a weekend
