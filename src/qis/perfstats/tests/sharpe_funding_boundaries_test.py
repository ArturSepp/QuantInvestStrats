"""Regression coverage for arithmetic-excess funding at price boundaries.

The arithmetic-excess table and compounded excess-NAV path must charge the same lagged
annualized rate over every realized price interval. Month-end dates provide unequal calendar-day
spans while retaining an unambiguous annualization factor of twelve. All expected funding costs,
periodic excess returns, means, and Sharpe ratios are calculated directly in this module without
using a QIS funding or return-reduction helper.
"""

from typing import cast
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# qis
from qis.perfstats.config import PerfParams, PerfStat
from qis.perfstats.perf_stats import compute_ra_perf_table, compute_risk_table
from qis.utils.generic import ColVar


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_PRICE_DATES = pd.date_range('2024-01-31', periods=4, freq='ME')
_ASSET_A_PRICES = (100.0, 105.0, 102.9, 108.045)
_ASSET_A_RETURNS = (0.05, -0.02, 0.05)
_ELAPSED_DAYS = (29.0, 31.0, 30.0)
_LAGGED_RATES = (0.13, 0.17, 0.11)

_ANNUALIZATION_FACTOR = 12.0
_FUNDING_DAYS_PER_YEAR = 365.0
_TOLERANCE = 1e-12


def _prices(include_ragged_asset: bool = False) -> pd.DataFrame:
    """Create the deterministic price panel used by the boundary tests.

    Args:
        include_ragged_asset: Add Asset B, whose first price occurs one month after Asset A.

    Returns:
        New price DataFrame with either one complete history or one complete and one ragged
        history.
    """
    data: dict[str, tuple[float, ...]] = {'Asset A': _ASSET_A_PRICES}
    if include_ragged_asset:
        data['Asset B'] = (np.nan, 200.0, 210.0, 207.9)
    return pd.DataFrame(data, index=_PRICE_DATES)


def _rates(include_pre_sample_rate: bool = True) -> pd.Series:
    """Create time-varying annualized rates observed at consecutive month-ends.

    Args:
        include_pre_sample_rate: Include the December observation used only by the initial
            zero-duration price boundary.

    Returns:
        New rate Series. Rates on the price dates fund the following price intervals.
    """
    rates = pd.Series(
        (0.07, 0.13, 0.17, 0.11, 0.19),
        index=_PRICE_DATES.insert(0, pd.Timestamp('2023-12-31')),
        name='risk_free_rate',
    )
    if not include_pre_sample_rate:
        rates = rates.iloc[1:]
    return rates


def _expected_funding_costs() -> NDArray[np.float64]:
    """Calculate the funding cost for each realized price interval.

    Returns:
        Array containing ``lagged annual rate * elapsed calendar days / 365`` for each interval.
    """
    elapsed_days = np.asarray(_ELAPSED_DAYS, dtype=float)
    lagged_rates = np.asarray(_LAGGED_RATES, dtype=float)
    return lagged_rates * elapsed_days / _FUNDING_DAYS_PER_YEAR


def _expected_asset_a_excess_returns() -> NDArray[np.float64]:
    """Calculate Asset A's three funded returns using explicit interval arithmetic.

    Returns:
        Array containing each simple price return less its independently calculated funding
        cost.
    """
    simple_returns = np.asarray(_ASSET_A_RETURNS, dtype=float)
    return simple_returns - _expected_funding_costs()


def _column_name(perf_stat: PerfStat) -> str:
    """Return a typed table label from a performance-statistic enum member.

    Args:
        perf_stat: Statistic whose full table label is required.

    Returns:
        Full column label stored by the enum's ``ColVar`` value.
    """
    return cast(ColVar, perf_stat.value).name


def _stat(table: pd.DataFrame, asset: str, perf_stat: PerfStat) -> float:
    """Extract one scalar statistic from a performance table.

    Args:
        table: Performance table indexed by asset name.
        asset: Asset row to select.
        perf_stat: Statistic column to select.

    Returns:
        Selected table value converted to a Python float.

    Raises:
        TypeError: If the selected statistic is not a real numeric scalar.
    """
    value = table.loc[asset, _column_name(perf_stat)]
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"expected a real numeric statistic, got {type(value)!r}")
    return float(value)


# =============================================================================
# First-interval arithmetic-excess regression
# =============================================================================

def test_compute_risk_table_charges_every_arithmetic_excess_interval() -> None:
    """Charge funding over the first realized return as well as later returns.

    Asset A earns 5%, -2%, and 5% over intervals of 29, 31, and 30 calendar days. The lagged
    annual rates are 13%, 17%, and 11%, so each expected excess return follows directly from
    ``return - rate * days / 365``. The expected mean, annualized mean, and Sharpe ratio are then
    reduced from those three independently calculated observations.
    """
    prices = _prices()
    rates = _rates()
    original_prices = prices.copy(deep=True)
    original_rates = rates.copy(deep=True)
    expected_returns = _expected_asset_a_excess_returns()
    expected_mean = float(np.mean(expected_returns))
    expected_sharpe = float(
        np.sqrt(_ANNUALIZATION_FACTOR)
        * expected_mean
        / np.std(expected_returns, ddof=1)
    )

    table = compute_risk_table(
        prices=prices,
        perf_params=PerfParams(freq_vol='ME', rates_data=rates),
    )

    np.testing.assert_allclose(
        _stat(table, 'Asset A', PerfStat.AVG_ARITH_EXCESS_RETURN),
        expected_mean,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    np.testing.assert_allclose(
        _stat(table, 'Asset A', PerfStat.AN_ARITH_EXCESS_RETURN),
        _ANNUALIZATION_FACTOR * expected_mean,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    np.testing.assert_allclose(
        _stat(table, 'Asset A', PerfStat.SHARPE_ARITH_EXCESS),
        expected_sharpe,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_series_equal(rates, original_rates)


def test_compute_ra_perf_table_reconciles_arithmetic_and_compounded_funding() -> None:
    """Reconcile arithmetic and compounded statistics to one periodic excess path.

    The compounded result multiplies ``1 + excess_return`` for the same three independently
    funded observations used by the arithmetic mean. Because this fixture is shorter than one
    year, the public compounded statistic reports that exact total rather than annualizing it.
    """
    expected_returns = _expected_asset_a_excess_returns()
    expected_compounded = float(np.prod(1.0 + expected_returns) - 1.0)
    expected_mean = float(np.mean(expected_returns))

    table = compute_ra_perf_table(
        prices=_prices(),
        perf_params=PerfParams(freq_vol='ME', rates_data=_rates()),
    )

    np.testing.assert_allclose(
        _stat(table, 'Asset A', PerfStat.PA_EXCESS_RETURN),
        expected_compounded,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    np.testing.assert_allclose(
        _stat(table, 'Asset A', PerfStat.AVG_ARITH_EXCESS_RETURN),
        expected_mean,
        rtol=0.0,
        atol=_TOLERANCE,
    )


# =============================================================================
# Shape and missing-history boundaries
# =============================================================================

def test_compute_ra_perf_table_matches_series_and_dataframe_arithmetic_excess() -> None:
    """Keep the public Series and one-column DataFrame paths numerically identical.

    ``compute_ra_perf_table`` accepts either pandas shape. Selecting the arithmetic-excess
    columns must produce the same labeled result regardless of whether the caller supplies the
    named Asset A Series or its one-column DataFrame representation.
    """
    prices = _prices()
    perf_params = PerfParams(freq_vol='ME', rates_data=_rates())

    series_table = compute_ra_perf_table(prices=prices['Asset A'], perf_params=perf_params)
    frame_table = compute_ra_perf_table(prices=prices, perf_params=perf_params)
    columns = [
        _column_name(PerfStat.AVG_ARITH_EXCESS_RETURN),
        _column_name(PerfStat.AN_ARITH_EXCESS_RETURN),
        _column_name(PerfStat.SHARPE_ARITH_EXCESS),
    ]

    pd.testing.assert_frame_equal(series_table[columns], frame_table[columns])


def test_compute_risk_table_preserves_ragged_funding_boundaries() -> None:
    """Fund complete and ragged histories without inventing an initial observation.

    Asset A is present at the first price boundary; Asset B begins one month later. A rate on
    January 31 is sufficient to fund Asset A's first return ending February 29, even without a
    December rate. Asset B's first realized return ends March 31 and uses the February rate.
    Leading missing prices remain missing and do not create a zero return in either reduction.
    """
    prices = _prices(include_ragged_asset=True)
    rates = _rates(include_pre_sample_rate=False)
    original_prices = prices.copy(deep=True)
    original_rates = rates.copy(deep=True)
    expected_a = _expected_asset_a_excess_returns()
    expected_b = np.asarray((0.05, -0.01), dtype=float) - _expected_funding_costs()[1:]
    expected_means = np.asarray((np.mean(expected_a), np.mean(expected_b)), dtype=float)
    expected_sharpes = np.sqrt(_ANNUALIZATION_FACTOR) * np.asarray(
        (
            np.mean(expected_a) / np.std(expected_a, ddof=1),
            np.mean(expected_b) / np.std(expected_b, ddof=1),
        ),
        dtype=float,
    )

    table = compute_risk_table(
        prices=prices,
        perf_params=PerfParams(freq_vol='ME', rates_data=rates),
    )

    np.testing.assert_allclose(
        table[_column_name(PerfStat.AVG_ARITH_EXCESS_RETURN)].to_numpy(),
        expected_means,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    np.testing.assert_allclose(
        table[_column_name(PerfStat.SHARPE_ARITH_EXCESS)].to_numpy(),
        expected_sharpes,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_series_equal(rates, original_rates)
