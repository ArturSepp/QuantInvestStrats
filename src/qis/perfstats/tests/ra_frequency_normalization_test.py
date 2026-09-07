"""Regression coverage for risk-adjusted table frequency normalization.

The p.a., log, excess, and Sortino ratios must pair return numerators with risk denominators
derived from the same ``freq_vol`` price boundaries. Mixed complete and leading-ragged histories
exercise column-local support, while off-grid endpoints distinguish ratio inputs from the native
return columns that the public table continues to expose. Expected statistics are calculated here
from explicit pandas and NumPy operations rather than through QIS performance helpers.
"""

from numbers import Real

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from qis.perfstats.config import PerfParams, PerfStat, ReturnTypes
from qis.perfstats.perf_stats import compute_ra_perf_table


_FREQUENCY = "QE"
_ANNUALIZATION_FACTOR = 4.0
_RETURN_DAYS_PER_YEAR = 365.25
_FUNDING_DAYS_PER_YEAR = 365.0
_ANNUAL_RATE = 0.04
_TOLERANCE = 1e-12
_ASSETS = ("Complete", "Leading ragged")

_RATIO_STATS = (
    PerfStat.SHARPE_RF0,
    PerfStat.SHARPE_EXCESS,
    PerfStat.SHARPE_LOG_AN,
    PerfStat.SHARPE_LOG_EXCESS,
    PerfStat.SORTINO_RATIO,
)


def _column_name(perf_stat: PerfStat) -> str:
    """Return the full table label carried by a performance statistic."""
    name = perf_stat.value.name
    if not isinstance(name, str):
        raise TypeError(f"expected a string column label, got {type(name)!r}")
    return name


def _stat(table: pd.DataFrame, asset: str, perf_stat: PerfStat) -> float:
    """Extract one real-valued statistic from a performance table."""
    value: object = table.loc[asset, _column_name(perf_stat)]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if not isinstance(value, Real):
        raise TypeError("expected a real statistic")
    return float(value)


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Select one column while keeping the pandas shape explicit to static checkers."""
    values: object = frame.loc[:, column]
    if not isinstance(values, pd.Series):
        raise TypeError("expected a Series column")
    return values


def _values(series: pd.Series) -> NDArray[np.float64]:
    """Convert a real-valued Series to a typed NumPy vector for reference arithmetic."""
    values: NDArray[np.float64] = series.to_numpy(dtype=np.float64)
    return values


def _elapsed_years(index: pd.DatetimeIndex) -> float:
    """Convert explicit datetime boundaries to the return annualization basis."""
    start: object = index[0]
    end: object = index[-1]
    if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp):
        raise TypeError("expected timestamp boundaries")
    return (end - start).days / _RETURN_DAYS_PER_YEAR


def _price_panel() -> pd.DataFrame:
    """Create complete and leading-ragged daily histories with off-grid endpoints."""
    dates = pd.bdate_range("2021-01-15", "2025-05-16")
    steps = np.arange(len(dates), dtype=float)
    log_returns = 0.0002 + 0.006 * np.sin(steps / 17.0) - 0.004 * np.cos(steps / 31.0)
    complete = 100.0 * np.exp(np.cumsum(log_returns))
    # Put a material move after the last complete quarter so native and sampled support differ.
    complete[-1] *= 1.12

    ragged = 0.75 * complete
    ragged[dates < pd.Timestamp("2021-08-17")] = np.nan
    return pd.DataFrame({"Complete": complete, "Leading ragged": ragged}, index=dates)


def _rates(prices: pd.DataFrame) -> pd.Series:
    """Create a constant annual rate on a calendar-daily observation grid."""
    price_index = pd.DatetimeIndex(prices.index)
    dates = pd.date_range(price_index[0] - pd.Timedelta(days=1), price_index[-1])
    return pd.Series(_ANNUAL_RATE, index=dates, name="risk_free_rate")


def _sample_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Sample complete quarters independently with pandas calendar operations."""
    price_index = pd.DatetimeIndex(prices.index)
    dates = pd.date_range(price_index[0], price_index[-1], freq=_FREQUENCY)
    return prices.ffill().reindex(dates, method="ffill").ffill()


def _annualized_compound_return(period_returns: pd.Series) -> float:
    """Annualize compounded periodic returns over their explicit calendar boundaries."""
    return_index = pd.DatetimeIndex(period_returns.index)
    num_years = _elapsed_years(return_index)
    # Row zero establishes the price boundary; it does not represent a realized return interval.
    total_return = float(np.prod(1.0 + _values(period_returns)[1:]) - 1.0)
    return float((1.0 + total_return) ** (1.0 / num_years) - 1.0)


def _expected_statistics(
    prices: pd.Series,
    return_type: ReturnTypes,
    with_rates: bool,
) -> dict[PerfStat, float]:
    """Calculate the ratio family from one canonical sampled price history."""
    prices = prices.dropna()
    price_index = pd.DatetimeIndex(prices.index)
    price_values = _values(prices)
    simple_values = np.full(price_values.shape, np.nan, dtype=np.float64)
    simple_values[1:] = price_values[1:] / price_values[:-1] - 1.0
    simple_returns = pd.Series(simple_values, index=price_index)
    # return_type selects the risk denominator; CAGR numerators compound simple returns in
    # both modes.
    if return_type == ReturnTypes.LOG:
        risk_values = np.diff(np.log(price_values))
    else:
        risk_values = simple_values[1:]

    vol = float(np.sqrt(_ANNUALIZATION_FACTOR) * np.std(risk_values, ddof=1))
    negative_values = risk_values[risk_values < 0.0]
    downside_vol = float(np.sqrt(_ANNUALIZATION_FACTOR) * np.std(negative_values, ddof=1))

    pa_return = _annualized_compound_return(simple_returns)
    excess_values = np.empty(simple_values.shape, dtype=np.float64)
    if with_rates:
        elapsed_days: NDArray[np.float64] = (
            price_index.to_series().diff().dt.days.to_numpy(dtype=np.float64)
        )
        excess_values[:] = simple_values - _ANNUAL_RATE * elapsed_days / _FUNDING_DAYS_PER_YEAR
        excess_values[0] = 0.0
        excess_returns = pd.Series(excess_values, index=price_index)
        pa_excess_return = _annualized_compound_return(excess_returns)
    else:
        excess_returns = simple_returns
        excess_values[:] = simple_values
        pa_excess_return = pa_return

    return {
        PerfStat.SHARPE_RF0: pa_return / vol,
        PerfStat.SHARPE_EXCESS: pa_excess_return / vol,
        PerfStat.SHARPE_LOG_AN: float(np.log1p(pa_return) / vol),
        PerfStat.SHARPE_LOG_EXCESS: float(np.log1p(pa_excess_return) / vol),
        PerfStat.SORTINO_RATIO: pa_excess_return / downside_vol,
        PerfStat.SHARPE_ARITH: float(
            np.sqrt(_ANNUALIZATION_FACTOR)
            * np.mean(simple_values[1:])
            / np.std(simple_values[1:], ddof=1)
        ),
        PerfStat.SHARPE_ARITH_EXCESS: float(
            np.sqrt(_ANNUALIZATION_FACTOR)
            * np.mean(excess_values[1:])
            / np.std(excess_values[1:], ddof=1)
        ),
    }


@pytest.mark.parametrize("return_type", [ReturnTypes.LOG, ReturnTypes.RELATIVE])
@pytest.mark.parametrize("with_rates", [False, True], ids=["zero-rate", "funded"])
def test_compute_ra_perf_table_aligns_ratio_numerators_with_risk_boundaries(
    return_type: ReturnTypes,
    with_rates: bool,
) -> None:
    """Match ratio outputs to one sampled-support oracle for complete and ragged assets."""
    prices = _price_panel()
    sampled_prices = _sample_prices(prices)
    rates = _rates(prices) if with_rates else None
    perf_params = PerfParams(
        freq_vol=_FREQUENCY,
        return_type=return_type,
        rates_data=rates,
    )

    actual = compute_ra_perf_table(prices=prices, perf_params=perf_params)
    same_support = compute_ra_perf_table(prices=sampled_prices, perf_params=perf_params)

    for asset in _ASSETS:
        expected = _expected_statistics(_series(sampled_prices, asset), return_type, with_rates)
        for perf_stat in _RATIO_STATS:
            # Pin each ratio to a calculation independent of the QIS performance helpers.
            np.testing.assert_allclose(
                _stat(actual, asset, perf_stat),
                expected[perf_stat],
                rtol=0.0,
                atol=_TOLERANCE,
            )
            # Equivalent pre-sampled input must converge to the same public result.
            np.testing.assert_allclose(
                _stat(actual, asset, perf_stat),
                _stat(same_support, asset, perf_stat),
                rtol=0.0,
                atol=_TOLERANCE,
            )


def test_compute_ra_perf_table_preserves_native_returns_and_independent_ratios() -> None:
    """Keep native return columns, arithmetic Sharpes, Calmar, and caller inputs unchanged."""
    prices = _price_panel()
    rates = _rates(prices)
    prices_before = prices.copy(deep=True)
    rates_before = rates.copy(deep=True)
    sampled_prices = _sample_prices(prices)
    perf_params = PerfParams(freq_vol=_FREQUENCY, rates_data=rates)

    table = compute_ra_perf_table(prices=prices, perf_params=perf_params)

    for asset in _ASSETS:
        native = _series(prices, asset).dropna()
        native_index = pd.DatetimeIndex(native.index)
        num_years = _elapsed_years(native_index)
        native_values = _values(native)
        total_return = float(native_values[-1] / native_values[0] - 1.0)
        pa_return = float((1.0 + total_return) ** (1.0 / num_years) - 1.0)
        native_return_values = np.full(native_values.shape, np.nan, dtype=np.float64)
        native_return_values[1:] = native_values[1:] / native_values[:-1] - 1.0
        native_elapsed_days: NDArray[np.float64] = (
            native_index.to_series().diff().dt.days.to_numpy(dtype=np.float64)
        )
        native_excess_values = (
            native_return_values - _ANNUAL_RATE * native_elapsed_days / _FUNDING_DAYS_PER_YEAR
        )
        native_excess_values[0] = 0.0
        native_excess_returns = pd.Series(native_excess_values, index=native_index)
        pa_excess_return = _annualized_compound_return(native_excess_returns)
        expected = _expected_statistics(
            _series(sampled_prices, asset).dropna(), ReturnTypes.LOG, True
        )

        visible_returns: dict[PerfStat, float] = {
            PerfStat.TOTAL_RETURN: total_return,
            PerfStat.PA_RETURN: pa_return,
            PerfStat.PA_EXCESS_RETURN: pa_excess_return,
            PerfStat.AN_LOG_RETURN: float(np.log1p(pa_return)),
            PerfStat.AN_LOG_EXCESS_RETURN: float(np.log1p(pa_excess_return)),
        }
        for perf_stat, expected_return in visible_returns.items():
            np.testing.assert_allclose(
                _stat(table, asset, perf_stat),
                expected_return,
                rtol=0.0,
                atol=_TOLERANCE,
            )
        # Arithmetic Sharpe already owns a same-support return path and must not move with this fix.
        for perf_stat in (PerfStat.SHARPE_ARITH, PerfStat.SHARPE_ARITH_EXCESS):
            np.testing.assert_allclose(
                _stat(table, asset, perf_stat),
                expected[perf_stat],
                rtol=0.0,
                atol=_TOLERANCE,
            )
        # Calmar deliberately pairs native excess return with its separately sampled drawdown.
        np.testing.assert_allclose(
            _stat(table, asset, PerfStat.CALMAR_RATIO),
            -_stat(table, asset, PerfStat.PA_EXCESS_RETURN) / _stat(table, asset, PerfStat.MAX_DD),
            rtol=0.0,
            atol=_TOLERANCE,
        )

    pd.testing.assert_frame_equal(prices, prices_before)
    pd.testing.assert_series_equal(rates, rates_before)


def test_compute_ra_perf_table_preserves_sampled_series_dataframe_parity() -> None:
    """Converge for already-sampled input and retain equivalent public pandas shapes."""
    sampled_series = _series(_sample_prices(_price_panel()), "Complete")
    sampled = sampled_series.to_frame()
    perf_params = PerfParams(freq_vol=_FREQUENCY)

    frame_table = compute_ra_perf_table(prices=sampled, perf_params=perf_params)
    series_table = compute_ra_perf_table(prices=sampled_series, perf_params=perf_params)

    pd.testing.assert_frame_equal(series_table, frame_table)
    expected = _expected_statistics(sampled_series, ReturnTypes.LOG, False)
    for perf_stat in _RATIO_STATS:
        np.testing.assert_allclose(
            _stat(frame_table, "Complete", perf_stat),
            expected[perf_stat],
            rtol=0.0,
            atol=_TOLERANCE,
        )
