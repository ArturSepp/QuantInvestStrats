"""Regression coverage for terminal support in risk-adjusted performance tables.

Unrelated later panel dates must not add flat returns after an asset's final observed price.
The mixed-panel fixtures cross ordinary and nullable storage, log and simple return modes,
already-sampled and downsampled histories, funding, extrema, and benchmark regressions. Expected
statistics are reduced directly with pandas and NumPy rather than through QIS table helpers.
"""

import math
from numbers import Real
from typing import cast

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy.stats import t as student_t  # pyright: ignore[reportMissingTypeStubs]

from qis.perfstats.config import PerfParams, PerfStat, ReturnTypes
from qis.perfstats.perf_stats import (
    compute_ra_perf_table,
    compute_ra_perf_table_with_benchmark,  # pyright: ignore[reportUnknownVariableType]
)


_FREQUENCY = "ME"
_ANNUALIZATION_FACTOR = 12.0
_RETURN_DAYS_PER_YEAR = 365.25
_FUNDING_DAYS_PER_YEAR = 365.0
_NANOSECONDS_PER_DAY = 86_400_000_000_000.0
_ANNUAL_RATE = 0.04
_TERMINAL_DATE = pd.Timestamp("2022-11-30")
_TOLERANCE = 1e-12
_CONTROL_ASSETS = ("Complete", "Leading", "Interior")


def _column_name(perf_stat: PerfStat) -> str:
    """Return the full table label carried by a performance statistic."""
    name = perf_stat.value.name
    if not isinstance(name, str):
        raise TypeError(f"expected a string column label, got {type(name)!r}")
    return name


def _stat(table: pd.DataFrame, asset: str, perf_stat: PerfStat) -> float:
    """Extract one real-valued statistic from a performance table."""
    value = cast(object, table.loc[asset, _column_name(perf_stat)])
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = cast(object, value.item())
    if not isinstance(value, Real):
        raise TypeError("expected a real statistic")
    return float(value)


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Select a DataFrame column with an explicit Series boundary."""
    return frame.loc[:, column]


def _values(series: pd.Series) -> NDArray[np.float64]:
    """Convert a numeric Series to a typed floating vector."""
    return series.to_numpy(dtype=np.float64, na_value=np.nan)


def _price_path(
    steps: NDArray[np.float64],
    base: float,
    drift: float,
    amplitude: float,
    phase: float,
) -> NDArray[np.float64]:
    """Create a deterministic positive path with non-degenerate periodic returns."""
    return base * np.exp(
        drift * steps + amplitude * np.sin(steps / 13.0 + phase) + 0.004 * np.cos(steps / 23.0)
    )


def _mixed_daily_prices(storage: str = "float64") -> pd.DataFrame:
    """Create all material column-support states on one daily panel."""
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    steps = np.arange(len(dates), dtype=np.float64)
    prices = pd.DataFrame(
        {
            "Complete": _price_path(steps, 100.0, 0.00018, 0.014, 0.0),
            "Leading": _price_path(steps, 90.0, 0.00015, 0.011, 0.4),
            "Interior": _price_path(steps, 110.0, 0.00012, 0.016, 0.8),
            "Terminated": _price_path(steps, 80.0, 0.00022, 0.013, 1.2),
            "All missing": np.full(len(dates), np.nan),
        },
        index=dates,
    )
    prices.loc[prices.index < pd.Timestamp("2020-09-30"), "Leading"] = np.nan
    prices.loc[pd.Timestamp("2022-04-01") : pd.Timestamp("2022-06-30"), "Interior"] = np.nan
    prices.loc[prices.index > _TERMINAL_DATE, "Terminated"] = np.nan
    if storage == "Float64":
        return prices.astype(pd.Float64Dtype())
    return prices.astype(np.float64)


def _sample_observed_period_ends(
    prices: pd.Series,
    frequency: str = _FREQUENCY,
) -> pd.Series:
    """Sample one asset only between its independently observed endpoints."""
    observed = prices.dropna()
    if observed.empty:
        return observed.astype(float)
    observed_index = pd.DatetimeIndex(observed.index)
    sample_index = pd.date_range(observed_index[0], observed_index[-1], freq=frequency)
    supported = prices.loc[observed_index[0] : observed_index[-1]].ffill()
    return supported.reindex(sample_index, method="ffill").dropna()


def _expected_risk_statistics(
    sampled_prices: pd.Series,
    return_type: ReturnTypes,
) -> dict[PerfStat, float]:
    """Reduce one sampled history with the documented QIS statistical conventions."""
    price_values = _values(sampled_prices)
    simple_returns = price_values[1:] / price_values[:-1] - 1.0
    risk_returns: NDArray[np.float64]
    if return_type == ReturnTypes.LOG:
        risk_returns = np.asarray(np.diff(np.log(price_values)), dtype=np.float64)
    else:
        risk_returns = simple_returns

    annualized_vol = float(np.sqrt(_ANNUALIZATION_FACTOR) * np.std(risk_returns, ddof=1))
    negative_returns = risk_returns[np.less(risk_returns, 0.0)]
    downside_vol = (
        float(np.sqrt(_ANNUALIZATION_FACTOR) * np.std(negative_returns, ddof=1))
        if len(negative_returns) > 1
        else 0.0
    )
    sampled_index = pd.DatetimeIndex(sampled_prices.index)
    sampled_nanoseconds = cast(
        NDArray[np.int64],
        sampled_index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
    )
    elapsed_years = (
        float(sampled_nanoseconds[-1] - sampled_nanoseconds[0])
        / _NANOSECONDS_PER_DAY
        / _RETURN_DAYS_PER_YEAR
    )
    price_ratio = float(price_values[-1] / price_values[0])
    pa_return = math.pow(price_ratio, 1.0 / elapsed_years) - 1.0

    centered = risk_returns - np.mean(risk_returns)
    n_obs = len(risk_returns)
    second_moment = float(np.mean(np.square(centered)))
    standardized_skew = float(np.mean(np.power(centered, 3)) / second_moment**1.5)
    skewness = float(np.sqrt(n_obs * (n_obs - 1)) / (n_obs - 2) * standardized_skew)
    standardized_kurtosis = float(np.mean(np.power(centered, 4)) / second_moment**2 - 3.0)
    kurtosis = float(
        (n_obs - 1) / ((n_obs - 2) * (n_obs - 3)) * ((n_obs + 1) * standardized_kurtosis + 6.0)
    )

    drawdowns = price_values / np.maximum.accumulate(price_values) - 1.0
    maximum_drawdown = float(np.min(drawdowns))
    arithmetic_vol = float(np.std(simple_returns, ddof=1))
    return {
        PerfStat.NUM_OBS: float(n_obs),
        PerfStat.VOL: annualized_vol,
        PerfStat.DOWNSIDE_VOL: downside_vol,
        PerfStat.AVG_LOG_RETURN: float(np.mean(risk_returns)),
        PerfStat.AVG_ARITH_RETURN: float(np.mean(simple_returns)),
        PerfStat.AN_ARITH_RETURN: float(_ANNUALIZATION_FACTOR * np.mean(simple_returns)),
        PerfStat.SHARPE_RF0: pa_return / annualized_vol,
        PerfStat.SHARPE_LOG_AN: float(np.log1p(pa_return) / annualized_vol),
        PerfStat.SHARPE_ARITH: float(
            np.sqrt(_ANNUALIZATION_FACTOR) * np.mean(simple_returns) / arithmetic_vol
        ),
        PerfStat.SORTINO_RATIO: pa_return / downside_vol if downside_vol > 0.0 else np.nan,
        PerfStat.MAX_DD: maximum_drawdown,
        PerfStat.CURRENT_DD: float(drawdowns[-1]),
        PerfStat.MAX_DD_VOL: maximum_drawdown / annualized_vol,
        PerfStat.WORST: float(np.min(simple_returns)),
        PerfStat.BEST: float(np.max(simple_returns)),
        PerfStat.SKEWNESS: skewness,
        PerfStat.KURTOSIS: kurtosis,
    }


def _expected_ols(
    benchmark_returns: NDArray[np.float64],
    asset_returns: NDArray[np.float64],
) -> tuple[float, float, float, float]:
    """Calculate intercept, slope, R-squared, and intercept p-value directly."""
    benchmark_centered = benchmark_returns - np.mean(benchmark_returns)
    asset_centered = asset_returns - np.mean(asset_returns)
    benchmark_ss = float(np.dot(benchmark_centered, benchmark_centered))
    beta = float(np.dot(benchmark_centered, asset_centered) / benchmark_ss)
    alpha = float(np.mean(asset_returns) - beta * np.mean(benchmark_returns))
    residuals = asset_returns - alpha - beta * benchmark_returns
    residual_ss = float(np.dot(residuals, residuals))
    r_squared = float(1.0 - residual_ss / np.dot(asset_centered, asset_centered))
    degrees_of_freedom = len(asset_returns) - 2
    alpha_variance = (
        residual_ss
        / degrees_of_freedom
        * (1.0 / len(asset_returns) + np.mean(benchmark_returns) ** 2 / benchmark_ss)
    )
    alpha_t = alpha / np.sqrt(alpha_variance)
    alpha_pvalue = float(
        2.0
        * student_t.sf(  # pyright: ignore[reportUnknownMemberType]
            abs(alpha_t), df=degrees_of_freedom
        )
    )
    return alpha, beta, r_squared, alpha_pvalue


@pytest.mark.parametrize("return_type", [ReturnTypes.LOG, ReturnTypes.RELATIVE])
@pytest.mark.parametrize("storage", ["float64", "Float64"])
def test_compute_ra_perf_table_stops_returns_at_each_assets_terminal_support(
    storage: str,
    return_type: ReturnTypes,
) -> None:
    """Keep every terminated statistic independent of unrelated later panel dates."""
    prices = _mixed_daily_prices(storage=storage)
    prices_before = prices.copy(deep=True)
    perf_params = PerfParams(
        freq_vol=_FREQUENCY,
        freq_drawdown=_FREQUENCY,
        freq_skewness=_FREQUENCY,
        return_type=return_type,
    )

    with pytest.warns(UserWarning, match="is all nans"):
        table = compute_ra_perf_table(prices=prices, perf_params=perf_params)
    expected = _expected_risk_statistics(
        _sample_observed_period_ends(_series(prices, "Terminated")),
        return_type,
    )

    for perf_stat, expected_value in expected.items():
        np.testing.assert_allclose(
            _stat(table, "Terminated", perf_stat),
            expected_value,
            rtol=0.0,
            atol=_TOLERANCE,
        )
    assert table.loc["Terminated", _column_name(PerfStat.START_DATE)] == pd.Timestamp("2020-01-01")
    assert table.loc["Terminated", _column_name(PerfStat.END_DATE)] == _TERMINAL_DATE
    assert _stat(table, "All missing", PerfStat.NUM_OBS) == 0.0
    assert np.isnan(_stat(table, "All missing", PerfStat.VOL))

    # Complete, leading, and interior histories are controls: embedding them beside a terminated
    # asset must leave their independently selected support unchanged.
    for asset in _CONTROL_ASSETS:
        isolated_prices = cast(  # pyright: ignore[reportUnnecessaryCast]
            pd.DataFrame,
            prices[[asset]],
        )
        isolated = compute_ra_perf_table(prices=isolated_prices, perf_params=perf_params)
        for perf_stat in (
            PerfStat.NUM_OBS,
            PerfStat.VOL,
            PerfStat.AVG_LOG_RETURN,
            PerfStat.SHARPE_RF0,
            PerfStat.MAX_DD,
            PerfStat.WORST,
            PerfStat.BEST,
        ):
            np.testing.assert_allclose(
                _stat(table, asset, perf_stat),
                _stat(isolated, asset, perf_stat),
                rtol=0.0,
                atol=_TOLERANCE,
            )
    pd.testing.assert_frame_equal(prices, prices_before)


def test_compute_ra_perf_table_preserves_terminal_support_on_an_existing_grid() -> None:
    """Guard the no-resampling branch where return conversion previously filled the tail."""
    dates = pd.date_range("2020-01-31", periods=60, freq=_FREQUENCY)
    steps = np.arange(len(dates), dtype=np.float64)
    terminated = pd.Series(
        _price_path(steps, 100.0, 0.012, 0.035, 0.0),
        index=dates,
        name="Terminated",
    )
    terminated.iloc[36:] = np.nan
    neighbor = pd.Series(
        _price_path(steps, 90.0, 0.008, 0.025, 0.6),
        index=dates,
        name="Neighbor",
    )
    prices = pd.concat([terminated, neighbor], axis=1)
    perf_params = PerfParams(
        freq_vol=_FREQUENCY,
        freq_drawdown=_FREQUENCY,
        freq_skewness=_FREQUENCY,
    )

    table = compute_ra_perf_table(prices=prices, perf_params=perf_params)
    expected = _expected_risk_statistics(
        cast(  # pyright: ignore[reportUnnecessaryCast]
            pd.Series,
            terminated.iloc[:36],
        ),
        ReturnTypes.LOG,
    )

    for perf_stat in (
        PerfStat.VOL,
        PerfStat.AVG_LOG_RETURN,
        PerfStat.AVG_ARITH_RETURN,
        PerfStat.SHARPE_RF0,
        PerfStat.SHARPE_ARITH,
        PerfStat.SKEWNESS,
        PerfStat.KURTOSIS,
    ):
        np.testing.assert_allclose(
            _stat(table, "Terminated", perf_stat),
            expected[perf_stat],
            rtol=0.0,
            atol=_TOLERANCE,
        )


def test_compute_ra_perf_table_stops_distinct_risk_frequencies_at_terminal_support() -> None:
    """Exercise separate volatility, drawdown, and moment resampling paths."""
    prices = _mixed_daily_prices()
    perf_params = PerfParams(
        freq_vol=_FREQUENCY,
        freq_drawdown="QE",
        freq_skewness="QE",
    )
    monthly_expected = _expected_risk_statistics(
        _sample_observed_period_ends(_series(prices, "Terminated")),
        ReturnTypes.LOG,
    )
    quarterly_expected = _expected_risk_statistics(
        _sample_observed_period_ends(_series(prices, "Terminated"), frequency="QE"),
        ReturnTypes.LOG,
    )

    with pytest.warns(UserWarning, match="is all nans"):
        table = compute_ra_perf_table(prices=prices, perf_params=perf_params)

    np.testing.assert_allclose(
        _stat(table, "Terminated", PerfStat.VOL),
        monthly_expected[PerfStat.VOL],
        rtol=0.0,
        atol=_TOLERANCE,
    )
    for perf_stat in (
        PerfStat.MAX_DD,
        PerfStat.CURRENT_DD,
        PerfStat.WORST,
        PerfStat.BEST,
        PerfStat.SKEWNESS,
        PerfStat.KURTOSIS,
    ):
        np.testing.assert_allclose(
            _stat(table, "Terminated", perf_stat),
            quarterly_expected[perf_stat],
            rtol=0.0,
            atol=_TOLERANCE,
        )


def test_compute_ra_perf_table_funds_only_observed_terminal_intervals() -> None:
    """Exclude post-termination cash charges from arithmetic excess-return statistics."""
    prices = cast(  # pyright: ignore[reportUnnecessaryCast]
        pd.DataFrame,
        _mixed_daily_prices()[["Terminated", "Complete"]],
    )
    price_index = pd.DatetimeIndex(prices.index)
    rates = pd.Series(
        _ANNUAL_RATE,
        index=pd.date_range(price_index[0] - pd.Timedelta(days=1), price_index[-1]),
        name="Rate",
    )
    perf_params = PerfParams(
        freq_vol=_FREQUENCY,
        freq_drawdown=_FREQUENCY,
        freq_skewness=_FREQUENCY,
        rates_data=rates,
    )
    sampled = _sample_observed_period_ends(_series(prices, "Terminated"))
    sampled_values = _values(sampled)
    simple_returns = sampled_values[1:] / sampled_values[:-1] - 1.0
    sampled_index = pd.DatetimeIndex(sampled.index)
    sampled_nanoseconds = cast(
        NDArray[np.int64],
        sampled_index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
    )
    elapsed_days = np.diff(sampled_nanoseconds).astype(np.float64) / _NANOSECONDS_PER_DAY
    excess_returns = simple_returns - _ANNUAL_RATE * elapsed_days / _FUNDING_DAYS_PER_YEAR
    expected_mean = float(np.mean(excess_returns))
    expected_sharpe = float(
        np.sqrt(_ANNUALIZATION_FACTOR) * expected_mean / np.std(excess_returns, ddof=1)
    )

    table = compute_ra_perf_table(prices=prices, perf_params=perf_params)

    np.testing.assert_allclose(
        _stat(table, "Terminated", PerfStat.AVG_ARITH_EXCESS_RETURN),
        expected_mean,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    np.testing.assert_allclose(
        _stat(table, "Terminated", PerfStat.AN_ARITH_EXCESS_RETURN),
        _ANNUALIZATION_FACTOR * expected_mean,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    np.testing.assert_allclose(
        _stat(table, "Terminated", PerfStat.SHARPE_ARITH_EXCESS),
        expected_sharpe,
        rtol=0.0,
        atol=_TOLERANCE,
    )


@pytest.mark.parametrize("is_log_returns", [False, True], ids=["simple", "log"])
def test_compute_ra_perf_table_with_benchmark_uses_joint_terminal_support(
    is_log_returns: bool,
) -> None:
    """Restrict benchmark regressions to the asset and benchmark's joint observed sample."""
    prices = cast(  # pyright: ignore[reportUnnecessaryCast]
        pd.DataFrame,
        _mixed_daily_prices()[["Complete", "Terminated"]],
    ).rename(
        columns={"Complete": "Benchmark"},
    )
    asset_prices = _sample_observed_period_ends(_series(prices, "Terminated"))
    benchmark_prices = (
        _series(prices, "Benchmark")
        .ffill()
        .reindex(
            asset_prices.index,
            method="ffill",
        )
    )
    if is_log_returns:
        asset_returns = np.diff(np.log(_values(asset_prices)))
        benchmark_returns = np.diff(np.log(_values(benchmark_prices)))
    else:
        asset_values = _values(asset_prices)
        benchmark_values = _values(benchmark_prices)
        asset_returns = asset_values[1:] / asset_values[:-1] - 1.0
        benchmark_returns = benchmark_values[1:] / benchmark_values[:-1] - 1.0
    alpha, beta, r_squared, alpha_pvalue = _expected_ols(
        benchmark_returns,
        asset_returns,
    )
    perf_params = PerfParams(
        freq_vol=_FREQUENCY,
        freq_drawdown=_FREQUENCY,
        freq_skewness=_FREQUENCY,
        freq_reg=_FREQUENCY,
    )

    table = compute_ra_perf_table_with_benchmark(
        prices=prices,
        benchmark="Benchmark",
        perf_params=perf_params,
        is_log_returns=is_log_returns,
    )

    expected = {
        PerfStat.ALPHA_AN: _ANNUALIZATION_FACTOR * alpha,
        PerfStat.BETA: beta,
        PerfStat.R2: r_squared,
        PerfStat.ALPHA_PVALUE: alpha_pvalue,
    }
    for perf_stat, expected_value in expected.items():
        np.testing.assert_allclose(
            _stat(table, "Terminated", perf_stat),
            expected_value,
            rtol=0.0,
            atol=_TOLERANCE,
        )


def test_compute_ra_perf_table_with_benchmark_preserves_existing_grid_regression() -> None:
    """Keep an already-sampled benchmark regression on its established joint support."""
    dates = pd.date_range("2020-01-31", periods=60, freq=_FREQUENCY)
    steps = np.arange(len(dates), dtype=np.float64)
    benchmark = pd.Series(
        _price_path(steps, 100.0, 0.007, 0.020, 0.0),
        index=dates,
        name="Benchmark",
    )
    terminated = pd.Series(
        _price_path(steps, 90.0, 0.010, 0.035, 0.4),
        index=dates,
        name="Terminated",
    )
    terminated.iloc[36:] = np.nan
    prices = pd.concat([benchmark, terminated], axis=1)
    benchmark_values = _values(
        cast(  # pyright: ignore[reportUnnecessaryCast]
            pd.Series,
            benchmark.iloc[:36],
        )
    )
    asset_values = _values(
        cast(  # pyright: ignore[reportUnnecessaryCast]
            pd.Series,
            terminated.iloc[:36],
        )
    )
    benchmark_returns = benchmark_values[1:] / benchmark_values[:-1] - 1.0
    asset_returns = asset_values[1:] / asset_values[:-1] - 1.0
    alpha, beta, r_squared, alpha_pvalue = _expected_ols(
        benchmark_returns,
        asset_returns,
    )

    table = compute_ra_perf_table_with_benchmark(
        prices=prices,
        benchmark="Benchmark",
        perf_params=PerfParams(
            freq_vol=_FREQUENCY,
            freq_drawdown=_FREQUENCY,
            freq_skewness=_FREQUENCY,
            freq_reg=_FREQUENCY,
        ),
    )

    expected = {
        PerfStat.ALPHA_AN: _ANNUALIZATION_FACTOR * alpha,
        PerfStat.BETA: beta,
        PerfStat.R2: r_squared,
        PerfStat.ALPHA_PVALUE: alpha_pvalue,
    }
    for perf_stat, expected_value in expected.items():
        np.testing.assert_allclose(
            _stat(table, "Terminated", perf_stat),
            expected_value,
            rtol=0.0,
            atol=_TOLERANCE,
        )


@pytest.mark.parametrize(
    ("direction", "perf_stat"),
    [(1.0, PerfStat.WORST), (-1.0, PerfStat.BEST)],
    ids=["positive-worst", "negative-best"],
)
def test_compute_ra_perf_table_excludes_flat_tail_from_return_extrema(
    direction: float,
    perf_stat: PerfStat,
) -> None:
    """Keep manufactured zero returns from replacing a terminated asset's true extreme."""
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    steps = np.arange(len(dates), dtype=np.float64)
    terminated = pd.Series(
        100.0 * np.exp(direction * 0.0005 * steps),
        index=dates,
        name="Terminated",
    ).where(dates <= _TERMINAL_DATE)
    neighbor = pd.Series(100.0 * np.exp(0.0002 * steps), index=dates, name="Neighbor")
    prices = pd.concat([terminated, neighbor], axis=1)
    sampled = _sample_observed_period_ends(terminated)
    sampled_values = _values(sampled)
    sampled_returns = sampled_values[1:] / sampled_values[:-1] - 1.0
    expected = float(np.min(sampled_returns) if direction > 0.0 else np.max(sampled_returns))

    table = compute_ra_perf_table(
        prices=prices,
        perf_params=PerfParams(
            freq_vol=_FREQUENCY,
            freq_drawdown=_FREQUENCY,
            freq_skewness=_FREQUENCY,
        ),
    )

    np.testing.assert_allclose(
        _stat(table, "Terminated", perf_stat),
        expected,
        rtol=0.0,
        atol=_TOLERANCE,
    )
