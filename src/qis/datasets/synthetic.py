"""
seeded synthetic market data for tests, examples and documentation

The panel is generated in-process from a fixed seed: no network, no data files, no vendor
licence. It carries the data defects that real multi-asset panels carry — ragged starts,
missing observations, stale prices, a delisted tail, fat tails, autocorrelated (smoothed)
returns, and a monthly-reported illiquid sleeve — so that code exercised against it meets the
same edge cases it meets in production.

FROZEN. Golden tests pin their expected output to this generator. Do not change the instrument
specs, the factor structure, the quirk parameters or the draw order once a baseline exists:
doing so silently moves every golden. To generate different data, add a new function.

The generator imports numpy, pandas and scipy only, never qis. That is deliberate: data used to
pin golden output must not move when the library under test moves.

`SyntheticInstrument.vol` is the volatility of the *economic* path. Instruments carrying
DataQuirk.SMOOTHED or DataQuirk.MONTHLY_REPORTED report a series whose realised volatility is
lower by sqrt((1 - phi) / (1 + phi)) — the appraisal-smoothing bias the qis unsmoothing models
are built to invert, reproduced here with a known true phi.

Design:
    log-returns are driven by a one-market-factor / one-group-factor structure

        z_i = beta_i * f_mkt + gamma_i * f_group(i) + sqrt(1 - beta_i^2 - gamma_i^2) * eps_i

    with f_mkt, f_group standard normal and eps_i standard normal or standardised Student-t.
    The construction is positive semi-definite by design, so no correlation matrix is repaired.
    Prices follow r_i = (mu_i - 0.5 sigma_i^2) dt + sigma_i sqrt(dt) z_i, P_i = 100 exp(sum r_i).
"""

# packages
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from scipy.signal import lfilter
from typing import Dict, List, Tuple

TRADING_DAYS_PER_YEAR = 260  # business-day grid, consistent with qis annualisation on 'B'
INITIAL_PRICE = 100.0


class DataQuirk(str, Enum):
    """Data defect applied to a synthetic instrument after its clean path is drawn."""
    CLEAN = 'clean'  # no defect: a liquid, fully reported instrument
    GAPS = 'gaps'  # scattered missing observations, as for a non-US listing on local holidays
    LATE_START = 'late_start'  # instrument launches after the panel starts: leading nans
    FAT_TAILS = 'fat_tails'  # Student-t idiosyncratic innovations
    STALE = 'stale'  # runs of repeated prices, as for an infrequently traded instrument
    DELISTED = 'delisted'  # trailing nans: instrument stops reporting before the panel ends
    SMOOTHED = 'smoothed'  # ar(1) smoothed returns, as for an appraisal-based nav
    MONTHLY_REPORTED = 'monthly_reported'  # ar(1) smoothed and reported at month ends only


@dataclass(frozen=True)
class SyntheticInstrument:
    """Specification of one synthetic instrument. Frozen: see the module docstring."""
    ticker: str
    group: str
    vol: float  # annualised volatility of the clean path
    drift: float  # annualised expected log return of the clean path
    beta_market: float  # loading on the single market factor
    beta_group: float  # loading on the group factor
    quirk: DataQuirk = DataQuirk.CLEAN

    def __post_init__(self) -> None:
        systematic_var = self.beta_market ** 2 + self.beta_group ** 2
        if not systematic_var < 1.0:
            raise ValueError(f"systematic variance must be below 1.0 for {self.ticker}, got "
                             f"beta_market={self.beta_market!r}, "
                             f"beta_group={self.beta_group!r}")
        if not self.vol > 0.0:
            raise ValueError(f"vol must be positive for {self.ticker}, got {self.vol!r}")


# the universe. Ticker prefix S marks the data as synthetic so it is never confused with a real
# instrument in a report. Frozen: see the module docstring.
SYNTHETIC_UNIVERSE: Tuple[SyntheticInstrument, ...] = (
    SyntheticInstrument(ticker='SEQ_US', group='Equities', vol=0.17, drift=0.080,
                        beta_market=0.90, beta_group=0.20, quirk=DataQuirk.CLEAN),
    SyntheticInstrument(ticker='SEQ_EU', group='Equities', vol=0.19, drift=0.055,
                        beta_market=0.78, beta_group=0.42, quirk=DataQuirk.GAPS),
    SyntheticInstrument(ticker='SEQ_EM', group='Equities', vol=0.22, drift=0.050,
                        beta_market=0.70, beta_group=0.45, quirk=DataQuirk.LATE_START),
    SyntheticInstrument(ticker='SBD_TSY', group='Bonds', vol=0.06, drift=0.030,
                        beta_market=-0.25, beta_group=0.80, quirk=DataQuirk.CLEAN),
    SyntheticInstrument(ticker='SBD_IG', group='Bonds', vol=0.07, drift=0.035,
                        beta_market=0.15, beta_group=0.82, quirk=DataQuirk.CLEAN),
    SyntheticInstrument(ticker='SBD_HY', group='Bonds', vol=0.10, drift=0.050,
                        beta_market=0.55, beta_group=0.55, quirk=DataQuirk.FAT_TAILS),
    SyntheticInstrument(ticker='SCM_GLD', group='Commodities', vol=0.15, drift=0.040,
                        beta_market=0.10, beta_group=0.55, quirk=DataQuirk.STALE),
    SyntheticInstrument(ticker='SCM_BCOM', group='Commodities', vol=0.14, drift=0.015,
                        beta_market=0.35, beta_group=0.60, quirk=DataQuirk.DELISTED),
    SyntheticInstrument(ticker='SAL_HF', group='Alternatives', vol=0.08, drift=0.050,
                        beta_market=0.60, beta_group=0.35, quirk=DataQuirk.SMOOTHED),
    SyntheticInstrument(ticker='SAL_PE', group='Alternatives', vol=0.11, drift=0.090,
                        beta_market=0.65, beta_group=0.40, quirk=DataQuirk.MONTHLY_REPORTED),
)

GROUP_ORDER: List[str] = ['Equities', 'Bonds', 'Commodities', 'Alternatives']

BENCHMARK_TICKER = 'SBM_6040'
BENCHMARK_WEIGHTS: Dict[str, float] = {'SEQ_US': 0.6, 'SBD_TSY': 0.4}

# quirk parameters. Frozen: see the module docstring.
GAP_PROBABILITY = 0.015  # fraction of observations dropped under DataQuirk.GAPS
LATE_START_FRACTION = 0.25  # fraction of the panel elapsed before a LATE_START instrument lists
DELISTED_FRACTION = 0.90  # fraction of the panel elapsed when a DELISTED instrument stops
STALE_RUN_PROBABILITY = 0.12  # probability that a STALE instrument repeats the previous price
FAT_TAIL_DF = 4.0  # Student-t degrees of freedom under DataQuirk.FAT_TAILS
SMOOTHING_AR1 = 0.35  # ar(1) coefficient applied under SMOOTHED and MONTHLY_REPORTED


@dataclass(frozen=True)
class SyntheticUniverseData:
    """
    Synthetic multi-asset panel and the metadata qis reporting expects alongside it.

    Attributes:
        prices: business-day price panel, columns = tickers, with nans where the instrument
            does not report
        benchmark_prices: business-day price panel for the synthetic 60/40 benchmark, no nans
        group_data: asset-class label per ticker, indexed by ticker
        group_order: asset-class display order
    """
    prices: pd.DataFrame
    benchmark_prices: pd.DataFrame
    group_data: pd.Series
    group_order: List[str]

    def __post_init__(self) -> None:
        if not self.prices.columns.equals(self.group_data.index):
            raise ValueError(f"group_data index must match prices columns, got "
                             f"{list(self.group_data.index)!r} vs "
                             f"{list(self.prices.columns)!r}")
        if not self.prices.index.equals(self.benchmark_prices.index):
            raise ValueError("benchmark_prices must share the prices index")


def generate_synthetic_prices(instruments: Tuple[SyntheticInstrument, ...] = SYNTHETIC_UNIVERSE,
                              start: str = '2005-01-03',
                              end: str = '2025-12-31',
                              seed: int = 20260725,
                              apply_quirks: bool = True,  # False gives clean paths, same draws
                              ) -> pd.DataFrame:
    """
    Draw a business-day synthetic price panel from a fixed seed.

    Args:
        instruments: instrument specifications; defaults to the frozen SYNTHETIC_UNIVERSE
        start: first business day of the panel, inclusive
        end: last business day of the panel, inclusive
        seed: numpy Generator seed; the same seed reproduces the panel exactly
        apply_quirks: when False the clean paths are returned with no nans and no staleness,
            which is what a numerical test that must not handle missing data should use

    Returns:
        price panel indexed by business day with columns in the order of ``instruments``

    Raises:
        ValueError: if ``instruments`` is empty or the date range contains no business day
    """
    if len(instruments) == 0:
        raise ValueError(f"instruments must not be empty, got {instruments!r}")
    dates = pd.bdate_range(start=start, end=end, name='date')
    if len(dates) == 0:
        raise ValueError(f"no business day in the requested range, "
                         f"got start={start!r}, end={end!r}")

    rng = np.random.default_rng(seed)
    num_dates, num_instruments = len(dates), len(instruments)
    groups = [instrument.group for instrument in instruments]
    unique_groups = list(dict.fromkeys(groups))

    # factor draws: one market factor, one factor per group, then idiosyncratic innovations
    market_factor = rng.standard_normal(num_dates)
    group_factors = rng.standard_normal((num_dates, len(unique_groups)))
    normal_idiosyncratic = rng.standard_normal((num_dates, num_instruments))
    # standardised Student-t: unit variance, so vol targeting is unaffected by the tail choice
    t_idiosyncratic = rng.standard_t(df=FAT_TAIL_DF, size=(num_dates, num_instruments))
    t_idiosyncratic = t_idiosyncratic / np.sqrt(FAT_TAIL_DF / (FAT_TAIL_DF - 2.0))

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    prices = {}
    for idx, instrument in enumerate(instruments):
        group_idx = unique_groups.index(instrument.group)
        idiosyncratic_var = 1.0 - instrument.beta_market ** 2 - instrument.beta_group ** 2
        is_fat_tailed = apply_quirks and instrument.quirk == DataQuirk.FAT_TAILS
        idiosyncratic = t_idiosyncratic[:, idx] if is_fat_tailed else normal_idiosyncratic[:, idx]
        z = (instrument.beta_market * market_factor
             + instrument.beta_group * group_factors[:, group_idx]
             + np.sqrt(idiosyncratic_var) * idiosyncratic)

        log_returns = ((instrument.drift - 0.5 * instrument.vol ** 2) * dt
                       + instrument.vol * np.sqrt(dt) * z)
        if apply_quirks and instrument.quirk in (DataQuirk.SMOOTHED, DataQuirk.MONTHLY_REPORTED):
            log_returns = _apply_ar1_smoothing(log_returns=log_returns, ar1=SMOOTHING_AR1)
        prices[instrument.ticker] = INITIAL_PRICE * np.exp(np.cumsum(log_returns))

    price_df = pd.DataFrame(data=prices, index=dates)
    if apply_quirks:
        price_df = _apply_reporting_quirks(prices=price_df, instruments=instruments, rng=rng)
    return price_df


def generate_synthetic_universe(start: str = '2005-01-03',
                                end: str = '2025-12-31',
                                seed: int = 20260725,
                                apply_quirks: bool = True,
                                ) -> SyntheticUniverseData:
    """
    Draw the synthetic panel together with the benchmark and group metadata.

    The benchmark is a fixed-weight 60/40 blend of the two clean instruments, rebalanced daily,
    so it is free of nans and can serve as a regression and regime benchmark.

    Args:
        start: first business day of the panel, inclusive
        end: last business day of the panel, inclusive
        seed: numpy Generator seed
        apply_quirks: when False the clean paths are returned with no nans and no staleness

    Returns:
        SyntheticUniverseData holding prices, benchmark_prices, group_data and group_order
    """
    prices = generate_synthetic_prices(start=start, end=end, seed=seed, apply_quirks=apply_quirks)
    benchmark_prices = _compute_benchmark_prices(prices=prices, weights=BENCHMARK_WEIGHTS)
    group_data = pd.Series(data=[instrument.group for instrument in SYNTHETIC_UNIVERSE],
                           index=[instrument.ticker for instrument in SYNTHETIC_UNIVERSE],
                           name='group')
    return SyntheticUniverseData(prices=prices,
                                 benchmark_prices=benchmark_prices,
                                 group_data=group_data,
                                 group_order=GROUP_ORDER)


def _apply_ar1_smoothing(log_returns: np.ndarray,
                         ar1: float,
                         ) -> np.ndarray:
    """
    Apply the appraisal-smoothing recursion r~_t = (1 - phi) r_t + phi r~_{t-1}.

    This is the data-generating process the qis unsmoothing models invert, so a panel carrying
    it exercises `adjust_returns_with_ar` and friends on data with a known true phi.

    Args:
        log_returns: unsmoothed log return path
        ar1: smoothing coefficient phi, in [0, 1)

    Returns:
        smoothed log return path of the same length

    Raises:
        ValueError: if ``ar1`` is outside [0, 1)
    """
    if not 0.0 <= ar1 < 1.0:
        raise ValueError(f"ar1 must lie in [0, 1), got {ar1!r}")
    return lfilter([1.0 - ar1], [1.0, -ar1], log_returns)


def _apply_reporting_quirks(prices: pd.DataFrame,
                            instruments: Tuple[SyntheticInstrument, ...],
                            rng: np.random.Generator,
                            ) -> pd.DataFrame:
    """
    Overwrite the clean paths with the reporting defects declared on each instrument.

    Args:
        prices: clean business-day price panel
        instruments: instrument specifications, aligned to the columns of ``prices``
        rng: the generator already used for the paths, so the draws stay reproducible

    Returns:
        price panel of the same shape with nans and repeated prices inserted
    """
    prices = prices.copy()
    num_dates = len(prices.index)
    # a date is a reporting month end when the following date falls in a different month
    monthly_periods = prices.index.to_period('M')
    is_month_end = np.append(monthly_periods[1:] != monthly_periods[:-1], True)
    month_end_dates = prices.index[is_month_end]

    for instrument in instruments:
        ticker = instrument.ticker
        if instrument.quirk == DataQuirk.GAPS:
            is_missing = rng.random(num_dates) < GAP_PROBABILITY
            prices.loc[is_missing, ticker] = np.nan

        elif instrument.quirk == DataQuirk.LATE_START:
            first_index = int(LATE_START_FRACTION * num_dates)
            prices.iloc[:first_index, prices.columns.get_loc(ticker)] = np.nan

        elif instrument.quirk == DataQuirk.DELISTED:
            last_index = int(DELISTED_FRACTION * num_dates)
            prices.iloc[last_index:, prices.columns.get_loc(ticker)] = np.nan

        elif instrument.quirk == DataQuirk.STALE:
            is_repeated = rng.random(num_dates) < STALE_RUN_PROBABILITY
            is_repeated[0] = False
            stale_prices = prices[ticker].where(~is_repeated, other=np.nan)
            prices[ticker] = stale_prices.ffill()

        elif instrument.quirk == DataQuirk.MONTHLY_REPORTED:
            prices[ticker] = prices[ticker].where(prices.index.isin(month_end_dates), other=np.nan)

    return prices


def _compute_benchmark_prices(prices: pd.DataFrame,
                              weights: Dict[str, float],
                              ) -> pd.DataFrame:
    """
    Build a daily-rebalanced fixed-weight benchmark from a subset of the panel.

    Args:
        prices: price panel containing every ticker in ``weights``
        weights: ticker to weight; weights must sum to one

    Returns:
        single-column price panel named by ``BENCHMARK_TICKER``

    Raises:
        ValueError: if a ticker is missing from ``prices`` or the weights do not sum to one
    """
    missing = [ticker for ticker in weights if ticker not in prices.columns]
    if len(missing) > 0:
        raise ValueError(f"benchmark tickers missing from prices, got {missing!r}")
    weight_sum = float(np.sum(list(weights.values())))
    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"benchmark weights must sum to one, got {weight_sum!r}")

    components = prices[list(weights.keys())]
    if components.isna().to_numpy().any():
        raise ValueError(f"benchmark components must be free of nans, got nans in "
                         f"{components.columns[components.isna().any()].tolist()!r}")
    # daily rebalanced, so the benchmark return is the weighted sum of simple returns. The
    # convention is arithmetic here and stated rather than implied; components carry no nans,
    # so fill_method is disabled explicitly rather than relying on the pandas default.
    arithmetic_returns = components.pct_change(fill_method=None).fillna(0.0)
    benchmark_returns = arithmetic_returns.mul(pd.Series(weights), axis=1).sum(axis=1)
    benchmark_nav = INITIAL_PRICE * (1.0 + benchmark_returns).cumprod()
    return benchmark_nav.to_frame(name=BENCHMARK_TICKER)


class LocalTest(Enum):
    """Enumeration of available local test cases."""
    UNIVERSE_SUMMARY = 1
    QUIRK_DIAGNOSTICS = 2


def run_local_test(local_test: LocalTest) -> None:
    """
    Run local tests for development and debugging purposes.

    Args:
        local_test: which test case to run
    """
    if local_test == LocalTest.UNIVERSE_SUMMARY:
        universe = generate_synthetic_universe()
        print(universe.prices.tail())
        print(universe.benchmark_prices.tail())
        print(universe.group_data)

    elif local_test == LocalTest.QUIRK_DIAGNOSTICS:
        prices = generate_synthetic_prices(apply_quirks=True)
        clean_prices = generate_synthetic_prices(apply_quirks=False)
        # both volatilities are annualised on the business-day grid, so the reported figure is
        # comparable to the clean figure only for daily-reporting instruments
        quirks = [instrument.quirk.value for instrument in SYNTHETIC_UNIVERSE]
        report = pd.DataFrame({'quirk': quirks,
                               'target_vol': [x.vol for x in SYNTHETIC_UNIVERSE],
                               'clean_vol': (np.log(clean_prices).diff().std()
                                             * np.sqrt(TRADING_DAYS_PER_YEAR)),
                               'num_reported': prices.count(),
                               'num_nans': prices.isna().sum(),
                               'first_valid': prices.apply(lambda x: x.first_valid_index()),
                               'last_valid': prices.apply(lambda x: x.last_valid_index())})
        print(report.to_string())


if __name__ == '__main__':

    run_local_test(local_test=LocalTest.QUIRK_DIAGNOSTICS)
