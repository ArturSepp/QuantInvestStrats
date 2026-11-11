#!/usr/bin/env python3
"""
Quarterly-rebalanced 25/25/25/25 SPY/TLT/TIP/GLD portfolio, backtested and reported with qis.

Requires: pip install -U "qis[data]"
"""

import contextlib
import sys

import matplotlib

matplotlib.use("Agg")  # headless; must precede the qis import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import qis

TICKERS = ["SPY", "TLT", "TIP", "GLD"]
WEIGHTS = {t: 0.25 for t in TICKERS}
START = "2002-07-31"
END_EXCL = "2026-10-01"  # yfinance end is exclusive
LAST = "2026-09-30"
INITIAL_NAV = 100.0
COST_RATE = 0.0010  # 10 bp of absolute traded notional
BENCHMARK = "SPY"
STRATEGY = "All-Weather 25x4"
FACTSHEET_FILE = "all_weather_factsheet"  # written as all_weather_factsheet.pdf

# Monthly log-return vol (ddof=1, x sqrt(12)), daily drawdown,
# compound p.a. return on 365.25-day years.
PERF_PARAMS = qis.PerfParams(freq_vol="ME", freq_drawdown="D", return_type=qis.ReturnTypes.LOG)


def load_prices() -> pd.DataFrame:
    raw = yf.download(TICKERS, start=START, end=END_EXCL, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        sys.exit("No data returned from Yahoo Finance.")
    px = raw["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(TICKERS[0])
    missing = [t for t in TICKERS if t not in px.columns or px[t].dropna().empty]
    if missing:
        sys.exit(f"No Yahoo Finance data for: {', '.join(missing)}")
    px = px[TICKERS]

    idx = pd.DatetimeIndex(px.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    px.index = idx.normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()

    if px.index.max() != pd.Timestamp(LAST):
        sys.exit(f"Last Yahoo observation is {px.index.max():%Y-%m-%d}, expected {LAST}.")

    # Business-day grid; ffill repeats prices on holidays and leaves NaN before each first quote.
    grid = pd.bdate_range(START, LAST)
    return px.reindex(grid).ffill()


def expected_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Spec: the start date plus the first business day on or after each calendar quarter-end."""
    qends = pd.period_range(START, LAST, freq="Q").to_timestamp(how="end").normalize()
    qends = qends[(qends >= index[0]) & (qends <= index[-1])]
    pos = index.searchsorted(qends, side="left")
    return index[[0]].append(index[pos]).unique()


def check(condition: bool, message: str) -> None:
    if not condition:
        sys.exit(f"Specification check failed: {message}")


def run():
    prices = load_prices()

    # Static targets over the live universe: an ETF without a price gets zero weight and the
    # rest are scaled pro rata; an ETF enters at the first rebalancing on or after its first quote.
    schedule = qis.generate_static_weights_schedule(
        prices=prices,
        weights=WEIGHTS,
        rebalancing_freq="QE",
        is_rescale_to_live_universe=True,
        include_start_date=True,
    )
    expected = expected_rebalance_dates(prices.index)
    check(
        schedule.index.equals(expected),
        "qis rebalancing dates differ from 'first business day on or after quarter-end'; "
        f"only in qis: {list(schedule.index.difference(expected)[:5])}, "
        f"missing in qis: {list(expected.difference(schedule.index)[:5])}",
    )
    check(
        np.allclose(schedule.sum(axis=1), 1.0),
        "target weights do not sum to 100% on every rebalancing date",
    )

    # Units held between rebalancings; positions sized on pre-cost value; costs debited to cash,
    # including the opening trade; no funding rate, so the cash balance earns nothing.
    portfolio = qis.backtest_model_portfolio(
        prices=prices,
        weights=schedule,
        rebalancing_freq=None,  # the schedule supplies the dates
        initial_nav=INITIAL_NAV,
        rebalancing_costs=COST_RATE,
        weight_implementation_lag=0,  # trade at the close of the rebalancing date
        ticker=STRATEGY,
    )
    nav = portfolio.get_portfolio_nav()
    check(
        nav.index[0] == pd.Timestamp(START),
        f"NAV starts on {nav.index[0]:%Y-%m-%d}, expected {START}",
    )
    check(
        np.isclose(nav.iloc[0], INITIAL_NAV * (1.0 - COST_RATE)),
        f"first NAV is {nav.iloc[0]:.6f}, expected the post-cost value "
        f"{INITIAL_NAV * (1.0 - COST_RATE):.6f}",
    )

    # Statistics from the qis risk-adjusted performance table,
    # measured from the post-cost first value.
    table = qis.compute_ra_perf_table(prices=nav, perf_params=PERF_PARAMS)
    row = table.iloc[0]

    def stat(s: qis.PerfStat) -> float:
        return float(row[s.to_str()])

    stats = {
        "terminal_nav": float(nav.iloc[-1]),
        "total_return": stat(qis.PerfStat.TOTAL_RETURN),
        "annual_return": stat(qis.PerfStat.PA_RETURN),
        "volatility": stat(qis.PerfStat.VOL),
        "sharpe": stat(qis.PerfStat.SHARPE_RF0),
        "max_drawdown": stat(qis.PerfStat.MAX_DD),
    }

    pd.DataFrame({"date": nav.index.strftime("%Y-%m-%d"), "nav": nav.to_numpy()}).to_csv(
        "nav.csv", index=False
    )

    return prices, portfolio, nav, stats


def write_factsheet(prices: pd.DataFrame, portfolio) -> str:
    figs = qis.generate_strategy_factsheet(
        portfolio_data=portfolio,
        benchmark_prices=prices[[BENCHMARK]],
        perf_params=PERF_PARAMS,  # same conventions as the printed statistics
        add_benchmarks_to_navs=True,
        factsheet_name=f"{STRATEGY} vs {BENCHMARK}",
    )
    path = qis.save_figs_to_pdf(
        figs, file_name=FACTSHEET_FILE, orientation="portrait", add_current_date=False
    )
    plt.close("all")
    return path


def main():
    # Keep stdout to the eight result lines; anything qis prints goes to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        prices, portfolio, nav, stats = run()

    print(f"first_date: {nav.index[0]:%Y-%m-%d}")
    print(f"last_date: {nav.index[-1]:%Y-%m-%d}")
    for k in (
        "terminal_nav",
        "total_return",
        "annual_return",
        "volatility",
        "sharpe",
        "max_drawdown",
    ):
        print(f"{k}: {stats[k]:.4f}")
    sys.stdout.flush()

    with contextlib.redirect_stdout(sys.stderr):
        path = write_factsheet(prices, portfolio)
        print(f"factsheet written to {path}")


if __name__ == "__main__":
    main()
