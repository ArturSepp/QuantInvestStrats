#!/usr/bin/env python3
"""
Backtest an equal-weight SPY/TLT/TIP/GLD ("all-weather") portfolio on Yahoo Finance data.

Requires: pip install yfinance pandas numpy
"""

import sys

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "TLT", "TIP", "GLD"]
WEIGHTS = pd.Series(0.25, index=TICKERS)
START = "2002-07-31"
END = "2026-09-30"  # inclusive
START_NAV = 100.0
REBALANCE = "monthly"  # "none", "daily", "monthly", "quarterly", "annual"
RISK_FREE = 0.0  # annual rate used in the Sharpe ratio
PERIODS = 252  # trading days per year for annualisation


def load_prices(tickers, start, end):
    """Total-return (dividend-adjusted) closes, trimmed to the common history."""
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )  # yfinance end is exclusive
    raw = yf.download(
        tickers,
        start=start,
        end=end_excl,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        sys.exit("No data returned from Yahoo Finance.")

    px = raw["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])

    missing = [t for t in tickers if t not in px.columns or px[t].dropna().empty]
    if missing:
        sys.exit(f"No Yahoo Finance data for: {', '.join(missing)}")
    px = px[tickers]

    idx = pd.DatetimeIndex(px.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    px.index = idx.normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()

    inception = px.apply(pd.Series.first_valid_index)
    common_start = inception.max()
    if common_start > pd.Timestamp(start):
        late = ", ".join(
            f"{t} {d:%Y-%m-%d}" for t, d in inception.items() if d > pd.Timestamp(start)
        )
        print(
            f"note: first price dates after {start}: {late}; "
            f"backtest starts {common_start:%Y-%m-%d}",
            file=sys.stderr,
        )

    # forward-fill isolated gaps after all series exist; never backfill
    return px.loc[common_start:].ffill().dropna()


def rebalance_mask(index, freq):
    """True on the trading days at whose close the portfolio is reset to target weights."""
    n = len(index)
    if freq == "none":
        mask = np.zeros(n, dtype=bool)
    elif freq == "daily":
        mask = np.ones(n, dtype=bool)
    else:
        alias = {"monthly": "M", "quarterly": "Q", "annual": "Y"}[freq]
        per = index.to_period(alias)
        mask = np.r_[np.asarray(per[1:] != per[:-1]), True]  # last trading day of each period
    mask[0] = True  # initial allocation
    return mask


def run_backtest(px, weights, freq, start_value):
    p = px.to_numpy(dtype=float)
    w = weights.reindex(px.columns).to_numpy(dtype=float)
    rebal = rebalance_mask(px.index, freq)

    nav = np.empty(len(p))
    units = start_value * w / p[0]
    for i in range(len(p)):
        nav[i] = units @ p[i]
        if rebal[i]:
            units = nav[i] * w / p[i]
    return pd.Series(nav, index=px.index, name="nav")


def performance(nav, rf=0.0, periods=252):
    r = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    cagr = (1.0 + total) ** (1.0 / years) - 1.0
    sd = r.std(ddof=1)
    rf_per = (1.0 + rf) ** (1.0 / periods) - 1.0
    return {
        "terminal_nav": nav.iloc[-1],
        "total_return": total,
        "annual_return": cagr,
        "volatility": sd * np.sqrt(periods),
        "sharpe": (r - rf_per).mean() / sd * np.sqrt(periods),
        "max_drawdown": (nav / nav.cummax() - 1.0).min(),
    }


def main():
    px = load_prices(TICKERS, START, END)
    nav = run_backtest(px, WEIGHTS, REBALANCE, START_NAV)

    pd.DataFrame({"date": nav.index.strftime("%Y-%m-%d"), "nav": nav.to_numpy()}).to_csv(
        "nav.csv", index=False
    )

    m = performance(nav, RISK_FREE, PERIODS)
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
        print(f"{k}: {m[k]:.4f}")


if __name__ == "__main__":
    main()
