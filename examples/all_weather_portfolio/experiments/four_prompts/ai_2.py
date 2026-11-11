#!/usr/bin/env python3
"""
Quarterly-rebalanced 25/25/25/25 SPY/TLT/TIP/GLD portfolio with 10 bp trading costs.

Requires: pandas, numpy, yfinance
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "TLT", "TIP", "GLD"]
TARGET = np.array([0.25, 0.25, 0.25, 0.25])
START = "2002-07-31"
END_EXCL = "2026-10-01"     # yfinance end is exclusive
LAST = "2026-09-30"
START_VALUE = 100.0
COST_RATE = 0.0010          # 10 bp of absolute traded notional


def load_prices():
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

    # Business-day grid; ffill repeats prices on holidays but leaves NaN before each first quote.
    grid = pd.bdate_range(START, LAST)
    return px.reindex(grid).ffill()


def rebalance_mask(grid):
    """True on the start date and on the first business day on or after each quarter-end."""
    qends = pd.period_range(START, LAST, freq="Q").to_timestamp(how="end").normalize()
    qends = qends[(qends >= grid[0]) & (qends <= grid[-1])]
    mask = np.zeros(len(grid), dtype=bool)
    mask[0] = True
    pos = grid.searchsorted(qends, side="left")
    mask[pos[pos < len(grid)]] = True
    return mask


def run_backtest(px):
    p = px.to_numpy(dtype=float)
    rebal = rebalance_mask(px.index)

    units = np.zeros(p.shape[1])
    cash = START_VALUE
    nav = np.empty(len(p))

    for i in range(len(p)):
        has = ~np.isnan(p[i])
        price = np.where(has, p[i], 0.0)
        holdings = units * price
        value = cash + holdings.sum()                      # value before trading costs

        if rebal[i]:
            w = np.where(has, TARGET, 0.0)
            if w.sum() <= 0:
                sys.exit(f"No prices available on rebalancing date {px.index[i]:%Y-%m-%d}.")
            w = w / w.sum()
            new_holdings = w * value
            cost = COST_RATE * np.abs(new_holdings - holdings).sum()
            units = np.divide(new_holdings, price, out=np.zeros_like(price), where=has)
            cash = value - new_holdings.sum() - cost       # = -cost; earns no interest
            value -= cost

        nav[i] = value

    return pd.Series(nav, index=px.index, name="nav")


def performance(nav):
    first, last = nav.iloc[0], nav.iloc[-1]
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    total = last / first - 1.0
    annual = (last / first) ** (1.0 / years) - 1.0

    month_end = nav.groupby(nav.index.to_period("M")).last()
    log_ret = np.log(month_end).diff().dropna()
    vol = log_ret.std(ddof=1) * np.sqrt(12)

    return {
        "terminal_nav": last,
        "total_return": total,
        "annual_return": annual,
        "volatility": vol,
        "sharpe": annual / vol,
        "max_drawdown": (nav / nav.cummax() - 1.0).min(),
    }


def main():
    px = load_prices()
    nav = run_backtest(px)

    pd.DataFrame({"date": nav.index.strftime("%Y-%m-%d"), "nav": nav.to_numpy()}) \
      .to_csv("nav.csv", index=False)

    m = performance(nav)
    print(f"first_date: {nav.index[0]:%Y-%m-%d}")
    print(f"last_date: {nav.index[-1]:%Y-%m-%d}")
    for k in ("terminal_nav", "total_return", "annual_return",
              "volatility", "sharpe", "max_drawdown"):
        print(f"{k}: {m[k]:.4f}")


if __name__ == "__main__":
    main()
