#!/usr/bin/env python3
"""All-weather (25/25/25/25) vs US 60/40, 2002-07-31--2026-09-30.

Run: python -m pip install -U numpy pandas requests yfinance
     python all_weather_vs_6040.py

Data: Yahoo Finance via yfinance (dividend/split-adjusted close) for SPY,
TLT, TIP, GLD, IEF, VIPSX; goldprice.com daily USD bullion closes for the
pre-GLD interval; FRED DGS3MO for the 3-month Treasury risk-free yield.
The pre-ETF periods are proxy simulations, not actual TIP/GLD ETF holdings.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import requests
import yfinance as yf

START = pd.Timestamp("2002-07-31")
END = pd.Timestamp("2026-09-30")
DOWNLOAD_END = "2026-10-01"  # Yahoo's end is exclusive.
FEE = 0.001  # 10 basis points of *total buys + sells*.
GOLD_PROXY_ANNUAL_EXPENSE = 0.004  # Mimic GLD's 0.40% yearly expense.
TIPS_SWITCH_CLOSE = pd.Timestamp("2003-12-31")
GOLD_SWITCH_CLOSE = pd.Timestamp("2004-12-31")
SYMBOLS = ("SPY", "TLT", "TIP", "GLD", "IEF", "VIPSX")
AW = {"SPY": .25, "TLT": .25, "TIPS": .25, "GOLD": .25}
SIXTY = {"SPY": .60, "IEF": .40}


def fetch_etfs() -> pd.DataFrame:
    out = {}
    for symbol in SYMBOLS:
        print(f"Downloading adjusted prices: {symbol}", flush=True)
        try:
            df = yf.Ticker(symbol).history(
                start="2002-07-20", end=DOWNLOAD_END,
                interval="1d", auto_adjust=True, actions=False
            )
        except Exception as exc:
            raise RuntimeError(f"Could not download {symbol}: {exc}") from exc
        if df.empty or "Close" not in df:
            raise RuntimeError(f"No adjusted close data returned for {symbol}")
        s = df["Close"].copy().astype(float)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index = s.index.normalize()
        s = s[~s.index.duplicated(keep="last")]
        out[symbol] = s.rename(symbol)
    return pd.concat(out.values(), axis=1).sort_index()


def fetch_csv_series(url: str, preferred: str) -> pd.Series:
    try:
        # Reproduction edit: FRED served DGS3MO with the default Requests user agent.
        headers = None if preferred == "DGS3MO" else {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, timeout=50, headers=headers)
        res.raise_for_status()
        df = pd.read_csv(io.StringIO(res.text))
    except Exception as exc:
        raise RuntimeError(f"Could not read publicly available CSV {url}: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        raise ValueError(f"No date column in {url}. Found: {list(df.columns)}")
    date_col = date_cols[0]
    value_cols = [c for c in df.columns if c.lower() == preferred.lower()]
    if not value_cols:
        value_cols = [c for c in df.columns if c != date_col]
    if len(value_cols) != 1:
        raise ValueError(f"Ambiguous numeric column in {url}: {list(df.columns)}")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_cols[0]].replace(".", np.nan), errors="coerce")
    series = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(dates),
                       name=preferred)
    series = series[series.index.notna()].sort_index()
    return series[~series.index.duplicated(keep="last")]


def prepare_returns(prices: pd.DataFrame, gold: pd.Series) -> pd.DataFrame:
    # SPY defines US ETF trading sessions. A missing required day is an error,
    # rather than silently dropping that day from both portfolios.
    dates = prices.loc[START:END, "SPY"].dropna().index
    if len(dates) < 5900 or dates.min() != START or dates.max() != END:
        raise ValueError(f"Incomplete SPY backtest dates: {dates.min()} .. {dates.max()}")
    p = prices.reindex(dates).copy()
    for symbol in SYMBOLS:
        first = p[symbol].first_valid_index()
        if first is None:
            raise ValueError(f"Entire ETF/fund history missing: {symbol}")
        p.loc[first:, symbol] = p.loc[first:, symbol].ffill(limit=2)
        if p.loc[first:, symbol].isna().any():
            bad = p.loc[first:, symbol][p.loc[first:, symbol].isna()].index[0]
            raise ValueError(f"Unexpected missing history for {symbol} on {bad.date()}")
    for symbol in ("SPY", "TLT", "IEF", "VIPSX"):
        if pd.isna(p.at[START, symbol]):
            raise ValueError(f"No {symbol} adjusted price on the initial date")
    if p.loc[:TIPS_SWITCH_CLOSE, "VIPSX"].isna().any():
        raise ValueError("Incomplete VIPSX pre-TIP history")
    if p.loc[TIPS_SWITCH_CLOSE:, "TIP"].isna().any():
        raise ValueError("TIP does not cover its 2003-12-31 switch date")
    if p.loc[GOLD_SWITCH_CLOSE:, "GLD"].isna().any():
        raise ValueError("GLD does not cover its 2004-12-31 switch date")

    spot = gold.reindex(dates, method="ffill")
    if spot.loc[:GOLD_SWITCH_CLOSE].isna().any():
        raise ValueError("Historical gold prices missing before the GLD switch")
    # Prevent quiet carrying over an excessively old gold observation.
    spot_before = gold.loc[:GOLD_SWITCH_CLOSE]
    if spot_before.empty or spot_before.index.max() < GOLD_SWITCH_CLOSE - pd.Timedelta(days=5):
        raise ValueError("Gold-price history ends before GLD switch")

    r = pd.DataFrame(index=dates, columns=["SPY", "TLT", "TIPS", "GOLD", "IEF"],
                     dtype=float)
    for symbol in ("SPY", "TLT", "IEF"):
        r[symbol] = p[symbol].pct_change(fill_method=None)
    tip_actual = p["TIP"].pct_change(fill_method=None)
    tip_proxy = p["VIPSX"].pct_change(fill_method=None)
    r["TIPS"] = tip_proxy.where(dates <= TIPS_SWITCH_CLOSE, tip_actual)
    gold_actual = p["GLD"].pct_change(fill_method=None)
    spot_returns = spot.pct_change(fill_method=None)
    # Economic proxy for gold before GLD can be purchased at quarter-end.
    # This is a cost/fee assumption, NOT a real ETF return series.
    gold_proxy = (1 + spot_returns) * (1 - GOLD_PROXY_ANNUAL_EXPENSE / 252) - 1
    r["GOLD"] = gold_proxy.where(dates <= GOLD_SWITCH_CLOSE, gold_actual)
    r.iloc[0] = 0.0  # July 31: first purchase at that day's close.
    if r.isna().any().any():
        bad = r.isna().stack()
        raise ValueError(f"Missing asset returns: {list(bad[bad].index[:8])}")
    return r


def rate_data(index: pd.DatetimeIndex) -> pd.Series:
    # FRED: annualized 3-month *investment-basis* government yield, in %.
    fred = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
    yields = fetch_csv_series(fred, "DGS3MO")
    yields = yields.reindex(index).ffill()
    if yields.isna().any():
        raise ValueError("No 3-month Treasury yield for some trading days")
    return yields / 100 / 252  # approximate per-session risk-free return


@dataclass
class Backtest:
    nav: pd.Series
    daily_returns: pd.Series
    gains: pd.DataFrame  # daily dollar P/L by asset, initial NAV exactly $1
    fees: pd.Series     # daily dollars charged for trading
    traded: pd.Series   # daily total absolute value traded


def rebalance(holdings: pd.Series, weights: pd.Series,
              switching: set[str]) -> tuple[pd.Series, float, float]:
    """Exact cost/target-weight solution, allowing round-trip proxy switches.

    For sleeve changing instrument (VIPSX -> TIP or bullion -> GLD), the full
    old sleeve is sold and the full new sleeve is bought. Otherwise only the
    difference between old and desired post-fee holdings is bought/sold.
    """
    value = float(holdings.sum())
    cost = 0.0
    for _ in range(50):
        target = weights * (value - cost)
        diffs = (target - holdings).abs()
        for sleeve in switching:
            diffs[sleeve] = holdings[sleeve] + target[sleeve]
        notional = float(diffs.sum())
        next_cost = FEE * notional
        if abs(next_cost - cost) < 1e-14:
            cost = next_cost
            break
        cost = next_cost
    final = weights * (value - cost)
    if not np.isclose(float(final.sum()) + cost, value, atol=1e-12):
        raise AssertionError("Rebalancing accounting failure")
    return final, cost, notional


def simulate(r: pd.DataFrame, target_weights: Dict[str, float],
             all_weather: bool) -> Backtest:
    weights = pd.Series(target_weights, dtype=float)
    dates = r.index
    nav_list, ret_list, gains_list, fees_list, traded_list = [], [], [], [], []
    capital = 1.0
    # Charge 10 bps on actual notional bought, NOT on an assumed $1 plus fee.
    first_fee = FEE / (1.0 + FEE)
    holdings = weights * (1.0 - first_fee)
    nav_list.append(float(holdings.sum()))
    ret_list.append(-first_fee)
    gains_list.append(pd.Series(0.0, index=weights.index))
    fees_list.append(first_fee)
    traded_list.append(1.0 - first_fee)
    capital = float(holdings.sum())

    for i in range(1, len(dates)):
        date = dates[i]
        dollar_gains = holdings * r.loc[date, weights.index]
        holdings = holdings + dollar_gains
        cost = turnover = 0.0
        if i == len(dates) - 1 or dates[i].quarter != dates[i + 1].quarter \
                or dates[i].year != dates[i + 1].year:
            switches = set()
            if all_weather and date == TIPS_SWITCH_CLOSE:
                switches.add("TIPS")
            if all_weather and date == GOLD_SWITCH_CLOSE:
                switches.add("GOLD")
            holdings, cost, turnover = rebalance(holdings, weights, switches)
        next_nav = float(holdings.sum())
        ret_list.append(next_nav / capital - 1.0)
        nav_list.append(next_nav)
        gains_list.append(dollar_gains)
        fees_list.append(cost)
        traded_list.append(turnover)
        capital = next_nav
    results = Backtest(
        nav=pd.Series(nav_list, index=dates),
        daily_returns=pd.Series(ret_list, index=dates),
        gains=pd.DataFrame(gains_list, index=dates)[list(weights.index)],
        fees=pd.Series(fees_list, index=dates),
        traded=pd.Series(traded_list, index=dates))
    check = 1.0 + results.gains.sum(axis=1).cumsum() - results.fees.cumsum()
    if not np.allclose(check.to_numpy(), results.nav.to_numpy(), atol=1e-10):
        raise AssertionError("Daily total gains/fees do not reconcile to NAV")
    return results


def annual_detail(bt: Backtest) -> tuple[pd.Series, pd.DataFrame]:
    """Arithmetic attribution, exactly additive even with quarterly rebalance.

    Contribution = sum of each sleeve's daily DOLLAR profit / NAV at beginning
    of year. Not simply quarter-end weight multiplied by annual asset return.
    """
    returns, rows = {}, []
    opening = 1.0
    for year, daily in bt.nav.groupby(bt.nav.index.year):
        closing = float(daily.iloc[-1])
        y = daily.index
        contributions = bt.gains.loc[y].sum() / opening
        contributions["Trading costs"] = -bt.fees.loc[y].sum() / opening
        expected = closing / opening - 1.0
        if not np.isclose(contributions.sum(), expected, atol=1e-10):
            raise AssertionError(f"Attribution does not add up for {year}")
        returns[int(year)] = expected
        rows.append(pd.Series(contributions, name=int(year)))
        opening = closing
    return pd.Series(returns), pd.DataFrame(rows).fillna(0.0)


def metrics(bt: Backtest, cash_per_session: pd.Series) -> dict[str, float]:
    years = (END - START).days / 365.25
    ret = bt.daily_returns
    excess = ret - cash_per_session.loc[ret.index]
    vol = ret.std(ddof=1) * math.sqrt(252)
    sharpe = excess.mean() / excess.std(ddof=1) * math.sqrt(252)
    peaks = np.maximum.accumulate(np.r_[1.0, bt.nav.to_numpy()])
    dd = np.r_[1.0, bt.nav.to_numpy()] / peaks - 1.0
    return {"CAGR": bt.nav.iloc[-1] ** (1 / years) - 1,
            "Volatility": vol, "Sharpe": sharpe,
            "Max drawdown": dd.min(),
            "Total fee drag (absolute $ per initial $1)": bt.fees.sum(),
            "Total traded (absolute $ per initial $1)": bt.traded.sum()}


def main() -> None:
    prices = fetch_etfs()
    bullion = fetch_csv_series("https://www.goldprice.com/gold-price-history.csv", "close")
    asset_returns = prepare_returns(prices, bullion)
    cash = rate_data(asset_returns.index)
    portfolios = {"All weather": simulate(asset_returns, AW, True),
                  "60/40": simulate(asset_returns, SIXTY, False)}

    print("\nPERIOD AND FIRST-DAY HOLDINGS")
    print(f"Prices: {asset_returns.index.min().date()} to {asset_returns.index.max().date()}; "
          f"{len(asset_returns):,} US ETF market sessions")
    print("2002-07-31 All weather: 25% SPY, 25% TLT, 25% VIPSX (TIPS proxy), "
          "25% bullion price proxy")
    print("2002-07-31 60/40: 60% SPY, 40% IEF")
    print("TIPS proxy exchanged for TIP at 2003-12-31 close; "
          "gold proxy exchanged for GLD at 2004-12-31 close")
    print("10 bp charged on buy + sell notional, including initial purchases "
          "and proxy exchanges")

    print("\nFULL-PERIOD METRICS")
    for name, bt in portfolios.items():
        m = metrics(bt, cash)
        print(f"{name:12}  CAGR {m['CAGR']:+8.2%}  vol {m['Volatility']:7.2%}  "
              f"Sharpe {m['Sharpe']:6.3f}  max drawdown {m['Max drawdown']:8.2%}")
        print(f"              trading fees ${m['Total fee drag (absolute $ per initial $1)']:.5f}; "
              f"gross notional ${m['Total traded (absolute $ per initial $1)']:.3f} "
              "per $1 initially invested")

    a_ret, a_attr = annual_detail(portfolios["All weather"])
    b_ret, b_attr = annual_detail(portfolios["60/40"])
    yearly = pd.DataFrame({"All weather": a_ret, "60/40": b_ret})
    yearly["AW minus 60/40"] = yearly["All weather"] - yearly["60/40"]
    print("\nALL CALENDAR-YEAR RETURNS (after costs; difference in pp)")
    for year, vals in yearly.iterrows():
        partial = " (PARTIAL)" if year in (2002, 2026) else ""
        print(f"{year}{partial:10} AW {vals['All weather']:+8.2%}  "
              f"60/40 {vals['60/40']:+8.2%}  gap "
              f"{100*vals['AW minus 60/40']:+7.2f} pp")
    winners = yearly.loc[2003:2025].query("`AW minus 60/40` > 0")
    print("\nCOMPLETE YEARS ALL WEATHER WON:", ", ".join(map(str, winners.index)))

    top = yearly.loc[2003:2025, "AW minus 60/40"].abs().nlargest(3).index
    print("\nTHREE LARGEST ABSOLUTE FULL-YEAR RETURN GAPS")
    for year in top:
        print(f"\n{year}: AW {yearly.at[year, 'All weather']:+.2%}; "
              f"60/40 {yearly.at[year, '60/40']:+.2%}; "
              f"gap {100*yearly.at[year, 'AW minus 60/40']:+.2f} pp")
        print("Exact contribution to EACH portfolio's yearly return (percentage points):")
        for name, attr in (("AW", a_attr), ("60/40", b_attr)):
            print("  " + name + ": " + ", ".join(
                f"{k} {100*v:+.2f}" for k, v in attr.loc[year].items()))
        gap_contr = a_attr.loc[year].subtract(b_attr.loc[year], fill_value=0)
        print("  Contribution to AW minus 60/40 (pp): " + ", ".join(
            f"{k} {100*v:+.2f}" for k, v in gap_contr.items()))
        print(f"  Check: contributions sum to {100*gap_contr.sum():+.2f} pp")

    yearly.to_csv("all_weather_vs_6040_annual.csv", float_format="%.10f")
    a_attr.to_csv("all_weather_attribution.csv", float_format="%.10f")
    b_attr.to_csv("6040_attribution.csv", float_format="%.10f")
    print("\nSaved: all_weather_vs_6040_annual.csv, "
          "all_weather_attribution.csv, 6040_attribution.csv")


if __name__ == "__main__":
    main()
