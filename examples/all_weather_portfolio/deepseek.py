"""
All-Weather vs 60/40 backtest
Period: 31 July 2002 through 30 September 2026
Quarterly rebalancing, 10 bp trading cost on traded notional
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402 - preserve the captured warning-filter setup above
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
START = "2002-07-31"
END = "2026-09-30"
DOWNLOAD_END = "2026-10-01"   # yfinance end is exclusive
COST_BPS = 10.0
TRADING_DAYS = 252

# All-Weather: 25% each
AW_WEIGHTS = {
    "SPY": 0.25,
    "TLT": 0.25,
    "TIP": 0.25,   # VIPSX proxy before TIP inception
    "GLD": 0.25,   # GC=F proxy before GLD inception
}

# 60/40: 60% US equities, 40% intermediate Treasuries
SIXTY_FORTY_WEIGHTS = {
    "SPY": 0.60,
    "IEF": 0.40,
}

# ---------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------
def download_adj_close(ticker, start="2000-01-01", end=DOWNLOAD_END):
    """Download adjusted close, retrying if Yahoo returns an empty frame."""
    last_err = None
    for auto_adjust in [True, False]:
        for attempt in range(3):
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=auto_adjust,
                    progress=False,
                    actions=False,
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if "Close" not in df.columns:
                    raise ValueError(f"No Close column for {ticker}")
                s = df["Close"].dropna()
                if len(s) > 0:
                    s.name = ticker
                    return s.sort_index()
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Could not download {ticker}: {last_err}")

def download_yield(ticker="^IRX"):
    """Download a yield series such as the 13-week T-bill yield."""
    df = yf.download(
        ticker,
        start=START,
        end=DOWNLOAD_END,
        auto_adjust=False,
        progress=False,
        actions=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.name = ticker
    return s.sort_index()

def pct_change(s):
    return s.sort_index().pct_change(fill_method=None)

def spliced_returns(
    etf_prices: pd.Series,
    proxy_prices: pd.Series,
    base_index: pd.DatetimeIndex,
    etf_name: str,
    proxy_name: str,
) -> pd.Series:
    """
    Use proxy returns before the ETF's first valid return; use ETF returns
    from the ETF's first valid return onward.
    """
    etf_r = pct_change(etf_prices).reindex(base_index)
    proxy_r = pct_change(proxy_prices).reindex(base_index)

    combined = etf_r.copy()
    first_etf = combined.first_valid_index()

    if first_etf is not None:
        pre_mask = combined.index < first_etf
        combined.loc[pre_mask] = proxy_r.loc[pre_mask]

    # If the ETF has a missing day after inception, fall back to the proxy
    # rather than forward-filling a stale price.
    combined = combined.fillna(proxy_r).fillna(0.0)
    combined.name = etf_name
    return combined

# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------
def simulate(returns: pd.DataFrame, weights: dict, cost_bps: float):
    """
    Daily simulation with quarterly rebalancing.
    Returns:
      values: daily portfolio value
      contributions: DataFrame of daily asset contributions to portfolio return
      cost_contrib: Series of rebalancing-cost contributions
    """
    dates = returns.index
    assets = list(weights.keys())
    w = pd.Series(weights, dtype=float)

    holdings = 1.0 * w.copy()
    total = 1.0

    values = []
    contribution_rows = []
    cost_rows = []

    # Last trading day of each calendar quarter
    quarter_ends = pd.date_range(dates[0], dates[-1], freq="QE")
    rebal_dates = set()
    for qe in quarter_ends:
        candidates = dates[dates <= qe]
        if len(candidates) > 0:
            rebal_dates.add(candidates[-1])

    for i, date in enumerate(dates):
        # Apply today's asset returns to yesterday's holdings
        if i > 0:
            r = returns.loc[date]
            pnl = holdings * r
            total_before_return = total

            for a in assets:
                contribution_rows.append(
                    (date, a, pnl[a] / total_before_return)
                )

            holdings = holdings * (1.0 + r)
            total = holdings.sum()

        # Rebalance after today's return, except on the initial day
        if date in rebal_dates and i > 0:
            target = total * w
            traded_notional = (target - holdings).abs().sum()
            cost = traded_notional * cost_bps / 10000.0

            total_before_cost = total
            total = total_before_cost - cost
            holdings = total * w

            cost_rows.append((date, -cost / total_before_cost))

        values.append((date, total))

    values = (
        pd.Series(dict(values), name="value")
        .sort_index()
    )
    contributions = (
        pd.DataFrame(
            contribution_rows,
            columns=["date", "asset", "contribution"],
        )
        .set_index("date")
        .sort_index()
    )
    cost_contrib = (
        pd.Series(dict(cost_rows), name="cost_contribution")
        .sort_index()
    )
    return values, contributions, cost_contrib

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def portfolio_metrics(values: pd.Series, rf_daily: pd.Series):
    daily_ret = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0
    vol = daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS)

    excess = daily_ret - rf_daily.reindex(daily_ret.index).fillna(0.0)
    sharpe = (
        np.sqrt(TRADING_DAYS)
        * excess.mean()
        / excess.std(ddof=1)
    )

    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
    }

def calendar_table(values: pd.Series) -> pd.DataFrame:
    rows = []

    # Partial 2002
    end_2002 = values.loc[: "2002-12-31"].iloc[-1]
    rows.append(
        ("2002 (partial)", end_2002 / values.iloc[0] - 1.0, "Partial")
    )

    # Complete calendar years
    for year in range(2003, 2026):
        start_val = values[values.index < f"{year}-01-01"].iloc[-1]
        end_val = values[values.index <= f"{year}-12-31"].iloc[-1]
        rows.append((str(year), end_val / start_val - 1.0, "Complete"))

    # Partial 2026
    start_2026 = values[values.index < "2026-01-01"].iloc[-1]
    end_2026 = values.loc[: "2026-09-30"].iloc[-1]
    rows.append(
        ("2026 (partial)", end_2026 / start_2026 - 1.0, "Partial")
    )

    return pd.DataFrame(rows, columns=["Year", "Return", "Type"])

def contribution_summary(
    contributions: pd.DataFrame,
    cost_contrib: pd.Series,
) -> pd.DataFrame:
    c = contributions.copy()
    c["year"] = c.index.year
    pivot = (
        c.groupby(["year", "asset"])["contribution"]
        .sum()
        .unstack(fill_value=0.0)
    )

    cost_by_year = cost_contrib.groupby(cost_contrib.index.year).sum()
    pivot["Trading cost"] = cost_by_year.reindex(pivot.index).fillna(0.0)
    pivot["Total"] = pivot.sum(axis=1)
    return pivot

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Downloading data...")

    spy = download_adj_close("SPY")
    tlt = download_adj_close("TLT")
    ief = download_adj_close("IEF")
    tip = download_adj_close("TIP")
    gld = download_adj_close("GLD")
    vipsx = download_adj_close("VIPSX")
    gold_futures = download_adj_close("GC=F")
    irx = download_yield("^IRX")

    # Use SPY trading calendar as the base
    base_index = spy.loc[START:END].index

    # Build return panel
    returns = pd.DataFrame(
        {
            "SPY": pct_change(spy).reindex(base_index),
            "TLT": pct_change(tlt).reindex(base_index),
            "IEF": pct_change(ief).reindex(base_index),
            "TIP": spliced_returns(
                tip, vipsx, base_index, "TIP", "VIPSX"
            ),
            "GLD": spliced_returns(
                gld, gold_futures, base_index, "GLD", "GC=F"
            ),
        }
    ).fillna(0.0)

    # Restrict to the requested period
    returns = returns.loc[START:END]

    # Daily risk-free rate
    rf_daily = (
        irx.reindex(returns.index)
        .ffill()
        .div(100.0)
        .div(TRADING_DAYS)
        .fillna(0.0)
    )

    # Simulate
    aw_values, aw_contrib, aw_cost = simulate(
        returns[["SPY", "TLT", "TIP", "GLD"]],
        AW_WEIGHTS,
        COST_BPS,
    )
    sf_values, sf_contrib, sf_cost = simulate(
        returns[["SPY", "IEF"]],
        SIXTY_FORTY_WEIGHTS,
        COST_BPS,
    )

    # Metrics
    aw_metrics = portfolio_metrics(aw_values, rf_daily)
    sf_metrics = portfolio_metrics(sf_values, rf_daily)

    print("\n" + "=" * 78)
    print("ETF choices")
    print("=" * 78)
    print(
        "All-Weather: SPY 25%, TLT 25%, TIP 25%, GLD 25%\n"
        "  - TIP sleeve uses VIPSX before TIP inception (4 Dec 2003).\n"
        "  - GLD sleeve uses GC=F before GLD inception (18 Nov 2004).\n"
        "60/40:       SPY 60%, IEF 40%"
    )

    print("\n" + "=" * 78)
    print("First investment date and initial holdings")
    print("=" * 78)
    print(f"First date: {aw_values.index[0].date()}")
    print(
        "All-Weather: SPY 25.0%, TLT 25.0%, "
        "TIP sleeve 25.0% (VIPSX proxy), "
        "GLD sleeve 25.0% (GC=F proxy)"
    )
    print("60/40:       SPY 60.0%, IEF 40.0%")

    print("\n" + "=" * 78)
    print("Q1. Full-period metrics")
    print("=" * 78)
    metrics_df = pd.DataFrame(
        {
            "All-Weather": aw_metrics,
            "60/40": sf_metrics,
        }
    ).T

    formatted = metrics_df.copy()
    formatted["CAGR"] = formatted["CAGR"].map(lambda x: f"{x:.2%}")
    formatted["Volatility"] = formatted["Volatility"].map(lambda x: f"{x:.2%}")
    formatted["Sharpe"] = formatted["Sharpe"].map(lambda x: f"{x:.3f}")
    formatted["Max Drawdown"] = formatted["Max Drawdown"].map(lambda x: f"{x:.2%}")
    print(formatted.to_string())

    print("\n" + "=" * 78)
    print("Q2. Calendar-year returns")
    print("=" * 78)
    aw_cal = calendar_table(aw_values).rename(columns={"Return": "AW"})
    sf_cal = calendar_table(sf_values).rename(columns={"Return": "60_40"})
    cal = aw_cal.merge(sf_cal[["Year", "60_40"]], on="Year")
    cal["Diff_pp"] = (cal["AW"] - cal["60_40"]) * 100.0

    display_cal = cal.copy()
    display_cal["AW"] = display_cal["AW"].map(lambda x: f"{x:.2%}")
    display_cal["60_40"] = display_cal["60_40"].map(lambda x: f"{x:.2%}")
    display_cal["Diff_pp"] = display_cal["Diff_pp"].map(lambda x: f"{x:+.2f} pp")
    print(display_cal.to_string(index=False))

    complete = cal[cal["Type"] == "Complete"].copy()
    beat = complete[complete["Diff_pp"] > 0].copy()
    print("\nComplete calendar years in which All-Weather beat 60/40:")
    if len(beat) == 0:
        print("  None")
    else:
        for _, row in beat.iterrows():
            print(
                f"  {row['Year']}: "
                f"AW {row['AW']:.2%}, 60/40 {row['60_40']:.2%}, "
                f"difference {row['Diff_pp']:+.2f} pp"
            )

    print("\nPartial years separately:")
    partial = cal[cal["Type"] == "Partial"]
    for _, row in partial.iterrows():
        print(
            f"  {row['Year']}: "
            f"AW {row['AW']:.2%}, 60/40 {row['60_40']:.2%}, "
            f"difference {row['Diff_pp']:+.2f} pp"
        )

    print("\n" + "=" * 78)
    print("Q3. Three complete years with largest absolute return gaps")
    print("=" * 78)

    complete["AbsDiff"] = complete["Diff_pp"].abs()
    top3 = complete.sort_values("AbsDiff", ascending=False).head(3)

    aw_contrib_summary = contribution_summary(aw_contrib, aw_cost) * 100.0
    sf_contrib_summary = contribution_summary(sf_contrib, sf_cost) * 100.0

    for _, row in top3.iterrows():
        year = int(row["Year"])
        print("\n" + "-" * 78)
        print(
            f"Year {year}: "
            f"AW {row['AW']:.2%}, 60/40 {row['60_40']:.2%}, "
            f"gap {row['Diff_pp']:+.2f} pp"
        )

        print("\nAll-Weather contributions (percentage points):")
        print(aw_contrib_summary.loc[year].round(2).to_string())

        print("\n60/40 contributions (percentage points):")
        print(sf_contrib_summary.loc[year].round(2).to_string())

        # Simple explanation of the dominant contributor
        aw_row = aw_contrib_summary.loc[year].drop(
            labels=["Total"], errors="ignore"
        )
        sf_row = sf_contrib_summary.loc[year].drop(
            labels=["Total"], errors="ignore"
        )
        print(
            f"\nLargest AW contributor: {aw_row.idxmax()} "
            f"({aw_row.max():+.2f} pp)"
        )
        print(
            f"Largest 60/40 contributor: {sf_row.idxmax()} "
            f"({sf_row.max():+.2f} pp)"
        )

    print("\n" + "=" * 78)
    print("Notes")
    print("=" * 78)
    print(
        "- Returns are daily total returns in USD from adjusted close prices.\n"
        "- Rebalancing occurs on the last trading day of each calendar quarter.\n"
        "- Trading cost is 10 bp of the sum of absolute trade notional and is\n"
        "  deducted from portfolio value at each rebalance.\n"
        "- Sharpe uses daily excess returns over the 13-week T-bill yield.\n"
        "- Contributions are arithmetic daily P&L contributions; they sum\n"
        "  approximately, not exactly, to the compounded annual return."
    )

if __name__ == "__main__":
    main()
