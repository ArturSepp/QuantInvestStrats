#!/usr/bin/env python3
"""
All-Weather (25/25/25/25) vs classic 60/40: quarterly-rebalanced ETF backtest.

Period : invested at the close of 2002-07-31, evaluated to the close of 2026-09-30
Costs  : 10 bp of traded ETF notional, including the initial purchase
Data   : Yahoo Finance via yfinance (split- and dividend-adjusted closes);
         FRED DTB3 3-month T-bill rate (fallback: Yahoo ^IRX)

    pip install "pandas>=2.0" numpy yfinance
    python aw_vs_6040.py --cache-dir data --outdir results
    python aw_vs_6040.py --missing renormalize     # sensitivity: pre-listing handling
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", FutureWarning)

START = pd.Timestamp("2002-07-31")
END = pd.Timestamp("2026-09-30")
TDAYS = 252
CASH = "CASH"
AW, SF = "AllWeather", "60/40"
PORTFOLIOS: dict[str, dict[str, float]] = {
    AW: {"SPY": 0.25, "TLT": 0.25, "TIP": 0.25, "GLD": 0.25},
    SF: {"SPY": 0.60, "IEF": 0.40},
}
ETFS = sorted({t for w in PORTFOLIOS.values() for t in w})
_DL_END = (END + pd.Timedelta(days=7)).strftime("%Y-%m-%d")


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


# ----------------------------------------------------------------------------- data
def _yf_close(tickers, start: str, end: str, auto_adjust: bool) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=auto_adjust, progress=False, threads=True
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers}")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers if isinstance(tickers, str) else tickers[0])
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)  # keep exchange-local dates
    close.index = idx.normalize()
    return close.sort_index().astype(float)


def load_prices(cache_dir: Path | None, refresh: bool) -> pd.DataFrame:
    f = cache_dir / "prices.csv" if cache_dir else None
    if f is not None and f.exists() and not refresh:
        px = pd.read_csv(f, index_col=0, parse_dates=True)
    else:
        px = _yf_close(ETFS, "2002-07-01", _DL_END, auto_adjust=True)[ETFS]
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            px.to_csv(f)
    return px.loc[:END]


def load_tbill(cache_dir: Path | None, refresh: bool) -> pd.Series:
    """3-month T-bill rate, discount basis, in percent."""
    f = cache_dir / "tbill.csv" if cache_dir else None
    if f is not None and f.exists() and not refresh:
        return pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
    try:
        s = pd.read_csv(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3",
            index_col=0,
            parse_dates=True,
            na_values=".",
        ).iloc[:, 0]
        s.name = "FRED_DTB3"
    except Exception as exc:  # network / format problems
        warn(f"FRED download failed ({exc}); using Yahoo ^IRX")
        s = _yf_close("^IRX", "2002-01-01", _DL_END, auto_adjust=False).iloc[:, 0]
        s.name = "YAHOO_IRX"
    s = s.astype(float).dropna().sort_index()
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(f)
    return s


def clean_prices(px: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    px = px.loc[px["SPY"].notna()].copy()  # SPY dates = NYSE trading calendar
    first = pd.Series({t: px[t].first_valid_index() for t in px.columns})
    for t in px.columns:
        if first[t] is None:
            raise RuntimeError(f"no price data for {t}")
        seg = px.loc[first[t] :, t]
        gaps = int(seg.isna().sum())
        if gaps:
            warn(f"{t}: {gaps} missing closes after listing, forward-filled")
            px.loc[first[t] :, t] = seg.ffill()
        seg = px.loc[first[t] :, t]
        jumps = (seg / seg.shift(1) - 1).abs()
        for d, v in jumps[jumps > 0.15].items():
            warn(f"{t}: {v:.1%} one-day move on {d:%Y-%m-%d}; check the data")
    return px, first


def daily_rf(tbill_pct: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """T-bill return earned from previous close to each close (calendar-day accrual)."""
    d = tbill_pct.reindex(tbill_pct.index.union(idx)).ffill().reindex(idx) / 100.0
    bey = 365.0 * d / (360.0 - 91.0 * d)  # discount rate -> bond-equivalent yield
    gap = idx.to_series().diff().dt.days
    return ((1.0 + bey.shift(1)) ** (gap / 365.0) - 1.0).fillna(0.0)


# ------------------------------------------------------------------------- backtest
@dataclass
class Result:
    nav: pd.Series  # end-of-day NAV after costs; capital = 1.0 before first trade
    pnl: pd.DataFrame  # daily P&L by sleeve
    cost: pd.Series  # trading cost paid each day
    turnover: pd.Series  # traded notional / pre-trade NAV on trade dates
    hold: pd.DataFrame  # end-of-day holdings by sleeve (after any rebalance)
    trades: list  # [(date, {sleeve: target weight})]


def backtest(
    target: dict[str, float], px: pd.DataFrame, rf: pd.Series, cost_bps: float, missing: str
) -> Result:
    idx = px.loc[START:END].index
    d0, dend = idx[0], idx[-1]
    etfs = list(target)
    cols = etfs + [CASH]
    ret = (px[etfs] / px[etfs].shift(1) - 1.0).reindex(idx)
    ret[CASH] = rf.reindex(idx)
    R = ret[cols].fillna(0.0).to_numpy()
    listed = px[etfs].reindex(idx).notna().to_numpy()
    tw = np.array([target[t] for t in etfs])
    c = cost_bps / 1e4
    qe = set(idx.to_series().groupby(idx.to_period("Q")).max())
    is_rebal = np.array([(d in qe) and (d0 < d < dend) for d in idx])

    def target_weights(k: int) -> np.ndarray:
        w = np.where(listed[k], tw, 0.0)
        if missing == "cash":  # unlisted sleeves earn T-bills
            return np.append(w, 1.0 - w.sum())
        return np.append(w / w.sum(), 0.0)  # renormalise over listed sleeves

    def rebalance(h: np.ndarray, w: np.ndarray):
        V = h.sum()
        Vp = V
        for _ in range(10):  # fixed point: V' = V - c*sum|w V' - h|
            Vp = V - c * np.abs(w[:-1] * Vp - h[:-1]).sum()
        traded = np.abs(w[:-1] * Vp - h[:-1]).sum()
        return w * Vp, V - Vp, traded / V

    n, m = len(idx), len(cols)
    H, P = np.zeros((n, m)), np.zeros((n, m))
    C, T = np.zeros(n), np.full(n, np.nan)
    trades = []
    h = np.zeros(m)
    h[-1] = 1.0  # start fully in cash
    w = target_weights(0)
    h, C[0], T[0] = rebalance(h, w)
    trades.append((d0, dict(zip(cols, w))))
    H[0] = h
    for k in range(1, n):
        P[k] = h * R[k]
        h = h + P[k]
        if is_rebal[k]:
            w = target_weights(k)
            h, C[k], T[k] = rebalance(h, w)
            trades.append((idx[k], dict(zip(cols, w))))
        H[k] = h
    hold = pd.DataFrame(H, index=idx, columns=cols)
    return Result(
        nav=hold.sum(axis=1),
        pnl=pd.DataFrame(P, index=idx, columns=cols),
        cost=pd.Series(C, index=idx),
        turnover=pd.Series(T, index=idx).dropna(),
        hold=hold,
        trades=trades,
    )


# --------------------------------------------------------------------------- stats
def perf(nav: pd.Series, rf: pd.Series, base: float) -> dict:
    dates = nav.index
    w = np.r_[1.0, nav.to_numpy() / base]  # wealth incl. pre-trade capital
    lab = [dates[0]] + list(dates)
    yrs = (dates[-1] - dates[0]).days / 365.25
    r = nav.pct_change().iloc[1:]
    ex = r - rf.reindex(r.index)
    peak = np.maximum.accumulate(w)
    dd = w / peak - 1.0
    k = int(dd.argmin())
    pk = int(np.flatnonzero(w[: k + 1] == peak[k])[-1])
    rec = np.flatnonzero(w[k:] >= peak[k])
    return {
        "Annualised return (CAGR)": w[-1] ** (1 / yrs) - 1,
        "Volatility (ann.)": r.std() * np.sqrt(TDAYS),
        "Sharpe (ann. excess over T-bills)": ex.mean() / ex.std() * np.sqrt(TDAYS),
        "Max drawdown": dd[k],
        "  MDD peak": f"{lab[pk]:%Y-%m-%d}",
        "  MDD trough": f"{lab[k]:%Y-%m-%d}",
        "  MDD recovered": f"{lab[k + rec[0]]:%Y-%m-%d}" if len(rec) else "not recovered",
        "Total return": w[-1] - 1,
        "T-bill CAGR": (1 + rf.reindex(r.index)).prod() ** (1 / yrs) - 1,
    }


def calendar_returns(nav: pd.Series) -> pd.Series:
    ye = nav.groupby(nav.index.year).last()
    return ye / ye.shift(1).fillna(1.0) - 1.0  # first year measured from capital 1.0


def classify_years(aw: Result) -> pd.Series:
    idx = aw.nav.index
    d0, d1 = idx[0], idx[-1]
    out = {}
    for y in sorted(set(idx.year)):
        days = idx[idx.year == y]
        if y == d0.year:
            out[y] = f"partial: {d0:%d %b %Y} close -> {days[-1]:%d %b %Y}"
        elif y == d1.year and d1 < pd.Timestamp(y, 12, 24):
            prev = idx[idx.year == y - 1][-1]
            out[y] = f"partial: {prev:%d %b %Y} close -> {d1:%d %b %Y}"
        else:
            miss = [t for t in PORTFOLIOS[AW] if (aw.hold.loc[days, t] <= 1e-12).any()]
            out[y] = (
                "complete"
                if not miss
                else f"complete, transitional (AW lacked {'/'.join(miss)} part of year)"
            )
    return pd.Series(out)


def attribution(res: Result, y: int):
    nav = res.nav
    prev = nav[nav.index.year < y]
    base = prev.iloc[-1] if len(prev) else 1.0
    m = nav.index.year == y
    contrib = res.pnl.loc[m].sum() / base
    contrib["Costs"] = -res.cost.loc[m].sum() / base
    total = nav.loc[m].iloc[-1] / base - 1.0
    assert np.isclose(contrib.sum(), total, atol=1e-10), "attribution does not add up"
    wts = res.hold.div(res.nav, axis=0).shift(1).loc[m].mean()  # avg start-of-day weight
    return contrib, wts, total


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--missing",
        choices=["cash", "renormalize"],
        default="cash",
        help="treatment of sleeves whose ETF is not yet listed",
    )
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--refresh", action="store_true", help="ignore cached data")
    ap.add_argument("--outdir", type=Path, default=None)
    a = ap.parse_args()
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    px, first = clean_prices(load_prices(a.cache_dir, a.refresh))
    tb = load_tbill(a.cache_dir, a.refresh)
    rf = daily_rf(tb, px.index)
    idx = px.loc[START:END].index
    if idx[0] != START:
        warn(f"first trading day on/after {START:%Y-%m-%d} in data is {idx[0]:%Y-%m-%d}")
    if idx[-1] != END:
        warn(f"data end on {idx[-1]:%Y-%m-%d}, not {END:%Y-%m-%d}")

    res = {n: backtest(w, px, rf, a.cost_bps, a.missing) for n, w in PORTFOLIOS.items()}

    print("=" * 100)
    print(
        f"Sample: {idx[0]:%Y-%m-%d} close -> {idx[-1]:%Y-%m-%d} close | {len(idx)} trading days"
        f" | cost {a.cost_bps:g} bp | pre-listing sleeves: {a.missing} | risk-free: {tb.name}"
    )
    print("First price in data:", ", ".join(f"{t} {d:%Y-%m-%d}" for t, d in first.items()))
    for n, r in res.items():
        d, w = r.trades[0]
        held = ", ".join(f"{k} {v:.1%}" for k, v in w.items() if v > 1e-12)
        print(f"{n}: first invested at close {d:%Y-%m-%d}: {held}")
        for t in PORTFOLIOS[n]:
            fi = next(dd for dd, ww in r.trades if ww[t] > 0)
            if fi != d:
                print(f"    {t} sleeve first bought at the {fi:%Y-%m-%d} rebalance")
        print(f"    {len(r.trades) - 1} quarterly rebalances")

    # ---------------- Q1
    print("\n" + "=" * 100 + "\nQ1  Full-period performance")
    yrs = (idx[-1] - idx[0]).days / 365.25
    tab = {}
    for n, r in res.items():
        s = perf(r.nav, rf, base=1.0)
        s["Avg annual turnover (rebalances)"] = r.turnover.iloc[1:].sum() / yrs
        s["Cost drag (ann., approx.)"] = (r.cost / (r.nav + r.cost)).sum() / yrs
        tab[n] = s
    q1 = pd.DataFrame(tab)
    pct_rows = [k for k in q1.index if "Sharpe" not in k and "MDD" not in k]
    print(
        q1.apply(
            lambda col: [
                f"{v:.2%}" if k in pct_rows else (f"{v:.3f}" if isinstance(v, float) else v)
                for k, v in col.items()
            ]
        ).to_string()
    )
    built = max(dd for dd, ww in res[AW].trades if all(ww[t] > 0 for t in PORTFOLIOS[AW]))
    built = min(dd for dd, ww in res[AW].trades if all(ww[t] > 0 for t in PORTFOLIOS[AW]))
    print(f"\nRobustness: from {built:%Y-%m-%d} (AW fully built) to {idx[-1]:%Y-%m-%d}")
    sub = pd.DataFrame(
        {n: perf(r.nav.loc[built:], rf, base=r.nav.loc[built]) for n, r in res.items()}
    )
    print(
        sub.apply(
            lambda col: [
                f"{v:.2%}" if k in pct_rows else (f"{v:.3f}" if isinstance(v, float) else v)
                for k, v in col.items()
            ]
        ).to_string()
    )

    # ---------------- Q2
    print("\n" + "=" * 100 + "\nQ2  Calendar-year returns")
    cal = pd.DataFrame({AW: calendar_returns(res[AW].nav), SF: calendar_returns(res[SF].nav)})
    cal["Gap_pp"] = (cal[AW] - cal[SF]) * 100
    cal["AW_beat"] = cal[AW] > cal[SF]
    cal["Status"] = classify_years(res[AW])
    show = cal.copy()
    show[AW] = (show[AW] * 100).map("{:7.2f}%".format)
    show[SF] = (show[SF] * 100).map("{:7.2f}%".format)
    show["Gap_pp"] = show["Gap_pp"].map("{:+7.2f}".format)
    is_complete = cal["Status"].str.startswith("complete")
    print("Complete calendar years (AW - 60/40 in percentage points):")
    print(show[is_complete].to_string())
    wins = cal[is_complete & cal["AW_beat"]].index.tolist()
    print(
        f"\nComplete years where All-Weather beat 60/40 ({len(wins)} of {int(is_complete.sum())}):"
        f" {wins}"
    )
    print("\nPartial years:")
    print(show[~is_complete].to_string())

    # ---------------- Q3
    print("\n" + "=" * 100 + "\nQ3  Largest absolute gaps: attribution")
    full = cal[cal["Status"] == "complete"]
    top = full["Gap_pp"].abs().nlargest(3).index.tolist()
    top_any = cal[is_complete]["Gap_pp"].abs().nlargest(3).index.tolist()
    print(f"Top-3 fully built complete years: {top}")
    if set(top_any) != set(top):
        print(
            f"(Including transitional years the top 3 would be {top_any}; transitional "
            "gaps largely reflect the T-bill sleeve, so they are excluded.)"
        )
    ye_px = px[ETFS].groupby(px.index.year).last()
    rows = ["SPY", "TLT", "TIP", "GLD", "IEF", CASH, "Costs"]
    attrib_out = {}
    for y in top:
        ca, wa, ta = attribution(res[AW], y)
        cs, ws, ts = attribution(res[SF], y)
        etf_r = ye_px.loc[y] / ye_px.loc[y - 1] - 1.0
        m = rf.index.year == y
        etf_r[CASH] = (1 + rf[m]).prod() - 1.0
        t = pd.DataFrame(index=rows)
        t["ETF_ret_%"] = etf_r.reindex(rows) * 100
        t["AW_avg_w_%"] = wa.reindex(rows) * 100
        t["AW_contrib_pp"] = ca.reindex(rows) * 100
        t["6040_avg_w_%"] = ws.reindex(rows) * 100
        t["6040_contrib_pp"] = cs.reindex(rows) * 100
        t = t.fillna(0.0)
        t.loc["Costs", ["ETF_ret_%", "AW_avg_w_%", "6040_avg_w_%"]] = np.nan
        t["Diff_pp (AW-6040)"] = t["AW_contrib_pp"] - t["6040_contrib_pp"]
        t.loc["TOTAL"] = [np.nan, np.nan, ta * 100, np.nan, ts * 100, (ta - ts) * 100]
        winner = AW if ta > ts else SF
        print(
            f"\n--- {y}: AW {ta:+.2%} vs 60/40 {ts:+.2%}  -> {winner} by "
            f"{abs(ta - ts) * 100:.2f} pp"
        )
        print(t.round(2).to_string(na_rep=""))
        wa_t, ws_t = PORTFOLIOS[AW], PORTFOLIOS[SF]
        d_eq = wa_t["SPY"] - ws_t["SPY"]
        bets = {f"Equity weight {d_eq:+.2f} x (SPY - IEF)": d_eq * (etf_r["SPY"] - etf_r["IEF"])}
        for k in ["TLT", "TIP", "GLD"]:
            bets[f"{wa_t[k]:.2f} x ({k} - IEF)"] = wa_t[k] * (etf_r[k] - etf_r["IEF"])
        bets["Residual (intra-quarter drift, cash, costs)"] = (ta - ts) - sum(bets.values())
        print("Gap as relative bets vs 60/40 (pp):")
        for k, v in bets.items():
            print(f"    {k:<45s} {v * 100:+7.2f}")
        attrib_out[y] = t

    if a.outdir:
        a.outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({n: r.nav for n, r in res.items()}).to_csv(a.outdir / "nav.csv")
        cal.to_csv(a.outdir / "calendar_years.csv")
        q1.to_csv(a.outdir / "q1_stats.csv")
        for y, t in attrib_out.items():
            t.to_csv(a.outdir / f"attribution_{y}.csv")
        print(f"\nCSV output written to {a.outdir}/")


if __name__ == "__main__":
    main()
