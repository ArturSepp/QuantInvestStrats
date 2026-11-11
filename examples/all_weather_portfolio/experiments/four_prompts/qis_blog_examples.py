"""
Runner for the two worked examples of the qis JOSS blog post
and the reference for the AI comparison.

Example 1: a quarterly rebalanced 60/40 SPY/IEF portfolio with 10 bp trading costs and its
strategy factsheet.

Example 2: an all-weather-style allocation of 25% each to SPY, TLT, TIP and GLD, whose ETFs were
launched in different years, under three treatments of the uneven history:
    truncated panel   prices.dropna(), so the backtest starts at the last launch
    static vector     the weight of an unpriced ETF stays in the cash balance
    live universe     qis.generate_static_weights_schedule allocates over the ETFs priced at
                      each rebalancing date, w*_i = E I_i b_i / sum_j I_j b_j
The live-universe portfolio is the reference for the AI comparison: its statistics are printed
in the fixed format the prompts request, and its nav is saved to nav_reference.csv.

Prices are Yahoo adjusted closes through yfinance: pip install "qis[data]". The sample ends on
30 September 2026, a business-day month-end, so the monthly statistics cover complete months.
Figures, factsheet PDFs and csv files go to the working directory, and every number depends on
the download date. This is a script for the post, not library code.
"""

# packages
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from typing import Dict, List

# qis
import qis as qis

SAMPLE_END_EXCLUSIVE: str = (
    "2026-10-01"  # yfinance end is exclusive: last observation 30 September 2026
)
REBALANCING_FREQ: str = "QE"
REBALANCING_COSTS: float = 0.0010  # fractional units: 10 bp of traded notional
PERF_COLUMNS: List[qis.PerfStat] = [
    qis.PerfStat.TOTAL_RETURN,
    qis.PerfStat.PA_RETURN,
    qis.PerfStat.VOL,
    qis.PerfStat.SHARPE_RF0,
    qis.PerfStat.MAX_DD,
]

BALANCED_WEIGHTS: Dict[str, float] = {"SPY": 0.6, "IEF": 0.4}
BALANCED_START: str = "2003-12-31"

ALL_WEATHER_WEIGHTS: Dict[str, float] = {"SPY": 0.25, "TLT": 0.25, "TIP": 0.25, "GLD": 0.25}
ALL_WEATHER_START: str = "2002-07-31"


class LocalTest(Enum):
    BALANCED_60_40 = 1  # example 1
    ALL_WEATHER_UNEVEN_HISTORY = 2  # example 2 and the reference for the AI comparison


def fetch_prices(tickers: List[str], start: str, end: str = SAMPLE_END_EXCLUSIVE) -> pd.DataFrame:
    """
    download adjusted closes on a business-day grid, keeping the leading NaNs of late launches.

    Args:
        tickers: Yahoo tickers, returned in this column order
        start: first date of the download, 'YYYY-MM-DD'
        end: exclusive end date of the download, 'YYYY-MM-DD'

    Returns:
        price panel indexed by business day; a ticker is NaN before its first quote

    Raises:
        ImportError: if yfinance is not installed
        ValueError: if a ticker has no quotes at all
    """
    try:
        import yfinance as yf
    except ImportError as error:
        raise ImportError(
            'yfinance is required for this script: pip install "qis[data]"'
        ) from error
    prices = yf.download(tickers, start=start, end=end, ignore_tz=True, auto_adjust=True)["Close"][
        tickers
    ]
    empty = [ticker for ticker in tickers if prices[ticker].notna().sum() == 0]
    if len(empty) > 0:
        raise ValueError(f"no quotes downloaded for {empty!r}")
    # Business-day grid: a holiday row repeats the previous row;
    # an ETF stays NaN before its launch.
    return prices.asfreq("B", method="ffill")


def compute_perf_table(navs: Dict[str, pd.Series]) -> pd.DataFrame:
    """Monthly-return statistics per NAV, each over its own history.

    Sharpe is compound p.a. with rf=0.
    """
    perf_params = qis.PerfParams(freq="ME")
    columns = [column.to_str() for column in PERF_COLUMNS]
    rows = {}
    for name, nav in navs.items():
        table = qis.compute_ra_perf_table(prices=nav.to_frame(name), perf_params=perf_params)[
            columns
        ]
        row = table.iloc[0].astype(float)
        row["First date"] = f"{nav.index[0]:%Y-%m-%d}"
        row["Years"] = (nav.index[-1] - nav.index[0]).days / 365.25
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def print_stats_block(nav: pd.Series) -> None:
    """print the statistics of one nav in the fixed format requested from the AI scripts."""
    stats = compute_perf_table(navs={"nav": nav}).iloc[0]
    print(f"first_date: {nav.index[0]:%Y-%m-%d}")
    print(f"last_date: {nav.index[-1]:%Y-%m-%d}")
    print(f"terminal_nav: {nav.iloc[-1]:.4f}")
    print(f"total_return: {stats[qis.PerfStat.TOTAL_RETURN.to_str()]:.4f}")
    print(f"annual_return: {stats[qis.PerfStat.PA_RETURN.to_str()]:.4f}")
    print(f"volatility: {stats[qis.PerfStat.VOL.to_str()]:.4f}")
    print(f"sharpe: {stats[qis.PerfStat.SHARPE_RF0.to_str()]:.4f}")
    print(f"max_drawdown: {stats[qis.PerfStat.MAX_DD.to_str()]:.4f}")


def run_local_test(local_test: LocalTest) -> None:
    """
    run one example and write its figures to the working directory.

    Args:
        local_test: which example to run
    """
    if local_test == LocalTest.BALANCED_60_40:
        tickers = list(BALANCED_WEIGHTS.keys())
        prices = fetch_prices(tickers=tickers, start=BALANCED_START).dropna()
        portfolio_data = qis.backtest_model_portfolio(
            prices=prices,
            weights=BALANCED_WEIGHTS,
            rebalancing_freq=REBALANCING_FREQ,
            rebalancing_costs=REBALANCING_COSTS,
            is_rebalanced_at_first_date=True,
            ticker="60/40",
        )
        figs = qis.factsheet(
            portfolio_data,
            benchmark_prices=prices["SPY"],
            reporting_frequency="monthly",
            add_rates_data=False,
            factsheet_name="60/40 SPY/IEF, quarterly rebalancing, 10 bp costs",
        )
        qis.save_figs_to_pdf(
            figs=figs,
            file_name="qis_blog_60_40_factsheet",
            orientation="portrait",
            local_path=None,
            add_current_date=False,
        )
        figs[0].savefig("qis_blog_fig1_60_40_factsheet.png", dpi=150)

        print(
            f"prices: {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}, "
            f"shape={prices.shape}"
        )
        display_weights = portfolio_data.get_weights(
            is_input_weights=True, freq=None).iloc[-1].round(4).to_dict()
        print(f"last target weights:   {display_weights}")
        print(f"last realised weights: {portfolio_data.weights.iloc[-1].round(4).to_dict()}")
        spy_weight = portfolio_data.weights["SPY"].iloc[1:]
        print(
            f"realised SPY weight over the sample: {spy_weight.min():.4f} to {spy_weight.max():.4f}"
        )
        navs = {"60/40": portfolio_data.get_portfolio_nav(), "SPY": prices["SPY"]}
        print(compute_perf_table(navs=navs).round(3).to_string())

    elif local_test == LocalTest.ALL_WEATHER_UNEVEN_HISTORY:
        tickers = list(ALL_WEATHER_WEIGHTS.keys())
        prices = fetch_prices(tickers=tickers, start=ALL_WEATHER_START)
        first_dates = prices.apply(pd.Series.first_valid_index)
        last_launch = first_dates.max()
        print("first quote per ETF:")
        print(first_dates.dt.strftime("%Y-%m-%d").to_string())

        # 1. Truncated panel: every ETF is priced on every row;
        # history before the last launch is gone.
        truncated = qis.backtest_model_portfolio(
            prices=prices.dropna(),
            weights=ALL_WEATHER_WEIGHTS,
            rebalancing_freq=REBALANCING_FREQ,
            rebalancing_costs=REBALANCING_COSTS,
            is_rebalanced_at_first_date=True,
            ticker="Truncated panel",
        )

        # 2. static vector on the ragged panel: qis warns that unpriced legs are not traded
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            static = qis.backtest_model_portfolio(
                prices=prices,
                weights=ALL_WEATHER_WEIGHTS,
                rebalancing_freq=REBALANCING_FREQ,
                rebalancing_costs=REBALANCING_COSTS,
                is_rebalanced_at_first_date=True,
                ticker="Static vector",
            )
        for warning in caught:
            if issubclass(warning.category, UserWarning):
                print(f"\nqis warning: {warning.message}")

        # 3. live universe: dated targets over the ETFs priced at each rebalancing date
        schedule = qis.generate_static_weights_schedule(
            prices=prices, weights=ALL_WEATHER_WEIGHTS, rebalancing_freq=REBALANCING_FREQ
        )
        live = qis.backtest_model_portfolio(
            prices=prices,
            weights=schedule,
            rebalancing_freq=None,
            rebalancing_costs=REBALANCING_COSTS,
            ticker="Live universe",
        )

        changes = schedule.loc[schedule.diff().abs().sum(axis=1) > 1e-12]
        print("\ntarget schedule: first row and the rows where the targets change")
        print(pd.concat([schedule.iloc[:1], changes]).round(4).to_string())

        table = compute_perf_table(
            navs={
                portfolio.ticker: portfolio.get_portfolio_nav()
                for portfolio in (truncated, static, live)
            }
        )
        for portfolio in (static, live):
            invested = portfolio.weights.sum(axis=1).iloc[1:].loc[:last_launch]
            table.loc[portfolio.ticker, "Min invested before last launch"] = invested.min()
        print(f"\nlast launch: {last_launch:%Y-%m-%d}")
        print(table.round(3).to_string())

        # reference for the AI comparison: same lines as the prompts request, and the nav as csv
        live_nav = live.get_portfolio_nav()
        print("\nreference (live universe):")
        print_stats_block(nav=live_nav)
        live_nav.rename("nav").rename_axis("date").to_csv("nav_reference.csv")

        figs = qis.factsheet(
            live,
            benchmark_prices=prices["SPY"],
            reporting_frequency="monthly",
            add_rates_data=False,
            factsheet_name=("25% SPY/TLT/TIP/GLD over the live universe, "
                            "quarterly rebalancing, 10 bp costs"),
        )
        qis.save_figs_to_pdf(
            figs=figs,
            file_name="qis_blog_all_weather_factsheet",
            orientation="portrait",
            local_path=None,
            add_current_date=False,
        )

        with sns.axes_style("darkgrid"):
            fig2, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
            qis.plot_stack(
                df=live.get_weights(freq="ME"),
                use_bar_plot=True,
                title="25% SPY/TLT/TIP/GLD over the live universe: realised month-end weights",
                legend_loc="upper left",
                bbox_to_anchor=(1.0, 1.0),
                var_format="{:.0%}",
                ax=ax,
            )
            fig2.savefig("qis_blog_fig2_live_universe_weights.png", dpi=150)

            fig3, axs = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
            navs = pd.concat([static.get_portfolio_nav(), live_nav], axis=1)
            qis.plot_prices_with_dd(
                prices=navs,
                perf_params=qis.PerfParams(freq="ME"),
                title="25% SPY/TLT/TIP/GLD: static vector against live universe",
                axs=axs,
            )
            fig3.savefig("qis_blog_fig3_static_vs_live.png", dpi=150)

    else:
        raise ValueError(f"unknown case: {local_test!r}")

    plt.close("all")


if __name__ == "__main__":
    for case in LocalTest:
        print(f"\n===== {case.name} =====")
        run_local_test(local_test=case)
