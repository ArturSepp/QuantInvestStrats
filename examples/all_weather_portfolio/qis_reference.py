"""
Runner for the blog post "All weather or 60/40? Exploring two portfolios with qis and AI".

Two quarterly rebalanced portfolios on Yahoo adjusted closes, net of 10 bp trading costs:
    All weather   25% each in SPY, TLT, TIP and GLD
    60/40         60% in SPY and 40% in IEF
Both are allocated over the live universe: at each rebalancing date the target of ETF i is
w*_i = I_i b_i / sum_j I_j b_j, with b_i the static weight and I_i = 1 when ETF i has a price.
TIP and GLD were launched after July 2002, so the all-weather portfolio starts with two ETFs.

Each figure of the post is one call on the qis.MultiPortfolioData that holds the two backtests,
or on one of its portfolios. Each case of LocalTest produces one figure and the tables behind it.

Statistics use monthly returns: the Sharpe ratio is the compound return p.a. over the
volatility with a zero risk-free rate, and regimes are quantiles of quarterly SPY returns.
The price panel is downloaded once and saved to prices.csv; every later run reads that file.
Figures, tables and the factsheet PDF go to the working directory. Run with --with-leverage
for the 2x allocation funded at the three-month Treasury bill quote (^IRX). That case also
saves the dated annual funding rates (decimal units) for reuse.
"""

# packages
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from pathlib import Path
from importlib import metadata
from typing import Dict, List

# qis
import qis as qis
from qis.portfolio.reports._paired_performance_inference import (
    compute_paired_performance_inference,
)

BUNDLED_DATA = Path(__file__).resolve().parent / "data"
PRICES_FILE: str = 'prices.csv'  # saved ETF panel for the baseline and leverage cases
SAMPLE_START: str = '2002-07-31'
SAMPLE_END_EXCLUSIVE: str = (
    '2026-10-01'  # yfinance end is exclusive: last observation 30 September 2026
)
REBALANCING_FREQ: str = 'QE'
REBALANCING_COSTS: float = 0.0010  # fractional units: 10 bp of traded notional
ALL_WEATHER_WEIGHTS: Dict[str, float] = {
    'SPY': 0.25,
    'TLT': 0.25,
    'TIP': 0.25,
    'GLD': 0.25,
}
BALANCED_WEIGHTS: Dict[str, float] = {'SPY': 0.60, 'IEF': 0.40}
BENCHMARK: str = 'SPY'  # equity regimes are quantiles of its quarterly returns
PERF_PARAMS: qis.PerfParams = qis.PerfParams(freq='ME')
REGIME_CLASSIFIER: qis.BenchmarkReturnsQuantilesRegime = qis.BenchmarkReturnsQuantilesRegime(
    freq='QE'
)
ATTRIBUTION_YEARS: List[int] = [2008, 2013, 2022, 2025]
SUB_PERIODS: Dict[str, qis.TimePeriod] = {
    'Jul 2002 to 2012': qis.TimePeriod('31Jul2002', '31Dec2012'),
    '2013 to 2021': qis.TimePeriod('31Dec2012', '31Dec2021'),
    '2022 to Sep 2026': qis.TimePeriod('31Dec2021', '30Sep2026'),
}


class LocalTest(Enum):
    PORTFOLIOS = 1  # section 1: the two backtests and the weights of the live universe
    PERFORMANCE = 2  # section 2: which one did better
    ANNUAL_RETURNS = 3  # section 3: in which years
    REGIMES = 4  # section 4: what kind of market favours it
    ATTRIBUTION = 5  # section 5: which ETF did the work
    DRAWDOWNS = 6  # section 6: how bad did it get
    SHARPE_DIFFERENCE = 7  # section 7: is the difference real
    FACTSHEET = 8  # section 8: the whole report in one call
    IMPLEMENTATION = 9
    CORRELATION_BETA = 10


def fetch_prices(file_name: str = PRICES_FILE) -> pd.DataFrame:
    """
    read the saved price panel, or download it once and save it.

    Args:
        file_name: csv file of the panel in the working directory

    Returns:
        adjusted closes on a business-day grid; an ETF is NaN before its first quote

    Raises:
        ImportError: if the file is absent and yfinance is not installed
        ValueError: if a ticker has no quotes at all
    """
    path = Path(file_name)
    if file_name == PRICES_FILE and not path.exists():
        path = BUNDLED_DATA / PRICES_FILE
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    try:
        import yfinance as yf
    except ImportError as error:
        raise ImportError(
            'yfinance is required for the download: pip install "qis[data]"'
        ) from error
    tickers = list(dict.fromkeys(list(ALL_WEATHER_WEIGHTS.keys()) + list(BALANCED_WEIGHTS.keys())))
    prices = yf.download(
        tickers,
        start=SAMPLE_START,
        end=SAMPLE_END_EXCLUSIVE,
        ignore_tz=True,
        auto_adjust=True,
    )['Close'][tickers]
    empty = [ticker for ticker in tickers if prices[ticker].notna().sum() == 0]
    if len(empty) > 0:
        raise ValueError(f"no quotes downloaded for {empty!r}")
    # Carry prices over holidays and keep pre-launch observations missing.
    prices = prices.asfreq('B', method='ffill').rename_axis('date')
    prices.to_csv(file_name)
    return prices


def backtest_static_allocation(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    ticker: str,
    funding_rate: pd.Series | None = None,
) -> qis.PortfolioData:
    """Rebalance over priced ETFs and accrue the supplied annual rate on cash."""
    prices = prices[list(weights.keys())]
    schedule = qis.generate_static_weights_schedule(
        prices=prices, weights=weights, rebalancing_freq=REBALANCING_FREQ
    )
    return qis.backtest_model_portfolio(
        prices=prices,
        weights=schedule,
        rebalancing_freq=None,
        rebalancing_costs=REBALANCING_COSTS,
        ticker=ticker,
        funding_rate=funding_rate,
    )


def build_portfolios(prices: pd.DataFrame) -> qis.MultiPortfolioData:
    """backtest the two portfolios and hold them in one object, with SPY as the regime benchmark."""
    all_weather = backtest_static_allocation(
        prices=prices, weights=ALL_WEATHER_WEIGHTS, ticker='All weather'
    )
    balanced = backtest_static_allocation(prices=prices, weights=BALANCED_WEIGHTS, ticker='60/40')
    return qis.MultiPortfolioData(
        portfolio_datas=[all_weather, balanced], benchmark_prices=prices[BENCHMARK]
    )


def report_table(table: pd.DataFrame, name: str, decimals: int = 4) -> None:
    """print a table and save it as <name>.csv in the working directory."""
    print(f"\n{name}:")
    print(table.to_string(float_format=lambda value: f"{value:.{decimals}f}"))
    table.to_csv(f"{name}.csv")


def run_local_test(local_test: LocalTest) -> None:
    """
    produce the figure and the tables of one section of the post in the working directory.

    Args:
        local_test: which section to produce
    """
    prices = fetch_prices()
    portfolios = build_portfolios(prices=prices)
    all_weather = portfolios.portfolio_datas[0]
    navs = portfolios.get_navs()
    perf_columns = [
        qis.PerfStat.TOTAL_RETURN.to_str(),
        qis.PerfStat.PA_RETURN.to_str(),
        qis.PerfStat.VOL.to_str(),
        qis.PerfStat.SHARPE_RF0.to_str(),
        qis.PerfStat.MAX_DD.to_str(),
    ]

    with sns.axes_style('darkgrid'):
        if local_test == LocalTest.PORTFOLIOS:
            print(
                f"prices: {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}, "
                f"shape={prices.shape}"
            )
            print("first quote per ETF:")
            print(prices.apply(pd.Series.first_valid_index).dt.strftime('%Y-%m-%d').to_string())
            for portfolio in portfolios.portfolio_datas:
                schedule = portfolio.get_weights(is_input_weights=True, freq=None)
                changes = schedule.loc[schedule.diff().abs().sum(axis=1) > 1e-12]
                print(
                    f"\n{portfolio.ticker}: target weights on the first date and where they change"
                )
                print(pd.concat([schedule.iloc[:1], changes]).round(4).to_string())
            navs.to_csv('navs.csv')
            fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
            qis.plot_stack(
                df=all_weather.get_weights(freq='ME'),
                use_bar_plot=True,
                title='All weather: realised month-end weights over the live universe',
                legend_loc='upper left',
                bbox_to_anchor=(1.0, 1.0),
                var_format='{:.0%}',
                ax=ax,
            )
            fig.savefig('aw6040_fig0_weights.png', dpi=150)

        elif local_test == LocalTest.PERFORMANCE:
            fig, axs = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
            portfolios.plot_nav(regime_benchmark=BENCHMARK, perf_params=PERF_PARAMS, ax=axs[0])
            portfolios.plot_drawdowns(regime_benchmark=BENCHMARK, ax=axs[1])
            fig.savefig('aw6040_fig1_performance.png', dpi=150)

            navs_with_benchmark = portfolios.get_navs(add_benchmarks_to_navs=True)
            report_table(
                qis.compute_ra_perf_table(prices=navs_with_benchmark, perf_params=PERF_PARAMS)[
                    perf_columns
                ],
                name='table_performance',
            )
            benchmark_stats = qis.compute_ra_perf_table_with_benchmark(
                prices=navs_with_benchmark, benchmark=BENCHMARK, perf_params=PERF_PARAMS
            )
            columns = [qis.PerfStat.BETA.to_str(), qis.PerfStat.R2.to_str()]
            report_table(benchmark_stats[columns], "table_spy_regression")
            for label, time_period in SUB_PERIODS.items():
                table = qis.compute_ra_perf_table(
                    prices=time_period.locate(navs_with_benchmark),
                    perf_params=PERF_PARAMS,
                )
                report_table(
                    table[perf_columns],
                    name=f"table_performance_{label.replace(' ', '_').lower()}",
                )

        elif local_test == LocalTest.ANNUAL_RETURNS:
            fig, ax = plt.subplots(1, 1, figsize=(14, 3.5), constrained_layout=True)
            portfolios.plot_periodic_returns(title='Calendar-year returns', ax=ax)
            fig.savefig('aw6040_fig2_annual_returns.png', dpi=150)

            annual = qis.compute_periodic_returns(
                prices=portfolios.get_navs(add_benchmarks_to_navs=True),
                freq='YE',
                date_format='%Y',
                add_total=False,
            )
            annual['All weather - 60/40'] = annual['All weather'] - annual['60/40']
            report_table(annual, name='table_annual_returns')
            ahead = annual.index[annual['All weather - 60/40'] > 0.0].to_list()
            print(
                f"all weather ahead in {len(ahead)} of {len(annual.index)} calendar periods, "
                f"the first and the last of them incomplete: {ahead}"
            )

        elif local_test == LocalTest.REGIMES:
            fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
            portfolios.plot_regime_data(
                benchmark=BENCHMARK,
                regime_data_to_plot=qis.RegimeData.REGIME_AVG,
                perf_params=PERF_PARAMS,
                regime_classifier=REGIME_CLASSIFIER,
                title='Average quarterly return in bear, normal and bull quarters of SPY',
                yvar_format='{:.1%}',
                ax=axs[0],
            )
            portfolios.plot_returns_scatter(benchmark=BENCHMARK, freq='QE', ax=axs[1])
            fig.savefig('aw6040_fig3_regimes.png', dpi=150)

            navs_with_benchmark = portfolios.get_navs(add_benchmarks_to_navs=True)
            _, regime_datas = REGIME_CLASSIFIER.compute_regimes_pa_perf_table(
                prices=navs_with_benchmark, benchmark=BENCHMARK, perf_params=PERF_PARAMS
            )
            report_table(
                regime_datas[qis.RegimeData.REGIME_AVG],
                name='table_regime_average_quarterly_returns',
            )
            report_table(regime_datas[qis.RegimeData.REGIME_SHARPE], "table_regime_sharpe")
            # how often all weather is ahead of 60/40, by the sign of the 60/40 return
            for freq in ['ME', 'QE']:
                returns = qis.to_returns(prices=navs, freq=freq, drop_first=True)
                is_ahead = returns['All weather'] > returns['60/40']
                is_down = returns['60/40'] < 0.0
                hits = pd.DataFrame(
                    {
                        'periods': [is_down.sum(), (~is_down).sum()],
                        'avg 60/40': [
                            returns.loc[is_down, '60/40'].mean(),
                            returns.loc[~is_down, '60/40'].mean(),
                        ],
                        'avg all weather': [
                            returns.loc[is_down, 'All weather'].mean(),
                            returns.loc[~is_down, 'All weather'].mean(),
                        ],
                        'share all weather ahead': [
                            is_ahead[is_down].mean(),
                            is_ahead[~is_down].mean(),
                        ],
                    },
                    index=['60/40 falls', '60/40 rises'],
                )
                report_table(hits, name=f"table_ahead_by_sign_{freq}")

        elif local_test == LocalTest.ATTRIBUTION:
            fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
            attributions = {}
            for year, ax in zip(ATTRIBUTION_YEARS, axs.flatten()):
                time_period = qis.TimePeriod(f"31Dec{year - 1}", f"31Dec{year}")
                all_weather.plot_performance_attribution(time_period=time_period, ax=ax)
                ax.set_title(
                    f"All weather in {year}: contributions to the compounded gross return",
                    fontsize=12,
                    pad=18,
                    color='darkblue',
                )
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
                for portfolio in portfolios.portfolio_datas:
                    attributions[(year, portfolio.ticker)] = portfolio.get_performance_attribution(
                        time_period=time_period
                    )
            fig.savefig('aw6040_fig4_attribution.png', dpi=150)
            for portfolio in portfolios.portfolio_datas:
                attributions[('Full sample', portfolio.ticker)] = (
                    portfolio.get_performance_attribution()
                )
            report_table(pd.DataFrame(attributions), name='table_attribution_daily_sum')
            linked = {}
            for label in [*ATTRIBUTION_YEARS, 'Full sample']:
                window = (
                    qis.TimePeriod(f'31Dec{label - 1}', f'31Dec{label}')
                    if isinstance(label, int)
                    else None
                )
                for portfolio in portfolios.portfolio_datas:
                    leg = portfolio.get_instruments_performance_attribution(time_period=window)
                    linked[label, portfolio.ticker] = pd.concat(
                        [leg, pd.Series({'Total': leg.sum()})]
                    )
            report_table(pd.DataFrame(linked), name='table_attribution')

            # Compare capital shares with volatility shares using monthly log-return covariance.
            returns = qis.to_returns(
                prices=prices[list(ALL_WEATHER_WEIGHTS.keys())],
                freq='ME',
                is_log_returns=True,
                drop_first=True,
            )
            covar = 12.0 * qis.compute_masked_covar_corr(data=returns, is_covar=True)
            weights = pd.Series(ALL_WEATHER_WEIGHTS)
            risk = pd.DataFrame(
                {
                    'capital share': weights,
                    'risk share': qis.compute_portfolio_risk_contribution_ratios(weights, covar),
                }
            )
            report_table(risk, name='table_risk_shares')

        elif local_test == LocalTest.DRAWDOWNS:
            fig, axs = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
            for portfolio, ax in zip(portfolios.portfolio_datas, axs):
                nav = portfolio.get_portfolio_nav()
                qis.plot_top_drawdowns_paths(
                    nav,
                    max_num=5,
                    title=f"{portfolio.ticker}: the five deepest drawdowns",
                    ax=ax,
                )
                label = portfolio.ticker.replace("/", "_").replace(" ", "_").lower()
                report_table(
                    qis.compute_drawdowns_stats_table(nav, max_num=5),
                    name=f"table_drawdowns_{label}",
                )
            fig.savefig('aw6040_fig5_drawdowns.png', dpi=150)

        elif local_test == LocalTest.IMPLEMENTATION:
            annual = {}
            for portfolio in portfolios.portfolio_datas:
                flows = pd.DataFrame(
                    {
                        "Turnover": portfolio.get_turnover(is_agg=True, roll_period=None),
                        "Trading costs": portfolio.get_costs(is_agg=True, roll_period=None),
                    }
                )
                annual[portfolio.ticker] = flows.loc["2005":"2025"].resample("YE").sum()
            report_table(pd.concat(annual, axis=1), name="table_annual_turnover_costs")

        elif local_test == LocalTest.CORRELATION_BETA:
            pair = prices[["SPY", "TLT"]]
            monthly = qis.to_returns(prices=pair, freq="ME", is_log_returns=True)
            windows = [("2005", "2012"), ("2013", "2021"), ("2022", "2026-09")]
            rows = {}
            for start, end in windows:
                sample = monthly.loc[start:end].dropna()
                corr = qis.compute_masked_covar_corr(data=sample, is_covar=False)
                rows[f"{start} to {end}"] = {
                    "Months": len(sample),
                    "Correlation": corr.at["SPY", "TLT"],
                }
            report_table(
                pd.DataFrame.from_dict(rows, orient="index"),
                "table_spy_tlt_correlation",
            )
            ew_corr = qis.compute_ewm_corr_df(df=monthly, span=36, init_type=qis.InitType.X0)
            betas = pd.DataFrame(
                {
                    portfolio.ticker: portfolio.compute_portfolio_benchmark_betas(
                        benchmark_prices=prices[[BENCHMARK]],
                        freq_beta="ME",
                        factor_beta_span=36,
                    )[BENCHMARK]
                    for portfolio in portfolios.portfolio_datas
                }
            )
            pd.concat([ew_corr, betas], axis=1).to_csv("series_correlation_beta.csv")
            fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
            qis.plot_returns_corr_matrix_time_series(
                prices=pair,
                freq="ME",
                span=36,
                ax=axes[0],
                title="36-month EW correlation: SPY and TLT",
            )
            portfolios.plot_factor_betas(
                benchmark_prices=prices[[BENCHMARK]],
                freq_beta="ME",
                factor_beta_span=36,
                axs=[axes[1]],
            )
            fig.savefig("aw6040_fig7_correlation_beta.png", dpi=150)

        elif local_test == LocalTest.SHARPE_DIFFERENCE:
            report_table(portfolios.get_sharpe_difference_table(), name='table_sharpe_difference')

        elif local_test == LocalTest.FACTSHEET:
            figs = qis.factsheet(portfolios, kind='strategy_benchmark')
            qis.save_figs_to_pdf(
                figs=figs,
                file_name='aw6040_factsheet',
                orientation='portrait',
                local_path=str(Path.cwd()),
                add_current_date=False,
            )
            figs[0].savefig('aw6040_fig6_factsheet.png', dpi=150)
            print(f"factsheet pages: {len(figs)}")
            groups = pd.Series(
                {
                    "SPY": "US equities",
                    "TLT": "Nominal Treasuries",
                    "IEF": "Nominal Treasuries",
                    "TIP": "Inflation-linked Treasuries",
                    "GLD": "Gold",
                }
            )
            ungrouped = portfolios.compute_brinson_attribution()[0]
            grouped = portfolios.compute_brinson_attribution(group_data=groups)[0]
            report_table(ungrouped, "table_brinson_instruments")
            report_table(grouped, "table_brinson_asset_classes")
            grouped_figs = qis.factsheet(portfolios, kind="strategy_benchmark", group_data=groups)
            grouped_figs[-1].savefig("aw6040_brinson_asset_classes.png", dpi=150)
            qis.save_figs_to_pdf(
                # Hold pages one and two fixed for a direct comparison of the attribution page.
                figs=[*figs[:-1], grouped_figs[-1]],
                file_name="aw6040_grouped_factsheet",
                orientation="portrait",
                local_path=str(Path.cwd()),
                add_current_date=False,
            )
            # Read the same paired inference series used on page two of the monthly factsheet.
            inference = compute_paired_performance_inference(
                strategy=all_weather,
                comparison=portfolios.portfolio_datas[1],
                benchmark_prices=prices[[BENCHMARK]],
                time_period=qis.TimePeriod(),
                freq="ME",
                freq_beta="ME",
                factor_beta_span=36,
            )
            inference.rolling.set_index("date").to_csv("series_rolling_sharpe_difference.csv")
            inference.alpha_windows.to_csv("table_marginal_alpha.csv")

        else:
            raise ValueError(f"unknown case: {local_test!r}")

    plt.close('all')


def fetch_funding_rate(dates: pd.DatetimeIndex) -> pd.Series:
    """Read or cache Yahoo's annual ^IRX quote in decimal units, including a pre-start quote."""
    path = Path("ust_3m_irx.csv")
    if not path.exists() and (BUNDLED_DATA / path.name).is_file():
        path = BUNDLED_DATA / path.name
    if path.exists():
        rates = pd.read_csv(path, index_col="date", parse_dates=True, float_precision="round_trip")[
            "annual_rate"
        ]
    else:
        import yfinance as yf

        quotes = yf.download(
            "^IRX",
            start="2002-07-01",
            end=SAMPLE_END_EXCLUSIVE,
            ignore_tz=True,
            auto_adjust=True,
            progress=False,
        )
        rates = (quotes["Close"]["^IRX"].dropna() / 100.0).rename("annual_rate")
        rates.index.name = "date"
    if (
        rates.empty
        or rates.index.min() > dates[0]
        or rates.index.max() < dates[-1]
        or rates.isna().any()
        or not rates.index.is_unique
    ):
        raise ValueError("Expected dated annual ^IRX quotes spanning the ETF sample")
    if not path.exists():
        rates.to_csv(path)
    return rates.sort_index()


def run_leverage(prices: pd.DataFrame) -> None:
    """Compare the funded 2x target with the unlevered portfolios and SPY on the same dates."""
    rates = fetch_funding_rate(prices.index)
    weights = {asset: 2.0 * weight for asset, weight in ALL_WEATHER_WEIGHTS.items()}
    levered = backtest_static_allocation(
        prices, weights, ticker="All weather 2x", funding_rate=rates
    )
    navs = build_portfolios(prices).get_navs()
    navs["All weather 2x"] = levered.get_portfolio_nav()
    navs[BENCHMARK] = prices[BENCHMARK]
    navs.to_csv("navs_leverage.csv")

    params = qis.PerfParams(freq="ME", rates_data=rates)
    stats = [
        qis.PerfStat.PA_RETURN,
        qis.PerfStat.VOL,
        qis.PerfStat.SHARPE_EXCESS,
        qis.PerfStat.MAX_DD,
    ]
    table = qis.compute_ra_perf_table(prices=navs, perf_params=params)[
        [stat.to_str() for stat in stats]
    ].copy()
    annual = qis.compute_periodic_returns(prices=navs, freq="YE", date_format="%Y", add_total=False)
    table["2022 return"] = annual.loc["2022"]
    report_table(table, name="table_leverage")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    plotted = navs[["All weather 2x", BENCHMARK]]
    qis.plot_prices(
        prices=plotted,
        perf_stats_labels=stats[:3],
        perf_params=params,
        title="Growth of one dollar",
        ax=axes[0],
    )
    qis.plot_rolling_drawdowns(
        prices=plotted,
        dd_legend_type=qis.DdLegendType.SIMPLE,
        title="Drawdown from the previous peak",
        ax=axes[1],
    )
    fig.savefig("aw6040_fig8_leverage.png", dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--with-leverage',
        action='store_true',
        help='also run the 2x allocation funded at the three-month bill rate',
    )
    parser.add_argument(
        "--output-dir", type=Path, help="write tables and figures in this directory"
    )
    args = parser.parse_args()
    output_dir = Path(qis.get_output_path(args.output_dir)).resolve()
    os.chdir(output_dir)

    print(f"python: {sys.version.split()[0]}")
    for package in ['qis', 'yfinance', 'pandas', 'numpy', 'matplotlib']:
        try:
            print(f"{package}: {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            print(f"{package}: not installed")

    for case in LocalTest:
        print(f"\n===== {case.name} =====")
        run_local_test(local_test=case)

    if args.with_leverage:
        print('\n===== LEVERAGE =====')
        run_leverage(fetch_prices())
