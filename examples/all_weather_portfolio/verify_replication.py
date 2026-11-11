"""Cross-check the article's tables with independent calculations on the frozen inputs."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import qis_reference as reference


def check(actual, expected, tolerance=1e-10):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance, equal_nan=True)


def read_levels(file):
    return pd.read_csv(file, index_col=0, parse_dates=True)


def verify(results):
    data = Path(__file__).resolve().parent / "data"
    for name, digest in {
        "prices.csv": "8242adce116b6e4223f83914f2791b3355e064098945d8b95e114d750ff42bfd",
        "ust_3m_irx.csv": "79b85cd189513906ee5cbf7269610aa7e63c5bc37259002aa6f70c3231fec21d",
    }.items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"The recorded {name} snapshot differs")

    prices = read_levels(data / "prices.csv")
    navs = read_levels(results / "navs.csv")
    paired = reference.build_portfolios(prices)
    daily = navs.assign(SPY=prices.SPY)
    annual = daily.resample("YE").last()
    annual_returns = annual.pct_change()
    annual_returns.iloc[0] = annual.iloc[0] / daily.iloc[0] - 1
    stored = pd.read_csv(results / "table_annual_returns.csv", index_col=0)
    check(stored[daily.columns], annual_returns)
    month_end = daily.resample("ME").last()
    monthly = month_end.pct_change().dropna()
    quarterly = pd.concat([daily.iloc[:1], daily.resample("QE").last()]).pct_change().dropna()
    lower, upper = quarterly.SPY.quantile([0.16, 0.84])
    regimes = [
        quarterly.SPY <= lower,
        quarterly.SPY.between(lower, upper, inclusive="neither"),
        quarterly.SPY >= upper,
    ]
    check(
        pd.read_csv(results / "table_regime_average_quarterly_returns.csv", index_col=0),
        pd.concat([quarterly.loc[mask].mean() for mask in regimes], axis=1),
    )
    for freq, returns in (
        ("ME", monthly),
        ("QE", daily.resample("QE").last().pct_change().dropna()),
    ):
        ahead = returns["All weather"] > returns["60/40"]
        down = returns["60/40"] < 0
        stored = pd.read_csv(results / f"table_ahead_by_sign_{freq}.csv", index_col=0)
        check(stored["share all weather ahead"], [ahead[down].mean(), ahead[~down].mean()])

    logs = np.log(prices.resample("ME").last()).diff()
    corr = pd.read_csv(results / "table_spy_tlt_correlation.csv", index_col=0)
    for label, (start, end) in zip(
        corr.index, [("2005", "2012"), ("2013", "2021"), ("2022", "2026-09")]
    ):
        r = logs.loc[start:end, ["SPY", "TLT"]].dropna()
        check(corr.loc[label], [len(r), r.SPY.corr(r.TLT)])
    detail = read_levels(results / "series_correlation_beta.csv")

    # Use the previous month-end's displayed holdings beta on this month's log returns.
    beta = detail[navs.columns]
    alpha = paired.compute_rolling_benchmark_alpha(prices.SPY, freq_beta="ME", factor_beta_span=36)
    log_returns = np.log(month_end).diff()
    residual = log_returns[navs.columns] - beta.shift().mul(log_returns.SPY, axis=0)
    expected_alpha = 12 * residual.ewm(span=36, adjust=False).mean().where(residual.notna())
    pd.testing.assert_index_equal(alpha.index, expected_alpha.index, check_names=False)
    for name in navs:
        assert beta[name].first_valid_index() == pd.Timestamp("2005-08-31")
        assert alpha[name].first_valid_index() == pd.Timestamp("2005-09-30")
        check(alpha[name], expected_alpha[name])

    def ew(x):
        return x.ewm(span=36, adjust=True).mean()

    cross = ew(logs.TLT * logs.SPY) / np.sqrt(ew(logs.TLT**2) * ew(logs.SPY**2))
    check(detail["TLT - SPY"].dropna(), cross.reindex(detail["TLT - SPY"].dropna().index))
    tickers = list(reference.ALL_WEATHER_WEIGHTS)
    covariance = logs[tickers].cov()
    weights = np.full(4, 0.25)
    check(
        pd.read_csv(results / "table_risk_shares.csv", index_col=0)["risk share"],
        weights * (covariance @ weights) / (weights @ covariance @ weights),
    )

    turnover = pd.read_csv(results / "table_annual_turnover_costs.csv", header=[0, 1], index_col=0)
    attribution = pd.read_csv(results / "table_attribution.csv", header=[0, 1], index_col=0)
    for portfolio in paired.portfolio_datas:
        label = portfolio.ticker
        assets = portfolio.prices.columns
        units = portfolio.units
        traded = (units.diff().abs() * prices[assets]).sum(axis=1) / navs[label]
        yearly = traded.loc["2005":"2025"].resample("YE").sum()
        check(turnover[label]["Turnover"], yearly)
        check(turnover[label]["Trading costs"], yearly * 0.001)
        instrument_returns = (units.shift() * prices[assets].ffill().diff()).div(
            navs[label].shift(), axis=0
        )
        for year in [*map(str, reference.ATTRIBUTION_YEARS), "Full sample"]:
            sample = (
                instrument_returns.iloc[1:]
                if year == "Full sample"
                else instrument_returns.loc[year]
            )
            growth = (1 + sample.sum(axis=1)).cumprod().shift(fill_value=1.0)
            contributions = sample.mul(growth, axis=0).sum()
            check(attribution.loc[assets, (year, label)], contributions)
            check(
                attribution.at["Total", (year, label)],
                (1 + sample.sum(axis=1)).prod() - 1,
            )
        # Independent weighted regression on the three mature dates quoted in the article.
        excess = logs - logs.ewm(span=36, adjust=False).mean()
        slopes = (
            excess.mul(excess.SPY, axis=0)
            .ewm(span=36, adjust=True)
            .mean()
            .div(ew(excess.SPY**2), axis=0)
        )
        w = portfolio.weights.resample("ME").last()
        dates = ["2021-12-31", "2023-12-31", "2026-09-30"]
        estimate = (slopes[assets] * w).sum(axis=1)
        check(detail.loc[dates, label], estimate.loc[dates], tolerance=0.0001)

    rolling = read_levels(results / "series_rolling_sharpe_difference.csv")
    for end, reported in rolling.estimate.items():
        sample = monthly.loc[
            (monthly.index > end - pd.DateOffset(years=5)) & (monthly.index <= end)
        ]
        sharpe = np.sqrt(12) * sample.mean() / sample.std(ddof=0)
        check(reported, sharpe["All weather"] - sharpe["60/40"])
    grouped = pd.read_csv(results / "table_brinson_asset_classes.csv", index_col=0)
    individual = pd.read_csv(results / "table_brinson_instruments.csv", index_col=0)
    for group, assets in {
        "US equities": ["SPY"],
        "Nominal Treasuries": ["TLT", "IEF"],
        "Inflation-linked Treasuries": ["TIP"],
        "Gold": ["GLD"],
    }.items():
        check(
            grouped.at[group, "Total\nActive"],
            individual.loc[assets, "Total\nActive"].sum(),
        )
        check(
            grouped.at[group, "Total\nActive"],
            grouped.loc[group, ["Asset\nAllocation", "Instrument\nSelection"]].sum(),
        )
    gross = attribution["Full sample"].loc["Total"]
    check(grouped.at["Total Sum", "Total\nActive"], gross["All weather"] - gross["60/40"])

    funded = read_levels(results / "navs_leverage.csv")
    table = pd.read_csv(results / "table_leverage.csv", index_col=0)
    rates = read_levels(data / "ust_3m_irx.csv")["annual_rate"]
    months = funded.resample("ME").last()
    years = (months.index[-1] - months.index[0]).days / 365.25
    vol = np.log(months).diff().std() * np.sqrt(12)
    rf = (
        rates.reindex(months.index, method="ffill").shift()
        * months.index.to_series().diff().dt.days
        / 365
    )
    excess = months.pct_change().sub(rf, axis=0).dropna()
    check(table["P.a. return"], (funded.iloc[-1] / funded.iloc[0]) ** (1 / years) - 1)
    check(table["Vol"], vol)
    check(table["Ex Sharpe"], ((1 + excess).prod() ** (1 / years) - 1) / vol)
    check(table["Max DD"], (funded / funded.cummax() - 1).min())
    check(
        table["2022 return"],
        months.loc["2022"].iloc[-1] / months.loc["2021"].iloc[-1] - 1,
    )
    print(
        "Passed: frozen inputs, annual returns, regimes, win rates, risk shares, turnover, costs,"
    )
    print(
        "wealth-linked contributions, correlations, quoted betas, rolling alpha and its warm-up, "
        "rolling Sharpe, "
        "Brinson and leverage."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", type=Path, nargs="?", default=Path(__file__).resolve().parent / "results"
    )
    verify(parser.parse_args().results)
