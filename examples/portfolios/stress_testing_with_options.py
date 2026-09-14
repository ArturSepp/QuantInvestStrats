"""Five yfinance stocks and ten short VOP-priced options under a four-factor EWMA model.

Run python -m examples.portfolios.stress_testing_with_options --output-dir <fresh directory>.
The first run downloads Yahoo data; subsequent runs replay a hashed local cache.
--refresh explicitly replaces the cache. No Bloomberg access or client data is used.
"""
import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qis
# User-facing example dependency only; VOP is never imported by the qis package.
import vanilla_option_pricers as vop  # noqa: TID251


STOCKS = ("AAPL", "MSFT", "AMZN", "GOOGL", "NVDA")
FACTORS = ("SPY", "TLT", "GLD", "USO")
AS_OF = "2025-12-31"
START = "2015-01-01"
SPAN = 52
ANNUALISATION = 52.0


def load_prices(cache_dir, as_of=AS_OF, refresh=False):
    """Download explicit raw/adjusted closes once, then replay their checked CSV bytes."""
    import yfinance as yf

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cut = pd.Timestamp(as_of).normalize()
    path = cache / f"yahoo_{START}_{cut.date()}.csv"
    record = path.with_suffix(".json")
    expected = {"tickers": list(STOCKS + FACTORS), "start": START,
                "as_of": str(cut.date()), "auto_adjust": False, "repair": False}
    if refresh or not path.exists():
        if path.exists() != record.exists() and not refresh:
            raise ValueError("Incomplete cache; use --refresh to rebuild it")
        yf.set_tz_cache_location(str(cache / "yfinance"))
        frame = yf.download(
            list(STOCKS + FACTORS), start=START, end=str((cut + pd.Timedelta(days=1)).date()),
            auto_adjust=False, repair=False, actions=False, threads=False, progress=False,
            group_by="column", timeout=30,
        )
        if frame is None or frame.empty:
            raise ValueError("Yahoo returned no prices; retry later or use an existing cache")
        frame = frame.loc[:, pd.MultiIndex.from_product(
            [["Close", "Adj Close"], STOCKS + FACTORS])].loc[:cut]
        if frame.notna().sum().min() < 260:
            raise ValueError("Every stock and factor needs at least 260 observed daily prices")
        frame.to_csv(path)
        record.write_text(json.dumps({**expected, "yfinance": version("yfinance"),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    metadata = json.loads(record.read_text(encoding="utf-8"))
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Cache request metadata differs; use the matching cache or --refresh")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Cached prices changed; use --refresh for an intentional new download")
    frame = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True,
                        float_precision="round_trip")
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    frame = frame.loc[:cut].sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate price dates")
    frame = frame.dropna()
    if frame.empty or (cut - frame.index[-1]).days > 4 or (frame <= 0).any().any():
        raise ValueError("Missing, stale or nonpositive common price panel")
    return frame, metadata


def fit_risk_model(prices):
    """Fit joint EWMA betas and annual factor/residual moments on weekly log returns."""
    adjusted = prices["Adj Close"]
    fit = qis.estimate_ewm_factor_model(
        asset_prices=adjusted[list(STOCKS)], factor_prices=adjusted[list(FACTORS)],
        freq="W-WED", span=SPAN, mean_adj_type=qis.MeanAdjType.NONE,
    )
    date = fit.x.index[fit.x.index <= adjusted.index[-1]].max()
    x, y = fit.x.loc[:date], fit.y.loc[:date]
    if len(x) < 3 * SPAN or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Need at least three EWMA spans of complete weekly observations")
    beta = fit.get_loadings_at_date(date).T.loc[list(STOCKS), list(FACTORS)]
    residuals = y - x @ beta.T
    factor_covar = pd.DataFrame(
        ANNUALISATION * qis.compute_ewm_covar(x.to_numpy(), span=SPAN),
        index=FACTORS, columns=FACTORS,
    )
    residual_var = pd.Series(np.diag(
        ANNUALISATION * qis.compute_ewm_covar(residuals.to_numpy(), span=SPAN)), index=STOCKS)
    # Assemble the assigned factor model; portfolio risk is computed exclusively by RiskModel.
    covariance = beta @ factor_covar @ beta.T + pd.DataFrame(
        np.diag(residual_var), index=STOCKS, columns=STOCKS)
    risk_model = qis.RiskModel(
        covar={date: covariance}, factor_loadings={date: beta},
        factor_covar={date: factor_covar}, residual_vars={date: residual_var},
    )
    raw_var = np.diag(ANNUALISATION * qis.compute_ewm_covar(y.to_numpy(), span=SPAN))
    diagnostics = pd.DataFrame({"r2": 1.0 - residual_var / raw_var,
                                "weekly_observations": len(x)}, index=STOCKS)
    history = qis.to_returns(adjusted[list(FACTORS)], freq="ME", is_log_returns=True,
                             drop_first=True).loc[:adjusted.index[-1]].dropna()
    return risk_model, date, history, diagnostics, x, y


@dataclass(frozen=True)
class BsmOptionPayoff:
    """Example-owned, USD European option mark with signed exchange-contract quantity.

    Attributes:
        underlying: Yahoo stock symbol identifying the actual USD quote.
        option_type: C or P; the direction is carried by contracts, not the type.
        strike: Strike in USD per share.
        expiry: Synthetic contract expiry date, strictly after valuation_date.
        valuation_date: Frozen mark date; no time elapses during instantaneous stresses.
        volatility: Assumed annual lognormal implied volatility, fixed under shocks.
        contracts: Signed number of contracts; negative for the short overlay.
        multiplier: Shares per contract, normally 100.
        rate: Continuously compounded USD rate, held fixed.
        dividend_yield: Continuously compounded dividend yield, held fixed.
    """
    underlying: str
    option_type: str
    strike: float
    expiry: pd.Timestamp
    valuation_date: pd.Timestamp
    volatility: float
    contracts: int
    multiplier: int = 100
    rate: float = 0.04
    dividend_yield: float = 0.0
    implementation_id = "qis.example.vop_european_option.v1"
    coverage = "European BSM mark; fixed IV/rate/TTM; no American exercise or margin model."
    boundary_policy = "Smooth BSM delta with positive TTM and volatility; no expiry crossing."

    def __post_init__(self):
        """Reject invalid pricing terms before any report computation."""
        if self.option_type not in ("C", "P") or self.ttm <= 0:
            raise ValueError("Use C/P and an expiry after the frozen valuation date")
        if self.strike <= 0 or self.volatility <= 0 or self.multiplier <= 0:
            raise ValueError("Strike, volatility and multiplier must be positive")

    @property
    def ttm(self):
        """Return ACT/365 time remaining at the frozen valuation date."""
        return (self.expiry - self.valuation_date).days / 365.0

    def unit_prices(self, spots):
        """Call the compiled VOP forward-grid pricer, returning USD per underlying share."""
        forwards = np.asarray(spots, dtype=float) * np.exp(
            (self.rate - self.dividend_yield) * self.ttm)
        return vop.compute_bsm_forward_grid_prices(
            ttm=self.ttm, forwards=forwards, strike=self.strike, vol=self.volatility,
            optiontype=self.option_type, discfactor=np.exp(-self.rate * self.ttm),
        )

    def spot_greeks(self, spot):
        """Convert VOP discounted forward delta and undiscounted forward gamma to spot Greeks."""
        carry = np.exp((self.rate - self.dividend_yield) * self.ttm)
        forward = float(spot * carry)
        discount = np.exp(-self.rate * self.ttm)
        delta = carry * vop.compute_bsm_vanilla_delta(
            self.ttm, forward, self.strike, self.volatility, self.option_type, discount)
        gamma = discount * carry**2 * vop.compute_bsm_vanilla_gamma(
            self.ttm, forward, self.strike, self.volatility)
        return delta, gamma

    def evaluate(self, context: qis.PayoffContext) -> pd.Series:
        """Return signed scenario option marks; all quotes in this example are USD."""
        if (context.reference_currency != "USD"
                or context.quote_currencies[self.underlying] != "USD"):
            raise ValueError("This example payoff supports USD quotes and reference currency only")
        quotes = context.quotes[self.underlying]
        values = self.contracts * self.multiplier * self.unit_prices(quotes.to_numpy())
        return pd.Series(values, index=quotes.index)

    def _jacobian(self, context, spot):
        """Apply the spot-to-log-quote chain rule and the public shared-response mapping."""
        delta, _ = self.spot_greeks(spot)
        dollars = self.contracts * self.multiplier * delta * spot
        return dollars * context.quote_response_jacobian.loc[self.underlying]

    def response_jacobian(self, context: qis.PayoffContext) -> pd.Series:
        """Return current signed dollar delta by underlying response."""
        return self._jacobian(context, context.baseline_quotes[self.underlying])

    def scenario_response_jacobian(self, context: qis.PayoffContext) -> pd.Series:
        """Recompute delta at the stressed spot for QIS conditional risk bands."""
        return self._jacobian(context, context.quotes.iloc[0][self.underlying])


def build_portfolio(prices, model, risk_date):
    """Buy round lots of five stocks and sell one call/put line per stock."""
    date = prices.index[-1]
    spots = prices["Close"].iloc[-1]
    returns = qis.to_returns(prices["Adj Close"][list(STOCKS)], is_log_returns=True,
                             drop_first=True)
    realised = qis.compute_ewm_vol(returns, span=63, annualize=True,
                                   annualization_factor=252).iloc[-1]
    quotes = {stock: qis.Underlying(stock, float(spots[stock]), "USD", stock,
                                    qis.ResponseBasis.LOCAL) for stock in STOCKS}
    holdings, inventory = [], []
    for i, stock in enumerate(STOCKS):
        spot = float(spots[stock])
        lots = int(2_000_000 / (100 * spot))
        if lots < 1:
            raise ValueError("Stock price exceeds the per-name teaching budget")
        shares = 100 * lots
        holdings.append(qis.PortfolioHolding(
            f"{stock} US Equity", stock, shares * spot,
            (qis.InstrumentLeg(qis.InstrumentType.DELTA_1, stock, shares),),
            metadata={"short_name": stock, "sleeve": "Stocks"},
        ))
        month = date.to_period("M") + (2, 3, 4, 5, 8)[i]
        expiry = pd.date_range(month.start_time, month.end_time, freq="W-FRI")[2]
        for kind, moneyness, quantity, skew in (
            ("C", (1.03, 1.05, 1.07, 1.04, 1.08)[i], -lots, 0.0),
            ("P", (0.97, 0.95, 0.93, 0.96, 0.92)[i], -int(1.5 * lots), 0.04),
        ):
            strike = float(5 * np.round(spot * moneyness / 5))
            option = BsmOptionPayoff(stock, kind, strike, expiry, date,
                                    float(max(0.15, 1.15 * realised[stock]) + skew), quantity)
            ticker = f"{stock} US {expiry:%m/%d/%y} {kind}{strike:g} Equity"
            unit_price = float(option.unit_prices(np.array([spot]))[0])
            delta, gamma = option.spot_greeks(spot)
            mark = quantity * 100 * unit_price
            holdings.append(qis.PortfolioHolding(
                ticker, ticker, mark, payoff=option,
                metadata={"short_name": f"{stock} {expiry:%b} {kind}{strike:g}",
                          "sleeve": "Short calls" if kind == "C" else "Short puts",
                          "expiry": str(expiry.date()), "strike": str(strike),
                          "contracts": str(quantity), "multiplier": "100",
                          "iv": str(option.volatility)},
            ))
            inventory.append({"ticker": ticker, "underlying": stock, "type": kind,
                "expiry": str(expiry.date()), "strike": strike, "contracts": quantity,
                "multiplier": 100, "iv": option.volatility, "unit_price": unit_price,
                "mtm_usd": mark, "spot": spot, "unit_spot_delta": delta,
                "dollar_delta": quantity * 100 * delta * spot,
                "dollar_gamma": quantity * 100 * gamma * spot**2})
    denominator = sum(holding.observed_mtm for holding in holdings)
    portfolio = qis.InstrumentPortfolio(
        tuple(holdings), quotes, model, risk_date, date, "USD", denominator,
        denominator_label="Net marked portfolio value",
    )
    return portfolio, pd.DataFrame(inventory).set_index("ticker")


def scenario_inputs():
    """Use independent requests, joint conditional comparisons and four correlated grids."""
    anchors = {f"{factor} {bump:+.0%}": {factor: bump}
               for factor, bumps in (("SPY", (-.3, -.2, -.1, .1, .2, .3)),
                                     ("TLT", (-.1, .1)), ("GLD", (-.2, .2)),
                                     ("USO", (-.3, .3))) for bump in bumps}
    anchors["No move"] = {factor: 0.0 for factor in FACTORS}
    requests = qis.StressScenarios(pd.DataFrame.from_dict(anchors, orient="index"),
                                   convention=qis.ShockConvention.SIMPLE)
    grids = {}
    for factor in FACTORS:
        limit = 30 if factor == "SPY" else 20
        axis = pd.Index(np.arange(-limit, limit + 1) / 100.0, name=f"{factor} simple return")
        grids[factor] = qis.StressScenarios(
            pd.DataFrame({factor: axis.to_numpy()}, index=axis),
            mode=qis.ScenarioMode.CONDITIONAL, convention=qis.ShockConvention.SIMPLE,
        )
    return requests, grids


def verify_example(portfolio, result, inventory, x, y):
    """Independently check EWMA least squares, option Greeks, parity and portfolio attribution."""
    decay = 1.0 - 2.0 / (SPAN + 1)
    weights = (1 - decay) * decay**np.arange(len(x) - 1, -1, -1)
    root_w = np.sqrt(weights)[:, None]
    reference_beta = np.linalg.lstsq(x.to_numpy() * root_w, y.to_numpy() * root_w, rcond=None)[0]
    np.testing.assert_allclose(result.factor_loadings, reference_beta.T, atol=1e-10)
    reference_residuals = y.to_numpy() - x.to_numpy() @ reference_beta
    np.testing.assert_allclose(result.residual_variances,
                               ANNUALISATION * weights @ reference_residuals**2, rtol=1e-10)
    assert len(portfolio.holdings) == 15
    assert inventory.type.value_counts().to_dict() == {"C": 5, "P": 5}
    assert inventory.contracts.lt(0).all() and inventory.dollar_gamma.lt(0).all()
    zero = pd.Series(0.0, index=FACTORS)
    np.testing.assert_allclose(portfolio.get_pnl(zero), 0.0, atol=1e-8)
    for holding in portfolio.holdings:
        if holding.payoff is None:
            continue
        option = holding.payoff
        spot = portfolio.underlyings[option.underlying].spot0
        h = spot * 1e-3
        values = option.unit_prices(np.array([spot - h, spot, spot + h]))
        delta, gamma = option.spot_greeks(spot)
        np.testing.assert_allclose(delta, (values[2] - values[0]) / (2*h), atol=2e-5)
        np.testing.assert_allclose(gamma, (values[2] - 2*values[1] + values[0]) / h**2,
                                   rtol=5e-4, atol=1e-6)
        call = replace(option, option_type="C").unit_prices(np.array([spot]))[0]
        put = replace(option, option_type="P").unit_prices(np.array([spot]))[0]
        parity = spot * np.exp(-option.dividend_yield * option.ttm) - option.strike * np.exp(
            -option.rate * option.ttm)
        np.testing.assert_allclose(call - put, parity, atol=1e-8)
    for factor in FACTORS:
        z = zero.copy()
        z[factor] = 1e-5
        derivative = (portfolio.get_pnl(z) - portfolio.get_pnl(-z)) / 2e-5
        np.testing.assert_allclose(derivative, result.holding_factor_exposures[factor],
                                   rtol=2e-5, atol=0.1)
    for name, value in result.valuations.items():
        np.testing.assert_allclose(result.attribution[name].sum(axis=1),
                                   value.portfolio_pnl, atol=1e-7)
    return {"status": "passed", "holdings": 15, "calls": 5, "puts": 5,
            "checks": ["weighted least-squares betas", "weighted residual moments",
                       "zero-shock marks", "spot delta/gamma finite differences",
                       "put-call parity", "factor delta finite differences", "P&L attribution"]}


def convexity_exhibit(portfolio, result, inventory):
    """Plot full repricing, frozen-delta comparison and option-sleeve contributions through QIS."""
    grid = result.grids["SPY"]
    nav = portfolio.reporting_denominator
    linear = grid.factor_log_shocks @ result.factor_exposures / nav
    curves = pd.DataFrame({"Full option repricing": grid.portfolio_pnl / nav,
                           "Current log-delta approximation": linear})
    sleeves = pd.DataFrame({name: grid.pnl[[holding.holding_id for holding in portfolio.holdings
                            if holding.metadata["sleeve"] == name]].sum(axis=1) / nav
                            for name in ("Stocks", "Short calls", "Short puts")})
    stock_options = pd.DataFrame({stock: grid.pnl[inventory.index[
        inventory.underlying.eq(stock)]].sum(axis=1) / nav for stock in STOCKS})
    table = inventory.groupby("underlying").agg(
        short_contracts=("contracts", "sum"), delta_usd=("dollar_delta", "sum"),
        gamma_usd=("dollar_gamma", "sum")).reindex(STOCKS)
    table["delta_usd"] = table.delta_usd.map(lambda x: f"{x/1e6:.2f}")
    table["gamma_usd"] = table.gamma_usd.map(lambda x: f"{x/1e6:.2f}")
    table.columns = ["Short contracts", "Option delta\n(USDm)", "Gamma x spot²\n(USDm)"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, data, title in ((axes[0, 0], curves, "Correlated SPY: value versus current delta"),
                            (axes[0, 1], sleeves, "Stock and short-option contributions"),
                            (axes[1, 0], stock_options, "Short-option contribution by underlying")):
        qis.plot_line(data, ax=ax, title=title, xlabel="SPY simple return",
                      ylabel="P&L / current net portfolio value", xvar_format="{:.0%}",
                      yvar_format="{:.0%}", linewidth=2, fontsize=10)
    qis.plot_df_table(table, ax=axes[1, 1], fontsize=10,
                      title="Short-option overlay: signed current Greeks")
    fig.suptitle("Five stocks + ten short options: the cost of negative convexity", fontsize=16)
    fig.text(.06, .025, "Other ETFs follow EWMA conditional co-moves. Fixed IV, rates and TTM. "
             "Gamma is signed option gamma multiplied by spot squared; premiums are model prices.",
             fontsize=10)
    fig.tight_layout(rect=(.02, .06, .98, .94))
    return fig, curves.join(sleeves), stock_options


def run_example(cache_dir, output_dir=None, as_of=AS_OF, refresh=False):
    """Compute the reproducible book; optionally save a standard report and convexity exhibit."""
    output = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if output is not None and output.exists():
        raise FileExistsError("Use a fresh example output directory")
    prices, source = load_prices(cache_dir, as_of, refresh)
    model, date, history, diagnostics, x, y = fit_risk_model(prices)
    portfolio, inventory = build_portfolio(prices, model, date)
    requests, grids = scenario_inputs()
    result = qis.run_portfolio_stress_test(
        portfolio, requests, history, grids,
        qis.StressTestConfig(horizon_years=1/12, confidence=.95),
    )
    verification = verify_example(portfolio, result, inventory, x, y)
    fig, curves, contributions = convexity_exhibit(portfolio, result, inventory)
    if output is not None:
        output.mkdir(parents=True)
        appendix = inventory[["expiry", "strike", "contracts", "iv", "unit_price"]].copy()
        for column in ("strike", "unit_price"):
            appendix[column] = appendix[column].map(lambda value: f"{value:.2f}")
        appendix.iv = appendix.iv.map(lambda value: f"{value:.1%}")
        config = qis.StressReportConfig(
            title="Five-stock short-option portfolio", model_name="EWMA",
            model_label="SPY / TLT / GLD / USO; weekly span 52", selected_grids=FACTORS,
            response_diagnostics=diagnostics, write_previews=True,
            appendix_table=appendix, appendix_title="Synthetic option terms and VOP prices",
            appendix_notes=(
                "Bloomberg-style IDs are synthetic teaching contracts, not fetched quotes.",
                "100 shares/contract. Covered calls; puts are 1.5x stock lots, rounded down.",
                "Fixed-IV European proxy; American exercise, margin and volatility shocks omitted.",
            ),
            notes=("Observed Yahoo stock/ETF history; synthetic option positions and model prices.",
                   "Risk uses adjusted closes; option marks use unadjusted closing spot quotes.",
                   "Local Gaussian bands exclude gamma, vega, parameter and regime uncertainty."),
        )
        artifacts = qis.generate_portfolio_stress_report(result, output / "report", config)
        fig.savefig(output / "short_convexity.pdf")
        fig.savefig(output / "short_convexity.png", dpi=150)
        inventory.to_csv(output / "option_inventory.csv")
        diagnostics.to_csv(output / "fit_diagnostics.csv")
        curves.to_csv(output / "spy_convexity.csv")
        contributions.to_csv(output / "spy_option_contributions.csv")
        prices.to_csv(output / "source_prices.csv")
        provenance = {"source": source, "valuation_date": str(portfolio.valuation_date.date()),
            "risk_date": str(date.date()), "span_weeks": SPAN, "annualisation": ANNUALISATION,
            "reference_currency": "USD", "net_marked_value": portfolio.reporting_denominator,
            "qis_distribution_version": version("qis"),
            "qis_file": str(Path(qis.__file__).resolve()),
            "qis_source_sha256": {str(path.relative_to(Path(qis.__file__).parent)):
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(qis.__file__).parent.rglob("*.py"))},
            "vop_version": version("vanilla-option-pricers"),
            "example_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "verification": verification}
        (output / "example_provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8")
        print("Report:", artifacts.pdf_path)
        print("Convexity exhibit:", output / "short_convexity.pdf")
    plt.close(fig)
    print(f"Valuation {portfolio.valuation_date.date()}; risk {date.date()}; "
          f"net marked value USD {portfolio.reporting_denominator:,.0f}; "
          "5 stocks + 5 short calls + 5 short puts; verification passed.")
    print(curves.loc[[-.3, -.2, -.1, 0., .1, .2, .3]].to_string(float_format="{:.2%}".format))
    return portfolio, result, inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    default_cache = cache_root / "qis/options_stress"
    parser.add_argument("--cache-dir", type=Path, default=default_cache)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--as-of", default=AS_OF)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    run_example(args.cache_dir, args.output_dir, args.as_of, args.refresh)
