"""Offline observed-index example: point/log fits, USD/EUR responses and account report.

Run: python -m examples.portfolios.instrument_portfolio_stress --case vix
Optional: --output-dir <fresh directory> writes two USD report packs and a comparison CSV.
All observations are synthetic, derived from the standard seeded QIS dataset. This is
an unpenalised EWLS teaching example, not a MAC estimate or an options sizing model.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import qis

from qis.datasets.synthetic import generate_synthetic_prices
from qis.perfstats.config import ReturnTypes
from qis.utils.regression import estimate_ewma_alpha_beta_hac


def run_example(output_dir=None):
    """Fit both index transforms and verify invariant regressors in USD and EUR."""
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
    prices = generate_synthetic_prices(start="2015-12-01", end="2025-12-31",
                                      apply_quirks=False).resample("ME").last()
    # Map the seeded equity shocks into positive, varying index levels. No rebasing,
    # volatility targeting, funding or FX operation is applied to the resulting index.
    seed = np.log(prices.SEQ_US).diff().fillna(0.)
    levels = (22. * np.exp(-5. * seed)).rename("VIX").to_frame()
    points = levels.VIX.diff()
    noise = np.log(prices.SBD_TSY).diff()
    usd_log = .003 - .002 * points + .15 * noise
    usd_nav = 100. * np.exp(usd_log.fillna(0.).cumsum())
    eurusd = 1.1 * (prices.SBD_IG / prices.SBD_IG.iloc[0])
    responses = {}
    for currency, nav, rate in (("USD", usd_nav, .03), ("EUR", usd_nav / eurusd, .02)):
        total = nav.pct_change(fill_method=None)
        cash = pd.Series(rate, index=nav.index)
        responses[currency] = np.log1p(qis.compute_excess_returns(total, cash)).dropna()
    assert not np.allclose(responses["USD"], responses["EUR"])
    records = []
    for kind in (ReturnTypes.DIFFERENCE, ReturnTypes.LOG):
        mapping = {"VIX": kind}
        x = qis.to_factor_returns(levels, mapping, is_first_zero=False).VIX.dropna()
        for currency, response in responses.items():
            # Currency affects only the already converted asset response.
            same_x = qis.to_factor_returns(levels, mapping, is_first_zero=False).VIX.dropna()
            pd.testing.assert_series_equal(x, same_x)
            fit = estimate_ewma_alpha_beta_hac(x, response.to_frame("fund"), span=36)
            beta, alpha = float(fit.beta.iloc[0]), float(fit.alpha.iloc[0])
            predicted = alpha + beta * x
            score = qis.prediction_r_squared(response, predicted, weights=fit.weights)
            np.testing.assert_allclose(score, fit.r_squared.iloc[0], atol=1e-12)
            assert beta < 0.
            records.append({"currency": currency, "transform": kind.name, "beta": beta,
                            "portfolio_r2": score, "observations": len(x),
                            "local_beta_per_point": beta if kind is ReturnTypes.DIFFERENCE
                            else beta / levels.VIX.iloc[-1]})
            if currency == "EUR":
                continue
            date = x.index[-1]
            loading = pd.DataFrame({"VIX": [beta]}, index=["fund"])
            cov = x.to_frame().cov() * 12.
            residual = pd.Series({"fund": float((response - predicted).var() * 12.)})
            model = qis.RiskModel(
                {date: loading @ cov @ loading.T + np.diag(residual)},
                {date: loading}, {date: cov}, {date: residual}, factor_return_types=mapping)
            nav = 1_000_000.
            ledger = pd.DataFrame({"mtm": [nav], "role": [qis.PositionRole.ASSET],
                "asset_class": [qis.AccountAssetClass.EQUITY], "currency": [currency]},
                index=["fund"])
            account = qis.PortfolioAccounting(ledger, net_equity=nav)
            holding = qis.PortfolioHolding("fund", "Synthetic index-sensitive fund", nav,
                (qis.InstrumentLeg(qis.InstrumentType.DELTA_1, "quote", nav / 100.),))
            portfolio = qis.InstrumentPortfolio((holding,),
                {"quote": qis.Underlying("quote", 100., currency, "fund",
                                         qis.ResponseBasis.REFERENCE)},
                model, date, date, currency, nav, account.denominator_label, accounting=account)
            spot = float(levels.VIX.iloc[-1])

            def changes(bumps):
                return bumps if kind is ReturnTypes.DIFFERENCE else np.log1p(bumps / spot)

            request = qis.StressScenarios(pd.DataFrame(
                {"VIX": changes(np.array([0., 5., 10., 20.]))},
                index=["No move", "+5 points", "+10 points", "+20 points"]),
                convention=qis.ShockConvention.MODEL)
            bumps = np.linspace(-min(10., .8 * spot), 30., 81)
            grid = qis.StressScenarios(pd.DataFrame({"VIX": changes(bumps)}, index=bumps),
                convention=qis.ShockConvention.MODEL, axis_label="VIX change (points)")
            selection = qis.HistoricalScenarioSelection("VIX", "2016-01-01", 10,
                                                        ascending=False, return_type=kind)
            result = qis.run_portfolio_stress_test(portfolio, request, x.to_frame(),
                {"VIX": grid}, qis.StressTestConfig(historical_selection=selection))
            expected = nav * np.expm1(beta * request.anchors.VIX)
            np.testing.assert_allclose(result.valuations["requested"].portfolio_pnl, expected)
            np.testing.assert_allclose(result.valuations["conditional"].portfolio_pnl, expected)
            if output_dir is not None:
                artifacts = qis.generate_portfolio_stress_report(result, output_dir / kind.name,
                    qis.StressReportConfig(title="Synthetic observed-index account",
                        model_name=f"VIX {kind.name}", portfolio_r2=score,
                        notes=("Synthetic data. EWLS with intercept; 36 monthly observations span.",
                               "VIX index is unconverted; asset responses use their own currency.",
                               "Option pricing and hedge sizing are outside this example.")))
                print(artifacts.pdf_path)
    comparison = pd.DataFrame(records).set_index(["currency", "transform"])
    assert comparison.loc[("USD", "DIFFERENCE"), "portfolio_r2"] > .95
    if output_dir is not None:
        comparison.to_csv(output_dir / "portfolio_comparison.csv")
    print(comparison.to_string())
    return comparison
