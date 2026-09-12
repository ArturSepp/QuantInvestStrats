"""Consumer-owned terminal knockout using the public HoldingPayoff extension.

Run: python -m examples.portfolios.composite_payoff_stress
Pass --output-dir <fresh directory> to export a standard report.
This synthetic terminal proxy ignores fixing paths, accrued units and settlement mechanics.
"""
import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import qis
from examples.portfolios.instrument_portfolio_stress import (
    build_model_and_history, build_portfolio, report_config, scenario_inputs,
)


@dataclass(frozen=True)
class TerminalKnockoutAccumulator:
    """Wrap public vanilla legs; no private QIS or consumer-library imports are needed.

    Attributes:
        vanilla: One original holding containing the continuing call/put decomposition.
        quote_id: Actual local quote checked against the barrier.
        knockout: Positive favorable-side terminal price, in the quote's local currency.
    """

    vanilla: qis.PortfolioHolding
    quote_id: str
    knockout: float
    implementation_id = "example.terminal_knockout_accumulator.v1"
    coverage = "Terminal intrinsic proxy; no accrued fixings, path-state or physical delivery."
    boundary_policy = "Zero payoff and local derivative at/above KO; wrapped strike-side policy."

    def __post_init__(self):
        """Reject inconsistent example terms before portfolio construction."""
        if not np.isfinite(self.knockout) or self.knockout <= 0:
            raise ValueError("knockout must be positive and finite")
        if not self.vanilla.legs or any(
            leg.underlying_id != self.quote_id
            or leg.instrument_type not in (qis.InstrumentType.CALL, qis.InstrumentType.PUT)
            for leg in self.vanilla.legs
        ):
            raise ValueError("wrapped holding must contain vanilla options on the barrier quote")

    def evaluate(self, context: qis.PayoffContext) -> pd.Series:
        """Use public vanilla valuation, which already performs local-currency conversion."""
        live = context.quotes[self.quote_id] < self.knockout
        return self.vanilla.model_payoff(context).where(live, 0.0)

    def response_jacobian(self, context: qis.PayoffContext) -> pd.Series:
        """Delegate the live-side Jacobian; at and beyond KO all proxy sensitivities are zero."""
        if context.baseline_quotes.loc[self.quote_id] < self.knockout:
            return self.vanilla.response_jacobian(context)
        return pd.Series(0.0, index=context.response_log_shocks.columns)


def run_example(output_dir=None):
    """Replace one vanilla holding with a composite and verify its factor derivative."""
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError("Use a fresh example output directory")
    model, date, history = build_model_and_history()
    portfolio = build_portfolio(model, date, derivatives=True)
    original = next(h for h in portfolio.holdings if h.holding_id == "accumulator")
    # Move away from the strike kink so a central finite difference has a unique reference.
    vanilla = replace(original, legs=tuple(replace(leg, strike=95.0) for leg in original.legs))
    custom = TerminalKnockoutAccumulator(vanilla, "share_eur", 115.0)
    assert isinstance(custom, qis.HoldingPayoff)
    holding = replace(
        vanilla, legs=(), payoff=custom,
        metadata={**vanilla.metadata, "knockout": "115 EUR", "strike": "95 EUR",
                  "policy": "terminal knockout; equality is knocked out"},
    )
    portfolio = replace(portfolio, holdings=tuple(
        holding if h.holding_id == holding.holding_id else h for h in portfolio.holdings
    ))
    requests, grids = scenario_inputs(model, date)
    result = qis.run_portfolio_stress_test(portfolio, requests, history, grids)
    # At Equity +40%, the EUR quote exceeds KO and the remaining intrinsic payoff is zero.
    # Current intrinsic is 2000 * (100 - 95) * EURUSD 1.2 = USD 12,000.
    assert result.valuations["requested"].pnl.loc["Equity up", "accumulator"] == -12_000.0
    np.testing.assert_allclose(result.valuations["requested"].pnl.loc["No move"], 0.0, atol=1e-10)
    epsilon = 1e-6
    finite_difference = {}
    for factor in model.factor_loadings[date].columns:
        bump = pd.Series(0.0, index=model.factor_loadings[date].columns)
        bump.loc[factor] = epsilon
        finite_difference[factor] = (
            portfolio.get_pnl(bump).loc[holding.holding_id]
            - portfolio.get_pnl(-bump).loc[holding.holding_id]
        ) / (2 * epsilon)
    np.testing.assert_allclose(
        pd.Series(finite_difference), result.holding_factor_exposures.loc[holding.holding_id],
        rtol=1e-7, atol=1e-4,
    )
    if output_dir is not None:
        config = report_config(portfolio, "Synthetic terminal-knockout portfolio")
        config = replace(config, appendix_notes=config.appendix_notes + (
            "One accumulator uses a consumer-owned terminal KO wrapper at EUR 115, strike EUR 95.",
        ))
        artifact = qis.generate_portfolio_stress_report(result, Path(output_dir), config)
        print(artifact.pdf_path)
    print("Composite: source mark retained; terminal payoff and finite-difference checks passed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    run_example(parser.parse_args().output_dir)
