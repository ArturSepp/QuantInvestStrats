"""Small labelled synthetic market shared by instrument stress contracts."""

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.risk.risk_model import RiskModel
from qis.portfolio.stress.instruments import ResponseBasis, Underlying
from qis.portfolio.stress.portfolio import InstrumentPortfolio


@pytest.fixture
def market():
    """Return a factory with stock, currency and independent response identities."""
    date = pd.Timestamp("2026-08-31")
    factors = ["Equity", "FX", "Credit", "Credit EM"]
    responses = ["stock", "eur", "proxy"]
    betas = pd.DataFrame(
        [[1.0, 0.4, 0.2, 0.0], [0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 1.0, 1.0]],
        index=responses,
        columns=factors,
    )
    cov = pd.DataFrame(
        [
            [0.04, 0.003, 0.006, 0.002],
            [0.003, 0.01, 0.0, 0.0],
            [0.006, 0.0, 0.01, 0.004],
            [0.002, 0.0, 0.004, 0.01],
        ],
        index=factors,
        columns=factors,
    )
    residual = pd.Series([0.01, 0.0, 0.0225], index=responses)
    asset_cov = betas @ cov @ betas.T + pd.DataFrame(
        np.diag(residual), index=responses, columns=responses
    )
    model = RiskModel({date: asset_cov}, {date: betas}, {date: cov}, {date: residual})

    def make(
        holdings,
        *,
        currency="USD",
        basis=ResponseBasis.REFERENCE,
        denominator=1000.0,
        spot=100.0,
        model_override=None,
        underlyings=None,
        **kwargs,
    ):
        """Build an absolute-position portfolio without an estimator or provider."""
        quotes = underlyings or {
            "actual": Underlying("actual", spot, currency, "stock", basis),
            "proxy_quote": Underlying("proxy_quote", 80.0, "USD", "proxy", basis),
        }
        fx = (
            {}
            if currency == "USD"
            else {"EUR": Underlying("EURUSD", 1.2, "USD", "eur", ResponseBasis.REFERENCE)}
        )
        return InstrumentPortfolio(
            holdings=tuple(holdings),
            underlyings=quotes,
            risk_model=model_override or model,
            risk_date=date,
            valuation_date=date,
            reference_currency="USD",
            reporting_denominator=denominator,
            denominator_label="Gross assets",
            fx_rates=fx,
            **kwargs,
        )

    return make
