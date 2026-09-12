"""Shared quote and FX construction for instrument portfolio valuation."""

import numpy as np
import pandas as pd

from qis.portfolio.stress.instruments import PayoffContext, ResponseBasis


def validate_shocks(shocks: pd.DataFrame, factors: pd.Index) -> pd.DataFrame:
    """Validate a complete labelled batch and align factors explicitly."""
    if not isinstance(shocks, pd.DataFrame) or shocks.empty:
        raise ValueError("factor_log_shocks must be a nonempty DataFrame")
    if shocks.index.has_duplicates or shocks.columns.has_duplicates:
        raise ValueError("scenario and factor IDs must be unique")
    if shocks.index.isna().any() or shocks.columns.isna().any():
        raise ValueError("scenario and factor IDs must not be missing")
    if set(shocks.columns) != set(factors):
        raise ValueError("factor_log_shocks must contain exactly the model factors")
    result = shocks.reindex(columns=factors).astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("factor_log_shocks must be finite")
    return result


def build_context(portfolio, shocks: pd.DataFrame) -> PayoffContext:
    """Cache actual quotes, currency conversion and shared response mappings."""
    model = portfolio.risk_model
    betas = model.factor_loadings[portfolio.risk_date]
    shocks = validate_shocks(shocks, betas.columns)
    responses = betas.index
    currencies = [portfolio.reference_currency, *portfolio.fx_rates]
    fx_map = pd.DataFrame(0.0, index=currencies, columns=responses)
    fx0 = pd.Series(1.0, index=currencies)
    for currency, quote in portfolio.fx_rates.items():
        fx0.loc[currency] = quote.spot0
        if quote.response_id is not None:
            fx_map.loc[currency, quote.response_id] = 1.0
    quote0 = pd.Series({key: item.spot0 for key, item in portfolio.underlyings.items()})
    quote_map = pd.DataFrame(0.0, index=quote0.index, columns=responses)
    for key, quote in portfolio.underlyings.items():
        if quote.response_id is not None:
            quote_map.loc[key, quote.response_id] = 1.0
        if quote.response_basis is ResponseBasis.REFERENCE:
            quote_map.loc[key] -= fx_map.loc[quote.currency]
    response_shocks = shocks @ betas.T
    with np.errstate(over="ignore", invalid="ignore"):
        quotes = np.exp(response_shocks @ quote_map.T).mul(quote0, axis=1)
        fx = np.exp(response_shocks @ fx_map.T).mul(fx0, axis=1)
    if (
        not np.isfinite(quotes.to_numpy()).all()
        or not np.isfinite(fx.to_numpy()).all()
        or (quotes <= 0).any().any()
        or (fx <= 0).any().any()
    ):
        raise ValueError("scenario produces nonfinite or nonpositive quote/FX levels")
    return PayoffContext(
        quotes,
        quote0,
        fx,
        fx0,
        shocks,
        quote_map,
        fx_map,
        portfolio.reference_currency,
        pd.Series({k: q.currency for k, q in portfolio.underlyings.items()}),
        response_shocks,
    )
