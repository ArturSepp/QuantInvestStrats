"""Deprecation contract for the legacy returns-based factor model."""
import pandas as pd
import pytest

from qis.portfolio.risk.factor_model import LinearModel


def test_compute_active_factor_risk_warns_exactly_once() -> None:
    date = pd.Timestamp('2024-01-31')
    factors = pd.Index(['Market'])
    assets = pd.Index(['Asset'])
    model = LinearModel(
        x=pd.DataFrame([[0.01]], index=[date], columns=factors),
        y=pd.DataFrame([[0.01]], index=[date], columns=assets),
        loadings={'Market': pd.DataFrame([[1.0]], index=[date], columns=assets)},
        x_covars={date: pd.DataFrame([[0.04]], index=factors, columns=factors)},
        residual_vars=pd.DataFrame([[0.01]], index=[date], columns=assets))
    portfolio_weights = pd.DataFrame([[1.0]], index=[date], columns=assets)
    benchmark_weights = pd.DataFrame([[0.8]], index=[date], columns=assets)

    with pytest.warns(
            DeprecationWarning,
            match=r"RiskModel\.compute_tre_decomposition_at_date.*compute_marginal_tre_at_date",
    ) as warnings_record:
        model.compute_active_factor_risk(
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights)

    assert len(warnings_record) == 1
