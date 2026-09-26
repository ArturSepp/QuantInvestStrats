"""RiskModel rejects covariances that are not positive semi-definite and negative variances.

The stress module already refused such inputs at evaluation time; RiskModel now refuses them at
construction, with a tolerance that accepts rounding noise.
"""

import numpy as np
import pandas as pd
import pytest

import qis

ASSETS = ['A1', 'A2', 'A3']
FACTORS = ['Equity', 'Rates']
DATE = pd.Timestamp('2025-12-31')
LOADINGS = pd.DataFrame([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], index=ASSETS, columns=FACTORS)
FACTOR_COVAR = pd.DataFrame([[0.04, 0.004], [0.004, 0.01]], index=FACTORS, columns=FACTORS)
RESIDUAL_VARS = pd.Series([0.01, 0.0004, 0.0004], index=ASSETS)


def _covar(residual_vars: pd.Series = RESIDUAL_VARS,
           factor_covar: pd.DataFrame = FACTOR_COVAR) -> pd.DataFrame:
    return LOADINGS @ factor_covar @ LOADINGS.T + pd.DataFrame(
        np.diag(residual_vars), index=ASSETS, columns=ASSETS)


def test_valid_factor_model_is_accepted() -> None:
    model = qis.RiskModel(covar={DATE: _covar()}, factor_loadings={DATE: LOADINGS},
                          factor_covar={DATE: FACTOR_COVAR}, residual_vars={DATE: RESIDUAL_VARS})
    assert model.dates.tolist() == [DATE]


def test_negative_residual_variance_is_rejected() -> None:
    residual_vars = pd.Series([0.01, -0.0004, 0.0004], index=ASSETS)
    with pytest.raises(ValueError, match=r"residual_vars\[.*\] has negative variances"):
        qis.RiskModel(covar={DATE: _covar()}, factor_loadings={DATE: LOADINGS},
                      factor_covar={DATE: FACTOR_COVAR}, residual_vars={DATE: residual_vars})


def test_indefinite_covariance_is_rejected() -> None:
    # Correlation 1.5 between A1 and A2: symmetric and finite, but not a covariance.
    covar = pd.DataFrame([[0.04, 0.06, 0.0], [0.06, 0.04, 0.0], [0.0, 0.0, 0.01]],
                         index=ASSETS, columns=ASSETS)
    with pytest.raises(ValueError, match=r"covar\[.*\] is not positive semi-definite"):
        qis.RiskModel(covar={DATE: covar})


def test_indefinite_factor_covariance_is_rejected() -> None:
    factor_covar = pd.DataFrame([[0.04, 0.03], [0.03, 0.01]], index=FACTORS, columns=FACTORS)
    with pytest.raises(ValueError, match=r"factor_covar\[.*\] is not positive semi-definite"):
        qis.RiskModel(covar={DATE: _covar()}, factor_loadings={DATE: LOADINGS},
                      factor_covar={DATE: factor_covar}, residual_vars={DATE: RESIDUAL_VARS})


def test_rounding_noise_is_tolerated() -> None:
    """A singular covariance and a zero residual variance off by rounding are accepted."""
    residual_vars = pd.Series([0.01, -1.0e-18, 0.0], index=ASSETS)
    singular = LOADINGS @ FACTOR_COVAR @ LOADINGS.T  # rank two: one eigenvalue is ~0
    eigenvalues = np.linalg.eigvalsh(singular.to_numpy())
    assert abs(eigenvalues.min()) < 1.0e-16
    shifted = singular - pd.DataFrame(np.eye(3) * 1.0e-15, index=ASSETS, columns=ASSETS)

    model = qis.RiskModel(covar={DATE: shifted}, factor_loadings={DATE: LOADINGS},
                          factor_covar={DATE: FACTOR_COVAR}, residual_vars={DATE: residual_vars})

    assert model.residual_vars[DATE].min() == -1.0e-18  # validated, not altered
