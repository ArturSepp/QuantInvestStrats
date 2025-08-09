"""
computation of risk contributions
"""
import numpy as np
import pandas as pd
from typing import Union


def compute_portfolio_risk_contributions(w: Union[np.ndarray, pd.Series],
                                         covar: Union[np.ndarray, pd.DataFrame]
                                         ) -> Union[np.ndarray, pd.Series]:
    """Computes the risk contribution of each asset to the portfolio's total risk.

    Args:
        w: Portfolio weights as array or Series.
        covar: Covariance matrix as array or DataFrame.

    Returns:
        Risk contributions for each asset.

    Raises:
        ValueError: If input types are not compatible.
        AssertionError: If dimensions don't match for numpy arrays.
    """
    if isinstance(covar, pd.DataFrame) and isinstance(w, pd.Series):  # make sure weights are alined
        w = w.reindex(index=covar.index).fillna(0.0)
    elif isinstance(covar, np.ndarray) and isinstance(w, np.ndarray):
        assert covar.shape[0] == covar.shape[1] == w.shape[0]
    else:
        raise ValueError(f"unnsuported types {type(w)} and {type(covar)}")
    portfolio_vol = np.sqrt(w.T @ covar @ w)
    marginal_risk_contribution = covar @ w.T
    rc = np.multiply(marginal_risk_contribution, w) / portfolio_vol
    return rc


def compute_benchmark_portfolio_risk_contributions(w_portfolio: Union[np.ndarray, pd.Series],
                                                    w_benchmark: Union[np.ndarray, pd.Series],
                                                    covar: Union[np.ndarray, pd.DataFrame],
                                                    is_independent_risk: bool = False
                                                    ) -> Union[np.ndarray, pd.Series]:
    """Computes risk contributions of active positions relative to benchmark.

    Args:
        w_portfolio: Portfolio weights as array or Series.
        w_benchmark: Benchmark weights as array or Series.
        covar: Covariance matrix as array or DataFrame.
        is_independent_risk: If True, assumes positions are independent (diagonal risk only).

    Returns:
        Risk contributions of active positions (portfolio - benchmark).

    Raises:
        ValueError: If input types are not compatible.
        AssertionError: If dimensions don't match for numpy arrays.
    """
    if isinstance(covar, pd.DataFrame) and isinstance(w_portfolio, pd.Series):  # make sure weights are alined
        w_portfolio = w_portfolio.reindex(index=covar.index).fillna(0.0)
    elif isinstance(covar, pd.DataFrame) and isinstance(w_benchmark, pd.Series):  # make sure weights are alined
        w_benchmark = w_benchmark.reindex(index=covar.index).fillna(0.0)
    elif isinstance(covar, np.ndarray) and isinstance(w_portfolio, np.ndarray) and isinstance(w_benchmark, np.ndarray):
        assert covar.shape[0] == covar.shape[1] == w_portfolio.shape[0] == w_benchmark.shape[0]
    else:
        raise ValueError(f"unnsuported types {type(w_portfolio)}, {type(w_benchmark)} and {type(covar)}")
    if is_independent_risk:
        rc = np.sqrt(np.multiply(np.square(w_portfolio-w_benchmark),  np.diag(covar)))
    else:
        portfolio_vol = np.sqrt(w_benchmark.T @ covar @ w_benchmark)
        marginal_risk_contribution = covar @ (w_portfolio-w_benchmark).T
        rc = np.multiply(marginal_risk_contribution, (w_portfolio-w_benchmark)) / portfolio_vol
    return rc
