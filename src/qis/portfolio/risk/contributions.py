"""
Euler risk decompositions: who owns which part of a portfolio's risk, given a covariance matrix.

``compute_portfolio_risk_contributions`` is the identity itself. With σ = sqrt(w' Σ w), asset i
contributes rc_i = w_i (Σ w)_i / σ, and Σ_i rc_i = w' Σ w / σ = σ exactly, so the parts add to
the whole and not to an approximation of it. ``compute_portfolio_risk_contribution_ratios``
normalizes that identity to dimensionless shares, and the grouped variant aggregates those shares
over a labelled partition. ``compute_benchmark_portfolio_risk_contributions`` applies the same
identity to active weights Δw = w_p - w_b, so its parts sum to the tracking error
sqrt(Δw' Σ Δw); ``is_independent_risk`` switches it to standalone position volatilities
|Δw_i| σ_i, which are not additive and sum to at least the tracking error.

``calculate_marginal_active_risk`` and ``calculate_active_risk_squared`` are the factor-model
pair and decompose active *variance*, not volatility: their marginal terms carry the factor 2
from ∂(w' Σ w)/∂w. Nothing here estimates Σ - units and annualisation are whatever the caller's
covariance carries. The factor-structured version through time is ``factor_model.py``.
"""
import numpy as np
import pandas as pd
from typing import Union, Tuple


def compute_portfolio_risk_contributions(w: Union[np.ndarray, pd.Series],
                                         covar: Union[np.ndarray, pd.DataFrame]
                                         ) -> Union[np.ndarray, pd.Series]:
    """Computes the risk contribution of each asset to the portfolio's total risk.

    Args:
        w: Portfolio weights as array or Series.
        covar: Covariance matrix as array or DataFrame.

    Returns:
        Risk contributions for each asset. A non-positive-variance portfolio
        has no risk to attribute and returns zeros.

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
    portfolio_var = float(w.T @ covar @ w)
    if portfolio_var <= 0.0:
        if isinstance(w, pd.Series):
            return pd.Series(0.0, index=w.index)
        return np.zeros_like(w, dtype=float)
    portfolio_vol = np.sqrt(portfolio_var)
    marginal_risk_contribution = covar @ w.T
    rc = np.multiply(marginal_risk_contribution, w) / portfolio_vol
    return rc


def compute_portfolio_risk_contribution_ratios(
        weights: Union[np.ndarray, pd.Series],
        covar: Union[np.ndarray, pd.DataFrame],
        ) -> Union[np.ndarray, pd.Series]:
    """Compute normalized Euler risk contributions that sum to one.

    This is the dimensionless counterpart of
    :func:`compute_portfolio_risk_contributions`. The existing function returns
    contributions in volatility units and remains the canonical Euler identity;
    this function divides those contributions by portfolio volatility. A
    zero-variance portfolio has no risk to attribute and returns zeros.

    Args:
        weights: Portfolio weights as an array or asset-indexed Series.
        covar: Covariance matrix as an array or consistently labelled DataFrame.

    Returns:
        Per-asset risk contribution ratios in the same container type as ``weights``.

    Raises:
        ValueError: If input types are not compatible.
        AssertionError: If dimensions do not match for NumPy arrays.
    """
    if isinstance(covar, pd.DataFrame) and isinstance(weights, pd.Series):
        aligned_w = weights.reindex(index=covar.index).fillna(0.0)
    elif isinstance(covar, np.ndarray) and isinstance(weights, np.ndarray):
        assert covar.shape[0] == covar.shape[1] == weights.shape[0]
        aligned_w = weights
    else:
        raise ValueError(f"unsupported types {type(weights)} and {type(covar)}")

    portfolio_var = float(aligned_w.T @ covar @ aligned_w)
    if portfolio_var <= 0.0:
        if isinstance(aligned_w, pd.Series):
            return pd.Series(0.0, index=aligned_w.index)
        return np.zeros_like(aligned_w, dtype=float)
    contributions = compute_portfolio_risk_contributions(w=aligned_w, covar=covar)
    return contributions / np.sqrt(portfolio_var)


def compute_group_portfolio_risk_contribution_ratios(
        weights: pd.Series,
        covar: pd.DataFrame,
        groups: pd.Series,
        ) -> pd.Series:
    """Aggregate normalized Euler risk contributions over supplied groups.

    Group labels may represent statistical clusters, sectors, asset classes, or
    any other complete partition of the covariance universe. Contributions retain
    their sign, follow first-seen group order, and reconcile to the asset-level
    total from :func:`compute_portfolio_risk_contribution_ratios`. A zero-variance
    portfolio returns zero for every group.

    Args:
        weights: Asset weights, which may cover a superset of covariance assets.
        covar: Labelled covariance matrix defining the risk universe.
        groups: One group label for every covariance asset.

    Returns:
        Normalized group risk contributions in first-seen group order.

    Raises:
        TypeError: If ``groups`` is not a Series.
        ValueError: If covariance or group labels cannot define a complete partition.
    """
    if not isinstance(groups, pd.Series):
        raise TypeError("groups must be a pandas Series")
    if covar.empty or covar.shape[0] != covar.shape[1]:
        raise ValueError("covar must be non-empty and square")
    if not covar.index.equals(covar.columns) or not covar.index.is_unique:
        raise ValueError("covar index and columns must be identical unique asset labels")
    if not groups.index.is_unique:
        raise ValueError("group asset labels must be unique")

    aligned_groups = groups.reindex(covar.index)
    if aligned_groups.isna().any():
        missing = aligned_groups.index[aligned_groups.isna()].tolist()
        raise ValueError(
            f"groups must classify every covariance asset; missing {missing[:5]}"
        )
    contributions = compute_portfolio_risk_contribution_ratios(weights=weights, covar=covar)
    grouped = contributions.groupby(aligned_groups, sort=False).sum()
    grouped.name = "risk_contribution"
    return grouped


def compute_benchmark_portfolio_risk_contributions(w_portfolio: Union[np.ndarray, pd.Series],
                                                    w_benchmark: Union[np.ndarray, pd.Series],
                                                    covar: Union[np.ndarray, pd.DataFrame],
                                                    is_independent_risk: bool = False
                                                    ) -> Union[np.ndarray, pd.Series]:
    """Computes the Euler contributions of active positions to tracking error.

    With active weights d = w_portfolio - w_benchmark and tracking error TE = sqrt(d' Σ d),
    asset i contributes d_i (Σ d)_i / TE, and the contributions sum to TE. They equal the
    ``mcte`` column of ``RiskModel.compute_marginal_tre_at_date`` for the same inputs; this
    function is the single-date, container-agnostic form of that decomposition.

    Args:
        w_portfolio: Portfolio weights as array or Series.
        w_benchmark: Benchmark weights as array or Series.
        covar: Covariance matrix as array or DataFrame.
        is_independent_risk: If True, return the standalone active risks ``abs(d_i) * σ_i`` with
            σ_i = sqrt(Σ_ii) instead. They ignore correlation and are not additive: their sum
            is the undiversified (perfectly correlated) figure and bounds TE from above.

    Returns:
        Contributions of the active positions in the units of ``sqrt(covar)``. With a
        labelled covariance, Series weights are aligned to its labels (missing labels count
        as zero weight, labels outside the covariance are dropped), array weights are taken
        in covariance order, and the result is a Series in covariance order. With array
        inputs the result is an array. A zero tracking error returns zeros, the convention of
        ``RiskModel``.

    Raises:
        ValueError: If input types are not compatible.
        AssertionError: If dimensions don't match.
    """
    if isinstance(covar, pd.DataFrame):
        aligned = []
        for weights in (w_portfolio, w_benchmark):
            if isinstance(weights, pd.Series):
                aligned.append(weights.reindex(index=covar.index).fillna(0.0))
            elif isinstance(weights, np.ndarray):
                assert covar.shape[0] == covar.shape[1] == weights.shape[0]
                aligned.append(pd.Series(weights, index=covar.index))
            else:
                raise ValueError(f"unsupported types {type(w_portfolio)}, {type(w_benchmark)} "
                                 f"and {type(covar)}")
        w_portfolio, w_benchmark = aligned
    elif (isinstance(covar, np.ndarray) and isinstance(w_portfolio, np.ndarray)
          and isinstance(w_benchmark, np.ndarray)):
        assert covar.shape[0] == covar.shape[1] == w_portfolio.shape[0] == w_benchmark.shape[0]
    else:
        raise ValueError(f"unsupported types {type(w_portfolio)}, {type(w_benchmark)} "
                         f"and {type(covar)}")
    active_weights = w_portfolio - w_benchmark
    if is_independent_risk:
        rc = np.abs(active_weights) * np.sqrt(np.diag(covar))
    else:
        rc = compute_portfolio_risk_contributions(w=active_weights, covar=covar)
    return rc


def calculate_marginal_active_risk(portfolio_weights: np.ndarray,
                                   benchmark_weights: np.ndarray,
                                   asset_betas: np.ndarray,
                                   factor_covar: np.ndarray,
                                   idiosyncratic_var: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient of active variance under a factor model, split into its two sources.

    With active weights d = w_p - w_b, active factor exposures e = B d and the model covariance
    B' Σ_x B + Ψ, the active variance is TE^2 = e' Σ_x e + d' Ψ d, and this function returns its
    gradient with respect to the portfolio weights, m = 2 B' Σ_x e + 2 Ψ d. That is the
    marginal active *variance*, not the marginal active risk ∂TE/∂w = m / (2 TE). By Euler's
    theorem for a degree-two function, Σ_i d_i m_i = 2 TE^2, so d_i m_i / (2 TE) are the
    contributions to TE and d_i m_i / (2 TE^2) are shares that sum to one.

    Args:
        portfolio_weights: Portfolio weights (N,)
        benchmark_weights: Benchmark weights (N,)
        asset_betas: Factor loadings for all assets (K x N), one row per factor
        factor_covar: Factor covariance matrix (K x K)
        idiosyncratic_var: Asset idiosyncratic variances (N,)

    Returns:
        marginal_risk: Gradient of active variance for each asset, m = m_sys + m_idio
        systematic_marginal: Systematic component 2 B' Σ_x e
        idiosyncratic_marginal: Idiosyncratic component 2 Ψ d
    """
    # Calculate benchmark factor exposures from its asset weights
    benchmark_factor_exposures = asset_betas @ benchmark_weights  # Shape: (K,)

    # Current portfolio factor exposures
    portfolio_factor_exposures = asset_betas @ portfolio_weights  # Shape: (K,)

    # Active factor exposures
    active_exposures = portfolio_factor_exposures - benchmark_factor_exposures  # Shape: (K,)

    # Weight differences
    weight_diff = portfolio_weights - benchmark_weights  # Shape: (N,)

    # Gradient of active variance for each asset
    marginal_risk = np.zeros(len(portfolio_weights))
    systematic_marginal = np.zeros(len(portfolio_weights))
    idiosyncratic_marginal = np.zeros(len(portfolio_weights))

    for i in range(len(portfolio_weights)):
        # Asset i's factor loadings
        asset_i_betas = asset_betas[:, i]  # Shape: (K,)

        # Systematic part of the gradient of active variance
        systematic_marginal[i] = 2.0 * asset_i_betas.T @ factor_covar @ active_exposures

        # Idiosyncratic part of the gradient of active variance
        idiosyncratic_marginal[i] = 2.0 * idiosyncratic_var[i] * weight_diff[i]

        # Gradient of active variance
        marginal_risk[i] = systematic_marginal[i] + idiosyncratic_marginal[i]

    return marginal_risk, systematic_marginal, idiosyncratic_marginal


def calculate_active_risk_squared(portfolio_weights: np.ndarray,
                                  benchmark_weights: np.ndarray,
                                  asset_betas: np.ndarray,
                                  factor_covar: np.ndarray,
                                  idiosyncratic_var: np.ndarray
                                  ) -> float:
    """Active variance TE^2 = e' Σ_x e + d' Ψ d under a factor model, with e = B d.

    Args:
        portfolio_weights: Portfolio weights (N,)
        benchmark_weights: Benchmark weights (N,)
        asset_betas: Factor loadings for all assets (K x N), one row per factor
        factor_covar: Factor covariance matrix (K x K)
        idiosyncratic_var: Asset idiosyncratic variances (N,)

    Returns:
        Active variance in the units of the covariances; its square root is the tracking error.
    """
    # Active factor exposures
    portfolio_exposures = asset_betas @ portfolio_weights
    benchmark_exposures = asset_betas @ benchmark_weights
    active_exposures = portfolio_exposures - benchmark_exposures

    # Systematic active risk
    systematic_risk_sq = active_exposures.T @ factor_covar @ active_exposures

    # Idiosyncratic active risk
    weight_diff = portfolio_weights - benchmark_weights
    idiosyncratic_risk_sq = weight_diff.T @ np.diag(idiosyncratic_var) @ weight_diff

    return systematic_risk_sq + idiosyncratic_risk_sq
