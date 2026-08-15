"""
principal component analysis of a covariance or correlation matrix, and what is built on it.

``apply_pca`` is the primitive: ``np.linalg.eigh`` on a symmetric matrix, reordered from the
largest eigenvalue down, with a sign convention so that loadings do not flip between refits.
``compute_pca_r2`` turns the eigenvalues into variance shares, raw or cumulative, and
``compute_data_pca_r2`` runs that through time over an EWM tensor of the input, correlation unless
``is_corr`` is False, sampled at ``freq``, one row per date and one column per component.
``compute_eigen_portfolio_weights`` returns the principal portfolios of a covariance matrix, one
per row, each scaled to unit variance.

``compute_eigen_portfolio_weights`` decomposes the correlation matrix rather than the covariance,
so the ranking is by explained correlation, and the volatilities re-enter in the scaling step
w_ij = v_ij / (σ_i sqrt(λ_j)).
"""
import numpy as np
import pandas as pd
import qis.utils.dates as da
import qis.models.linear.ewm as ewm


def compute_eigen_portfolio_weights(covar: np.ndarray) -> np.ndarray:
    """
    return weights for pca portolios with unit variance
    covar = eigen_vectors @ np.diag(eigen_values) @ eigen_vectors.T
    rows are principal portolio weights ranked
    """
    vols = np.sqrt(np.diag(covar))
    inv_vol = np.reciprocal(vols)
    norm = np.outer(inv_vol, inv_vol)
    corr = norm * covar
    eigen_values, eigen_vectors = apply_pca(cmatrix=corr, is_max_sign_positive=True)
    # eigen_values, eigen_vectors = np.linalg.eigh(corr)
    scale = np.outer(vols, np.sqrt(eigen_values).T)
    weights = np.reciprocal(scale) * eigen_vectors
    return weights.T


def apply_pca(cmatrix: np.ndarray,
              is_max_sign_positive: bool = True,
              eigen_signs: np.ndarray = None
              ) -> (np.ndarray, np.ndarray):
    """
    eigen decomposition of a symmetric matrix, ordered from the largest eigenvalue down.

    Uses ``np.linalg.eigh``, so ``cmatrix`` must be symmetric; only the lower triangle is read.
    Eigenvectors are columns: ``cmatrix @ vectors[:, i] == values[i] * vectors[:, i]``.

    The sign of an eigenvector is arbitrary, which makes loadings flip between refits. Two
    conventions are offered to pin it down.

    Args:
        cmatrix: symmetric covariance or correlation matrix, shape (n, n)
        is_max_sign_positive: flip each eigenvector so its largest-magnitude element is
            positive. Ignored when ``eigen_signs`` is given
        eigen_signs: explicit sign per eigenvector, shape (n,); use to carry the sign
            convention of a previous fit forward so loadings stay comparable across dates

    Returns:
        (eigenvalues, eigenvectors), descending by eigenvalue, eigenvectors as columns
    """
    # from sample covar_model
    eig_vals, eig_vecs = np.linalg.eigh(cmatrix)

    # Make a list of (eigenvalue, eigenvector) tuples for sorting
    eig_pairs = [(eig_vals[i], eig_vecs[:, i]) for i in range(len(eig_vals))]

    # reverse (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.reverse()

    # get back to ndarrays
    eigen_values = np.array([eig_pair[0] for eig_pair in eig_pairs]).T
    eigen_vectors = np.array([eig_pair[1] for eig_pair in eig_pairs]).T

    if is_max_sign_positive and eigen_signs is None:

        signed_eigen_vectors = eigen_vectors
        for idx, eigen_vector in enumerate(eigen_vectors.T):
            arg_max = np.argmax(np.abs(eigen_vector))
            if eigen_vector[arg_max] < 0.0:
                eigen_vector = - eigen_vector
                signed_eigen_vectors[:, idx] = eigen_vector
        eigen_vectors = signed_eigen_vectors

    elif eigen_signs is not None:

        # eigen_vectors = eigen_signs * eigen_vectors
        signed_eigen_vectors = eigen_vectors.copy()
        for idx, eigen_vector in enumerate(eigen_vectors):
            if np.sign(eigen_vector[0]) != eigen_signs[idx]:
               signed_eigen_vectors[idx] = - eigen_vector
        eigen_vectors = signed_eigen_vectors

    return eigen_values, eigen_vectors


def compute_pca_r2(cmatrix: np.ndarray, is_cumulative: bool = False) -> (np.ndarray, np.ndarray):
    eigen_values, _ = apply_pca(cmatrix=cmatrix)
    if is_cumulative:
        out = np.cumsum(eigen_values) / np.sum(eigen_values)
    else:
        out = eigen_values / np.sum(eigen_values)
    return out


def compute_data_pca_r2(data: pd.DataFrame,
                        freq: str = 'ME',
                        time_period: da.TimePeriod = None,
                        ewm_lambda: float = 0.94,
                        is_corr: bool = True
                        ) -> pd.DataFrame:

    corr_tensor_txy = ewm.compute_ewm_covar_tensor(a=data.to_numpy(),
                                                     ewm_lambda=ewm_lambda,
                                                     is_corr=is_corr)

    if time_period is None:
        time_period = da.get_time_period(df=data)
    sample_dates = time_period.to_pd_datetime_index(freq=freq)
    original_idx = pd.Series(range(len(data.index)), index=data.index)
    resampled_index = original_idx.reindex(index=sample_dates, method='ffill')

    pca_r2s = {}
    for date, date_idx in zip(resampled_index.index, resampled_index.to_numpy()):
        pca_r2s[date] = compute_pca_r2(cmatrix=corr_tensor_txy[date_idx])

    pca_r2s = pd.DataFrame.from_dict(pca_r2s,
                                     orient='index',
                                     columns=[f"PC{n+1}" for n in range(len(data.columns))])
    return pca_r2s
