"""
principal component analysis of a covariance or correlation matrix, and what is built on it.

``apply_pca`` is the primitive: ``np.linalg.eigh`` on a symmetric matrix, reordered from the
largest eigenvalue down, with a sign convention for each eigenvector. The default convention
makes the largest-magnitude loading positive; it pins the sign only while that loading keeps its
identity, so a near tie between the two largest loadings can still flip an eigenvector between
refits, and a repeated eigenvalue has no unique eigenvectors at all. ``eigen_signs`` sets the sign
of each eigenvector's first loading instead.
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
import qis.utils.np_ops as npo
import qis.models.linear.ewm as ewm


def compute_eigen_portfolio_weights(covar: np.ndarray) -> np.ndarray:
    """
    Return ranked PCA portfolio weights scaled to unit variance.

    Args:
        covar: Square covariance matrix with finite, materially positive asset variances.

    Returns:
        Principal-portfolio weights by row, ranked by descending correlation eigenvalue.

    Raises:
        ValueError: If an asset variance or correlation eigenvalue cannot support unit-variance
            scaling.
    """
    corr = npo.covar_to_corr(covar)
    if not np.isfinite(corr).all():
        raise ValueError(
            "unit-variance eigen-portfolios require finite, positive asset variances"
        )
    vols = np.sqrt(np.diag(covar))
    eigen_values, eigen_vectors = apply_pca(cmatrix=corr, is_max_sign_positive=True)
    eigenvalue_scale = max(1.0, float(np.max(np.abs(eigen_values))))
    eigenvalue_tolerance = 100.0 * np.finfo(float).eps * eigenvalue_scale
    if np.any(eigen_values <= eigenvalue_tolerance):
        raise ValueError(
            "unit-variance eigen-portfolios require materially positive correlation eigenvalues"
        )
    scale = np.outer(vols, np.sqrt(eigen_values).T)
    weights = eigen_vectors / scale
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
    conventions are offered to pin it down; each flips whole eigenvectors (columns), so the
    result remains an eigen-decomposition. Neither can rule out every flip: under the default,
    a near tie in magnitude between the two largest loadings of opposite sign lets the sign
    change between refits, and eigenvectors of a repeated eigenvalue are not unique.

    Args:
        cmatrix: symmetric covariance or correlation matrix, shape (n, n)
        is_max_sign_positive: flip each eigenvector so its largest-magnitude element is
            positive. Ignored when ``eigen_signs`` is given
        eigen_signs: explicit sign per eigenvector, shape (n,), entry ``j`` for the ``j``-th
            eigenvector in descending order. Eigenvector ``j`` is flipped when its first
            loading (asset 0) has the opposite sign to ``eigen_signs[j]``; a zero first loading
            or a zero sign leaves it unchanged. Use to carry the convention of a previous fit
            forward so loadings stay comparable across dates

    Returns:
        (eigenvalues, eigenvectors), descending by eigenvalue, eigenvectors as columns

    Raises:
        ValueError: if ``eigen_signs`` does not have one entry per eigenvector
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

        eigen_signs = np.asarray(eigen_signs, dtype=float).reshape(-1)
        if eigen_signs.shape[0] != eigen_vectors.shape[1]:
            raise ValueError(f"eigen_signs must have one entry per eigenvector: expected "
                             f"{eigen_vectors.shape[1]}, got {eigen_signs.shape[0]}")
        # flip whole eigenvectors (columns) whose first loading disagrees with the target sign
        signed_eigen_vectors = eigen_vectors.copy()
        for idx in range(eigen_vectors.shape[1]):
            if np.sign(eigen_vectors[0, idx]) * np.sign(eigen_signs[idx]) < 0.0:
                signed_eigen_vectors[:, idx] = - eigen_vectors[:, idx]
        eigen_vectors = signed_eigen_vectors

    return eigen_values, eigen_vectors


def compute_pca_r2(cmatrix: np.ndarray, is_cumulative: bool = False) -> np.ndarray:
    """
    explained-variance shares of the eigenvalues of a symmetric matrix.

    The shares are ``nu_j / sum_k nu_k`` with the eigenvalues ``nu_j`` of :func:`apply_pca` in
    descending order. They lie in [0, 1] and sum to one only for a positive semi-definite input;
    a negative eigenvalue, as a pairwise-complete matrix can have, gives a negative share.

    Args:
        cmatrix: symmetric covariance or correlation matrix, shape (n, n)
        is_cumulative: return the cumulative shares ``sum_{k<=j} nu_k / sum_k nu_k`` instead

    Returns:
        the shares, shape (n,), largest component first
    """
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
    """
    explained-variance shares through time, from an EWM correlation or covariance tensor.

    Runs :func:`qis.compute_ewm_covar_tensor` on the rows of ``data`` as supplied, uncentred
    (no mean removed), from a zero seed and with the kernel's default forward fill of missing
    entries, then applies :func:`compute_pca_r2` on each sampling date. A sampling date takes the
    last row on or before it, so every row of the output is point in time. Missing values are
    held rather than skipped, so fill or align a ragged panel first.

    Args:
        data: returns, rows are dates and columns are assets
        freq: frequency of the sampling dates generated over ``time_period``
        time_period: span of the sampling dates. None uses the span of ``data``
        ewm_lambda: EWM decay of the tensor
        is_corr: decompose the correlation tensor. False decomposes the second-moment tensor,
            whose components are dominated by the most volatile assets

    Returns:
        one row per sampling date and one column per component, ``PC1`` to ``PCn``
    """
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
