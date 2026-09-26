"""apply_pca applies eigen_signs per eigenvector and keeps the eigen-decomposition."""

# packages
import numpy as np
import pytest

# qis
from qis.models.linear.pca import apply_pca, compute_pca_r2

RHO = np.array([[1.0, 0.5, -0.2],
                [0.5, 1.0, 0.3],
                [-0.2, 0.3, 1.0]])


@pytest.mark.parametrize('signs', [(1.0, -1.0, 1.0), (-1.0, -1.0, -1.0), (1.0, 1.0, -1.0)])
def test_eigen_signs_flip_whole_eigenvectors(signs) -> None:
    """Each column keeps rho v = nu v, and its first loading carries the requested sign."""
    eigen_signs = np.array(signs)
    nu, vectors = apply_pca(cmatrix=RHO, eigen_signs=eigen_signs)
    np.testing.assert_allclose(RHO @ vectors, vectors * nu, atol=1e-12)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=1e-12)
    np.testing.assert_array_equal(np.sign(vectors[0]), eigen_signs)
    # the same vectors as the default convention, up to one sign per column
    _, default_vectors = apply_pca(cmatrix=RHO)
    np.testing.assert_allclose(np.abs(vectors), np.abs(default_vectors), atol=1e-12)


def test_eigen_signs_wrong_length_raises() -> None:
    """A sign vector that does not have one entry per eigenvector is rejected."""
    with pytest.raises(ValueError, match='eigen_signs'):
        apply_pca(cmatrix=RHO, eigen_signs=np.array([1.0, -1.0]))


def test_compute_pca_r2_returns_one_array() -> None:
    """The explained-variance shares are one array summing to one."""
    shares = compute_pca_r2(cmatrix=RHO)
    assert isinstance(shares, np.ndarray) and shares.shape == (3,)
    assert shares.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(compute_pca_r2(cmatrix=RHO, is_cumulative=True)[-1], 1.0,
                               atol=1e-12)
