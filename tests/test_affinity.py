import numpy as np
import pytest

from spectral_clustering.graphs import (
    pairwise_distance,
    rbf_affinity,
    degree_matrix,
    unnormalized_laplacian,
)


def test_pairwise_distance_known_triangle():
    """Khoảng cách tính tay của tam giác 3–4–5."""
    X = np.array([
        [0, 0],
        [3, 0],
        [0, 4],
    ])

    expected = np.array([
        [0, 3, 4],
        [3, 0, 5],
        [4, 5, 0],
    ])

    np.testing.assert_allclose(pairwise_distance(X), expected)


def test_rbf_known_values_and_laplacian():
    """Đối chiếu RBF với khoảng cách biết trước."""
    X = np.array([
        [0, 0],
        [3, 0],
        [0, 4],
    ])

    expected_W = np.array([
        [0, np.exp(-0.9), np.exp(-1.6)],
        [np.exp(-0.9), 0, np.exp(-2.5)],
        [np.exp(-1.6), np.exp(-2.5), 0],
    ])

    W = rbf_affinity(X, gamma=0.1)
    D = degree_matrix(W)
    L = unnormalized_laplacian(W)

    np.testing.assert_allclose(W, expected_W)
    np.testing.assert_allclose(W, W.T)
    np.testing.assert_allclose(np.diag(W), 0)
    assert np.all(W >= 0)

    np.testing.assert_allclose(np.diag(D), expected_W.sum(axis=1))
    np.testing.assert_allclose(L, D - W)
    np.testing.assert_allclose(L.sum(axis=1), 0, atol=1e-12)


@pytest.mark.parametrize("gamma", [0, -1, np.nan, np.inf])
def test_rbf_rejects_invalid_gamma(gamma):
    X = np.array([[0, 0], [1, 0]])

    with pytest.raises(ValueError):
        rbf_affinity(X, gamma=gamma)