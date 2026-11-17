import numpy as np
import pytest

from spectral_clustering.datasets import make_synthetic
from spectral_clustering.graphs import (
    degree_matrix,
    graph_matrices_from_edges,
    pairwise_distance,
    rbf_affinity,
    unnormalized_laplacian,
)


def test_four_node_graph_matches_hand_calculation() -> None:
    edges = [(0, 1, 5), (1, 3, 2), (2, 3, 9), (0, 3, 3)]

    adjacency, degree, laplacian = graph_matrices_from_edges(4, edges)

    expected_adjacency = np.array(
        [
            [0, 5, 0, 3],
            [5, 0, 0, 2],
            [0, 0, 0, 9],
            [3, 2, 9, 0],
        ],
        dtype=float,
    )
    expected_degree = np.diag([8, 7, 9, 14]).astype(float)
    expected_laplacian = np.array(
        [
            [8, -5, 0, -3],
            [-5, 7, 0, -2],
            [0, 0, 9, -9],
            [-3, -2, -9, 14],
        ],
        dtype=float,
    )

    np.testing.assert_array_equal(adjacency, expected_adjacency)
    np.testing.assert_array_equal(degree, expected_degree)
    np.testing.assert_array_equal(laplacian, expected_laplacian)
    np.testing.assert_allclose(laplacian.sum(axis=1), 0)


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        graph_matrices_from_edges(2, [(0, 1, -1)])


def test_pairwise_distance_and_rbf_known_points() -> None:
    X = np.array([[0.0, 0.0], [0.3, 0.4]])

    distances = pairwise_distance(X)
    np.testing.assert_allclose(distances, [[0.0, 0.5], [0.5, 0.0]])

    W = rbf_affinity(X, gamma=1.0)
    expected_weight = np.exp(-0.25)
    np.testing.assert_allclose(W, [[0.0, expected_weight], [expected_weight, 0.0]])
    np.testing.assert_allclose(np.diag(W), 0.0)


def test_rbf_graph_matrices_properties() -> None:
    X = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])

    W = rbf_affinity(X, gamma=0.1)
    D = degree_matrix(W)
    L = unnormalized_laplacian(W)

    np.testing.assert_allclose(W, W.T)
    assert np.all(W >= 0)
    np.testing.assert_allclose(np.diag(W), 0.0, atol=1e-12)
    np.testing.assert_allclose(D, np.diag(W.sum(axis=1)))
    np.testing.assert_allclose(L, D - W)
    np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-12)


def test_make_synthetic_reproducibility() -> None:
    expected_clusters = {
        "blobs": 3,
        "moons": 2,
        "circles": 2,
        "anisotropic": 3,
    }
    for name, n_classes in expected_clusters.items():
        X1, y1 = make_synthetic(name, seed=42)
        X2, y2 = make_synthetic(name, seed=42)

        assert X1.shape == (300, 2)
        assert y1.shape == (300,)
        assert len(np.unique(y1)) == n_classes
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

