import numpy as np
import pytest

from spectral_clustering.graphs import graph_matrices_from_edges


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
