"""Build graph matrices used by unnormalized spectral clustering."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

Edge = tuple[int, int, float]


def weighted_adjacency_matrix(
    n_nodes: int,
    edges: Iterable[Edge],
) -> NDArray[np.float64]:
    """Return a symmetric weighted adjacency matrix with a zero diagonal."""
    if not isinstance(n_nodes, int) or isinstance(n_nodes, bool) or n_nodes <= 0:
        raise ValueError("n_nodes must be a positive integer")

    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    seen_edges: set[tuple[int, int]] = set()

    for u, v, weight in edges:
        if not isinstance(u, int) or isinstance(u, bool):
            raise TypeError("edge endpoints must be integers")
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError("edge endpoints must be integers")
        if not 0 <= u < n_nodes or not 0 <= v < n_nodes:
            raise ValueError("edge endpoint is outside the graph")
        if u == v:
            raise ValueError("self-loops are not allowed because diag(W) must be zero")

        value = float(weight)
        if not np.isfinite(value) or value < 0:
            raise ValueError("edge weights must be finite and non-negative")

        edge_key = (min(u, v), max(u, v))
        if edge_key in seen_edges:
            raise ValueError(f"duplicate undirected edge: {edge_key}")
        seen_edges.add(edge_key)

        adjacency[u, v] = adjacency[v, u] = value

    return adjacency


def _validate_adjacency(adjacency: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("W must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("W must contain only finite values")
    if np.any(matrix < 0):
        raise ValueError("W must contain only non-negative weights")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("W must be symmetric for an undirected graph")
    if not np.allclose(np.diag(matrix), 0):
        raise ValueError("diag(W) must be zero")
    return matrix


def degree_matrix(adjacency: ArrayLike) -> NDArray[np.float64]:
    """Return D = diag(W.sum(axis=1)) for an undirected graph."""
    matrix = _validate_adjacency(adjacency)
    return np.diag(matrix.sum(axis=1))


def unnormalized_laplacian(adjacency: ArrayLike) -> NDArray[np.float64]:
    """Return the basic, unnormalized graph Laplacian L = D - W."""
    matrix = _validate_adjacency(adjacency)
    return np.diag(matrix.sum(axis=1)) - matrix


def graph_matrices_from_edges(
    n_nodes: int,
    edges: Iterable[Edge],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build W, D and the unnormalized L from a weighted edge list."""
    adjacency = weighted_adjacency_matrix(n_nodes, edges)
    degree = degree_matrix(adjacency)
    laplacian = unnormalized_laplacian(adjacency)
    return adjacency, degree, laplacian

# Compute Euclid distance
def pairwise_distance(X: ArrayLike) -> NDArray[np.float64]:
    points = np.asarray(X, dtype=float)
    if points.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features)")
    if points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("X must contain samples and features")
    if not np.all(np.isfinite(points)):
        raise ValueError("X must contain only finite values")

    n_samples = points.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=float)

    for i in range(n_samples):
        differences = points[i + 1:] - points[i]
        values = np.linalg.norm(differences, axis=1)

        distances[i, i + 1:] = values
        distances[i + 1:, i] = values

    return distances


def rbf_affinity(
    X: ArrayLike,
    gamma: float = 1.0,
) -> NDArray[np.float64]:
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")

    distances = pairwise_distance(X)
    adjacency = np.exp(-gamma * distances**2)
    np.fill_diagonal(adjacency, 0.0)

    return adjacency