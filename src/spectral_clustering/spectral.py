import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans

from .graphs import normalized_laplacian, rbf_affinity


def spectral_clustering(
    X: ArrayLike,
    n_clusters: int,
    gamma: float = 1.0,
    seed: int = 42,
) -> NDArray[np.int64]:
    """Cluster samples using an RBF graph and a normalized spectral embedding."""
    W = rbf_affinity(X, gamma=gamma)
    n_samples = W.shape[0]

    if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
        raise ValueError("n_clusters must be an integer")
    if not 1 <= n_clusters <= n_samples:
        raise ValueError("n_clusters must be between 1 and n_samples")

    L_sym = normalized_laplacian(W)
    _, eigenvectors = np.linalg.eigh(L_sym)
    U = eigenvectors[:, :n_clusters]

    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    if np.any(row_norms == 0):
        raise ValueError("Cannot normalize a zero row in the spectral embedding")
    T = U / row_norms

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return model.fit_predict(T).astype(np.int64)