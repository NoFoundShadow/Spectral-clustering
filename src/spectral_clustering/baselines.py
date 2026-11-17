"""Các thuật toán phân cụm cơ sở: K-Means và DBSCAN."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import DBSCAN, KMeans


def run_kmeans(
    X: ArrayLike,
    n_clusters: int,
    random_state: int = 42,
) -> NDArray[np.int64]:
    """Chạy K-Means và trả về nhãn cụm cho từng mẫu."""
    labels = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state,
    ).fit_predict(X)
    return np.asarray(labels, dtype=np.int64)


def run_dbscan(
    X: ArrayLike,
    eps: float,
    min_samples: int,
) -> NDArray[np.int64]:
    """Chạy DBSCAN với khoảng cách Euclid, giữ -1 cho điểm nhiễu (noise)."""
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
    ).fit_predict(X)
    return np.asarray(labels, dtype=np.int64)
