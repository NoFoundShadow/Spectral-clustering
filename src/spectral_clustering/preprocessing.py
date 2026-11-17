"""Chuẩn hóa đặc trưng dữ liệu."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import StandardScaler


def standardize_features(X: ArrayLike) -> NDArray[np.float64]:
    """Chuẩn hóa dữ liệu về trung bình 0 và độ lệch chuẩn 1."""
    return np.asarray(StandardScaler().fit_transform(X), dtype=np.float64)
