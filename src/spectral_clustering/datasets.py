"""Tạo các tập dữ liệu nhân tạo 2D (Blobs, Moons, Circles, Anisotropic Blobs)."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_blobs, make_circles, make_moons


def _get_random_state(
    random_state: int | np.random.RandomState | None = None,
    seed: int | None = None,
) -> int | np.random.RandomState | None:
    """Lấy seed/random_state ưu tiên seed nếu có."""
    return seed if seed is not None else random_state


def generate_blobs(
    n_samples: int = 500,
    centers: int = 3,
    cluster_std: float = 1.0,
    random_state: int | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Tạo dữ liệu Blobs 2D (cụm dạng hình cầu)."""
    r_state = _get_random_state(random_state, seed)
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=centers,
        cluster_std=cluster_std,
        random_state=r_state,
    )
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def generate_moons(
    n_samples: int = 500,
    noise: float = 0.05,
    random_state: int | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Tạo dữ liệu Moons 2D (2 nửa mặt trăng đan xen)."""
    r_state = _get_random_state(random_state, seed)
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=r_state,
    )
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def generate_circles(
    n_samples: int = 500,
    factor: float = 0.5,
    noise: float = 0.05,
    random_state: int | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Tạo dữ liệu Circles 2D (2 vòng tròn đồng tâm)."""
    r_state = _get_random_state(random_state, seed)
    X, y = make_circles(
        n_samples=n_samples,
        factor=factor,
        noise=noise,
        random_state=r_state,
    )
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def generate_anisotropic_blobs(
    n_samples: int = 500,
    centers: int = 3,
    cluster_std: float = 1.0,
    transformation: np.ndarray | None = None,
    random_state: int | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Tạo dữ liệu Anisotropic Blobs 2D (cụm Gaussian bị kéo dãn)."""
    r_state = _get_random_state(random_state, seed)
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=centers,
        cluster_std=cluster_std,
        random_state=r_state,
    )
    # Ma trận biến đổi afin để kéo dãn dữ liệu
    if transformation is None:
        transformation = np.array([[0.6, -0.6], [-0.4, 0.8]], dtype=np.float64)
    else:
        transformation = np.asarray(transformation, dtype=np.float64)

    X_aniso = np.dot(X, transformation)
    return np.asarray(X_aniso, dtype=np.float64), np.asarray(y, dtype=np.int64)


make_blobs_dataset = generate_blobs
make_moons_dataset = generate_moons
make_circles_dataset = generate_circles
make_anisotropic_dataset = generate_anisotropic_blobs

_GENERATORS = {
    "blobs": generate_blobs,
    "moons": generate_moons,
    "circles": generate_circles,
    "anisotropic": generate_anisotropic_blobs,
}


def make_synthetic(
    name: str,
    n_samples: int = 300,
    seed: int = 42,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Tạo bộ dữ liệu chuẩn theo tên ('blobs', 'moons', 'circles', 'anisotropic')."""
    key = name.strip().lower()
    if key not in _GENERATORS:
        raise ValueError(
            f"Tên dataset không hợp lệ: {name!r}. Chọn trong: {tuple(_GENERATORS)}"
        )
    return _GENERATORS[key](n_samples=n_samples, seed=seed, **kwargs)

