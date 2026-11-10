"""Các hàm sinh dữ liệu nhân tạo 2D (Blobs, Moons, Circles, Anisotropic Blobs).

Cung cấp 4 hàm sinh dữ liệu phục vụ nghiên cứu và so sánh các thuật toán phân cụm
(K-Means, DBSCAN, Spectral Clustering). Tất cả các hàm đều trả về tuple (X, y)
với X có shape (n_samples, 2) và y có shape (n_samples,).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.datasets import (
    make_blobs as _sk_make_blobs,
    make_circles as _sk_make_circles,
    make_moons as _sk_make_moons,
)


def _resolve_random_state(
    random_state: int | np.random.RandomState | None = None,
    seed: int | None = None,
) -> int | np.random.RandomState | None:
    """Xử lý đồng nhất tham số random_state và seed."""
    if seed is not None and random_state is not None:
        if isinstance(random_state, int) and seed != random_state:
            raise ValueError(
                "Không thể truyền đồng thời seed và random_state với giá trị khác nhau "
                f"(seed={seed}, random_state={random_state})."
            )
        return seed
    if seed is not None:
        return seed
    return random_state


def _validate_n_samples(n_samples: Any) -> None:
    """Kiểm tra tính hợp lệ của tham số n_samples."""
    if isinstance(n_samples, int):
        if n_samples <= 0:
            raise ValueError(f"n_samples phải là số nguyên dương, nhận được {n_samples}")
    elif isinstance(n_samples, (tuple, list)):
        if len(n_samples) != 2 or any(not isinstance(n, int) or n <= 0 for n in n_samples):
            raise ValueError("Khi n_samples là tuple/list, phải có 2 phần tử nguyên dương")
    else:
        raise TypeError("n_samples phải là int hoặc tuple[int, int]")


def generate_blobs(
    n_samples: int = 500,
    centers: int | ArrayLike = 3,
    cluster_std: float | Sequence[float] = 1.0,
    center_box: tuple[float, float] = (-10.0, 10.0),
    *,
    random_state: int | np.random.RandomState | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Sinh tập dữ liệu dạng các đám mây điểm Gaussian đẳng hướng (Isotropic Gaussian Blobs).

    Phù hợp cho các thuật toán phân cụm giả định hình cầu lồi như K-Means.

    Parameters
    ----------
    n_samples : int, default=500
        Tổng số điểm mẫu.
    centers : int | ArrayLike, default=3
        Số cụm hoặc tọa độ các tâm cụm.
    cluster_std : float | Sequence[float], default=1.0
        Độ lệch chuẩn của mỗi cụm.
    center_box : tuple[float, float], default=(-10.0, 10.0)
        Khoảng giá trị biên ngẫu nhiên cho mỗi tâm cụm khi centers là số nguyên.
    random_state : int | np.random.RandomState | None, default=None
        Hạt giống sinh số ngẫu nhiên.
    seed : int | None, default=None
        Bí danh (alias) cho random_state.

    Returns
    -------
    X : NDArray[np.float64]
        Mảng tọa độ điểm dữ liệu shape (n_samples, 2).
    y : NDArray[np.int64]
        Mảng nhãn cụm thực tế (ground truth) shape (n_samples,).
    """
    _validate_n_samples(n_samples)
    rng = _resolve_random_state(random_state, seed)

    X, y = _sk_make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=centers,
        cluster_std=cluster_std,
        center_box=center_box,
        random_state=rng,
    )

    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def generate_moons(
    n_samples: int | tuple[int, int] = 500,
    noise: float = 0.05,
    *,
    random_state: int | np.random.RandomState | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Sinh tập dữ liệu hai nửa mặt trăng đan xen (Two Interleaving Half Circles / Moons).

    Cấu trúc phi tuyến tính, không lồi; K-Means thường thất bại trong khi Spectral
    Clustering và DBSCAN nhận diện tốt.

    Parameters
    ----------
    n_samples : int | tuple[int, int], default=500
        Tổng số điểm mẫu (hoặc tuple số lượng cho từng mặt trăng).
    noise : float, default=0.05
        Độ lệch chuẩn của nhiễu Gaussian thêm vào dữ liệu.
    random_state : int | np.random.RandomState | None, default=None
        Hạt giống sinh số ngẫu nhiên.
    seed : int | None, default=None
        Bí danh cho random_state.

    Returns
    -------
    X : NDArray[np.float64]
        Mảng tọa độ điểm dữ liệu shape (n_samples, 2).
    y : NDArray[np.int64]
        Mảng nhãn cụm thực tế shape (n_samples,).
    """
    _validate_n_samples(n_samples)
    if noise < 0:
        raise ValueError(f"noise phải không âm, nhận được {noise}")

    rng = _resolve_random_state(random_state, seed)

    X, y = _sk_make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=rng,
    )

    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def generate_circles(
    n_samples: int | tuple[int, int] = 500,
    factor: float = 0.5,
    noise: float = 0.05,
    *,
    random_state: int | np.random.RandomState | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Sinh tập dữ liệu hai vòng tròn đồng tâm (Concentric Circles).

    Cấu trúc phi tuyến tính, đa thang đo khoảng cách; K-Means không thể tách được
    vì các miền phân cách không lồi, trong khi Spectral Clustering tách được.

    Parameters
    ----------
    n_samples : int | tuple[int, int], default=500
        Tổng số điểm mẫu.
    factor : float, default=0.5
        Tỉ lệ bán kính giữa vòng tròn trong và vòng tròn ngoài (0 < factor < 1).
    noise : float, default=0.05
        Độ lệch chuẩn của nhiễu Gaussian.
    random_state : int | np.random.RandomState | None, default=None
        Hạt giống sinh số ngẫu nhiên.
    seed : int | None, default=None
        Bí danh cho random_state.

    Returns
    -------
    X : NDArray[np.float64]
        Mảng tọa độ điểm dữ liệu shape (n_samples, 2).
    y : NDArray[np.int64]
        Mảng nhãn cụm thực tế shape (n_samples,).
    """
    _validate_n_samples(n_samples)
    if not (0.0 <= factor < 1.0):
        raise ValueError(f"factor phải nằm trong khoảng [0, 1), nhận được {factor}")
    if noise < 0:
        raise ValueError(f"noise phải không âm, nhận được {noise}")

    rng = _resolve_random_state(random_state, seed)

    X, y = _sk_make_circles(
        n_samples=n_samples,
        factor=factor,
        noise=noise,
        random_state=rng,
    )

    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def generate_anisotropic_blobs(
    n_samples: int = 500,
    centers: int | ArrayLike = 3,
    cluster_std: float | Sequence[float] = 1.0,
    transformation: ArrayLike | None = None,
    *,
    random_state: int | np.random.RandomState | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Sinh tập dữ liệu các cụm Gaussian bị biến dạng dị hướng (Anisotropic Blobs).

    Các cụm bị kéo dài theo các hướng khác nhau thông qua ma trận biến đổi afin/tuyến tính.
    K-Means giả định phương sai đẳng hướng nên dễ cắt cụm sai.

    Parameters
    ----------
    n_samples : int, default=500
        Tổng số điểm mẫu.
    centers : int | ArrayLike, default=3
        Số cụm hoặc tọa độ các tâm cụm ban đầu.
    cluster_std : float | Sequence[float], default=1.0
        Độ lệch chuẩn ban đầu của các cụm.
    transformation : ArrayLike | None, default=None
        Ma trận biến đổi 2x2. Nếu None, dùng ma trận chuẩn [[0.6, -0.6], [-0.4, 0.8]].
    random_state : int | np.random.RandomState | None, default=None
        Hạt giống sinh số ngẫu nhiên.
    seed : int | None, default=None
        Bí danh cho random_state.

    Returns
    -------
    X : NDArray[np.float64]
        Mảng tọa độ điểm dữ liệu shape (n_samples, 2).
    y : NDArray[np.int64]
        Mảng nhãn cụm thực tế shape (n_samples,).
    """
    _validate_n_samples(n_samples)
    rng = _resolve_random_state(random_state, seed)

    if transformation is None:
        trans_matrix = np.array([[0.6, -0.6], [-0.4, 0.8]], dtype=np.float64)
    else:
        trans_matrix = np.asarray(transformation, dtype=np.float64)
        if trans_matrix.shape != (2, 2):
            raise ValueError(
                f"transformation phải có shape (2, 2), nhận được {trans_matrix.shape}"
            )
        if not np.all(np.isfinite(trans_matrix)):
            raise ValueError("transformation chỉ được chứa các giá trị hữu hạn")

    X_orig, y = generate_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=rng,
    )

    X_aniso = np.dot(X_orig, trans_matrix)

    return np.asarray(X_aniso, dtype=np.float64), y


# Aliases tương thích với quy ước đặt tên sklearn (make_*) và tên ngắn gọn
make_blobs = generate_blobs
make_moons = generate_moons
make_circles = generate_circles
make_anisotropic_blobs = generate_anisotropic_blobs
generate_anisotropic = generate_anisotropic_blobs
make_anisotropic = generate_anisotropic_blobs


SYNTHETIC_DATASETS = {
    "blobs": generate_blobs,
    "moons": generate_moons,
    "circles": generate_circles,
    "anisotropic": generate_anisotropic_blobs,
    "anisotropic_blobs": generate_anisotropic_blobs,
}


def get_synthetic_dataset(
    name: str,
    *,
    n_samples: int = 500,
    random_state: int | np.random.RandomState | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Truy xuất dataset nhân tạo theo tên gọi chuẩn hóa.

    Parameters
    ----------
    name : str
        Tên tập dữ liệu: 'blobs', 'moons', 'circles', 'anisotropic' (hoặc 'anisotropic_blobs').
    n_samples : int, default=500
        Số lượng điểm mẫu.
    random_state : int | np.random.RandomState | None, default=None
        Hạt giống ngẫu nhiên.
    seed : int | None, default=None
        Bí danh cho random_state.
    **kwargs : Any
        Các tham số bổ sung tùy chỉnh cho từng hàm tương ứng.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.int64]]
        (X, y) với X shape (n_samples, 2) và y shape (n_samples,).
    """
    key = name.strip().lower()
    if key not in SYNTHETIC_DATASETS:
        valid_keys = ", ".join(repr(k) for k in ["blobs", "moons", "circles", "anisotropic"])
        raise ValueError(
            f"Không tìm thấy dataset '{name}'. Các tên hợp lệ: {valid_keys}"
        )

    return SYNTHETIC_DATASETS[key](
        n_samples=n_samples,
        random_state=random_state,
        seed=seed,
        **kwargs,
    )


__all__ = [
    "SYNTHETIC_DATASETS",
    "generate_anisotropic",
    "generate_anisotropic_blobs",
    "generate_blobs",
    "generate_circles",
    "generate_moons",
    "get_synthetic_dataset",
    "make_anisotropic",
    "make_anisotropic_blobs",
    "make_blobs",
    "make_circles",
    "make_moons",
]
