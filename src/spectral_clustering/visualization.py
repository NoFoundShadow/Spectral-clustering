"""Các hàm trực quan hóa kết quả phân cụm trên dữ liệu hai chiều."""

from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

PathLike: TypeAlias = str | Path


def _validate_points(X: ArrayLike) -> NDArray[np.float64]:
    try:
        points = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("X phải là dữ liệu số có shape (n_samples, 2)") from error

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("X phải có shape (n_samples, 2)")
    if points.shape[0] == 0:
        raise ValueError("X không được rỗng")
    if not np.all(np.isfinite(points)):
        raise ValueError("X chỉ được chứa các giá trị hữu hạn, không có NaN/Inf")
    return points


def _validate_labels(
    labels: ArrayLike,
    n_samples: int,
    name: str = "labels",
) -> NDArray[np.int64]:
    try:
        numeric_labels = np.asarray(labels, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} phải là mảng một chiều gồm các số nguyên") from error

    if numeric_labels.ndim != 1:
        raise ValueError(f"{name} phải có shape (n_samples,)")
    if numeric_labels.shape[0] != n_samples:
        raise ValueError(f"{name} phải có cùng số phần tử với số dòng của X")
    if not np.all(np.isfinite(numeric_labels)):
        raise ValueError(f"{name} chỉ được chứa các số nguyên hữu hạn")
    if not np.all(numeric_labels == np.floor(numeric_labels)):
        raise ValueError(f"{name} chỉ được chứa các số nguyên")
    return numeric_labels.astype(np.int64)


def _save_figure(figure: Figure, save_path: PathLike) -> None:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")


def plot_clusters(
    X: ArrayLike,
    labels: ArrayLike | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    save_path: PathLike | None = None,
) -> tuple[Figure, Axes]:
    """Vẽ dữ liệu 2D và nhãn cụm; trả về ``(Figure, Axes)`` để chỉnh tiếp.

    ``X`` có shape ``(n_samples, 2)``; ``labels`` có shape ``(n_samples,)``
    hoặc là ``None``. Nhãn ``-1`` được vẽ như noise của DBSCAN.
    """
    points = _validate_points(X)
    cluster_labels = (
        None if labels is None else _validate_labels(labels, points.shape[0])
    )

    if ax is None:
        figure, ax = plt.subplots(figsize=(5, 5))
    else:
        figure = ax.figure

    if cluster_labels is None:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=28,
            color="#4c72b0",
            label="Dữ liệu",
        )
    else:
        unique_labels = np.unique(cluster_labels)
        regular_labels = unique_labels[unique_labels != -1]
        color_map = colormaps["tab20"].resampled(max(len(regular_labels), 1))

        for color_index, cluster_id in enumerate(regular_labels):
            in_cluster = cluster_labels == cluster_id
            ax.scatter(
                points[in_cluster, 0],
                points[in_cluster, 1],
                s=28,
                color=color_map(color_index),
                label=f"Cụm {cluster_id}",
            )

        if -1 in unique_labels:
            is_noise = cluster_labels == -1
            ax.scatter(
                points[is_noise, 0],
                points[is_noise, 1],
                s=36,
                color="#777777",
                marker="x",
                linewidths=1.2,
                label="Noise (-1)",
            )

    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    if save_path is not None:
        _save_figure(figure, save_path)
    return figure, ax


def plot_baseline_comparison(
    X: ArrayLike,
    kmeans_labels: ArrayLike,
    dbscan_labels: ArrayLike,
    *,
    title: str | None = None,
    save_path: PathLike | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """So sánh dữ liệu, K-Means và DBSCAN; trả về ``(Figure, Axes)``.

    ``X`` là cùng dữ liệu 2D đã đưa vào hai thuật toán; hai mảng nhãn có
    shape ``(n_samples,)``. ``title`` là tiêu đề chung tùy chọn của hình.
    """
    points = _validate_points(X)
    kmeans = _validate_labels(kmeans_labels, points.shape[0], "kmeans_labels")
    dbscan = _validate_labels(dbscan_labels, points.shape[0], "dbscan_labels")

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        sharex=True,
        sharey=True,
    )
    plot_clusters(points, ax=axes[0], title="Dữ liệu đầu vào")
    plot_clusters(points, kmeans, ax=axes[1], title="K-Means")
    plot_clusters(points, dbscan, ax=axes[2], title="DBSCAN")

    x_limits = axes[0].get_xlim()
    y_limits = axes[0].get_ylim()
    for axis in axes:
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_aspect("equal", adjustable="box")

    if title is not None:
        figure.suptitle(title)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        figure.tight_layout()

    if save_path is not None:
        _save_figure(figure, save_path)
    return figure, axes
