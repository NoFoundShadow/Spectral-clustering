# Phương pháp thực nghiệm

## 1. Dataset nhân tạo

Toàn bộ dữ liệu thử nghiệm 2D được sinh qua điểm gọi chung `make_synthetic(name, seed=42)` trong `src/spectral_clustering/datasets.py` với `n_samples=300` mặc định. Bốn hàm sinh riêng (`generate_blobs`, `generate_moons`, `generate_circles`, `generate_anisotropic_blobs`) giữ mặc định `n_samples=500` cho các thử nghiệm bổ sung.

Cách gọi chuẩn để B và D dùng lại đúng dữ liệu:

```python
from spectral_clustering.datasets import make_synthetic

X, y = make_synthetic("moons", seed=42)
```

Thông số mặc định của 4 bộ dữ liệu chuẩn (`n_samples=300`, `seed=42`):

| Tên (`name`) | Số mẫu | Số cụm | Thông số riêng |
| --- | --- | --- | --- |
| `blobs` | 300 | 3 | `centers=3`, `cluster_std=1.0` |
| `moons` | 300 | 2 | `noise=0.05` |
| `circles` | 300 | 2 | `factor=0.5`, `noise=0.05` |
| `anisotropic` | 300 | 3 | `centers=3`, `cluster_std=1.0`, nhân ma trận `[[0.6, -0.6], [-0.4, 0.8]]` |

Kiểm tra tính tái lập: đã gọi `make_synthetic(name, seed=42)` hai lần liên tiếp cho cả 4 tên. Mỗi lần đều trả về `X` kích thước `(300, 2)` và `y` kích thước `(300,)` trùng khớp từng phần tử, số nhãn tương ứng là `3, 2, 2, 3`.

## 2. Cấu hình Baseline (K-Means & DBSCAN)

### Chuẩn hóa dữ liệu

Mỗi bộ dữ liệu được chuẩn hóa một lần bằng hàm `standardize_features(X)` trong `src/spectral_clustering/preprocessing.py` (dùng `StandardScaler().fit_transform(X)`). Cả K-Means và DBSCAN đều chạy trên cùng mảng `X_scaled` này; nhãn thật `y` được giữ riêng làm tham chiếu.

### Tham số K-Means

Chạy qua `run_kmeans(X_scaled, n_clusters, random_state=42)` trong `src/spectral_clustering/baselines.py` với `n_init=10` và `n_clusters` lần lượt là `3, 2, 2, 3` theo thứ tự `blobs`, `moons`, `circles`, `anisotropic`.

### Kết quả thử lưới tham số DBSCAN

Chạy qua `run_dbscan(X_scaled, eps, min_samples)` (`metric="euclidean"`) trên `X_scaled` của từng bộ với lưới `eps` thuộc `{0.15, 0.25, 0.40}` và `min_samples` thuộc `{5, 10}`. Số cụm dưới đây không tính nhãn noise `-1`.

| Dataset | `eps` | `min_samples` | Số cụm tìm được | Tỉ lệ noise |
| --- | --- | --- | --- | --- |
| `blobs` | 0.15 | 5 | 3 | 2.3% (7/300) |
| `blobs` | 0.15 | 10 | 3 | 6.7% (20/300) |
| `blobs` | 0.25 | 5 | 3 | 0.3% (1/300) |
| `blobs` | 0.25 | 10 | 3 | 1.0% (3/300) |
| `blobs` | 0.40 | 5 | 3 | 0.0% (0/300) |
| `blobs` | 0.40 | 10 | 3 | 0.0% (0/300) |
| `moons` | 0.15 | 5 | 14 | 6.3% (19/300) |
| `moons` | 0.15 | 10 | 7 | 70.0% (210/300) |
| `moons` | 0.25 | 5 | 2 | 0.0% (0/300) |
| `moons` | 0.25 | 10 | 3 | 2.0% (6/300) |
| `moons` | 0.40 | 5 | 2 | 0.0% (0/300) |
| `moons` | 0.40 | 10 | 2 | 0.0% (0/300) |
| `circles` | 0.15 | 5 | 18 | 41.0% (123/300) |
| `circles` | 0.15 | 10 | 1 | 96.7% (290/300) |
| `circles` | 0.25 | 5 | 7 | 0.0% (0/300) |
| `circles` | 0.25 | 10 | 1 | 50.0% (150/300) |
| `circles` | 0.40 | 5 | 2 | 0.0% (0/300) |
| `circles` | 0.40 | 10 | 3 | 0.3% (1/300) |
| `anisotropic` | 0.15 | 5 | 3 | 3.7% (11/300) |
| `anisotropic` | 0.15 | 10 | 3 | 5.7% (17/300) |
| `anisotropic` | 0.25 | 5 | 3 | 0.0% (0/300) |
| `anisotropic` | 0.25 | 10 | 3 | 1.3% (4/300) |
| `anisotropic` | 0.40 | 5 | 3 | 0.0% (0/300) |
| `anisotropic` | 0.40 | 10 | 3 | 0.0% (0/300) |

### Cấu hình DBSCAN chọn để vẽ baseline

Kết quả chi tiết từng bộ được lưu trong `baseline_results` tại `notebooks/01_baselines.ipynb` với các khóa `X_scaled`, `y`, `kmeans_labels`, `dbscan_labels`, `n_clusters`, `eps`, `min_samples`:

- `blobs`: `eps=0.25, min_samples=5` (tách đúng 3 cụm, giữ được điểm biên sát thực tế với 0.3% noise).
- `moons`: `eps=0.25, min_samples=5` (tách gọn 2 nửa mặt trăng, không đứt cụm và 0.0% noise).
- `circles`: `eps=0.40, min_samples=5` (lựa chọn duy nhất trong lưới tách đúng 2 vòng tròn đồng tâm mà không vỡ cụm hay mất mẫu; các mức `eps` nhỏ hơn bị chia thành 7 hoặc 18 cụm nhỏ hoặc gán 50% đến 96.7% mẫu thành noise).
- `anisotropic`: `eps=0.25, min_samples=5` (tách đúng 3 cụm kéo dãn với 0.0% noise).
