# Phương pháp thực nghiệm

## 1. Dataset nhân tạo

Nhóm tạo 4 bộ dữ liệu 2D qua hàm `make_synthetic(name, seed=42)` trong `src/spectral_clustering/datasets.py` với mặc định `n_samples=300`. Bốn hàm sinh riêng (`generate_blobs`, `generate_moons`, `generate_circles`, `generate_anisotropic_blobs`) vẫn giữ mặc định `n_samples=500` cho các thử nghiệm khác. Cùng một giá trị `seed=42` luôn trả về cặp mảng `X` có kích thước `(300, 2)` và `y` có kích thước `(300,)` giống nhau qua mỗi lần gọi.

Ví dụ cách gọi hàm sinh dữ liệu chuẩn:

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

## 2. Cấu hình Baseline (K-Means & DBSCAN)

### Chuẩn hóa dữ liệu

Trước khi phân cụm, ta chuẩn hóa từng bộ dữ liệu một lần qua hàm `standardize_features(X)` trong `src/spectral_clustering/preprocessing.py` (dùng `StandardScaler().fit_transform(X)`). Cả K-Means và DBSCAN đều chạy trên cùng mảng `X_scaled` này, còn nhãn thật `y` được giữ riêng để đối chiếu kết quả.

### Tham số K-Means

Ta chạy `run_kmeans(X_scaled, n_clusters, random_state=42)` trong `src/spectral_clustering/baselines.py` với `n_init=10` và số cụm `n_clusters` lần lượt là `3, 2, 2, 3` cho `blobs`, `moons`, `circles`, `anisotropic`.

### Kết quả thử lưới tham số DBSCAN

Hàm `run_dbscan(X_scaled, eps, min_samples)` (`metric="euclidean"`) được chạy trên `X_scaled` của từng bộ dữ liệu qua lưới `eps` gồm `{0.15, 0.25, 0.40}` và `min_samples` gồm `{5, 10}`. Số cụm tìm được trong bảng không tính nhãn noise `-1`.

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

Với `blobs`, `moons` và `anisotropic`, cặp tham số `eps=0.25, min_samples=5` tách đúng số cụm kỳ vọng (`3, 2, 3`) và hầu như không làm mất mẫu (0.3% noise trên `blobs`, 0.0% trên hai bộ còn lại). Riêng với `circles`, các mức `eps` nhỏ hơn đều khiến hai vòng tròn bị vỡ thành 7 đến 18 cụm nhỏ hoặc có từ 50% đến 96.7% điểm rơi vào noise, nên ta chọn `eps=0.40, min_samples=5` vì đây là cấu hình duy nhất trong lưới tách được trọn vẹn 2 vòng tròn với 0.0% noise.

Kết quả chạy baseline của cả 4 bộ được lưu sẵn trong từ điển `baseline_results` ở `notebooks/01_baselines.ipynb`, gồm các khóa `X_scaled`, `y`, `kmeans_labels`, `dbscan_labels`, `n_clusters`, `eps` và `min_samples`.

## 3. So sánh Spectral Clustering với Scikit-learn

Mục tiêu: So sánh kết quả phân cụm của thuật toán Spectral Clustering do nhóm triển khai với SpectralClustering trong thư viện Scikit-learn.

Điều kiện so sánh: Để đảm bảo tính công bằng, cả hai phương pháp sử dụng chung các thiết lập:
-Dữ liệu đầu vào: Cùng một tập dữ liệu X.
-Ma trận trọng số: Cùng ma trận affinity W được xây dựng từ dữ liệu đầu vào bằng RBF kernel.
-Số cụm: Cùng giá trị n_clusters = k.
-Random seed: Cùng giá trị random_state = 42.

### Cấu hình Scikit-learn

Sử dụng SpectralClustering với các tham số tương ứng:
'from sklearn.cluster import SpectralClustering

sklearn_model = SpectralClustering(
    n_clusters=k,
    affinity="precomputed",
    assign_labels="kmeans",
    random_state=42
)

sklearn_labels = sklearn_model.fit_predict(W)'

Trong đó:
-n_clusters: Số cụm cần tìm.
-affinity="precomputed": Sử dụng trực tiếp ma trận trọng số W đã được xây dựng, thay vì để Scikit-learn tự tính affinity.
-assign_labels="kmeans": Sau khi thực hiện Spectral Clustering và biến đổi dữ liệu sang không gian phổ (spectral space), sử dụng thuật toán K-Means để chia các điểm dữ liệu thành k cụm.
-random_state=42: Cố định seed để tăng khả năng tái lập kết quả.

> **Note:** affinity="precomputed" rất quan trọng: nếu không có nó, Scikit-learn sẽ tự xây dựng ma trận affinity, khiến hai phương pháp không còn sử dụng cùng W.

### Đánh giá kết quả

-Sử dụng hai chỉ số Adjusted Rand Index (ARI) và Normalized Mutual Information (NMI) để đánh giá mức độ tương đồng giữa kết quả của nhóm và Scikit-learn. Do nhãn cụm không có thứ tự cố định, không so sánh trực tiếp giá trị nhãn giữa hai phương pháp. Thay vào đó, sử dụng ARI và NMI để đánh giá mức độ tương đồng của cách phân chia dữ liệu.
-Ngoài ra, có thể so sánh trực quan kết quả phân cụm thông qua biểu đồ scatter plot, sử dụng cùng dữ liệu đầu vào để quan sát sự khác biệt giữa hai phương pháp.