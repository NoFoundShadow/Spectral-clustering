# Đồ án Spectral Clustering

Đồ án hướng đến việc tìm hiểu và so sánh ba phương pháp phân cụm: **K-Means**, **DBSCAN** và **Spectral Clustering**.

## Cấu trúc dự kiến

```text
Spectral-clustering/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/
│   └── spectral_clustering/
│       ├── __init__.py
│       ├── datasets.py
│       ├── preprocessing.py
│       ├── graphs.py
│       ├── spectral.py
│       ├── baselines.py
│       ├── evaluation.py
│       └── visualization.py
├── notebooks/
│   ├── 01_baselines.ipynb
│   ├── 02_graph_to_embedding.ipynb
│   ├── 03_synthetic_experiments.ipynb
│   └── 04_real_data_experiments.ipynb
├── scripts/
│   ├── run_experiments.py
│   └── demo.py
├── tests/
│   ├── test_graphs.py
│   └── test_spectral.py
├── docs/
│   ├── progress.md
│   ├── theory.md
│   ├── methodology.md
│   ├── references.md
│   ├── defense.md
│   └── hand_calc/
├── data/
│   ├── raw/
│   │   ├── banknote/
│   │   ├── image_segmentation/
│   │   └── digits/
│   └── processed/
├── results/
│   ├── tables/
│   │   ├── verification.csv
│   │   ├── results_synthetic.csv
│   │   ├── results_real.csv
│   │   └── results_final.csv
│   └── figures/
│       ├── baselines/
│       ├── spectral/
│       └── experiments/
└── reports/
    ├── report.md
    ├── report.pdf
    └── slides/
        ├── presentation.pptx
        └── presentation.pdf
```

Đây là cấu trúc dự kiến cho toàn bộ 6 tuần, không có nghĩa là tất cả file trong cây đã tồn tại. Chỉ tạo file khi nhóm bắt đầu triển khai đầu việc tương ứng; không tạo hàng loạt file rỗng chỉ để khớp cây.

- `src/` chứa các hàm dùng chung; notebook gọi lại các hàm này thay vì chép lại code.
- `results/` chứa bảng và hình do chương trình xuất ra.
- `data/raw/` giữ dữ liệu gốc; `data/processed/` giữ dữ liệu đã xử lý.
- `reports/` chứa báo cáo và slide thuyết trình.

## Chuẩn bị môi trường trên Windows

Môi trường đã được kiểm tra với **Python 3.12.10 (64-bit)**.

Mở PowerShell tại thư mục gốc của dự án rồi chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m ipykernel install --user --name spectral-clustering --display-name "Python (Spectral-clustering)"
```

Mỗi thành viên tự tạo `.venv` trên máy của mình; không sao chép hoặc đưa `.venv` lên Git.

Mở Jupyter Notebook từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m notebook
```

Trong Jupyter, chọn kernel **Python (Spectral-clustering)**. Trong VS Code, chọn Python interpreter tại `.venv\Scripts\python.exe`, sau đó chọn cùng kernel này cho notebook.

Kiểm tra nhanh phần đồ thị đã tích hợp:

```powershell
.\.venv\Scripts\python.exe scripts\demo.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Trực quan hóa kết quả phân cụm

`src/spectral_clustering/visualization.py` cung cấp hàm vẽ dữ liệu hai chiều
và hình so sánh baseline. Hai hàm chỉ vẽ dữ liệu và nhãn đã có, không tự chạy
thuật toán hay chuẩn hóa dữ liệu.

```python
import matplotlib.pyplot as plt

from spectral_clustering.visualization import (
    plot_baseline_comparison,
    plot_clusters,
)

# X, kmeans_labels và dbscan_labels đã được tạo bởi pipeline baseline.
figure_1, axis = plot_clusters(X, kmeans_labels, title="K-Means")
figure_2, axes = plot_baseline_comparison(
    X,
    kmeans_labels,
    dbscan_labels, 
    y=y,
    title="Moons - seed=42",
    save_path="results/figures/baselines/moons.png",
)

plt.show()
plt.close(figure_1)
plt.close(figure_2)
```

`X` phải có shape `(n_samples, 2)` và mỗi mảng nhãn phải có shape
`(n_samples,)`. Với DBSCAN, nhãn `-1` được quy ước là noise. Mã màu của một
cụm trong hai thuật toán là độc lập vì ID cụm chỉ là nhãn tùy ý. Người gọi
chịu trách nhiệm gọi `plt.show()` hoặc `plt.close()`.

Hình baseline chính thức được lưu trong `results/figures/baselines/`. Khi vẽ,
phải truyền đúng `X` đã dùng để chạy thuật toán. Nhãn thật chỉ dùng để đánh giá
hoặc làm panel đối chiếu có ghi rõ; không đưa nhãn thật vào đặc trưng và không
dùng nhãn thật thay cho nhãn dự đoán trong hình kết quả.

## Quy trình Git

Nhóm dùng `master` làm nhánh tích hợp. Mỗi đầu việc được thực hiện trên một nhánh riêng và đưa vào `master` qua Pull Request.

1. Cập nhật `master` (`git switch master`, rồi `git pull`).
2. Tạo branch mới theo đầu việc.
3. Commit và push branch đó lên remote.
4. Mở Pull Request vào `master`.
5. Người phụ trách tích hợp kiểm tra và merge. Pull Request của người phụ trách tích hợp cũng đi qua quy trình này; người đó tự kiểm tra rồi merge.

Mỗi notebook chỉ có **một người chỉnh sửa chính tại một thời điểm** để hạn chế xung đột khi merge.
