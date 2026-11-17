# Tiến độ học tập và ghi chú nhóm

Mỗi thành viên tự điền trực tiếp vào dòng của mình theo từng giai đoạn.

## W1+W2: Học chung Phần 1 (Baselines: K-Means và DBSCAN)

| Thành viên | Điều quan sát được trên 4 bộ dữ liệu | Điểm còn thắc mắc / cần thảo luận |
| --- | --- | --- |
| A | | |
| B | | |
| C | K-Means tách tốt `blobs` nhưng cắt ngang cấu trúc cong của `moons`, `circles` và cắt lệch cụm kéo dãn `anisotropic`. DBSCAN với `eps=0.25, min_samples=5` tách gọn `blobs`, `moons`, `anisotropic`, còn `circles` phải tăng lên `eps=0.40, min_samples=5` mới không bị vỡ thành nhiều cụm nhỏ. | Cách chọn `eps` và `gamma` khi sang dữ liệu nhiều chiều không vẽ scatter 2D được. |
| D | | |

## W1+W2: Học chung Phần 2 (Tính tay W, D, L và đối chiếu chéo)

### Bài giải tay và kết quả quiz

| Thành viên | File bài tính tay | Thứ tự đỉnh và số cạnh | Kết quả kiểm tra bằng NumPy | Kết quả chấm chéo / lỗi cần sửa |
| --- | --- | --- | --- | --- |
| A | | | | |
| B | | | | |
| C | | | | |
| D | | | | |

### Đối chiếu graph mẫu với code (`graph_matrices_from_edges`)

- Người kiểm tra (A): (ghi kết quả so khớp từng ô `W`, `D`, `L` giữa bài mẫu trong `docs/theory.md` và hàm `graph_matrices_from_edges`, cùng ghi chú nếu có lệch ký hiệu).

## W3: Học chung (Mini-lab Spectral Clustering trên Moons 40 điểm)

| Thành viên | Kết quả chạy mini-lab (`eigh(L)` + K-Means trên `U`, thử đảo nhãn `0/1`) | Câu hỏi / điểm chưa hiểu |
| --- | --- | --- |
| A | | |
| B | | |
| C | | |
| D | | |

## W6: Tổng duyệt và luyện phản biện

| Thành viên | Kết quả lần tập 1 | Kết quả lần tập 2 (đổi chéo phần) | Phần còn yếu cần bổ sung |
| --- | --- | --- | --- |
| A | | | |
| B | | | |
| C | | | |
| D | | | |
