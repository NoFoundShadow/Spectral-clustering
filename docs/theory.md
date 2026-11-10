# Giới hạn của K-Means

K-Means phù hợp nhất khi các cụm tương đối lồi, gần dạng hình cầu, có độ phân tán tương tự nhau và được mô tả tốt bằng khoảng cách Euclidean. Thuật toán có thể cho kết quả không phản ánh cấu trúc thật của dữ liệu trong các trường hợp dưới đây.

## Các trường hợp thường gặp

1. **Cụm không lồi hoặc có hình dạng phức tạp.** Hai vòng tròn đồng tâm, hai nửa mặt trăng, hình chữ S và các cụm cong không thể được tách tốt bằng các miền Voronoi quanh tâm cụm. Biên giữa hai tâm K-Means là một siêu phẳng, nên thường cắt ngang cấu trúc phi tuyến.
2. **Cụm có kích thước, phương sai hoặc mật độ khác nhau.** K-Means tối thiểu hóa tổng bình phương khoảng cách đến tâm; thuật toán có thể tách một cụm lớn hoặc gộp các vùng không phù hợp để giảm hàm mục tiêu.
3. **Có điểm ngoại lai.** Vì tâm cụm là trung bình và khoảng cách được bình phương, một điểm ở rất xa có thể kéo tâm lệch đáng kể.
4. **Khởi tạo tâm không tốt.** Thuật toán có thể hội tụ đến nghiệm cục bộ kém. Nên dùng K-Means++ và nhiều lần khởi tạo. Khi chạy thí nghiệm, cần ghi rõ `random_state` và `n_init` để tái lập kết quả.
5. **Các đặc trưng có thang đo khác nhau.** Đặc trưng có độ lớn số học lớn có thể chi phối khoảng cách Euclidean. Có thể dùng `StandardScaler` hoặc `MinMaxScaler` khi phù hợp với dữ liệu và mục tiêu phân tích.
6. **Chọn sai số cụm K.** K quá nhỏ làm gộp các nhóm khác biệt; K quá lớn làm tách một nhóm tự nhiên. Có thể tham khảo elbow, silhouette score và kiến thức của bài toán, nhưng các tiêu chí này không tự bảo đảm một K duy nhất đúng.
7. **Dữ liệu nhiều chiều hoặc chứa đặc trưng nhiễu.** Khoảng cách giữa các điểm có thể kém phân biệt hơn, còn đặc trưng không liên quan làm sai lệch độ tương đồng. Có thể chọn đặc trưng hoặc giảm chiều bằng PCA khi phù hợp.
8. **Dữ liệu chồng lấn, có thuộc tính phân loại hoặc thay đổi theo thời gian.** Đây là các tình huống cần xem lại cách biểu diễn dữ liệu, độ đo khoảng cách và thời điểm huấn luyện; không phải mọi trường hợp đều là lỗi riêng của K-Means.

## Góc nhìn từ hàm mục tiêu

K-Means tối thiểu hóa within-cluster sum of squares:

\[
J = \sum_{i=1}^{n} \lVert x_i - \mu_{c_i} \rVert^2,
\]

trong đó `x_i` là điểm dữ liệu thứ `i`, `c_i` là nhãn cụm được gán và `mu_(c_i)` là tâm của cụm đó. Mỗi vòng lặp gồm hai bước: gán điểm vào tâm gần nhất, rồi cập nhật mỗi tâm bằng trung bình các điểm trong cụm.

Hai bước này làm `J` giảm hoặc giữ nguyên, nhưng không bảo đảm tìm được nghiệm toàn cục hay một phân hoạch phản ánh đúng cấu trúc mong muốn. Với hai vòng tròn đồng tâm, một đường chia gần tuyến tính có thể cho `J` nhỏ hơn cách gán theo vòng trong và vòng ngoài, dù cách chia đó không đúng theo cấu trúc hình học cần nhận diện.

## Khi cân nhắc thuật toán khác

- **DBSCAN** có thể phù hợp với cụm có hình dạng phức tạp và dữ liệu có nhiễu hoặc điểm ngoại lai. Nhãn noise `-1` cần được giữ nguyên khi đánh giá. DBSCAN có thể gặp khó khăn khi mật độ giữa các cụm khác nhau đáng kể.
- **Spectral Clustering** có thể phù hợp với cấu trúc phi tuyến như hai vòng tròn đồng tâm hoặc hai nửa mặt trăng. Phương pháp xây dựng ma trận tương đồng rồi dùng trị riêng và vector riêng để tạo biểu diễn trước khi phân cụm.

Việc thay thuật toán cần dựa trên cấu trúc dữ liệu và mục tiêu đánh giá, không chỉ dựa trên một ví dụ trực quan.
