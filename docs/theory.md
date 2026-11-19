# Giới hạn của K-Means

K-Means phù hợp nhất khi các cụm tương đối lồi, gần dạng hình cầu, có độ phân tán tương tự nhau và được mô tả tốt bằng khoảng cách Euclidean. Thuật toán có thể cho kết quả không phản ánh cấu trúc thật của dữ liệu trong các trường hợp dưới đây.

Các trường hợp thường gặp
Cụm không lồi hoặc có hình dạng phức tạp. Hai vòng tròn đồng tâm, hai nửa mặt trăng, hình chữ S và các cụm cong không thể được tách tốt bằng các miền Voronoi quanh tâm cụm. Biên giữa hai tâm K-Means là một siêu phẳng, nên thường cắt ngang cấu trúc phi tuyến.
Cụm có kích thước, phương sai hoặc mật độ khác nhau. K-Means tối thiểu hóa tổng bình phương khoảng cách đến tâm; thuật toán có thể tách một cụm lớn hoặc gộp các vùng không phù hợp để giảm hàm mục tiêu.
Có điểm ngoại lai. Vì tâm cụm là trung bình và khoảng cách được bình phương, một điểm ở rất xa có thể kéo tâm lệch đáng kể.
Khởi tạo tâm không tốt. Thuật toán có thể hội tụ đến nghiệm cục bộ kém. Nên dùng K-Means++ và nhiều lần khởi tạo. Khi chạy thí nghiệm, cần ghi rõ random_state và n_init để tái lập kết quả.
Các đặc trưng có thang đo khác nhau. Đặc trưng có độ lớn số học lớn có thể chi phối khoảng cách Euclidean. Có thể dùng StandardScaler hoặc MinMaxScaler khi phù hợp với dữ liệu và mục tiêu phân tích.
Chọn sai số cụm K. K quá nhỏ làm gộp các nhóm khác biệt; K quá lớn làm tách một nhóm tự nhiên. Có thể tham khảo elbow, silhouette score và kiến thức của bài toán, nhưng các tiêu chí này không tự bảo đảm một K duy nhất đúng.
Dữ liệu nhiều chiều hoặc chứa đặc trưng nhiễu. Khoảng cách giữa các điểm có thể kém phân biệt hơn, còn đặc trưng không liên quan làm sai lệch độ tương đồng. Có thể chọn đặc trưng hoặc giảm chiều bằng PCA khi phù hợp.
Dữ liệu chồng lấn, có thuộc tính phân loại hoặc thay đổi theo thời gian. Đây là các tình huống cần xem lại cách biểu diễn dữ liệu, độ đo khoảng cách và thời điểm huấn luyện; không phải mọi trường hợp đều là lỗi riêng của K-Means.
Góc nhìn từ hàm mục tiêu

K-Means tối thiểu hóa within-cluster sum of squares:

$$J = \sum_{i=1}^{n} \lVert x_i - \mu_{c_i} \rVert^2,$$

trong đó x_i là điểm dữ liệu thứ i, c_i là nhãn cụm được gán và mu_(c_i) là tâm của cụm đó. Mỗi vòng lặp gồm hai bước: gán điểm vào tâm gần nhất, rồi cập nhật mỗi tâm bằng trung bình các điểm trong cụm.

Hai bước này làm J giảm hoặc giữ nguyên, nhưng không bảo đảm tìm được nghiệm toàn cục hay một phân hoạch phản ánh đúng cấu trúc mong muốn. Với hai vòng tròn đồng tâm, một đường chia gần tuyến tính có thể cho J nhỏ hơn cách gán theo vòng trong và vòng ngoài, dù cách chia đó không đúng theo cấu trúc hình học cần nhận diện.

Khi cân nhắc thuật toán khác
DBSCAN có thể phù hợp với cụm có hình dạng phức tạp và dữ liệu có nhiễu hoặc điểm ngoại lai. Nhãn noise -1 cần được giữ nguyên khi đánh giá. DBSCAN có thể gặp khó khăn khi mật độ giữa các cụm khác nhau đáng kể.
Spectral Clustering có thể phù hợp với cấu trúc phi tuyến như hai vòng tròn đồng tâm hoặc hai nửa mặt trăng. Phương pháp xây dựng ma trận tương đồng rồi dùng trị riêng và vector riêng để tạo biểu diễn trước khi phân cụm.

Việc thay thuật toán cần dựa trên cấu trúc dữ liệu và mục tiêu đánh giá, không chỉ dựa trên một ví dụ trực quan.

# Vì sao centroid khó tách cụm cong

Centroid (tâm cụm) khó tách các cụm có hình dạng cong vì nó phân cụm dựa trên khoảng cách từ mỗi điểm đến tâm cụm, chứ không dựa trên cấu trúc hình dạng hoặc sự kết nối giữa các điểm dữ liệu.

Để hiểu rõ, hãy xét cách K-Means sử dụng centroid. Giả sử có hai centroid $C_1$ và $C_2$. K-Means gán mỗi điểm dữ liệu $x$ vào centroid gần nó nhất:

$$Cluster(x)=\arg\min_{j}||x-C_j||^2$$

Khi có hai centroid, ranh giới phân chia hai cụm là tập hợp các điểm có khoảng cách đến hai centroid bằng nhau:

[

||x-C_1||^2=||x-C_2||^2 ]

Đây chính là phương trình của một đường thẳng trong không gian 2 chiều (một siêu phẳng trong không gian nhiều chiều). Nói cách khác, K-Means thường tạo ra những vùng phân chia có dạng lồi (convex), không thể linh hoạt uốn cong theo hình dạng dữ liệu.

# Giải thích similarity RBF

Trong Spectral Clustering, Similarity RBF (Radial Basis Function) được sử dụng để đo mức độ tương đồng giữa hai điểm dữ liệu $x_i$ và $x_j$ trong cùng một tập dữ liệu $X$.

### Công thức RBF:

$$W_{ij}=\exp(-\gamma\|x_i-x_j\|^2)$$

Trong đó:

$x_i,x_j$: Hai điểm dữ liệu trong tập $X$.
$\|x_i-x_j\|^2$: Bình phương khoảng cách Euclidean giữa hai điểm.
$\gamma$: Tham số điều chỉnh mức độ suy giảm của similarity.
$W_{ij}$: Trọng số thể hiện mức độ tương đồng giữa hai điểm.

Ý nghĩa trực quan:

Hai điểm gần nhau: Khoảng cách nhỏ → $W_{ij}$ lớn → Mức độ tương đồng cao. Hai điểm xa nhau: Khoảng cách lớn → $W_{ij}$ nhỏ → Mức độ tương đồng thấp.

Giả sử $\gamma=0.5$. Khoảng cách bình phương Trọng số RBF

0 1

1 0.6065

2 0.3679

4 0.1353

6 0.0498

Ta thấy khoảng cách càng lớn thì trọng số càng tiến gần về 0.

Vai trò của gamma: $\gamma$ quyết định mức độ nhạy của RBF đối với khoảng cách:

$\gamma$ lớn: Trọng số giảm nhanh khi khoảng cách tăng. Chỉ những điểm rất gần nhau mới có similarity đáng kể.
$\gamma$ nhỏ: Trọng số giảm chậm. Những điểm ở xa hơn vẫn có thể có similarity đáng kể.

Sau khi tính similarity giữa các cặp điểm trong tập $X$, ta thu được ma trận trọng số $W$.

## Quy ước của ma trận W và ví dụ
Quy ước $W_{ii}=0$. Nếu áp dụng trực tiếp công thức RBF cho $i=j$ thì $W_{ii}=\exp(0)=1$. Tuy nhiên, trong đồ thị tương đồng ta không xét cạnh nối một đỉnh với chính nó (không có self-loop), nên đặt riêng:

$$W_{ii}=0 \quad \text{với mọi } i.$$

Do đó đường chéo của $W$ luôn bằng 0, $W$ đối xứng và mọi phần tử không âm. Quy ước này cũng làm cho $D_{ii}$ chỉ cộng trọng số các cạnh nối với đỉnh khác. Hàm rbf_affinity trong mã nguồn và test test_pairwise_distance_and_rbf_known_points đều kiểm tra đường chéo bằng 0.

Từ $W$, ta tiếp tục tính:

$$D_{ii}=\sum_j W_{ij}$$

Sau đó ta tính được [ L=D-W ]

Ví dụ: đồ thị 4 đỉnh

Đây là đồ thị trong test_four_node_graph_matches_hand_calculation (test_graphs.py).

Thứ tự đỉnh: $0, 1, 2, 3$. Hàng và cột của $W$, $D$, $L$ đều theo thứ tự này.

Cạnh vô hướng và trọng số (dạng (i, j, w)):

Cạnh	Trọng số
(0, 1)	5
(1, 3)	2
(2, 3)	9
(0, 3)	3

Vì đồ thị vô hướng nên $W_{ij}=W_{ji}$; các cặp không có cạnh có trọng số 0, và $W_{ii}=0$.

$$W=\begin{bmatrix}
0 & 5 & 0 & 3\\
5 & 0 & 0 & 2\\
0 & 0 & 0 & 9\\
3 & 2 & 9 & 0
\end{bmatrix}$$

Bậc của mỗi đỉnh là tổng hàng tương ứng của $W$:

$D_{00}=5+3=8$
$D_{11}=5+2=7$
$D_{22}=9$
$D_{33}=3+2+9=14$
$$D=\begin{bmatrix}
8 & 0 & 0 & 0\\
0 & 7 & 0 & 0\\
0 & 0 & 9 & 0\\
0 & 0 & 0 & 14
\end{bmatrix}$$
$$L=D-W=\begin{bmatrix}
8 & -5 & 0 & -3\\
-5 & 7 & 0 & -2\\
0 & 0 & 9 & -9\\
-3 & -2 & -9 & 14
\end{bmatrix}$$

Kiểm tra: tổng mỗi hàng của $L$ bằng 0, tức $L\mathbf{1}=\mathbf{0}$, do $D_{ii}=\sum_j W_{ij}$ và $W_{ii}=0$.

Điểm quan trọng: RBF không trực tiếp phân cụm dữ liệu. Nó chỉ giúp chuyển dữ liệu từ không gian đặc trưng ban đầu thành một đồ thị có trọng số, làm cơ sở cho Spectral Clustering thực hiện phân cụm.