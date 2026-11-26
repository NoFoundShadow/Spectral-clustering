from numpy.typing import ArrayLike
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def calculate_ari(y_true: ArrayLike, y_pred:ArrayLike):
    """Tính Adjusted Rand Index giữa nhãn thật và nhãn dự đoán."""
    return float(adjusted_rand_score(y_true, y_pred))


def calculate_nmi(y_true: ArrayLike, y_pred: ArrayLike):
    """Tính Normalized Mutual Information giữa nhãn thật và nhãn dự đoán."""
    return float(normalized_mutual_info_score(y_true, y_pred))


# 1 test nhỏ vụ đảo nhãn [0, 0, 1, 1] với [1, 1, 0, 0]
y_true = [0, 0, 1, 1]
y_pred = [1, 1, 0, 0]
if calculate_ari(y_true, y_pred) == 1.0 and calculate_nmi(y_true, y_pred) == 1.0:
    print("OK: cả 2 test đều pass")
