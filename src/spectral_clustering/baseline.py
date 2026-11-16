#import data từ file datasets.py
from datasets import generate_circles #(loại dữ liệu)
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

#DBSCAN của mẫu dữ liệu circles từ file datasets.py
X, y = generate_circles(seed=0)
labels = DBSCAN(eps=0.15, min_samples=5).fit_predict(X)

#Kết quả phân loại mẫu của DBSCAN
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()

