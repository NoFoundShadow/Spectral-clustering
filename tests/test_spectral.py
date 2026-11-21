import numpy as np
from spectral_clustering.graphs import normalized_laplacian
from spectral_clustering.spectral import spectral_clustering


def test_spectral_rbf_separates_two_pairs():
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    labels = spectral_clustering(X, n_clusters=2, gamma=0.5, seed=42)

    assert labels.shape == (4,)
    assert labels[0] == labels[1]  # 0.0 và 0.1 cùng cụm
    assert labels[2] == labels[3]  # 5.0 và 5.1 cùng cụm
    assert labels[0] != labels[2]  # Hai cặp thuộc hai cụm khác nhau

    from spectral_clustering.graphs import normalized_laplacian


def test_normalized_laplacian_three_nodes():
    W = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]], dtype=float)

    L = normalized_laplacian(W)
    a = 1 / np.sqrt(2)
    expected = np.array([[1, -a, 0],
                         [-a, 1, -a],
                         [0, -a, 1]])

    np.testing.assert_allclose(L, expected)
    np.testing.assert_allclose(np.linalg.eigvalsh(L), [0, 1, 2], atol=1e-12)