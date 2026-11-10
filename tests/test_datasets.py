import numpy as np
import pytest

from spectral_clustering.datasets import (
    SYNTHETIC_DATASETS,
    generate_anisotropic_blobs,
    generate_blobs,
    generate_circles,
    generate_moons,
    get_synthetic_dataset,
    make_anisotropic_blobs,
    make_blobs,
    make_circles,
    make_moons,
)


def test_generate_blobs_shapes_and_types() -> None:
    X, y = generate_blobs(n_samples=200, centers=3, random_state=42)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (200, 2)
    assert y.shape == (200,)
    assert X.dtype == np.float64
    assert y.dtype == np.int64
    assert len(np.unique(y)) == 3
    assert np.all(np.isfinite(X))


def test_generate_moons_shapes_and_types() -> None:
    X, y = generate_moons(n_samples=150, noise=0.08, random_state=42)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (150, 2)
    assert y.shape == (150,)
    assert X.dtype == np.float64
    assert y.dtype == np.int64
    assert set(np.unique(y)) == {0, 1}
    assert np.all(np.isfinite(X))


def test_generate_circles_shapes_and_types() -> None:
    X, y = generate_circles(n_samples=160, factor=0.4, noise=0.05, random_state=42)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (160, 2)
    assert y.shape == (160,)
    assert X.dtype == np.float64
    assert y.dtype == np.int64
    assert set(np.unique(y)) == {0, 1}
    assert np.all(np.isfinite(X))


def test_generate_anisotropic_blobs_shapes_and_types() -> None:
    X, y = generate_anisotropic_blobs(n_samples=250, centers=4, random_state=42)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (250, 2)
    assert y.shape == (250,)
    assert X.dtype == np.float64
    assert y.dtype == np.int64
    assert len(np.unique(y)) == 4
    assert np.all(np.isfinite(X))


def test_reproducibility_with_seed_and_random_state() -> None:
    # random_state vs seed should produce identical outputs
    X_rs, y_rs = generate_blobs(n_samples=100, random_state=123)
    X_seed, y_seed = generate_blobs(n_samples=100, seed=123)
    np.testing.assert_array_equal(X_rs, X_seed)
    np.testing.assert_array_equal(y_rs, y_seed)

    X_m_rs, y_m_rs = generate_moons(n_samples=100, random_state=77)
    X_m_seed, y_m_seed = generate_moons(n_samples=100, seed=77)
    np.testing.assert_array_equal(X_m_rs, X_m_seed)
    np.testing.assert_array_equal(y_m_rs, y_m_seed)

    X_c_rs, y_c_rs = generate_circles(n_samples=100, random_state=88)
    X_c_seed, y_c_seed = generate_circles(n_samples=100, seed=88)
    np.testing.assert_array_equal(X_c_rs, X_c_seed)
    np.testing.assert_array_equal(y_c_rs, y_c_seed)

    X_a_rs, y_a_rs = generate_anisotropic_blobs(n_samples=100, random_state=99)
    X_a_seed, y_a_seed = generate_anisotropic_blobs(n_samples=100, seed=99)
    np.testing.assert_array_equal(X_a_rs, X_a_seed)
    np.testing.assert_array_equal(y_a_rs, y_a_seed)


def test_conflicting_seed_and_random_state_raises() -> None:
    with pytest.raises(ValueError, match="Không thể truyền đồng thời seed và random_state"):
        generate_blobs(seed=42, random_state=43)


def test_consistent_seed_and_random_state_accepted() -> None:
    X1, y1 = generate_blobs(seed=42, random_state=42)
    X2, y2 = generate_blobs(random_state=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_custom_transformation_anisotropic() -> None:
    custom_trans = [[1.5, 0.0], [0.0, 0.5]]
    X, y = generate_anisotropic_blobs(n_samples=100, transformation=custom_trans, seed=1)
    assert X.shape == (100, 2)
    assert y.shape == (100,)


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="n_samples phải là số nguyên dương"):
        generate_blobs(n_samples=0)

    with pytest.raises(ValueError, match="n_samples phải là số nguyên dương"):
        generate_blobs(n_samples=-10)

    with pytest.raises(ValueError, match="noise phải không âm"):
        generate_moons(noise=-0.1)

    with pytest.raises(ValueError, match="factor phải nằm trong khoảng"):
        generate_circles(factor=1.5)

    with pytest.raises(ValueError, match="factor phải nằm trong khoảng"):
        generate_circles(factor=-0.1)

    with pytest.raises(ValueError, match="transformation phải có shape"):
        generate_anisotropic_blobs(transformation=[[1, 0, 0], [0, 1, 0]])


def test_aliases() -> None:
    assert make_blobs is generate_blobs
    assert make_moons is generate_moons
    assert make_circles is generate_circles
    assert make_anisotropic_blobs is generate_anisotropic_blobs


def test_get_synthetic_dataset() -> None:
    for name in ["blobs", "moons", "circles", "anisotropic", "anisotropic_blobs"]:
        assert name in SYNTHETIC_DATASETS
        X, y = get_synthetic_dataset(name, n_samples=120, seed=42)
        assert X.shape == (120, 2)
        assert y.shape == (120,)

    with pytest.raises(ValueError, match="Không tìm thấy dataset"):
        get_synthetic_dataset("non_existent_dataset")
