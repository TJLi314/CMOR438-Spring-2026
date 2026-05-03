import numpy as np
import pytest

from rice_ml import PCA


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def simple_data():
    """
    Create a simple correlated dataset.
    """
    np.random.seed(0)
    x = np.random.rand(50)
    y = x * 2 + np.random.normal(0, 0.01, 50)  # highly correlated
    z = np.random.rand(50)
    return np.vstack([x, y, z]).T


# ----------------------------
# Tests
# ----------------------------

def test_fit_sets_attributes(simple_data):
    """
    After fitting, PCA should have components and mean.
    """
    pca = PCA(num_components=2)
    pca.fit(simple_data)

    assert pca.pc_axes_ is not None
    assert pca.data_mean_ is not None
    assert pca.pc_axes_.shape == (2, 3)


def test_transform_shape(simple_data):
    """
    Transform should reduce dimensionality correctly.
    """
    pca = PCA(num_components=2)
    pca.fit(simple_data)

    transformed = pca.project(simple_data)

    assert transformed.shape == (simple_data.shape[0], 2)


def test_fit_project_equivalence(simple_data):
    """
    fit_project should match fit + project.
    """
    pca1 = PCA(num_components=2)
    pca2 = PCA(num_components=2)

    out1 = pca1.fit_project(simple_data)
    pca2.fit(simple_data)
    out2 = pca2.project(simple_data)

    assert np.allclose(out1, out2)


def test_inverse_transform_reconstruction(simple_data):
    """
    Reconstruction should approximately recover original data.
    (lossy but close for low-dimensional projection)
    """
    pca = PCA(num_components=2)
    pca.fit(simple_data)

    reduced = pca.project(simple_data)
    reconstructed = pca.reconstruct(reduced)

    assert reconstructed.shape == simple_data.shape

    # reconstruction should be reasonably close
    error = np.mean((simple_data - reconstructed) ** 2)
    assert error < 0.1


def test_variance_ratio_properties(simple_data):
    """
    Variance ratios should be between 0 and 1 and sum <= 1.
    """
    pca = PCA(num_components=2)
    pca.fit(simple_data)

    vr = pca.var_ratio_

    assert np.all(vr >= 0)
    assert np.all(vr <= 1)
    assert np.sum(vr) <= 1.0


def test_principal_components_orthogonality(simple_data):
    """
    Principal components should be orthonormal.
    """
    pca = PCA(num_components=3)
    pca.fit(simple_data)

    comps = pca.pc_axes_

    # dot product should be close to identity matrix
    identity = comps @ comps.T
    assert np.allclose(identity, np.eye(3), atol=1e-6)


def test_n_components_too_large():
    """
    Should raise error if num_components > features.
    """
    X = np.random.rand(10, 3)

    with pytest.raises(ValueError):
        PCA(num_components=5).fit(X)


def test_non_2d_input_raises():
    """
    PCA should reject 1D input.
    """
    X = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError):
        PCA(num_components=1).fit(X)


def test_unfitted_transform_raises():
    """
    Calling transform before fit should raise error.
    """
    X = np.random.rand(5, 3)
    pca = PCA(num_components=2)

    with pytest.raises(RuntimeError):
        pca.project(X)


def test_deterministic_output(simple_data):
    """
    PCA should be deterministic for same input.
    """
    pca1 = PCA(num_components=2)
    pca2 = PCA(num_components=2)

    out1 = pca1.fit_project(simple_data)
    out2 = pca2.fit_project(simple_data)

    assert np.allclose(out1, out2)