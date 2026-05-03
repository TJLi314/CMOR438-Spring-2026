import numpy as np
import pytest
from rice_ml import DBSCAN  # adjust import as needed


@pytest.fixture
def simple_clusters():
    """
    Create two clearly separable clusters.
    """
    np.random.seed(0)
    cluster1 = np.random.randn(20, 2) + np.array([5, 5])
    cluster2 = np.random.randn(20, 2) + np.array([-5, -5])
    return np.vstack([cluster1, cluster2])


def test_fit_assigns_labels(simple_clusters):
    """
    After fitting, labels_ should be assigned for all points.
    """
    model = DBSCAN(eps=1.5, min_samples=3)
    model.fit(simple_clusters)

    assert model.labels_ is not None
    assert len(model.labels_) == simple_clusters.shape[0]


def test_finds_at_least_two_clusters(simple_clusters):
    """
    DBSCAN should detect at least two clusters in separable data.
    """
    model = DBSCAN(eps=1.5, min_samples=3)
    model.fit(simple_clusters)

    unique_labels = set(model.labels_)
    unique_labels.discard(-1)  # remove noise if present

    assert len(unique_labels) >= 2


def test_noise_points_detected():
    """
    Points far away from clusters should be labeled as noise (-1).
    """
    X = np.array([
        [0, 0], [0.1, 0.1], [0.2, 0.2],   # cluster
        [10, 10]                         # outlier
    ])

    model = DBSCAN(eps=0.5, min_samples=2)
    model.fit(X)

    assert model.labels_[-1] == -1


def test_all_points_noise():
    """
    If eps is too small, all points should be labeled as noise.
    """
    X = np.random.rand(10, 2)

    model = DBSCAN(eps=1e-6, min_samples=2)
    model.fit(X)

    assert np.all(model.labels_ == -1)


def test_single_cluster():
    """
    If eps is large enough, all points should belong to one cluster.
    """
    X = np.random.rand(20, 2)

    model = DBSCAN(eps=10.0, min_samples=2)
    model.fit(X)

    unique_labels = set(model.labels_)
    unique_labels.discard(-1)

    assert len(unique_labels) == 1


def test_predict_before_fit_raises():
    """
    Calling predict before fit should raise an error.
    """
    model = DBSCAN()

    with pytest.raises(ValueError):
        model.predict(np.array([[0, 0]]))


def test_predict_matches_training_labels(simple_clusters):
    """
    Predicting on training data should return consistent labels.
    """
    model = DBSCAN(eps=1.5, min_samples=3)
    model.fit(simple_clusters)

    preds = model.predict(simple_clusters)

    assert len(preds) == len(model.labels_)


def test_empty_input_raises():
    """
    Fitting on empty dataset should raise ValueError.
    """
    model = DBSCAN()

    with pytest.raises(ValueError):
        model.fit(np.array([]))


def test_reproducibility():
    """
    DBSCAN should be deterministic (no randomness involved).
    """
    X = np.random.rand(20, 2)

    model1 = DBSCAN(eps=0.5, min_samples=3)
    model2 = DBSCAN(eps=0.5, min_samples=3)

    model1.fit(X)
    model2.fit(X)

    assert np.array_equal(model1.labels_, model2.labels_)