import numpy as np
import pytest
from rice_ml import KMeans   # change to your actual import


def test_fit_basic_clustering():
    """
    KMeans should correctly cluster two clearly separable groups.
    """
    X = np.array([
        [0, 0], [0, 1], [1, 0],     # cluster 1
        [10, 10], [10, 11], [11, 10]  # cluster 2
    ])

    model = KMeans(K=2, max_iters=100)
    model.fit(X)

    labels = model.labels_
    
    # Ensure two clusters exist
    assert set(labels) == {0, 1}

    # Check that first 3 points are in same cluster
    assert len(set(labels[:3])) == 1

    # Check that last 3 points are in same cluster
    assert len(set(labels[3:])) == 1

    # Ensure the two groups are different clusters
    assert labels[0] != labels[3]


def test_predict_cluster_matches_fit():
    """
    predict_cluster should match labels_ after fitting.
    """
    X = np.array([
        [1, 2], [1, 3], [2, 2],
        [8, 8], [9, 8], [8, 9]
    ])

    model = KMeans(K=2)
    model.fit(X)

    preds = model.predict_cluster(X)

    assert np.array_equal(preds, model.labels_)


def test_invalid_empty_dataset():
    """
    Should raise ValueError if dataset is empty.
    """
    model = KMeans(K=2)
    
    with pytest.raises(ValueError):
        model.fit(np.array([]))


def test_invalid_k_greater_than_samples():
    """
    Should raise ValueError if K > number of samples.
    """
    X = np.array([[1, 2], [3, 4]])

    model = KMeans(K=3)
    
    with pytest.raises(ValueError):
        model.fit(X)


def test_centroids_shape():
    """
    Centroids should have shape (K, n_features).
    """
    X = np.random.rand(10, 3)

    model = KMeans(K=4)
    model.fit(X)

    assert model.centroids.shape == (4, 3)


def test_labels_length():
    """
    labels_ should have same length as number of samples.
    """
    X = np.random.rand(15, 2)

    model = KMeans(K=3)
    model.fit(X)

    assert len(model.labels_) == len(X)


def test_reproducibility_with_random_state():
    """
    Same random_state should produce identical centroids (up to permutation).
    """
    X = np.random.rand(20, 2)

    model1 = KMeans(K=3, random_state=42)
    model2 = KMeans(K=3, random_state=42)

    model1.fit(X)
    model2.fit(X)

    # Sort centroids for order-independent comparison
    c1 = np.array(sorted(model1.centroids.tolist()))
    c2 = np.array(sorted(model2.centroids.tolist()))

    assert np.allclose(c1, c2)


def test_single_cluster():
    """
    With K=1, all points should belong to the same cluster.
    """
    X = np.random.rand(10, 2)

    model = KMeans(K=1)
    model.fit(X)

    assert set(model.labels_) == {0}


def test_predict_without_fit():
    """
    Predicting before fitting should raise an error.
    """
    X = np.random.rand(5, 2)

    model = KMeans(K=2)

    with pytest.raises(Exception):
        model.predict_cluster(X)