import numpy as np
import pytest

from rice_ml import KNN  # change this import


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def simple_classification_data():
    """
    Linearly separable dataset for classification.
    """
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [5, 5],
        [6, 5],
        [5, 6]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def simple_regression_data():
    """
    Simple dataset for regression.
    """
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    return X, y


@pytest.fixture
def knn_classifier():
    return KNN(k=3, regression=False)


@pytest.fixture
def knn_regressor():
    return KNN(k=2, regression=True)


# ----------------------------
# Tests
# ----------------------------

def test_distance():
    """
    Distance between two points should match Euclidean distance.
    """
    knn = KNN()
    p = np.array([0, 0])
    q = np.array([3, 4])

    assert knn.distance(p, q) == pytest.approx(5.0)


def test_train_sets_data(simple_classification_data, knn_classifier):
    """
    Training should correctly store X_train and y_train.
    """
    X, y = simple_classification_data
    knn_classifier.train(X, y)

    assert knn_classifier.X_train is not None
    assert knn_classifier.y_train is not None
    assert knn_classifier.X_train.shape == X.shape


def test_predict_classification(simple_classification_data, knn_classifier):
    """
    KNN should correctly classify a simple point.
    """
    X, y = simple_classification_data
    knn_classifier.train(X, y)

    pred = knn_classifier.predict(np.array([0, 0]))
    assert pred == 0

    pred = knn_classifier.predict(np.array([5, 5]))
    assert pred == 1


def test_predict_regression(simple_regression_data, knn_regressor):
    """
    Regression should return average of nearest neighbors.
    """
    X, y = simple_regression_data
    knn_regressor.train(X, y)

    pred = knn_regressor.predict(np.array([1.5]))
    
    # nearest neighbors are 1 and 2 → average = 1.5
    assert pred == pytest.approx(1.5)


def test_predict_multiple(simple_classification_data, knn_classifier):
    """
    Batch prediction should return correct shape and values.
    """
    X, y = simple_classification_data
    knn_classifier.train(X, y)

    preds = knn_classifier.predict_multiple(X)

    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_classification_error(simple_classification_data, knn_classifier):
    """
    Classification error should be low on training data.
    """
    X, y = simple_classification_data
    knn_classifier.train(X, y)

    error = knn_classifier.classification_error(X, y)

    assert 0 <= error <= 1
    assert error < 0.5  # should perform reasonably well


def test_untrained_model_raises_error(knn_classifier):
    """
    Predicting before training should raise ValueError.
    """
    with pytest.raises(ValueError):
        knn_classifier.predict(np.array([0, 0]))


def test_invalid_k_raises_error(simple_classification_data):
    """
    k larger than dataset size should raise ValueError.
    """
    X, y = simple_classification_data
    knn = KNN(k=10)
    knn.train(X, y)

    with pytest.raises(ValueError):
        knn.predict(np.array([0, 0]))


def test_k_equals_one(simple_classification_data):
    """
    With k=1, prediction should match nearest neighbor exactly.
    """
    X, y = simple_classification_data
    knn = KNN(k=1)
    knn.train(X, y)

    for xi, yi in zip(X, y):
        assert knn.predict(xi) == yi