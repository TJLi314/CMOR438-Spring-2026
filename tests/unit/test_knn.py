import numpy as np
import pytest

from rice_ml import KNN


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def simple_classification_data():
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
    knn = KNN()
    p = np.array([0, 0])
    q = np.array([3, 4])

    assert knn.distance(p, q) == pytest.approx(5.0)


def test_train_sets_data(simple_classification_data, knn_classifier):
    X, y = simple_classification_data
    knn_classifier.train(X, y)

    assert knn_classifier.X_train is not None
    assert knn_classifier.y_train is not None
    assert knn_classifier.X_train.shape == X.shape


def test_predict_classification(simple_classification_data, knn_classifier):
    X, y = simple_classification_data
    knn_classifier.train(X, y)

    pred = knn_classifier.predict(np.array([0, 0]))
    assert pred[0] == 0

    pred = knn_classifier.predict(np.array([5, 5]))
    assert pred[0] == 1


def test_predict_regression(simple_regression_data, knn_regressor):
    X, y = simple_regression_data
    knn_regressor.train(X, y)

    pred = knn_regressor.predict(np.array([1.5]))

    assert pred[0] == pytest.approx(1.5)


def test_batch_prediction(simple_classification_data, knn_classifier):
    X, y = simple_classification_data
    knn_classifier.train(X, y)

    preds = knn_classifier.predict(X)

    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_classification_error(simple_classification_data, knn_classifier):
    X, y = simple_classification_data
    knn_classifier.train(X, y)

    error = knn_classifier.classification_error(X, y)

    assert 0 <= error <= 1
    assert error < 0.5


def test_untrained_model_raises_error(knn_classifier):
    with pytest.raises(ValueError):
        knn_classifier.predict(np.array([0, 0]))


def test_invalid_k_raises_error(simple_classification_data):
    X, y = simple_classification_data
    knn = KNN(k=10)
    knn.train(X, y)

    with pytest.raises(ValueError):
        knn.predict(np.array([0, 0]))


def test_k_equals_one(simple_classification_data):
    X, y = simple_classification_data
    knn = KNN(k=1)
    knn.train(X, y)

    preds = knn.predict(X)

    assert np.array_equal(preds, y)