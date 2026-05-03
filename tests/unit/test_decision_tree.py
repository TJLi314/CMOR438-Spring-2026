import numpy as np
import pytest

from rice_ml import DecisionTreeClassifier, DecisionTreeRegressor


# ============================================================
# Helpers
# ============================================================

def toy_classification_data():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([0, 0, 1, 1])
    return X, y


def toy_regression_data():
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    return X, y


# ============================================================
# CLASSIFIER TESTS
# ============================================================

def test_classifier_fit_predict_shapes():
    X, y = toy_classification_data()

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape


def test_classifier_perfect_fit_small_data():
    X, y = toy_classification_data()

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)

    acc = model.score(X, y)

    assert acc == 1.0


def test_classifier_prediction_values_valid():
    X, y = toy_classification_data()

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    preds = model.predict(X)

    # predictions must be only known class labels
    assert set(preds).issubset(set(y))


def test_classifier_untrained_model_error():
    model = DecisionTreeClassifier()

    X = np.array([[1, 2]])

    with pytest.raises(RuntimeError):
        model.predict(X)


def test_classifier_single_class():
    X = np.random.randn(10, 2)
    y = np.zeros(10, dtype=int)

    model = DecisionTreeClassifier()
    model.fit(X, y)

    preds = model.predict(X)

    assert np.all(preds == 0)


# ============================================================
# REGRESSOR TESTS
# ============================================================

def test_regressor_fit_predict_shape():
    X, y = toy_regression_data()

    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape


def test_regressor_perfect_fit_small_data():
    X, y = toy_regression_data()

    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X, y)

    preds = model.predict(X)

    # should be very close (trees can interpolate small datasets)
    assert np.allclose(preds, y, atol=1e-6)


def test_regressor_score_range():
    X, y = toy_regression_data()

    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X, y)

    r2 = model.score(X, y)

    assert r2 <= 1.0
    assert r2 >= -1e-9  # numerical tolerance


def test_regressor_constant_target():
    X = np.random.randn(20, 3)
    y = np.ones(20) * 7.0

    model = DecisionTreeRegressor()
    model.fit(X, y)

    preds = model.predict(X)

    assert np.allclose(preds, 7.0)


def test_regressor_untrained_error():
    model = DecisionTreeRegressor()

    X = np.array([[1, 2]])

    with pytest.raises(RuntimeError):
        model.predict(X)


# ============================================================
# CONSISTENCY TESTS
# ============================================================

def test_tree_deterministic_behavior():
    X, y = toy_classification_data()

    m1 = DecisionTreeClassifier(max_depth=3).fit(X, y)
    m2 = DecisionTreeClassifier(max_depth=3).fit(X, y)

    assert np.array_equal(m1.predict(X), m2.predict(X))


def test_regressor_deterministic_behavior():
    X, y = toy_regression_data()

    m1 = DecisionTreeRegressor(max_depth=3).fit(X, y)
    m2 = DecisionTreeRegressor(max_depth=3).fit(X, y)

    assert np.allclose(m1.predict(X), m2.predict(X))