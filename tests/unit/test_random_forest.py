import numpy as np
import pytest

from rice_ml import RandomForestClassifier, RandomForestRegressor


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def classification_data():
    # Simple linearly separable dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 2],
        [2, 3],
        [3, 2],
        [3, 3],
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    return X, y


@pytest.fixture
def regression_data():
    # Simple linear relationship: y = x0 + x1
    X = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
    ])
    y = np.array([0, 2, 4, 6, 8])
    return X, y


# -----------------------------
# Classifier Tests
# -----------------------------

def test_rf_classifier_fit_predict_shape(classification_data):
    X, y = classification_data

    model = RandomForestClassifier(n_trees=10, max_depth=3, seed=42)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)).issubset(set(y))


def test_rf_classifier_overfits_small_data(classification_data):
    X, y = classification_data

    model = RandomForestClassifier(n_trees=20, max_depth=None, seed=1)
    model.fit(X, y)

    preds = model.predict(X)

    # On tiny dataset, RF should fit very well
    acc = (preds == y).mean()
    assert acc >= 0.75


def test_rf_classifier_deterministic_with_seed(classification_data):
    X, y = classification_data

    model1 = RandomForestClassifier(n_trees=10, seed=123)
    model2 = RandomForestClassifier(n_trees=10, seed=123)

    model1.fit(X, y)
    model2.fit(X, y)

    np.testing.assert_array_equal(model1.predict(X), model2.predict(X))


def test_rf_classifier_requires_fit():
    model = RandomForestClassifier()

    with pytest.raises(RuntimeError):
        model.predict(np.array([[1, 2]]))


# -----------------------------
# Regressor Tests
# -----------------------------

def test_rf_regressor_fit_predict_shape(regression_data):
    X, y = regression_data

    model = RandomForestRegressor(n_trees=10, max_depth=3, seed=42)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == (X.shape[0],)
    assert np.issubdtype(preds.dtype, np.floating)


def test_rf_regressor_reasonable_fit(regression_data):
    X, y = regression_data

    model = RandomForestRegressor(n_trees=30, max_depth=None, seed=0)
    model.fit(X, y)

    preds = model.predict(X)
    r2 = model.score(X, y)

    # Should strongly fit this simple linear function
    assert r2 > 0.9


def test_rf_regressor_deterministic(regression_data):
    X, y = regression_data

    model1 = RandomForestRegressor(n_trees=10, seed=99)
    model2 = RandomForestRegressor(n_trees=10, seed=99)

    model1.fit(X, y)
    model2.fit(X, y)

    np.testing.assert_allclose(model1.predict(X), model2.predict(X))


def test_rf_regressor_requires_fit():
    model = RandomForestRegressor()

    with pytest.raises(RuntimeError):
        model.predict(np.array([[1, 2]]))


# -----------------------------
# Edge Case Tests
# -----------------------------

def test_single_feature_case():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])

    clf = RandomForestClassifier(n_trees=5, seed=0)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == y.shape


def test_constant_target_regression():
    X = np.random.randn(20, 3)
    y = np.ones(20)

    model = RandomForestRegressor(n_trees=5, seed=0)
    model.fit(X, y)

    preds = model.predict(X)

    # Should predict constant values close to 1
    assert np.allclose(preds, 1.0, atol=0.2)