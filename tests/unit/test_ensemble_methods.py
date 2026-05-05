import numpy as np

from rice_ml import DecisionTreeClassifier, StumpBaggingClassifier, StumpBaggingRegressor, MajorityVoteClassifier


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def make_classification_data():
    """
    Simple linearly separable dataset.
    """
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [5, 5],
        [5, 6],
        [6, 5]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def make_regression_data():
    """
    Simple linear regression dataset: y = x0 + x1
    """
    X = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4]
    ])
    y = np.array([0, 2, 4, 6, 8])
    return X, y


# -------------------------------------------------------------------
# 1. StumpBaggingClassifier Tests
# -------------------------------------------------------------------

def test_stump_bagging_classifier_fit_predict():
    X, y = make_classification_data()

    model = StumpBaggingClassifier(n_models=5, seed=42)
    model.fit(X, y)

    preds = model.predict(X)

    # shape check
    assert preds.shape == y.shape

    # should achieve perfect or near-perfect accuracy on easy dataset
    acc = model.score(X, y)
    assert acc >= 0.83  # allow slight randomness tolerance


def test_stump_bagging_classifier_deterministic():
    X, y = make_classification_data()

    m1 = StumpBaggingClassifier(n_models=5, seed=123).fit(X, y)
    m2 = StumpBaggingClassifier(n_models=5, seed=123).fit(X, y)

    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


# -------------------------------------------------------------------
# 2. StumpBaggingRegressor Tests
# -------------------------------------------------------------------

def test_stump_bagging_regressor_fit_predict():
    X, y = make_regression_data()

    model = StumpBaggingRegressor(n_models=5, seed=42)
    model.fit(X, y)

    preds = model.predict(X)

    # shape check
    assert preds.shape == y.shape

    # should fit linear trend reasonably well
    r2 = model.score(X, y)
    assert r2 > 0.65


# -------------------------------------------------------------------
# 3. Majority Vote Classifier Tests
# -------------------------------------------------------------------

def test_majority_vote_classifier_basic():
    X, y = make_classification_data()

    # Base models (must be pre-trained or trained in wrapper)
    model1 = DecisionTreeClassifier(max_depth=1)
    model2 = DecisionTreeClassifier(max_depth=2)
    model3 = DecisionTreeClassifier(max_depth=3)

    # train models externally (as intended by your design)
    model1.fit(X, y)
    model2.fit(X, y)
    model3.fit(X, y)

    ensemble = MajorityVoteClassifier([model1, model2, model3])

    preds = ensemble.predict(X)

    # shape check
    assert preds.shape == y.shape

    # should perform at least as good as weak trees
    acc = ensemble.score(X, y)
    assert acc >= 0.83


def test_majority_vote_consistency():
    X, y = make_classification_data()

    model1 = DecisionTreeClassifier(max_depth=1).fit(X, y)
    model2 = DecisionTreeClassifier(max_depth=1).fit(X, y)

    ensemble = MajorityVoteClassifier([model1, model2])

    preds_a = ensemble.predict(X)
    preds_b = ensemble.predict(X)

    # deterministic output
    np.testing.assert_array_equal(preds_a, preds_b)