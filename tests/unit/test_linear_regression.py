import numpy as np
import pytest

from rice_ml import LinearRegression


# -----------------------------
# Helper: simple synthetic data
# -----------------------------
def simple_dataset():
    # y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])
    return X, y


# -----------------------------
# OLS TEST
# -----------------------------
def test_ols_correctness():
    X, y = simple_dataset()

    model = LinearRegression(method="ols")
    model.fit(X, y)

    preds = model.predict(X)

    # Should recover near-perfect linear relationship
    assert np.allclose(preds, y, atol=1e-6)

    # Check learned parameters
    assert np.isclose(model.w_[0], 2.0, atol=1e-6)
    assert np.isclose(model.bias, 1.0, atol=1e-6)


# -----------------------------
# GD TEST (convergence)
# -----------------------------
def test_gd_convergence():
    X, y = simple_dataset()

    model = LinearRegression(method="gd", eta=0.01, epochs=2000)
    model.fit(X, y)

    preds = model.predict(X)

    # Should converge close to true values
    assert np.allclose(preds, y, atol=1e-1)

    # Loss should generally decrease over time
    assert model.errors_[0] > model.errors_[-1]


# -----------------------------
# PREDICTION SHAPE TEST
# -----------------------------
def test_predict_shape():
    X, y = simple_dataset()

    model = LinearRegression(method="ols")
    model.fit(X, y)

    X_test = np.array([[10], [20], [30]])
    preds = model.predict(X_test)

    assert preds.shape == (3,)


# -----------------------------
# INPUT VALIDATION TEST
# -----------------------------
def test_invalid_method():
    with pytest.raises(ValueError):
        LinearRegression(method="invalid")


# -----------------------------
# SHAPE SAFETY TEST
# -----------------------------
def test_y_flattening():
    X = np.array([[1], [2], [3]])
    y = np.array([[2], [4], [6]])  # column vector

    model = LinearRegression(method="ols")
    model.fit(X, y)

    # should still work correctly
    preds = model.predict(X)
    assert preds.shape == (3,)