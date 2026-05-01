import numpy as np
import pytest

from rice_ml import LogisticRegression 


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def simple_linearly_separable_data():
    """
    Simple 2D dataset that is linearly separable.
    """
    X = np.array([
        [0.0, 0.0],
        [0.2, 0.1],
        [0.1, 0.3],
        [0.9, 0.8],
        [1.0, 1.0],
        [0.8, 0.9]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def model():
    """
    Small deterministic model for testing.
    """
    return LogisticRegression(eta=0.1, epochs=10)


# ----------------------------
# Tests
# ----------------------------

def test_sigmoid_bounds(model):
    z = np.array([-100, -1, 0, 1, 100])
    out = model.sigmoid(z)

    assert np.all(out >= 0)
    assert np.all(out <= 1)


def test_cross_entropy_loss_non_negative(model):
    """
    Cross entropy loss should always be >= 0.
    """
    loss = model.cross_entropy_loss(0.9, 1)
    assert loss >= 0


def test_train_updates_weights(simple_linearly_separable_data, model):
    """
    Training should change weights (sanity check).
    """
    X, y = simple_linearly_separable_data

    initial_weights = model.w_ if model.w_ is not None else None

    model.train(X, y)

    assert model.w_ is not None

    # weights should not stay identical (very high probability they change)
    if initial_weights is not None:
        assert not np.allclose(initial_weights, model.w_)


def test_predict_shape(simple_linearly_separable_data, model):
    """
    Predict should return correct shape and binary values.
    """
    X, y = simple_linearly_separable_data

    model.train(X, y)
    preds = model.predict(X)

    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})


def test_predict_proba_range(simple_linearly_separable_data, model):
    """
    Predict probabilities should be in [0, 1].
    """
    X, y = simple_linearly_separable_data

    model.train(X, y)
    probs = model.predict_proba(X)

    assert np.all(probs >= 0)
    assert np.all(probs <= 1)


def test_loss_decreases_over_time(simple_linearly_separable_data, model):
    """
    Loss should generally decrease or stay stable for simple data.
    (Not strictly monotonic due to SGD, but trend should improve.)
    """
    X, y = simple_linearly_separable_data

    model.train(X, y)
    errors = model.errors_

    assert len(errors) == model.epochs
    assert np.mean(errors[:3]) >= np.mean(errors[-3:])


def test_decision_boundary_runs(simple_linearly_separable_data, model):
    """
    Smoke test: decision boundary function should run without crashing.
    """
    X, y = simple_linearly_separable_data

    model.train(X, y)

    # should not throw
    model.plot_decision_boundary(X, y)