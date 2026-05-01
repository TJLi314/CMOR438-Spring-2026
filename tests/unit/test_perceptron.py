import numpy as np
import pytest

from rice_ml import Perceptron

def test_initialization():
    model = Perceptron(learning_rate=0.1, epochs=10)
    assert model.learning_rate == 0.1
    assert model.epochs == 10
    assert model.weights_ is None
    assert model.errors_ == []


def test_predict_before_training_raises_error():
    model = Perceptron()
    X = np.array([1, 2])
    
    with pytest.raises(TypeError):
        model.predict(X)


def test_training_updates_weights():
    X = np.array([[1, 1], [2, 2]])
    y = np.array([1, 1])

    model = Perceptron(epochs=1)
    model.train(X, y)

    assert model.weights_ is not None
    assert len(model.weights_) == X.shape[1] + 1


def test_linearly_separable_data_converges():
    # Simple AND-like dataset (linearly separable)
    X = np.array([
        [1, 1],
        [2, 2],
        [-1, -1],
        [-2, -2]
    ])
    y = np.array([1, 1, -1, -1])

    model = Perceptron(learning_rate=0.1, epochs=100)
    model.train(X, y)

    predictions = np.array([model.predict(xi) for xi in X])
    assert np.array_equal(predictions, y)


def test_errors_decrease_or_reach_zero():
    X = np.array([
        [1, 1],
        [2, 2],
        [-1, -1],
        [-2, -2]
    ])
    y = np.array([1, 1, -1, -1])

    model = Perceptron(learning_rate=0.1, epochs=20)
    model.train(X, y)

    # Either converged early or errors trend downward
    assert len(model.errors_) > 0
    assert model.errors_[-1] == 0 or model.errors_[-1] <= model.errors_[0]


def test_predict_output_values():
    X = np.array([[1, 1], [-1, -1]])
    y = np.array([1, -1])

    model = Perceptron(learning_rate=0.1, epochs=50)
    model.train(X, y)

    preds = model.predict(X)
    assert set(preds).issubset({-1, 1})


def test_net_input_shape():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, -1])

    model = Perceptron(epochs=5)
    model.train(X, y)

    output = model.net_input(X)
    assert output.shape == (2,)