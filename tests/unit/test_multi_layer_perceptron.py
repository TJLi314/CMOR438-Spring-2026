import numpy as np

from rice_ml import MLP

def make_toy_data():
    """
    Simple 2-class separable dataset.
    """
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.2],
        [0.2, 0.1],
        [1.0, 1.0],
        [1.1, 1.2],
        [1.2, 1.1],
    ])

    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_forward_output_shape():
    model = MLP([2, 5, 3])
    X = np.random.randn(10, 2)

    probs = model.forward(X)

    assert probs.shape == (10, 3)


def test_softmax_properties():
    model = MLP([2, 4, 3])
    X = np.random.randn(5, 2)

    probs = model.forward(X)

    # probabilities sum to 1
    sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-6)

    # probabilities in [0, 1]
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)


def test_loss_is_finite():
    model = MLP([2, 4, 3])
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 3, size=10)

    loss = model.loss(X, y)

    assert np.isfinite(loss)
    assert loss > 0


def test_predict_shape_and_range():
    model = MLP([2, 6, 4, 3])
    X = np.random.randn(7, 2)

    preds = model.predict(X)

    assert preds.shape == (7,)
    assert np.all(preds >= 0)
    assert np.all(preds < 3)


def test_training_reduces_loss():
    X, y = make_toy_data()

    model = MLP([2, 10, 2], lr=0.1, reg=0.0)

    initial_loss = model.loss(X, y)
    model.train(X, y, steps=200)

    final_loss = model.loss(X, y)

    assert final_loss < initial_loss


def test_model_can_overfit_small_dataset():
    """
    If the implementation is correct, the model should be able to
    nearly memorize a tiny dataset.
    """
    X, y = make_toy_data()

    model = MLP([2, 20, 2], lr=0.1, reg=0.0)

    model.train(X, y, steps=1500, verbose=False)

    preds = model.predict(X)

    accuracy = np.mean(preds == y)

    assert accuracy >= 0.95


def test_backward_returns_grad_shapes():
    model = MLP([2, 5, 3])
    X = np.random.randn(8, 2)
    y = np.random.randint(0, 3, size=8)

    grads, dW_out, db_out = model.backward(X, y)

    # output layer gradients
    assert dW_out.shape == model.W_out.shape
    assert db_out.shape == model.b_out.shape

    # hidden layer gradients
    assert len(grads) == len(model.blocks)

    for (dW, db), block in zip(grads, model.blocks):
        assert dW.shape == block.W.shape
        assert db.shape == block.b.shape


def test_parameters_change_after_training():
    model = MLP([2, 5, 2], lr=0.1, reg=0.0)

    X = np.random.randn(10, 2)
    y = np.random.randint(0, 2, size=10)

    W_before = model.W_out.copy()
    b_before = model.b_out.copy()

    model.train(X, y, steps=5, verbose=False)

    assert not np.allclose(W_before, model.W_out)
    assert not np.allclose(b_before, model.b_out)