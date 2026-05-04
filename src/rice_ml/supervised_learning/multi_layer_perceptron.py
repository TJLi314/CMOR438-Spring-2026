import numpy as np

def activate(x, mode="tanh"):
    """
    Apply a nonlinear activation function elementwise. 
    Computes the activation of input array `x` using the specified mode.

    Parameters
    ----------
    x : np.ndarray
        Input array (pre-activation values).
    mode : str
        Activation type. Options: "tanh", "sigmoid", "relu".

    Returns
    -------
    np.ndarray
        Activated output.
    """
    if mode == "tanh":
        return np.tanh(x)
    if mode == "sigmoid":
        return 1 / (1 + np.exp(-x))
    if mode == "relu":
        return np.maximum(0, x)
    raise ValueError("Unknown activation")


def activate_grad(x, mode="tanh"):
    """
    Compute derivative of activation function.
    Returns gradient of the activation function evaluated at `x`.

    Parameters
    ----------
    x : np.ndarray
        Pre-activation values.
    mode : str
        Activation type. Options: "tanh", "sigmoid", "relu".

    Returns
    -------
    np.ndarray
        Elementwise derivative of activation.
    """
    if mode == "tanh":
        return 1 - np.tanh(x) ** 2
    if mode == "sigmoid":
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)
    if mode == "relu":
        return (x > 0).astype(float)
    raise ValueError("Unknown activation")


def softmax(x):
    """
    Compute softmax probabilities.
    Converts raw logits into probability distribution over classes.

    Parameters
    ----------
    x : np.ndarray
        Logits of shape (n_samples, n_classes).

    Returns
    -------
    np.ndarray
        Softmax probabilities of same shape as input.
    """
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ---------------------------
# Dense Layer
# ---------------------------

class Layer:
    """
    Fully connected neural network layer with activation.
    Performs affine transformation followed by nonlinear activation.
    Stores intermediate values for backpropagation.
    """

    def __init__(self, in_size, out_size, nonlin="tanh", seed=None):
        """
        Initialize layer parameters.

        Parameters
        ----------
        in_size : int
            Number of input features.
        out_size : int
            Number of output neurons.
        nonlin : str
            Activation function type.
        seed : int or None
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)

        self.W = rng.standard_normal((in_size, out_size)) / np.sqrt(in_size)
        self.b = np.zeros((1, out_size))

        self.nonlin = nonlin

        self.x_in = None
        self.lin_out = None
        self.act_out = None

    def forward(self, x):
        """
        Forward propagation through layer.

        Parameters
        ----------
        x : np.ndarray
            Input data of shape (batch_size, in_size).

        Returns
        -------
        np.ndarray
            Activated output of shape (batch_size, out_size).
        """
        self.x_in = x
        self.lin_out = x @ self.W + self.b
        self.act_out = activate(self.lin_out, self.nonlin)
        return self.act_out

    def backward(self, grad_out):
        """
        Backpropagation through layer.

        Description
        -----------
        Computes gradients of weights, bias, and input.

        Parameters
        ----------
        grad_out : np.ndarray
            Gradient flowing from next layer.

        Returns
        -------
        tuple
            dx : np.ndarray
                Gradient with respect to input.
            dW : np.ndarray
                Gradient with respect to weights.
            db : np.ndarray
                Gradient with respect to bias.
        """
        dz = grad_out * activate_grad(self.lin_out, self.nonlin)

        dW = self.x_in.T @ dz / self.x_in.shape[0]
        db = np.sum(dz, axis=0, keepdims=True) / self.x_in.shape[0]
        dx = dz @ self.W.T

        return dx, dW, db

class MLP:
    """
    Fully connected multi-layer perceptron classifier.

    Description
    -----------
    Implements forward pass, loss computation, backpropagation,
    and gradient descent training.
    """

    def __init__(self, shape, activation="tanh", lr=0.01, reg=0.01, seed=0):
        """
        Initialize neural network architecture.

        Parameters
        ----------
        shape : list of int
            Layer sizes including input and output dimensions.
        activation : str
            Activation function for hidden layers.
        lr : float
            Learning rate.
        reg : float
            L2 regularization strength.
        seed : int
            Random seed.
        """
        self.blocks = []
        self.lr = lr
        self.reg = reg
        self.activation = activation

        for i in range(len(shape) - 2):
            self.blocks.append(
                Layer(shape[i], shape[i + 1], activation, seed + i)
            )

        rng = np.random.default_rng(seed)
        self.W_out = rng.standard_normal((shape[-2], shape[-1])) / np.sqrt(shape[-2])
        self.b_out = np.zeros((1, shape[-1]))

    def forward(self, X):
        """
        Forward pass through entire network.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Class probabilities from softmax.
        """
        h = X
        for block in self.blocks:
            h = block.forward(h)

        self.logits = h @ self.W_out + self.b_out
        self.probs = softmax(self.logits)
        return self.probs

    def loss(self, X, y):
        """
        Compute cross-entropy loss with L2 regularization.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Integer class labels.

        Returns
        -------
        float
            Scalar loss value.
        """
        n = X.shape[0]
        self.forward(X)

        data_loss = np.mean(-np.log(self.probs[np.arange(n), y]))

        reg_loss = 0.5 * self.reg * (
            np.sum(self.W_out ** 2)
            + sum(np.sum(b.W ** 2) for b in self.blocks)
        )

        return data_loss + reg_loss

    def backward(self, X, y):
        """
        Perform backpropagation through entire network.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Integer class labels.

        Returns
        -------
        tuple
            grads : list of tuples
                Gradients for hidden layers (dW, db).
            dW_out : np.ndarray
                Gradient of output weights.
            db_out : np.ndarray
                Gradient of output bias.
        """
        n = X.shape[0]
        self.forward(X)

        grad_logits = self.probs.copy()
        grad_logits[np.arange(n), y] -= 1
        grad_logits /= n

        dW_out = self.blocks[-1].act_out.T @ grad_logits
        db_out = np.sum(grad_logits, axis=0, keepdims=True)

        dW_out += self.reg * self.W_out

        grad_hidden = grad_logits @ self.W_out.T

        grads = []

        for i in reversed(range(len(self.blocks))):
            grad_hidden, dW, db = self.blocks[i].backward(grad_hidden)
            dW += self.reg * self.blocks[i].W
            grads.insert(0, (dW, db))

        return grads, dW_out, db_out

    def train(self, X, y, steps=10000, verbose=True, track_loss=False):
        loss_history = []

        for t in range(steps):
            grads, dW_out, db_out = self.backward(X, y)

            self.W_out -= self.lr * dW_out
            self.b_out -= self.lr * db_out

            for i, (dW, db) in enumerate(grads):
                self.blocks[i].W -= self.lr * dW
                self.blocks[i].b -= self.lr * db

            if track_loss and t % 50 == 0:
                loss_history.append(self.loss(X, y))

            if verbose and t % 1000 == 0:
                print(f"step {t}, loss = {self.loss(X, y):.4f}")

        return loss_history

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Predicted class indices.
        """
        return np.argmax(self.forward(X), axis=1)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Softmax probability matrix.
        """
        return self.forward(X)