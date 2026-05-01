import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch using stochastic gradient descent.

    This model supports binary classification using the sigmoid activation function and
    cross-entropy loss. Training is performed using online (per-sample) gradient updates.
    """

    def __init__(self, eta=0.01, epochs=1000):
        """
        Initialize the Logistic Regression model.

        Parameters
        ----------
        eta : float
            Learning rate for gradient descent updates.
        epochs : int
            Number of passes over the training dataset.
        """
        self.eta = eta
        self.epochs = epochs
        self.w_ = None
        self.errors_ = []

    def sigmoid(self, z):
        """
        Compute the sigmoid activation function.

        Parameters
        ----------
        z : float or np.ndarray
            Linear combination of inputs and weights.

        Returns
        -------
        float or np.ndarray
            Output in range (0, 1), interpreted as probability.
        """
        return 1.0 / (1.0 + np.exp(-z))

    def cross_entropy_loss(self, y_hat, y):
        """
        Compute binary cross-entropy loss for a single prediction.

        Parameters
        ----------
        y_hat : float
            Predicted probability (output of sigmoid).
        y : float
            True binary label (0 or 1).

        Returns
        -------
        float
            Cross-entropy loss value.
        """
        # Clip predictions to prevent log(0) numerical instability
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def train(self, X, y):
        """
        Train the logistic regression model using stochastic gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray, shape (n_samples,)
            Binary target labels.

        Returns
        -------
        self : LogisticRegression
            Fitted model instance.
        """
        # Initialize weights (including bias term as last weight)
        self.w_ = np.random.rand(X.shape[1] + 1)
        self.errors_ = []
        N = X.shape[0]

        # Loop over epochs
        for _ in range(self.epochs):
            errors = 0

            # Stochastic gradient descent (one sample at a time)
            for xi, target in zip(X, y):

                # Linear combination + bias
                z = np.dot(xi, self.w_[:-1]) + self.w_[-1]

                # Predicted probability
                y_hat = self.sigmoid(z)

                # Gradient signal
                error = y_hat - target

                # Update weights (feature weights + bias)
                self.w_[:-1] -= self.eta * error * xi
                self.w_[-1] -= self.eta * error

                # Accumulate loss
                errors += self.cross_entropy_loss(y_hat, target)

            # Store average epoch loss
            self.errors_.append(errors / N)

        return self

    def predict_proba(self, X):
        """
        Compute predicted probabilities for input samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted probabilities for class 1.
        """
        z = np.dot(X, self.w_[:-1]) + self.w_[-1]
        return self.sigmoid(z)

    def predict(self, X):
        """
        Predict binary class labels for input samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        return (self.predict_proba(X) >= 0.5).astype(int)

    def plot_cost_function(self):
        """
        Plot the training loss (cross-entropy) over epochs.

        Returns
        -------
        None
        """
        _, axs = plt.subplots(figsize=(10, 8))

        # Plot loss curve
        axs.plot(range(1, len(self.errors_) + 1),
                 self.errors_,
                 label="Cost function")

        axs.set_xlabel("epochs", fontsize=15)
        axs.set_ylabel("Cost", fontsize=15)
        axs.legend(fontsize=15)
        axs.set_title("Cost Calculated after Epoch During Training", fontsize=18)
        plt.show()

    def plot_decision_boundary(self, X, y, xstring="x", ystring="y"):
        """
        Visualize the decision boundary learned by the model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (must be 2D for visualization).
        y : np.ndarray
            Binary target labels.
        xstring : str
            Label for x-axis.
        ystring : str
            Label for y-axis.

        Returns
        -------
        None
        """
        plt.figure(figsize=(10, 8))

        # Uses mlxtend utility to plot classification regions
        plot_decision_regions(X, y, clf=self)

        plt.title("Neuron Decision Boundary", fontsize=18)
        plt.xlabel(xstring, fontsize=15)
        plt.ylabel(ystring, fontsize=15)
        plt.show()