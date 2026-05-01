import numpy as np

class LinearRegression:
    """
    Linear Regression supporting:
    - Ordinary Least Squares (closed-form)
    - Gradient Descent optimization
    """

    def __init__(self, method="gd", eta=0.01, epochs=1000):
        """
        Parameters
        ----------
        method : str
            Optimization method: "gd" (gradient descent) or "ols" (closed form)
        eta : float
            Learning rate for gradient descent updates
        epochs : int
            Number of passes over the dataset (used only in GD)
        """
        if method not in ["gd", "ols"]:
            raise ValueError("method must be 'gd' or 'ols'")

        self.method = method
        self.eta = eta
        self.epochs = epochs

        self.w_ = None
        self.bias = None
        self.errors_ = []

        np.random.seed(42)

    def _fit_closed_form(self, X, y):
        """
        Fit linear regression using the closed-form solution.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix
        y : np.ndarray of shape (n_samples,)
            Target values
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Solve normal equation using pseudoinverse for stability
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.bias = theta[0]
        self.w_ = theta[1:]

    def _fit_gradient_descent(self, X, y):
        """
        Fit linear regression using stochastic gradient descent.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix
        y : np.ndarray of shape (n_samples,)
            Target values
        """
        self.w_ = np.random.rand(X.shape[1])
        self.bias = 0.0
        self.errors_ = []

        N = X.shape[0]

        for _ in range(self.epochs):
            epoch_error = 0.0

            for xi, target in zip(X, y):
                error = self.predict(xi) - target

                self.w_ -= self.eta * error * xi
                self.bias -= self.eta * error

                epoch_error += 0.5 * (error ** 2)

            self.errors_.append(epoch_error / N)

    def fit(self, X, y):
        """
        Train the linear regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : LinearRegression
            Fitted model
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if self.method == "ols":
            self._fit_closed_form(X, y)
        else:
            self._fit_gradient_descent(X, y)

        return self

    def predict(self, X):
        """
        Predict target values using learned model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        np.ndarray
            Predicted values
        """
        return np.dot(X, self.w_) + self.bias