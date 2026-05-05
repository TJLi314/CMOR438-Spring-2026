"""
Ensemble Methods
----------------
This module implements simple ensemble learning techniques:

1. StumpBaggingClassifier
   - Bagging using decision stumps (depth = 1 trees)
   - Uses bootstrap sampling
   - Aggregates via majority vote

2. StumpBaggingRegressor
   - Same idea for regression
   - Aggregates via averaging

3. MajorityVoteClassifier
   - Combines multiple models via hard voting
   - Works with any models that implement predict()
"""

import numpy as np
from collections import Counter

from rice_ml import DecisionTreeClassifier, DecisionTreeRegressor


# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def _bootstrap_sample(n, rng):
    """
    Generate bootstrap sample indices.

    Description
    -----------
    Randomly samples indices from 0 to n-1 with replacement.

    Arguments
    ----------
    n : int
        Number of training samples.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of sampled indices (with replacement).
    """
    return rng.integers(0, n, size=n)


def _majority_vote(pred_matrix):
    """
    Perform majority vote across model predictions.

    Description
    -----------
    Takes predictions from multiple models and returns
    the most frequent label for each sample.

    Arguments
    ----------
    pred_matrix : np.ndarray
        Shape (n_samples, n_models). Each row contains predictions
        from all models for a single sample.

    Returns
    -------
    np.ndarray
        Final predicted class labels after voting.
    """

    def vote(row):
        # Count occurrences of each label
        counts = Counter(row)

        # Find highest vote count
        max_count = max(counts.values())

        # Resolve ties by choosing smallest label
        return min([k for k, v in counts.items() if v == max_count])

    return np.array([vote(row) for row in pred_matrix])


def _average(pred_matrix):
    """
    Compute mean prediction across models.

    Description
    -----------
    Used for regression: averages predictions from all models.

    Arguments
    ----------
    pred_matrix : np.ndarray
        Shape (n_samples, n_models)

    Returns
    -------
    np.ndarray
        Averaged predictions for each sample.
    """
    return np.mean(pred_matrix, axis=1)


# -------------------------------------------------------------------
# Bagging Classifier (Decision Stumps)
# -------------------------------------------------------------------

class StumpBaggingClassifier:
    """
    Bagging classifier using decision stumps (depth=1 trees).
    """

    def __init__(self, n_models=10, seed=None):
        """
        Initialize ensemble.

        Arguments
        ----------
        n_models : int
            Number of decision stumps in the ensemble.
        seed : int or None
            Random seed for reproducibility.
        """
        self.n_models = n_models
        self.seed = seed
        self.models = []

    def fit(self, X, y):
        """
        Train ensemble using bootstrap sampling.

        Description
        -----------
        Each model is trained on a bootstrap sample of the data.
        Each base learner is a decision stump (max_depth=1).

        Arguments
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target labels (n_samples,)

        Returns
        -------
        self
            Fitted ensemble.
        """
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.seed)

        self.models = []  # reset model list

        for _ in range(self.n_models):
            # Step 1: create bootstrap sample
            idx = _bootstrap_sample(n_samples, rng)
            X_sample = X[idx]
            y_sample = y[idx]

            # Step 2: train a weak learner (decision stump)
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X_sample, y_sample)

            # Step 3: store trained model
            self.models.append(stump)

        return self

    def predict(self, X):
        """
        Predict class labels using majority vote.

        Arguments
        ----------
        X : np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted class labels.

        Raises
        ------
        RuntimeError
            If model has not been trained.
        """
        if not self.models:
            raise RuntimeError("Model not trained")

        # Collect predictions from all models
        preds = np.array([m.predict(X) for m in self.models]).T

        # Aggregate via majority vote
        return _majority_vote(preds)

    def score(self, X, y):
        """
        Compute classification accuracy.

        Arguments
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True labels.

        Returns
        -------
        float
            Accuracy score in [0, 1].
        """
        preds = self.predict(X)
        return np.mean(preds == y)


# -------------------------------------------------------------------
# Bagging Regressor (Decision Stumps)
# -------------------------------------------------------------------

class StumpBaggingRegressor:
    """
    Bagging regressor using decision stumps (depth=1 trees).
    """

    def __init__(self, n_models=10, seed=None):
        """
        Initialize ensemble.

        Arguments
        ----------
        n_models : int
            Number of base regressors.
        seed : int or None
            Random seed.
        """
        self.n_models = n_models
        self.seed = seed
        self.models = []
    
    def fit(self, X, y):
        """
        Train the bagging regressor using bootstrap aggregation.

        Description
        -----------
        Builds an ensemble of decision stumps (depth=1 regression trees),
        each trained on a bootstrap sample of the training data. The models
        are stored internally in a list for later prediction via averaging.

        Arguments
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        self
            Fitted StumpBaggingRegressor instance.
        """
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.seed)

        self.models = []

        for _ in range(self.n_models):
            idx = _bootstrap_sample(n_samples, rng)
            X_sample = X[idx]
            y_sample = y[idx]

            stump = DecisionTreeRegressor(max_depth=1)
            stump.fit(X_sample, y_sample)

            self.models.append(stump)

        return self

    def predict(self, X):
        """
        Predict continuous values using averaged ensemble output.

        Arguments
        ----------
        X : np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted values.

        Raises
        ------
        RuntimeError
            If model has not been trained.
        """
        if not self.models:
            raise RuntimeError("Model not trained")

        # Collect predictions from all regressors
        preds = np.array([m.predict(X) for m in self.models]).T

        # Average predictions
        return _average(preds)

    def score(self, X, y):
        """
        Compute R² score.

        Arguments
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True target values.

        Returns
        -------
        float
            R² score.
        """
        preds = self.predict(X)

        # residual sum of squares
        ss_res = np.sum((y - preds) ** 2)

        # total variance
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res / ss_tot


# -------------------------------------------------------------------
# Hard Voting Classifier (General Models)
# -------------------------------------------------------------------

class MajorityVoteClassifier:
    """
    Hard voting classifier.

    Description
    -----------
    Combines predictions from multiple models using majority vote.
    Models must already be trained externally.
    """

    def __init__(self, models):
        """
        Initialize voting ensemble.

        Arguments
        ----------
        models : list
            List of trained models with a .predict() method.
        """
        self.models = models

    def predict(self, X):
        """
        Predict class labels via majority vote.

        Arguments
        ----------
        X : np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted class labels.

        Raises
        ------
        RuntimeError
            If no models are provided.
        """
        if not self.models:
            raise RuntimeError("No models provided")

        # Collect predictions from each model
        preds = np.array([m.predict(X) for m in self.models]).T

        # Majority vote aggregation
        return _majority_vote(preds)

    def score(self, X, y):
        """
        Compute classification accuracy.

        Arguments
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True labels.

        Returns
        -------
        float
            Accuracy score.
        """
        preds = self.predict(X)
        return np.mean(preds == y)