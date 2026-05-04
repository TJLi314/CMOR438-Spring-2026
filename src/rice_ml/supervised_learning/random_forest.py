"""
Random Forest

------------------------------------------------------
This module implements two ensemble learning models:

1. RandomForestClassifier
   - Uses bootstrap aggregation (bagging)
   - Builds multiple DecisionTreeClassifier models
   - Uses feature subsampling at each tree
   - Predicts via majority voting

2. RandomForestRegressor
   - Uses bootstrap aggregation (bagging)
   - Builds multiple DecisionTreeRegressor models
   - Uses feature subsampling at each tree
   - Predicts via averaging across trees

Both models are built on top of custom decision tree implementations
from the `rice_ml` package and use NumPy for all computations.
"""

import numpy as np
from rice_ml import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier:
    """
    Random Forest classifier using an ensemble of decision trees.
    """

    def __init__(self, n_trees=100, max_depth=None, feature_mode="sqrt", seed=None):
        """
        Initialize the random forest classifier.

        Parameters
        ----------
        n_trees : int
            Number of decision trees in the ensemble.
        max_depth : int or None
            Maximum depth of each decision tree.
        feature_mode : str
            Feature sampling strategy ("sqrt" or "all").
        seed : int or None
            Random seed for reproducibility.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.feature_mode = feature_mode
        self.seed = seed

        # Stores (trained_tree, feature_indices)
        self._forest = []

    def _sample_features(self, n_features):
        """
        Select a subset of features for a single tree.

        Parameters
        ----------
        n_features : int
            Total number of available features.

        Returns
        -------
        np.ndarray
            Indices of selected features.
        """
        # Use sqrt(n_features) features by default (classic RF behavior)
        if self.feature_mode == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
            return self._rng.choice(n_features, k, replace=False)

        # Otherwise use all features
        return np.arange(n_features)

    def fit(self, X, y):
        """
        Train the random forest classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : RandomForestClassifier
            Fitted model.
        """
        self._rng = np.random.default_rng(self.seed)
        self._forest = []

        n_samples, n_features = X.shape

        # Build each tree independently
        for _ in range(self.n_trees):

            # Bootstrap sampling (sample with replacement)
            boot_idx = self._rng.integers(0, n_samples, size=n_samples)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]

            # Random feature subset
            feat_idx = self._sample_features(n_features)

            # Train decision tree on subset of data/features
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_boot[:, feat_idx], y_boot)

            # Store tree + feature mapping
            self._forest.append((tree, feat_idx))

        return self

    def predict(self, X):
        """
        Predict class labels using majority vote across all trees.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if not self._forest:
            raise RuntimeError("Forest is empty. Call fit first.")

        # Collect predictions from each tree
        all_preds = []

        for tree, feats in self._forest:
            preds = tree.predict(X[:, feats])
            all_preds.append(preds)

        # Shape: (n_samples, n_trees)
        all_preds = np.vstack(all_preds).T

        # Majority vote per sample
        return np.array([
            np.bincount(row.astype(int)).argmax()
            for row in all_preds
        ])

    def score(self, X, y):
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            True labels.

        Returns
        -------
        float
            Accuracy score in [0, 1].
        """
        preds = self.predict(X)
        return np.mean(preds == y)


class RandomForestRegressor:
    """
    Random Forest regressor using an ensemble of decision trees.
    """

    def __init__(self, n_trees=100, max_depth=None, feature_mode="sqrt", seed=None):
        """
        Initialize the random forest regressor.

        Parameters
        ----------
        n_trees : int
            Number of decision trees in the ensemble.
        max_depth : int or None
            Maximum depth of each decision tree.
        feature_mode : str
            Feature sampling strategy ("sqrt" or "all").
        seed : int or None
            Random seed for reproducibility.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.feature_mode = feature_mode
        self.seed = seed

        # Stores (trained_tree, feature_indices)
        self._forest = []

    def _choose_features(self, n_features):
        """
        Select subset of features for a regression tree.

        Parameters
        ----------
        n_features : int
            Total number of features.

        Returns
        -------
        np.ndarray
            Selected feature indices.
        """
        if self.feature_mode == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
            return np.random.choice(n_features, k, replace=False)

        return np.arange(n_features)

    def fit(self, X, y):
        """
        Train the random forest regressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RandomForestRegressor
            Fitted model.
        """
        rng = np.random.default_rng(self.seed)
        self._forest = []

        n_samples, n_features = X.shape

        # Train each tree independently
        for _ in range(self.n_trees):

            # Bootstrap sample
            sample_ids = rng.integers(0, n_samples, size=n_samples)
            X_s = X[sample_ids]
            y_s = y[sample_ids]

            # Random feature subset
            feat_ids = self._choose_features(n_features)

            # Train decision tree regressor
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_s[:, feat_ids], y_s)

            self._forest.append((tree, feat_ids))

        return self

    def predict(self, X):
        """
        Predict continuous values by averaging tree outputs.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted values.
        """
        if len(self._forest) == 0:
            raise RuntimeError("Model not trained yet.")

        # Collect predictions from each tree
        preds = np.zeros((X.shape[0], len(self._forest)))

        for i, (tree, feats) in enumerate(self._forest):
            preds[:, i] = tree.predict(X[:, feats])

        # Average across all trees
        return preds.mean(axis=1)

    def score(self, X, y):
        """
        Compute R² score for regression performance.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            True target values.

        Returns
        -------
        float
            R² score.
        """
        y_hat = self.predict(X)

        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)