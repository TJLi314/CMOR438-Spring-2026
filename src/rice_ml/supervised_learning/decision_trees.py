"""
Overview
--------
This file implements two fundamental machine learning models from scratch
using only NumPy:

1. DecisionTreeClassifier
   - A binary recursive decision tree for classification tasks
   - Uses entropy and information gain to choose splits
   - Outputs discrete class labels

2. DecisionTreeRegressor
   - A regression tree for continuous-valued targets
   - Uses variance reduction to select optimal splits
   - Outputs real-valued predictions

Tree Representation
-------------------
Each internal node is stored as:

    (feature_index, threshold, left_subtree, right_subtree)

Leaf nodes store either:
    - class label (classifier)
    - numeric value (regressor)
"""

import numpy as np


class DecisionTreeClassifier:
    """
    Implements a binary decision tree classifier using:
    - Entropy as impurity measure
    - Information gain for split selection
    - Recursive depth-limited tree construction
    """

    def __init__(self, max_depth=None):
        """
        Initialize the classifier.

        Parameters
        ----------
        max_depth : int or None
            Maximum depth of the tree. If None, the tree grows until
            all leaves are pure or no valid splits remain.

        Attributes
        ----------
        depth_limit : int or None
            Stores maximum allowed tree depth.
        root_node : tuple or scalar
            Root of the constructed decision tree.
        """
        self.depth_limit = max_depth
        self.root_node = None

    def fit(self, features, labels):
        """
        Build decision tree from training data.

        Parameters
        ----------
        features : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.
        labels : np.ndarray, shape (n_samples,)
            Integer class labels.

        Returns
        -------
        self : DecisionTreeClassifier
            Trained model instance.

        Raises
        ------
        ValueError
            If dataset is empty.
        """
        if len(features) == 0:
            raise ValueError("Empty dataset provided")

        # Start recursive tree construction
        self.root_node = self._grow(features, labels, depth=0)
        return self

    def predict(self, features):
        """
        Predict class labels for input samples.

        Parameters
        ----------
        features : np.ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class labels.

        Raises
        ------
        RuntimeError
            If model has not been trained.
        """
        if self.root_node is None:
            raise RuntimeError("Model not trained")

        # Traverse tree for each sample
        return np.array([self._traverse(x, self.root_node) for x in features])

    def score(self, features, labels):
        """
        Compute classification accuracy.

        Parameters
        ----------
        features : np.ndarray
            Input data.
        labels : np.ndarray
            Ground truth labels.

        Returns
        -------
        float
            Accuracy in range [0, 1].
        """
        preds = self.predict(features)
        return np.mean(preds == labels)

    def _entropy(self, y):
        """
        Compute Shannon entropy of label distribution.

        Parameters
        ----------
        y : np.ndarray
            Class labels.

        Returns
        -------
        float
            Entropy value.
        """
        counts = np.bincount(y)
        probs = counts[counts > 0] / len(y)
        return -np.sum(probs * np.log2(probs))

    def _gain(self, parent, left, right):
        """
        Compute information gain from a split.

        Parameters
        ----------
        parent : np.ndarray
            Labels before split.
        left : np.ndarray
            Labels in left branch.
        right : np.ndarray
            Labels in right branch.

        Returns
        -------
        float
            Information gain value.
        """
        lp = len(left) / len(parent)
        rp = 1 - lp

        return self._entropy(parent) - (
            lp * self._entropy(left) + rp * self._entropy(right)
        )

    def _best_cut(self, X, y):
        """
        Find best feature and threshold to split on.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Labels.

        Returns
        -------
        (int or None, float or None)
            Best feature index and threshold.
        """
        best = (-1, None, None)

        n_features = X.shape[1]

        # Try every feature
        for f in range(n_features):

            # Try every unique threshold value
            for t in np.unique(X[:, f]):

                mask_left = X[:, f] <= t
                mask_right = ~mask_left

                # Skip invalid splits
                if mask_left.sum() == 0 or mask_right.sum() == 0:
                    continue

                gain = self._gain(y, y[mask_left], y[mask_right])

                if gain > best[0]:
                    best = (gain, f, t)

        return best[1], best[2]

    def _grow(self, X, y, depth):
        """
        Recursively build decision tree.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix at node.
        y : np.ndarray
            Labels at node.
        depth : int
            Current recursion depth.

        Returns
        -------
        tuple or int
            Tree node or leaf class label.
        """
        classes, counts = np.unique(y, return_counts=True)
        majority = classes[np.argmax(counts)]

        # Stop if pure node
        if len(classes) == 1:
            return majority

        # Stop if depth limit reached
        if self.depth_limit is not None and depth >= self.depth_limit:
            return majority

        feat, thresh = self._best_cut(X, y)

        # No valid split found
        if feat is None:
            return majority

        left_mask = X[:, feat] <= thresh

        # Recursive split
        return (
            feat,
            thresh,
            self._grow(X[left_mask], y[left_mask], depth + 1),
            self._grow(X[~left_mask], y[~left_mask], depth + 1),
        )

    def _traverse(self, x, node):
        """
        Traverse tree for a single sample.

        Parameters
        ----------
        x : np.ndarray
            Single input sample.
        node : tuple or int
            Current tree node.

        Returns
        -------
        int
            Predicted class label.
        """
        if not isinstance(node, tuple):
            return node

        feat, thresh, left, right = node

        if x[feat] <= thresh:
            return self._traverse(x, left)
        return self._traverse(x, right)


class DecisionTreeRegressor:
    """
    Decision Tree Regressor
    
    Implements regression trees using:
    - Variance reduction for split selection
    - Recursive binary partitioning
    - Mean prediction at leaf nodes
    """

    def __init__(self, max_depth=None):
        """
        Initialize regression tree.

        Parameters
        ----------
        max_depth : int or None
            Maximum tree depth.
        """
        self.depth_limit = max_depth
        self.root = None

    def fit(self, X, y):
        """
        Train regression tree.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        self
        """
        self.root = self._build(X, y, depth=0)
        return self

    def predict(self, X):
        """
        Predict continuous values.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray
        """
        if self.root is None:
            raise RuntimeError("Model not fitted yet")
        
        return np.array([self._walk(x, self.root) for x in X])

    def score(self, X, y):
        """
        Compute R² score.

        Returns
        -------
        float
        """
        preds = self.predict(X)

        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res / ss_tot

    def _var(self, y):
        """
        Compute variance.

        Returns
        -------
        float
        """
        return np.var(y) if len(y) else 0.0

    def _var_gain(self, parent, left, right):
        """
        Compute variance reduction.

        Returns
        -------
        float
        """
        lp = len(left) / len(parent)
        rp = 1 - lp

        return self._var(parent) - (
            lp * self._var(left) + rp * self._var(right)
        )

    def _best_split(self, X, y):
        """
        Find best split based on variance reduction.

        Returns
        -------
        (int or None, float or None)
        """
        best_score = -1
        best_feat, best_thr = None, None

        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):

                mask = X[:, f] <= t
                if mask.sum() == 0 or (~mask).sum() == 0:
                    continue

                score = self._var_gain(y, y[mask], y[~mask])

                if score > best_score:
                    best_score = score
                    best_feat = f
                    best_thr = t

        return best_feat, best_thr

    def _build(self, X, y, depth):
        """
        Recursively build regression tree.

        Returns
        -------
        tuple or float
        """
        if len(np.unique(y)) == 1:
            return np.mean(y)

        if self.depth_limit is not None and depth >= self.depth_limit:
            return np.mean(y)

        feat, thr = self._best_split(X, y)

        if feat is None:
            return np.mean(y)

        mask = X[:, feat] <= thr

        return (
            feat,
            thr,
            self._build(X[mask], y[mask], depth + 1),
            self._build(X[~mask], y[~mask], depth + 1),
        )

    def _walk(self, x, node):
        """
        Traverse regression tree.

        Returns
        -------
        float
        """
        if not isinstance(node, tuple):
            return node

        feat, thr, left, right = node

        if x[feat] <= thr:
            return self._walk(x, left)
        return self._walk(x, right)