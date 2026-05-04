import numpy as np


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Works for any numeric dataset.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Compute mean and std for each feature.

        Args:
            X (np.ndarray): shape (n_samples, n_features)
        """
        X = np.asarray(X)

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Avoid division by zero
        self.std_[self.std_ == 0] = 1

        return self

    def transform(self, X):
        """
        Standardize using previously computed mean and std.
        """
        X = np.asarray(X)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        Fit and transform in one step.
        """
        return self.fit(X).transform(X)


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None, stratify=None):
    """
    Split dataset into train and test sets.

    Args:
        X (np.ndarray)
        y (np.ndarray)
        test_size (float): proportion of test data
        shuffle (bool)
        random_state (int)
        stratify (np.ndarray or None): maintain class distribution

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    test_count = int(n_samples * test_size)

    if stratify is not None:
        # Stratified split
        unique_classes = np.unique(y)
        train_indices = []
        test_indices = []

        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]

            if shuffle:
                np.random.shuffle(cls_indices)

            split = int(len(cls_indices) * (1 - test_size))
            train_indices.extend(cls_indices[:split])
            test_indices.extend(cls_indices[split:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

    else:
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        test_indices = indices[:test_count]
        train_indices = indices[test_count:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


class LabelEncoder:
    """
    Encode categorical labels as integers.

    Example:
        ["cat", "dog", "cat"] -> [0, 1, 0]
    """

    def __init__(self):
        self.classes_ = None
        self.class_to_index_ = None

    def fit(self, y):
        """
        Learn mapping from classes to integers.

        Parameters
        ----------
        y : array-like
            Input labels
        """
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.class_to_index_ = {
            cls: idx for idx, cls in enumerate(self.classes_)
        }

        return self

    def transform(self, y):
        """
        Convert labels to integers.

        Parameters
        ----------
        y : array-like

        Returns
        -------
        np.ndarray
        """
        y = np.asarray(y)

        if self.class_to_index_ is None:
            raise ValueError("LabelEncoder has not been fitted yet.")

        return np.array([self.class_to_index_[label] for label in y])

    def fit_transform(self, y):
        """
        Fit and transform in one step.
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        """
        Convert integers back to original labels.

        Parameters
        ----------
        y : array-like of int

        Returns
        -------
        np.ndarray
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder has not been fitted yet.")

        y = np.asarray(y)
        return np.array([self.classes_[i] for i in y])