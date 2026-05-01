import numpy as np

class KNN:
    """
    K-Nearest Neighbors (KNN) algorithm for classification and regression.

    This implementation stores the training data and predicts outputs for new
    samples by finding the k closest training points using Euclidean distance.
    
    - For classification: predicts the majority label among neighbors.
    - For regression: predicts the average value of neighbors.
    """

    def __init__(self, k=3, regression=False, X_train=None, y_train=None):
        """
        Initialize the KNN model.

        Parameters
        ----------
        k : int
            Number of nearest neighbors to consider.
        regression : bool
            If True, performs regression; otherwise performs classification.
        X_train : np.ndarray or None
            Optional initial training feature matrix.
        y_train : np.ndarray or None
            Optional initial training labels/targets.
        """
        self.k = k
        self.regression = regression
        self.X_train = X_train
        self.y_train = y_train
    
    def distance(self, p, q):
        """
        Compute Euclidean distance between two points.

        Parameters
        ----------
        p : np.ndarray
            First point.
        q : np.ndarray
            Second point.

        Returns
        -------
        float
            Euclidean distance between p and q.
        """
        return np.sqrt((p - q) @ (p - q))
    
    def train(self, X, y):
        """
        Store the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target labels (classification) or values (regression).

        Returns
        -------
        self : KNN
            Fitted model instance.
        """
        # Convert input data to NumPy arrays for efficient computation
        self.X_train = np.array(X, dtype=float)

        # Store labels as integers for classification, floats for regression
        self.y_train = np.array(y, dtype=float) if self.regression else np.array(y, dtype=int)
        
        return self
    
    def predict(self, xi):
        """
        Predict the output for a single input sample.

        Parameters
        ----------
        xi : np.ndarray of shape (n_features,)
            Input feature vector.

        Returns
        -------
        int or float
            Predicted class label (classification) or value (regression).
        """
        # Ensure model has been trained
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been trained yet.")

        # Ensure k is valid
        if self.k > len(self.X_train):
            raise ValueError("k cannot be larger than number of training samples")
        
        neighbors = []
        
        # Compute distance from xi to every training point
        for p, label in zip(self.X_train, self.y_train):
            d = self.distance(xi, p)

            # Store point, label, and distance
            temp_data = [p, label, d]
            neighbors.append(temp_data)
        
        # Sort neighbors by distance (ascending)
        neighbors.sort(key=lambda x: x[-1])

        # Select k nearest neighbors
        neighbors = neighbors[:self.k]
        
        # Classification: majority vote
        if self.regression == False:
            labels = [x[1] for x in neighbors]
            return max(labels, key=labels.count)
        
        # Regression: average of neighbor values
        else:
            return sum(x[1] for x in neighbors) / self.k

    def predict_multiple(self, X):
        """
        Predict outputs for multiple input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels or values for each input sample.
        """
        # Apply predict() to each sample
        return np.array([self.predict(xi) for xi in X])

    def classification_error(self, X, y):
        """
        Compute classification error rate.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Proportion of incorrectly classified samples.
        """
        # Generate predictions for all samples
        preds = self.predict_multiple(X)

        # Compute fraction of incorrect predictions
        return np.mean(preds != y)