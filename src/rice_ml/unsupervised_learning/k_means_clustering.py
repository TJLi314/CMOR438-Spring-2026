import numpy as np

class KMeans:
    """
    K-Means clustering algorithm.

    This class implements the K-Means clustering algorithm, which partitions
    data into K clusters by minimizing the variance within each cluster.

    Attributes:
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations for convergence.
        tol (float): Tolerance for convergence (based on centroid movement).
        centroids (np.ndarray): Coordinates of cluster centroids.
        labels_ (np.ndarray): Cluster assignments for training data after fitting.
    """

    def __init__(self, K=3, max_iters=100, tol=1e-4, random_state=42):
        """
        Initialize the KMeans model.

        Args:
            K (int): Number of clusters.
            max_iters (int): Maximum number of iterations to run the algorithm.
            tol (float): Convergence threshold. If centroid movement is below
                         this value, the algorithm stops early.
            random_state (int): Seed for random centroid initialization.
        """
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels_ = None

        # Set random seed for reproducibility
        np.random.seed(random_state)
    
    def fit(self, X):
        """
        Fit the K-Means model to the data.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Input dataset.

        Returns:
            self: Fitted KMeans instance.
        
        Raises:
            ValueError: If dataset is empty or K > number of samples.
        """
        # Convert input to NumPy array for consistency
        X = np.array(X, dtype=float)

        # Validate input
        if X.size == 0:
            raise ValueError("Need at least one point in the dataset")

        if self.K > X.shape[0]:
            raise ValueError("K cannot be larger than number of samples")
        
        # Randomly initialize centroids from data points
        random_indices = np.random.choice(X.shape[0], self.K, replace=False)
        self.centroids = X[random_indices]
        
        # Main K-Means iteration loop
        for _ in range(self.max_iters):
            # Assign each point to the nearest centroid
            labels = self.cluster_assign(X)

            centroids_new = []

            for k in range(self.K):
                points = X[labels == k]

                if len(points) == 0:
                    # reinitialize to a random data point
                    centroids_new.append(X[np.random.randint(0, X.shape[0])])
                else:
                    centroids_new.append(points.mean(axis=0))

            centroids_new = np.array(centroids_new)
            
            # Check for convergence (centroid movement is small)
            if np.linalg.norm(centroids_new - self.centroids) < self.tol:
                self.centroids = centroids_new
                break
                
            # Update centroids
            self.centroids = centroids_new
            
        # Store final cluster assignments
        self.labels_ = self.cluster_assign(X)
        return self
            
    def cluster_assign(self, X):
        """
        Assign each data point to the nearest centroid.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Input dataset.

        Returns:
            np.ndarray: Array of cluster indices for each data point.
        """
        X = np.array(X, dtype=float)

        # Compute distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

        # Assign each point to the closest centroid
        return np.argmin(distances, axis=1)

    def predict_cluster(self, X):
        """
        Predict cluster assignments for new data.

        Args:
            X (array-like of shape (n_samples, n_features)):
                New data points.

        Returns:
            np.ndarray: Predicted cluster indices.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before prediction.")

        return self.cluster_assign(X)