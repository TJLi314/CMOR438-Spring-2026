import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

        DBSCAN is a clustering algorithm that groups together points that are
        closely packed (i.e., have many nearby neighbors) and marks points that
        lie alone in low-density regions as noise (outliers).

        Parameters
        ----------
        eps : float, default=0.5
            The maximum distance between two samples for them to be considered
            as neighbors. This defines the radius of the neighborhood around a point.

        min_samples : int, default=5
            The minimum number of samples required in a neighborhood for a point
            to be considered a core point. This includes the point itself.

        Attributes
        ----------
        labels_ : np.ndarray of shape (n_samples,)
            Cluster labels assigned to each point in the dataset.
            -1 indicates noise (outlier points).

        _X_train : np.ndarray of shape (n_samples, n_features)
            The training data used during fitting. Stored for use in prediction.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self._X_train = None

    def fit(self, X):
        """
        Fit the DBSCAN model to the dataset.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input dataset to cluster.

        Returns
        -------
        self : DBSCAN
            The fitted DBSCAN instance with computed cluster labels.
        """
        if X.size == 0:
            raise ValueError("Input dataset must not be empty.")

        n_samples = X.shape[0]
        self._X_train = X

        # Initialize all points as noise (-1)
        self.labels_ = -1 * np.ones(n_samples, dtype=int)

        # Track whether a point has already been processed
        visited = np.zeros(n_samples, dtype=bool)

        cluster_id = 0

        for i in range(n_samples):
            # Skip already processed points
            if visited[i]:
                continue

            visited[i] = True

            # Find neighbors within eps radius
            neighbors = self._region_query(X, i)

            # If not enough neighbors → not a core point → remains noise
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                # Start a new cluster and expand it
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1

        return self

    def _region_query(self, X, idx):
        """
        Find all points within eps distance of a given point.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The dataset.

        idx : int
            Index of the query point.

        Returns
        -------
        neighbors : np.ndarray
            Indices of all points within `eps` distance of the query point,
            including the point itself.
        """
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, idx, neighbors, cluster_id, visited):
        """
        Expand a cluster starting from a core point.

        This method performs a breadth-first expansion of a cluster by iteratively
        adding all density-reachable points.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The dataset.

        idx : int
            Index of the initial core point.

        neighbors : np.ndarray
            Indices of neighboring points within eps of the core point.

        cluster_id : int
            The ID of the current cluster being formed.

        visited : np.ndarray of shape (n_samples,)
            Boolean array indicating whether each point has been visited.

        Returns
        -------
        None
            This method updates `self.labels_` in-place.
        """
        # Assign the starting core point to the cluster
        self.labels_[idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            # If neighbor hasn't been processed yet
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = self._region_query(X, neighbor_idx)

                # If neighbor is also a core point, merge neighborhoods
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, neighbor_neighbors))

            # Assign cluster label if it's currently noise
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id

            i += 1

    def predict(self, X):
        """
        Assign cluster labels to new data points.

        This method assigns each new point to the cluster of its nearest neighbor
        within `eps`, if such a neighbor exists. Otherwise, the point is labeled
        as noise (-1).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data points to assign cluster labels.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Predicted cluster labels for each point.
            -1 indicates noise.
        """
        if self.labels_ is None or self._X_train is None:
            raise ValueError("Model has not been fitted yet.")

        labels = []

        for point in X:
            # Compute distances to all training points
            distances = np.linalg.norm(self._X_train - point, axis=1)

            # Find neighbors within eps
            neighbors = np.where(distances <= self.eps)[0]

            if len(neighbors) == 0:
                labels.append(-1)  # No nearby cluster → noise
            else:
                labels.append(self.labels_[neighbors[0]])

        return np.array(labels)