import numpy as np
from typing import Optional


class PCA:
    """
    Principal Component Analysis (PCA) implementation from scratch.

    This class performs dimensionality reduction by projecting data onto
    orthogonal directions that maximize variance.

    It uses eigen-decomposition of the covariance matrix to compute
    principal directions (components).
    """

    def __init__(self, num_components: int):
        """
        Initialize PCA model.

        Parameters
        ----------
        num_components : int
            Number of principal components to retain.
        """
        if num_components < 1:
            raise ValueError("num_components must be >= 1")

        self.num_components = num_components

        # Learned attributes
        self.pc_axes_: Optional[np.ndarray] = None
        self.data_mean_: Optional[np.ndarray] = None
        self.eigen_values_: Optional[np.ndarray] = None
        self.var_ratio_: Optional[np.ndarray] = None

    def fit(self, data_matrix: np.ndarray):
        """
        Learn principal components from data.

        Parameters
        ----------
        data_matrix : np.ndarray of shape (n_samples, n_features)
            Input dataset.

        Returns
        -------
        self : PrincipalComponentAnalyzer
            Fitted PCA model.
        """
        data_matrix = np.asarray(data_matrix, dtype=float)

        if data_matrix.ndim != 2:
            raise ValueError("Input must be a 2D array")

        n_samples, n_features = data_matrix.shape

        if self.num_components > n_features:
            raise ValueError("num_components cannot exceed feature dimension")

        # Step 1: center data
        self.data_mean_ = np.mean(data_matrix, axis=0)
        centered = data_matrix - self.data_mean_

        # Step 2: covariance matrix
        cov_matrix = (centered.T @ centered) / (n_samples - 1)

        # Step 3: eigen decomposition
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

        # Sort by descending variance
        order = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[order]
        eig_vecs = eig_vecs[:, order]

        # Step 4: store principal axes
        self.pc_axes_ = eig_vecs[:, : self.num_components].T
        self.eigen_values_ = eig_vals[: self.num_components]

        # Explained variance ratio
        total_var = np.sum(eig_vals)
        self.var_ratio_ = self.eigen_values_ / total_var

        return self

    def project(self, data_matrix: np.ndarray):
        """
        Project data into lower-dimensional PCA space.

        Parameters
        ----------
        data_matrix : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        projected : np.ndarray of shape (n_samples, num_components)
        """
        if self.pc_axes_ is None or self.data_mean_ is None:
            raise RuntimeError("Model must be fitted before calling project")

        data_matrix = np.asarray(data_matrix, dtype=float)

        centered = data_matrix - self.data_mean_
        return centered @ self.pc_axes_.T

    def reconstruct(self, reduced_data: np.ndarray):
        """
        Reconstruct original data from reduced PCA space.

        Parameters
        ----------
        reduced_data : np.ndarray of shape (n_samples, num_components)

        Returns
        -------
        reconstructed : np.ndarray of shape (n_samples, n_features)
        """
        if self.pc_axes_ is None or self.data_mean_ is None:
            raise RuntimeError("Model must be fitted before calling reconstruct")

        reduced_data = np.asarray(reduced_data, dtype=float)

        return reduced_data @ self.pc_axes_ + self.data_mean_

    def fit_project(self, data_matrix: np.ndarray):
        """
        Fit PCA and return projected data in one step.

        Parameters
        ----------
        data_matrix : np.ndarray

        Returns
        -------
        np.ndarray
            Reduced representation of input data.
        """
        self.fit(data_matrix)
        return self.project(data_matrix)