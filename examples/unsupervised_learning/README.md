# Unsupervised Learning Notebook Demonstrations

## Overview

This directory contains a collection of **unsupervised learning Jupyter notebooks** built using the RiceML library.

The goal of these notebooks is to explore how machine learning models can uncover **hidden structure in unlabeled data**, without relying on target variables (y).

These examples emphasize:
- Intuition behind clustering and dimensionality reduction
- Sensitivity to hyperparameters
- Geometric interpretation of learned representations
- Visualization of latent structure in data

---

## Algorithms Covered

### Clustering Methods

- **K-Means Clustering**  
  Partitions data into K clusters by iteratively updating centroids.  
  Focus: centroid convergence, cluster assignment stability, and choice of K.

- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**  
  Groups points based on density rather than distance to centroids.  
  Focus: discovery of arbitrarily shaped clusters and identification of noise points.

---

### Dimensionality Reduction

- **Principal Component Analysis (PCA)**  
  Projects high-dimensional data into a lower-dimensional subspace that maximizes variance.  
  Focus: explained variance, eigen decomposition, and geometric interpretation of projections.

---

## Datasets

The notebooks use a combination of real-world and benchmark datasets:

- **Wine Dataset**  
  Chemical composition of wines used to explore clustering structure and separability.

- **Seeds Dataset**  
  Geometric measurements of wheat kernels from different varieties.

- **Wisconsin Breast Cancer Dataset**  
  Used in an unsupervised setting to explore latent structure in medical data.

- **Synthetic Datasets**  
  Generated datasets (e.g., blobs, moons, circular patterns) used to visualize clustering behavior and algorithm limitations.

Each notebook includes:
- Feature scaling (when needed)
- Visualization of raw data distributions
- Dimensionality considerations

---

## Key Learning Goals

Across all notebooks, the unsupervised learning examples emphasize:

- How structure can emerge without labels
- Differences between centroid-based and density-based clustering
- The importance of distance metrics and scaling
- How dimensionality reduction preserves (or loses) information
- Visual interpretation of learned representations