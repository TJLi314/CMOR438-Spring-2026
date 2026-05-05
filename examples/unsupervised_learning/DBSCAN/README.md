# DBSCAN Clustering Demo (Custom Implementation)

## Overview

This project demonstrates a **custom implementation of DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** applied to a synthetic dataset. The goal is to show how DBSCAN can identify **non-linearly separable clusters** and detect **noise points (outliers)** without requiring the number of clusters in advance.

We use the `make_moons` dataset from scikit-learn because it produces two interleaving crescent-shaped clusters that are not separable using linear boundaries or well-suited for K-Means.

---

## What is DBSCAN?

DBSCAN is a density-based clustering algorithm that groups points based on neighborhood density.

Key ideas:
- Points within a distance `eps` are considered neighbors
- A point is a **core point** if it has at least `min_samples` neighbors
- Clusters are formed by expanding from core points
- Points that do not belong to any cluster are labeled as **noise (-1)**

Advantages:
- No need to specify number of clusters
- Can find arbitrarily shaped clusters
- Robust to outliers

---

## Dataset: make_moons

We use:

- `n_samples = 300`
- `noise = 0.05`
- `random_state = 42`

This dataset is ideal because:
- It contains two curved, interleaving clusters
- It is not linearly separable
- It highlights DBSCAN’s strengths over K-Means

---

## Visualizations

We plot the resulting clusters:
- Each color represents a cluster
- Noise points are labeled as `-1`
- We compare DBSCAN predictions with true labels from `make_moons`.

Note:
- Cluster IDs are arbitrary and may not match true labels
- Evaluation is purely visual

---

## Results

The model successfully:
- Identifies both crescent-shaped clusters
- Separates dense regions correctly
- Marks ambiguous boundary points as noise

This demonstrates DBSCAN’s strength in handling **non-linear cluster structures**.