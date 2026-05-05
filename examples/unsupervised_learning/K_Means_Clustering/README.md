# K-Means Clustering Demo (Custom Implementation)

This notebook demonstrates a **from-scratch K-Means clustering workflow using a custom implementation** applied to a synthetic dataset.

---

## 📌 Overview

We use a synthetic dataset generated via `make_blobs` to visualize how K-Means clustering behaves on clearly separable data.

The workflow includes:
- Generating synthetic data
- Visualizing raw (unlabeled) data
- Fitting a custom K-Means model
- Visualizing learned clusters
- Comparing predictions with true labels (for evaluation only)

---

## 📊 Dataset: Synthetic Blobs

We use `sklearn.datasets.make_blobs` to generate a dataset with:
- 300 samples
- 4 clusters
- Controlled variance (`cluster_std=1.2`)
- Fixed randomness for reproducibility

### Why this dataset?
- Well-separated clusters
- Easy to visualize in 2D
- True labels available for comparison (not used in training)
