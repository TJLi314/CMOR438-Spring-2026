# Notebook Demonstrations (RiceML Examples)

## Overview

This directory contains a curated collection of **machine learning Jupyter notebooks** demonstrating both supervised and unsupervised learning algorithms implemented in the RiceML package.

The notebooks are designed to emphasize:

- Conceptual understanding of machine learning algorithms
- Step-by-step visualization of model behavior
- Practical application on real and synthetic datasets
- Comparison of different model families under consistent evaluation settings

Each notebook is self-contained and can be run independently.

---

## Supervised Learning

The supervised learning notebooks focus on learning predictive models from **labeled data (X, y)**. These examples cover both classification and regression tasks and illustrate how different algorithms encode assumptions about data structure, complexity, and generalization.

### Topics include:

- Linear Regression
- Logistic Regression
- k-Nearest Neighbors (KNN)
- Perceptron
- Decision Trees
- Random Forests
- Ensemble Methods (e.g., bagging with weak learners)
- Multi-Layer Perceptron (MLP)

These notebooks highlight:
- Decision boundaries
- Loss behavior during training (when applicable)
- Bias-variance tradeoffs

---

## Unsupervised Learning

The unsupervised learning notebooks focus on discovering structure in **unlabeled data**. These methods are used for clustering, density estimation, and dimensionality reduction.

### Topics include:

- K-Means Clustering
- DBSCAN
- Principal Component Analysis (PCA)

These notebooks demonstrate:
- Cluster formation and visualization
- Sensitivity to hyperparameters (e.g., epsilon in DBSCAN, k in K-Means)
- Variance explained in PCA
- Projection of high-dimensional data into lower-dimensional spaces

---

## Datasets

The notebooks use a mixture of **real-world datasets and curated benchmark datasets**, including:

- **Wisconsin Breast Cancer Dataset**  
  Binary classification of malignant vs benign tumors using diagnostic features.

- **Wine Dataset**  
  Multi-class classification based on chemical composition of wines.

- **Seeds Dataset**  
  Classification of wheat varieties using geometric kernel measurements.

- **Car Evaluation Dataset**  
  Categorical classification problem based on car attributes such as price, safety, and capacity.

- **Banknote Authentication Dataset**  
  Binary classification using features extracted from wavelet-transformed images of banknotes.

Each notebook includes:
- Dataset description
- Preprocessing steps (if required)
- Train/test splitting strategy
- Feature normalization or encoding (when applicable)

---

## Requirements

Most notebooks rely on standard Python scientific computing libraries:

- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn (for dataset loading and comparison baselines)
- networkX
- mlxtend

All ML algorithms used in experiments are implemented in the **RiceML package**, not sklearn (unless explicitly noted for comparison or datasets).

---

## Usage

Each notebook is fully self-contained and should be executed top-to-bottom.

Recommended workflow:

1. Open a notebook (e.g., `k_means_clustering.ipynb`)
2. Run all cells sequentially
3. Observe visualizations and model outputs
4. Modify hyperparameters to explore behavior