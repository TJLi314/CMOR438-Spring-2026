# Supervised Learning Notebook Demonstrations

## Overview

This directory contains a collection of **supervised learning Jupyter notebooks** built using the RiceML library.

The goal of these notebooks is to provide an intuitive, hands-on understanding of how classical machine learning models learn from **labeled data (X, y)** and how different algorithmic assumptions affect performance, generalization, and decision boundaries.

Each notebook is self-contained and includes:
- Dataset loading and preprocessing
- Model training from scratch (RiceML implementations)
- Evaluation and performance analysis
- Visualizations of predictions, decision boundaries, or error trends

---

## Algorithms Covered

### Linear Models

- **Linear Regression**  
  Models continuous targets using least squares optimization.  
  Focus: regression fitting, residual behavior, and R² evaluation.

- **Logistic Regression**  
  Binary classification using the sigmoid function and gradient descent.  
  Focus: probability outputs, decision boundaries, and convergence behavior.

- **Perceptron**  
  A foundational linear classifier updated via misclassification-driven learning.  
  Focus: linear separability and convergence properties.

---

### Instance-Based Learning

- **K-Nearest Neighbors (KNN)**  
  Classifies samples based on distance to training points.  
  Focus: effect of `k`, distance metrics, and decision boundary smoothness.

---

### Tree-Based Models

- **Decision Trees**  
  Recursive partitioning of feature space using impurity reduction.  
  Focus: interpretability, overfitting, and tree depth effects.

- **Random Forest**  
  Ensemble of decision trees using bootstrap aggregation and feature subsampling.  
  Focus: variance reduction and robustness improvements.

---

### Ensemble Methods

- **Ensemble Methods (Bagging / Stump Bagging)**  
  Combines weak learners (e.g., decision stumps) to improve stability and accuracy.  
  Focus: bias-variance tradeoff and ensemble voting behavior.

---

### Neural Networks

- **Multi-Layer Perceptron (MLP)**  
  Feedforward neural network trained via backpropagation.  
  Focus: non-linear decision boundaries, loss curves, and learning dynamics.

---

## Datasets

These notebooks use a mix of real-world and benchmark datasets:

- **Wisconsin Breast Cancer Dataset**  
  Binary classification of tumor types using diagnostic features.

- **Wine Dataset**  
  Multi-class classification based on chemical composition of wines.

- **Seeds Dataset**  
  Classification of wheat varieties using geometric measurements.

- **Car Evaluation Dataset**  
  Categorical classification based on structured attributes (price, safety, etc.).

- **Banknote Authentication Dataset**  
  Binary classification using image-derived statistical features.

Each notebook includes dataset-specific preprocessing such as:
- Feature scaling
- Encoding categorical variables
- Train/test splitting