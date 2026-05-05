# rice_ml: A From-Scratch Machine Learning Library
### CMOR 438/INDE 577: Data Science & Machine Learning
### Tianjian Li, tl107

---

## Overview

**rice_ml** is a lightweight, from-scratch machine learning library built for **education, experimentation, and deep understanding of core ML algorithms**.

It includes:

- Supervised and unsupervised learning algorithms
- Preprocessing and postprocessing utilities
- Fully reproducible examples via Jupyter notebooks
- Comprehensive unit tests for correctness and robustness

---

## Project Structure

```
.
├── LICENSE
├── README.md
├── examples
│   ├── README.md
│   ├── supervised_learning
│   │   ├── Decision_Trees
│   │   ├── Ensemble_Methods
│   │   ├── K_Nearest_Neighbors
│   │   ├── Linear_Regression
│   │   ├── Logisitic_Regression
│   │   ├── Multi-layer_Perceptron
│   │   ├── Perceptron
│   │   └── Random_Forest
│   └── unsupervised_learning
│       ├── DBSCAN
│       ├── K_Means_Clustering
│       └── PCA
├── pyproject.toml
├── requirements.txt
├── src
│   └── rice_ml
│       ├── processing
│       ├── supervised_learning
│       └── unsupervised_learning
└── tests
    └── unit
```

---

## Algorithms Included

### Supervised Learning

- **Linear Regression**  
  Ordinary Least Squares (OLS) regression with analytical solution.

- **Logistic Regression**  
  Binary classification using sigmoid activation and gradient descent.

- **K-Nearest Neighbors (KNN)**  
  Distance-based classification algorithm with efficient vectorized prediction.

- **Decision Trees**  
  Tree-based model using recursive splitting and impurity minimization.

- **Random Forest**  
  Ensemble of decision trees using bootstrap aggregation and feature subsampling.

- **Ensemble Methods (Stump Bagging)**  
  Bagging with weak learners (decision stumps) to reduce variance.

- **Perceptron**  
  Single-layer linear classifier for binary classification.

- **Multi-Layer Perceptron (MLP)**  
  Feedforward neural network with backpropagation.

---

### Unsupervised Learning

- **K-Means Clustering**  
  Centroid-based clustering with iterative refinement.

- **DBSCAN**  
  Density-based clustering capable of detecting arbitrary-shaped clusters and noise.

- **PCA (Principal Component Analysis)**  
  Dimensionality reduction via variance maximization and orthogonal projections.

---

## Utilities

### Preprocessing

Located in `rice_ml/processing/preprocessing.py`:

- Feature scaling
- Data normalization
- Input preparation for models

### Postprocessing

Located in `rice_ml/processing/postprocessing.py`:

- Output formatting
- Prediction transformations
- Evaluation helpers

---

## Example Notebooks

Each algorithm includes a **fully worked Jupyter notebook** in the `examples/` directory with:

- Step-by-step explanations
- Visualizations
- Training and evaluation workflows

Example topics include:

- Decision boundaries for classifiers
- Clustering visualizations
- Dimensionality reduction plots
- Model performance comparisons

---

## Example Datasets

The example notebooks use a mix of real-world and synthetic datasets to demonstrate different machine learning tasks:

- **Banknote Authentication Dataset**  
  Contains features extracted from images of banknotes using wavelet transforms.  
  Task: Binary classification (authentic vs forged).

- **Wisconsin Breast Cancer Dataset**  
  Medical dataset with features computed from digitized images of breast masses.  
  Task: Binary classification (malignant vs benign tumors).

- **Wine Dataset**  
  Chemical analysis of wines derived from three different cultivars.  
  Task: Multi-class classification.

- **Seeds Dataset**  
  Measurements of geometrical properties of wheat kernels across different varieties.  
  Task: Multi-class classification.

- **Car Evaluation Dataset**  
  Contains categorical features describing car attributes (price, safety, capacity, etc.).  
  Task: Multi-class classification with structured/categorical inputs.

---

## Testing

RiceML includes a comprehensive unit test suite covering:

- Algorithm correctness
- Edge cases and numerical stability
- Preprocessing and postprocessing utilities

Note that users will have to do the development installation to use testing tools.

Run tests with:

```bash
pytest
```

---

## Installation

Clone the repository and install locally:

```bash
git clone <repo_url>
cd CMOR438-Spring-2026
pip install -e .
```

---

### Development installation (includes testing tools)

```bash
git clone <repo_url>
cd CMOR438-Spring-2026
pip install -e .[dev]
```

This installs:
- `pytest` for running unit tests