# src (RiceML Core Package)

## Overview

This directory contains the core implementation of the **RiceML** library, a from-scratch machine learning package designed for **education, experimentation, and algorithmic understanding**.

The project implements classic machine learning algorithms using only **NumPy and core Python*.

---

## Installation & Setup

RiceML should be installed from the **project root directory (not inside `src/`)**.

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

#### Activate environment:

```bash
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

---

### 2. Install dependencies

If needed, install core dependencies manually:

```bash
pip install numpy pandas matplotlib pytest scikit-learn mlxtend
```

---

### 3. Install RiceML in editable mode

From the **project root (where `pyproject.toml` is located)**:

```bash
pip install -e .
```

This allows you to import the package directly and reflects code changes immediately.

---

### 4. Example Imports

After installation:

```python
import rice_ml

from rice_ml import LinearRegression
from rice_ml import RandomForest
from rice_ml import PCA
from rice_ml import *
```

---

## Design Goals

RiceML is built around the following principles:

- Mimic a simplified `scikit-learn`-style API
- Use only NumPy and standard Python (no heavy ML frameworks)
- Keep implementations readable and educational
- Ensure all models are unit-testable
- Separate supervised and unsupervised learning cleanly
- Maintain consistent method signatures across models

---

## Common API Pattern

Most models follow this interface:

### Supervised Learning
```python
fit(X, y)       # Train model
predict(X)      # Generate predictions
```

### Unsupervised Learning
```python
fit(X)              # Learn structure
predict(X)          # (if applicable)
transform(X)        # Dimensionality reduction / encoding
fit_transform(X)    # Combined operation
```

---

## Package Structure

```
rice_ml/
│
├── processing/
│   ├── preprocessing.py
│   └── postprocessing.py
│
├── supervised_learning/
│   ├── decision_trees.py
│   ├── ensemble_methods.py
│   ├── k_nearest_neightbors.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── multi_layer_perceptron.py
│   ├── perceptron.py
│   └── random_forest.py
│
├── unsupervised_learning/
│   ├── dbscan.py
│   ├── k_means_clustering.py
│   └── pca.py
```

Each module is self-contained, with minimal dependencies across files.

---

## Supervised Learning

These algorithms learn from labeled data `(X, y)`:

- Linear Regression
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees (Classifier & Regressor)
- Random Forest
- Ensemble Methods (Bagging / Stumps)
- Perceptron
- Multi-Layer Perceptron (MLP)

All models support prediction and evaluation workflows using standard ML metrics.

---

## Unsupervised Learning

These algorithms learn structure from unlabeled data:

- K-Means Clustering
- DBSCAN
- Principal Component Analysis (PCA)

They are used for:
- Clustering
- Dimensionality reduction
- Pattern discovery

---

## Preprocessing & Postprocessing

Located in `rice_ml/processing/`:

preprocessing.py
- train_test_split
- scaling utilities (Standardization, MinMax scaling)
- encoding utilities (categorical handling)

postprocessing.py
- prediction formatting
- output transformations
- evaluation helpers

These utilities are intentionally lightweight and NumPy-based.