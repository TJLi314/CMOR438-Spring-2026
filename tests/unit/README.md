# Unit Testing (RiceML)

## Overview

This project includes a comprehensive suite of **unit tests** to ensure correctness, numerical stability, and robustness of all machine learning implementations in the `rice_ml` package.

All tests are written using **pytest** and are organized by algorithm and module.

The goal of the test suite is to verify:

- Correct implementation of ML algorithms from scratch
- Proper handling of edge cases (empty inputs, invalid shapes, degenerate data)
- Stability of numerical computations
- Consistency of preprocessing and postprocessing utilities
- Correct integration across the full ML pipeline

---

## Test Structure

All unit tests are located in:

```
tests/unit/
```

Each file corresponds to a specific module or algorithm:

### Supervised Learning Tests
- `test_linear_regression.py`
- `test_logisitic_regression.py`
- `test_knn.py`
- `test_decision_tree.py`
- `test_random_forest.py`
- `test_ensemble_methods.py`
- `test_perceptron.py`
- `test_multi_layer_perceptron.py`

### Unsupervised Learning Tests
- `test_k_means_clustering.py`
- `test_dbscan.py`
- `test_pca.py`

### Preprocessing & Utilities
- `test_preprocessing.py`
- `test_postprocessing.py`

---

## Running Tests

From the **project root directory**, run:

```bash
pytest
```

Pytest will automatically discover all tests in the `tests/` directory using the configuration defined in `pyproject.toml`.