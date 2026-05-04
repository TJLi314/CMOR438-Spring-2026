# K-Nearest Neighbors (From Scratch)

This notebook implements the K-Nearest Neighbors (KNN) algorithm from scratch using NumPy and evaluates it on a real-world classification dataset.

KNN is a non-parametric, instance-based learning method that makes predictions by finding the closest training samples to a given input and aggregating their labels.

---

## 🧠 Algorithm Overview

K-Nearest Neighbors is a distance-based learning algorithm. Instead of learning explicit model parameters during training, it stores the training dataset and performs computation at prediction time.

For a given input sample:
- Compute the distance to all training samples
- Select the **k closest neighbors**
- Predict based on:
  - **Classification:** majority vote of neighbors
  - **Regression:** average of neighbor values

---

## 📊 Dataset Used

This notebook uses the **Adult Income Dataset** from the UCI Machine Learning Repository.

It contains demographic and employment-related attributes such as:
- Age
- Education level
- Occupation
- Hours worked per week
- Capital gain/loss

The task is to predict whether an individual earns:
- `>50K` income (class 1)
- `<=50K` income (class 0)

This dataset is well-suited for KNN because:
- It contains both categorical and numerical structure
- The decision boundary is highly non-linear
- Feature similarity is meaningful for classification

---

## 📈 Experiments

The notebook includes multiple experiments:

### 1. Model Accuracy
Evaluates classification performance on the test set.

### 2. Effect of K
Analyzes how different values of `k` affect model performance and error rate.

### 3. Decision Boundary Visualization
Uses two features (e.g., age and hours-per-week) to visualize decision regions learned by the model.

---