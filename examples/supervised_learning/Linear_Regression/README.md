# 📈 Linear Regression from Scratch (NumPy Implementation)

This repository contains a complete from-scratch implementation and demonstration of **Linear Regression**, supporting two optimization methods:

- Ordinary Least Squares (OLS) — closed-form solution
- Gradient Descent (GD) — iterative optimization

The model is implemented using only NumPy and is evaluated on a real-world regression dataset.

---

# 📌 Algorithm Overview

Linear Regression models the relationship between a continuous target variable \( y \) and input features \( X \) using a linear function:

\[
\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b
\]

Where:
- \( w \): learned weights (coefficients)
- \( b \): bias (intercept)
- \( X \): feature matrix
- \( \hat{y} \): predicted output

The objective is to minimize the Mean Squared Error (MSE):

\[
MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
\]

---

# ⚙️ Optimization Methods

## 1. Ordinary Least Squares (OLS)

- Closed-form solution using linear algebra (Normal Equation)
- Computes the exact global minimum of the loss function

**Best for:**
- Small to medium datasets
- Fast, deterministic solutions

---

## 2. Gradient Descent (GD)

- Iterative optimization method
- Updates weights using gradient of the loss function

**Update rule:**
\[
w := w - \eta \frac{\partial L}{\partial w}
\]

**Best for:**
- Large datasets
- Memory-efficient training
- Cases where matrix inversion is expensive

---

# 📊 Dataset

This project uses the:

### :contentReference[oaicite:0]{index=0}

### Dataset Description:
- 10 numerical input features
- Continuous target variable (disease progression score)
- Standard benchmark dataset for regression tasks

---

# ⚠️ Data Requirements

## Input Format

### Features (X)
- Must be a 2D NumPy array
- Shape: `(n_samples, n_features)`
- Must contain numeric values only

### Target (y)
- Must be a 1D NumPy array
- Must contain continuous numeric values

---

# ⚡ Preprocessing Requirements

## Feature Scaling (Required for GD)

Gradient Descent requires feature scaling for stable convergence.

Recommended method:
- Standardization (zero mean, unit variance)

Without scaling:
- Convergence may be slow or unstable
- Loss may oscillate

---

# 🧠 Project Workflow

The notebook demonstrates a full machine learning pipeline:

1. Load dataset
2. Standardize features
3. Train-test split
4. Train Linear Regression model:
   - OLS (closed-form solution)
   - Gradient Descent (iterative optimization)
5. Evaluate model performance
6. Visualize training behavior and predictions

---

# 📈 Outputs

The model produces:

- Learned weight vector \( w \)
- Bias term \( b \)
- Predicted values \( \hat{y} \)
- Training loss curve (Gradient Descent only)

---

# 📉 Evaluation Metric

## Mean Squared Error (MSE)

\[
MSE = \frac{1}{N} \sum (y - \hat{y})^2
\]

Lower values indicate better model performance.

---

# 📊 Key Visualizations

The notebook includes:

- 📉 Gradient Descent convergence curve
- 📊 Predicted vs Actual scatter plot
- 📌 Weight interpretation output

---

# 🧪 Key Insights

This project demonstrates:

- Difference between analytical vs iterative optimization
- Importance of feature scaling in gradient-based methods
- Interpretability of linear models
- Trade-offs between computational efficiency and flexibility
- Behavior of loss convergence over time

---

# 🎯 Learning Objectives

After reviewing this project, you will understand:

- How Linear Regression works mathematically
- How OLS computes exact solutions
- How Gradient Descent optimizes iteratively
- Why feature scaling is critical for optimization
- How to evaluate regression models properly

---

# 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/your-username/linear-regression-from-scratch.git

cd linear-regression-from-scratch

# Install dependencies
pip install numpy matplotlib seaborn scikit-learn

# Run notebook
jupyter notebook