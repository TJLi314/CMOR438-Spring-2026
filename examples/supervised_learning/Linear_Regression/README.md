# 📈 Linear Regression from Scratch (NumPy Implementation)

This repository contains a complete from-scratch implementation and demonstration of **Linear Regression**, supporting two optimization methods:

- Ordinary Least Squares (OLS) — closed-form solution
- Gradient Descent (GD) — iterative optimization

The model is implemented using only NumPy and is evaluated on a real-world regression dataset.

---

# 📌 Algorithm Overview

Linear Regression models the relationship between a continuous target variable (y) and input features (X) using a linear function:

y_hat = w1*x1 + w2*x2 + ... + wn*xn + b

Where:
- w: learned weights (coefficients)
- b: bias (intercept)
- X: feature matrix
- y_hat: predicted output

The goal is to learn weights and bias that minimize prediction error using Mean Squared Error (MSE).

---

# ⚙️ Optimization Methods

## 1. Ordinary Least Squares (OLS)

- Closed-form analytical solution
- Computes the exact best-fit parameters using linear algebra

**Best for:**
- Small to medium datasets
- Fast and exact solutions

---

## 2. Gradient Descent (GD)

- Iterative optimization method
- Repeatedly updates weights using gradient-based updates

**Update idea:**
- Adjust weights in the direction that reduces prediction error

**Best for:**
- Large datasets
- Cases where matrix inversion is too expensive

---

# 📊 Dataset

This project uses the **Diabetes Dataset** from scikit-learn, a standard regression benchmark dataset.

### Dataset Description:
- 10 numerical features
- Continuous target variable (disease progression score)
- Standard benchmark dataset for regression

---


# 🧠 Project Workflow

The notebook demonstrates a full machine learning pipeline:

1. Load dataset
2. Standardize features
3. Split into training and test sets
4. Train Linear Regression model using:
   - OLS (closed-form solution)
   - Gradient Descent (iterative optimization)
5. Evaluate performance
6. Visualize results

---

# 📈 Outputs

The model produces:

- Learned weight vector (w)
- Bias term (b)
- Predictions (y_hat)
- Training loss curve (for GD)

---

# 📉 Evaluation Metric

## Mean Squared Error (MSE)

Used to measure prediction quality:
- Lower values indicate better performance
- Penalizes large prediction errors more heavily

---

# 📊 Key Visualizations

The notebook includes:

- Gradient Descent convergence plot
- Predicted vs actual values plot
- Weight/bias inspection

---
