# 📈 Linear Regression from Scratch (NumPy Implementation)

This repository contains a complete from-scratch implementation and demonstration of **Linear Regression**, supporting two optimization methods:

- Ordinary Least Squares (OLS) — closed-form solution
- Gradient Descent (GD) — iterative optimization

The model is implemented using only NumPy and is evaluated on a real-world regression dataset.

---

# 📌 Algorithm Overview

Linear Regression models the relationship between a continuous target variable \( y \) and input features \( X \) using a linear function:

\[\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b\]

Where:
- \( w \): learned weights (coefficients)
- \( b \): bias (intercept)
- \( X \): feature matrix
- \( \hat{y} \): predicted output

The objective is to minimize the Mean Squared Error (MSE):

\[MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2\]

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
\[w := w - \eta \frac{\partial L}{\partial w}\]

**Best for:**
- Large datasets
- Memory-efficient training
- Cases where matrix inversion is expensive

---

# 📊 Dataset

This project uses the **Diabetes Dataset** from scikit-learn, a standard regression benchmark dataset.

### Dataset Description:
- 10 numerical input features
- Continuous target variable (disease progression score)
- Standard benchmark dataset for regression tasks

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

\[MSE = \frac{1}{N} \sum (y - \hat{y})^2\]

Lower values indicate better model performance.

---

# 📊 Key Visualizations

The notebook includes:

- 📉 Gradient Descent convergence curve
- 📊 Predicted vs Actual scatter plot
- 📌 Weight interpretation output