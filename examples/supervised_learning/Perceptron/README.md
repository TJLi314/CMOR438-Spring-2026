# 🧠 Perceptron from Scratch (NumPy Implementation)

This repository contains a complete from-scratch implementation and demonstration of the **Perceptron algorithm**, one of the simplest neural network models for binary classification.

The model is implemented using only NumPy and is evaluated on real-world datasets with visualization of decision boundaries and training behavior.

---

# 📌 Algorithm Overview

The Perceptron is a linear binary classifier that learns a decision boundary separating two classes.

It makes predictions using a linear function:

net_input = w1*x1 + w2*x2 + ... + wn*xn + b

Prediction rule:
- If net_input >= 0 → class 1
- If net_input < 0 → class -1

The model learns by updating weights whenever it makes a mistake.

---

# ⚙️ Learning Rule

For each misclassified sample, the Perceptron updates:

- Weights are adjusted in the direction of the correct class
- Bias is updated similarly

This process is repeated over multiple epochs until convergence or until the maximum number of iterations is reached.

---

# 📊 Dataset

This project demonstrates the Perceptron on real-world binary classification data:

We use the Breast Cancer Wisconsin Dataset, which contains features computed from digitized images of breast mass samples.

### Dataset Description:
- 30 numerical features
- Binary classification:
  - Malignant
  - Benign
- Used for medical diagnosis prediction

---


# 📊 Key Visualizations

The notebook includes:

- Decision boundary plots (2D feature slices)
- Training error over epochs
- Feature pair decision regions
- Model convergence behavior

---