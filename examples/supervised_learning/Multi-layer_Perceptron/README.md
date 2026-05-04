# 🍷 Neural Network from Scratch — MLP on UCI Wine Dataset

This project implements a fully connected **Multi-Layer Perceptron (MLP)** from scratch using only NumPy and evaluates it on the **UCI Wine dataset**. The goal is to demonstrate how a neural network works under the hood, including forward propagation, backpropagation, and gradient descent optimization.

---

## 🍷 Dataset: UCI Wine

The **Wine dataset** contains chemical analysis results of wines grown in the same region in Italy but derived from three different cultivars.

### Key properties:
- 178 samples
- 13 numerical features
- 3 classes (wine types)
- No missing values

This makes it a great dataset for multiclass classification and neural network experimentation.

---

## 📊 Visualizations Included

This notebook includes several visualizations to better understand the data and model behavior:

### 1. 📉 Training Loss Curve
Shows how the model converges over time.

### 2. 📊 Feature Distributions
Histograms of input features grouped by class.

### 3. 🎯 Confusion Matrix
Visual breakdown of model predictions vs true labels.

---

## 📈 Results

After training, the model achieves:

- **High training accuracy** (fits training distribution well)
- **Strong test accuracy** (good generalization on unseen data)

Example output:
