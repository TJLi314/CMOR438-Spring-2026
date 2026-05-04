# 🧠 Logistic Regression from Scratch (NumPy Implementation)

This repository contains a complete from-scratch implementation and demonstration of **Logistic Regression**, built using only NumPy. The model is trained using stochastic gradient descent and evaluated on a real-world binary classification dataset.

The implementation includes probability prediction, cross-entropy loss optimization, training visualization, and decision boundary plotting.

---

# 📌 Algorithm Overview

Logistic Regression is a linear classifier used for binary classification problems. It models the probability that a given input belongs to class 1.

The model computes a linear combination of input features and passes it through the sigmoid function to produce a probability between 0 and 1:

- Linear score: weighted sum of input features plus bias
- Sigmoid activation: converts score into probability

Prediction rule:
- If probability ≥ 0.5 → class 1
- If probability < 0.5 → class 0

---

# ⚙️ Learning Mechanism

The model is trained using **stochastic gradient descent (SGD)**:

- Each training sample is processed individually
- The prediction error is computed using cross-entropy loss
- Weights and bias are updated after each sample
- This process is repeated over multiple epochs

The objective is to minimize cross-entropy loss, which penalizes incorrect predictions more heavily when the model is confident but wrong.

---

# 📊 Dataset

This project uses a real-world dataset containing statistical features extracted from images of banknotes.

### Dataset Description:
- 4 numerical features extracted from image statistics
- Binary classification target:
  - 0 → Fake banknote
  - 1 → Genuine banknote
- A widely used dataset for evaluating binary classification models

---

# 📈 Model Capabilities

This implementation supports:

- Probability prediction using sigmoid activation
- Binary classification (0/1 output)
- Cross-entropy loss computation
- Stochastic gradient descent optimization
- Training loss tracking over epochs
- Decision boundary visualization (2D feature projections)

---

# 📊 Key Components

## Sigmoid Activation
Transforms linear outputs into probabilities.

## Cross-Entropy Loss
Measures how far predicted probabilities are from true labels.

## Stochastic Gradient Descent
Updates model parameters incrementally using one sample at a time for efficient learning.

---


# 🌐 Visualization Features

The notebook includes:

- Training loss curve over epochs
- Decision boundary visualization using 2D feature projections
- Probability output inspection
- Class separation visualization

# 🧠 Summary

This implementation demonstrates how logistic regression learns a linear decision boundary by optimizing cross-entropy loss using stochastic gradient descent.

It provides a clear understanding of:
- How probabilities are modeled in classification
- How gradient-based optimization updates parameters
- How linear decision boundaries separate classes