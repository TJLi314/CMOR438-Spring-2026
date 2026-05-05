# 📧 Stump Bagging Classifier Demo (Spambase)

## Overview

This notebook demonstrates the implementation and performance of a **Stump Bagging Classifier**, an ensemble method that combines multiple weak learners (decision stumps) to produce a stronger classifier.

The experiment uses the **UCI Spambase dataset**, a classic binary classification problem for detecting spam emails.

---

## 📦 What is Stump Bagging?

**Stump Bagging** is a form of **bootstrap aggregation (bagging)** where:

- Each base learner is a **decision stump** (a decision tree with depth = 1)
- Each model is trained on a **bootstrap sample** of the data
- Predictions are combined using **majority voting**

### Why it works

- Decision stumps are **weak learners** (high bias)
- Bagging reduces **variance** by averaging across models
- The ensemble captures more robust patterns than any single stump

---

## 📊 Dataset: Spambase

This notebook uses the **UCI Spambase dataset**, which classifies emails as spam or not spam.

### Features

- 57 continuous features extracted from email content
- Include:
  - Word frequencies (e.g., "free", "money")
  - Character frequencies (e.g., `!`, `$`)
  - Capitalization patterns

### Target

- `1` → Spam  
- `0` → Not Spam  

### Why this dataset?

- High-dimensional and noisy → ideal for testing ensembles
- Weak learners alone struggle, making improvements from bagging clear
- Widely used benchmark for classification tasks

---

## 🧠 Models Implemented

### 1. Decision Stump
- A shallow decision tree (`max_depth = 1`)
- Serves as a weak baseline model

### 2. StumpBaggingClassifier
- Ensemble of multiple decision stumps
- Uses:
  - Bootstrap sampling
  - Majority voting

---

## 🔍 Experiments

The notebook walks through:

1. **Data preprocessing**
   - Train/test split
   - Feature scaling (if applied)

2. **Baseline model**
   - Train a single decision stump
   - Evaluate accuracy

3. **Ensemble model**
   - Train the StumpBaggingClassifier
   - Compare performance vs baseline

---

## ⚙️ Implementation Details

The models are implemented from scratch using:

- `numpy` for computation
- Custom implementations of:
  - `DecisionTreeClassifier`
  - `StumpBaggingClassifier`

### Core Components

- **Bootstrap Sampling**
  - Random sampling with replacement from training data

- **Majority Voting**
  - Final prediction determined by the most common class across models

---