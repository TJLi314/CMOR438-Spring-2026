# 🌲 Random Forest from Scratch (NumPy)

This project implements a **Random Forest algorithm from scratch using only NumPy** and demonstrates its performance on a real-world dataset from the UCI Machine Learning Repository.

The accompanying Jupyter notebook walks through:
- Data loading and preprocessing
- Training a custom Random Forest model
- Evaluating performance
- Understanding ensemble behavior

---

## ⚙️ How It Works

Each tree in the forest is trained independently using:

### 1. Bootstrap Sampling (Bagging)
- Each tree is trained on a **random sample (with replacement)** of the training data

### 2. Feature Subsampling
- At each split, only a **random subset of features** is considered
- This decorrelates trees and improves ensemble performance

### 3. Aggregation

- **Classification** → majority voting  
- **Regression** → averaging predictions  

---

## 🎯 Dataset

This notebook uses a dataset from the **UCI Machine Learning Repository**.

### 🔍 Task
Predict a target variable based on structured input features.

### 📊 Features
- A mix of numerical and/or categorical variables (encoded for model compatibility)

### 🧠 Target
- Depends on the dataset used in the notebook (classification or regression)

### 💡 Why this dataset?
- Real-world structured data
- Works well with tree-based models
- Allows clear demonstration of ensemble learning

---

## 📈 Evaluation

The notebook evaluates model performance using:

- Training vs testing accuracy (or error)
- Model predictions vs ground truth
- Observations of overfitting vs generalization