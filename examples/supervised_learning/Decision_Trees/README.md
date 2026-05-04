# 🌳 Decision Tree from Scratch (NumPy)

This project implements a **Decision Tree Classifier from scratch using only NumPy** and demonstrates its performance on a real-world dataset from the UCI Machine Learning Repository.

It includes a full Jupyter notebook walkthrough with:
- Data loading and preprocessing
- Model training and evaluation
- Confusion matrix analysis
- Feature importance approximation
- Tree visualization using NetworkX

---

## 🎯 Dataset Used

### 🚗 UCI Car Evaluation Dataset

We use the classic **Car Evaluation dataset** from the UCI repository.

### 🔍 Task
Predict the **acceptability of a car** based on its attributes.

### 🧠 Target Classes

- `unacc` → unacceptable
- `acc` → acceptable
- `good` → good
- `vgood` → very good

### 📊 Input Features

All features are categorical:

- Buying price (vhigh, high, med, low)
- Maintenance cost (vhigh, high, med, low)
- Number of doors (2, 3, 4, 5more)
- Passenger capacity (2, 4, more)
- Luggage boot size (small, med, big)
- Safety rating (low, med, high)

---

## 📈 Evaluation Metrics

The notebook evaluates performance using:

- Accuracy (train/test)
- Confusion matrix
- Feature split frequency (approximate importance)

---

## 🌳 Visualization

The decision tree is visualized using **NetworkX**, where:

- 🟡 Root node = gold
- 🔵 Decision nodes = light blue
- 🟢 Leaf nodes = light green

Edges represent True/False splits.

This helps interpret how the model makes decisions.