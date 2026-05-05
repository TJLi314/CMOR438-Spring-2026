---
name: 🐛 Bug Report
about: Report something that is broken or producing incorrect results
title: "[Bug] "
labels: bug
assignees: <your-profile-name>
---

## 🐛 Bug Summary

Briefly describe the issue.

---

## 📌 Algorithm / Module Affected

Which part of RiceML is affected?

- [ ] Linear Regression
- [ ] Logistic Regression
- [ ] KNN
- [ ] Decision Trees
- [ ] Random Forest
- [ ] MLP
- [ ] PCA
- [ ] K-Means
- [ ] DBSCAN
- [ ] Preprocessing
- [ ] Other: ___________

---

## 📊 Dataset Used (if applicable)

Please specify the dataset:

- Dataset name:
- Source (if known):
- Shape of data (if known):

---

## 🔁 Steps to Reproduce

Please provide a minimal reproducible example:

```python
# Example code
from rice_ml.supervised_learning.linear_regression import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

---

## ❌ Expected Behavior

What did you expect to happen?

---

## ⚠️ Actual Behavior

What actually happened instead?

---

## 📷 Screenshots / Logs (optional)

If applicable, add plots, error messages, or output logs.

---

## 💡 Additional Context

Anything else that might help diagnose the issue?