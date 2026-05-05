# PCA from Scratch — UCI Seeds Dataset

This repository contains a Jupyter notebook demonstrating a **Principal Component Analysis (PCA)** implementation built completely from scratch using NumPy. The goal is to show how PCA works under the hood and how it can be used for dimensionality reduction, visualization, and reconstruction.

---

# 📊 Dataset

We use the **UCI Seeds Dataset**, which contains measurements of wheat kernels from three different varieties.

- Source: UCI Machine Learning Repository  
- Samples: 210  
- Features: 7 continuous numerical features  
- Classes: 3 wheat types  

Each sample includes geometric measurements such as area, perimeter, kernel length, and more.

---

# 🎯 Objective

The notebook demonstrates:

- PCA implementation from scratch (no sklearn PCA used)
- Eigen-decomposition of covariance matrix
- Dimensionality reduction (7D → 2D)
- Visualization of projected data
- Explained variance analysis
- Data reconstruction from reduced space
- Information loss due to compression

---

# 📉 Key Outputs

The notebook produces:

### ✔ Explained Variance Ratio
Shows how much information each principal component retains.

### ✔ 2D Projection Plot
Visualizes dataset in reduced space:
- reveals class structure
- shows separability of wheat types

### ✔ Reconstruction Error
Measures information loss after compression:
- lower = better preservation

---

# 📌 Key Insights

- PCA finds directions of maximum variance in data
- The first two components capture most of the structure
- Even without labels, PCA reveals natural clustering tendencies
- Dimensionality reduction introduces some loss, but preserves global structure well