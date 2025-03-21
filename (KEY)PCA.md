# Principal Component Analysis (PCA) 
---
## 1. Explanation of PCA
**Principal Component Analysis (PCA)** is a dimensionality reduction technique commonly used in Machine Learning and Data Science to reduce the number of input features while preserving as much variability in the data as possible. It transforms the original variables into new, uncorrelated variables called **principal components**, which are ordered by the amount of variance they explain in the data.

## 2. Definition of PCA
PCA is a statistical method that:

- Identifies the directions (**principal components**) in which the data varies the most.
- Projects the original data onto these principal components.
- Reduces redundancy and noise by focusing on the most significant components.


---
### **Summary of Steps** (Without Sckit Learning)
1. **Standardizing the Data** → Makes sure all features have mean **0** and variance **1**.
2. **Computing the Covariance Matrix** → Measures relationships between features.
3. **Computing Eigenvalues & Eigenvectors** → Determines principal components.
4. **Sorting Eigenvectors** → Selects the most important features.
5. **Projecting Data** → Reduces dimensions while preserving the most variance.

#### **1. Standardizing the Data**  
Ensuring each feature has **mean = 0** and **unit variance = 1**.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset (only features, ignoring labels for PCA)
iris = load_iris()
X = iris.data  

# Standardizing the data (zero mean, unit variance)
mean = np.mean(X, axis=0)  # Mean of each feature
std_dev = np.std(X, axis=0)  # Standard deviation of each feature

X_standardized = (X - mean) / std_dev  # Standardization formula

# Print standardized data statistics (should have mean ≈ 0, variance ≈ 1)
print("Mean after standardization:\n", np.mean(X_standardized, axis=0))
print("Standard deviation after standardization:\n", np.std(X_standardized, axis=0))
```
```
# Mean after standardization (≈ 0):
[-1.69031455e-15, -1.84297022e-15, -1.69864123e-15, -1.40924309e-15]

# Standard deviation after standardization (≈ 1):
[1., 1., 1., 1.]
```

---

#### **2. Computing the Covariance Matrix**  
This matrix tells us how much each feature varies in relation to others.

```python
# Compute covariance matrix
cov_matrix = np.cov(X_standardized.T)  # Transpose because np.cov expects variables as rows

print("Covariance Matrix:\n", cov_matrix)
```
```
[[ 1.00671141, -0.11835884,  0.87760447,  0.82343066],
 [-0.11835884,  1.00671141, -0.43131554, -0.36858315],
 [ 0.87760447, -0.43131554,  1.00671141,  0.96932762],
 [ 0.82343066, -0.36858315,  0.96932762,  1.00671141]]
```

---

#### **3. Computing Eigenvalues and Eigenvectors**  
Eigenvalues and eigenvectors determine the **principal components**.

```python
# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```
- **Eigenvalues** indicate how much variance is captured by each principal component.
- **Eigenvectors** represent the direction of these components.
```python
# Eigenvalues (unsorted):
[2.93808505, 0.9201649 , 0.14774182, 0.02085386]
```

---

#### **4. Sorting Eigenvectors**  
Selecting the top eigenvectors based on eigenvalues.

```python
# Sorting eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]  # Get indices of sorted eigenvalues (descending order)

eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]  # Sort eigenvectors accordingly

print("Sorted Eigenvalues:\n", eigenvalues_sorted)
print("Sorted Eigenvectors:\n", eigenvectors_sorted)
```
```python
# Sorted Eigenvalues (descending order):
[2.93808505, 0.9201649 , 0.14774182, 0.02085386]
# Sorted Eigenvectors:
[[ 0.52106591, -0.37741762, -0.71956635,  0.26128628],
 [-0.26934744, -0.92329566,  0.24438178, -0.12350962],
 [ 0.5804131 , -0.02449161,  0.14212637, -0.80144925],
 [ 0.56485654, -0.06694199,  0.63427274,  0.52359713]]
```

---

#### **5. Projecting the Data**  
Transforming data into the new lower-dimensional space.

```python
# Choose the number of principal components (e.g., 2 for visualization)
num_components = 2
top_eigenvectors = eigenvectors_sorted[:, :num_components]  # Select the top eigenvectors

# Project data onto new principal components
X_pca = X_standardized.dot(top_eigenvectors)

print("Shape of reduced data:", X_pca.shape)

# Visualizing the transformed data
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis', edgecolors='k', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Iris Dataset')
plt.colorbar(label='Target Classes')
plt.show()
```
```python
# PCA data shape (manual calculation):
(150, 2)
```
---

## With Sckit Learning

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Step 1: Load Dataset
iris = load_iris()
X = iris.data  # Features

# Step 2: Standardizing the Data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)  # Automatically normalizes data (mean=0, variance=1)

# Step 3: Apply PCA
pca = PCA(n_components=2)  # Reducing to 2 dimensions
X_pca = pca.fit_transform(X_standardized)

# Step 4: Visualizing the PCA-Transformed Data
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis', edgecolors='k', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Iris Dataset')
plt.colorbar(label='Target Classes')
plt.show()
```
![image](https://github.com/user-attachments/assets/c4a54f02-0795-4f82-9b00-43e3b21936f5)

---
## Determine Optimal Number of Components (Explained Variance):
To find the optimal number of components, examine the explained variance:

```python
# PCA with all components
pca_full = PCA()
pca_full.fit(X_scaled)

# Explained variance ratio
explained_variance = pca_full.explained_variance_ratio_
cum_explained_variance = np.cumsum(explained_variance)

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(cum_explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of PCA Components')
plt.grid()
plt.show()
```
Interpretation:
Select the number of components where you retain approximately 90-95% of variance.
![image](https://github.com/user-attachments/assets/1cebf942-632a-43e1-928e-655198664458)
---

## 3. Comparison and Contrast of PCA with Other Methods

| Feature | PCA | t-SNE | LDA | Autoencoders |
|---------|-----|------|-----|--------------|
| **Purpose** | Dimensionality reduction | Visualization | Classification-based reduction | Non-linear feature learning |
| **Linear/Non-linear** | Linear | Non-linear | Linear | Non-linear |
| **Supervised/Unsupervised** | Unsupervised | Unsupervised | Supervised | Unsupervised (but can be supervised) |
| **Computational Cost** | Low to Moderate | High | Moderate | High |
| **Best Used For** | Feature extraction and compression | Data visualization | Feature selection for classification | Learning compressed representations |
| **Data Preservation** | Maximizes variance | Preserves local structure | Maximizes class separability | Learns hierarchical representations |

PCA is best suited for cases where the primary goal is to **reduce features while retaining variance**, while methods like t-SNE are more for **visualizing clusters** and LDA is for **supervised dimensionality reduction**.

---

## Conclusion
- PCA is a **linear** technique for dimensionality reduction.
- It helps in **feature selection**, **noise removal**, and **data visualization**.
- It is different from t-SNE (which is non-linear) and LDA (which is supervised).
- PCA is widely used in **machine learning**, **computer vision**, and **data analysis**.

---
