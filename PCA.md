## 1. Explanation of PCA
Principal Component Analysis (PCA) is a dimensionality reduction technique commonly used in Machine Learning and Data Science to reduce the number of input features while preserving as much variability in the data as possible. It transforms the original variables into new, uncorrelated variables called **principal components**, which are ordered by the amount of variance they explain in the data.

## 2. Definition of PCA
PCA is a statistical method that:

- Identifies the directions (**principal components**) in which the data varies the most.
- Projects the original data onto these principal components.
- Reduces redundancy and noise by focusing on the most significant components.


---

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

---

#### **2. Computing the Covariance Matrix**  
This matrix tells us how much each feature varies in relation to others.

```python
# Compute covariance matrix
cov_matrix = np.cov(X_standardized.T)  # Transpose because np.cov expects variables as rows

print("Covariance Matrix:\n", cov_matrix)
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

## 4. Python Implementation of PCA
Let's implement PCA using Python with **Scikit-learn**.

## **Summary of Steps** (Without Sckit Learning)
1. **Standardizing the Data** → Makes sure all features have mean **0** and variance **1**.
2. **Computing the Covariance Matrix** → Measures relationships between features.
3. **Computing Eigenvalues & Eigenvectors** → Determines principal components.
4. **Sorting Eigenvectors** → Selects the most important features.
5. **Projecting Data** → Reduces dimensions while preserving the most variance.

### Example 1: Basic PCA for Dimensionality Reduction
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (reduce to 2 dimensions for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA-transformed data
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset')
plt.colorbar(label='Target')
plt.show()
```
**Explanation:**
- The dataset is standardized for PCA.
- PCA is applied to reduce dimensions from 4 to 2.
- The transformed data is visualized in a scatter plot.

### Example 2: Explained Variance in PCA
```python
# Fit PCA and check explained variance ratio
pca = PCA(n_components=4)
pca.fit(X_scaled)

# Plot explained variance
plt.figure(figsize=(8, 6))
plt.bar(range(1, 5), pca.explained_variance_ratio_, alpha=0.7, align='center')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.title('Explained Variance by PCA Components')
plt.show()
```
**Key Insights:**
- The **explained variance ratio** shows how much variance each principal component captures.
- We can decide the optimal number of components based on this.

### Example 3: PCA for Image Compression
PCA is commonly used in image processing to reduce dimensions while preserving details.

```python
from sklearn.decomposition import PCA
import cv2

# Load grayscale image
img = cv2.imread('image.jpg', 0)  # Load image in grayscale
img = cv2.resize(img, (256, 256))  # Resize for processing

# Apply PCA for compression
pca = PCA(n_components=50)  # Reduce to 50 principal components
transformed_img = pca.fit_transform(img)
reconstructed_img = pca.inverse_transform(transformed_img)  # Reconstruct image

# Display original and compressed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_img, cmap='gray')
plt.title('Compressed Image with PCA')

plt.show()
```
**Takeaway:** PCA can significantly reduce storage size while preserving essential image details.

## 5. Applications of PCA
- **Feature Extraction**: Reducing redundant features in high-dimensional datasets.
- **Image Compression**: Reducing pixel redundancy in images while retaining key details.
- **Anomaly Detection**: Identifying unusual patterns by capturing normal variance in data.
- **Noise Reduction**: Filtering out less important features while retaining core information.
- **Data Visualization**: Reducing dimensions for easier plotting and interpretation.

## Conclusion
- PCA is a **linear** technique for dimensionality reduction.
- It helps in **feature selection**, **noise removal**, and **data visualization**.
- It is different from t-SNE (which is non-linear) and LDA (which is supervised).
- PCA is widely used in **machine learning**, **computer vision**, and **data analysis**.

---
