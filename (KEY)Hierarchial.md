# Hierarchical Clustering Overview

Hierarchical Clustering is an unsupervised machine learning algorithm used to group similar data points into clusters. Unlike K-Means, it does not require specifying the number of clusters beforehand. Instead, it creates a tree-like structure (dendrogram) that helps determine the optimal number of clusters.

---

### What is Hierarchical Clustering?

Hierarchical clustering builds nested clusters by either:

1. **Agglomerative Approach (Bottom-Up)**: Starts with individual data points as separate clusters and merges the most similar ones iteratively until one cluster remains.
2. **Divisive Approach (Top-Down)**: Starts with all data points in a single cluster and splits them into smaller clusters recursively.

Hierarchical clustering is widely used in market segmentation, image analysis, and gene expression data classification.

---

## The Process Behind Hierarchical Clustering

### Hierarchical Agglomerative Clustering (Bottom-Up)

Hierarchical Agglomerative Clustering (HAC) follows a **bottom-up** approach where each data point starts as its own cluster. The process proceeds as follows:

1. **Compute Distance Matrix**: Calculate the pairwise distance between all data points.
2. **Merge Closest Clusters**: Identify the two closest clusters and merge them into a single cluster.
3. **Update Distance Matrix**: Recalculate distances between the newly formed cluster and the remaining clusters based on a linkage criterion.
4. **Repeat Until One Cluster Remains**: Continue merging clusters until a single cluster containing all data points is formed.

### Divisive Hierarchical Clustering (Top-Down)

Divisive Hierarchical Clustering follows a **top-down** approach. The process proceeds as follows:

1. **Start with a Single Cluster**: All data points initially belong to one large cluster.
2. **Compute Dissimilarity Matrix**: Calculate pairwise distances to measure the dissimilarity between data points.
3. **Split the Cluster**: The cluster is divided into two sub-clusters based on the greatest dissimilarity (often using techniques like Principal Component Analysis or Max Variance Splitting).
4. **Repeat Recursively**: The splitting continues until each data point is its own cluster or a predefined stopping condition is met (e.g., a specific number of clusters is reached).
5. **Construct a Dendrogram**: The hierarchical splits are represented in a dendrogram, which helps visualize the clustering structure.

The divisive approach is computationally expensive compared to agglomerative clustering because it requires evaluating multiple possible splits at each step.

#### Comparison:
- **Agglomerative Clustering** starts with individual points and progressively merges them into larger clusters, forming a bottom-up dendrogram.
- **Divisive Clustering (simulated)** starts with all points in one cluster and recursively splits them into smaller groups, forming a top-down dendrogram.
![output (5)](https://github.com/user-attachments/assets/9bc2234d-a05a-4236-b4a0-4d7b1f6a3f52)


---

### Types of Linkage Criteria in Hierarchical Agglomerative Clustering

There are three common linkage methods used to determine the distance between clusters:

1. **Single Linkage**: Uses the smallest distance between any two points in two different clusters. This results in elongated and chain-like clusters.
2. **Complete Linkage**: Uses the largest distance between any two points in two different clusters. This results in compact and well-separated clusters.
3. **Average Linkage**: Uses the average of all pairwise distances between points in two different clusters. It provides a balance between single and complete linkage.
![image](https://github.com/user-attachments/assets/d802a653-69b4-45cf-b099-b968fd1c83b0)

### Purpose of a Dendrogram

A **dendrogram** is a tree-like diagram that represents the merging process in hierarchical clustering. It provides a visual summary of the clustering process and helps determine the optimal number of clusters by identifying a cutoff threshold.

---

## How is Hierarchical Clustering Different from K-Means?

| Feature                      | Hierarchical Clustering                                     | K-Means Clustering                                    |
| ---------------------------- | ----------------------------------------------------------- | ----------------------------------------------------- |
| **Cluster Formation**        | Creates a dendrogram and merges/splits clusters iteratively | Assigns points to fixed K clusters based on centroids |
| **Number of Clusters**       | No need to predefine K                                      | Must specify K beforehand                             |
| **Algorithm Type**           | Can be agglomerative (bottom-up) or divisive (top-down)     | Non-hierarchical (flat clustering)                    |
| **Computational Complexity** | More expensive (O(n²) or O(n³))                             | More efficient (O(n))                                 |
| **Handling of Outliers**     | Better at handling outliers                                 | Sensitive to outliers                                 |
| **Scalability**              | Slow for large datasets                                     | Works well for large datasets                         |

---

## Steps in Hierarchical Clustering Algorithm (Agglomerative)

1. **Compute Distance Matrix**: Calculate the pairwise distance between data points (e.g., Euclidean distance).
2. **Merge Closest Clusters**: Find the two closest clusters and merge them into one.
3. **Update Distance Matrix**: Recompute distances between the new cluster and existing clusters using linkage methods.
4. **Repeat Until One Cluster Remains**: Continue merging until all data points form a single cluster.

## Steps in Hierarchical Clustering Algorithm (Divisive)

1. **Start with One Cluster**: All data points are initially in one large cluster.
2. **Compute Dissimilarity Matrix**: Calculate pairwise distances among all points.
3. **Split the Cluster**: Identify the most dissimilar points and split the cluster into two groups.
4. **Recursively Split Sub-clusters**: Continue breaking down each sub-cluster into smaller clusters using the same criteria.
5. **Repeat Until Desired Number of Clusters is Reached**: The process stops when each point is in its own cluster or a predefined stopping condition is met.
6. **Visualize Using a Dendrogram**: The hierarchical structure is represented in a dendrogram for interpretation.

---

## Python Implementation of Hierarchical Clustering

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Step 1: Generate Sample Data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Step 2: Compute Linkage Matrix
linkage_matrix = linkage(X, method='ward')  # Ward minimizes variance within clusters

# Step 3: Plot the Dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=10)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
```

### Determining the Optimal Number of Clusters

The dendrogram helps determine the number of clusters by cutting at an appropriate distance threshold.

```python
# Step 4: Extract Clusters (Set Threshold)
k = 4  # Define number of clusters
clusters = fcluster(linkage_matrix, k, criterion='maxclust')

# Step 5: Plot the Clustered Data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6, edgecolors='k', marker='o')
plt.title("Hierarchical Clustering Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

---

## Evaluating Hierarchical Clustering
### 1. **Cophenetic Correlation Coefficient**
Measures how well the hierarchical clustering preserves the original distances.
```python
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cophenet

coph_corr, _ = cophenet(linkage_matrix, pdist(X))
print(f"Cophenetic Correlation Coefficient: {coph_corr:.4f}")
```

### 2. **Silhouette Score**
Evaluates how well data points fit within their assigned clusters.
```python
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, clusters)
print(f"Silhouette Score: {silhouette_avg:.4f}")
```

---

## Advantages and Disadvantages of Hierarchical Clustering
### **Advantages:**
- Does not require specifying K beforehand.
- Creates an interpretable dendrogram.
- Works well for small to medium-sized datasets.

### **Disadvantages:**
- Computationally expensive for large datasets.
- Hard to modify clusters once merged.
- Sensitive to noise and outliers.

## Conclusion
Hierarchical clustering is a powerful technique for analyzing data relationships and does not require predefining the number of clusters. While it is computationally expensive, it provides interpretability through dendrograms, making it useful in applications like bioinformatics, market segmentation, and text mining.

