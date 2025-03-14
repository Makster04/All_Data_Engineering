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

Hierarchical Clustering is a clustering technique that builds a hierarchy of clusters either in an **agglomerative** (bottom-up) or **divisive** (top-down) manner. Below, I'll demonstrate the steps for performing **Agglomerative Hierarchical Clustering** using Python.

### **Steps for Hierarchical Clustering**
1. **Import Required Libraries**  
2. **Generate Sample Data or Load a Dataset**  
3. **Compute the Distance Matrix**  
4. **Perform Agglomerative Clustering**  
5. **Visualize the Dendrogram**  
6. **Determine the Optimal Number of Clusters**  
7. **Assign Cluster Labels and Visualize Results**  

---

### **Python Code Implementation**
I'll demonstrate this using synthetic data.

#### **Step 1: Import Required Libraries**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
```

#### **Step 2: Generate Sample Data**
```python
# Create a dataset with 2 features
X, _ = make_blobs(n_samples=200, centers=3, random_state=42, cluster_std=1.2)

# Convert to DataFrame
df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])

# Visualize the data points
plt.scatter(df['Feature 1'], df['Feature 2'], c='gray')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Dataset for Clustering")
plt.show()
```
<img src="https://github.com/user-attachments/assets/c8295f61-cbcd-45ee-9313-3e7997abe52e" width="500" />


#### **Step 3: Compute the Distance Matrix & Plot Dendrogram**
```python
# Create a dendrogram
plt.figure(figsize=(10, 5))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

# Add labels
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()
```
- The **dendrogram** helps determine the optimal number of clusters by looking at the longest vertical line that can be cut without crossing horizontal lines.
<img src="https://github.com/user-attachments/assets/6598687a-ea3a-4fa1-bc8a-9ea25a8cbfc7" width="500" />

#### **Step 4: Perform Hierarchical Clustering**
```python
# Applying Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Assign cluster labels to DataFrame
df['Cluster'] = y_hc
```

#### **Step 5: Visualize the Clusters**
```python
# Scatter plot of clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=50, c='green', label='Cluster 3')

# Mark cluster centers
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Clusters Identified by Hierarchical Clustering")
plt.legend()
plt.show()
```
<img src="https://github.com/user-attachments/assets/1ec4eece-0c21-4d31-bb6c-d65914e7b359" width="500" />
---
### **Explanation of Each Step**
1. **Dendrogram Analysis**:  
   - The **dendrogram** shows how clusters are merged at different distances.  
   - The **optimal number of clusters** can be chosen by setting a horizontal line at the largest vertical gap.  

2. **Agglomerative Clustering**:  
   - Uses **Ward’s linkage**, which minimizes variance when merging clusters.  
   - Distance metric used: **Euclidean Distance**.  

3. **Cluster Visualization**:  
   - Data points are colored based on their assigned clusters.

---
Here is a Clustergram of it
```python
# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.datasets import make_blobs

# Generating synthetic dataset
X, _ = make_blobs(n_samples=200, centers=3, random_state=42, cluster_std=1.2)

# Creating a DataFrame
df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])

# Creating a clustergram (clustered heatmap)
sns.clustermap(df, method='ward', cmap='viridis', figsize=(10, 8))

# Displaying the plot
plt.show()
```
Here's the **clustergram (clustered heatmap)** visualizing hierarchical clustering combined with the intensity of your synthetic data. Rows represent data points, columns represent features, and the dendrograms indicate clusters and similarities across data points and features.
<img src="https://github.com/user-attachments/assets/c77bb8e0-b663-4933-a598-6e27ff00f995" width="500" />

---
## Evaluating Hierarchical Clustering
To assess the performance of the clustering model, we can use the following evaluation metrics:

1. **Silhouette Score**  
   - Measures how well-separated the clusters are.
   - Range: $$\(-1\)$$ (poor separation) to $$\(1\)$$ (clear separation).
2. **Davies-Bouldin Index**  
   - Lower values indicate better clustering quality.
3. **Calinski-Harabasz Index**  
   - Higher values indicate better clustering.
4. **Inertia (Within-Cluster Sum of Squares - WCSS)**  
   - Measures compactness within clusters.

Results:
```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Compute evaluation metrics
silhouette_avg = silhouette_score(X, y_hc)
davies_bouldin = davies_bouldin_score(X, y_hc)
calinski_harabasz = calinski_harabasz_score(X, y_hc)

# Display results
evaluation_results = pd.DataFrame({
    "Metric": ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"],
    "Value": [silhouette_avg, davies_bouldin, calinski_harabasz]
})

import ace_tools as tools
tools.display_dataframe_to_user(name="Hierarchical Clustering Evaluation", dataframe=evaluation_results)
```
```
Result
                    Metric        Value
0         Silhouette Score     0.816140
1     Davies-Bouldin Index     0.264168
2  Calinski-Harabasz Index  2392.312365
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

