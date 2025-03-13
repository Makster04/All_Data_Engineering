# **K-Means Clustering Overview**
---
K-Means is a popular **unsupervised machine learning algorithm** used for **clustering data into K groups**. It assigns data points to clusters based on their similarity, measured using **Euclidean distance**. The algorithm is widely used in customer segmentation, image recognition, and anomaly detection.

---
## What is Clustering?
**Cluster:** A group of similar data points based on certain features.

**K**: Number of groups specified in clustering algorithms. 

**Clustering Techniques** are among the most popular unsupervised machine learning algorithms. The main idea behind clustering is that you want to group objects into similar classes, in a way that:
- intra-class similarity is high (similarity amongst members of the same group is high) *(Example: Similarity between two players from the same team.)*
- inter-class similarity is low (similarity of different groups is low) *(Example: Differences between players from different teams.)*

**Why is Clustering Important?** Clustering groups similar data, helps in pattern discovery, and improves decision-making in various applications like recommendation systems, image segmentation, and market analysis.
- **Example:** In the NBA, clustering can group players based on playing style (e.g., shooters, defenders, playmakers), helping teams make better recruitment and game strategy decisions.

**How can you determine if they're similar?**: The closer two points are, the more similar they are. It is useful to make a distinction between hierarchical and nonhierarchical clustering algorithms:

**Agglomerative Hierarchical:**
1. It starts with **n** clusters (where n is the number of observations, so each ovservation is a cluster) 
2. Then combines the two most similar clusters combines
3. Then combines the next two most similar cluster, and continues.
*(A divise one does the exact oppoiste going from 1 to n clusters)*
   
**Agglomerative Nonhierarchical**
- Chooses **k** intitial cluster *(Refers to k starting points, centroids, chosen in k-mean clustering to begin grouping data points)* and reassigns observations *(The data points are moved between clusters)* until no imporovments can be obtained *(until the clusters stabilize, meaning no data points need to switch clusters because their assignments no longer improve the clustering (i.e., they are already closest to their optimal centroid)*

You basically try to group data together without knowing what actual cluster/classes are.

---


## **Clustering Methods: A Comparison**
There are multiple clustering techniques, each with unique strengths and weaknesses:

| **Method**            | **Description**  | **Pros**  | **Cons** |
|----------------------|----------------|----------|----------|
| **K-Means (Non-Hierarchical)** | Partitions data into **K clusters** based on Euclidean distance | Fast, scalable, simple | Sensitive to outliers and initial centroid placement |
| **Hierarchical Clustering** | Creates a **tree-like structure (dendrogram)** of clusters | No need to specify K | Computationally expensive for large datasets |
| **DBSCAN** | Groups data based on **density** rather than distance | Detects arbitrarily shaped clusters, handles noise well | Struggles with varying densities and high-dimensional data |
| **Gaussian Mixture Model (GMM)** | Uses **probabilistic clustering** based on Gaussian distributions | Works well with overlapping clusters | Requires choosing K, computationally intensive |

---
## Non-Hierarchical Clustering With K-Means Clustering

K-means is a popular **non-hierarchical clustering** method that finds **K cluster centers** as the mean of data points in each cluster. The number **K** must be chosen beforehand. The algorithm iteratively:  

### **Steps in the K-Means Algorithm**
1. **Select K:** Choose the number of clusters.
2. **Initialize Centroids:** Randomly select **K points** as the initial cluster centers.
3. **Assign Data Points:** Each data point is assigned to the **nearest centroid**.
4. **Update Centroids:** Compute the new centroids as the **mean of all assigned points**.
5. **Repeat:** Iterate until centroids no longer change (convergence).

It assumes clusters are formed based on the **arithmetic mean**, and points are reassigned if they become closer to a different centroid.

How to do it:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
```
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Step 1: Generate Sample Data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Step 2: Select K → Chose 𝐾 = 4
K = 4  # Ensure it's an integer

# Step 3: Initialize K-Means
kmeans = KMeans(n_clusters=K, init='random', n_init=10, max_iter=300, random_state=42)

# Step 4: Fit the model and predict cluster assignments
y_kmeans = kmeans.fit_predict(X)

# Step 5: Plot the Results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6, edgecolors='k', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroids')
plt.title("K-Means Clustering Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
```
<img src="https://github.com/user-attachments/assets/b8683285-4a82-41de-bb72-75d2727255e7" width="500">

---
### Evaluating Cluster Fitness in K-Means

Evaluating the quality of clusters formed by K-Means is crucial to ensure that the algorithm is effectively grouping similar data points. Here are some common methods to assess cluster fitness:

#### **1. Within-Cluster Sum of Squares (WCSS) / Inertia**
- Measures the compactness of clusters by summing the squared distances of each point to its assigned centroid.
- Lower WCSS values indicate better clustering.

#### **2. The Elbow Method**
- Plots WCSS against different values of K.
- The optimal K is the "elbow point," where the reduction in WCSS slows down significantly.

#### **3. Silhouette Score**
- Measures how similar a data point is to its assigned cluster compared to other clusters.
- Ranges from -1 to 1, where higher values indicate better clustering.

#### **4. Davies-Bouldin Index**
- Measures the average similarity ratio between each cluster and the most similar one.
- Lower values indicate better clustering.

#### **5. Calinski-Harabasz Index**
- Measures the variance ratio between clusters and within clusters.
- Higher values indicate well-separated clusters.

---

### **Python Code to Evaluate Cluster Fitness**
Let's implement these evaluation methods in Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Step 1: Generate Sample Data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Step 2: Apply K-Means Clustering with Different K Values
wcss = []
silhouette_scores = []
db_scores = []
ch_scores = []
K_values = range(2, 11)

for k in K_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Compute evaluation metrics
    wcss.append(kmeans.inertia_)  # WCSS
    silhouette_scores.append(silhouette_score(X, y_kmeans))  # Silhouette Score
    db_scores.append(davies_bouldin_score(X, y_kmeans))  # Davies-Bouldin Index
    ch_scores.append(calinski_harabasz_score(X, y_kmeans))  # Calinski-Harabasz Index

# Step 3: Plot the Elbow Method (WCSS)
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(K_values, wcss, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method: Optimal K")

# Step 4: Plot Silhouette Score
plt.subplot(1, 3, 2)
plt.plot(K_values, silhouette_scores, marker='s', linestyle='--', color='g')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. K")

# Step 5: Plot Davies-Bouldin and Calinski-Harabasz Index
plt.subplot(1, 3, 3)
plt.plot(K_values, db_scores, marker='^', linestyle='-.', color='r', label="Davies-Bouldin")
plt.plot(K_values, ch_scores, marker='d', linestyle='-', color='purple', label="Calinski-Harabasz")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Index Value")
plt.title("Davies-Bouldin & Calinski-Harabasz")
plt.legend()

plt.tight_layout()
plt.show()
```
![output (3)](https://github.com/user-attachments/assets/d1864b13-33f1-485e-b4bf-c4d5da461c56)

---

### **Interpreting the Results**
1. **Elbow Method (WCSS)**:
   - The point where WCSS starts to level off indicates the best K value.
   
2. **Silhouette Score**:
   - Higher values (closer to 1) indicate that clusters are well-separated.

3. **Davies-Bouldin Index**:
   - Lower values indicate that clusters are well-separated.

4. **Calinski-Harabasz Index**:
   - Higher values indicate better-defined clusters.

---

### **Use Case for NBA Clustering**
If clustering NBA players based on their performance metrics (e.g., points, rebounds, assists), these evaluation metrics help determine the optimal number of player categories (e.g., scorers, playmakers, defenders, etc.).

Would you like me to adapt this to a real NBA dataset for better insights? 🚀

